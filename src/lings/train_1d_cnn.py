from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis_spectral import load_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cnn1d"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a compact 1D CNN for yarn spectral anomaly detection.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_inputs(x_full: np.ndarray, wavelength: np.ndarray):
    upper = min(2450.0, float(wavelength[-1]))
    mask = (wavelength >= 1050.0) & (wavelength <= upper)
    indices = np.flatnonzero(mask)[::4]
    mean = x_full.mean(axis=1, keepdims=True)
    std = x_full.std(axis=1, keepdims=True)
    snv_full = (x_full - mean) / np.maximum(std, 1e-8)
    derivative_full = savgol_filter(
        snv_full,
        window_length=31,
        polyorder=2,
        deriv=1,
        delta=0.5,
        axis=1,
    )
    x = np.stack([snv_full[:, indices], derivative_full[:, indices]], axis=1).astype(np.float32)
    return x, wavelength[indices].astype(np.float32)


class SpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 16, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def normalize_from_train(x: np.ndarray, train_idx: np.ndarray):
    center = x[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    scale = x[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    return center, scale


def make_loader(x, y, indices, center, scale, batch_size, shuffle):
    x_tensor = torch.from_numpy(((x[indices] - center) / scale).astype(np.float32))
    y_tensor = torch.from_numpy(y[indices].astype(np.float32))
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator().manual_seed(20260831)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits.append(model(xb).cpu().numpy())
        labels.append(yb.numpy())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels).astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return labels, probabilities


def safe_auc(y_true, probability):
    return float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan")


def choose_threshold(y_true, probability):
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr - fpr
    score[~finite] = -np.inf
    return float(thresholds[int(np.argmax(score))])


def metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_abnormal": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": safe_auc(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one(
    x,
    y,
    train_idx,
    val_idx,
    test_idx,
    device,
    batch_size,
    max_epochs,
    patience,
    seed,
    protocol,
    fold_name,
):
    seed_everything(seed)
    center, scale = normalize_from_train(x, train_idx)
    train_loader = make_loader(x, y, train_idx, center, scale, batch_size, True)
    val_loader = make_loader(x, y, val_idx, center, scale, batch_size * 2, False)
    test_loader = make_loader(x, y, test_idx, center, scale, batch_size * 2, False)

    model = SpectralCNN().to(device)
    n_pos = int((y[train_idx] == 1).sum())
    n_neg = int((y[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)

        val_labels, val_probability = predict(model, val_loader, device)
        val_auc = safe_auc(val_labels, val_probability)
        val_threshold = choose_threshold(val_labels, val_probability)
        val_metrics = metrics(val_labels, val_probability, val_threshold)
        record = {
            "protocol": protocol,
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "val_auc": val_auc,
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_threshold": val_threshold,
        }
        history.append(record)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    val_labels, val_probability = predict(model, val_loader, device)
    threshold = choose_threshold(val_labels, val_probability)
    test_labels, test_probability = predict(model, test_loader, device)
    result = metrics(test_labels, test_probability, threshold)
    result.update(
        {
            "protocol": protocol,
            "fold": fold_name,
            "best_epoch": best_epoch,
            "val_auc": best_auc,
            "train_normal": n_neg,
            "train_abnormal": n_pos,
            "seconds": time.time() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    return result, test_probability, center, scale, model.state_dict(), history


def combined_metrics(y_true, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_abnormal": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": safe_auc(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_folds(records):
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    return {
        key: {
            "mean": float(np.mean([r[key] for r in records])),
            "std": float(np.std([r[key] for r in records], ddof=1)) if len(records) > 1 else 0.0,
        }
        for key in keys
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(args.seed)

    rows, wavelength, x_full, errors = load_data()
    if errors:
        raise RuntimeError(f"Failed to parse {len(errors)} input files")
    x, model_wavelength = build_inputs(x_full, wavelength)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    batches = np.asarray([r["batch"] for r in rows])
    device = torch.device("cpu")

    all_fold_metrics = []
    all_history = []
    predictions = [
        {
            "file": r["file"],
            "sample_name": r["sample_name"],
            "batch": r["batch"],
            "label": r["label"],
            "label_name": r["label_name"],
        }
        for r in rows
    ]

    # 1) Random stratified out-of-fold evaluation.
    random_probability = np.full(len(y), np.nan, dtype=float)
    random_prediction = np.full(len(y), -1, dtype=int)
    random_records = []
    splitter = StratifiedKFold(n_splits=args.random_folds, shuffle=True, random_state=args.seed)
    for fold, (train_pool, test_idx) in enumerate(splitter.split(x, y), start=1):
        train_idx, val_idx = train_test_split(
            train_pool,
            test_size=0.15,
            stratify=y[train_pool],
            random_state=args.seed + fold,
        )
        result, probability, _, _, _, history = train_one(
            x, y, train_idx, val_idx, test_idx, device,
            args.batch_size, args.max_epochs, args.patience,
            args.seed + fold, "random_stratified_cv", f"fold_{fold}",
        )
        random_probability[test_idx] = probability
        random_prediction[test_idx] = (probability >= result["threshold"]).astype(int)
        random_records.append(result)
        all_fold_metrics.append(result)
        all_history.extend(history)
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

    random_combined = combined_metrics(y, random_probability, random_prediction)
    random_combined["protocol"] = "random_stratified_cv_combined"
    random_combined["macro_fold_summary"] = summarize_folds(random_records)

    # 2) Leave-one-batch-out on the three batches that contain both classes.
    common_mask = np.isin(batches, ["DN", "DW", "DV"])
    batch_probability = np.full(len(y), np.nan, dtype=float)
    batch_prediction = np.full(len(y), -1, dtype=int)
    batch_records = []
    for offset, held_out in enumerate(["DN", "DW", "DV"], start=1):
        test_idx = np.flatnonzero(batches == held_out)
        train_pool = np.flatnonzero(common_mask & (batches != held_out))
        train_idx, val_idx = train_test_split(
            train_pool,
            test_size=0.20,
            stratify=y[train_pool],
            random_state=args.seed + 100 + offset,
        )
        result, probability, _, _, _, history = train_one(
            x, y, train_idx, val_idx, test_idx, device,
            args.batch_size, args.max_epochs, args.patience,
            args.seed + 100 + offset, "leave_one_batch_out", held_out,
        )
        batch_probability[test_idx] = probability
        batch_prediction[test_idx] = (probability >= result["threshold"]).astype(int)
        result["test_batch"] = held_out
        result["train_batches"] = "+".join(sorted(set(batches[train_pool])))
        batch_records.append(result)
        all_fold_metrics.append(result)
        all_history.extend(history)
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

    common_idx = np.flatnonzero(common_mask)
    batch_combined = combined_metrics(y[common_idx], batch_probability[common_idx], batch_prediction[common_idx])
    batch_combined.update(
        {
            "protocol": "leave_one_batch_out_combined",
            "macro_fold_summary": summarize_folds(batch_records),
            "macro_auc": float(np.mean([r["roc_auc"] for r in batch_records])),
            "weighted_auc": float(np.average([r["roc_auc"] for r in batch_records], weights=[r["n"] for r in batch_records])),
        }
    )

    for i in range(len(rows)):
        predictions[i].update(
            {
                "random_cv_probability": float(random_probability[i]),
                "random_cv_prediction": int(random_prediction[i]),
                "batch_cv_probability": float(batch_probability[i]) if np.isfinite(batch_probability[i]) else "",
                "batch_cv_prediction": int(batch_prediction[i]) if batch_prediction[i] >= 0 else "",
            }
        )

    # 3) Fit a final reusable model with a held-out calibration split.
    full_idx = np.arange(len(y))
    final_train_idx, final_val_idx = train_test_split(
        full_idx,
        test_size=0.15,
        stratify=y,
        random_state=args.seed + 999,
    )
    final_result, _, final_center, final_scale, final_state, final_history = train_one(
        x, y, final_train_idx, final_val_idx, final_val_idx, device,
        args.batch_size, args.max_epochs, args.patience,
        args.seed + 999, "final_model", "all_batches",
    )
    all_history.extend(final_history)
    checkpoint = {
        "model_class": "SpectralCNN",
        "model_state_dict": final_state,
        "normalization_center": torch.from_numpy(final_center),
        "normalization_scale": torch.from_numpy(final_scale),
        "wavelength_nm": torch.from_numpy(model_wavelength),
        "threshold": final_result["threshold"],
        "preprocessing": {
            "channels": ["SNV absorbance", "Savitzky-Golay first derivative of SNV"],
            "source_range_nm": [float(wavelength[0]), float(wavelength[-1])],
            "model_range_nm": [float(model_wavelength[0]), float(model_wavelength[-1])],
            "spacing_nm": float(np.median(np.diff(model_wavelength))),
            "savgol_window_points_at_0.5nm": 31,
            "savgol_polyorder": 2,
        },
        "validation": final_result,
    }
    torch.save(checkpoint, args.output_dir / "spectral_1d_cnn.pt")

    results = {
        "dataset": {
            "spectra": int(len(y)),
            "normal": int((y == 0).sum()),
            "abnormal": int((y == 1).sum()),
            "abnormal_rate": float(y.mean()),
            "source_wavelength_nm": [float(wavelength[0]), float(wavelength[-1])],
            "model_wavelength_nm": [float(model_wavelength[0]), float(model_wavelength[-1])],
            "model_points": int(len(model_wavelength)),
            "counts_by_batch": {
                batch: {
                    "normal": int(((batches == batch) & (y == 0)).sum()),
                    "abnormal": int(((batches == batch) & (y == 1)).sum()),
                }
                for batch in ["DN", "DW", "DV", "DY"]
            },
        },
        "model": {
            "architecture": "2-channel compact 1D CNN",
            "parameters": int(sum(p.numel() for p in SpectralCNN().parameters())),
            "loss": "BCEWithLogitsLoss with training-fold positive class weight",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "selection": "early stopping on validation ROC-AUC; threshold chosen on validation Youden J",
            "device": str(device),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "random_stratified_cv_folds": random_records,
        "random_stratified_cv_combined": random_combined,
        "leave_one_batch_out_folds": batch_records,
        "leave_one_batch_out_combined": batch_combined,
        "final_model_validation": final_result,
        "limitations": [
            "DY contains no abnormal sample, so abnormal sensitivity on DY is unknown.",
            "DW has only 6 abnormal samples and DN has 22; per-batch estimates have high uncertainty.",
            "Random stratified CV mixes batches and is optimistic for deployment to a new batch.",
            "The supplied files end at 2250 nm, not 2500 nm.",
        ],
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "fold_metrics.csv", all_fold_metrics)
    write_csv(args.output_dir / "training_history.csv", all_history)
    write_csv(args.output_dir / "out_of_fold_predictions.csv", predictions)
    print(json.dumps({
        "random_combined": random_combined,
        "batch_combined": batch_combined,
        "checkpoint": str((args.output_dir / "spectral_1d_cnn.pt").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
