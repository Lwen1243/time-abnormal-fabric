from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)


def read_spectrum(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()
    sample_name = lines[0].rstrip("\r\n").split(";", 1)[-1].strip()
    start = next(i for i, line in enumerate(lines) if line.startswith("#;Wavelength")) + 1
    wl = []
    absorbance = []
    for line in lines[start:]:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        try:
            wl.append(float(parts[1]))
            absorbance.append(float(parts[2]))
        except ValueError:
            continue
    return sample_name, np.asarray(wl, dtype=np.float32), np.asarray(absorbance, dtype=np.float32)


def load_data():
    rows = []
    spectra = []
    wavelength = None
    errors = []
    for label_name, folder, label in [
        ("正常", ROOT / "data" / "正常数据集", 0),
        ("异常", ROOT / "data" / "异常数据集", 1),
    ]:
        for path in sorted(folder.rglob("*.csv")):
            match = re.search(r"采谱(DN|DW|DV|DY)-", path.name)
            batch = match.group(1) if match else "未知"
            try:
                sample_name, wl, y = read_spectrum(path)
                if wavelength is None:
                    wavelength = wl
                if wl.shape != wavelength.shape or not np.allclose(wl, wavelength, atol=1e-6):
                    errors.append({"file": str(path.relative_to(ROOT)), "reason": "wavelength_grid_mismatch"})
                    continue
                if not np.all(np.isfinite(y)):
                    errors.append({"file": str(path.relative_to(ROOT)), "reason": "nonfinite_values"})
                    continue
                rows.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "sample_name": sample_name,
                        "batch": batch,
                        "label": label,
                        "label_name": label_name,
                    }
                )
                spectra.append(y)
            except Exception as exc:
                errors.append({"file": str(path.relative_to(ROOT)), "reason": repr(exc)})
    return rows, wavelength, np.stack(spectra).astype(np.float32), errors


def preprocess_snv_derivative(x_full, wavelength):
    # SNV removes multiplicative/scatter offsets. A 15.5 nm Savitzky-Golay
    # first derivative suppresses broad baseline drift while preserving bands.
    mean = x_full.mean(axis=1, keepdims=True)
    std = x_full.std(axis=1, keepdims=True)
    snv = (x_full - mean) / np.maximum(std, 1e-8)
    deriv = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    upper = min(2450.0, float(wavelength[-1]))
    mask = (wavelength >= 1050) & (wavelength <= upper)
    indices = np.flatnonzero(mask)[::4]  # 2 nm spacing for modelling
    return deriv[:, indices].astype(np.float32), wavelength[indices]


def make_model():
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.03, class_weight="balanced", dual="auto", max_iter=15000),
    )


def metric_record(y_true, scores, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_abnormal": int((y_true == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_val_predictions(x, y, groups, splitter):
    scores = np.full(len(y), np.nan, dtype=float)
    preds = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
        model = make_model()
        model.fit(x[tr], y[tr])
        scores[te] = model.decision_function(x[te])
        preds[te] = model.predict(x[te])
        fold_ids[te] = fold
    return scores, preds, fold_ids


def leave_one_batch_out(x, y, batches):
    records = []
    all_scores = np.full(len(y), np.nan, dtype=float)
    all_preds = np.full(len(y), -1, dtype=int)
    eligible = np.zeros(len(y), dtype=bool)
    for batch in ["DN", "DW", "DV"]:
        te = batches == batch
        tr = np.isin(batches, [b for b in ["DN", "DW", "DV"] if b != batch])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_model()
        model.fit(x[tr], y[tr])
        scores = model.decision_function(x[te])
        pred = model.predict(x[te])
        rec = metric_record(y[te], scores, pred)
        rec["test_batch"] = batch
        rec["train_batches"] = "+".join(sorted(set(batches[tr])))
        records.append(rec)
        all_scores[te] = scores
        all_preds[te] = pred
        eligible[te] = True
    combined = metric_record(y[eligible], all_scores[eligible], all_preds[eligible])
    combined["test_batch"] = "合并"
    combined["train_batches"] = "逐批次留一"
    return records, combined, all_scores, all_preds, eligible


def cohen_d_by_batch(x, y, batches, wavelength):
    ds = {}
    for batch in ["DN", "DW", "DV"]:
        a = x[(batches == batch) & (y == 1)]
        n = x[(batches == batch) & (y == 0)]
        v1 = a.var(axis=0, ddof=1) if len(a) > 1 else np.zeros(x.shape[1])
        v0 = n.var(axis=0, ddof=1) if len(n) > 1 else np.zeros(x.shape[1])
        pooled = np.sqrt(((len(a) - 1) * v1 + (len(n) - 1) * v0) / max(len(a) + len(n) - 2, 1))
        ds[batch] = (a.mean(axis=0) - n.mean(axis=0)) / np.maximum(pooled, 1e-8)
    stack = np.stack([ds[b] for b in ["DN", "DW", "DV"]])
    direction_consistency = np.abs(np.sign(stack).sum(axis=0)) / 3.0
    robust = np.median(np.abs(stack), axis=0) * direction_consistency
    peaks, _ = find_peaks(robust, distance=max(1, int(40 / float(wavelength[1] - wavelength[0]))))
    ranked = peaks[np.argsort(robust[peaks])[::-1]]
    bands = []
    for idx in ranked[:8]:
        bands.append(
            {
                "wavelength_nm": float(wavelength[idx]),
                "robust_effect": float(robust[idx]),
                "direction_consistency": float(direction_consistency[idx]),
                **{f"d_{b}": float(ds[b][idx]) for b in ["DN", "DW", "DV"]},
            }
        )
    return ds, robust, bands


def main():
    rows, wavelength, x_full, errors = load_data()
    y = np.asarray([r["label"] for r in rows], dtype=int)
    batches = np.asarray([r["batch"] for r in rows])
    sample_names = np.asarray([r["sample_name"] for r in rows])

    # Prefix label in groups so coincident names across classes cannot couple folds.
    groups = np.asarray([f"{r['label']}::{r['sample_name']}" for r in rows])
    x, model_wl = preprocess_snv_derivative(x_full, wavelength)

    naive_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    naive_scores, naive_pred, _ = cross_val_predictions(x, y, groups, naive_split)
    naive_metrics = metric_record(y, naive_scores, naive_pred)

    grouped_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grouped_scores, grouped_pred, _ = cross_val_predictions(x, y, groups, grouped_split)
    grouped_metrics = metric_record(y, grouped_scores, grouped_pred)

    lobo_rows, lobo_combined, lobo_scores, lobo_pred, lobo_eligible = leave_one_batch_out(x, y, batches)

    # PCA is for visualization only; model metrics above do not use this global fit.
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=4, random_state=42)
    pcs = pca.fit_transform(x_scaled)
    rng = np.random.default_rng(42)
    selected = []
    for label in [0, 1]:
        for batch in ["DN", "DW", "DV", "DY"]:
            idx = np.flatnonzero((y == label) & (batches == batch))
            if len(idx):
                take = min(len(idx), 120)
                selected.extend(rng.choice(idx, take, replace=False).tolist())
    selected = sorted(set(selected))

    # Class mean spectra for clean line charts, downsampled to 5 nm.
    chart_idx = np.arange(0, len(wavelength), 10)
    spectra_rows = []
    for idx in chart_idx:
        rec = {"wavelength_nm": float(wavelength[idx])}
        for label, label_name in [(0, "正常"), (1, "异常")]:
            vals = x_full[y == label, idx]
            rec[f"{label_name}_mean"] = float(vals.mean())
            rec[f"{label_name}_sd"] = float(vals.std(ddof=1))
        for batch in ["DN", "DW", "DV", "DY"]:
            for label, label_name in [(0, "正常"), (1, "异常")]:
                vals = x_full[(batches == batch) & (y == label), idx]
                rec[f"{batch}_{label_name}_mean"] = float(vals.mean()) if len(vals) else None
        spectra_rows.append(rec)

    ds, robust, bands = cohen_d_by_batch(x, y, batches, model_wl)

    class_batch_counts = defaultdict(lambda: {"spectra": 0, "samples": set()})
    for r in rows:
        key = (r["label_name"], r["batch"])
        class_batch_counts[key]["spectra"] += 1
        class_batch_counts[key]["samples"].add(r["sample_name"])
    count_rows = []
    for label_name in ["正常", "异常"]:
        for batch in ["DN", "DW", "DV", "DY"]:
            val = class_batch_counts[(label_name, batch)]
            count_rows.append(
                {
                    "label": label_name,
                    "batch": batch,
                    "spectra": val["spectra"],
                    "unique_samples": len(val["samples"]),
                }
            )

    duplicate_distribution = Counter(Counter(groups).values())
    sample_overlap = sorted(set(sample_names[y == 0]).intersection(set(sample_names[y == 1])))

    quality = {
        "n_files_loaded": len(rows),
        "n_errors": len(errors),
        "wavelength_start_nm": float(wavelength[0]),
        "wavelength_end_nm": float(wavelength[-1]),
        "wavelength_step_nm": float(np.median(np.diff(wavelength))),
        "n_wavelengths": int(len(wavelength)),
        "absorbance_min": float(x_full.min()),
        "absorbance_max": float(x_full.max()),
        "near_flat_spectra": int((x_full.std(axis=1) < 1e-5).sum()),
        "duplicate_measurement_group_size_distribution": {str(k): v for k, v in sorted(duplicate_distribution.items())},
        "sample_names_shared_across_labels": sample_overlap,
    }

    results = {
        "quality": quality,
        "counts": count_rows,
        "naive_random_cv": naive_metrics,
        "sample_grouped_cv": grouped_metrics,
        "leave_one_batch_out": lobo_rows,
        "leave_one_batch_out_combined": lobo_combined,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "top_consistent_wavelengths": bands,
        "method": {
            "preprocessing": f"SNV + Savitzky-Golay first derivative (31 points, polyorder 2), 1050-{model_wl[-1]:.0f} nm, 2 nm spacing",
            "classifier": "StandardScaler + class-balanced linear SVM (C=0.03)",
            "random_cv": "5-fold stratified",
            "sample_grouped_cv": "5-fold stratified group CV by label + sample name",
            "batch_cv": "Leave-one-batch-out on DN/DW/DV; DY excluded because no abnormal examples",
        },
        "errors": errors,
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(name, fieldnames, data):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv("sample_counts.csv", ["label", "batch", "spectra", "unique_samples"], count_rows)
    write_csv("spectra_means.csv", list(spectra_rows[0]), spectra_rows)
    write_csv(
        "pca_points.csv",
        ["label", "batch", "sample_name", "pc1", "pc2"],
        [
            {
                "label": rows[i]["label_name"],
                "batch": batches[i],
                "sample_name": sample_names[i],
                "pc1": float(pcs[i, 0]),
                "pc2": float(pcs[i, 1]),
            }
            for i in selected
        ],
    )
    write_csv(
        "validation_metrics.csv",
        ["validation", "test_batch", "train_batches", "n", "n_normal", "n_abnormal", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "tn", "fp", "fn", "tp"],
        [
            {"validation": "随机5折", "test_batch": "混合", **naive_metrics},
            {"validation": "按样品分组5折", "test_batch": "混合", **grouped_metrics},
            *[{"validation": "留一批次", **r} for r in lobo_rows],
            {"validation": "留一批次", **lobo_combined},
        ],
    )
    write_csv(
        "top_wavelengths.csv",
        ["wavelength_nm", "robust_effect", "direction_consistency", "d_DN", "d_DW", "d_DV"],
        bands,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
