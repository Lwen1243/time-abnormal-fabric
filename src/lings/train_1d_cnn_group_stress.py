from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from analysis_spectral import load_data
from train_1d_cnn import build_inputs, combined_metrics, summarize_folds, train_one


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Group-aware stress tests for the spectral 1D CNN.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "cnn1d_group_stress")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def specification_group(sample_name: str):
    value = re.sub(r"^\d+-", "", sample_name.strip())
    value = re.sub(r"(?:[-+]\d{1,2}|间僵)$", "", value)
    return value


def acquisition_date(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8-sig") as f:
        for _ in range(6):
            line = f.readline().strip()
            if line.startswith("Determination start;"):
                return line.split(";", 1)[1][:10]
    return "unknown"


def grouped_inner_split(train_pool, y, groups, folds, seed):
    n_groups = len(np.unique(groups[train_pool]))
    n_splits = max(2, min(folds, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = []
    for inner_train_rel, val_rel in splitter.split(np.zeros(len(train_pool)), y[train_pool], groups[train_pool]):
        inner_train = train_pool[inner_train_rel]
        val = train_pool[val_rel]
        if len(np.unique(y[inner_train])) == 2 and len(np.unique(y[val])) == 2:
            candidates.append((inner_train, val))
    if not candidates:
        raise RuntimeError("Could not create a group-preserving validation split containing both classes")
    # Choose the candidate closest to the overall anomaly rate.
    target = float(y[train_pool].mean())
    return min(candidates, key=lambda pair: abs(float(y[pair[1]].mean()) - target))


def run_protocol(name, x, y, groups, args, device):
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    probability = np.full(len(y), np.nan, dtype=float)
    prediction = np.full(len(y), -1, dtype=int)
    fold_id = np.full(len(y), -1, dtype=int)
    records = []
    history = []
    assignments = []

    for fold, (train_pool, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        train_idx, val_idx = grouped_inner_split(train_pool, y, groups, args.folds, args.seed + fold)
        result, fold_probability, _, _, _, fold_history = train_one(
            x,
            y,
            train_idx,
            val_idx,
            test_idx,
            device,
            args.batch_size,
            args.max_epochs,
            args.patience,
            args.seed + fold,
            name,
            f"fold_{fold}",
        )
        fold_prediction = (fold_probability >= result["threshold"]).astype(int)
        result["average_precision"] = float(average_precision_score(y[test_idx], fold_probability))
        result["mcc"] = float(matthews_corrcoef(y[test_idx], fold_prediction))
        result["test_groups"] = int(len(np.unique(groups[test_idx])))
        result["train_groups"] = int(len(np.unique(groups[train_idx])))
        probability[test_idx] = fold_probability
        prediction[test_idx] = fold_prediction
        fold_id[test_idx] = fold
        records.append(result)
        history.extend(fold_history)
        assignments.extend(
            {
                "index": int(i),
                "fold": fold,
                "group": str(groups[i]),
                "probability": float(p),
                "prediction": int(q),
            }
            for i, p, q in zip(test_idx, fold_probability, fold_prediction)
        )
        print(json.dumps({"completed": result}, ensure_ascii=False), flush=True)

    combined = combined_metrics(y, probability, prediction)
    combined["protocol"] = name + "_combined"
    combined["average_precision"] = float(average_precision_score(y, probability))
    combined["mcc"] = float(matthews_corrcoef(y, prediction))
    combined["macro_fold_summary"] = summarize_folds(records)
    combined["macro_average_precision"] = float(np.mean([r["average_precision"] for r in records]))
    combined["macro_mcc"] = float(np.mean([r["mcc"] for r in records]))
    return records, combined, history, assignments


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

    rows, wavelength, x_full, errors = load_data()
    if errors:
        raise RuntimeError(f"Failed to parse {len(errors)} files")
    x, model_wavelength = build_inputs(x_full, wavelength)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    specification_groups = np.asarray([specification_group(r["sample_name"]) for r in rows])
    date_groups = np.asarray([acquisition_date(r["file"]) for r in rows])
    device = torch.device("cpu")

    results = {
        "dataset": {
            "spectra": int(len(y)),
            "specification_groups": int(len(np.unique(specification_groups))),
            "acquisition_dates": int(len(np.unique(date_groups))),
            "abnormal_rate": float(y.mean()),
            "model_wavelength_nm": [float(model_wavelength[0]), float(model_wavelength[-1])],
        },
        "protocols": {},
    }
    all_metrics = []
    all_history = []
    all_assignments = []
    for name, groups in [
        ("specification_group_cv", specification_groups),
        ("acquisition_date_group_cv", date_groups),
    ]:
        records, combined, history, assignments = run_protocol(name, x, y, groups, args, device)
        results["protocols"][name] = {"folds": records, "combined": combined}
        all_metrics.extend(records)
        all_history.extend(history)
        for row in assignments:
            row["protocol"] = name
            all_assignments.append(row)

    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "fold_metrics.csv", all_metrics)
    write_csv(args.output_dir / "training_history.csv", all_history)
    write_csv(args.output_dir / "fold_assignments.csv", all_assignments)
    print(json.dumps({name: data["combined"] for name, data in results["protocols"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
