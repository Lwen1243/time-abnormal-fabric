"""用已保存的模型权重(.pth)在新数据上推理评估(跨数据泛化测试)。

将旧数据训练得到的 `*_final.pth` 直接用于当前 data/ 下的全部数据,
并按数据子集(采谱 / 原始数据 / 各批次)分别报告指标。

用法:
    uv run python src/lings/evaluate_checkpoint_on_data.py
    uv run python src/lings/evaluate_checkpoint_on_data.py --checkpoints-dir outputs/multimodel_tuned/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from analysis_spectral import load_data
from train_1d_cnn import build_inputs, make_loader, metrics, predict
from train_multi_model import (
    SpectralCNN_MPS,
    SpectralCNN_Tuned,
    SpectralGRU,
    SpectralMLP,
    SpectralTransformer,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = ROOT / "outputs" / "multimodel_tuned" / "checkpoints"


def build_model(model_class: str, input_points: int):
    """按 checkpoint 中记录的结构名重建模型。"""
    if model_class == "SpectralMLP":
        return SpectralMLP(input_points)
    if model_class == "SpectralCNN_Tuned":
        return SpectralCNN_Tuned()
    if model_class == "SpectralCNN_MPS":
        return SpectralCNN_MPS()
    if model_class == "SpectralGRU":
        return SpectralGRU()
    if model_class == "SpectralTransformer":
        return SpectralTransformer()
    raise ValueError(f"未知模型结构: {model_class}")


def report(y_true, probability, threshold):
    result = metrics(y_true, probability, threshold)
    if len(np.unique(y_true)) == 2:
        result["auprc"] = float(average_precision_score(y_true, probability))
    else:
        result["auprc"] = float("nan")
    result["g_mean"] = float(np.sqrt(max(result["sensitivity"], 0) * max(result["specificity"], 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description="旧权重在新数据上的推理评估")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--models", default="mlp,cnn,gru,transformer")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "old_models_on_newdata.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cpu")
    rows, wavelength, x_full, errors = load_data()
    if errors:
        print(f"⚠️ {len(errors)} 个文件解析失败,已跳过", file=sys.stderr)
    x, model_wavelength = build_inputs(x_full, wavelength)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    batch_arr = np.asarray([r["batch"] for r in rows])
    is_raw = np.asarray(["原始数据" in r["file"] for r in rows])
    is_caipu = ~is_raw
    all_idx = np.arange(len(y))

    print(f"数据: {len(y)} 条(正常 {int((y == 0).sum())} / 异常 {int((y == 1).sum())})")
    print(f"模型权重目录: {args.checkpoints_dir}\n")

    records = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        ckpt_path = args.checkpoints_dir / f"{name}_final.pth"
        if not ckpt_path.is_file():
            print(f"⚠️ 缺少 {ckpt_path},跳过 {name}", file=sys.stderr)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model(ckpt["model_class"], x.shape[2])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        loader = make_loader(x, y, all_idx, ckpt["normalization_center"].numpy(),
                             ckpt["normalization_scale"].numpy(), args.batch_size, False)
        labels, probability = predict(model, loader, device)
        threshold = float(ckpt["threshold"])
        print(f"==== 模型: {name} (老权重, 阈值 {threshold:.4f}) ====")

        subsets = [
            ("全部数据", all_idx),
            ("仅采谱", np.flatnonzero(is_caipu)),
            ("仅原始数据", np.flatnonzero(is_raw)),
        ]
        for b in sorted(set(batch_arr)):
            subsets.append((f"批次 {b}", np.flatnonzero(batch_arr == b)))

        for subset_name, idx in subsets:
            r = report(labels[idx], probability[idx], threshold)
            r["model"] = name
            r["subset"] = subset_name
            r["n"] = int(len(idx))
            r["threshold"] = threshold
            records.append(r)
            print(
                f"  {subset_name:<10} n={r['n']:>4}  Acc={r['accuracy']:.4f}  BalAcc={r['balanced_accuracy']:.4f}  "
                f"Prec={r['precision']:.4f}  Rec={r['sensitivity']:.4f}  F1={r['f1']:.4f}  "
                f"AUPRC={r['auprc']:.4f}  AUC={r['roc_auc']:.4f}"
            )
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "subset", "n", "n_normal", "n_abnormal", "accuracy", "balanced_accuracy", "precision",
                  "sensitivity", "specificity", "f1", "auprc", "g_mean", "roc_auc",
                  "tn", "fp", "fn", "tp", "threshold"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"✅ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
