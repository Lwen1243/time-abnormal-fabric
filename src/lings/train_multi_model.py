"""多模型对比训练:PyTorch 实现多个模型对光谱样本做正常/异常二分类。

模型集合:
    mlp         SpectralMLP         全连接网络(展平 2 通道光谱)
    cnn         SpectralCNN         1D 卷积网络(与 train_1d_cnn 同结构)
    gru         SpectralGRU         卷积降采样 + 双向 GRU
    transformer SpectralTransformer 卷积降采样 + Transformer Encoder

评估协议(与 train_1d_cnn 的 random_stratified_cv 一致):
    随机分层 5 折交叉验证,每折 15% 留作验证(早停),再在全体数据上
    按 85/15 重训一个最终模型。

产物(默认 outputs/multimodel/):
    checkpoints/<模型>_fold<k>.pth   每折最佳模型
    checkpoints/<模型>_final.pth     最终模型(含归一化参数与阈值,可直接推理)
    comparison.csv / comparison.json 全部指标
    history/<模型>.json              每折训练曲线

用法:
    uv run python train_multi_model.py
    uv run python train_multi_model.py --models mlp,cnn --folds 3 --max-epochs 15
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from analysis_spectral import load_data
from train_1d_cnn import (
    build_inputs,
    make_loader,
    metrics,
    normalize_from_train,
    predict,
    safe_auc,
    seed_everything,
    summarize_folds,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "multimodel"


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
class SpectralMLP(nn.Module):
    def __init__(self, input_points: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * input_points, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class SpectralCNN_MPS(nn.Module):
    """与 train_1d_cnn.SpectralCNN 同结构的 1D CNN,但用全局平均池化
    替代 AdaptiveAvgPool1d(16),以兼容 MPS(不支持非整除输入)。"""

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
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class SpectralCNN_Tuned(nn.Module):
    """加深版 1D CNN(4 层卷积 + 逐级池化 + 全局平均池化),MPS 兼容。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class SpectralGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.gru = nn.GRU(16, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(2 * 64 + 2 * 64, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)          # [N, L', 16]
        out, _ = self.gru(h)                        # [N, L', 128]
        pooled = torch.cat([out[:, -1], out.mean(dim=1)], dim=1)  # 末步 + 平均
        return self.head(pooled).squeeze(1)


class SpectralTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=128,
            dropout=0.15, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)          # [N, L', 32]
        h = self.encoder(h)
        pooled = h.mean(dim=1)                      # 全局平均池化
        return self.head(pooled).squeeze(1)


def model_factory(name: str, input_points: int) -> nn.Module:
    if name == "mlp":
        return SpectralMLP(input_points)
    if name == "cnn":
        return SpectralCNN_Tuned()
    if name == "gru":
        return SpectralGRU()
    if name == "transformer":
        return SpectralTransformer()
    raise ValueError(f"未知模型: {name}")


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def choose_threshold_f1(y_true: np.ndarray, probability: np.ndarray) -> float:
    """选择使 F1 最大的阈值(不平衡分类比 Youden 更合适)。"""
    _, _, thresholds = roc_curve(y_true, probability)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        if not np.isfinite(t):
            continue
        pred = (probability >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    aug_noise: float = 0.0,
    oversample: int = 1,
):
    """训练单个模型,按验证集 AUC 早停,返回最佳状态与历史。

    aug_noise > 0 时在训练输入上叠加高斯噪声(正则化/增强)。
    oversample > 1 时把训练集中的异常样本重复 oversample 倍(不平衡过采样,
    配合噪声增强)。
    """
    seed_everything(seed)
    if oversample > 1:
        pos = train_idx[y[train_idx] == 1]
        train_idx = np.concatenate([train_idx, np.tile(pos, oversample - 1)])
    center, scale = normalize_from_train(x, train_idx)
    train_loader = make_loader(x, y, train_idx, center, scale, batch_size, True)
    val_loader = make_loader(x, y, val_idx, center, scale, batch_size * 2, False)

    model = model.to(device)
    n_pos = int((y[train_idx] == 1).sum())
    n_neg = int((y[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_state = None
    best_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            if len(xb) <= 1:  # BatchNorm 需要 batch > 1,跳过末尾单样本 batch
                continue
            xb, yb = xb.to(device), yb.to(device)
            if aug_noise > 0:
                xb = xb + torch.randn_like(xb) * aug_noise
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)
        scheduler.step()

        val_labels, val_probability = predict(model, val_loader, device)
        val_auc = safe_auc(val_labels, val_probability)
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_n, 1), "val_auc": val_auc})

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
    threshold = choose_threshold_f1(val_labels, val_probability)
    return {
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "center": center,
        "scale": scale,
        "state": best_state,
        "history": history,
    }


def evaluate(model, x, y, test_idx, center, scale, device, batch_size, threshold):
    test_loader = make_loader(x, y, test_idx, center, scale, batch_size * 2, False)
    test_labels, test_probability = predict(model, test_loader, device)
    result = metrics(test_labels, test_probability, threshold)
    # 类别不平衡场景下更有意义的指标
    if len(np.unique(test_labels)) == 2:
        result["auprc"] = float(average_precision_score(test_labels, test_probability))
    else:
        result["auprc"] = float("nan")
    result["g_mean"] = float(np.sqrt(result["sensitivity"] * result["specificity"]))
    return result, test_probability


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="多模型光谱二分类对比训练")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default="mlp,cnn,gru,transformer",
                        help="逗号分隔的模型列表: mlp,cnn,gru,transformer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=3,
                        help="异常样本过采样倍数(>=1,默认 3)")
    parser.add_argument("--aug-noise", type=float, default=0.02,
                        help="训练输入高斯噪声强度(默认 0.02)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-filter", default="",
                        help="只使用文件名含此子串的样本(如 --file-filter 采谱)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}", flush=True)

    rows, wavelength, x_full, errors = load_data()
    if errors:
        print(f"⚠️ {len(errors)} 个文件解析失败(空壳/格式异常),已跳过:", flush=True)
        for e in errors[:5]:
            print(f"   - {e['file']}: {e['reason']}", flush=True)
    if args.file_filter:
        keep = np.asarray([args.file_filter in r["file"] for r in rows])
        rows = [r for i, r in enumerate(rows) if keep[i]]
        x_full = x_full[keep]
        print(f"按文件名过滤 '{args.file_filter}': 保留 {len(rows)} 条", flush=True)
    x, model_wavelength = build_inputs(x_full, wavelength)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    input_points = x.shape[2]
    print(f"数据: {len(y)} 条光谱(正常 {(y == 0).sum()} / 异常 {(y == 1).sum()}),输入形状 {tuple(x.shape)}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fold_records: list[dict] = []
    final_records: list[dict] = []

    for name in models:
        print(f"\n{'=' * 70}\n模型: {name}\n{'=' * 70}", flush=True)
        model_records = []
        fold_histories = []
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        for fold, (train_pool, test_idx) in enumerate(splitter.split(x, y), start=1):
            train_idx, val_idx = train_test_split(
                train_pool, test_size=0.15, stratify=y[train_pool],
                random_state=args.seed + fold,
            )
            t0 = time.time()
            model = model_factory(name, input_points)
            outcome = train_model(
                model, x, y, train_idx, val_idx, device,
                args.batch_size, args.max_epochs, args.patience, args.seed + fold,
                aug_noise=args.aug_noise, oversample=args.oversample,
            )
            result, _ = evaluate(
                model, x, y, test_idx, outcome["center"], outcome["scale"],
                device, args.batch_size, outcome["threshold"],
            )
            result.update({
                "model": name,
                "fold": fold,
                "best_epoch": outcome["best_epoch"],
                "val_auc": outcome["best_auc"],
                "seconds": round(time.time() - t0, 1),
                "parameters": sum(p.numel() for p in model.parameters()),
            })
            model_records.append(result)
            fold_records.append(result)
            fold_histories.append(outcome["history"])

            save_checkpoint(
                args.output_dir / "checkpoints" / f"{name}_fold{fold}.pth",
                {
                    "model_class": type(model).__name__,
                    "model_state_dict": outcome["state"],
                    "normalization_center": torch.from_numpy(outcome["center"]),
                    "normalization_scale": torch.from_numpy(outcome["scale"]),
                    "wavelength_nm": torch.from_numpy(model_wavelength),
                    "threshold": outcome["threshold"],
                    "validation": {k: result[k] for k in ("balanced_accuracy", "roc_auc", "f1")},
                },
            )
            print(json.dumps({
                "model": name, "fold": fold,
                "val_auc": round(result["val_auc"], 4),
                "test_roc_auc": round(result["roc_auc"], 4),
                "test_balanced_accuracy": round(result["balanced_accuracy"], 4),
                "test_f1": round(result["f1"], 4),
                "best_epoch": result["best_epoch"],
                "seconds": result["seconds"],
            }, ensure_ascii=False), flush=True)

        # 最终模型:全体数据 85/15 重训
        full_idx = np.arange(len(y))
        final_train_idx, final_val_idx = train_test_split(
            full_idx, test_size=0.15, stratify=y, random_state=args.seed + 999,
        )
        model = model_factory(name, input_points)
        outcome = train_model(
            model, x, y, final_train_idx, final_val_idx, device,
            args.batch_size, args.max_epochs, args.patience, args.seed + 999,
            aug_noise=args.aug_noise, oversample=args.oversample,
        )
        final_result, _ = evaluate(
            model, x, y, final_val_idx, outcome["center"], outcome["scale"],
            device, args.batch_size, outcome["threshold"],
        )
        final_result.update({
            "model": name,
            "fold": "final",
            "best_epoch": outcome["best_epoch"],
            "val_auc": outcome["best_auc"],
            "parameters": sum(p.numel() for p in model.parameters()),
        })
        final_records.append(final_result)
        fold_records.append(final_result)

        save_checkpoint(
            args.output_dir / "checkpoints" / f"{name}_final.pth",
            {
                "model_class": type(model).__name__,
                "model_state_dict": outcome["state"],
                "normalization_center": torch.from_numpy(outcome["center"]),
                "normalization_scale": torch.from_numpy(outcome["scale"]),
                "wavelength_nm": torch.from_numpy(model_wavelength),
                "threshold": outcome["threshold"],
                "preprocessing": {
                    "channels": ["SNV absorbance", "Savitzky-Golay first derivative of SNV"],
                    "source_range_nm": [float(wavelength[0]), float(wavelength[-1])],
                    "model_range_nm": [float(model_wavelength[0]), float(model_wavelength[-1])],
                    "spacing_nm": float(np.median(np.diff(model_wavelength))),
                },
                "validation": final_result,
            },
        )
        print(json.dumps({"model": name, "final": final_result}, ensure_ascii=False), flush=True)

        # 写训练历史(每折的 train_loss / val_auc 曲线)
        (args.output_dir / "history").mkdir(exist_ok=True)
        (args.output_dir / "history" / f"{name}.json").write_text(
            json.dumps({"folds": fold_histories}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ 汇总
    comparison = []
    for name in models:
        recs = [r for r in fold_records if r["model"] == name and r["fold"] != "final"]
        summary = {"model": name, "folds": len(recs), "parameters": recs[0]["parameters"]}
        summary.update(summarize_folds(recs))
        for k in ("auprc", "g_mean"):
            vals = [r[k] for r in recs]
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        final = next(r for r in final_records if r["model"] == name)
        summary["final_accuracy"] = final["accuracy"]
        summary["final_balanced_accuracy"] = final["balanced_accuracy"]
        summary["final_roc_auc"] = final["roc_auc"]
        summary["final_f1"] = final["f1"]
        summary["final_auprc"] = final["auprc"]
        summary["best_epoch"] = final["best_epoch"]
        comparison.append(summary)

    write_csv(args.output_dir / "comparison.csv", fold_records)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"models": comparison, "folds": fold_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印对比表(突出不平衡场景下有意义的指标)
    keys = ["balanced_accuracy", "precision", "sensitivity", "f1", "auprc", "g_mean", "roc_auc"]
    zh = {"balanced_accuracy": "平衡准确率", "precision": "精确率", "sensitivity": "召回率",
          "f1": "F1", "auprc": "AUPRC", "g_mean": "G-mean", "roc_auc": "ROC-AUC"}
    baseline_acc = 1.0 - float(y.mean())
    print("\n" + "=" * 112)
    print(f"⚠️ 数据不平衡: 异常率 {y.mean():.1%}。若全预测「正常」,准确率即达 {baseline_acc:.1%},故准确率仅供参考。")
    print(f"   调优: 异常样本过采样 {args.oversample}x + 噪声增强 + F1 最优阈值")
    print("=" * 112)
    header = f"{'模型':<12}{'参数量':>10}" + "".join(f"{zh[k]:>14}" for k in keys)
    print(header)
    for row in comparison:
        line = f"{row['model']:<12}{row['parameters']:>10}"
        for k in keys:
            m, s = row[k]["mean"], row[k]["std"]
            line += f"{m:>10.4f}±{s:<3.4f}"
        print(line)
    print("=" * 112)
    print("5 折分层交叉验证均值±标准差")
    print()
    print("最终模型(85/15 训练/验证):")
    for row in comparison:
        print(f"  {row['model']:<12} BalAcc={row['final_balanced_accuracy']:.4f}  F1={row['final_f1']:.4f}  "
              f"AUPRC={row['final_auprc']:.4f}  AUC={row['final_roc_auc']:.4f}  最佳epoch={row['best_epoch']}")
    print(f"\n全部结果: {args.output_dir}")


if __name__ == "__main__":
    main()
