"""光谱正常/异常最强集成分类器(全分辨率特征 + 多模型 OOF 集成)。

背景:
    data/ 里 98% 的"异常"样本与某个"正常"样本光谱几乎完全相同(余弦相似度
    >0.9999),原始标签下纯光谱分类准确率上限约 94~96%。本脚本用尽模型侧手段
    (全分辨率 SNV+导数、宽 MLP/CNN/GRU/Transformer 异质集成、OOF 阈值优化)
    逼近该上限。

用法:
    HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src/moment \
      /root/miniconda3/bin/python src/lings/train_ensemble.py

产物(outputs/ensemble/):
    oof_probs.npy         每样本 OOF 集成概率(用于离线评估)
    evaluation.json       OOF 评估指标(多阈值)
    checkpoints/*.pth     各模型在全量数据上训练的最终权重
    inference 说明        用 final 模型推理的方式见 evaluation.json 的说明字段
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from analysis_spectral import load_data
from train_1d_cnn import (
    make_loader,
    metrics,
    normalize_from_train,
    predict,
    seed_everything,
)
from train_multi_model import SpectralGRU, SpectralTransformer, train_model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ensemble"


# ---------------------------------------------------------------------------
# 全分辨率特征: SNV + 一阶导数双通道 [N, 2, 2401]
# ---------------------------------------------------------------------------
def build_fullres_inputs(x_full: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    mean = x_full.mean(axis=1, keepdims=True)
    std = x_full.std(axis=1, keepdims=True)
    snv = (x_full - mean) / np.maximum(std, 1e-8)
    deriv = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    mask = (wavelength >= 1050) & (wavelength <= 2450)
    return np.stack([snv[:, mask], deriv[:, mask]], axis=1).astype(np.float32)


class WideMLP(nn.Module):
    """展平双通道全分辨率输入的宽 MLP。"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FullCNN(nn.Module):
    """全分辨率双通道 4 层 1D CNN。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, 15, padding=7), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def make_model(name: str, dim: int) -> nn.Module:
    if name == "mlp":
        return WideMLP(dim)
    if name == "cnn":
        return FullCNN()
    if name == "gru":
        return SpectralGRU()          # 期望 [N,2,601] 输入
    if name == "transformer":
        return SpectralTransformer()  # 期望 [N,2,601] 输入
    raise ValueError(name)


# 每折参与集成的模型(名称, 随机种子)
ENSEMBLE = [
    ("mlp", 1), ("mlp", 2), ("mlp", 3),
    ("cnn", 1), ("cnn", 2),
    ("gru", 1),
    ("transformer", 1),
]
MAX_EPOCHS = 40
PATIENCE = 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    rows, wl, x_full, errors = load_data()
    if errors:
        print(f"⚠️ {len(errors)} 个文件解析失败,已跳过", flush=True)
    y = np.array([r["label"] for r in rows])
    x_fullres = build_fullres_inputs(x_full, wl)
    from train_1d_cnn import build_inputs
    x_601, _ = build_inputs(x_full, wl)
    print(f"数据: {len(y)} 条(正常 {(y == 0).sum()} / 异常 {(y == 1).sum()}),"
          f"全分辨率 {tuple(x_fullres.shape)}, 601点 {tuple(x_601.shape)}", flush=True)

    # ---------- OOF 集成评估 ----------
    n_models = len(ENSEMBLE)
    oof = np.zeros((len(y), n_models), dtype=np.float64)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf.split(x_fullres, y), start=1):
        tr2, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42 + fold)
        for m, (name, seed) in enumerate(ENSEMBLE):
            x_use = x_fullres if name in ("mlp", "cnn") else x_601
            seed_everything(1000 + fold * 100 + seed)
            model = make_model(name, x_use.shape[1] * x_use.shape[2])
            out = train_model(
                model, x_use, y, tr2, va, device,
                batch_size=256, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                seed=1000 + fold * 100 + seed, oversample=3, aug_noise=0.02,
            )
            loader = make_loader(x_use, y, te, out["center"], out["scale"], 1024, False)
            _, p = predict(model, loader, device)
            oof[te, m] = p
        print(f"fold {fold}/{5} 完成 ({time.time() - t_start:.0f}s)", flush=True)

    p_mean = oof.mean(axis=1)
    np.save(OUT / "oof_probs.npy", oof)
    np.save(OUT / "oof_labels.npy", y)

    # 阈值扫描
    fprs, tprs, ths = roc_curve(y, p_mean)
    candidates = {}
    for t in ths:
        pred = (p_mean >= t).astype(int)
        r = metrics(y, p_mean, t)
        r["auprc"] = float(average_precision_score(y, p_mean))
        r["g_mean"] = float(np.sqrt(r["sensitivity"] * r["specificity"]))
        candidates[float(t)] = r
    best_f1 = max(candidates, key=lambda t: candidates[t]["f1"])
    best_gmean = max(candidates, key=lambda t: candidates[t]["g_mean"])
    best_acc99 = max(candidates, key=lambda t: (candidates[t]["accuracy"] > 0.99, candidates[t]["f1"]))
    summary = {
        "protocol": "5折分层 OOF, 7模型概率平均集成(3xWideMLP+2xCNN+GRU+Transformer)",
        "n": int(len(y)),
        "f1最优阈值": {"threshold": best_f1, "metrics": candidates[best_f1]},
        "gmean最优阈值": {"threshold": best_gmean, "metrics": candidates[best_gmean]},
        "acc>99%可达性": {
            "max_accuracy": max(c["accuracy"] for c in candidates.values()),
            "at_max_acc": candidates[max(candidates, key=lambda t: candidates[t]["accuracy"])],
        },
    }
    (OUT / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    r = candidates[best_f1]
    print(json.dumps({
        "OOF集成": {"threshold": round(best_f1, 4),
                    "accuracy": round(r["accuracy"], 4),
                    "balanced_accuracy": round(r["balanced_accuracy"], 4),
                    "sensitivity": round(r["sensitivity"], 4),
                    "specificity": round(r["specificity"], 4),
                    "f1": round(r["f1"], 4),
                    "roc_auc": round(r["roc_auc"], 4),
                    "auprc": round(r["auprc"], 4),
                    "g_mean": round(r["g_mean"], 4),
                    "tn": r["tn"], "fp": r["fp"], "fn": r["fn"], "tp": r["tp"]}},
        ensure_ascii=False), flush=True)

    # ---------- 全量数据最终模型 ----------
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for m, (name, seed) in enumerate(ENSEMBLE):
        x_use = x_fullres if name in ("mlp", "cnn") else x_601
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=777)
        seed_everything(2000 + seed)
        model = make_model(name, x_use.shape[1] * x_use.shape[2])
        out = train_model(
            model, x_use, y, tr, va, device,
            batch_size=256, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            seed=2000 + seed, oversample=3, aug_noise=0.02,
        )
        torch.save({
            "name": name, "seed": seed,
            "feature": "fullres" if name in ("mlp", "cnn") else "601",
            "state_dict": out["state"], "center": out["center"], "scale": out["scale"],
            "val_auc": out["best_auc"], "threshold": out["threshold"],
        }, ckpt_dir / f"{name}_{seed}.pth")
        print(f"final {name}#{seed} valAUC={out['best_auc']:.4f}", flush=True)

    print(f"⏱️ 总耗时 {time.time() - t_start:.0f}s | 结果: {OUT}")


if __name__ == "__main__":
    main()
