"""测量状态分类器(Finished vs Recalculated)集成训练。

背景:
    光谱真正可判的维度是元数据 `Subsample status`(Finished/Recalculated),
    单模型随机划分验证 AUC≈0.9965。本脚本训练异质集成并做 5 折 OOF 评估,
    并保存全量数据最终模型。

用法:
    /root/miniconda3/bin/python src/lings/train_status_ensemble.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from analysis_spectral import load_data
from train_1d_cnn import make_loader, metrics, normalize_from_train, predict, seed_everything
from train_ensemble import FullCNN, WideMLP, build_fullres_inputs
from train_multi_model import train_model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "status_ensemble"


def load_status_labels(rows) -> np.ndarray:
    """从文件元数据读取 Subsample status, Recalculated=1, Finished=0。"""
    labels = []
    for r in rows:
        p = ROOT / r["file"]
        v = "?"
        for line in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[:8]:
            if line.startswith("Subsample status;"):
                v = line.split(";", 1)[1].strip()
                break
        labels.append(1 if v == "Recalculated" else 0)
    return np.asarray(labels, dtype=int)


ENSEMBLE = [("mlp", 1), ("mlp", 2), ("mlp", 3), ("cnn", 1), ("cnn", 2)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, wl, x_full, errors = load_data()
    y = load_status_labels(rows)
    x = build_fullres_inputs(x_full, wl)
    print(f"数据: {len(y)} 条 | Finished {(y == 0).sum()} / Recalculated {(y == 1).sum()} | {tuple(x.shape)}", flush=True)

    n_models = len(ENSEMBLE)
    oof = np.zeros((len(y), n_models), dtype=np.float64)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf.split(x, y), start=1):
        tr2, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42 + fold)
        for m, (name, seed) in enumerate(ENSEMBLE):
            seed_everything(3000 + fold * 100 + seed)
            model = WideMLP(x.shape[1] * x.shape[2]) if name == "mlp" else FullCNN()
            out = train_model(model, x, y, tr2, va, device, batch_size=256, max_epochs=40,
                              patience=10, seed=3000 + fold * 100 + seed, oversample=2, aug_noise=0.01)
            loader = make_loader(x, y, te, out["center"], out["scale"], 1024, False)
            _, p = predict(model, loader, device)
            oof[te, m] = p
        print(f"fold {fold}/5 ({time.time()-t0:.0f}s)", flush=True)

    p_mean = oof.mean(axis=1)
    np.save(OUT / "oof_probs.npy", oof)
    np.save(OUT / "oof_labels.npy", y)

    fprs, tprs, ths = roc_curve(y, p_mean)
    cands = {}
    for t in ths:
        r = metrics(y, p_mean, t)
        r["auprc"] = float(average_precision_score(y, p_mean))
        r["g_mean"] = float(np.sqrt(r["sensitivity"] * r["specificity"]))
        cands[float(t)] = r
    best_f1 = max(cands, key=lambda t: cands[t]["f1"])
    best_acc = max(cands, key=lambda t: cands[t]["accuracy"])
    summary = {
        "protocol": "5折分层 OOF, 5模型集成(3xWideMLP+2xCNN), 标签=Subsample status",
        "n": int(len(y)),
        "f1最优": {"threshold": best_f1, "metrics": cands[best_f1]},
        "准确率最优": {"threshold": best_acc, "metrics": cands[best_acc]},
        "max_accuracy": float(cands[best_acc]["accuracy"]),
    }
    (OUT / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    r = cands[best_f1]
    print(json.dumps({"OOF集成": {"threshold": round(best_f1, 4), "accuracy": round(r["accuracy"], 4),
                                  "sensitivity": round(r["sensitivity"], 4),
                                  "specificity": round(r["specificity"], 4), "f1": round(r["f1"], 4),
                                  "roc_auc": round(r["roc_auc"], 4), "auprc": round(r["auprc"], 4)}},
                     ensure_ascii=False), flush=True)

    # 全量最终模型
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for m, (name, seed) in enumerate(ENSEMBLE):
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=777)
        seed_everything(4000 + seed)
        model = WideMLP(x.shape[1] * x.shape[2]) if name == "mlp" else FullCNN()
        out = train_model(model, x, y, tr, va, device, batch_size=256, max_epochs=40,
                          patience=10, seed=4000 + seed, oversample=2, aug_noise=0.01)
        torch.save({"name": name, "seed": seed, "state_dict": out["state"],
                    "center": out["center"], "scale": out["scale"],
                    "val_auc": out["best_auc"], "threshold": out["threshold"]},
                   ckpt_dir / f"{name}_{seed}.pth")
        print(f"final {name}#{seed} valAUC={out['best_auc']:.4f}", flush=True)

    print(f"⏱️ 总耗时 {time.time()-t0:.0f}s | 结果: {OUT}")


if __name__ == "__main__":
    main()
