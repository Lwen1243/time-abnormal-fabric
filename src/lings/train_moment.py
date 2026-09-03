"""MOMENT 时间序列基础模型微调(光谱二分类)。

流程:
    1. 加载数据 → SNV+导数特征([N,2,601]) → 降采样到 MOMENT 要求的 512 点
    2. 从 HuggingFace 加载预训练 MOMENT(默认 MOMENT-1-small),冻结主干,训练分类头
    3. 5 折分层交叉验证 + 最终模型,指标与 train_multi_model 一致

用法:
    uv run python src/lings/train_moment.py
    uv run python src/lings/train_moment.py --model AutonLab/MOMENT-1-small --finetune
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from analysis_spectral import load_data
from train_1d_cnn import (
    build_inputs,
    metrics,
    safe_auc,
    seed_everything,
    summarize_folds,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "moment"
DEFAULT_MODEL = "AutonLab/MOMENT-1-small"
DEFAULT_SEQ_LEN = 512  # MOMENT 固定上下文长度


def downsample_to(x: np.ndarray, target: int = DEFAULT_SEQ_LEN) -> np.ndarray:
    """把 [N, C, L] 线性插值到 [N, C, target]。"""
    n, c, length = x.shape
    src = np.linspace(0.0, 1.0, length)
    dst = np.linspace(0.0, 1.0, target)
    out = np.empty((n, c, target), dtype=np.float32)
    for i in range(n):
        for j in range(c):
            out[i, j] = np.interp(dst, src, x[i, j])
    return out


def build_moment(model_id: str, device: torch.device):
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={
            "task_name": "classification",
            "n_channels": 2,
            "num_class": 2,
        },
    )
    model.init()
    # MPS 上 transformers 的 SDPA 注意力不支持 dropout,统一关闭(冻结主干也避免
    # dropout 在 train() 模式下带来扰动)。注意 T5 的注意力 dropout 是 float 属性。
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if hasattr(m, "dropout") and isinstance(getattr(m, "dropout"), float):
            m.dropout = 0.0
    return model.to(device)


def choose_threshold_f1_impl(y_true, probability):
    """选择使 F1 最大的阈值。"""
    _, _, thresholds = roc_curve(y_true, probability)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        if not np.isfinite(t):
            continue
        f1 = float(f1_score(y_true, (probability >= t).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


@torch.inference_mode()
def predict_logits(model, x_tensor, batch_size, device):
    model.eval()
    logits = []
    for i in range(0, len(x_tensor), batch_size):
        xb = x_tensor[i : i + batch_size].to(device)
        out = model(x_enc=xb)
        logits.append(out.logits.cpu().numpy())
    logits = np.concatenate(logits)
    prob = softmax2(logits)[:, 1]
    return prob


def softmax2(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def parse_args():
    parser = argparse.ArgumentParser(description="MOMENT 基础模型微调光谱二分类")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MOMENT 预训练模型 id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率(默认: 微调主干 2e-4 / 只训分类头 1e-3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune", action="store_true", help="微调整个模型(默认只训分类头)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 2e-4 if args.finetune else 1e-3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device} | 模型: {args.model}", flush=True)

    # ---------- 数据 ----------
    rows, wavelength, x_full, errors = load_data()
    if errors:
        print(f"⚠️ {len(errors)} 个文件解析失败,已跳过", flush=True)
    x, _ = build_inputs(x_full, wavelength)
    x = downsample_to(x, DEFAULT_SEQ_LEN)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    print(f"数据: {len(y)} 条(正常 {(y == 0).sum()} / 异常 {(y == 1).sum()}),输入 {tuple(x.shape)}", flush=True)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y.astype(np.int64))

    fold_records = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    for fold, (train_pool, test_idx) in enumerate(splitter.split(x, y), start=1):
        train_idx, val_idx = train_test_split(
            train_pool, test_size=0.15, stratify=y[train_pool], random_state=args.seed + fold,
        )
        t0 = time.time()
        model = build_moment(args.model, device)
        if not args.finetune:
            # 只训练分类头(主干保持冻结)
            head_params = [p for name, p in model.named_parameters() if p.requires_grad]
        else:
            head_params = [p for p in model.parameters()]
        print(f"  可训练参数: {sum(p.numel() for p in head_params):,}", flush=True)

        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        pos_frac = (y[train_idx] == 1).mean()
        weight = torch.tensor([1.0 - pos_frac, pos_frac], dtype=torch.float32, device=device)
        # 放大幅度:异常类权重 = 正常/异常 比例
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        best_state = None
        best_auc = -math.inf
        best_epoch = 0
        wait = 0
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx))
            total_loss, total_n = 0.0, 0
            for i in range(0, len(perm), args.batch_size):
                ids_t = perm[i : i + args.batch_size]
                ids_np = train_idx[ids_t.numpy()]
                if len(ids_np) <= 1:
                    continue
                xb = x_tensor[torch.from_numpy(ids_np)].to(device)
                yb = y_tensor[torch.from_numpy(ids_np)].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x_enc=xb)
                loss = criterion(out.logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(ids_np)
                total_n += len(ids_np)
            scheduler.step()

            val_prob = predict_logits(model, x_tensor[torch.from_numpy(val_idx)], args.batch_size * 2, device)
            val_auc = safe_auc(y[val_idx], val_prob)
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        model.load_state_dict(best_state)
        val_prob = predict_logits(model, x_tensor[torch.from_numpy(val_idx)], args.batch_size * 2, device)
        threshold = choose_threshold_f1_impl(y[val_idx], val_prob)
        test_prob = predict_logits(model, x_tensor[torch.from_numpy(test_idx)], args.batch_size * 2, device)
        result = metrics(y[test_idx], test_prob, threshold)
        result["auprc"] = float(average_precision_score(y[test_idx], test_prob)) if len(np.unique(y[test_idx])) == 2 else float("nan")
        result["g_mean"] = float(np.sqrt(result["sensitivity"] * result["specificity"]))
        result.update({
            "model": args.model.split("/")[-1],
            "fold": fold,
            "best_epoch": best_epoch,
            "val_auc": best_auc,
            "seconds": round(time.time() - t0, 1),
            "parameters": sum(p.numel() for p in model.parameters()),
        })
        fold_records.append(result)
        print(json.dumps({"fold": fold, "val_auc": round(best_auc, 4), "test_f1": round(result["f1"], 4),
                          "test_auc": round(result["roc_auc"], 4), "best_epoch": best_epoch}, ensure_ascii=False), flush=True)

        # 保存本折最佳模型
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": args.model,
                "model_state_dict": best_state,
                "threshold": threshold,
                "seq_len": DEFAULT_SEQ_LEN,
                "validation": result,
            },
            args.output_dir / "checkpoints" / f"moment_fold{fold}.pth",
        )

    # 最终模型
    full_idx = np.arange(len(y))
    final_train_idx, final_val_idx = train_test_split(
        full_idx, test_size=0.15, stratify=y, random_state=args.seed + 999,
    )
    model = build_moment(args.model, device)
    if not args.finetune:
        head_params = [p for name, p in model.named_parameters() if p.requires_grad]
    else:
        head_params = [p for p in model.parameters()]
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    best_state = None
    best_auc = -math.inf
    wait = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(final_train_idx))
        for i in range(0, len(perm), args.batch_size):
            ids = final_train_idx[perm[i : i + args.batch_size].numpy()]
            if len(ids) <= 1:
                continue
            xb = x_tensor[torch.from_numpy(ids)].to(device)
            yb = y_tensor[torch.from_numpy(ids)].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_enc=xb).logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_prob = predict_logits(model, x_tensor[torch.from_numpy(final_val_idx)], args.batch_size * 2, device)
        val_auc = safe_auc(y[final_val_idx], val_prob)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    val_prob = predict_logits(model, x_tensor[torch.from_numpy(final_val_idx)], args.batch_size * 2, device)
    threshold = choose_threshold_f1_impl(y[final_val_idx], val_prob)
    final_prob = predict_logits(model, x_tensor[torch.from_numpy(final_val_idx)], args.batch_size * 2, device)
    final_result = metrics(y[final_val_idx], final_prob, threshold)
    final_result["auprc"] = float(average_precision_score(y[final_val_idx], final_prob))
    final_result["g_mean"] = float(np.sqrt(final_result["sensitivity"] * final_result["specificity"]))
    final_result.update({"model": args.model.split("/")[-1], "fold": "final", "best_epoch": best_epoch,
                         "val_auc": best_auc, "parameters": sum(p.numel() for p in model.parameters())})
    fold_records.append(final_result)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": best_state,
            "threshold": threshold,
            "seq_len": DEFAULT_SEQ_LEN,
            "validation": final_result,
        },
        args.output_dir / "checkpoints" / "moment_final.pth",
    )

    # 汇总
    summary = {"model": args.model, "folds": args.folds,
               "parameters": sum(p.numel() for p in model.parameters())}
    summary.update(summarize_folds([r for r in fold_records if r["fold"] != "final"]))
    write_csv(args.output_dir / "comparison.csv", fold_records)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"summary": summary, "folds": fold_records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "final": final_result}, ensure_ascii=False), flush=True)
    print(f"⏱️ 总耗时 {time.time() - t_start:.1f}s | 结果: {args.output_dir}")


if __name__ == "__main__":
    main()
