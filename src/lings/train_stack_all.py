"""Train a diverse set of models on multiscale features and fuse them.

Strategy (mirrors top Kaggle approaches for this type of problem):
1. Multiscale features: SNV + 1st derivative + 2nd derivative (concatenated).
2. Diverse base models: LightGBM (3-seed bag), XGBoost, RandomForest,
   ExtraTrees, plus existing ConvNeXt-1D / MLP-ensemble OOF probabilities.
3. Fusion: (a) convex weight grid search over the strongest base models,
   (b) logistic-regression stacking with inner CV as a comparison.
4. Report the best configuration on 5-fold OOF (threshold = best F1).

Outputs (analysis/):
  oof_xgb_multiscale.npy, oof_rf_multiscale.npy, oof_et_multiscale.npy,
  oof_lgbm_multiscale.npy, oof_stack.npy, stack_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402

import lightgbm as lgb  # noqa: E402

SEED = 42


def load_multiscale():
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = (spectra - mean) / np.maximum(std, 1e-8)
    g1 = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    g2 = savgol_filter(snv, window_length=41, polyorder=3, deriv=2, delta=0.5, axis=1)
    mask = (wavelength >= 1050) & (wavelength <= 2450)
    idx = np.flatnonzero(mask)[::4]
    X = np.concatenate([snv[:, idx], g1[:, idx], g2[:, idx]], axis=1).astype(np.float32)
    return X, y, rows


def best_f1_metrics(y, scores):
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(scores, 4)):
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (scores >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(best_t),
        "acc": accuracy_score(y, pred),
        "f1": best_f1,
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auc": roc_auc_score(y, scores),
        "auprc": average_precision_score(y, scores),
        "cm": [int(tn), int(fp), int(fn), int(tp)],
    }


def oof_skf(estimator_factory, X, y, seed=SEED):
    """5-fold stratified OOF probabilities, one model per fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        if hasattr(est, "predict_proba"):
            oof[va] = est.predict_proba(X[va])[:, 1]
        else:
            raise TypeError("estimator must expose predict_proba")
    return oof


def run():
    print("loading data ...")
    X, y, rows = load_multiscale()
    print(f"X = {X.shape}, abnormal = {int(y.sum())}")

    print("training XGBoost (multiscale) ...")
    xgb_oof = oof_skf(
        lambda: XGBClassifier(
            n_estimators=800, learning_rate=0.04, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=20, random_state=SEED,
        ),
        X, y,
    )
    np.save(OUT / "oof_xgb_multiscale.npy", xgb_oof)
    print("XGBoost:", best_f1_metrics(y, xgb_oof)["auc"])

    print("training RandomForest (multiscale) ...")
    rf_oof = oof_skf(
        lambda: RandomForestClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=3,
            n_jobs=20, random_state=SEED, class_weight="balanced_subsample",
        ),
        X, y,
    )
    np.save(OUT / "oof_rf_multiscale.npy", rf_oof)
    print("RandomForest:", best_f1_metrics(y, rf_oof)["auc"])

    print("training ExtraTrees (multiscale) ...")
    et_oof = oof_skf(
        lambda: ExtraTreesClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=3,
            n_jobs=20, random_state=SEED, class_weight="balanced_subsample",
        ),
        X, y,
    )
    np.save(OUT / "oof_et_multiscale.npy", et_oof)
    print("ExtraTrees:", best_f1_metrics(y, et_oof)["auc"])

    print("training LightGBM 3-seed bag (multiscale) ...")
    lgbm_oofs = []
    for s in (42, 2024, 7):
        lgbm_oofs.append(
            oof_skf(
                lambda: lgb.LGBMClassifier(
                    n_estimators=1200, learning_rate=0.03, num_leaves=63,
                    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                    min_child_samples=30, reg_lambda=1.0, n_jobs=20,
                    random_state=s, verbose=-1,
                ),
                X, y,
            )
        )
    lgbm_ms = np.mean(lgbm_oofs, axis=0)
    np.save(OUT / "oof_lgbm_multiscale.npy", lgbm_ms)
    print("LightGBM multiscale bag:", best_f1_metrics(y, lgbm_ms)["auc"])

    # existing OOFs
    lgbm_bag = np.load(OUT / "oof_lgbm_bag.npy")
    cnn = np.load(OUT / "oof_cnn_convnext.npy")
    ens = np.load(OUT / "oof_ensemble_probs.npy").mean(axis=1)

    candidates = {
        "lgbm_ms": lgbm_ms,
        "lgbm_bag": lgbm_bag,
        "xgb": xgb_oof,
        "rf": rf_oof,
        "et": et_oof,
        "cnn": cnn,
        "mlp_ens": ens,
    }
    print("\nsingle model AUC/AUPRC:")
    for k, v in candidates.items():
        m = best_f1_metrics(y, v)
        print(f"  {k:10s} AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} ACC={m['acc']:.4f}")

    # ---- (a) convex weight grid search over the top-4 strongest models ----
    ranked = sorted(candidates.items(), key=lambda kv: -best_f1_metrics(y, kv[1])["auc"])
    top = [k for k, _ in ranked[:4]]
    print("\ngrid search over:", top)
    best_w, best_auc, best_mix = None, -1.0, None
    grid = np.arange(0, 1.0001, 0.05)
    n = len(top)
    # simple Dirichlet-like loop over 2 weights then remainder split
    for w0 in grid:
        for w1 in grid[:] if w0 < 1.0 else [0.0]:
            for w2 in grid[:] if w0 + w1 < 1.0 else [0.0]:
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                    continue
                w3 = max(w3, 0.0)
                s = w0 + w1 + w2 + w3
                if s <= 0:
                    continue
                mix = (
                    w0 * candidates[top[0]]
                    + w1 * candidates[top[1]]
                    + w2 * candidates[top[2]]
                    + w3 * candidates[top[3]]
                ) / s
                auc = roc_auc_score(y, mix)
                if auc > best_auc:
                    best_auc, best_w, best_mix = auc, [w0, w1, w2, w3], mix
    print(f"best convex weights {dict(zip(top, [round(float(w),3) for w in best_w]))} AUC={best_auc:.5f}")
    np.save(OUT / "oof_best_fusion.npy", best_mix)

    # ---- (b) logistic-regression stacking with inner 5-fold CV ----
    print("\nlogistic stacking with inner CV ...")
    names = list(candidates.keys())
    Z = np.stack([candidates[k] for k in names], axis=1)
    # rank-transform each component to make stacking scale-free
    from scipy.stats import rankdata

    Zr = np.stack([rankdata(Z[:, i]) / len(y) for i in range(Z.shape[1])], axis=1)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stack_oof = np.zeros(len(y))
    for tr, va in inner.split(Zr, y):
        lr = LogisticRegression(C=1.0, max_iter=2000)
        scaler = StandardScaler()
        lr.fit(scaler.fit_transform(Zr[tr]), y[tr])
        stack_oof[va] = lr.predict_proba(scaler.transform(Zr[va]))[:, 1]
    np.save(OUT / "oof_stack.npy", stack_oof)
    s_m = best_f1_metrics(y, stack_oof)
    print(f"stack AUC={s_m['auc']:.4f} AUPRC={s_m['auprc']:.4f} ACC={s_m['acc']:.4f} F1={s_m['f1']:.4f}")

    # ---- final comparison ----
    print("\nfinal comparison (threshold = best F1):")
    report = {}
    for label, scores in [
        ("lgbm_bag", lgbm_bag),
        ("lgbm_multiscale", lgbm_ms),
        ("convex_fusion_best", best_mix),
        ("lr_stack", stack_oof),
    ]:
        m = best_f1_metrics(y, scores)
        report[label] = m
        print(
            f"  {label:20s} AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"ACC={m['acc']:.4f} F1={m['f1']:.4f} sens={m['sensitivity']:.4f} "
            f"spec={m['specificity']:.4f} cm={m['cm']}"
        )

    with open(OUT / "stack_results.json", "w", encoding="utf-8") as fh:
        json.dump({"candidates": {k: best_f1_metrics(y, v) for k, v in candidates.items()},
                   "grid_top": top,
                   "best_weights": [float(w) for w in best_w],
                   "convex_fusion": report["convex_fusion_best"],
                   "lr_stack": report["lr_stack"]}, fh, indent=2, default=str)


if __name__ == "__main__":
    run()
