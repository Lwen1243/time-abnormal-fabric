"""Second-round boost: full-resolution features + stronger meta-learner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402

import lightgbm as lgb  # noqa: E402

SEED = 42


def metrics_at(y, scores, t):
    pred = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, scores),
        "auprc": average_precision_score(y, scores),
        "cm": [int(tn), int(fp), int(fn), int(tp)],
    }


def best_thresholds(y, scores):
    """Best-F1, Youden-J and best-acc thresholds."""
    uniq = np.unique(np.round(scores, 4))
    best_f1 = max(uniq, key=lambda t: f1_score(y, (scores >= t).astype(int), zero_division=0))
    tpr = np.array([(scores[y == 1] >= t).mean() for t in uniq])
    fpr = np.array([(scores[y == 0] >= t).mean() for t in uniq])
    youden = uniq[int(np.argmax(tpr - fpr))]
    best_acc = max(uniq, key=lambda t: accuracy_score(y, (scores >= t).astype(int)))
    return {"best_f1": metrics_at(y, scores, best_f1),
            "youden": metrics_at(y, scores, youden),
            "best_acc": metrics_at(y, scores, best_acc)}


def load_fullres():
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = (spectra - mean) / np.maximum(std, 1e-8)
    g1 = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    g2 = savgol_filter(snv, window_length=41, polyorder=3, deriv=2, delta=0.5, axis=1)
    mask = (wavelength >= 1050) & (wavelength <= 2450)
    idx = np.flatnonzero(mask)
    X = np.concatenate([snv[:, idx], g1[:, idx], g2[:, idx]], axis=1).astype(np.float32)
    return X, y, rows


def oof_skf(est_factory, X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        est = est_factory()
        est.fit(X[tr], y[tr])
        oof[va] = est.predict_proba(X[va])[:, 1]
    return oof


def main():
    print("loading full-resolution features ...")
    X, y, rows = load_fullres()
    print(f"X = {X.shape}")

    print("training LightGBM on full-resolution multiscale (1 seed) ...")
    oof_full = oof_skf(
        lambda: lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, num_leaves=63,
            feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
            min_child_samples=30, reg_lambda=1.0, n_jobs=20,
            random_state=SEED, verbose=-1,
        ),
        X, y,
    )
    np.save(OUT / "oof_lgbm_fullres.npy", oof_full)
    m = best_thresholds(y, oof_full)["best_f1"]
    print(f"LightGBM full-res: AUC={m['auc']:.4f} ACC={m['acc']:.4f}")

    # ---- stronger meta-learner ----
    bases = {
        "lgbm_bag": np.load(OUT / "oof_lgbm_bag.npy"),
        "lgbm_ms": np.load(OUT / "oof_lgbm_multiscale.npy"),
        "lgbm_full": oof_full,
        "xgb": np.load(OUT / "oof_xgb_multiscale.npy"),
        "rf": np.load(OUT / "oof_rf_multiscale.npy"),
        "et": np.load(OUT / "oof_et_multiscale.npy"),
        "cnn": np.load(OUT / "oof_cnn_convnext.npy"),
        "mlp_ens": np.load(OUT / "oof_ensemble_probs.npy").mean(axis=1),
    }
    names = list(bases)
    Z = np.stack([rankdata(bases[k]) / len(y) for k in names], axis=1)
    # add pairwise products of the 5 strongest components for meta features
    top5 = ["lgbm_bag", "lgbm_ms", "xgb", "lgbm_full", "et"]
    Zi = np.stack([rankdata(bases[k]) / len(y) for k in top5], axis=1)
    prods = [Zi[:, i] * Zi[:, j] for i in range(5) for j in range(i + 1, 5)]
    Zm = np.concatenate([Z, np.stack(prods, axis=1)], axis=1)
    print(f"meta features = {Zm.shape}")

    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    meta = {
        "lr": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "lgbm": lambda: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=50, reg_lambda=5.0, n_jobs=20,
            random_state=SEED, verbose=-1,
        ),
    }
    results = {}
    for mname, factory in meta.items():
        oof = np.zeros(len(y))
        for tr, va in inner.split(Zm, y):
            est = factory()
            scaler = StandardScaler()
            est.fit(scaler.fit_transform(Zm[tr]), y[tr])
            if hasattr(est, "predict_proba"):
                oof[va] = est.predict_proba(scaler.transform(Zm[va]))[:, 1]
            else:
                oof[va] = est.decision_function(scaler.transform(Zm[va]))
        np.save(OUT / f"oof_meta_{mname}.npy", oof)
        results[mname] = best_thresholds(y, oof)["best_f1"]
        print(f"meta-{mname}: AUC={results[mname]['auc']:.4f} AUPRC={results[mname]['auprc']:.4f} "
              f"ACC={results[mname]['acc']:.4f} F1={results[mname]['f1']:.4f}")

    # threshold-strategy report on the current best stack
    stack = np.load(OUT / "oof_stack.npy")
    thr = best_thresholds(y, stack)
    print("\nthreshold strategies on LR stack:")
    for k, v in thr.items():
        print(f"  {k:8s} t={v['threshold']:.4f} ACC={v['acc']:.4f} F1={v['f1']:.4f} cm={v['cm']}")

    with open(OUT / "boost_round2.json", "w", encoding="utf-8") as fh:
        json.dump({
            "lgbm_fullres": best_thresholds(y, oof_full),
            "meta": {k: best_thresholds(y, np.load(OUT / f"oof_meta_{k}.npy")) for k in meta},
            "stack_thresholds": thr,
        }, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
