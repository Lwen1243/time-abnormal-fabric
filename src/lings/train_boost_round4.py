"""Round-4 boost: KNN on multiscale, LGBM-DART, stronger XGB, 13-seed meta."""
from __future__ import annotations

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402

import lightgbm as lgb  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def load_multi():
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


def oof_skf(est_factory, X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        est = est_factory()
        est.fit(X[tr], y[tr])
        if hasattr(est, "predict_proba"):
            oof[va] = est.predict_proba(X[va])[:, 1]
        else:
            oof[va] = est.decision_function(X[va])
    return oof


def best_f1(y, s):
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(s, 4)):
        f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (s >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "t": float(best_t),
        "acc": accuracy_score(y, pred),
        "f1": best_f1,
        "auc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "cm": [int(tn), int(fp), int(fn), int(tp)],
    }


def main():
    X, y, rows = load_multi()
    print(f"X = {X.shape}")

    print("KNN multiscale ...")
    knn_m = oof_skf(
        lambda: KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=20),
        X, y,
    )
    np.save(OUT / "oof_knn_multiscale.npy", knn_m)
    print("  ", {k: round(v, 4) for k, v in best_f1(y, knn_m).items() if k != "cm"})

    print("LightGBM DART ...")
    dart = oof_skf(
        lambda: lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            min_child_samples=30, boosting_type="dart", drop_rate=0.1,
            skip_drop=0.5, n_jobs=20, random_state=42, verbose=-1,
        ),
        X, y,
    )
    np.save(OUT / "oof_lgbm_dart.npy", dart)
    print("  ", {k: round(v, 4) for k, v in best_f1(y, dart).items() if k != "cm"})

    print("XGBoost v2 ...")
    xgb2 = oof_skf(
        lambda: XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.7,
            colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=20, random_state=42,
        ),
        X, y,
    )
    np.save(OUT / "oof_xgb_v2.npy", xgb2)
    print("  ", {k: round(v, 4) for k, v in best_f1(y, xgb2).items() if k != "cm"})

    # ---- meta v4 ----
    bases = {
        "lgbm_bag": np.load(OUT / "oof_lgbm_bag.npy"),
        "lgbm_ms": np.load(OUT / "oof_lgbm_multiscale.npy"),
        "lgbm_full": np.load(OUT / "oof_lgbm_fullres.npy"),
        "lgbm_dart": dart,
        "xgb_bag": np.load(OUT / "oof_xgb_bag.npy"),
        "xgb_v2": xgb2,
        "et_bag": np.load(OUT / "oof_et_bag.npy"),
        "rf": np.load(OUT / "oof_rf_multiscale.npy"),
        "cnn": np.load(OUT / "oof_cnn_convnext.npy"),
        "mlp_ens": np.load(OUT / "oof_ensemble_probs.npy").mean(axis=1),
        "knn": np.load(OUT / "oof_knn.npy"),
        "knn_m": knn_m,
        "lda": np.load(OUT / "oof_lda.npy"),
    }
    names = list(bases)
    Z = np.stack([rankdata(bases[k]) / len(y) for k in names], axis=1)
    prods = np.stack([Z[:, i] * Z[:, j] for i in range(len(names)) for j in range(i + 1, len(names))], axis=1)
    Zm = np.concatenate([Z, prods], axis=1)
    print(f"meta dim = {Zm.shape}, bases = {len(names)}")

    seeds = [42, 2024, 7, 123, 321, 111, 999, 55, 77, 88, 404, 505, 606]
    oofs = []
    for s in seeds:
        inner = StratifiedKFold(5, shuffle=True, random_state=s)
        o = np.zeros(len(y))
        for tr, va in inner.split(Zm, y):
            sc = StandardScaler()
            lr = LogisticRegression(C=1.0, max_iter=3000)
            lr.fit(sc.fit_transform(Zm[tr]), y[tr])
            o[va] = lr.predict_proba(sc.transform(Zm[va]))[:, 1]
        oofs.append(o)
    meta13 = np.mean(oofs, axis=0)
    np.save(OUT / "oof_boost_round4.npy", meta13)
    m = best_f1(y, meta13)
    print("\nMETA v4:", {k: (round(v, 5) if k != "cm" else v) for k, v in m.items()})

    prev = np.load(OUT / "oof_boost_final2.npy")
    print("META v3:", {k: (round(v, 5) if k != "cm" else v) for k, v in best_f1(y, prev).items()})

    best = None
    for w in np.arange(0, 1.001, 0.05):
        s = w * meta13 + (1 - w) * prev
        auc = roc_auc_score(y, s)
        if best is None or auc > best[0]:
            best = (auc, w, s)
    auc, w, s = best
    mm = best_f1(y, s)
    print(f"\nbest blend v4={w:.2f}: AUC={auc:.5f} ACC={mm['acc']:.5f} F1={mm['f1']:.5f} cm={mm['cm']}")
    np.save(OUT / "oof_boost_final3.npy", s)


if __name__ == "__main__":
    main()
