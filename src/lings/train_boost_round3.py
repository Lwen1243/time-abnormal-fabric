"""Round-3 boost: add KNN/LDA base models, multi-seed bagged trees, richer meta."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402

SEED = 42


def load_features():
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = (spectra - mean) / np.maximum(std, 1e-8)
    g1 = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    g2 = savgol_filter(snv, window_length=41, polyorder=3, deriv=2, delta=0.5, axis=1)
    mask = (wavelength >= 1050) & (wavelength <= 2450)
    idx = np.flatnonzero(mask)[::4]
    multi = np.concatenate([snv[:, idx], g1[:, idx], g2[:, idx]], axis=1).astype(np.float32)
    snv_low = snv[:, idx].astype(np.float32)
    return multi, snv_low, y, rows


def oof_skf(est_factory, X, y, seed=SEED):
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
    print("loading features ...")
    multi, snv_low, y, rows = load_features()
    print(f"multi={multi.shape} snv_low={snv_low.shape}")

    print("KNN OOF ...")
    knn = oof_skf(
        lambda: KNeighborsClassifier(n_neighbors=30, weights="distance", n_jobs=20),
        snv_low, y,
    )
    np.save(OUT / "oof_knn.npy", knn)
    print("  KNN:", {k: round(v, 4) for k, v in best_f1(y, knn).items() if k != "cm"})

    print("LDA OOF ...")
    lda = oof_skf(
        lambda: _LdaPipeline(),
        multi, y,
    )
    np.save(OUT / "oof_lda.npy", lda)
    print("  LDA:", {k: round(v, 4) for k, v in best_f1(y, lda).items() if k != "cm"})

    print("XGBoost 2-seed bag ...")
    xgb_oofs = [np.load(OUT / "oof_xgb_multiscale.npy")]
    for s in (2024, 7):
        xgb_oofs.append(
            oof_skf(
                lambda: XGBClassifier(
                    n_estimators=800, learning_rate=0.04, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
                    eval_metric="logloss", tree_method="hist", n_jobs=20, random_state=s,
                ),
                multi, y, seed=s,
            )
        )
    xgb_bag = np.mean(xgb_oofs, axis=0)
    np.save(OUT / "oof_xgb_bag.npy", xgb_bag)
    print("  XGB bag:", {k: round(v, 4) for k, v in best_f1(y, xgb_bag).items() if k != "cm"})

    print("ExtraTrees 2-seed bag ...")
    et_oofs = [np.load(OUT / "oof_et_multiscale.npy")]
    for s in (2024, 7):
        et_oofs.append(
            oof_skf(
                lambda: ExtraTreesClassifier(
                    n_estimators=600, max_features="sqrt", min_samples_leaf=3,
                    n_jobs=20, random_state=s, class_weight="balanced_subsample",
                ),
                multi, y, seed=s,
            )
        )
    et_bag = np.mean(et_oofs, axis=0)
    np.save(OUT / "oof_et_bag.npy", et_bag)
    print("  ET bag:", {k: round(v, 4) for k, v in best_f1(y, et_bag).items() if k != "cm"})

    # ---- richer meta ----
    bases = {
        "lgbm_bag": np.load(OUT / "oof_lgbm_bag.npy"),
        "lgbm_ms": np.load(OUT / "oof_lgbm_multiscale.npy"),
        "lgbm_full": np.load(OUT / "oof_lgbm_fullres.npy"),
        "xgb_bag": xgb_bag,
        "et_bag": et_bag,
        "rf": np.load(OUT / "oof_rf_multiscale.npy"),
        "cnn": np.load(OUT / "oof_cnn_convnext.npy"),
        "mlp_ens": np.load(OUT / "oof_ensemble_probs.npy").mean(axis=1),
        "knn": knn,
        "lda": lda,
    }
    names = list(bases)
    Z = np.stack([rankdata(bases[k]) / len(y) for k in names], axis=1)
    prods = np.stack([Z[:, i] * Z[:, j] for i in range(len(names)) for j in range(i + 1, len(names))], axis=1)
    Zm = np.concatenate([Z, prods], axis=1)
    print(f"meta dim = {Zm.shape}")

    seeds = [42, 2024, 7, 123, 321, 111, 999]
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
    meta7 = np.mean(oofs, axis=0)
    np.save(OUT / "oof_boost_round3.npy", meta7)
    m = best_f1(y, meta7)
    print("\nMETA round3:", {k: (round(v, 5) if k != "cm" else v) for k, v in m.items()})

    # compare with previous best
    prev = np.load(OUT / "oof_boost_final.npy")
    print("previous boost:", {k: (round(v, 5) if k != "cm" else v) for k, v in best_f1(y, prev).items()})

    # blend of new meta and previous boost (grid)
    best = None
    for w in np.arange(0, 1.001, 0.05):
        s = w * meta7 + (1 - w) * prev
        auc = roc_auc_score(y, s)
        if best is None or auc > best[0]:
            best = (auc, w, s)
    auc, w, s = best
    mm = best_f1(y, s)
    print(f"\nbest blend meta7={w:.2f}: AUC={auc:.5f} ACC={mm['acc']:.5f} F1={mm['f1']:.5f} cm={mm['cm']}")
    np.save(OUT / "oof_boost_final2.npy", s)


class _LdaPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, X, y):
        self.lda.fit(self.scaler.fit_transform(X), y)
        return self

    def decision_function(self, X):
        return self.lda.decision_function(self.scaler.transform(X))


if __name__ == "__main__":
    main()
