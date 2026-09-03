"""Round-6: autoencoder reconstruction-error features + final meta."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # noqa: E402


class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(X_train_norm, X_all, device, epochs=25, batch=256, lr=1e-3):
    """Train AE on `X_train_norm`, return per-sample MSE on `X_all`."""
    ae = AutoEncoder(X_all.shape[1]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train_norm, dtype=torch.float32)),
        batch_size=batch, shuffle=True,
    )
    ae.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            loss = nn.functional.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_all), 2048):
            xb = torch.as_tensor(X_all[i : i + 2048], dtype=torch.float32, device=device)
            errs.append(((ae(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs).astype(np.float64)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = ((spectra - mean) / np.maximum(std, 1e-8)).astype(np.float32)
    mask = wavelength >= 1050
    X = snv[:, mask]

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    err_norm = np.zeros(len(y))
    err_all = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        tr_normal = tr[y[tr] == 0]
        err_norm[va] = train_ae(X[tr_normal], X[va], device)
        err_all[va] = train_ae(X[tr], X[va], device)
        print(f"fold{fold} err_norm AUC={roc_auc_score(y[va], -err_norm[va]):.4f} "
              f"err_all AUC={roc_auc_score(y[va], -err_all[va]):.4f}", flush=True)

    np.save(OUT / "ae_err_normal.npy", err_norm)
    np.save(OUT / "ae_err_all.npy", err_all)
    print(f"[AE-normal] OOF AUC={roc_auc_score(y, -err_norm):.4f} "
          f"[AE-all] OOF AUC={roc_auc_score(y, -err_all):.4f}")

    # ---- meta v6 ----
    bases = {
        "lgbm_bag": np.load(f"{OUT}/oof_lgbm_bag.npy"),
        "lgbm_ms": np.load(f"{OUT}/oof_lgbm_multiscale.npy"),
        "lgbm_full": np.load(f"{OUT}/oof_lgbm_fullres.npy"),
        "lgbm_dart": np.load(f"{OUT}/oof_lgbm_dart.npy"),
        "xgb_bag": np.load(f"{OUT}/oof_xgb_bag.npy"),
        "xgb_v2": np.load(f"{OUT}/oof_xgb_v2.npy"),
        "et_bag": np.load(f"{OUT}/oof_et_bag.npy"),
        "rf": np.load(f"{OUT}/oof_rf_multiscale.npy"),
        "cnn": np.load(f"{OUT}/oof_cnn_convnext.npy"),
        "mlp_ens": np.load(f"{OUT}/oof_ensemble_probs.npy").mean(axis=1),
        "knn": np.load(f"{OUT}/oof_knn.npy"),
        "knn_m": np.load(f"{OUT}/oof_knn_multiscale.npy"),
        "lda": np.load(f"{OUT}/oof_lda.npy"),
    }
    names = list(bases)
    Z = np.stack([rankdata(bases[k]) / len(y) for k in names], axis=1)
    prods = np.stack([Z[:, i] * Z[:, j] for i in range(len(names)) for j in range(i + 1, len(names))], axis=1)
    ae_feat = np.stack([rankdata(-err_norm) / len(y), rankdata(-err_all) / len(y),
                        rankdata(err_norm) / len(y), rankdata(err_all) / len(y)], axis=1)
    Zm = np.concatenate([Z, prods, ae_feat], axis=1)
    print("meta dim:", Zm.shape)

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
    meta6 = np.mean(oofs, axis=0)
    np.save(f"{OUT}/oof_boost_round6.npy", meta6)

    def bf(s):
        bt, bf1 = 0.5, -1
        for t in np.unique(np.round(s, 4)):
            f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
            if f1 > bf1:
                bf1, bt = f1, t
        p = (s >= bt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        return accuracy_score(y, p), bf1, roc_auc_score(y, s), average_precision_score(y, s), [tn, fp, fn, tp]

    acc, f1, auc, ap, cm = bf(meta6)
    print(f"\nMETA v6: AUC={auc:.5f} AUPRC={ap:.5f} ACC={acc:.5f} F1={f1:.5f} cm={cm}")
    v4 = np.load(f"{OUT}/oof_boost_round4.npy")
    acc4, f14, auc4, ap4, cm4 = bf(v4)
    print(f"META v4: AUC={auc4:.5f} AUPRC={ap4:.5f} ACC={acc4:.5f} F1={f14:.5f} cm={cm4}")


if __name__ == "__main__":
    main()
