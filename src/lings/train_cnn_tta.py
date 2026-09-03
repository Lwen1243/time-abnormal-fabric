"""Round-5a: ConvNeXt-1D with test-time augmentation (GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402
from train_1d_cnn import seed_everything, make_loader  # noqa: E402
from train_multi_model import train_model  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim, ks=15):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, ks, padding=ks // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(dim * 4, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 1e-6)

    def forward(self, x):
        r = x
        x = self.dw(x); x = self.norm(x); x = self.pw1(x)
        x = self.act(x); x = self.pw2(x)
        return r + self.gamma * x


class SpectralConvNeXt(nn.Module):
    def __init__(self, in_ch=2, widths=(32, 64, 128, 256), blocks=(2, 2, 4, 2)):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, widths[0], 15, padding=7)
        self.norm0 = nn.BatchNorm1d(widths[0])
        stages = []
        for i, (w, nb) in enumerate(zip(widths, blocks)):
            if i > 0:
                stages.append(nn.Conv1d(widths[i - 1], w, 3, stride=2, padding=1))
                stages.append(nn.BatchNorm1d(w))
                stages.append(nn.GELU())
            stages.extend([ConvNeXt1DBlock(w) for _ in range(nb)])
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(widths[-1], 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.norm0(x)
        x = torch.nn.functional.gelu(x)
        x = self.stages(x)
        return self.head(x).squeeze(1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    from scipy.signal import savgol_filter
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = (spectra - mean) / np.maximum(std, 1e-8)
    deriv = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)
    mask = wavelength >= 1050
    X = np.stack([snv[:, mask], deriv[:, mask]], axis=1).astype(np.float32)
    print("features:", X.shape)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)
    for fi, (tr, va) in enumerate(skf.split(X, y), 1):
        seed_everything(42 + fi)
        model = SpectralConvNeXt()
        res = train_model(
            model, X, y, tr, va, device=device,
            batch_size=256, max_epochs=40, patience=10,
            seed=42 + fi, oversample=3, aug_noise=0.02,
        )
        model.load_state_dict(res["state"])
        model.to(device)
        model.eval()
        center = torch.as_tensor(res["center"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(res["scale"], dtype=torch.float32, device=device)
        # TTA: 8 variants averaged (input normalized exactly as in training)
        with torch.no_grad():
            Xv = torch.as_tensor(X[va], dtype=torch.float32, device=device)
            Xv = (Xv - center) / scale
            ps = []
            for xb in torch.split(Xv, 512):
                outs = [torch.sigmoid(model(xb))]
                for s in (-2, -1, 1, 2):
                    xs = torch.roll(xb, s, dims=-1)
                    if s > 0:
                        xs[:, :, :s] = xs[:, :, s : s + 1]
                    else:
                        xs[:, :, s:] = xs[:, :, s - 1 : s]
                    outs.append(torch.sigmoid(model(xs)))
                for _ in range(3):
                    outs.append(torch.sigmoid(model(xb + torch.randn_like(xb) * 0.005)))
                ps.append(torch.stack(outs).mean(dim=0))
            oof[va] = torch.cat(ps).cpu().numpy()
        auc = roc_auc_score(y[va], oof[va])
        print(f"fold{fi} valAUC_tta={auc:.4f} (best_auc={res['best_auc']:.4f})", flush=True)

    np.save(OUT / "oof_cnn_tta.npy", oof)
    print(f"[CNN-TTA OOF] AUC={roc_auc_score(y, oof):.4f} AUPRC={average_precision_score(y, oof):.4f}")


if __name__ == "__main__":
    main()
