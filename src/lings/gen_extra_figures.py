"""Generate extra PPT figures (13-20): PCA, wavelength importance, samples,
batch means, method evolution, threshold tradeoff, misclassified spectra,
and a pipeline diagram."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data  # noqa: E402

C_NORMAL = "#2b7ab6"
C_ABNORMAL = "#d1495b"
C_V4 = "#17becf"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def fig13():
    """PCA scatter from precomputed points."""
    pts = list(csv.DictReader(open(ROOT / "analysis" / "pca_points.csv", encoding="utf-8-sig")))
    pc1 = np.array([float(r["pc1"]) for r in pts])
    pc2 = np.array([float(r["pc2"]) for r in pts])
    lab = np.array([r["label"] == "异常" for r in pts])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for m, c, n in [(~lab, C_NORMAL, "Normal"), (lab, C_ABNORMAL, "Abnormal")]:
        ax.scatter(pc1[m], pc2[m], s=6, alpha=0.35, color=c, label=n, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA embedding of raw spectra")
    ax.legend(frameon=False, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT / "13_pca.png")
    plt.close(fig)


def fig14():
    """Discriminative wavelengths from top_wavelengths.csv."""
    rows = list(csv.DictReader(open(ROOT / "analysis" / "top_wavelengths.csv", encoding="utf-8-sig")))
    wl = np.array([float(r["wavelength_nm"]) for r in rows])
    eff = np.array([float(r["robust_effect"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(wl, eff, width=1.2, color=np.where(eff > 0, C_ABNORMAL, C_NORMAL), alpha=0.85)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Robust effect (class difference)")
    ax.set_title("Most discriminative wavelengths between normal and abnormal")
    fig.tight_layout()
    fig.savefig(OUT / "14_wavelength_importance.png")
    plt.close(fig)


def fig15(wavelength, spectra, y):
    """Random sample of raw spectra per class."""
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lab, c, name in [
        (axes[0], 0, C_NORMAL, "Normal"),
        (axes[1], 1, C_ABNORMAL, "Abnormal"),
    ]:
        idx = rng.choice(np.flatnonzero(y == lab), size=10, replace=False)
        for i in idx:
            ax.plot(wavelength, spectra[i], color=c, lw=0.8, alpha=0.55)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"10 randomly sampled {name.lower()} spectra")
    fig.suptitle("Raw NIR spectra", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "15_sample_spectra.png")
    plt.close(fig)


def fig16():
    """Mean spectra per batch and class."""
    rows = list(csv.DictReader(open(ROOT / "analysis" / "spectra_means.csv", encoding="utf-8-sig")))
    wl = np.array([float(r["wavelength_nm"]) for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, batch in zip(axes.ravel(), ["DN", "DW", "DV", "DY"]):
        def col(key):
            return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])
        n = col(f"{batch}_正常_mean")
        a = col(f"{batch}_异常_mean")
        ax.plot(wl, n, color=C_NORMAL, lw=1.8, label="Normal")
        ax.plot(wl, a, color=C_ABNORMAL, lw=1.8, label="Abnormal")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"Batch {batch}")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Mean spectra per measurement batch", y=1.01, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "16_batch_means.png")
    plt.close(fig)


def fig17():
    """Method evolution: accuracy and AUC across the project."""
    stages = ["MLP ensemble", "LightGBM\n3-seed bag", "LR\nstacking", "Meta-stack\nv3", "Meta-stack\nv4"]
    acc = [0.940, 0.9629, 0.9669, 0.9698, 0.9718]
    auc = [0.9755, 0.9900, 0.9910, 0.9922, 0.9925]
    x = np.arange(len(stages))
    fig, ax1 = plt.subplots(figsize=(9.5, 4.6))
    ax1.plot(x, acc, "o-", color=C_V4, lw=2.5, ms=8, label="Accuracy")
    ax1.set_ylim(0.93, 0.98)
    ax1.set_ylabel("Accuracy")
    ax2 = ax1.twinx()
    ax2.plot(x, auc, "s--", color=C_ABNORMAL, lw=2, ms=7, label="AUC")
    ax2.set_ylim(0.97, 1.0)
    ax2.set_ylabel("AUC")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    for i in range(len(stages)):
        ax1.annotate(f"{acc[i]:.3f}", (x[i], acc[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#333")
        ax2.annotate(f"{auc[i]:.4f}", (x[i], auc[i]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=C_ABNORMAL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_title("Model improvement across project stages")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "17_method_evolution.png")
    plt.close(fig)


def fig18(y, scores):
    """Metrics vs decision threshold for the final model."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    ts = np.linspace(scores.min() + 1e-4, scores.max() - 1e-4, 400)
    accs, f1s, sens, spec = [], [], [], []
    for t in ts:
        p = (scores >= t).astype(int)
        tn = ((p == 0) & (y == 0)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        tp = ((p == 1) & (y == 1)).sum()
        accs.append((tn + tp) / len(y))
        f1s.append(f1_score(y, p, zero_division=0))
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ts, accs, color=C_V4, lw=2.2, label="Accuracy")
    ax.plot(ts, f1s, color="#9c755f", lw=2.2, label="F1")
    ax.plot(ts, sens, color=C_ABNORMAL, lw=2, ls="--", label="Sensitivity (recall)")
    ax.plot(ts, spec, color=C_NORMAL, lw=2, ls="--", label="Specificity")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Metric trade-off vs decision threshold (Meta-stack v4)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "18_threshold_tradeoff.png")
    plt.close(fig)


def fig19(wavelength, spectra, y, scores, rows):
    """Spectra of false negatives / positives vs class means."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1
    for t in np.unique(np.round(scores, 4)):
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (scores >= best_t).astype(int)
    fn_idx = np.flatnonzero((y == 1) & (pred == 0))
    fp_idx = np.flatnonzero((y == 0) & (pred == 1))
    rng = np.random.default_rng(7)
    fn_show = rng.choice(fn_idx, size=min(6, len(fn_idx)), replace=False)
    fp_show = rng.choice(fp_idx, size=min(6, len(fp_idx)), replace=False)
    mu_n = spectra[y == 0].mean(axis=0)
    mu_a = spectra[y == 1].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(wavelength, mu_n, color=C_NORMAL, lw=2, label="Normal class mean")
    axes[0].plot(wavelength, mu_a, color=C_ABNORMAL, lw=2, label="Abnormal class mean")
    for i in fn_show:
        axes[0].plot(wavelength, spectra[i], color="gray", lw=0.7, alpha=0.6)
    axes[0].set_title(f"False negatives (true abnormal, n={len(fn_idx)})")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Absorbance")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(wavelength, mu_n, color=C_NORMAL, lw=2, label="Normal class mean")
    axes[1].plot(wavelength, mu_a, color=C_ABNORMAL, lw=2, label="Abnormal class mean")
    for i in fp_show:
        axes[1].plot(wavelength, spectra[i], color="gray", lw=0.7, alpha=0.6)
    axes[1].set_title(f"False positives (true normal, n={len(fp_idx)})")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Absorbance")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Misclassified samples vs class means (Meta-stack v4)", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "19_misclassified_spectra.png")
    plt.close(fig)


def fig20():
    """Pipeline diagram of the final Meta-stack v4 method."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f7fb", ec="#4e79a7", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color="#555555", lw=1.3))

    box(0.2, 4.4, 2.2, 1.1, "NIR spectra\n1000-2250 nm", fc="#eaf3fb", ec=C_NORMAL)
    box(3.0, 4.4, 2.6, 1.1, "Preprocessing\nSNV + SG 1st/2nd deriv", fc="#eaf3fb", ec=C_NORMAL)
    arrow(2.4, 4.95, 3.0, 4.95)
    box(6.2, 4.4, 3.4, 1.1, "13 base models\nLGBM x4, XGB x2, ET, RF,\nKNN x2, LDA, CNN, MLP", fc="#fdf3e7", ec="#f28e2b")
    arrow(5.6, 4.95, 6.2, 4.95)
    box(10.1, 4.4, 1.6, 1.1, "5-fold OOF\nprobabilities", fc="#fdf3e7", ec="#f28e2b")
    arrow(9.6, 4.95, 10.1, 4.95)

    box(10.1, 2.3, 1.6, 1.1, "rank transform\n+ pair products\n(91 features)", fc="#e8f7f8", ec=C_V4)
    arrow(10.9, 4.4, 10.9, 3.4)
    box(7.0, 2.3, 2.2, 1.1, "Logistic meta-learner\n13-seed inner CV", fc="#e8f7f8", ec=C_V4)
    arrow(10.1, 2.85, 9.2, 2.85)
    box(3.6, 2.3, 2.4, 1.1, "Final score\nP(abnormal)", fc="#e8f7f8", ec=C_V4)
    arrow(7.0, 2.85, 6.0, 2.85)
    box(0.6, 2.3, 2.0, 1.1, "Decision\nthreshold = best F1", fc="#e8f7f8", ec=C_V4)
    arrow(3.6, 2.85, 2.6, 2.85)

    ax.text(0.2, 0.7, "AUC = 0.99247   |   Accuracy = 97.18%   |   F1 = 0.903",
            fontsize=13, fontweight="bold", color=C_V4)
    ax.set_title("Meta-stack v4 pipeline", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "20_pipeline_diagram.png")
    plt.close(fig)


def main():
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    scores = np.load(ROOT / "analysis" / "oof_boost_round4.npy")
    fig13()
    fig14()
    fig15(wavelength, spectra, y)
    fig16()
    fig17()
    fig18(y, scores)
    fig19(wavelength, spectra, y, scores, rows)
    fig20()
    print("extra figures saved:")
    for name in sorted(p.name for p in OUT.iterdir() if p.name.startswith(("13_", "14_", "15_", "16_", "17_", "18_", "19_", "20_"))):
        print(" ", name)


if __name__ == "__main__":
    main()
