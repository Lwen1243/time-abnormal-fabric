"""Generate a PPT-ready figure suite for the NIR yarn anomaly detection project.

All figures are saved to outputs/ppt_figures/ with English labels
(no CJK font is installed in this environment; titles can be re-done in PPT).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src" / "lings"))

from analysis_spectral import load_data, preprocess_snv_derivative, read_spectrum  # noqa: E402

C_NORMAL = "#2b7ab6"
C_ABNORMAL = "#d1495b"
C_NEUTRAL = "#6c757d"

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


def best_f1_metrics(y, scores):
    """Metrics at the threshold maximizing F1 on the OOF scores."""
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


def fig01(rows):
    """Dataset overview: label counts and batch x label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = np.array([r["label"] for r in rows])
    n_normal = int((labels == 0).sum())
    n_abnormal = int((labels == 1).sum())

    axes[0].bar(
        ["Normal", "Abnormal"],
        [n_normal, n_abnormal],
        color=[C_NORMAL, C_ABNORMAL],
        width=0.55,
    )
    for i, v in enumerate([n_normal, n_abnormal]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Samples")
    axes[0].set_title(f"Class balance (N = {n_normal + n_abnormal:,})")
    axes[0].set_ylim(0, n_normal * 1.15)

    batch_order = ["DN", "DW", "DV", "DY", "Unknown"]
    counts = Counter((r["batch"], r["label"]) for r in rows)
    x = np.arange(len(batch_order))
    normal = [counts[(b, 0)] for b in batch_order]
    abnormal = [counts[(b, 1)] for b in batch_order]
    axes[1].bar(x, normal, 0.6, label="Normal", color=C_NORMAL)
    axes[1].bar(x, abnormal, 0.6, bottom=normal, label="Abnormal", color=C_ABNORMAL)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(batch_order)
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Samples per measurement batch")
    axes[1].legend(frameon=False)
    fig.suptitle("Dataset overview", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "01_dataset_overview.png")
    plt.close(fig)


def fig02(wavelength, spectra, labels):
    """Mean spectra +/- 1 std band for the two classes."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for lab, color, name in [(0, C_NORMAL, "Normal"), (1, C_ABNORMAL, "Abnormal")]:
        x = spectra[labels == lab]
        mu, sd = x.mean(axis=0), x.std(axis=0)
        ax.plot(wavelength, mu, color=color, lw=2, label=name)
        ax.fill_between(wavelength, mu - sd, mu + sd, color=color, alpha=0.25)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Mean absorbance spectra (band = +/-1 std)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "02_mean_spectra.png")
    plt.close(fig)


def fig03(wavelength, spectra, labels):
    """Mean SNV + 1st-derivative curves of the two classes."""
    full = preprocess_snv_derivative(spectra, wavelength)[0]
    wl_mask = (wavelength >= 1050) & (wavelength <= 2450)
    wl_sel = wavelength[wl_mask]
    deriv_full = preprocess_snv_derivative(spectra, wavelength)
    # recompute derivative on full grid for plotting (no downsampling)
    from scipy.signal import savgol_filter

    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    snv = (spectra - mean) / np.maximum(std, 1e-8)
    deriv = savgol_filter(snv, window_length=31, polyorder=2, deriv=1, delta=0.5, axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for lab, color, name in [(0, C_NORMAL, "Normal"), (1, C_ABNORMAL, "Abnormal")]:
        x = deriv[labels == lab]
        mu, sd = x.mean(axis=0), x.std(axis=0)
        ax.plot(wavelength, mu, color=color, lw=2, label=name)
        ax.fill_between(wavelength, mu - sd, mu + sd, color=color, alpha=0.25)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("1st derivative (SNV)")
    ax.set_title("Preprocessed spectra: SNV + Savitzky-Golay 1st derivative (band = +/-1 std)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "03_preprocessed_means.png")
    plt.close(fig)


def fig04():
    """t-SNE embedding of all 7905 spectra."""
    xy = np.load(ROOT / "analysis" / "tsne_xy.npy")
    lab = np.load(ROOT / "analysis" / "tsne_labels.npy")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for l, color, name in [(0, C_NORMAL, "Normal"), (1, C_ABNORMAL, "Abnormal")]:
        m = lab == l
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.35, color=color, label=name, linewidths=0)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE embedding of raw spectra (perplexity 30)")
    ax.legend(frameon=False, markerscale=4, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUT / "04_tsne.png")
    plt.close(fig)


def fig05():
    """Cosine similarity of each abnormal sample to its nearest normal sample."""
    vals = []
    with open(ROOT / "analysis" / "suspicious_labels.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            vals.append(float(row["与最近正常相似度"]))
    vals = np.array(vals)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(vals, bins=60, color=C_ABNORMAL, alpha=0.8)
    axes[0].set_xlabel("Cosine similarity to nearest normal sample")
    axes[0].set_ylabel("Abnormal samples")
    axes[0].set_title("Abnormal spectra are nearly identical to normal spectra")

    thresh = 0.9999
    sorted_vals = np.sort(vals)
    frac = (vals > thresh).mean()
    axes[1].plot(sorted_vals, np.arange(1, len(vals) + 1) / len(vals), color=C_ABNORMAL, lw=2)
    axes[1].axvline(thresh, color="k", ls="--", lw=1)
    axes[1].text(thresh, 0.5, f"{frac:.1%} above cos = {thresh}", rotation=90, va="center")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Cumulative distribution")
    fig.suptitle("Spectral overlap between classes", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "05_cross_similarity.png")
    plt.close(fig)


def fig06():
    """Examples of nearly identical normal/abnormal spectrum pairs."""
    with open(ROOT / "analysis" / "abnormal_normal_same_pairs.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, row in zip(axes.ravel(), rows):
        wl, ya = None, None
        name_a, wl, ya = read_spectrum(ROOT / row["异常文件"])
        name_n, wl_n, yn = read_spectrum(ROOT / row["正常文件"])
        ax.plot(wl, ya, color=C_ABNORMAL, lw=1.6, label="Abnormal")
        ax.plot(wl_n, yn, color=C_NORMAL, lw=1.2, alpha=0.85, label="Normal")
        ax.set_title(f"r = {float(row['皮尔逊相关']):.7f}, max|dA| = {float(row['最大吸光度差']):.4f}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Same spectrum, different labels (contradictory pairs)", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_pair_overlap.png")
    plt.close(fig)


def fig07_08(y, methods):
    """ROC and PR curves of every method."""
    fig, ax = plt.subplots(figsize=(7.6, 6))
    for name, color, scores in methods:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (5-fold OOF)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "07_roc_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6))
    from sklearn.metrics import precision_recall_curve

    for name, color, scores in methods:
        prec, rec, _ = precision_recall_curve(y, scores)
        auprc = average_precision_score(y, scores)
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AUPRC = {auprc:.4f})")
    baseline = y.mean()
    ax.axhline(baseline, color="k", ls="--", lw=1, alpha=0.6, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (5-fold OOF)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "08_pr_curves.png")
    plt.close(fig)


def fig09(y, methods):
    """Faceted bar chart: one panel per metric, one bar per method."""
    metrics = ["auc", "auprc", "acc", "f1"]
    names = [m[0] for m in methods]
    colors = [m[1] for m in methods]
    recs = [best_f1_metrics(y, m[2]) if m[2] is not None else dict(m[3]) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, k in zip(axes.ravel(), metrics):
        vals = np.array([recs[i][k] for i in range(len(recs))])
        best_i = int(np.argmax(vals))
        colors_bars = [c if i != best_i else c for i, c in enumerate(colors)]
        bars = ax.bar(
            range(len(names)), vals, width=0.62, color=colors_bars,
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        bars[best_i].set_edgecolor("#333333")
        bars[best_i].set_linewidth(1.8)
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.02, (hi - lo) * 0.25)
        ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
        for i, v in enumerate(vals):
            ax.text(
                i, v + (hi - lo) * 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if i == best_i else "normal",
                color="#1a1a1a" if i == best_i else "#555555",
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_title(k.upper(), fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(
        handles, names, frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), fontsize=10,
    )
    fig.suptitle("Method comparison on 5-fold OOF (threshold = best F1)",
                 fontweight="bold", fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "09_method_comparison.png")
    plt.close(fig)

    with open(OUT / "method_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "threshold", "acc", "f1", "precision", "sensitivity", "specificity", "auc", "auprc", "tn", "fp", "fn", "tp"])
        for name, rec in zip(names, recs):
            row = [name]
            for k in ("threshold", "acc", "f1", "precision", "sensitivity", "specificity", "auc", "auprc"):
                v = rec.get(k, np.nan)
                row.append(f"{v:.5f}" if np.isfinite(v) else "")
            row += [str(x) for x in rec.get("cm", [])]
            w.writerow(row)


def fig10(y, scores):
    """Score distributions of the best model by true class."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lab, color, name in [(0, C_NORMAL, "Normal"), (1, C_ABNORMAL, "Abnormal")]:
        axes[0].hist(scores[y == lab], bins=50, alpha=0.65, color=color, label=name)
    axes[0].set_xlabel("Predicted P(abnormal)")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Meta-stack v4 OOF score distribution")
    axes[0].legend(frameon=False)

    t = best_f1_metrics(y, scores)["threshold"]
    axes[1].hist(scores[y == 0], bins=50, alpha=0.6, color=C_NORMAL, label="Normal", density=True)
    axes[1].hist(scores[y == 1], bins=50, alpha=0.6, color=C_ABNORMAL, label="Abnormal", density=True)
    axes[1].axvline(t, color="k", ls="--", lw=1, label=f"threshold = {t:.3f}")
    axes[1].set_xlabel("Predicted P(abnormal)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence separation at the decision boundary")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "10_score_distribution.png")
    plt.close(fig)


def fig11(y, scores, rows):
    """Confusion matrix heatmap + error-score histogram."""
    rec = best_f1_metrics(y, scores)
    tn, fp, fn, tp = rec["cm"]
    cm = np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Pred Normal", "Pred Abnormal"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["True Normal", "True Abnormal"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
    axes[0].set_title(f"Confusion matrix (acc = {rec['acc']:.4f})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pred = (scores >= rec["threshold"]).astype(int)
    fn_scores = scores[(y == 1) & (pred == 0)]
    fp_scores = scores[(y == 0) & (pred == 1)]
    axes[1].hist(fp_scores, bins=30, alpha=0.7, color=C_NORMAL, label=f"False positives (n={len(fp_scores)})")
    axes[1].hist(fn_scores, bins=30, alpha=0.7, color=C_ABNORMAL, label=f"False negatives (n={len(fn_scores)})")
    axes[1].axvline(rec["threshold"], color="k", ls="--", lw=1)
    axes[1].set_xlabel("Predicted P(abnormal)")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Errors concentrate near the decision boundary")
    axes[1].legend(frameon=False)
    fig.suptitle("Meta-stack v4 error analysis", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "11_error_analysis.png")
    plt.close(fig)


def fig12(rows, y, scores):
    """Predicted scores of the contradictory pairs: the model is also confused."""
    file2idx = {r["file"]: i for i, r in enumerate(rows)}
    pair_scores_a, pair_scores_n = [], []
    with open(ROOT / "analysis" / "abnormal_normal_same_pairs.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ia, inn = file2idx.get(row["异常文件"]), file2idx.get(row["正常文件"])
            if ia is not None and inn is not None:
                pair_scores_a.append(scores[ia])
                pair_scores_n.append(scores[inn])
    pair_scores_a = np.array(pair_scores_a)
    pair_scores_n = np.array(pair_scores_n)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(pair_scores_a, bins=bins, alpha=0.6, color=C_ABNORMAL, label="Abnormal members of pairs")
    ax.hist(pair_scores_n, bins=bins, alpha=0.6, color=C_NORMAL, label="Normal members of pairs")
    ax.set_xlabel("Meta-stack v4 predicted P(abnormal)")
    ax.set_ylabel("Samples")
    ax.set_title(
        "Model scores on 1,151 near-identical cross-label pairs\n"
        f"mean abnormal-member score = {pair_scores_a.mean():.3f}  |  mean normal-member score = {pair_scores_n.mean():.3f}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "12_contradictory_pairs.png")
    plt.close(fig)


def main():
    print("loading data ...")
    rows, wavelength, spectra, errors = load_data()
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    print(f"data loaded: {spectra.shape}, abnormal = {y.sum()}")

    lgbm_bag = np.load(ROOT / "analysis" / "oof_lgbm_bag.npy")
    lgbm = np.load(ROOT / "analysis" / "oof_lgbm.npy")
    cnn = np.load(ROOT / "analysis" / "oof_cnn_convnext.npy")
    ens = np.load(ROOT / "analysis" / "oof_ensemble_probs.npy").mean(axis=1)

    methods = [
        ("LightGBM 3-seed bag", "#4e79a7", lgbm_bag),
        ("LightGBM", "#59a14f", lgbm),
        ("MLP ensemble", "#f28e2b", ens),
        ("ConvNeXt-1D", "#e15759", cnn),
    ]

    fig01(rows)
    fig02(wavelength, spectra, y)
    fig03(wavelength, spectra, y)
    fig04()
    fig05()
    fig06()
    fig07_08(y, methods)
    fig09(y, methods)
    fig10(y, lgbm_bag)
    fig11(y, lgbm_bag, rows)
    fig12(rows, y, lgbm_bag)

    print("saved figures:")
    for p in sorted(OUT.iterdir()):
        print(" ", p.name)
    print("\nmethod metrics:")
    print(open(OUT / "method_metrics.csv", encoding="utf-8").read())


if __name__ == "__main__":
    main()
