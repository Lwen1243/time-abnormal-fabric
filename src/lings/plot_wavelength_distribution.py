"""输出每个波长处吸光度的分布(正常 vs 异常)。

产物:
    1. outputs/absorbance_distribution_bands.png   全波长分位数带图
    2. outputs/absorbance_distribution_boxplot.png 代表性波长箱线图(正常/异常并排)
    3. outputs/absorbance_distribution_by_wavelength.csv  每个波长的分布统计表

用法:
    uv run python plot_wavelength_distribution.py [--out DIR] [--show] [--dpi N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_all_spectra import ABNORMAL_DIR, NORMAL_DIR, load_class
from visualize_spectrum import ROOT, parse_spectrum, setup_chinese_font

PERCENTILES = [5, 25, 50, 75, 95]


def wavelength_stats(wl: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    pct = np.percentile(Y, PERCENTILES, axis=0)
    return {
        "n": np.full(len(wl), Y.shape[0]),
        "min": Y.min(axis=0),
        "max": Y.max(axis=0),
        "mean": Y.mean(axis=0),
        "std": Y.std(axis=0, ddof=1),
        **{f"p{p}": pct[i] for i, p in enumerate(PERCENTILES)},
    }


def main():
    parser = argparse.ArgumentParser(description="每个波长的吸光度分布")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--show", action="store_true", help="生成后弹出窗口显示")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    print(f"读取正常样本: {NORMAL_DIR}")
    normal_paths, Y_normal, skipped_n = load_class(NORMAL_DIR)
    print(f"读取异常样本: {ABNORMAL_DIR}")
    abnormal_paths, Y_abnormal, skipped_a = load_class(ABNORMAL_DIR)
    if Y_normal is None or Y_abnormal is None:
        print("❌ 没有读取到有效光谱数据", file=sys.stderr)
        sys.exit(1)

    wl = parse_spectrum(normal_paths[0])["segments"][0][1]
    if Y_abnormal.shape[1] != len(wl):
        print("❌ 两类波长网格不一致,请检查数据", file=sys.stderr)
        sys.exit(1)

    stats_n = wavelength_stats(wl, Y_normal)
    stats_a = wavelength_stats(wl, Y_abnormal)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # ---------- 1. 导出每个波长的分布统计 CSV ----------
    csv_path = out_dir / "absorbance_distribution_by_wavelength.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("wavelength_nm,")
        keys = ["n", "min", "p5", "p25", "p50", "p75", "p95", "max", "mean", "std"]
        f.write(",".join(f"normal_{k}" for k in keys) + ",")
        f.write(",".join(f"abnormal_{k}" for k in keys) + "\n")
        for i, lam in enumerate(wl):
            f.write(f"{lam:.1f},")
            f.write(",".join(f"{stats_n[k][i]:.6f}" if k == "n" else f"{stats_n[k][i]:.6f}" for k in keys) + ",")
            f.write(",".join(f"{stats_a[k][i]:.6f}" if k == "n" else f"{stats_a[k][i]:.6f}" for k in keys) + "\n")
    print(f"✅ 分布统计表已保存: {csv_path}")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    setup_chinese_font()

    C_N = "#2b6cff"   # 正常:蓝
    C_A = "#ffcc00"   # 异常:黄

    # ---------- 2. 全波长分位数带图 ----------
    fig1, ax1 = plt.subplots(figsize=(14, 7))

    def draw_bands(ax, s_n, s_a):
        ax.fill_between(wl, s_n["p5"], s_n["p95"], color=C_N, alpha=0.12, linewidth=0)
        ax.fill_between(wl, s_n["p25"], s_n["p75"], color=C_N, alpha=0.25, linewidth=0)
        ax.plot(wl, s_n["p50"], color="#0a2f7a", lw=2.2, label=f"正常中位数 (n={Y_normal.shape[0]})")
        ax.fill_between(wl, s_a["p5"], s_a["p95"], color=C_A, alpha=0.28, linewidth=0)
        ax.fill_between(wl, s_a["p25"], s_a["p75"], color=C_A, alpha=0.40, linewidth=0)
        ax.plot(wl, s_a["p50"], color="#8a6d00", lw=2.2, ls="--", label=f"异常中位数 (n={Y_abnormal.shape[0]})")

    draw_bands(ax1, stats_n, stats_a)

    legend1 = [
        Line2D([0], [0], color="#0a2f7a", lw=2.2, label="正常中位数"),
        Patch(facecolor=C_N, alpha=0.25, label="正常 25%~75% 区间"),
        Patch(facecolor=C_N, alpha=0.12, label="正常 5%~95% 区间"),
        Line2D([0], [0], color="#8a6d00", lw=2.2, ls="--", label="异常中位数"),
        Patch(facecolor=C_A, alpha=0.40, label="异常 25%~75% 区间"),
        Patch(facecolor=C_A, alpha=0.28, label="异常 5%~95% 区间"),
    ]
    ax1.legend(handles=legend1, fontsize=10, loc="best", ncol=2)
    ax1.set_xlabel("波长 / nm", fontsize=13)
    ax1.set_ylabel("吸光度 / AU", fontsize=13)
    ax1.set_title(
        f"每个波长处吸光度的分布带  正常(蓝) n={Y_normal.shape[0]} · 异常(黄) n={Y_abnormal.shape[0]}",
        fontsize=14,
    )
    ax1.grid(True, ls="--", alpha=0.35)
    fig1.tight_layout()
    band_path = out_dir / "absorbance_distribution_bands.png"
    fig1.savefig(band_path, dpi=args.dpi)
    print(f"✅ 分位数带图已保存: {band_path}")

    # ---------- 3. 代表性波长箱线图(每 100 nm 一个) ----------
    sel_idx = [int(np.argmin(np.abs(wl - lam))) for lam in range(1000, 2300, 100)]
    sel_wl = [float(wl[i]) for i in sel_idx]

    n_sel = len(sel_wl)
    positions = np.arange(n_sel) * 2.0
    width = 0.75

    fig2, ax2 = plt.subplots(figsize=(16, 7))
    bp_n = ax2.boxplot(
        [Y_normal[:, i] for i in sel_idx],
        positions=positions - width / 2, widths=width,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#0a2f7a", lw=1.8),
        boxprops=dict(facecolor=C_N, alpha=0.55, edgecolor="#0a2f7a"),
        whiskerprops=dict(color="#0a2f7a"), capprops=dict(color="#0a2f7a"),
    )
    bp_a = ax2.boxplot(
        [Y_abnormal[:, i] for i in sel_idx],
        positions=positions + width / 2, widths=width,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#8a6d00", lw=1.8),
        boxprops=dict(facecolor=C_A, alpha=0.75, edgecolor="#8a6d00"),
        whiskerprops=dict(color="#8a6d00"), capprops=dict(color="#8a6d00"),
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"{lam:.0f}" for lam in sel_wl], fontsize=11)
    ax2.set_xlabel("波长 / nm", fontsize=13)
    ax2.set_ylabel("吸光度 / AU", fontsize=13)
    ax2.set_title("代表性波长处吸光度分布(箱线图:每波长左=正常蓝,右=异常黄)", fontsize=14)
    ax2.grid(True, axis="y", ls="--", alpha=0.35)

    handles2 = [
        Patch(facecolor=C_N, alpha=0.55, edgecolor="#0a2f7a", label=f"正常 (n={Y_normal.shape[0]})"),
        Patch(facecolor=C_A, alpha=0.75, edgecolor="#8a6d00", label=f"异常 (n={Y_abnormal.shape[0]})"),
    ]
    ax2.legend(handles=handles2, fontsize=11, loc="best")
    fig2.tight_layout()
    box_path = out_dir / "absorbance_distribution_boxplot.png"
    fig2.savefig(box_path, dpi=args.dpi)
    print(f"✅ 箱线图已保存: {box_path}")

    # ---------- 摘要:两类中位数差异最大的 5 个波长 ----------
    diff = stats_n["p50"] - stats_a["p50"]
    top = np.argsort(np.abs(diff))[::-1][:5]
    print("\n两类中位数差异最大的 5 个波长:")
    print(f"{'波长 nm':>10}{'正常中位数':>14}{'异常中位数':>14}{'差异(正-异)':>14}")
    for i in top:
        print(f"{wl[i]:>10.1f}{stats_n['p50'][i]:>14.4f}{stats_a['p50'][i]:>14.4f}{diff[i]:>+14.4f}")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
