"""在指定波长处绘制吸光度分布直方图(正常 vs 异常)。

对每个目标波长(默认 1000~2000 nm,每 100 nm 一个),统计该波长下
正常/异常样本落在各吸光度区间(如 0.2~0.22)内的数量并画直方图。

产物:
    outputs/absorbance_histograms_by_wavelength.png

用法:
    uv run python plot_absorbance_histograms.py [--wavelengths 1000,1100,...,2000]
                                                [--bin-width 0.02] [--out DIR] [--show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_all_spectra import ABNORMAL_DIR, NORMAL_DIR, load_class
from visualize_spectrum import ROOT, parse_spectrum, setup_chinese_font


def main():
    parser = argparse.ArgumentParser(description="指定波长处的吸光度分布直方图")
    parser.add_argument(
        "--wavelengths",
        default="1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000",
        help="逗号分隔的目标波长列表(nm)",
    )
    parser.add_argument("--bin-width", type=float, default=0.02, help="吸光度直方图区间宽度(默认 0.02)")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--show", action="store_true", help="生成后弹出窗口显示")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    targets = [float(x) for x in args.wavelengths.split(",") if x.strip()]

    print(f"读取正常样本: {NORMAL_DIR}")
    normal_paths, Y_normal, _ = load_class(NORMAL_DIR)
    print(f"读取异常样本: {ABNORMAL_DIR}")
    abnormal_paths, Y_abnormal, _ = load_class(ABNORMAL_DIR)
    if Y_normal is None or Y_abnormal is None:
        print("❌ 没有读取到有效光谱数据", file=sys.stderr)
        sys.exit(1)

    wl = parse_spectrum(normal_paths[0])["segments"][0][1]

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    setup_chinese_font()

    C_N = "#2b6cff"   # 正常:蓝
    C_A = "#ffcc00"   # 异常:黄

    n_targets = len(targets)
    ncols = 4
    nrows = int(np.ceil(n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    summary = []
    for k, target in enumerate(targets):
        ax = axes[k]
        idx = int(np.argmin(np.abs(wl - target)))
        actual_wl = float(wl[idx])
        vals_n = Y_normal[:, idx]
        vals_a = Y_abnormal[:, idx]

        lo = min(vals_n.min(), vals_a.min())
        hi = max(vals_n.max(), vals_a.max())
        lo = np.floor(lo / args.bin_width) * args.bin_width
        hi = np.ceil(hi / args.bin_width) * args.bin_width
        bins = np.arange(lo, hi + args.bin_width / 2, args.bin_width)

        ax.hist([vals_n, vals_a], bins=bins, color=[C_N, C_A], alpha=0.6,
                edgecolor="white", lw=0.4, label=["正常", "异常"])
        # 标注两类均值
        for vals, color, ls in [(vals_n, "#0a2f7a", "-"), (vals_a, "#8a6d00", "--")]:
            ax.axvline(vals.mean(), color=color, ls=ls, lw=1.8, alpha=0.9)

        ax.set_title(f"{actual_wl:.0f} nm", fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.set_xlabel("吸光度 / AU", fontsize=9)
        ax.set_ylabel("样本数量", fontsize=9)

        summary.append((target, actual_wl, vals_n.mean(), vals_a.mean(), vals_a.mean() - vals_n.mean()))

    # 隐藏多余子图
    for k in range(n_targets, len(axes)):
        axes[k].set_visible(False)

    handles = [
        Patch(facecolor=C_N, alpha=0.6, label=f"正常 (n={Y_normal.shape[0]})"),
        Patch(facecolor=C_A, alpha=0.6, label=f"异常 (n={Y_abnormal.shape[0]})"),
        plt.Line2D([0], [0], color="#0a2f7a", lw=1.8, label="正常均值"),
        plt.Line2D([0], [0], color="#8a6d00", lw=1.8, ls="--", label="异常均值"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"各波长处吸光度分布直方图(区间宽度 {args.bin_width})  正常(蓝) n={Y_normal.shape[0]} · 异常(黄) n={Y_abnormal.shape[0]}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "absorbance_histograms_by_wavelength.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ 直方图已保存: {out_path}")

    print("\n各波长吸光度均值对比:")
    print(f"{'目标 nm':>9}{'实际 nm':>9}{'正常均值':>12}{'异常均值':>12}{'差异':>10}")
    for target, actual_wl, mn, ma, d in summary:
        print(f"{target:>9.0f}{actual_wl:>9.1f}{mn:>12.4f}{ma:>12.4f}{d:>+10.4f}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
