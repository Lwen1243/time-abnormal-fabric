"""吸光度分布的 3D 场可视化(波长 × 吸光度 × 密度)。

把每个波长处的吸光度直方图沿波长轴排开,连成 3D 密度曲面:
    X = 波长 / nm
    Y = 吸光度 / AU(区间中心)
    Z = 该波长处落在该吸光度区间内的样本占比(密度)

正常样本画蓝色曲面,异常样本画黄色曲面,并在底面投下等高线投影。

产物:
    outputs/absorbance_3d_field.png

用法:
    uv run python plot_absorbance_3d_field.py [--step 25] [--bin-width 0.02] [--out DIR] [--show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_all_spectra import ABNORMAL_DIR, NORMAL_DIR, load_class
from visualize_spectrum import ROOT, parse_spectrum, setup_chinese_font


def build_density_field(wl: np.ndarray, Y: np.ndarray, sample_step: float, edges: np.ndarray):
    """返回 (W, B, Z):W=采样波长, B=吸光度区间中心, Z=密度(样本占比)。"""
    n = Y.shape[0]
    centers = (edges[:-1] + edges[1:]) / 2

    # 波长采样
    sel_idx = [int(np.argmin(np.abs(wl - lam))) for lam in np.arange(wl.min(), wl.max() + sample_step / 2, sample_step)]
    sel_idx = list(dict.fromkeys(sel_idx))
    W = wl[np.asarray(sel_idx, dtype=int)]

    Z = np.zeros((len(W), len(centers)))
    for k, i in enumerate(sel_idx):
        hist, _ = np.histogram(Y[:, i], bins=edges)
        Z[k, :] = hist / n  # 密度 = 样本占比
    return W, centers, Z


def main():
    parser = argparse.ArgumentParser(description="吸光度分布 3D 场")
    parser.add_argument("--step", type=float, default=25.0, help="波长采样步长 nm(默认 25)")
    parser.add_argument("--bin-width", type=float, default=0.02, help="吸光度区间宽度(默认 0.02)")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--show", action="store_true", help="生成后弹出窗口显示(可旋转查看)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    print(f"读取正常样本: {NORMAL_DIR}")
    normal_paths, Y_normal, _ = load_class(NORMAL_DIR)
    print(f"读取异常样本: {ABNORMAL_DIR}")
    abnormal_paths, Y_abnormal, _ = load_class(ABNORMAL_DIR)
    if Y_normal is None or Y_abnormal is None:
        print("❌ 没有读取到有效光谱数据", file=sys.stderr)
        sys.exit(1)

    wl = parse_spectrum(normal_paths[0])["segments"][0][1]

    # 统一吸光度区间网格(取两类合并后的 0.5%~99.5% 分位范围)
    lo, hi = np.percentile(np.concatenate([Y_normal, Y_abnormal]), [0.5, 99.5])
    lo = np.floor(lo / args.bin_width) * args.bin_width
    hi = np.ceil(hi / args.bin_width) * args.bin_width
    edges = np.arange(lo, hi + args.bin_width / 2, args.bin_width)

    W_n, B, Z_n = build_density_field(wl, Y_normal, args.step, edges)
    W_a, _, Z_a = build_density_field(wl, Y_abnormal, args.step, edges)
    W = W_n

    print(f"正常密度场: {Z_n.shape[0]} 个波长 × {len(B)} 个吸光度区间")
    print(f"异常密度场: {Z_a.shape[0]} 个波长 × {len(B)} 个吸光度区间")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, colormaps

    setup_chinese_font()

    X, Y = np.meshgrid(W, B, indexing="ij")

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 正常:蓝色系曲面
    blues = colormaps["Blues"]
    ax.plot_surface(X, Y, Z_n, cmap=blues, alpha=0.60, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, zorder=2)

    # 异常:自定义黄色系曲面
    yellow_cmap = colors.LinearSegmentedColormap.from_list(
        "yellow_field", ["#fff6c8", "#ffe066", "#e6a800", "#8a6d00"]
    )
    ax.plot_surface(X, Y, Z_a, cmap=yellow_cmap, alpha=0.75, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, zorder=3)

    # 底面等高线投影(z=0)
    z_floor = 0.0
    ax.contourf(X, Y, Z_n, zdir="z", offset=z_floor, cmap=blues, alpha=0.30, levels=8)
    ax.contourf(X, Y, Z_a, zdir="z", offset=z_floor, cmap=yellow_cmap, alpha=0.45, levels=8)

    # 顶面线框(增强"场"的立体感,仅画正常场)
    ax.plot_wireframe(X[::3, ::2], Y[::3, ::2], Z_n[::3, ::2],
                      color="#0a2f7a", alpha=0.18, linewidth=0.4, zorder=4)

    ax.set_xlabel("波长 / nm", fontsize=12, labelpad=10)
    ax.set_ylabel("吸光度 / AU", fontsize=12, labelpad=10)
    ax.set_zlabel("密度(样本占比)", fontsize=12, labelpad=10)
    ax.set_zlim(z_floor, max(Z_n.max(), Z_a.max()) * 1.15)
    ax.view_init(elev=26, azim=-58)
    ax.set_title(
        f"吸光度分布 3D 密度场  蓝=正常 (n={Y_normal.shape[0]}) · 黄=异常 (n={Y_abnormal.shape[0]})\n"
        f"波长步长 {args.step:.0f} nm · 吸光度区间 {args.bin_width:g} · 底面为密度等高线投影",
        fontsize=14,
        pad=18,
    )

    # 图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=blues(0.75), alpha=0.85, label="正常样本密度场"),
        Patch(facecolor=yellow_cmap(0.75), alpha=0.85, label="异常样本密度场"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=11, bbox_to_anchor=(1.01, 0.95))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "absorbance_3d_field.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ 3D 密度场已保存: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
