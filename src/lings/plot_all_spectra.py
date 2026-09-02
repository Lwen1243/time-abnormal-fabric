"""将所有正常样本(蓝)与异常样本(黄)画在同一张图上。

用法:
    uv run python plot_all_spectra.py [--out DIR] [--show] [--dpi N]

输出:
    outputs/all_spectra_overview.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from visualize_spectrum import ROOT, parse_spectrum, setup_chinese_font

NORMAL_DIR = ROOT / "data" / "正常数据集"
ABNORMAL_DIR = ROOT / "data" / "异常数据集"


def load_class(folder: Path) -> tuple[list[Path], np.ndarray | None, list[Path]]:
    """读取文件夹下所有光谱 CSV(兼容「采谱XX-」与「XX原始数据-」命名),返回 (文件列表, 光谱矩阵, 跳过列表)。"""
    paths = sorted(folder.rglob("*.csv"))
    wavelength: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    skipped: list[Path] = []

    t0 = time.time()
    for n, path in enumerate(paths, 1):
        try:
            data = parse_spectrum(path)
            wl, y = data["segments"][0][1], data["segments"][0][2]
        except (ValueError, OSError) as e:
            skipped.append(path)
            continue
        if wavelength is None:
            wavelength = wl
        if wl.shape != wavelength.shape or not np.allclose(wl, wavelength, atol=1e-6):
            skipped.append(path)
            continue
        if not np.all(np.isfinite(y)):
            skipped.append(path)
            continue
        spectra.append(y)
        if n % 500 == 0:
            print(f"    {folder.name}: 已读取 {n}/{len(paths)} ... ({time.time() - t0:.1f}s)", flush=True)

    matrix = np.vstack(spectra) if spectra else None
    return paths, matrix, skipped


def main():
    parser = argparse.ArgumentParser(description="全样本光谱总览图(正常蓝 / 异常黄)")
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

    # 波长网格取正常类第一个文件的网格为基准,并校验异常类一致
    wl = parse_spectrum(normal_paths[0])["segments"][0][1]
    if Y_abnormal.shape[1] != len(wl):
        print("❌ 两类波长网格不一致,请检查数据", file=sys.stderr)
        sys.exit(1)

    print(f"\n正常样本: {Y_normal.shape[0]} 条 | 异常样本: {Y_abnormal.shape[0]} 条")
    print(f"波长点数: {len(wl)} | 范围: {wl.min():.1f} ~ {wl.max():.1f} nm")
    if skipped_n:
        print(f"跳过正常文件 {len(skipped_n)} 个")
    if skipped_a:
        print(f"跳过异常文件 {len(skipped_a)} 个")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_chinese_font()

    COLOR_NORMAL = "#2b6cff"   # 正常样本:蓝
    COLOR_ABNORMAL = "#ffcc00"  # 异常样本:黄

    fig, ax = plt.subplots(figsize=(14, 8))

    # 所有样本细线叠加(2D 矩阵一次绘制,速度快)
    ax.plot(wl, Y_normal.T, color=COLOR_NORMAL, lw=0.35, alpha=0.10, rasterized=True)
    ax.plot(wl, Y_abnormal.T, color=COLOR_ABNORMAL, lw=0.45, alpha=0.45, rasterized=True)

    # 每类均值粗线
    mean_normal = Y_normal.mean(axis=0)
    mean_abnormal = Y_abnormal.mean(axis=0)
    ax.plot(wl, mean_normal, color="#0a2f7a", lw=2.5, label=f"正常均值 (n={Y_normal.shape[0]})", zorder=5)
    ax.plot(wl, mean_abnormal, color="#8a6d00", lw=2.5, ls="--", label=f"异常均值 (n={Y_abnormal.shape[0]})", zorder=5)

    # 图例:细线颜色说明 + 均值线
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLOR_NORMAL, lw=2, alpha=0.8, label=f"正常样本 (n={Y_normal.shape[0]})"),
        Line2D([0], [0], color=COLOR_ABNORMAL, lw=2, alpha=0.9, label=f"异常样本 (n={Y_abnormal.shape[0]})"),
        Line2D([0], [0], color="#0a2f7a", lw=2.5, label="正常均值"),
        Line2D([0], [0], color="#8a6d00", lw=2.5, ls="--", label="异常均值"),
    ]
    ax.legend(handles=handles, fontsize=11, loc="best")

    ax.set_xlabel("波长 / nm", fontsize=13)
    ax.set_ylabel("吸光度 / AU", fontsize=13)
    ax.set_title(
        f"全样本光谱总览  正常(蓝) n={Y_normal.shape[0]} · 异常(黄) n={Y_abnormal.shape[0]}",
        fontsize=15,
    )
    ax.grid(True, ls="--", alpha=0.35)

    # 打印均值对比摘要
    print("\n" + "-" * 66)
    print(f"{'指标':<22}{'正常均值':>14}{'异常均值':>14}{'差异':>12}")
    print(f"{'吸光度整体均值':<22}{mean_normal.mean():>14.4f}{mean_abnormal.mean():>14.4f}{mean_abnormal.mean() - mean_normal.mean():>+12.4f}")
    print(f"{'吸光度最大值':<22}{mean_normal.max():>14.4f}{mean_abnormal.max():>14.4f}{mean_abnormal.max() - mean_normal.max():>+12.4f}")
    print(f"{'吸光度最小值':<22}{mean_normal.min():>14.4f}{mean_abnormal.min():>14.4f}{mean_abnormal.min() - mean_normal.min():>+12.4f}")
    i_max = int(np.argmax(np.abs(mean_abnormal - mean_normal)))
    print(f"{'最大差异处波长 nm':<22}{wl[i_max]:>14.1f}{'':>14}{mean_abnormal[i_max] - mean_normal[i_max]:>+12.4f}")
    print("-" * 66)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "all_spectra_overview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    print(f"\n✅ 图片已保存: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
