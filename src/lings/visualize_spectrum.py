"""指定光谱 CSV 文件的可视化脚本。

用法:
    uv run python visualize_spectrum.py <光谱文件.csv> [更多文件...] [选项]

选项:
    -o, --out DIR    输出图片目录(默认 outputs)
    --show           生成图片后弹出窗口显示
    --dpi N          输出图片分辨率(默认 150)
    --no-smooth      关闭 Savitzky-Golay 平滑叠加曲线

示例:
    uv run python visualize_spectrum.py "异常数据集/异常数据集/采谱DN-2026-06-17-16-49-43-887.csv"
    uv run python visualize_spectrum.py 文件A.csv 文件B.csv --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def parse_spectrum(path: Path) -> dict:
    """解析采谱 CSV:返回元数据 + 一段或多段 (波长, 吸光度)。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()

    meta: dict[str, str] = {}
    segments: list[tuple[str, np.ndarray, np.ndarray]] = []

    i = 0
    # 读取数据段之前的元数据(key;value 形式)
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if line.startswith("#;Wavelength"):
            break
        parts = line.split(";", 1)
        if len(parts) == 2 and not parts[0].startswith(("#", "MEAS SPEC", "MP list")):
            meta[parts[0].strip()] = parts[1].strip()
        i += 1

    # 逐段读取光谱数据(文件中可能包含多条"记录光谱")
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if line.startswith("#;Wavelength"):
            seg_name = ""
            # 向上找最近的 "MEAS SPEC;记录光谱 N" 作为该段名称
            for j in range(i - 1, max(i - 6, -1), -1):
                if lines[j].startswith("MEAS SPEC"):
                    seg_name = lines[j].rstrip("\r\n").split(";", 1)[1].strip()
                    break
            wl, y = [], []
            i += 1
            while i < len(lines):
                parts = lines[i].strip().split(";")
                if len(parts) < 3:
                    i += 1
                    continue
                try:
                    wl.append(float(parts[1]))
                    y.append(float(parts[2]))
                except ValueError:
                    break
                i += 1
            if wl:
                segments.append(
                    (seg_name or f"光谱 {len(segments) + 1}",
                     np.asarray(wl, dtype=np.float64),
                     np.asarray(y, dtype=np.float64))
                )
        else:
            i += 1

    if not segments:
        raise ValueError(f"{path.name}: 未找到光谱数据段(#;Wavelength ...)")
    return {"meta": meta, "segments": segments}


def smooth_savgol(wl: np.ndarray, y: np.ndarray, window: int = 31, order: int = 3) -> np.ndarray:
    """简单 Savitzky-Golay 平滑(numpy 卷积实现,避免引入 scipy)。"""
    if len(y) < window:
        return y.copy()
    window -= 1  # 保持奇数窗口
    window += window % 2
    half = window // 2
    order = min(order, window - 1)
    t = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(t, N=order + 1, increasing=True)
    # 求解卷积核 k,使得对窗口内数据的内积等于中心点的多项式拟合值
    e0 = np.zeros(order + 1)
    e0[0] = 1.0
    k = np.linalg.lstsq(a.T, e0, rcond=None)[0]
    # 对首尾用边缘镜像填充
    padded = np.pad(y, (half, half), mode="edge")
    return np.convolve(padded, k[::-1], mode="valid")


def pick_peaks(wl: np.ndarray, y: np.ndarray, min_dist: float = 20.0, height_frac: float = 0.02):
    """基于局部极大值找峰,返回 (波长, 吸光度) 列表。"""
    if len(y) < 3:
        return []
    peaks = []
    ymin, ymax = y.min(), y.max()
    threshold = ymin + height_frac * (ymax - ymin)
    # 最小间隔对应的索引数
    if len(wl) > 1:
        step = (wl[-1] - wl[0]) / (len(wl) - 1)
        min_idx = max(1, int(min_dist / max(step, 1e-9)))
    else:
        min_idx = 1
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= threshold:
            if peaks and i - peaks[-1][0] < min_idx:
                if y[i] > peaks[-1][1]:
                    peaks[-1] = (i, y[i])
            else:
                peaks.append((i, y[i]))
    return [(float(wl[i]), float(y[i])) for i, _ in peaks]


def setup_chinese_font():
    """配置 matplotlib 中文字体(macOS 常见字体,依次尝试)。"""
    import matplotlib
    from matplotlib import font_manager

    candidates = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "STHeiti"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_spectra(file_paths: list[Path], out_dir: Path, show: bool, dpi: int, use_smooth: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_chinese_font()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_files = len(file_paths)
    fig, ax = plt.subplots(figsize=(12, 6.5))

    all_stats: list[dict] = []
    legend_handles = []

    for k, path in enumerate(file_paths):
        data = parse_spectrum(path)
        meta = data["meta"]
        base_color = colors[k % len(colors)]

        for s, (seg_name, wl, y) in enumerate(data["segments"]):
            label = path.stem if n_files == 1 and len(data["segments"]) == 1 else f"{path.stem} · {seg_name}"
            if n_files > 1 and len(data["segments"]) > 1:
                alpha = 0.85
            else:
                alpha = 1.0
            color = base_color if len(data["segments"]) == 1 else colors[(k + s) % len(colors)]
            (line,) = ax.plot(wl, y, lw=1.2, alpha=alpha, color=color, label=label)
            legend_handles.append(line)

            if use_smooth:
                ys = smooth_savgol(wl, y)
                ax.plot(wl, ys, lw=1.0, alpha=0.55, color=color, linestyle="--",
                        label=f"{label}(平滑)")

            # 标注主要峰
            for px, py_ in pick_peaks(wl, y, height_frac=0.05)[:6]:
                ax.plot(px, py_, marker="v", ms=5, color=color, alpha=0.8)
                ax.annotate(f"{px:.1f}", (px, py_), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center", color=color, alpha=0.9)

            all_stats.append({
                "文件": path.name,
                "段": seg_name,
                "样品": meta.get("Sample name", "?"),
                "点数": len(y),
                "波长范围 nm": f"{wl.min():.1f} ~ {wl.max():.1f}",
                "吸光度范围": f"{y.min():.3f} ~ {y.max():.3f}",
                "吸光度均值": f"{y.mean():.4f}",
            })

    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("吸光度 / AU")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_title("光谱可视化" + (" · " + all_stats[0]["样品"] if all_stats else ""))
    if legend_handles:
        ax.legend(fontsize=8, loc="best")

    out_dir.mkdir(exist_ok=True)
    stems = [p.stem for p in file_paths]
    out_name = stems[0] if n_files == 1 else f"{stems[0]}_等{n_files}个文件"
    out_path = out_dir / f"{out_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"✅ 图片已保存: {out_path}")

    # 打印统计摘要
    print("\n" + "-" * 70)
    header = f"{'文件':<42} {'点数':>6} {'波长范围 nm':>16} {'吸光度范围':>18}"
    print(header)
    for st in all_stats:
        print(f"{st['文件']:<42} {st['点数']:>6} {st['波长范围 nm']:>16} {st['吸光度范围']:>18}")
    print("-" * 70)

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="可视化指定光谱 CSV 文件")
    parser.add_argument("files", nargs="+", help="一个或多个光谱 CSV 文件路径")
    parser.add_argument("-o", "--out", default=str(ROOT / "outputs"), help="输出图片目录")
    parser.add_argument("--show", action="store_true", help="生成后弹出窗口显示")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片分辨率")
    parser.add_argument("--no-smooth", action="store_true", help="不绘制平滑曲线")
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.is_file():
            print(f"❌ 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    plot_spectra(paths, Path(args.out), args.show, args.dpi, use_smooth=not args.no_smooth)


if __name__ == "__main__":
    main()
