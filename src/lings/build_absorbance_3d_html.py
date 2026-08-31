"""生成可旋转交互的吸光度 3D 密度场 HTML(Plotly,内联库可离线打开)。

产物:
    outputs/absorbance_3d_field.html

依赖:
    outputs/vendor/plotly.min.js(不存在或不完整时自动下载,失败则回退 CDN 引用)

用法:
    uv run python build_absorbance_3d_html.py [--step 10] [--bin-width 0.02] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from plot_all_spectra import ABNORMAL_DIR, NORMAL_DIR, load_class
from visualize_spectrum import ROOT, parse_spectrum

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_SOURCES = [
    PLOTLY_URL,
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",
    "https://unpkg.com/plotly.js-dist-min@2.35.2/plotly.min.js",
]
PLOTLY_MIN_BYTES = 3_000_000  # 小于该大小视为下载不完整


def _download_resumable(url: str, local: Path, max_rounds: int = 8) -> bytes | None:
    """断点续传下载,按服务器 Content-Length 校验完整性。"""
    offset = local.stat().st_size if local.is_file() else 0
    for _ in range(max_rounds):
        headers = {"User-Agent": "Mozilla/5.0"}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                status = resp.status
                total = None
                cr = resp.headers.get("Content-Range")
                if cr:
                    total = int(cr.rsplit("/", 1)[1])
                cl = resp.headers.get("Content-Length")
                if cl:
                    total = int(cl)
                chunk = resp.read()
                if status == 206:
                    with local.open("ab") as f:
                        f.write(chunk)
                    offset += len(chunk)
                else:
                    local.write_bytes(chunk)
                    offset = len(chunk)
                if total is not None:
                    if offset >= total:
                        return local.read_bytes()
                    print(f"      续传 {url}: {offset / 1e6:.2f} / {total / 1e6:.2f} MB", flush=True)
                else:
                    # 无 Content-Length,以大小阈值判断
                    if offset >= PLOTLY_MIN_BYTES:
                        return local.read_bytes()
        except Exception as e:  # noqa: BLE001
            print(f"      {url} 请求中断({e}),继续续传 ...", flush=True)
    if local.is_file() and local.stat().st_size >= PLOTLY_MIN_BYTES:
        return local.read_bytes()
    return None


def ensure_plotly(vendor_dir: Path) -> str | None:
    """返回 plotly 库源码;若无法取得则返回 None(由调用方回退 CDN)。"""
    local = vendor_dir / "plotly.min.js"
    if local.is_file() and local.stat().st_size >= PLOTLY_MIN_BYTES:
        return local.read_text(encoding="utf-8")
    print("本地 plotly.min.js 缺失或不完整,尝试多源下载(支持断点续传)...", flush=True)
    for url in PLOTLY_SOURCES:
        print(f"  源: {url}", flush=True)
        content = _download_resumable(url, local)
        if content is not None and len(content) >= PLOTLY_MIN_BYTES:
            local.write_bytes(content)
            print(f"✅ 已下载 plotly.min.js ({len(content) / 1e6:.2f} MB)")
            return content.decode("utf-8")
        print(f"⚠️ 该源下载不完整,尝试下一个源", file=sys.stderr)
    return None


def build_density_field(wl: np.ndarray, Y: np.ndarray, sample_step: float, edges: np.ndarray):
    n = Y.shape[0]
    centers = (edges[:-1] + edges[1:]) / 2
    sel_idx = [int(np.argmin(np.abs(wl - lam))) for lam in np.arange(wl.min(), wl.max() + sample_step / 2, sample_step)]
    sel_idx = list(dict.fromkeys(sel_idx))
    W = wl[np.asarray(sel_idx, dtype=int)]
    Z = np.zeros((len(W), len(centers)))
    for k, i in enumerate(sel_idx):
        hist, _ = np.histogram(Y[:, i], bins=edges)
        Z[k, :] = hist / n
    return W, centers, Z


def gaussian_smooth(Z: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """对密度场做高斯平滑(沿两个轴),让曲面柔和连续。"""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    axes = (1, 0)
    Zs = Z
    for axis in axes:
        padded = np.pad(Zs, ((0, 0), (radius, radius)) if axis == 1 else ((radius, radius), (0, 0)), mode="edge")
        if axis == 1:
            Zs = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, padded)
        else:
            Zs = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded)
    return Zs


def main():
    parser = argparse.ArgumentParser(description="生成交互式 3D 密度场 HTML")
    parser.add_argument("--step", type=float, default=10.0, help="波长采样步长 nm(默认 10)")
    parser.add_argument("--bin-width", type=float, default=0.02, help="吸光度区间宽度(默认 0.02)")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    print(f"读取正常样本: {NORMAL_DIR}")
    normal_paths, Y_normal, _ = load_class(NORMAL_DIR)
    print(f"读取异常样本: {ABNORMAL_DIR}")
    abnormal_paths, Y_abnormal, _ = load_class(ABNORMAL_DIR)
    if Y_normal is None or Y_abnormal is None:
        print("❌ 没有读取到有效光谱数据", file=sys.stderr)
        sys.exit(1)

    wl = parse_spectrum(normal_paths[0])["segments"][0][1]

    lo, hi = np.percentile(np.concatenate([Y_normal, Y_abnormal]), [0.5, 99.5])
    lo = np.floor(lo / args.bin_width) * args.bin_width
    hi = np.ceil(hi / args.bin_width) * args.bin_width
    edges = np.arange(lo, hi + args.bin_width / 2, args.bin_width)

    W, B, Z_n = build_density_field(wl, Y_normal, args.step, edges)
    _, _, Z_a = build_density_field(wl, Y_abnormal, args.step, edges)

    # 高斯平滑让曲面更柔和(sigma 随网格变密适当增大)
    sigma = max(1.0, 8.0 / max(args.step, 1.0))
    Z_n = gaussian_smooth(Z_n, sigma)
    Z_a = gaussian_smooth(Z_a, sigma)

    Z_n = gaussian_smooth(Z_n, sigma=1.2)
    Z_a = gaussian_smooth(Z_a, sigma=1.2)

    payload = {
        "wavelength": np.round(W, 1).tolist(),
        "absorbance": np.round(B, 4).tolist(),
        "normal": np.round(Z_n, 6).tolist(),
        "abnormal": np.round(Z_a, 6).tolist(),
        "counts": {"normal": int(Y_normal.shape[0]), "abnormal": int(Y_abnormal.shape[0])},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    vendor_dir = out_dir / "vendor"
    vendor_dir.mkdir(exist_ok=True)

    template = (Path(__file__).resolve().parent / "templates" / "absorbance_3d_template.html").read_text(encoding="utf-8")
    plotly_src = ensure_plotly(vendor_dir)
    if plotly_src is not None:
        html = template.replace("__PLOTLY__", plotly_src)
        print("✅ plotly.js 已内联(离线可用)")
    else:
        html = template.replace("__PLOTLY__", "")
        # 回退:使用 CDN 外部引用
        html = html.replace(
            '<script id="plotly-inline"></script>',
            f'<script src="{PLOTLY_URL}"></script>',
        )
        print("⚠️ 未内联 plotly.js,HTML 打开时需要联网加载 CDN")

    html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out_path = out_dir / "absorbance_3d_field.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ 交互式 3D 密度场已保存: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   网格: {len(W)} 波长 × {len(B)} 吸光度区间,浏览器中拖拽可旋转")


if __name__ == "__main__":
    main()
