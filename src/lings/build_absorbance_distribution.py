from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis_spectral import load_data


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "absorbance_distribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rounded(values):
    return np.round(np.asarray(values, dtype=float), 6).tolist()


def distribution(values):
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    median = quantiles[2]
    return {
        "n": int(len(values)),
        "mean": rounded(values.mean(axis=0)),
        "std": rounded(values.std(axis=0, ddof=1)),
        "min": rounded(values.min(axis=0)),
        "p05": rounded(quantiles[0]),
        "p25": rounded(quantiles[1]),
        "p50": rounded(median),
        "p75": rounded(quantiles[3]),
        "p95": rounded(quantiles[4]),
        "max": rounded(values.max(axis=0)),
        "iqr": rounded(quantiles[3] - quantiles[1]),
        "mad": rounded(np.median(np.abs(values - median), axis=0)),
    }


rows, wavelength, spectra, errors = load_data()
if errors:
    raise RuntimeError(f"Failed to parse {len(errors)} spectrum files")

labels = np.asarray([row["label"] for row in rows], dtype=int)
batches = np.asarray([row["batch"] for row in rows])
groups = {}
for group_name, mask in [
    ("overall_normal", labels == 0),
    ("overall_abnormal", labels == 1),
]:
    groups[group_name] = distribution(spectra[mask])

for batch in ["DN", "DW", "DV", "DY"]:
    for label, label_name in [(0, "normal"), (1, "abnormal")]:
        mask = (batches == batch) & (labels == label)
        if mask.any():
            groups[f"{batch}_{label_name}"] = distribution(spectra[mask])

payload = {
    "metadata": {
        "source_files": len(rows),
        "wavelength_count": int(len(wavelength)),
        "wavelength_start_nm": float(wavelength[0]),
        "wavelength_end_nm": float(wavelength[-1]),
        "wavelength_step_nm": float(np.median(np.diff(wavelength))),
        "absorbance_unit": "AU",
        "statistics": ["mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max", "iqr", "mad"],
        "note": "All arrays align by index with wavelength_nm. DY_abnormal is absent because no abnormal DY spectrum was supplied.",
    },
    "wavelength_nm": rounded(wavelength),
    "groups": groups,
}

json_path = OUTPUT_DIR / "absorbance-distribution-all-wavelengths.json"
json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

long_path = OUTPUT_DIR / "absorbance-distribution-long.jsonl"
with long_path.open("w", encoding="utf-8") as f:
    for group_name, group in groups.items():
        parts = group_name.split("_", 1)
        if parts[0] == "overall":
            scope, batch, label = "overall", None, parts[1]
        else:
            scope, batch, label = "batch", parts[0], parts[1]
        for index, wavelength_nm in enumerate(payload["wavelength_nm"]):
            record = {
                "scope": scope,
                "batch": batch,
                "label": label,
                "wavelength_nm": wavelength_nm,
                "n": group["n"],
                **{stat: group[stat][index] for stat in payload["metadata"]["statistics"]},
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

visual_payload = {
    "wavelength_nm": payload["wavelength_nm"],
    "normal": groups["overall_normal"],
    "abnormal": groups["overall_abnormal"],
}
template = (Path(__file__).resolve().parent / "templates" / "absorbance_distribution_template.html").read_text(encoding="utf-8")
html_path = OUTPUT_DIR / "absorbance-distribution.html"
html_path.write_text(
    template.replace("__DATA__", json.dumps(visual_payload, ensure_ascii=False, separators=(",", ":"))),
    encoding="utf-8",
)

print(json.dumps({"json": str(json_path.resolve()), "jsonl": str(long_path.resolve()), "html": str(html_path.resolve())}, ensure_ascii=False))
