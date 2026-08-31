from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_csv(name):
    with (ROOT / "analysis" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


spectra = []
for row in load_csv("spectra_means.csv"):
    out = {"wavelength_nm": float(row["wavelength_nm"])}
    for key, value in row.items():
        if key == "wavelength_nm":
            continue
        out[key] = float(value) if value not in ("", None) else None
    spectra.append(out)

counts = []
for row in load_csv("sample_counts.csv"):
    counts.append({"label": row["label"], "batch": row["batch"], "spectra": int(row["spectra"])})

metrics = json.loads((ROOT / "analysis" / "results.json").read_text(encoding="utf-8"))
validation = [
    {
        "test_batch": row["test_batch"],
        "sensitivity": row["sensitivity"],
        "specificity": row["specificity"],
        "roc_auc": row["roc_auc"],
    }
    for row in metrics["leave_one_batch_out"]
]
combined = metrics["leave_one_batch_out_combined"]
validation.append(
    {
        "test_batch": "合并",
        "sensitivity": combined["sensitivity"],
        "specificity": combined["specificity"],
        "roc_auc": combined["roc_auc"],
    }
)

payload = json.dumps({"spectra": spectra, "counts": counts, "validation": validation}, ensure_ascii=False, separators=(",", ":"))
template = (Path(__file__).resolve().parent / "templates" / "visualization_template.html").read_text(encoding="utf-8")
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
(output_dir / "spectral-detectability.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
