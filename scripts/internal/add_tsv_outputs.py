"""Generate example_output.tsv next to each example_output.json.

Reads variant-report JSONs (produced by report.to_dict()) and writes
a flat TSV with one row per track per allele. Matches the columns of
VariantReport.to_dataframe().
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

BASE = "/PHShome/lp698/chorus/examples/applications"

COLUMNS = [
    "allele", "layer", "assay_id", "assay_type", "cell_type",
    "description", "ref_value", "alt_value", "raw_score",
    "quantile_score", "ref_signal_percentile", "note",
]


def json_to_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    alleles = data.get("alleles", {})
    for allele, payload in alleles.items():
        seen: set[tuple] = set()
        layers = payload.get("scores_by_layer", {})
        for layer, tracks in layers.items():
            for t in tracks:
                key = (allele, t.get("assay_id"), layer)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "allele": allele,
                    "layer": layer,
                    "assay_id": t.get("assay_id", ""),
                    "assay_type": t.get("assay_type", ""),
                    "cell_type": t.get("cell_type", ""),
                    "description": t.get("description", ""),
                    "ref_value": t.get("ref_value", ""),
                    "alt_value": t.get("alt_value", ""),
                    "raw_score": t.get("raw_score", ""),
                    "quantile_score": t.get("quantile_score", ""),
                    "ref_signal_percentile": t.get("ref_signal_percentile", ""),
                    "note": t.get("note", ""),
                })
    return rows


def write_tsv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    pattern = os.path.join(BASE, "**", "example_output.json")
    count = 0
    for json_path in glob.glob(pattern, recursive=True):
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"skip {json_path}: {exc}")
            continue
        if "alleles" not in data:
            continue
        rows = json_to_rows(data)
        if not rows:
            continue
        tsv_path = json_path.replace("example_output.json", "example_output.tsv")
        write_tsv(rows, tsv_path)
        print(f"wrote {tsv_path} ({len(rows)} rows)")
        count += 1
    print(f"\n{count} TSVs written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
