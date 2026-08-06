"""Regenerate README.md's per-oracle background table from the shipped artefacts.

The table was hand-maintained and had drifted badly: it claimed 10,000 effect samples
and 31,500 activity samples per track for six of its seven rows, when the shipped
counts range from 5,949 to 104,033 and vary by oracle *and* by layer. It also omitted
Cherimoya entirely — one of the eight shipped backgrounds simply missing from the
table that enumerates them.

Hand-maintaining a table of numbers that live in binary artefacts is the same defect
as the stale walkthrough READMEs: nothing compares the prose to the data. So the table
is generated, and ``tests/test_documented_track_counts.py`` fails if it drifts.

Idempotent: run it after any rebuild. It rewrites only the fenced block between the
two markers, so surrounding prose is preserved.

Usage:
  python scripts/refresh_readme_background_table.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Resolved through the data-dir mechanism, not hardcoded to $HOME. Every
# background-handling script had this literal; CHORUS_BACKGROUNDS_DIR applies
# the legacy ~/.chorus compatibility itself, per kind.
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
BG = CHORUS_BACKGROUNDS_DIR
README = REPO / "README.md"

BEGIN = "<!-- BEGIN GENERATED: background-table -->"
END = "<!-- END GENERATED: background-table -->"

DISPLAY = {
    "alphagenome": "AlphaGenome", "enformer": "Enformer", "borzoi": "Borzoi",
    "chrombpnet": "ChromBPNet", "cherimoya": "Cherimoya (CATv1)", "sei": "Sei",
    "legnet": "LegNet", "epinformerseq": "EPInformer-seq",
}
ORDER = ["alphagenome", "enformer", "borzoi", "chrombpnet", "cherimoya",
         "sei", "legnet", "epinformerseq"]


def _rng(counts) -> str:
    if counts is None:
        return "—"
    u = np.unique(np.asarray(counts)[np.asarray(counts) > 0])
    if u.size == 0:
        return "—"
    return f"{int(u.min()):,}" if u.size == 1 else f"{int(u.min()):,}–{int(u.max()):,}"


def _size(path: Path) -> str:
    mb = path.stat().st_size / 1e6
    return f"{mb:.0f} MB" if mb >= 1 else f"{mb * 1000:.0f} KB"


def build_table() -> str:
    from chorus.analysis.normalization import PerTrackNormalizer

    norm = PerTrackNormalizer(cache_dir=str(BG))
    rows = [
        "| Oracle | Tracks | Effect samples / track | Activity samples / track "
        "| Effect reference population | NPZ size |",
        "|---|---|---|---|---|---|",
    ]
    for name in ORDER:
        path = BG / f"{name}_pertrack.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            n = len(data["track_ids"])
            eff = _rng(data["effect_counts"] if "effect_counts" in data.files else None)
            act = _rng(data["summary_counts"] if "summary_counts" in data.files else None)
        cfg = norm.provenance(name) or {}
        # Read the stamp; fall back only to facts, never to a guess. Defaulting to
        # "uniform random" mislabelled AlphaGenome, whose stamp records the
        # gene-anchored rule under an older key.
        region = cfg.get("effect_region_set")
        if not region and cfg.get("effect_region_rule"):
            rule = str(cfg["effect_region_rule"])
            region = "gene-anchored" if "gene-anchored" in rule else rule[:40]
        if not region:
            # The two that were peak-anchored before the field existed; read from
            # their builders, not assumed.
            region = {"chrombpnet": "uniform + DHS summits",
                      "cherimoya": "uniform + DHS summits"}.get(name)
        if not region:
            region = "(unrecorded)"
        rows.append(f"| {DISPLAY[name]} | {n:,} | {eff} | {act} | {region} "
                    f"| {_size(path)} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if the README table is stale, without writing.")
    args = ap.parse_args()

    if not BG.is_dir():
        raise SystemExit(f"no backgrounds at {BG}")
    table = build_table()
    text = README.read_text()

    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"README.md has no generated-table markers. Add\n  {BEGIN}\n  {END}\n"
            f"around the per-oracle background table."
        )
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{table}\n{END}{tail}"

    if new == text:
        print("README background table already current")
        return 0
    if args.check:
        print("README background table is STALE — run without --check to refresh")
        return 1
    README.write_text(new)
    print("refreshed README background table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
