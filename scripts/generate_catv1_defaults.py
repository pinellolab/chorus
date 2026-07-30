"""Regenerate ``chorus/oracles/cherimoya_source/catv1_defaults.py``.

CATv1 has 1,518 ENCODE experiments but only 492 unique
``(assay, biosample)`` pairs — 162 of those pairs hold more than one
experiment (up to 83, for ``DNase-seq / head of caudate nucleus``), and
1,188 of the 1,518 experiments live in an ambiguous pair.  Even the
flagship cell lines are ambiguous: K562 has 4 ATAC + 1 DNase experiment.

So ``load_pretrained_model(assay='ATAC', cell_type='K562')`` has to pick
one.  That choice is **committed to a generated table** rather than
computed at runtime, because a runtime ``argmax`` over
``performance.tsv`` would silently change which experiment a bare
``(assay, cell_type)`` resolves to if the metrics file were ever
updated — and the per-track background CDFs are keyed on the resulting
``ASSAY:ENCSR`` track id.

Selection rules, in order:

1. **ChromBPNet parity.** For the nine ``(assay, biosample)`` pairs that
   Chorus's ChromBPNet oracle also covers, pick the CATv1 experiment
   trained on the *same underlying ENCODE experiment*.  CATv1's
   ``annotation_accession`` column records the ENCODE ChromBPNet
   annotation per experiment, which is exactly the accession
   ``CHROMBPNET_MODELS_DICT`` keys on, so the join is exact.  This makes
   ``ATAC:K562`` mean the same experiment on both oracles.
2. **Best fold-0 count correlation.** Otherwise take the highest
   ``count_pearson`` at fold 0, tie-broken by accession for determinism.

Run from the repo root:

    python scripts/generate_catv1_defaults.py
"""

import os
import sys
from pathlib import Path

import pandas

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SOURCE_DIR = REPO_ROOT / "chorus" / "oracles" / "cherimoya_source"
OUT_PATH = SOURCE_DIR / "catv1_defaults.py"

# ENCODE ChromBPNet *annotation* accessions for the human models in
# Chorus's CHROMBPNET_MODELS_DICT.  That dict stores ENCFF (file) ids
# with the ENCSR annotation accession in a trailing comment, so these
# cannot be imported programmatically -- they are transcribed here from
# chorus/oracles/chrombpnet_source/chrombpnet_globals.py.  Verified: all
# nine resolve to exactly one CATv1 row via `annotation_accession`.
#
# ChromBPNet also ships mouse models (neural_tube, limb_E12.5, ...).
# CATv1 is GRCh38-only, so those have no counterpart and are omitted.
CHROMBPNET_HUMAN_ANNOTATIONS = {
    ("ATAC", "K562"): "ENCSR467RSV",
    ("ATAC", "HepG2"): "ENCSR380YGX",
    ("ATAC", "GM12878"): "ENCSR389HIH",
    ("ATAC", "IMR-90"): "ENCSR978WIX",
    ("DNASE", "HepG2"): "ENCSR006CUK",
    ("DNASE", "IMR-90"): "ENCSR137OLC",
    ("DNASE", "GM12878"): "ENCSR003WJE",
    ("DNASE", "K562"): "ENCSR296UHQ",
    ("DNASE", "H1"): "ENCSR085MTT",
}

ASSAY_TERM_TO_CHORUS = {"DNase-seq": "DNASE", "ATAC-seq": "ATAC"}


def main() -> int:
    meta = pandas.read_csv(SOURCE_DIR / "CATv1-metadata.tsv", sep="\t")
    perf = pandas.read_csv(SOURCE_DIR / "CATv1-performance-fold0.tsv", sep="\t")

    meta["assay"] = meta["assay_term_name"].map(ASSAY_TERM_TO_CHORUS)
    unmapped = meta["assay"].isna().sum()
    if unmapped:
        raise SystemExit(
            f"{unmapped} rows have an assay_term_name outside "
            f"{sorted(ASSAY_TERM_TO_CHORUS)} -- update ASSAY_TERM_TO_CHORUS."
        )

    count_r = dict(zip(perf["experiment_accession"], perf["count_pearson"]))
    by_annotation = dict(zip(meta["annotation_accession"], meta["experiment_accession"]))

    # Sanity-check rule 1 before relying on it.
    for pair, annot in CHROMBPNET_HUMAN_ANNOTATIONS.items():
        if annot not in by_annotation:
            raise SystemExit(
                f"ChromBPNet annotation {annot} for {pair} is not in "
                f"CATv1-metadata.tsv -- the join assumption is broken."
            )

    rows = []
    grouped = meta.groupby(["assay", "experiment_biosample_term_name"])
    for (assay, biosample), group in grouped:
        pair = (assay, biosample)
        accessions = sorted(group["experiment_accession"])

        annot = CHROMBPNET_HUMAN_ANNOTATIONS.get(pair)
        if annot is not None:
            chosen = by_annotation[annot]
            reason = "chrombpnet-parity"
            if chosen not in accessions:
                raise SystemExit(
                    f"{pair}: ChromBPNet-matched {chosen} is not in that "
                    f"pair's experiments {accessions}."
                )
        else:
            # Highest fold-0 count_pearson; ties broken by accession.
            chosen = max(accessions, key=lambda a: (count_r.get(a, float("-inf")), a))
            reason = "best-count-pearson" if len(accessions) > 1 else "only-candidate"

        rows.append((assay, biosample, chosen, len(accessions), reason))

    rows.sort(key=lambda r: (r[0], r[1]))

    n_ambiguous = sum(1 for r in rows if r[3] > 1)
    n_parity = sum(1 for r in rows if r[4] == "chrombpnet-parity")

    lines = [
        '"""Committed default CATv1 experiment per ``(assay, biosample)`` pair.',
        "",
        "GENERATED FILE -- do not edit by hand.  Regenerate with:",
        "",
        "    python scripts/generate_catv1_defaults.py",
        "",
        "Maps a ``(assay, biosample)`` pair to the single CATv1 experiment that",
        "``load_pretrained_model(assay=..., cell_type=...)`` resolves to.  See the",
        "generator's module docstring for why this is committed rather than",
        "computed at runtime, and for the selection rules.",
        "",
        f"{len(rows)} pairs, of which {n_ambiguous} hold more than one experiment.",
        f"{n_parity} are pinned to ChromBPNet's experiment for cross-oracle parity.",
        '"""',
        "",
        "# (assay, biosample) -> (experiment_accession, n_candidates, reason)",
        "CATV1_DEFAULT_EXPERIMENT: dict[tuple[str, str], tuple[str, int, str]] = {",
    ]
    for assay, biosample, chosen, n, reason in rows:
        lines.append(
            f'    ({assay!r}, {biosample!r}): ({chosen!r}, {n}, {reason!r}),'
        )
    lines.append("}")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")
    print(f"  {len(rows)} pairs | {n_ambiguous} ambiguous | {n_parity} chrombpnet-pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
