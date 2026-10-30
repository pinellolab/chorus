"""Regenerate ``chorus/oracles/cherimoya_source/batv1_defaults.py``.

BATv1 has 3,429 ENCODE ChIP-seq experiments but only 2,474 unique
``(biosample, target)`` pairs -- 437 of those pairs hold more than one
experiment (up to 59, for ``K562 / POLR2A``).  So
``load_pretrained_model(assay='CHIP', cell_type='K562', TF='GATA1')`` has to
pick one, and for the same reason as CATv1 that choice is **committed to a
generated table** rather than computed at runtime: the per-track background
CDFs are keyed on the resulting ``CHIP:ENCSR`` track id, and a runtime
``argmax`` over the metrics file would silently move a bare pair onto a
different row if the file were ever updated.

Selection rules, in order:

1. **BPNet parity.** Chorus's ChromBPNet oracle serves 744 BPNet ChIP models
   from the JASPAR DeepLearning 2026 release, and ``SOURCE_ID`` in
   ``chrombpnet_source/chrombpnet_JASPAR_metadata.tsv`` is the ENCODE
   experiment each was trained on.  Every one of the 744 is a BATv1 experiment,
   so where a ``(biosample, target)`` pair has a BPNet model, the BATv1 default
   is the model trained on the *same underlying ENCODE experiment*: the join is
   by accession, which is exact where a join by name is not (JASPAR spells five
   biosamples differently from ENCODE, e.g. ``middle frontal area 46`` for
   ENCODE's ``dorsolateral prefrontal cortex``).  This is the counterpart of
   CATv1's ``chrombpnet-parity`` rule and makes ``CHIP:K562:GATA1`` mean the
   same experiment on both oracles.
2. **Not revoked, when there is a choice.** ENCODE has revoked 5 experiments
   (all ZNF143); their models are released and flagged but are not chosen as a
   pair's default when another experiment exists.  This outranks rule 1: four
   of the five are BPNet pins, and parity with a withdrawn dataset is not worth
   a default, so ``K562 / ZNF143`` resolves to its released experiment.
3. **Best fold-0 count correlation.** Otherwise the highest validation
   ``count_pearson`` at fold 0, tie-broken by accession for determinism.  The
   release's other ``quality_flags`` do not affect selection, matching CATv1;
   they are on the metadata row for the caller to read.

Run from the repo root (after ``scripts/vendor_batv1_metadata.py`` when the
vendored TSVs change):

    python scripts/generate_batv1_defaults.py
"""

import sys
from pathlib import Path

import pandas

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SOURCE_DIR = REPO_ROOT / "chorus" / "oracles" / "cherimoya_source"
JASPAR_TSV = REPO_ROOT / "chorus" / "oracles" / "chrombpnet_source" / "chrombpnet_JASPAR_metadata.tsv"
OUT_PATH = SOURCE_DIR / "batv1_defaults.py"


def main() -> int:
    meta = pandas.read_csv(SOURCE_DIR / "BATv1-metadata.tsv", sep="\t", dtype=str)
    perf = pandas.read_csv(SOURCE_DIR / "BATv1-performance-fold0.tsv", sep="\t")
    jaspar = pandas.read_csv(JASPAR_TSV, sep="\t", dtype=str)

    if set(meta["assay_term_name"]) != {"ChIP-seq"}:
        raise SystemExit(
            f"BATv1-metadata.tsv holds assays {sorted(set(meta['assay_term_name']))}; "
            f"expected ChIP-seq only."
        )

    count_r = dict(zip(perf["experiment_accession"], perf["count_pearson"]))
    status = dict(zip(meta["experiment_accession"], meta["experiment_status"]))
    by_accession = meta.set_index("experiment_accession")

    # Rule 1: the BPNet model chorus serves for each (TF, cell line), by the ENCODE
    # experiment it was trained on. `iter_unique_bpnet_models` keeps the first row per
    # pair; the same dedupe here keeps the two oracles pointing at the same experiment.
    jaspar = jaspar.drop_duplicates(["TF_NAME", "CELL_LINE"])
    bpnet_pins: dict = {}
    for source_id in jaspar["SOURCE_ID"]:
        if source_id not in by_accession.index:
            raise SystemExit(
                f"BPNet model trained on {source_id} is not a BATv1 experiment -- "
                f"the join assumption is broken."
            )
        row = by_accession.loc[source_id]
        pair = (row["experiment_biosample_term_name"], row["target_label"])
        if pair in bpnet_pins and bpnet_pins[pair] != source_id:
            raise SystemExit(f"{pair} is pinned to two BPNet experiments.")
        bpnet_pins[pair] = source_id

    rows = []
    grouped = meta.groupby(["experiment_biosample_term_name", "target_label"])
    for (biosample, target), group in grouped:
        pair = (biosample, target)
        accessions = sorted(group["experiment_accession"])

        # Rule 2 first, so a revoked experiment is skipped even when it is the BPNet pin:
        # four of the five revoked ZNF143 experiments are BPNet pins, and parity with a
        # withdrawn dataset is not worth a default. Where every candidate is revoked
        # (GM12878 / ZNF143) the pin stands.
        candidates = [a for a in accessions if status[a] != "revoked"] or accessions
        pinned = bpnet_pins.get(pair)
        if pinned in candidates:
            chosen, reason = pinned, "bpnet-parity"
        else:
            chosen = max(candidates, key=lambda a: (count_r.get(a, float("-inf")), a))
            reason = "best-count-pearson" if len(accessions) > 1 else "only-candidate"

        rows.append((biosample, target, chosen, len(accessions), reason))

    rows.sort(key=lambda r: (r[0], r[1]))

    n_ambiguous = sum(1 for r in rows if r[3] > 1)
    n_parity = sum(1 for r in rows if r[4] == "bpnet-parity")
    n_skipped = sum(
        1 for biosample, target, _, _, reason in rows
        if (biosample, target) in bpnet_pins and reason != "bpnet-parity"
    )
    if n_parity + n_skipped != len(bpnet_pins):
        raise SystemExit(f"{len(bpnet_pins)} BPNet pins but {n_parity} parity rows.")

    lines = [
        '"""Committed default BATv1 experiment per ``(biosample, target)`` pair.',
        "",
        "GENERATED FILE -- do not edit by hand.  Regenerate with:",
        "",
        "    python scripts/generate_batv1_defaults.py",
        "",
        "Maps a ``(biosample, target)`` pair to the single BATv1 experiment that",
        "``load_pretrained_model(assay='CHIP', cell_type=..., TF=...)`` resolves to.  See",
        "the generator's module docstring for why this is committed rather than",
        "computed at runtime, and for the selection rules.",
        "",
        f"{len(rows)} pairs, of which {n_ambiguous} hold more than one experiment.",
        f"{n_parity} are pinned to the ENCODE experiment behind chorus's BPNet model",
        f"for cross-oracle parity; {n_skipped} such pin(s) skipped as revoked.",
        '"""',
        "",
        "# (biosample, target) -> (experiment_accession, n_candidates, reason)",
        "BATV1_DEFAULT_EXPERIMENT: dict[tuple[str, str], tuple[str, int, str]] = {",
    ]
    for biosample, target, chosen, n, reason in rows:
        lines.append(
            f"    ({biosample!r}, {target!r}): ({chosen!r}, {n}, {reason!r}),"
        )
    lines.append("}")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")
    print(f"  {len(rows)} pairs | {n_ambiguous} ambiguous | {n_parity} bpnet-pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
