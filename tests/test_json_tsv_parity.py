"""Every score in a walkthrough's JSON must appear in its TSV.

The TSV is the artefact a reader actually opens in a spreadsheet, and a dropped row
leaves no trace in it — there is no gap, no null, nothing to notice. So this is
exactly the kind of loss that survives review.

It had. ``scripts/regenerate_remaining_examples.py`` de-duplicated its rows on
``(allele, assay_id, layer)``, a key that omits ``region_label``. RNA and CAGE emit
one row per **gene** per track — same allele, same assay, same layer, different gene
— so every gene beyond the first was discarded:

===============================  =========  ========
walkthrough                      JSON rows  TSV rows
===============================  =========  ========
validation/TERT_chr5_1295046            99        18
sequence_engineering/region_swap        32         4
sequence_engineering/integration        55         3
===============================  =========  ========

TERT's TSV showed one ``tss_activity`` row where there were fifteen, one per nearby
gene TSS (BRD9, CLPTM1L, LPCAT1, MRPL36, ...). The counts were identical before and
after the 2026-08-04 rebuild, so this was long-standing rather than a regression —
and it is the same defect class as the rest of that work: two copies of one routine
(``regenerate_examples.py`` has its own ``_write_tsv``, which was correct)
disagreeing with nothing comparing them.

Walkthroughs whose report shape carries no ``alleles.*.all_scores`` — batch scoring,
causal prioritisation, discovery, the consolidated multi-oracle view — are skipped
rather than asserted, because their TSVs are legitimately a different granularity.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

WALKTHROUGHS = Path(__file__).resolve().parent.parent / "examples" / "walkthroughs"


def _cases():
    if not WALKTHROUGHS.is_dir():
        return []
    out = []
    for js in sorted(WALKTHROUGHS.rglob("example_output.json")):
        tsv = js.with_name("example_output.tsv")
        if tsv.exists():
            out.append(pytest.param(js, tsv,
                                    id=str(js.parent.relative_to(WALKTHROUGHS))))
    return out


@pytest.mark.parametrize("json_path,tsv_path", _cases())
def test_every_json_score_reaches_the_tsv(json_path: Path, tsv_path: Path):
    doc = json.loads(json_path.read_text())
    alleles = doc.get("alleles")
    if not isinstance(alleles, dict):
        pytest.skip("report shape has no alleles map")

    scores = [s for payload in alleles.values()
              for s in ((payload or {}).get("all_scores") or [])]
    if not scores:
        pytest.skip("no all_scores — different report granularity")

    with open(tsv_path) as fh:
        tsv_rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(tsv_rows) == len(scores), (
        f"{json_path.parent.name}: JSON has {len(scores)} scores but the TSV has "
        f"{len(tsv_rows)} rows. A de-dup key that omits region_label collapses "
        f"per-gene RNA/CAGE rows into one and loses the rest silently."
    )

    # And the identities must match, not merely the counts — an equal count with
    # different rows would be a worse failure than an unequal one.
    def key(assay, layer, label):
        return (str(assay or ""), str(layer or ""), str(label or ""))

    # Both sides key on region_label. The TSV column is region_label because
    # to_dataframe() is the canonical writer; the old hand-rolled one renamed it to
    # "description", which already means the TRACK description in to_dict(). That
    # collision is why the two artefacts could not be compared at all.
    want = {key(s.get("assay_id"), s.get("layer"), s.get("region_label"))
            for s in scores}
    got = {key(r.get("assay_id"), r.get("layer"), r.get("region_label"))
           for r in tsv_rows}
    missing = want - got
    assert not missing, f"{len(missing)} identities absent from the TSV: {sorted(missing)[:4]}"


@pytest.mark.parametrize("json_path,tsv_path", _cases())
def test_no_walkthrough_tsv_is_empty(json_path: Path, tsv_path: Path):
    """A zero-row TSV is what ``_write_tsv`` produces when handed an empty list —
    it returns early and leaves whatever was on disk, so an empty one means either
    a stale file or a silent build failure."""
    with open(tsv_path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert rows, f"{tsv_path.parent.name}/example_output.tsv has no data rows"
