"""A batch row's headline numbers must be findable in its own per-track detail.

``score_variant_batch`` computes ``max_effect`` / ``max_quantile`` /
``per_layer_scores`` by ranking over **every** ``TrackScore`` the report emits,
then stored the detail in a dict keyed on ``assay_id`` alone. CAGE emits one row
per nearby gene TSS **plus** a variant-site row, all sharing one assay_id, so all
but the last were overwritten — silently, leaving no gap and no null to notice.

The artefact then contradicted itself. Measured on the committed
``examples/walkthroughs/batch_scoring/example_output.json`` before the fix:

===========  ================================  ======================  =========
variant      headline                          the CAGE(-) row shown   verdict
===========  ================================  ======================  =========
rs7528419    max_effect −0.06217 on CAGE(−)    +0.001582               opposite
rs4970836    max_effect +0.05817 on CAGE(−)    −0.003435               opposite
all 5        max_quantile                      —                       absent
all 5        per_layer_scores.tss_activity     —                       absent
===========  ================================  ======================  =========

and the TSV under-reported CAGE by ~700x (−0.002 where the winning gene-TSS row
was +1.20). The aggregates were right — they saw the whole population — so the
fix is to make the detail as fine-grained as the aggregates, not to shrink the
aggregates to the lossy subset, which would have made the ranking wrong too.

``test_committed_artefacts_reconcile`` reads the committed JSON, so it stays red
until ``scripts/regenerate_examples.py`` reruns the batch walkthrough.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chorus.analysis import batch_scoring as bs
from chorus.analysis.batch_scoring import BatchResult, score_variant_batch
from chorus.analysis.variant_report import TrackScore

REPO = Path(__file__).resolve().parents[1]
WALKTHROUGHS = REPO / "examples" / "walkthroughs"

# Exact equality would hold in-process, but the artefact path round-trips through
# JSON, so compare with a tolerance far tighter than any real difference between
# two distinct track scores.
TOL = 1e-9


def _cage(label: str, raw: float, q: float | None) -> TrackScore:
    """One CAGE row — same assay_id every time, distinguished only by region."""
    return TrackScore(
        assay_id="CAGE/hCAGE EFO:0001187/-", assay_type="CAGE", cell_type="HepG2",
        layer="tss_activity", ref_value=10.0, alt_value=11.0, raw_score=raw,
        quantile_score=q, description="CAGE:HepG2", region_label=label,
    )


_DNASE = TrackScore(
    assay_id="DNASE/EFO:0001187 DNase-seq/.", assay_type="DNASE", cell_type="HepG2",
    layer="chromatin_accessibility", ref_value=100.0, alt_value=95.0,
    raw_score=-0.0548, quantile_score=0.7933, description="DNASE:HepG2",
)

# The winning tss_activity row is a gene TSS in the MIDDLE of the list, so a
# last-wins dict drops it and an argmax over the shown rows disagrees with the
# headline in both magnitude and sign.
_ROWS = [
    _DNASE,
    _cage("variant site", 0.001582, 0.235),
    _cage("SORT1 TSS", 1.2044, 0.8632),
    _cage("PSRC1 TSS", -0.06217, 0.4001),
    _cage("CELSR2 TSS", 0.0031, 0.1102),
]


def _batch(rows_per_variant: list[list[TrackScore]]) -> BatchResult:
    """Run score_variant_batch over stubbed reports (no oracle, no GPU)."""
    oracle = MagicMock()
    oracle.name = "alphagenome"
    reports = iter(rows_per_variant)

    def _fake_report(*_a, **_kw):
        return SimpleNamespace(allele_scores={"T": next(reports)}, gene_name="SORT1")

    variants = [
        {"chrom": "chr1", "pos": 109274968 + i, "ref": "G", "alt": "T",
         "id": f"rs{i}"}
        for i in range(len(rows_per_variant))
    ]
    with patch.object(bs, "build_variant_report", side_effect=_fake_report):
        return score_variant_batch(oracle, variants, ["CAGE/hCAGE EFO:0001187/-"])


def _close(value: float, pool: list[float]) -> bool:
    return any(v is not None and abs(value - v) <= TOL for v in pool)


# ── The keying itself ────────────────────────────────────────────────────────


def test_every_report_row_reaches_track_scores():
    s = _batch([_ROWS])
    assert len(s.scores[0].track_scores) == len(_ROWS), (
        "rows sharing an assay_id were collapsed: "
        f"{len(_ROWS)} scored -> {len(s.scores[0].track_scores)} kept"
    )


def test_track_scores_keys_are_assay_plus_region():
    ts = _batch([_ROWS]).scores[0].track_scores
    assert {(t.assay_id, t.region_label) for t in ts.values()} == {
        (r.assay_id, r.region_label) for r in _ROWS
    }


def test_column_labels_are_unique_per_row():
    """Renderers derive column names from the display name, and pandas silently
    overwrites a repeated column key — so a duplicate label loses data even when
    track_scores itself is complete."""
    ts = _batch([_ROWS]).scores[0].track_scores
    labels = [bs._track_display_name(t) for t in ts.values()]
    assert len(set(labels)) == len(labels), f"duplicate column labels: {labels}"


# ── The check that would have caught it ──────────────────────────────────────


def test_headline_values_are_present_among_the_rows_own_track_scores():
    for s in _batch([_ROWS, _ROWS]).scores:
        raws = [t.raw_score for t in s.track_scores.values()]
        quants = [t.quantile_score for t in s.track_scores.values()]

        assert _close(s.max_effect, raws), (
            f"{s.variant_id}: max_effect {s.max_effect} is in no per-track "
            f"raw_score {sorted(r for r in raws if r is not None)}"
        )
        assert s.max_quantile is not None and _close(s.max_quantile, quants), (
            f"{s.variant_id}: max_quantile {s.max_quantile} is in no per-track "
            f"quantile_score {sorted(q for q in quants if q is not None)}"
        )
        for layer, value in s.per_layer_scores.items():
            layer_raws = [t.raw_score for t in s.track_scores.values()
                          if t.layer == layer]
            assert _close(value, layer_raws), (
                f"{s.variant_id}: per_layer_scores[{layer}]={value} is in no "
                f"{layer} row {sorted(r for r in layer_raws if r is not None)}"
            )
        assert s.top_track in s.track_scores, (
            f"{s.variant_id}: top_track {s.top_track!r} names no row in the "
            "detail table printed beside it"
        )


def test_the_top_track_row_is_the_one_that_set_max_effect():
    s = _batch([_ROWS]).scores[0]
    assert s.track_scores[s.top_track].raw_score == pytest.approx(s.max_effect)
    assert s.track_scores[s.top_track].region_label == "SORT1 TSS"


# ── The artefacts ────────────────────────────────────────────────────────────


def test_tsv_shows_the_winning_row_not_the_last_one():
    tsv = _batch([_ROWS]).to_tsv().splitlines()
    cells = dict(zip(tsv[0].split("\t"), tsv[1].split("\t")))
    fcs = {k: float(v) for k, v in cells.items()
           if k.endswith("_log2fc") and k.startswith("CAGE")}
    assert len(fcs) == 4, f"expected 4 CAGE columns, got {sorted(fcs)}"
    assert max(fcs.values()) == pytest.approx(1.2044)


def test_json_track_scores_carry_assay_id_and_region_label():
    entry = _batch([_ROWS]).to_dict()["scores"][0]
    assert len(entry["track_scores"]) == len(_ROWS)
    got = {(t["assay_id"], t["region_label"]) for t in entry["track_scores"].values()}
    assert got == {(r.assay_id, r.region_label) for r in _ROWS}


def test_a_variant_missing_a_row_shares_the_others_column():
    """A per-gene row absent for one variant must leave a hole in the existing
    column, not open a second all-null column for the same track."""
    lean = [_DNASE, _cage("variant site", 0.02, 0.5)]
    df = _batch([_ROWS, lean]).to_dataframe()
    fc_cols = [c for c in df.columns if c.endswith("_log2fc")]
    assert len(fc_cols) == len(_ROWS)
    assert df["CAGE:HepG2 (-) — SORT1 TSS_log2fc"].isna().tolist() == [False, True]


# ── The committed artefacts ──────────────────────────────────────────────────


def _batch_artefacts():
    if not WALKTHROUGHS.is_dir():
        return []
    out = []
    for js in sorted(WALKTHROUGHS.rglob("example_output.json")):
        doc = json.loads(js.read_text())
        if isinstance(doc.get("scores"), list) and any(
            isinstance(s, dict) and s.get("track_scores") for s in doc["scores"]
        ):
            out.append(pytest.param(js, id=str(js.parent.relative_to(WALKTHROUGHS))))
    return out


@pytest.mark.parametrize("json_path", _batch_artefacts())
def test_committed_artefacts_reconcile(json_path: Path):
    doc = json.loads(json_path.read_text())
    problems: list[str] = []
    for s in doc["scores"]:
        tracks = s.get("track_scores") or {}
        if not tracks:
            continue
        vid = s.get("variant_id", "?")
        raws = [t.get("raw_score") for t in tracks.values()]
        quants = [t.get("quantile_score") for t in tracks.values()]

        if s.get("max_effect") and not _close(s["max_effect"], raws):
            problems.append(f"{vid}: max_effect {s['max_effect']} in no row")
        if s.get("max_quantile") is not None and not _close(s["max_quantile"], quants):
            problems.append(f"{vid}: max_quantile {s['max_quantile']} in no row")
        for layer, value in (s.get("per_layer_scores") or {}).items():
            layer_raws = [t.get("raw_score") for t in tracks.values()
                          if t.get("layer") == layer]
            if not _close(value, layer_raws):
                problems.append(f"{vid}: per_layer[{layer}] {value} in no row")
        if s.get("top_track") and s["top_track"] not in tracks:
            problems.append(f"{vid}: top_track {s['top_track']!r} names no row")

    assert not problems, (
        f"{json_path.parent.name}: {len(problems)} headline values cannot be "
        f"located in their own track_scores — {problems[:6]}"
    )
