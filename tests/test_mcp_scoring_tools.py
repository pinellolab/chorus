"""The three MCP tools that need a loaded model, and a silent-null they were hiding.

`oracle_status`, `score_prediction_region` and `score_variant_effect_at_region` all call
`state.get_oracle(...)`, so they were left out of the annotation-tool module. Testing them
found that two of the three return a **well-formed success response with every score
null** — no error field, no explanation:

    {"scores": {"alt_1": {"LentiMPRA:HepG2": {"ref_score": null, "alt_score": null,
                                              "effect": null}}}}

An agent reads a populated `scores` key as success and proceeds on nothing. Measured on
LegNet, `score_prediction_region` returned null for score regions of 10, 40, 50 and
100 bp; only the full 200 bp window scored (0.372).

The cause is a geometry inconsistency, not a slice bug. LegNet declares
`resolution = 50` over a 200 bp interval — implying 4 bins — while `values` holds a
single scalar. `region_bin_span` computes bins 1..3 for a 100 bp sub-region, clamps
`end_bin` to `len(values) = 1`, gets `start_bin >= end_bin`, and returns None. Only the
full window maps to bin 0.

This is the fabricated-`resolution` hazard the repo already documents elsewhere ("the
field fabricated as 1 for the track that caused the 131 MB incident"), in a second place.
Rather than change LegNet's declared geometry — which would move its background and every
committed artefact — the tools now explain the null.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

REGION = "chr1:109274868-109275068"          # 200 bp, LegNet's native window
TRACK = "LentiMPRA:HepG2"


def _call(name: str, /, **kw):
    import chorus.mcp.server as server
    fn = getattr(server, name)
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)
    return fn(**kw)


@pytest.fixture(scope="module")
def loaded():
    """LegNet: the cheapest oracle to load, and the one that exposes the defect."""
    r = _call("load_oracle", oracle_name="legnet")
    if not isinstance(r, dict) or r.get("status") != "loaded":
        pytest.skip(f"could not load legnet: {r}")
    return r


# ---------------------------------------------------------------------------
# oracle_status
# ---------------------------------------------------------------------------


def test_oracle_status_reports_what_was_loaded(loaded):
    out = _call("oracle_status")
    assert "loaded_oracles" in out
    names = [o["name"] for o in out["loaded_oracles"]]
    assert "legnet" in names, out
    entry = next(o for o in out["loaded_oracles"] if o["name"] == "legnet")
    assert entry["load_time_seconds"] > 0
    # backgrounds_loaded is the count of CDF rows; legnet ships 3 cell types.
    assert entry.get("backgrounds_loaded") == 3, entry


# ---------------------------------------------------------------------------
# score_prediction_region
# ---------------------------------------------------------------------------


def test_the_full_window_scores(loaded):
    out = _call("score_prediction_region", oracle_name="legnet", region=REGION,
                assay_ids=[TRACK], score_region=REGION)
    assert out["scores"][TRACK] is not None
    assert "score_notes" not in out, "a successful score must not carry a null note"


def test_a_null_score_explains_itself_rather_than_reading_as_success(loaded):
    """The defect. A populated `scores` key with None inside reads as success."""
    out = _call("score_prediction_region", oracle_name="legnet", region=REGION,
                assay_ids=[TRACK], score_region="chr1:109274918-109275018")
    assert out["scores"][TRACK] is None
    note = out.get("score_notes", {}).get(TRACK)
    assert note, (
        "score is None with no note — an agent cannot tell this from a real zero. "
        f"response keys: {sorted(out)}"
    )
    # The note must name the arithmetic, not just say "failed".
    assert "value(s)" in note and "implies" in note, note
    assert "score the full window" in note.lower() or "whole window" in note.lower()


def test_a_non_overlapping_region_is_reported_as_such(loaded):
    out = _call("score_prediction_region", oracle_name="legnet", region=REGION,
                assay_ids=[TRACK], score_region="chr2:1-1000")
    assert out["scores"][TRACK] is None
    note = out.get("score_notes", {}).get(TRACK, "")
    assert "does not overlap" in note, note


# ---------------------------------------------------------------------------
# score_variant_effect_at_region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["at_variant", "score_region"])
def test_both_variant_scoring_modes_explain_a_null(loaded, mode):
    kw = dict(oracle_name="legnet", position="chr1:109274968", ref_allele="G",
              alt_alleles=["T"], assay_ids=[TRACK])
    kw.update({"at_variant": True} if mode == "at_variant"
              else {"score_region": "chr1:109274918-109275018"})
    out = _call("score_variant_effect_at_region", **kw)
    inner = out["scores"]["alt_1"][TRACK]
    assert inner["ref_score"] is None and inner["effect"] is None
    note = out.get("score_notes", {}).get(TRACK)
    assert note, f"{mode}: all scores null with no note; keys {sorted(out)}"
    assert "no positional resolution" in note or "implies" in note, note


def test_the_variant_tool_returns_the_variant_it_was_asked_about(loaded):
    out = _call("score_variant_effect_at_region", oracle_name="legnet",
                position="chr1:109274968", ref_allele="G", alt_alleles=["T"],
                assay_ids=[TRACK], at_variant=True)
    vi = out["variant_info"]
    assert vi["position"] == "chr1:109274968" and vi["ref"] == "G"
    assert vi["alts"] == ["T"]


# ---------------------------------------------------------------------------
# The underlying inconsistency, asserted directly
# ---------------------------------------------------------------------------


def test_legnets_declared_geometry_disagrees_with_its_values(loaded):
    """Pinned so the tools' notes stay truthful, and so a fix trips this test.

    If LegNet's declared resolution is ever corrected to the full window span, this
    fails — at which point the notes above are wrong and the sub-region path should
    start working. That is a deliberate coupling: the explanation and the defect must
    change together.
    """
    import chorus.mcp.server as server
    from chorus.mcp.server import _parse_region

    oracle = server._state().get_oracle("legnet")
    pred = oracle.predict(_parse_region(REGION), [TRACK])
    tr = pred[TRACK]
    iv = tr.prediction_interval.reference
    implied = (iv.end - iv.start) // tr.resolution
    assert len(tr.values) == 1
    assert implied == 4, implied
    assert len(tr.values) != implied, (
        "LegNet's declared geometry now agrees with its values — the sub-region path "
        "should work, and the null-explanation notes need revisiting"
    )
