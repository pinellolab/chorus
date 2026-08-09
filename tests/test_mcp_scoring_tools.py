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

**RESOLVED 2026-08-09, and the resolution reversed the decision recorded here.** The
original call was: "rather than change LegNet's declared geometry — which would move its
background and every committed artefact — the tools now explain the null." The premise was
false. `resolution = 50` is LegNet's sliding STEP, correct for a multi-window query and
wrong for the single-window case; deriving it from the array
(`len(prediction_interval) // len(preds)`) gives 200 for one window and leaves 50 for six
or eight. No committed artefact moved — the shipped LegNet pkl already carried
`resolution = 200`, because `regenerate_multioracle.py` sets `step = win = 200` itself, and
a full sweep confirmed no artefact diff.

So the sub-region path now WORKS rather than explaining itself, and the tests below assert
that. `test_legnets_declared_geometry_disagrees_with_its_values` was written to trip exactly
when this happened — "a deliberate coupling: the explanation and the defect must change
together" — and it did. The null-explanation machinery is still exercised, on a region that
genuinely does not overlap, which is a real null rather than a manufactured one.
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


def test_a_sub_region_now_scores_instead_of_returning_a_null(loaded):
    """What the geometry fix bought.

    This asserted `scores[TRACK] is None` plus a well-crafted note explaining why.
    A good response to a defect that was believed unfixable; the defect turned out to
    be one line. LegNet's single 200 bp value now maps to bin 0 for any sub-region of
    its window, so the tool returns the number.
    """
    out = _call("score_prediction_region", oracle_name="legnet", region=REGION,
                assay_ids=[TRACK], score_region="chr1:109274918-109275018")
    score = out["scores"][TRACK]
    assert score is not None, (
        f"sub-region scoring regressed to None; notes={out.get('score_notes')}"
    )
    assert isinstance(score, (int, float)) and score == score  # not NaN
    # And no leftover explanation for a null that no longer happens.
    assert not out.get("score_notes", {}).get(TRACK), (
        f"a null-explanation note survives on a successful score: "
        f"{out['score_notes'][TRACK]!r}"
    )


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
    assert inner["ref_score"] is not None, (
        f"{mode}: ref_score is still None; notes={out.get('score_notes')}"
    )
    assert inner["alt_score"] is not None and inner["effect"] is not None
    # effect must be the difference the payload claims, not an independent number.
    assert abs((inner["alt_score"] - inner["ref_score"]) - inner["effect"]) < 1e-9, inner


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


def test_legnets_declared_geometry_agrees_with_its_values(loaded):
    """The inverse of what this test used to assert, and that is the point.

    It previously pinned the DEFECT (`implied == 4` bins against 1 value) with a note
    saying a fix should trip it, "a deliberate coupling: the explanation and the defect
    must change together". The fix came, this tripped, and the coupling worked as
    designed — so it now pins the corrected geometry instead.

    `n_values * resolution == len(prediction_interval)` is the invariant that makes
    `positions`, `pos2bin` and every IGV span mean anything.
    """
    import chorus.mcp.server as server
    from chorus.mcp.server import _parse_region

    oracle = server._state().get_oracle("legnet")
    tr = oracle.predict(_parse_region(REGION), [TRACK])[TRACK]
    iv = tr.prediction_interval.reference
    span = iv.end - iv.start
    assert len(tr.values) == 1, "a 200 bp query should give LegNet one value"
    assert tr.resolution == span, (
        f"resolution {tr.resolution} over a {span} bp interval holding "
        f"{len(tr.values)} value(s) implies {span // tr.resolution} bins that do not "
        f"exist. Derive it from the array, not from the sliding step."
    )
    assert len(tr.values) * tr.resolution == span
    assert tr.pos2bin("chr1", 109_274_968) == 0
