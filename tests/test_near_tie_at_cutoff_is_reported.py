"""When "the top N" isn't stable between runs, say so.

Cross-process drift is accepted rather than eliminated (2026-08-17 decision, recorded in
`docs/BACKGROUND_NULL_PROTOCOL.md`): median drift is 0.016% and published percentiles move at most
0.69%, so rebuilding the Enformer and ChromBPNet nulls to chase bit-exactness was not worth changing
two oracles' published numbers.

That trade is sound for *values*. It is not automatically sound for *rankings*, and this is the gap it
leaves: `_rank_and_select` cuts hard at `top_n_per_layer`, and the measured rank-12↔13 gap in
`tf_binding` was **4.25%** against a **4.29%** worst-case drift. So two runs of identical code can report
a different track in the top N while no number moves enough for a reader to question it — the one way
the accepted drift can still change a conclusion.

The fix is not to change the selection. It is to stop presenting a coin-flip as a result.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from chorus.analysis.discovery import NEAR_TIE_RELATIVE_GAP, _near_tie_at_cutoff


def _te(score: float, name: str = "t", cell: str = "K562"):
    return SimpleNamespace(ranking_score=score, assay_id=name, cell_type=cell)


def test_a_drift_sized_gap_at_the_cutoff_is_flagged():
    """The measured case: ranks 12 and 13 separated by less than the drift band."""
    effects = [_te(10.0 - i * 0.001, f"t{i}") for i in range(14)]
    tie = _near_tie_at_cutoff(effects, 12)

    assert tie is not None, (
        "a gap inside the accepted drift band was not flagged, so which track appears in the top 12 "
        "varies between runs with nothing in the output saying so"
    )
    assert tie["rank"] == 12
    assert tie["kept"]["assay_id"] == "t11" and tie["dropped"]["assay_id"] == "t12"
    assert 0 <= tie["relative_gap"] <= NEAR_TIE_RELATIVE_GAP
    assert "not stable between runs" in tie["note"]


def test_a_real_separation_is_not_flagged():
    """The signal is worthless if it fires on genuine rankings."""
    effects = [_te(10.0, "a"), _te(9.0, "b"), _te(2.0, "c"), _te(1.0, "d")]
    assert _near_tie_at_cutoff(effects, 2) is None


@pytest.mark.parametrize("effects,top_n,why", [
    ([_te(1.0, "a"), _te(0.9, "b")], 2, "nothing was dropped, so there is no boundary"),
    ([_te(1.0, "a")], 5, "top_n exceeds the list"),
    ([], 3, "no effects at all"),
    ([_te(0.0, "a"), _te(0.0, "b")], 1, "all scores zero — no meaningful relative gap"),
    ([_te(None, "a"), _te(None, "b")], 1, "scores missing"),
])
def test_it_stays_silent_when_there_is_nothing_to_say(effects, top_n, why):
    assert _near_tie_at_cutoff(effects, top_n) is None, why


def test_the_threshold_sits_above_the_measured_percentile_drift():
    """1% is not arbitrary: it must exceed the drift it exists to describe.

    Published `quantile_score` was measured moving at most 0.69% cross-process. A threshold below that
    would call genuinely-affected boundaries clean; far above it would flag every ranking.
    """
    assert 0.0069 < NEAR_TIE_RELATIVE_GAP <= 0.05, NEAR_TIE_RELATIVE_GAP


def test_selection_itself_is_unchanged():
    """This reports; it must not re-rank. The top N stay the top N."""
    from chorus.analysis.discovery import _rank_and_select

    effects = [
        SimpleNamespace(ranking_score=10.0, assay_id="a", cell_type="K562", layer="tf_binding",
                        effect_pctile=0.99, abs_score=1.0),
        SimpleNamespace(ranking_score=9.999, assay_id="b", cell_type="HepG2", layer="tf_binding",
                        effect_pctile=0.99, abs_score=1.0),
        SimpleNamespace(ranking_score=1.0, assay_id="c", cell_type="H1", layer="tf_binding",
                        effect_pctile=0.99, abs_score=1.0),
    ]
    selected, rankings, near_ties = _rank_and_select(effects, top_n_per_layer=1)

    assert [t.assay_id for t in selected] == ["a"], "selection changed; this should only report"
    assert "tf_binding" in near_ties, "a 0.01% boundary was not reported"
    assert near_ties["tf_binding"]["dropped"]["assay_id"] == "b"


def test_the_payload_carries_the_key_even_when_clean():
    """Callers should be able to treat presence of a *non-empty* dict as the signal."""
    import inspect

    from chorus.analysis import discovery

    src = inspect.getsource(discovery.discover_variant_effects)
    assert '"near_ties_at_cutoff"' in src, (
        "discover_variant_effects no longer returns near_ties_at_cutoff, so a consumer cannot tell a "
        "stable top-N from a coin-flip"
    )
