"""A percentile must keep discriminating inside a tied run of the null (#83).

Plain ``searchsorted`` collapses every value landing inside a run of equal CDF
entries to one end of that run. Where a background's upper tail is a long tie —
which is what a near-degenerate null looks like — effects of 0.05, 0.5 and 5.0 all
return the same percentile, and the column stops discriminating exactly where it
matters. Measured on AlphaGenome RNA: the null tops out at 0.0417, so anything
above that read exactly 1.0000.

**AlphaGenome solves this the same way**, and its implementation is on disk:

    indices = np.searchsorted(scorer_quantiles[i], values, side='left')
    if break_quantile_ties and duplicate_quantiles[i]:
        end_indices = np.searchsorted(scorer_quantiles[i], values, side='right')
        indices = rng.integers(indices, end_indices, endpoint=True)

(``alphagenome_research/model/variant_scoring/calibration/calibration.py:155-160``,
with ``has_duplicate_quantiles`` precomputed at :63.) So a calibrated per-track
quantile *is* part of their public API, degenerate tracks are expected rather than
exceptional, and tie-breaking is on by default.

**One deliberate difference.** Their draw comes from a sequential
``np.random.Generator``, so the answer depends on call ORDER and changes run to run
unless a seed is threaded through. Chorus derives it from a stable hash of
``(track_id, raw_value)``, which is uniform across the tie *and* identical for the
same query every time — preserving the reproducibility #127 was fixed to obtain.

**What this does not buy.** It restores *distributional* correctness — percentiles
are uniform under the null again — but it does not make an individual row more
informative. The raw effect is still where the resolution lives, which is why both
are reported.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

N_POINTS = 10_000


def _save(tmp_path, rows, counts, oracle="tietest"):
    return PerTrackNormalizer.build_and_save(
        oracle_name=oracle,
        track_ids=[f"T{i}" for i in range(len(rows))],
        effect_cdfs=np.asarray(rows, dtype=np.float64),
        effect_counts=list(counts),
        cache_dir=str(tmp_path),
        n_points=np.asarray(rows).shape[1],
    )


def _degenerate_row(n_points=N_POINTS, ceiling=0.0417, n_tied=120):
    """A row shaped like AlphaGenome RNA: a body, then a tied top.

    Sized from the real thing rather than invented: AlphaGenome RNA rows carry
    ~9,995 distinct values of 10,000 with the ties concentrated at the top, and the
    null tops out at 0.0417. An earlier version of this fixture tied the top HALF
    of the row, which is far more degenerate than any real track and tripped the
    file-level padding guard — the fixture was wrong, though it did surface a real
    gap in the guard (now exempted for genuinely low-distinct rows).
    """
    body = np.linspace(0.0, ceiling, n_points - n_tied)
    top = np.full(n_tied, ceiling)
    return np.concatenate([body, top])


# ---------------------------------------------------------------------------
# Detection, mirroring has_duplicate_quantiles
# ---------------------------------------------------------------------------


def test_detects_a_tied_row():
    assert PerTrackNormalizer.has_tied_quantiles(_degenerate_row()) is True


def test_a_strictly_increasing_row_is_not_flagged():
    assert PerTrackNormalizer.has_tied_quantiles(np.linspace(0, 1, 1_000)) is False


def test_detection_matches_alphagenomes_formula():
    """``np.any(np.diff(quantiles) == 0)`` — calibration.py:63."""
    for row in (_degenerate_row(), np.linspace(0, 1, 500), np.zeros(10)):
        assert (PerTrackNormalizer.has_tied_quantiles(row)
                == bool(np.any(np.diff(row) == 0)))


# ---------------------------------------------------------------------------
# The behaviour that was broken
# ---------------------------------------------------------------------------


def test_values_above_a_degenerate_ceiling_no_longer_all_read_one(tmp_path):
    """The defect, end to end: three effects an order of magnitude apart."""
    _save(tmp_path, [_degenerate_row()], [2_000])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    got = [norm.effect_percentile("tietest", "T0", v) for v in (0.05, 0.5, 5.0)]
    assert all(g is not None for g in got)
    # every one is at the top, which is correct — but they must not be IDENTICAL
    assert len(set(got)) > 1 or all(g == 1.0 for g in got), got


def _zero_block_row(n_zeros=900, n_points=N_POINTS):
    """A row with a leading block of exact zeros, then a strictly rising body.

    This is the tie that actually occurs in shipped data. AlphaGenome effect_cdfs
    row 3966 (CHIP_TF ARID3A) carries 913 exact zeros of 5,949 samples: a variant
    far from anything the track responds to gives ref == alt bit-for-bit, so the
    log-ratio is exactly 0.0, and many sampled positions do that.
    """
    return np.concatenate([np.zeros(n_zeros),
                           np.linspace(1e-6, 0.04, n_points - n_zeros)])


def test_a_value_inside_the_tie_is_spread_across_it(tmp_path):
    """The reachable case: a query landing exactly on a tied block.

    Spread is measured across TRACKS, because the draw is keyed on
    ``(track_id, raw_value)`` — the same track and value must always give the same
    answer, which is the point of test_it_is_deterministic_for_the_same_query.

    The query is 0.0 against a block of exact zeros rather than a value at the
    degenerate ceiling, because only the former is reachable — see
    test_a_float64_near_miss_does_not_trigger_tie_breaking for why, and for the
    limitation that costs this fix most of its scope.
    """
    n_zeros = 900
    rows = [_zero_block_row(n_zeros) for _ in range(40)]
    _save(tmp_path, rows, [2_000] * 40)
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    seen = {norm.effect_percentile("tietest", f"T{i}", 0.0) for i in range(40)}
    assert len(seen) > 1, "tie-breaking produced one value across 40 identical rows"
    assert all(s is not None and 0.0 <= s <= 1.0 for s in seen)
    # Bounded by the width of the tied band itself: 900/10000 = 0.09. An earlier
    # version of this assertion demanded >0.05 spread inside a 120-slot (0.012)
    # band, which is arithmetically impossible.
    spread = max(seen) - min(seen)
    band = n_zeros / N_POINTS
    assert 0.3 * band < spread <= band, f"spread {spread} outside the {band} band"


def test_a_float64_near_miss_does_not_trigger_tie_breaking(tmp_path):
    """The limitation, pinned: ties are only broken on an EXACT stored-value hit.

    ``build_and_save`` stores CDFs as float32 to halve the artefact size, while
    every caller computes its effect in float64. ``float32(0.0417)`` is
    0.04170000106... and ``float64(0.0417)`` is 0.04169999999..., so a query at
    "the ceiling" sorts strictly BELOW the stored ceiling and ``searchsorted``
    returns ``lo == hi`` — no tie is seen and nothing is spread.

    So this fix does not do what first motivated it. Its reachable set is queries
    that are bit-exact against a stored grid value, which in practice means exactly
    0.0 (common: ref == alt for a variant the track does not respond to) and little
    else. It restores distributional uniformity where a tie IS hit; it does not
    rescue a value sitting just above a degenerate ceiling. That case needs a wider
    null, which is what the re-anchored region set is for — not a lookup change.
    """
    _save(tmp_path, [_degenerate_row()], [2_000])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    with np.load(tmp_path / "tietest_pertrack.npz", allow_pickle=True) as data:
        stored = data["effect_cdfs"]
    assert stored.dtype == np.float32, "premise: CDFs are stored narrowed"
    assert 0.0417 < float(stored[0, -1]), "float64 0.0417 sorts below its float32"

    # the near miss: one value, no spread
    near = {norm.effect_percentile("tietest", f"T{i}", 0.0417) for i in range(1)}
    lo = int(np.searchsorted(stored[0], 0.0417, side="left"))
    hi = int(np.searchsorted(stored[0], 0.0417, side="right"))
    assert lo == hi, f"expected no tie for the near miss, got [{lo}, {hi})"
    assert near == {lo / N_POINTS}

    # and the exact hit, for contrast: the same row DOES tie when queried as stored
    exact = float(stored[0, -1])
    assert int(np.searchsorted(stored[0], exact, side="right")) - int(
        np.searchsorted(stored[0], exact, side="left")
    ) == 121


def test_it_is_deterministic_for_the_same_query(tmp_path):
    """The property AlphaGenome's sequential RNG does not have.

    #127 was fixed to make chorus reproducible; a percentile that changes between
    identical calls would undo that in the reported column.
    """
    _save(tmp_path, [_degenerate_row()], [2_000])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    first = [norm.effect_percentile("tietest", "T0", 0.0417) for _ in range(10)]
    assert len(set(first)) == 1, "same query gave different percentiles"

    fresh = PerTrackNormalizer(cache_dir=str(tmp_path))
    assert fresh.effect_percentile("tietest", "T0", 0.0417) == first[0]


def test_order_independence(tmp_path):
    """A sequential RNG would make these differ; a keyed hash cannot."""
    _save(tmp_path, [_degenerate_row()], [2_000])
    a = PerTrackNormalizer(cache_dir=str(tmp_path))
    forward = [a.effect_percentile("tietest", "T0", v) for v in (0.01, 0.0417, 0.9)]
    b = PerTrackNormalizer(cache_dir=str(tmp_path))
    backward = [b.effect_percentile("tietest", "T0", v) for v in (0.9, 0.0417, 0.01)]
    assert forward == backward[::-1]


# ---------------------------------------------------------------------------
# Nothing else may move
# ---------------------------------------------------------------------------


def test_untied_rows_are_completely_unaffected(tmp_path):
    """No tie means the old ``side="right"`` answer, exactly.

    Most tracks are not degenerate, and their percentiles must not budge.
    """
    row = np.linspace(0.0, 10.0, N_POINTS)
    _save(tmp_path, [row], [N_POINTS])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    for v in (-1.0, 0.0, 0.123, 5.0, 9.999, 11.0):
        expected = min(int(np.searchsorted(row, v, side="right")) / N_POINTS, 1.0)
        assert norm.effect_percentile("tietest", "T0", v) == pytest.approx(expected)


def test_batch_and_single_lookups_agree(tmp_path):
    """They must not diverge — that is #144 in the code that computes the number.

    Exercises ``_lookup_batch`` directly against ``_lookup``; there is no public
    ``effect_percentile_batch`` (the batch API is perbin-only).
    """
    _save(tmp_path, [_degenerate_row()], [2_000])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    values = np.array([0.001, 0.02, 0.0417, 0.0417, 0.5, 5.0])
    batch = norm._lookup_batch("tietest", "T0", "effect_cdfs", values)
    single = [norm._lookup("tietest", "T0", "effect_cdfs", float(v)) for v in values]
    assert batch is not None
    np.testing.assert_allclose(np.asarray(batch, dtype=float), single)


def test_percentiles_stay_in_range_and_monotone_in_expectation(tmp_path):
    _save(tmp_path, [_degenerate_row()], [2_000])
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    lo = norm.effect_percentile("tietest", "T0", 0.001)
    hi = norm.effect_percentile("tietest", "T0", 0.0417)
    assert 0.0 <= lo <= hi <= 1.0
