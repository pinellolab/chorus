"""Percentile denominator must match the population `rank` is measured on.

`rank` comes from `np.searchsorted` over the stored CDF row, so it lives on
`[0, cdf_width]`. Dividing it by the raw sample count — which is what
`_get_denominator` used to do whenever `counts < cdf_width` — inflates every
percentile by `cdf_width / counts` and clamps the top of the range to 1.0.

Short rows are not padded: `ReservoirSampler.to_cdf_matrix` interpolates a
short sample onto the full grid, so every entry is a real quantile estimate.
"""

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

WIDTH = 10_000
TRACK = "TEST:track"


def _normalizer(n_samples: int, width: int = WIDTH) -> PerTrackNormalizer:
    """A normalizer over one synthetic track whose CDF is 0..width-1.

    `n_samples` is what the builder recorded in `*_counts`; the row is
    always `width` wide, mirroring how the real NPZs are written.
    """
    row = np.arange(width, dtype=np.float64)[None, :]
    nz = PerTrackNormalizer()
    nz._loaded["synthetic"] = {
        "track_ids": [TRACK],
        "track_index": {TRACK: 0},
        "effect_cdfs": row,
        "summary_cdfs": row.copy(),
        "effect_counts": np.array([n_samples], dtype=np.int64),
        "summary_counts": np.array([n_samples], dtype=np.int64),
        "signed_flags": np.array([False]),
    }
    return nz


# The real per-oracle effect_counts, so this test tracks shipped reality.
#   alphagenome ~1_909 (the severe case), the ~9_60x cluster for
#   borzoi/enformer/epinformerseq/legnet/sei, and a >width case.
@pytest.mark.parametrize("n_samples", [1_909, 9_608, 9_609, WIDTH, 18_672])
def test_denominator_is_the_grid_width(n_samples):
    nz = _normalizer(n_samples)
    assert nz._get_denominator(nz._loaded["synthetic"], "effect_cdfs", 0) == WIDTH


@pytest.mark.parametrize("n_samples", [1_909, 9_608, WIDTH, 18_672])
def test_midpoint_of_the_background_is_the_50th_percentile(n_samples):
    """The value at grid index 5000 is the median by construction."""
    nz = _normalizer(n_samples)
    # searchsorted(row, 4999.0, side='right') == 5000
    pct = nz.effect_percentile("synthetic", TRACK, 4_999.0)
    assert pct == pytest.approx(0.5, abs=1e-6), (
        f"n_samples={n_samples}: got {pct}, expected 0.5. A count-based "
        f"denominator would give {min(5000 / min(n_samples, WIDTH), 1.0)}."
    )


@pytest.mark.parametrize("n_samples", [1_909, 9_608, WIDTH, 18_672])
def test_only_the_very_top_reaches_the_100th_percentile(n_samples):
    """Nothing below the maximum may pin to 1.0.

    With the old denominator, everything at or above rank `n_samples`
    clamped — for AlphaGenome that was the top 80.9% of the range.
    """
    nz = _normalizer(n_samples)
    assert nz.effect_percentile("synthetic", TRACK, float(WIDTH - 1)) == pytest.approx(1.0)
    # one grid step below the top must be strictly below 1.0
    just_below = nz.effect_percentile("synthetic", TRACK, float(WIDTH - 2))
    assert just_below < 1.0
    assert just_below == pytest.approx((WIDTH - 1) / WIDTH, abs=1e-9)


def test_no_inflation_across_the_whole_range():
    """Reported percentile == grid position, for the severe AlphaGenome case."""
    nz = _normalizer(1_909)
    for gi in (0, 1, 100, 2_500, 5_000, 7_500, 9_000, 9_999):
        pct = nz.effect_percentile("synthetic", TRACK, float(gi))
        assert pct == pytest.approx((gi + 1) / WIDTH, abs=1e-9), f"grid index {gi}"


def test_zero_count_still_reports_no_background():
    """The counts array is still load-bearing for _has_samples."""
    nz = _normalizer(0)
    assert nz.effect_percentile("synthetic", TRACK, 5_000.0) is None


def test_signed_lookup_is_centred_after_the_fix():
    """A signed track's median must map to 0.0, not saturate at +1.0."""
    nz = _normalizer(1_909)
    assert nz.effect_percentile(
        "synthetic", TRACK, 4_999.0, signed=True
    ) == pytest.approx(0.0, abs=1e-6)
