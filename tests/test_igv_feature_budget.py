"""No single IGV track may emit an unbounded number of JSON features.

`_calculate_track_bin_size` lets each oracle pick a preferred bin size, and
most fall through to a ~3,000-feature budget. Two did not: ChromBPNet pins
20 bp (deliberately — max-pooling a 1 bp profile over 20 bp preserves peak
shape, see PR #79) and LegNet returned the bare `resolution`, which makes
`bins_per = max(1, bin_size // resolution) == 1` in `_downsample_to_features`
— i.e. **no downsampling at all**, one JSON feature per input bin.

Measured before the fix, features emitted per track:

    legnet      res=1  window=200,000     ->   200,000
    legnet      res=1  window=1,048,576   -> 1,048,576
    chrombpnet  res=1  window=200,000     ->    10,000
    sei         res=1  window=4,096       ->     4,096
    chrombpnet  res=1  window=2,114       ->       106   (fine — real window)
    alphagenome res=1  window=1,048,576   ->     3,005
    enformer    res=128 window=114,688    ->       896
    borzoi      res=32  window=524,288    ->     3,277

That is how `rs12740374_SORT1_legnet_report.html` reached **131 MB** from
1.29 MB, and the consolidated multi-oracle report 139 MB — both above
GitHub's hard 100 MiB per-file limit, so neither could be committed at all.
`audits/AUDIT_CHECKLIST.md:172` had recorded it as P0 with a *manual* guard
("check `find examples -name '*.html' -size +50M` is empty before
regenerating"), which was duly forgotten.

See https://github.com/pinellolab/chorus/issues/129.
"""

import numpy as np
import pytest

from chorus.analysis._igv_report import (
    _MAX_FEATURES_PER_TRACK,
    _calculate_track_bin_size,
    _downsample_to_features,
)

# Every oracle chorus registers, with the window and native resolution it
# actually renders at, plus the pathological large-window cases that the
# multi-oracle and causal reports can produce.
ORACLE_WINDOWS = [
    ("chrombpnet", 1, 2_114),
    ("chrombpnet", 1, 200_000),
    ("legnet", 1, 200),
    ("legnet", 1, 200_000),
    ("legnet", 1, 1_048_576),
    ("alphagenome", 1, 1_048_576),
    ("enformer", 128, 114_688),
    ("borzoi", 32, 524_288),
    ("sei", 1, 4_096),
    ("cherimoya", 1, 2_114),
    ("epinformerseq", 1, 2_114),
    # An unknown oracle must also be bounded — the budget is the floor for
    # everyone, not a per-oracle opt-in.
    ("some_future_oracle", 1, 500_000),
]


def _feature_count(resolution: int, window_bp: int, oracle: str) -> int:
    """Features `_downsample_to_features` would emit for a full window."""
    bin_size, _agg = _calculate_track_bin_size(resolution, window_bp, oracle)
    n_input = window_bp // resolution
    bins_per = max(1, bin_size // resolution)
    return (n_input + bins_per - 1) // bins_per


@pytest.mark.parametrize("oracle,resolution,window_bp", ORACLE_WINDOWS)
def test_feature_count_is_bounded(oracle, resolution, window_bp):
    """Arithmetic check: no (oracle, window) exceeds the budget."""
    n = _feature_count(resolution, window_bp, oracle)
    assert n <= _MAX_FEATURES_PER_TRACK, (
        f"{oracle} at resolution={resolution}, window={window_bp:,} would emit "
        f"{n:,} features (budget {_MAX_FEATURES_PER_TRACK:,}). An unbounded "
        f"track is what made the LegNet report 131 MB."
    )


@pytest.mark.parametrize("oracle,resolution,window_bp", ORACLE_WINDOWS)
def test_downsample_actually_respects_the_budget(oracle, resolution, window_bp):
    """End-to-end through the real downsampler, with skip_zeros disabled.

    `skip_zeros=True` would mask the problem by dropping near-zero bins, and
    callers pass `skip_zeros=not (floor_ok or signed_track)` — so for any
    normalised or signed track (LegNet is signed) it is already `False` and
    nothing is dropped. Test the unmasked path.
    """
    bin_size, agg = _calculate_track_bin_size(resolution, window_bp, oracle)
    n_input = window_bp // resolution
    # A non-zero ramp: nothing is near-zero, so no bin could be skipped even
    # if skip_zeros were on.
    values = np.linspace(1.0, 2.0, n_input)

    features = _downsample_to_features(
        values, "chr1", 1_000, resolution, bin_size,
        skip_zeros=False, aggregation_method=agg,
    )

    assert len(features) <= _MAX_FEATURES_PER_TRACK, (
        f"{oracle} at resolution={resolution}, window={window_bp:,} emitted "
        f"{len(features):,} features (budget {_MAX_FEATURES_PER_TRACK:,})"
    )


def test_chrombpnet_keeps_its_20bp_binning_in_its_real_window():
    """The budget must not silently undo a deliberate per-oracle choice.

    ChromBPNet's 20 bp max-pooling exists because mean-pooling a 1 bp profile
    dilutes sharp peaks below the floor-rescale threshold (PR #79). At its
    actual 2,114 bp input window that is well inside the budget, so the cap
    must leave it alone.
    """
    bin_size, agg = _calculate_track_bin_size(1, 2_114, "chrombpnet")
    assert bin_size == 20
    assert agg == "max"


def test_max_pooling_is_preserved_where_it_was_chosen():
    """Widening a bin must not change the aggregation an oracle asked for."""
    for oracle in ("chrombpnet", "legnet"):
        _bin, agg = _calculate_track_bin_size(1, 1_048_576, oracle)
        assert agg == "max", f"{oracle} lost its max-pooling"


def test_bin_size_is_never_below_resolution():
    """A bin smaller than the native resolution is meaningless.

    `bins_per = max(1, bin_size // resolution)` silently floors it to 1, which
    is exactly the no-downsampling bug. Assert the invariant directly.
    """
    for oracle, resolution, window_bp in ORACLE_WINDOWS:
        bin_size, _agg = _calculate_track_bin_size(resolution, window_bp, oracle)
        assert bin_size >= resolution, (
            f"{oracle}: bin_size {bin_size} < resolution {resolution}"
        )
