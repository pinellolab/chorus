"""Every oracle must have a declared IGV display pooling, or this fails.

``_calculate_track_bin_size`` chose max-pooling from a hardcoded list of two oracle
NAMES, with everything else falling through to mean. Cherimoya is a BPNet-family model
with the same 1 bp point-profile output as ChromBPNet, was not in the list, and so was
mean-pooled over 349 bp bins on the SORT1 multi-oracle panel.

Measured consequence, DNASE:ENCSR149XIL at chr1:109,274,968 over the committed
1,048,396 bp window:

    ensemble 1 bp profile peak      11.1031
    max-pooled  -> rendered          3.0000   (the ceiling ChromBPNet also reaches)
    mean-pooled -> rendered          0.5467   <- what shipped

A 5.5x display-only dilution, on the same 0-3 axis as ChromBPNet, in a report whose
entire purpose is cross-oracle comparison. Scores were never affected -- the 501 bp
window sum is linear, so Cherimoya's log2FC 1.4576 vs ChromBPNet's 1.3756 was right
all along.

The fix is not a cleverer predicate. Two candidate universal rules were measured and
both fail to express the intent:

  * ``resolution <= 1`` happens to select the right set today only because of which
    oracles exist; it carries no statement about *why*, so the next oracle inherits a
    coincidence rather than a decision.
  * "spikiness" read off the shipped artefact points the wrong way: perbin max/p99 is
    22 for Cherimoya against 65 for AlphaGenome.

So pooling started as a declared per-oracle fact. It is no longer: when a track's values
are display-scaled, ``choose_aggregation`` measures what max-pooling does to the floor and
overrides the declaration. The frozensets survive as the **fallback** for un-rescaled tracks,
which have no display scale to ask the question against -- and this test keeps that fallback
honest, so an oracle added without a decision fails here rather than silently rendering at a
fifth of its height. The measured behaviour is pinned in
tests/test_display_scale_is_measured_not_declared.py.

AlphaGenome was initially left on mean-pooling on the argument that its 1 bp output is
dense coverage, where max over a 349 bp bin lifts the baseline as well as the peak. That
argument is still true and is now measured
(:func:`test_alphagenome_max_pools_by_maintainer_decision` has the table), but the
maintainer decided on 2026-08-10 to max-pool it anyway, so that every panel in a
cross-oracle report is computed the same way. Recorded here because the earlier revision
of this docstring asserted the opposite, and a test file arguing against its own
assertions is worse than either choice.
"""
from __future__ import annotations

import pytest

from chorus.analysis._igv_report import (
    _COVERAGE_ORACLES,
    _POINT_PROFILE_ORACLES,
    _calculate_track_bin_size,
)


def _registered_oracles() -> set[str]:
    """Oracle names as the rest of chorus knows them."""
    from chorus.oracles import ORACLES

    return set(ORACLES)


def test_every_registered_oracle_has_a_declared_pooling():
    """The guard against the next Cherimoya."""
    declared = _POINT_PROFILE_ORACLES | _COVERAGE_ORACLES | {"legnet"}
    undeclared = sorted(_registered_oracles() - declared)
    assert not undeclared, (
        f"{undeclared} have no declared IGV display pooling, so they fall through to "
        f"mean-pooling. If an oracle emits a base-resolution point profile (BPNet "
        f"family) add it to _POINT_PROFILE_ORACLES; if it emits coverage add it to "
        f"_COVERAGE_ORACLES. Falling through silently is how Cherimoya shipped at 5.5x "
        f"below its true height."
    )


def test_the_two_sets_are_disjoint():
    overlap = _POINT_PROFILE_ORACLES & _COVERAGE_ORACLES
    assert not overlap, f"{sorted(overlap)} declared both point-profile and coverage"


@pytest.mark.parametrize("oracle", sorted(_POINT_PROFILE_ORACLES))
def test_point_profile_oracles_max_pool_at_every_window_width(oracle: str):
    """Including the wide windows where the budget bound rewrites bin_size.

    The bound must widen the bin without turning max-pooling into mean-pooling --
    that was already true and is re-pinned here because Cherimoya now depends on it.
    """
    for window_bp in (2_114, 100_000, 1_048_396, 5_000_000):
        bin_size, aggregation = _calculate_track_bin_size(
            resolution=1, window_bp=window_bp, source_oracle=oracle,
        )
        assert aggregation == "max", (
            f"{oracle} at window {window_bp} bp pools by {aggregation}; a one-base "
            f"peak would be diluted by the bin width"
        )
        assert bin_size >= 1


def test_cherimoya_and_chrombpnet_agree_on_aggregation():
    """The specific comparison the multi-oracle report exists to make.

    Both are BPNet-family 1 bp models drawn on the same 0-3 axis. If they disagree on
    pooling, the panel compares heights that were computed differently -- which is the
    defect this test was written for.
    """
    window = 1_048_396
    _, cherimoya_agg = _calculate_track_bin_size(
        resolution=1, window_bp=window, source_oracle="cherimoya",
    )
    _, chrombpnet_agg = _calculate_track_bin_size(
        resolution=1, window_bp=window, source_oracle="chrombpnet",
    )
    assert cherimoya_agg == chrombpnet_agg == "max"


def test_coverage_oracles_still_mean_pool():
    """The pre-binned oracles stay on mean, and that is not an oversight.

    Enformer (128 bp) and Borzoi (32 bp) have already integrated over their own bin, so
    a display bin collapses only 2-11 native bins; max there adds noise without
    recovering anything. Sei and EPInformer-seq emit a window statistic rather than a
    profile.

    AlphaGenome used to be in this set and is deliberately no longer: see
    ``test_alphagenome_max_pools_by_maintainer_decision``.
    """
    for oracle in sorted(_COVERAGE_ORACLES):
        _, aggregation = _calculate_track_bin_size(
            resolution=1, window_bp=1_048_396, source_oracle=oracle,
        )
        assert aggregation == "mean", (
            f"{oracle} is declared coverage but pools by {aggregation}"
        )


def test_alphagenome_max_pools_by_maintainer_decision():
    """AlphaGenome is max-pooled, and the trade it makes is pinned here.

    It emits DNase/CAGE at 1 bp, so mean-pooling divided a one-base feature by the
    display bin width exactly as it did for Cherimoya. Maintainer decision on
    2026-08-10 was to treat it like the BPNet-family models so that every panel in a
    cross-oracle report is computed the same way.

    It is NOT the same case as ChromBPNet/Cherimoya, and the measurement is recorded so
    the trade is visible rather than discovered later. A BPNet profile is sparse spikes
    on a near-zero floor (Cherimoya null p50 0.075, p99 3.38), so max recovers the peak
    and leaves the floor. AlphaGenome's 1 bp output is dense coverage (p50 0.020,
    p99 0.285), so the max of ~349 dense samples lands near the upper tail almost
    everywhere. Measured on the committed SORT1 panel, 1,048,396 bp window:

                          peak            bins > 1.0        mean displayed
        DNASE:HepG2   2.918 -> 3.000   1.96% -> 32.61%     0.0800 -> 0.9915
        CAGE:HepG2    2.567 -> 3.000   1.40% -> 22.06%     0.0630 -> 0.6417
        H3K27ac       3.000 -> 3.000   2.08% ->  2.39%     0.0709 -> 0.0838

    So the peak became comparable and the baseline rose by more than 12x on DNase. The
    128 bp histone tracks are effectively untouched, which is the control: it confirms
    the effect comes from collapsing many native bins, not from the pooling choice per se.

    If this reads badly in review, the revert is moving two names between the two
    frozensets -- this test is what makes that a deliberate act rather than a silent one.
    """
    for oracle in ("alphagenome", "alphagenome_pt"):
        for window_bp in (2_114, 1_048_396):
            _, aggregation = _calculate_track_bin_size(
                resolution=1, window_bp=window_bp, source_oracle=oracle,
            )
            assert aggregation == "max", (
                f"{oracle} at window {window_bp} pools by {aggregation}; it is declared "
                f"a base-resolution track and must match ChromBPNet/Cherimoya"
            )

    # The 128 bp tracks collapse only 2 native bins, which is why they barely move.
    bin_size, _ = _calculate_track_bin_size(
        resolution=128, window_bp=1_048_396, source_oracle="alphagenome",
    )
    assert bin_size // 128 == 2, (
        f"expected a 128 bp AlphaGenome track to collapse 2 native bins at this window, "
        f"got {bin_size // 128}; the docstring's histone control assumes it"
    )


def test_mean_pooling_a_point_profile_is_what_it_would_cost():
    """Quantify the defect from first principles, so the number in the docstring is
    not merely asserted in prose.

    A single-base spike of height H in a bin of width W mean-pools to H/W. At the
    committed window the bin is 349 bp, so an 11.10 peak becomes 0.0318 before the
    floor-rescale -- the rescale then lifts it to 0.547 against a p99 of 3.377,
    against 3.000 when max-pooled. This test pins the mechanism, not the rendering.
    """
    import numpy as np

    profile = np.zeros(1_000)
    profile[426] = 11.1031          # the measured ensemble peak, at its measured bin

    bin_size, aggregation = _calculate_track_bin_size(
        resolution=1, window_bp=1_048_396, source_oracle="cherimoya",
    )
    assert aggregation == "max"

    chunk = profile[349:698]        # one display bin containing the peak
    assert chunk.max() == pytest.approx(11.1031)
    assert chunk.mean() == pytest.approx(11.1031 / 349, rel=1e-6)
    # Ratio of what max preserves to what mean would have kept.
    assert chunk.max() / chunk.mean() == pytest.approx(349.0, rel=1e-6)
