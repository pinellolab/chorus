"""The display scale and the pooling operator are chosen from data, and stay scoped.

Two render decisions used to be hardcoded per oracle name, and both were wrong for at least
one track:

  POOLING   ``_calculate_track_bin_size`` max-pooled ChromBPNet and LegNet by name and
            mean-pooled everything else, so Cherimoya -- a BPNet-family 1 bp model -- rendered
            at 0.547 instead of 3.000 on the same axis as ChromBPNet.
  SCALE     ``floor=p95, peak=p99, linear`` assumes signal decays smoothly out of the
            background. AlphaGenome's CAGE has p99=0.0405 against a max of 852, so every real
            TSS from strength 1 to 3000 rendered at exactly 3.00 -- 13.1% of its display bins
            pinned at the ceiling.

Both are now measured per track. This file pins the scope of each, because the measurements
that got there were wrong three times before they were right, and each wrong version looked
plausible:

  * the pooling criterion was first the *median* of max-pooled values -- blind to CAGE, whose
    median was 0.023 while 13% of bins clipped;
  * then the *ink* fraction -- which flipped Cherimoya and ChromBPNet to mean and re-broke the
    original defect, because Cherimoya legitimately inks 41% of its bins and still reads well;
  * the scale anchors were first p99.9/p99.99, which erased CAGE (peak 1.24 of 3.0);
  * and the scale TRIGGER was first a genome-wide CDF ratio (max/p99.9 > 50), which fixed
    CAGE but would also have log-scaled 104 ChromBPNet ChIP tracks, 10 Enformer and 8 Borzoi
    CAGE tracks and a Cherimoya DNase track. All four CDF statistics tried overlap between
    the tracks that must change and the tracks that must not; the panel is measured instead.

Saturation, not ink, is what makes a panel unreadable. Cherimoya inks 41% and looks right
because only 1.3% of its bins clip.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis._igv_report import (
    _calculate_track_bin_size,
    choose_aggregation,
)

# Measured on the rendered SORT1 multi-oracle panel, 1,048,576 bp / 349 bp display bins.
# (track, saturated fraction) for the panels judged readable, and CAGE before the fix.
READABLE_SATURATION = {
    "chrombpnet DNASE": 0.003,
    "cherimoya DNASE": 0.013,
    "alphagenome DNASE": 0.000,
    "alphagenome CHIP:CEBPA": 0.007,
    "alphagenome CHIP:H3K27ac": 0.005,
}
CAGE_SATURATION_BEFORE = 0.131
CAGE_SATURATION_AFTER = 0.013


def test_no_genome_wide_cdf_statistic_decides_the_scale():
    """The scale is not predicted from the CDF, and this is why.

    Four genome-wide statistics were measured as candidate triggers across 20,343 tracks,
    splitting them into "must get the log band" (AlphaGenome CAGE/splice/PROCAP) and "was
    working, must not move" (everything else). Every one of them overlaps:

        max/p99.9        must-log down to 172, must-stay-linear up to 4212 (ChromBPNet ChIP)
        p99.9/p99        must-log p5 5.7, must-stay-linear p95 8.6
        p99/p95          must-log p5 3.0, must-stay-linear p95 8.1
        predicted clip   must-log p5 0.0028, must-stay-linear p95 0.0044

    ``max/p99.9`` at threshold 50 shipped briefly and looked clean at 41x separation -- only
    because ChromBPNet's ChIP tracks had been left out of the protected set. It would have
    log-scaled 104 ChromBPNet ChIP tracks, 10 Enformer and 8 Borzoi CAGE tracks and a
    Cherimoya DNase track. On a 10,000-point grid ``int(0.9999*n)`` IS the last slot, so that
    statistic is a ratio to the single extreme order statistic the null protocol warns
    against.

    So this test asserts an ABSENCE: no CDF-derived bimodality predicate may come back.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "chorus" / "analysis"
           / "normalization.py").read_text()
    for banned in ("_BIMODAL_TAIL_JUMP", "display_scale_for", "0.9999"):
        assert banned not in src, (
            f"normalization.py references {banned!r} again. The display scale is decided by "
            f"measuring the rendered panel (_igv_report.escalate_scale_if_saturated), not by "
            f"a genome-wide CDF statistic -- all four candidates overlap between the tracks "
            f"that must change and the tracks that must not."
        )
    # And the log band must stay where the panel measurement put it.
    from chorus.analysis._igv_report import _LOG_FLOOR_PCTILE, _LOG_PEAK_PCTILE
    assert (_LOG_FLOOR_PCTILE, _LOG_PEAK_PCTILE) == (0.995, 0.999), (
        "p99.9/p99.99 anchors were measured to drop CAGE's peak to 1.24 of 3.0, erasing it"
    )


def test_a_saturated_track_escalates_to_the_log_band():
    """The CAGE case, end to end, with a stub normalizer.

    A near-zero floor plus a few enormous peaks: on the linear band almost every peak-bearing
    display bin clips, and the log band separates them again.
    """
    from chorus.analysis import _igv_report as ig

    bins_per, n_bins = 349, 400
    raw = np.zeros(n_bins * bins_per)
    rng = np.random.default_rng(7)
    raw[:] = rng.uniform(0.0, 0.004, raw.size)            # background
    for i in range(0, n_bins, 6):                          # real TSSs, 1 to 3000
        raw[i * bins_per + 40] = 1.0 + 3000.0 * rng.random()

    # rescale_for_display gates on isinstance(normalizer, PerTrackNormalizer), so the
    # stub has to be one -- with a no-op __init__ so no artefact is loaded.
    from chorus.analysis.normalization import PerTrackNormalizer

    class _Stub(PerTrackNormalizer):
        """Linear band p95/p99 = 0.005/0.0405; the measured AlphaGenome CAGE geometry."""
        def __init__(self):
            pass

        def is_signed(self, *_a, **_k):
            return False

        def perbin_floor_rescale_batch(self, _o, _t, values, floor_pctile=0.95,
                                       peak_pctile=0.99, max_value=3.0, log_scale=False):
            floor, peak = (0.005, 0.0405) if not log_scale else (5.0, 25.0)
            v = np.asarray(values, dtype=float)
            if log_scale:
                v = np.log1p(np.maximum(v, 0.0))
                floor, peak = np.log1p(0.35), np.log1p(60.0)
            return np.clip((v - floor) / (peak - floor), 0.0, max_value)

    stub = _Stub()
    ok, lin_ref, lin_alt, signed = ig.apply_floor_rescale(
        stub, "alphagenome", "CAGE:HepG2", "tss_activity", raw, raw)
    assert ok and not signed
    lin_sat, _ = ig._display_saturation(lin_ref, bins_per, "max")
    assert lin_sat > ig._MAX_DISPLAY_SATURATION, (
        f"the fixture is meant to reproduce the CAGE defect but only clips {lin_sat:.3f}"
    )

    out_ref, _out_alt, used_log = ig.escalate_scale_if_saturated(
        stub, "alphagenome", "CAGE:HepG2", "tss_activity", raw, raw,
        lin_ref, lin_alt, bins_per, "max")
    assert used_log, "a track clipping more than the limit must be re-rendered on log"
    sat, peak = ig._display_saturation(out_ref, bins_per, "max")
    assert sat < lin_sat and peak >= 1.0, (
        f"log band left saturation {sat:.3f} (was {lin_sat:.3f}) and peak {peak:.2f}"
    )


def test_a_readable_track_is_left_alone():
    """The don't-touch-what-works guard: no escalation when nothing clips.

    This is the property the CDF-statistic version could not deliver -- it would have
    log-scaled 104 ChromBPNet ChIP tracks that render correctly today.
    """
    from chorus.analysis import _igv_report as ig

    bins_per, n_bins = 349, 400
    rng = np.random.default_rng(3)
    disp = rng.uniform(0.0, 0.4, n_bins * bins_per)
    # Two clipped display bins out of 400 = 0.5% saturation, the ChromBPNet/Cherimoya regime.
    # Note the spacing is in NATIVE bins: an earlier version of this fixture used
    # ``[::bins_per * 7]``, which is one clipped bin in every 7th DISPLAY bin -- 14%, and it
    # (correctly) triggered.
    for i in (11, 250):
        disp[i * bins_per + 5] = 3.0

    calls = []

    from chorus.analysis.normalization import PerTrackNormalizer

    class _Stub(PerTrackNormalizer):
        def __init__(self):
            pass

        def is_signed(self, *_a, **_k):
            return False

        def perbin_floor_rescale_batch(self, *_a, **kw):
            calls.append(kw.get("log_scale"))
            return np.zeros(10)

    out_ref, out_alt, used_log = ig.escalate_scale_if_saturated(
        _Stub(), "chrombpnet", "CHIP:K562:ZBTB11", "tf_binding",
        disp, disp, disp, disp, bins_per, "max")
    assert not used_log
    assert out_ref is disp and out_alt is disp, "an unsaturated track must be untouched"
    assert not calls, "an unsaturated track must not even be re-rescaled"


def test_the_log_band_is_rejected_when_it_erases_the_track():
    """Two-sided acceptance. Reducing saturation is not sufficient.

    p99.9/p99.99 anchors dropped CAGE's rendered peak to 1.24 of a 3.0 axis -- saturation
    went to zero because the signal was gone. Acceptance requires the peak to survive.
    """
    from chorus.analysis import _igv_report as ig

    bins_per = 349
    raw = np.zeros(200 * bins_per)
    raw[::bins_per] = 500.0
    lin = np.full(200 * bins_per, 3.0)             # fully saturated -> triggers

    from chorus.analysis.normalization import PerTrackNormalizer

    class _Erasing(PerTrackNormalizer):
        def __init__(self):
            pass

        def is_signed(self, *_a, **_k):
            return False

        def perbin_floor_rescale_batch(self, _o, _t, values, log_scale=False, **_kw):
            v = np.asarray(values, dtype=float)
            # a band so high that everything lands near zero
            return np.clip(v / 1e6, 0.0, 3.0) if log_scale else np.full(v.size, 3.0)

    out, _alt, used_log = ig.escalate_scale_if_saturated(
        _Erasing(), "alphagenome", "CAGE:HepG2", "tss_activity", raw, raw,
        lin, lin, bins_per, "max")
    assert not used_log, "a log band that erases the peak must be rejected"
    assert out is lin


def test_saturation_is_measured_as_drawn_not_natively():
    """Pooling is what turns a 1.2% native clip rate into a 13.1% displayed one.

    Measured natively, AlphaGenome CAGE (0.005-0.014) is indistinguishable from the ChIP
    tracks (0.001-0.008) that must not move -- because CAGE is 1 bp and collapses 349 native
    bins while ChIP is 128 bp and collapses 2. The separation only exists after pooling, so
    the trigger must be measured there.
    """
    from chorus.analysis import _igv_report as ig

    bins_per = 349
    v = np.zeros(300 * bins_per)
    v[::bins_per] = 3.0                            # one clipped native bin per display bin

    native = float((v >= 3.0 - 1e-3).mean())
    drawn, _ = ig._display_saturation(v, bins_per, "max")
    assert native < 0.01 < drawn, (
        f"native {native:.4f} vs drawn {drawn:.4f}: max-pooling must be what the "
        f"saturation measurement sees"
    )
    assert drawn == pytest.approx(1.0), "every display bin inherits its clipped native bin"
    mean_drawn, _ = ig._display_saturation(v, bins_per, "mean")
    assert mean_drawn == 0.0, "mean-pooling dilutes the same bin instead"


def test_pooling_keeps_a_sparse_spiky_track_on_max():
    """The Cherimoya guard. A track that is mostly floor with real spikes must keep max.

    Written because an 'ink fraction' criterion failed exactly here: Cherimoya's ink is 41%
    of display bins, and punishing that flipped it to mean and re-created the 5.5x defect.
    """
    rng = np.random.default_rng(0)
    n_bins, bins_per = 400, 349
    v = rng.uniform(0.0, 0.01, n_bins * bins_per)      # near-zero floor
    for i in range(0, n_bins, 3):                      # a third of bins carry a real spike
        v[i * bins_per + 100] = 3.0
    assert choose_aggregation(v, bins_per) == "max"


def test_pooling_moves_a_dense_track_to_mean():
    """The AlphaGenome-DNase guard: max over a dense bin lifts the floor into the signal."""
    rng = np.random.default_rng(1)
    v = rng.uniform(0.4, 0.9, 400 * 349)
    assert choose_aggregation(v, 349) == "mean"


def test_pooling_is_a_no_op_when_nothing_is_collapsed():
    v = np.linspace(0, 3, 500)
    assert choose_aggregation(v, 1) == "max"


def test_every_downsample_call_site_goes_through_choose_aggregation():
    """Three files duplicate the render path, and I patched one and reported a null result.

    ``_igv_report`` builds single-oracle panels, ``multi_oracle_report`` the consolidated
    one, and ``causal`` the causal report -- each with its own copy of the
    bin-size + downsample sequence. A fourth copy, or a fifth call site, must not silently
    skip the measured decision.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "chorus" / "analysis"
    offenders = []
    for name in ("_igv_report.py", "multi_oracle_report.py", "causal.py"):
        src = (root / name).read_text()
        calls = len(re.findall(r"_downsample_to_features\(", src))
        # the definition itself lives in _igv_report and is not a call site
        if name == "_igv_report.py":
            calls -= 1
        missing = [fn for fn in ("choose_aggregation", "escalate_scale_if_saturated")
                   if fn not in src]
        if calls and missing:
            offenders.append(f"{name} ({calls} call(s), missing {missing})")
    assert not offenders, (
        f"these render paths downsample without the measured pooling decision: {offenders}. "
        f"Patching only one of the three is how a change to this logic came back reporting "
        f"byte-identical output."
    )


def test_the_saturation_target_is_recorded():
    """Guard the number the fix was aimed at, not just the mechanism.

    A panel is unreadable when a large share of its bins clip, and the readable tracks on the
    shipped SORT1 panel sit at 0.0-1.3%. CAGE was at 13.1%; the fix brought it to 1.3%.
    """
    assert max(READABLE_SATURATION.values()) <= 0.02
    assert CAGE_SATURATION_BEFORE > 0.10, "CAGE's pre-fix saturation is the defect being fixed"
    assert CAGE_SATURATION_AFTER <= max(READABLE_SATURATION.values()) + 0.002, (
        "after the fix CAGE should clip no more than the tracks that already read well"
    )


@pytest.mark.parametrize("window_bp", [2_114, 100_000, 1_048_576])
def test_bin_size_still_respects_the_feature_budget(window_bp: int):
    """The scale work must not have disturbed the JSON feature cap (issue #129)."""
    for oracle in ("alphagenome", "cherimoya", "chrombpnet", "enformer", "borzoi"):
        bin_size, _ = _calculate_track_bin_size(
            resolution=1, window_bp=window_bp, source_oracle=oracle)
        assert window_bp // max(bin_size, 1) <= 4_000 + 1, (
            f"{oracle} at {window_bp} bp would emit more than the 4,000-feature budget"
        )
