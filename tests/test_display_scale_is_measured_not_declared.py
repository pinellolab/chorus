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
    CAGE but would also have log-scaled 130 other tracks, including 102 ChromBPNet ChIP and 7
    AlphaGenome TF-ChIP tracks. All four CDF statistics tried overlap between
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
        p99.9/p99        must-log p5 5.7;  must-stay p95 15.6
        p99/p95          must-log p5 3.0;  must-stay p95 10.0
        predicted clip   must-log p5 0.0028; must-stay p95 0.0045

    ``max/p99.9`` at threshold 50 shipped briefly and looked clean at 41x separation -- only
    because ChromBPNet's ChIP tracks had been left out of the protected set. It would have
    log-scaled 130 tracks: 102 ChromBPNet ChIP, 10 Enformer and 8 Borzoi CAGE, 7 AlphaGenome
    TF-ChIP, 2 ChromBPNet DNase and 1 Cherimoya DNase. On a 10,000-point grid ``int(0.9999*n)`` IS the last slot, so that
    statistic is a ratio to the single extreme order statistic the null protocol warns
    against.

    So this test asserts an ABSENCE: no CDF-derived bimodality predicate may come back.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "chorus"
    # Sweep the whole package, not just normalization.py: a first version of this guard read
    # one file, so a dangling ``normalization._BIMODAL_TAIL_JUMP`` pointer in _igv_report's
    # own docstring survived it and sent readers hunting for a symbol that never existed in
    # any committed tree.
    src = "\n".join(f.read_text() for f in sorted(root.rglob("*.py")))
    for banned in ("_BIMODAL_TAIL_JUMP", "display_scale_for"):
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
                                       peak_pctile=0.99, max_value=None, log_scale=False):
            from chorus.analysis._igv_report import _DISPLAY_MAX
            max_value = _DISPLAY_MAX if max_value is None else max_value
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
    log-scaled 102 ChromBPNet ChIP tracks that render correctly today.
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
        disp[i * bins_per + 5] = ig._DISPLAY_MAX

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
            from chorus.analysis._igv_report import _DISPLAY_MAX
            return (np.clip(v / 1e6, 0.0, _DISPLAY_MAX) if log_scale
                    else np.full(v.size, _DISPLAY_MAX))

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

    # Uses _DISPLAY_MAX rather than a literal: this fixture silently stopped reaching the
    # ceiling when the axis moved 3.0 -> 4.0, which is the same hardcoded-literal failure the
    # mutation review found in the limits themselves.
    bins_per = 349
    v = np.zeros(300 * bins_per)
    v[::bins_per] = ig._DISPLAY_MAX               # one clipped native bin per display bin

    native = float((v >= ig._DISPLAY_MAX - 1e-3).mean())
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


def test_a_log_scaled_track_says_so_in_its_label():
    """A log panel's 1.0 is p99.9, not p99 -- all three render paths must disclose it.

    Two same-assay panels in one report can legitimately land on different bands: BCL11A's
    two CAGE:K562 tracks measured 0.053 and 0.036 saturation and only the first escalated.
    The 0-3 axis was always per-track (1.0 is *this* track's percentile, never a shared raw
    value), so mixing bands is not new -- shipping it unlabelled would be. ``legnet`` sets
    the precedent with its ``(per-track norm)`` suffix.
    """
    import re
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent / "chorus" / "analysis"
    from chorus.analysis._igv_report import _LOG_SCALE_LABEL
    assert _LOG_SCALE_LABEL.strip(), "the label must not be empty"

    for name in ("_igv_report.py", "multi_oracle_report.py", "causal.py"):
        src = (root / name).read_text()
        if "escalate_scale_if_saturated" not in src:
            continue
        assert "_LOG_SCALE_LABEL" in src, (
            f"{name} can re-render a track on the log band but never labels it, so the panel "
            f"claims 1.0 = p99 while drawing 1.0 = p99.9"
        )
        # and the escalation result must be bound to a real name, not thrown away
        assert re.search(r"(?<!_)used_log\s*=\s*escalate_scale_if_saturated", src) or \
               re.search(r",\s*used_log\s*=\s*escalate", src), (
            f"{name} discards the escalation flag, so the label can never be applied"
        )


def test_an_epsilon_improvement_is_not_accepted_as_a_fix():
    """``log_sat < sat`` alone would relabel a still-broken panel as fixed.

    Adversarial review demonstrated the regime with a real CDF row: linear saturation 0.550,
    log saturation 0.500. Under a bare "did it go down" test that ships with half the panel
    pinned at the ceiling -- 12x the limit -- having paid the log band's full cost (compressed
    peaks, floor moved p95 -> p99.5) for five percentage points. Acceptance now requires the
    band to clear the limit outright or at least halve the clipping.
    """
    from chorus.analysis import _igv_report as ig
    from chorus.analysis.normalization import PerTrackNormalizer

    bins_per = 1
    n = 2000
    raw = np.linspace(0.0, 10.0, n)
    from chorus.analysis._igv_report import _DISPLAY_MAX as _DM
    lin = np.full(n, _DM)
    lin[: int(n * 0.45)] = 0.5                     # linear: 55% clipped

    class _Marginal(PerTrackNormalizer):
        def __init__(self):
            pass

        def is_signed(self, *_a, **_k):
            return False

        def perbin_floor_rescale_batch(self, _o, _t, values, log_scale=False, **_kw):
            v = np.asarray(values, dtype=float)
            if not log_scale:
                return lin.copy()
            out = np.full(v.size, _DM)
            out[: int(v.size * 0.50)] = 0.5        # log: 50% clipped -- barely better
            return out

    out, _alt, used_log = ig.escalate_scale_if_saturated(
        _Marginal(), "alphagenome", "CAGE:HepG2", "tss_activity", raw, raw,
        lin, lin, bins_per, "max")
    assert not used_log, (
        "a 0.550 -> 0.500 saturation change is not a fix; accepting it relabels a broken "
        "panel as log-scaled while leaving it broken"
    )
    assert out is lin


def test_a_collapsed_log_band_is_rejected_rather_than_shipped_as_a_barcode():
    """Degenerate anchors render two levels only, and clipping guarantees the peak test.

    Reachable from shipped data: chrombpnet CHIP:HEK293:ZNF24 has p99.5 = -7.4e-07 and
    p99.9 = -3.3e-10, which the log path's ``max(x, 0.0)`` maps to the same 0.0, leaving
    ``denom`` pinned at its 1e-9 floor. Every value a hair above the floor then renders at
    exactly 3.0 and everything else at exactly 0.0 -- saturation can even fall, and the peak
    guard passes trivially, so without this check the track ships with all peak-height
    information destroyed.
    """
    from chorus.analysis import _igv_report as ig
    from chorus.analysis.normalization import PerTrackNormalizer

    bins_per, n = 1, 1000
    raw = np.zeros(n)
    raw[::10] = np.linspace(1.0, 500.0, raw[::10].size)
    from chorus.analysis._igv_report import _DISPLAY_MAX as _DM2
    lin = np.full(n, _DM2)
    lin[: int(n * 0.85)] = 0.0                     # 15% clipped -> triggers

    class _Collapsed(PerTrackNormalizer):
        def __init__(self):
            pass

        def is_signed(self, *_a, **_k):
            return False

        def perbin_floor_rescale_batch(self, _o, _t, values, log_scale=False, **_kw):
            v = np.asarray(values, dtype=float)
            if not log_scale:
                return lin.copy()
            # denom collapsed: a two-level barcode, and 13% clipped so saturation "improved"
            return np.where(v > 3e-9, _DM2, 0.0)

    out, _alt, used_log = ig.escalate_scale_if_saturated(
        _Collapsed(), "chrombpnet", "CHIP:HEK293:ZNF24", "tf_binding", raw, raw,
        lin, lin, bins_per, "max")
    assert not used_log, "a two-level barcode must be rejected, not shipped as a fix"
    assert out is lin


def test_the_limit_leaves_the_committed_enformer_and_borzoi_panels_alone():
    """The threshold is calibrated on the corpus, not on one panel.

    Measured over all 346 subtracks of the 19 committed IGV panels at the released baseline
    there is a clean gap with nothing in it:

        0.1085   alphagenome CAGE, linear band as drawn   <- must escalate
        ------ gap, 2.49x ------
        0.0435   enformer CAGE substantia nigra           <- must not move (corpus top)
        0.0022   corpus median

    A limit of 0.04 sat below that gap and would have escalated 45 subtracks (13%), including
    seven Enformer CAGE tracks at 0.042-0.063 that render acceptably -- silently invalidating
    committed panels that were never regenerated.
    """
    from chorus.analysis._igv_report import _MAX_DISPLAY_SATURATION

    # Re-measured 2026-08-12 for the 4.0 display ceiling; the previous pair (0.0656 / 0.0899)
    # was measured at 3.0 and no longer describes the axis. CAGE's trigger value has to come
    # from a fresh linear-band rescale because the committed panels show it post-escalation.
    HIGHEST_ACCEPTABLE_MEASURED = 0.0435     # enformer CAGE substantia nigra, SORT1 panel
    LOWEST_BROKEN_MEASURED = 0.1085          # alphagenome CAGE, linear band as drawn
    assert HIGHEST_ACCEPTABLE_MEASURED < _MAX_DISPLAY_SATURATION < LOWEST_BROKEN_MEASURED, (
        f"_MAX_DISPLAY_SATURATION={_MAX_DISPLAY_SATURATION} is outside the measured gap "
        f"({HIGHEST_ACCEPTABLE_MEASURED}, {LOWEST_BROKEN_MEASURED}); below it the committed "
        f"Enformer CAGE panels escalate and go stale, above it the SORT1 AlphaGenome panels "
        f"stay broken"
    )


def test_signed_tracks_are_excluded_from_the_measured_pooling():
    """Max-pooling signed data deletes the repressive half of every display bin.

    ``choose_aggregation`` asks whether max-pooling lifts the floor. A signed track has no
    floor at zero, so the question is meaningless and the answer is actively harmful: max over
    a bin holding a strong repression and a weak activation returns the activation. Measured
    on borzoi ENCFF734OLC+ (signed, 32 bp, 11 native bins per display bin), the measured
    choice flips mean -> max and takes displayed saturation 0.000 -> 0.138.

    2,253 shipped tracks are signed: borzoi 1,543, alphagenome 667, sei 40, legnet 3. They
    must keep the static geometry-based choice, which is what shipped and works.
    """
    import re
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent / "chorus" / "analysis"
    for name in ("_igv_report.py", "multi_oracle_report.py", "causal.py"):
        src = (root / name).read_text()
        if "choose_aggregation(" not in src:
            continue
        for m in re.finditer(r"agg_method\s*=\s*choose_aggregation\(", src):
            before = src[max(0, m.start() - 700):m.start()]
            assert "if not signed_track:" in before, (
                f"{name}: choose_aggregation is reached without the signed-track guard, so "
                f"signed tracks get max-pooled and their repressive half disappears"
            )


def test_every_igv_legend_qualifies_the_axis_when_a_track_can_be_log_scaled():
    """The legend used to promise p95/p99 for every track, unconditionally.

    All three reports print "1.0 = top 1% of bins genome-wide ... tracks comparable". Once a
    track can be re-rendered on log1p between p99.5 and p99.9, that sentence is false for it,
    and a reader comparing its peak height to a linear neighbour is comparing different units.
    """
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent / "chorus" / "analysis"
    for name in ("multi_oracle_report.py", "variant_report.py", "causal.py"):
        src = (root / name).read_text()
        if "top 1% of bins" not in src:
            continue
        assert "log scale" in src, (
            f"{name} prints the p95/p99 legend but never mentions the log band, so an "
            f"escalated track sits on the same axis with different semantics and no caveat"
        )


def test_the_pooling_limit_sits_inside_its_own_measured_gap():
    """Companion pin to the saturation limit, for the same reason.

    Adversarial mutation showed ``_MAX_POOL_FLOOR_LIMIT`` was free to move anywhere in
    (0.00999, 0.89886) with every test green -- so either of the two track fixes could be
    undone silently. At 0.8, AlphaGenome DNase (max-pooled floor 0.707) stays on max and the
    lifted-floor defect returns; at 0.011, ChromBPNet DNase flips to mean and Cherimoya's
    original 5.5x dilution comes back.

    Medians of the max-pooled DISPLAY values, read off the committed panels:

        keeps max                          flips to mean
          chrombpnet DNASE:HepG2   0.0000    alphagenome DNASE:K562 (BCL11A)  0.1990
          cherimoya  DNASE:HepG2   0.0000    alphagenome DNASE:HepG2 (SORT1)  0.7072
          alphagenome CAGE:HepG2   0.0644    alphagenome ATAC:HepG2  (SORT1)  0.9056
    """
    from chorus.analysis._igv_report import _MAX_POOL_FLOOR_LIMIT

    HIGHEST_KEEPING_MAX = 0.0644      # alphagenome CAGE:HepG2, SORT1 panel
    LOWEST_FLIPPING_TO_MEAN = 0.1990  # alphagenome DNASE:K562, BCL11A panel
    assert HIGHEST_KEEPING_MAX < _MAX_POOL_FLOOR_LIMIT < LOWEST_FLIPPING_TO_MEAN, (
        f"_MAX_POOL_FLOOR_LIMIT={_MAX_POOL_FLOOR_LIMIT} is outside the measured gap "
        f"({HIGHEST_KEEPING_MAX}, {LOWEST_FLIPPING_TO_MEAN}): below it a sparse BPNet profile "
        f"flips to mean and Cherimoya's 5.5x dilution returns, above it AlphaGenome's dense "
        f"1 bp tracks keep max and their floor stays lifted"
    )


def test_the_real_log_band_is_exercised_end_to_end():
    """Both escalation fixtures stub the rescaler, so nothing pinned the log band itself.

    Adversarial mutation: deleting ``if log_scale: floor_p, peak_p = _LOG_FLOOR_PCTILE,
    _LOG_PEAK_PCTILE`` leaves the log band anchored at p95/p99 like the linear one, and
    deleting the ``log1p`` block in ``perbin_floor_rescale_batch`` turns ``log_scale=True``
    into a plain wider linear band. Both left every test green.

    This runs the real normalizer against a real CDF and compares with the transform computed
    by hand, so the anchors AND the log1p are both pinned.
    """
    from chorus.analysis._igv_report import (
        _DISPLAY_MAX,
        _LOG_FLOOR_PCTILE,
        _LOG_PEAK_PCTILE,
        rescale_for_display,
    )
    from chorus.analysis.normalization import get_normalizer

    norm = get_normalizer(oracle_name="alphagenome")
    if norm is None:
        pytest.skip("no alphagenome normalizer")
    entry = norm._ensure_loaded("alphagenome")
    if entry is None:
        pytest.skip("alphagenome artefact not present")
    track_index = entry.get("track_index", {})
    tid = next((t for t in track_index if t.startswith("CAGE/")), None)
    if tid is None:
        pytest.skip("no CAGE track in the artefact")
    cdf = norm._find_matching_cdf(entry, track_index[tid], tid)
    if cdf is None:
        pytest.skip("no CDF for the CAGE track")

    c = np.asarray(cdf, dtype=float)
    n = len(c)
    raw = np.array([0.0, 0.001, 0.05, 1.0, 25.0, 400.0, 5000.0])

    def expected(floor_p, peak_p, log):
        f = float(c[min(int(floor_p * n), n - 1)])
        p = float(c[min(int(peak_p * n), n - 1)])
        v = raw.astype(float)
        if log:
            v = np.log1p(np.maximum(v, 0.0))
            f, p = float(np.log1p(max(f, 0.0))), float(np.log1p(max(p, 0.0)))
        return np.clip((v - f) / max(p - f, 1e-9), 0.0, _DISPLAY_MAX)

    got_log, cfg_log = rescale_for_display(
        raw, "tss_activity", normalizer=norm, oracle_name="alphagenome",
        assay_id=tid, log_scale=True)
    assert cfg_log["rescaled"] and cfg_log.get("log_scale") is True
    assert cfg_log["floor_pctile"] == _LOG_FLOOR_PCTILE, (
        f"log band anchored at p{cfg_log['floor_pctile']*100:g}, not the measured "
        f"p{_LOG_FLOOR_PCTILE*100:g} -- the linear anchors would leave CAGE saturated"
    )
    assert cfg_log["peak_pctile"] == _LOG_PEAK_PCTILE
    np.testing.assert_allclose(
        got_log, expected(_LOG_FLOOR_PCTILE, _LOG_PEAK_PCTILE, True), rtol=1e-12,
        err_msg="the log path does not apply log1p to the values AND both band anchors")

    # And it must NOT be reproducible by any linear band: a linear map preserves ratios of
    # differences, so if log1p were dropped this equality would hold for some (f, p).
    lin = expected(0.95, 0.99, False)
    unclipped = (got_log > 0.0) & (got_log < _DISPLAY_MAX) & (lin > 0.0) & (lin < _DISPLAY_MAX)
    if unclipped.sum() >= 3:
        a, b = got_log[unclipped], lin[unclipped]
        ratios = np.diff(a) / np.where(np.diff(b) == 0, np.nan, np.diff(b))
        finite = ratios[np.isfinite(ratios)]
        if finite.size >= 2:
            assert not np.allclose(finite, finite[0], rtol=1e-6), (
                "the log band is an affine rescale of the linear one, so log1p was dropped"
            )


def test_the_browser_reduction_is_max_for_every_track():
    """igv.js re-reduces features to pixels, and that stage wants max everywhere.

    This is the SECOND pooling stage, and it takes the opposite default from the first because
    the collapse factors differ by two orders of magnitude. The feature stage reduces ~349
    native bins per display bin, where max lifted AlphaGenome DNase's floor to 0.707 and so has
    to be measured per track. igv.js collapses only 2-3 of those already-pooled features per
    pixel, where max barely lifts a floor but mean still dilutes a sharp peak.

    Measured on the committed SORT1 panel at the browser's 3:1 ratio, peak lost by using mean:
    legnet 2.33x, alphagenome DNase 1.56x, chrombpnet 1.38x, CAGE 1.31x, cherimoya 1.14x.
    Mean also costs UNEQUALLY -- 1.38x for ChromBPNet against 1.14x for Cherimoya is a 1.2x
    relative distortion between the two tracks a cross-oracle panel exists to compare -- and it
    cancels signed tracks against themselves, which is why LegNet is worst.

    It used to be ``"max" if source_model in _HIGH_RES_ORACLES else "mean"``, a two-name list
    that gave Cherimoya and AlphaGenome CAGE ``mean`` in the browser right after the feature
    stage had measured them into ``max`` -- the original 5.5x dilution, one stage further down.
    """
    import re
    from pathlib import Path as _P

    from chorus.analysis._igv_report import _IGV_WINDOW_FUNCTION

    from chorus.analysis._igv_report import browser_window_function
    assert _IGV_WINDOW_FUNCTION == "max"
    # max for every track EXCEPT a log-scaled one, where the compressed top means max
    # promotes many near-ceiling bins over it: CAGE's saturation went 0.003 -> 0.023 (7.7x)
    # and the clipped flat tops read as coverage rather than TSS peaks.
    assert browser_window_function(False) == "max"
    assert browser_window_function(True) == "mean"

    root = _P(__file__).resolve().parent.parent / "chorus" / "analysis"
    offenders = []
    for name in ("_igv_report.py", "multi_oracle_report.py", "causal.py"):
        src = (root / name).read_text()
        for m in re.finditer(r'"windowFunction"\s*:\s*([^,\n]+)', src):
            val = m.group(1).strip()
            if val not in ("_IGV_WINDOW_FUNCTION", "wf",
                           "browser_window_function(used_log)"):
                offenders.append(f"{name}:{src[:m.start()].count(chr(10)) + 1} -> {val}")
        # and the per-oracle list must not be what decides it
        if re.search(r'"windowFunction".{0,60}_HIGH_RES_ORACLES', src, re.S):
            offenders.append(f"{name}: windowFunction keyed on _HIGH_RES_ORACLES again")
    assert not offenders, (
        f"these emit a browser reduction that is not the single measured default: {offenders}. "
        f"A per-oracle name list here is how Cherimoya got mean-reduced in the browser "
        f"immediately after the feature stage measured it into max."
    )


def test_the_axis_and_the_saturation_limit_stay_coupled():
    """Changing the display ceiling invalidates the saturation calibration. Say so loudly.

    ``_MAX_DISPLAY_SATURATION`` is a fraction of bins sitting AT ``_DISPLAY_MAX``, so its value
    is only meaningful for one ceiling. When the ceiling moved 3.0 -> 4.0 the previously measured
    gap (0.0656 -> 0.0899) silently became wrong, and three fixtures in this file quietly stopped
    reaching the ceiling at all -- passing for the wrong reason. Mutation testing showed the
    ceiling was free to move with every test green, which is what let that happen.

    So this pins the pair. If you change the ceiling, re-derive the limit: regenerate the panels,
    re-measure saturation across every committed subtrack, and re-measure the escalating track's
    LINEAR-band value separately (the panels show it post-escalation, so it cannot be read off
    them). Then update this test and the constant's docstring table together.
    """
    from chorus.analysis._igv_report import _DISPLAY_MAX, _MAX_DISPLAY_SATURATION

    assert _DISPLAY_MAX == 4.0, (
        f"_DISPLAY_MAX is {_DISPLAY_MAX}, but _MAX_DISPLAY_SATURATION={_MAX_DISPLAY_SATURATION} "
        f"was calibrated against a 4.0 ceiling (measured gap: enformer CAGE 0.0435 against "
        f"alphagenome CAGE 0.1085 on the linear band). Every saturation figure scales with the "
        f"ceiling, so the limit must be re-derived before this change is safe -- see the "
        f"docstring on _MAX_DISPLAY_SATURATION for the procedure."
    )
