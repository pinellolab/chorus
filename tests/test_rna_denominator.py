"""The RNA statistic must divide by the exon mask's extent, not its interval count.

Instance 4 of #144. AlphaGenome's shipped reference implementation
(``alphagenome_research/model/variant_scoring/gene_mask.py:53-56``) is::

    gene_length = gene_mask.sum(axis=0)
    ref_mean    = einsum('lt,lg->gt', ref, gene_mask) / gene_length
    return jnp.log(alt_mean + 1e-3) - jnp.log(ref_mean + 1e-3)

so: **mean over the mask, natural log, pseudocount 1e-3**, with the divisor being
the mask's **extent**. chorus's ``ln`` and its ``0.001`` were already right — the
one divergence was dividing by ``len(gene_exons)``, the interval count.

Measured on real GENCODE v48 genes, extent / interval-count:

=========  =========  ===========  =========
gene       n_exons    exonic bp    bp/exon
=========  =========  ===========  =========
CELSR2            34       10,981        323
PSRC1              7        1,932        276
SORT1             21        7,307        348
TERT              16        4,023        251
BCL11A             8       13,884      1,736
MYC                5        5,344      1,069
=========  =========  ===========  =========

Median **347x** too small, range 251-1,736x. Because the ``1e-3`` pseudocount is
fixed it does **not** cancel in the log ratio, so a too-small denominator leaves
the effect *under*-damped — chorus **overstated** RNA effects. That matters for
reading the direction of the fix: RNA effects looking tiny was never explained by
this, and correcting it makes them smaller, not larger.

**Why bins and not bases.** The two are not interchangeable:

* at AlphaGenome's ``resolution=1`` a bin *is* a base, so bins == ``gene_mask.sum()``
  and the formula is the reference one exactly;
* at Borzoi's 32 bp, bins matches that builder's own
  ``np.mean(pred[rna_exon_bins])`` units, so its 1,543 RNA background rows stay
  valid. A **bases** denominator would swap today's overstatement for a fresh
  ~32x understatement of the opposite sign;
* bins counts what was actually summed, so exons clipped by the prediction window
  are weighted correctly instead of silently.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from chorus.core.interval import GenomeRef, Interval
from chorus.core.result import OraclePredictionTrack

PRED_START = 1_000_000


def _track(resolution: int, n_bins: int, fill: float = 1.0):
    ref = GenomeRef(
        chrom="chr1", start=PRED_START,
        end=PRED_START + n_bins * resolution, fasta="/nonexistent.fa",
    )
    interval = Interval.make(ref) if hasattr(Interval, "make") else Interval(reference=ref)
    return OraclePredictionTrack(
        source_model="test", assay_id="RNA_SEQ/test", assay_type="RNA",
        cell_type="TEST", query_interval=interval, prediction_interval=interval,
        input_interval=interval, resolution=resolution,
        values=np.full(n_bins, fill, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# region_bin_count: the denominator
# ---------------------------------------------------------------------------


def test_bin_count_equals_bases_at_resolution_1():
    """Which is why this is the reference formula for AlphaGenome's RNA tracks."""
    track = _track(1, 20_000)
    start = PRED_START + 100
    assert track.region_bin_count("chr1", start, start + 350) == 350


def test_bin_count_is_in_bins_not_bases_at_coarse_resolution():
    """Borzoi at 32 bp: a 320 bp exon is 10 bins, and 10 is the right divisor."""
    track = _track(32, 4096)
    start = PRED_START + 32 * 10
    assert track.region_bin_count("chr1", start, start + 320) == 10


def test_bin_count_matches_what_score_region_actually_summed():
    """The invariant that keeps numerator and denominator consistent."""
    for res in (1, 32, 128):
        track = _track(res, 4096, fill=1.0)
        for length in (1, res, res * 3 + 7, res * 50):
            start = PRED_START + res * 7 + 3
            total = track.score_region("chr1", start, start + length, "sum")
            bins = track.region_bin_count("chr1", start, start + length)
            # values are all 1.0, so the sum IS the bin count
            assert total == float(bins), f"res={res} len={length}"


def test_bin_count_none_when_region_misses_the_prediction():
    track = _track(128, 1024)
    assert track.region_bin_count("chr1", 1, 2) is None
    assert track.region_bin_count("chr2", PRED_START, PRED_START + 500) is None


def test_bin_span_is_shared_by_score_region_and_bin_count():
    """They must never disagree about which bins a region covers (#144)."""
    track = _track(32, 4096)
    start = PRED_START + 999
    span = track.region_bin_span("chr1", start, start + 700)
    assert span is not None
    assert span[1] - span[0] == track.region_bin_count("chr1", start, start + 700)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def _rna_effect(ref_track, alt_track, exons):
    from chorus.analysis.scorers import LAYER_CONFIGS, score_track_effect

    return score_track_effect(
        ref_track, alt_track, "chr1", PRED_START + 500,
        layer_config=LAYER_CONFIGS["gene_expression"], gene_exons=exons,
    )


def test_rna_value_is_the_per_bin_mean_over_the_mask():
    """ref_value must be sum/extent, so a uniform track returns its own level."""
    ref = _track(1, 20_000, fill=4.0)
    alt = _track(1, 20_000, fill=4.0)
    exons = [
        {"chrom": "chr1", "start": PRED_START + 100, "end": PRED_START + 400},
        {"chrom": "chr1", "start": PRED_START + 1000, "end": PRED_START + 1600},
    ]
    out = _rna_effect(ref, alt, exons)
    assert out is not None
    # 900 bp of mask at 4.0 each -> mean 4.0, NOT 3600/2 = 1800
    assert out["ref_value"] == pytest.approx(4.0)
    assert out["raw_score"] == pytest.approx(0.0)


def test_interval_count_denominator_would_inflate_by_the_exon_length():
    """Guards the regression directly: /n_exons gives 450x this value here."""
    ref = _track(1, 20_000, fill=4.0)
    alt = _track(1, 20_000, fill=4.0)
    exons = [
        {"chrom": "chr1", "start": PRED_START + 100, "end": PRED_START + 400},
        {"chrom": "chr1", "start": PRED_START + 1000, "end": PRED_START + 1600},
    ]
    out = _rna_effect(ref, alt, exons)
    total = 300 * 4.0 + 600 * 4.0
    assert out["ref_value"] != pytest.approx(total / len(exons))
    assert out["ref_value"] == pytest.approx(total / 900)


def test_matches_the_reference_formula_including_the_pseudocount():
    """ln(alt_mean + 1e-3) - ln(ref_mean + 1e-3), evaluated end to end."""
    ref = _track(1, 20_000, fill=0.5)
    alt = _track(1, 20_000, fill=2.0)
    exons = [{"chrom": "chr1", "start": PRED_START + 10, "end": PRED_START + 1010}]
    out = _rna_effect(ref, alt, exons)
    expected = math.log(2.0 + 1e-3) - math.log(0.5 + 1e-3)
    assert out["raw_score"] == pytest.approx(expected, rel=1e-12)


def test_the_mean_and_sum_framings_are_the_same_statistic():
    """``ln(A/L + e) - ln(R/L + e) == ln(A + eL) - ln(R + eL)``.

    So "mean over the exon mask" (AlphaGenome's wording) and "sum of a
    transcript's exons" (Luca's) are algebraically identical — the only content of
    the choice is whether the pseudocount scales with mask length. Verified to
    machine precision, which is why no sum-vs-mean change was needed.
    """
    rng = np.random.default_rng(0)
    e = 1e-3
    A = rng.exponential(5, 50_000)
    R = rng.exponential(5, 50_000)
    L = rng.integers(50, 50_000, 50_000)
    lhs = np.log(A / L + e) - np.log(R / L + e)
    rhs = np.log(A + e * L) - np.log(R + e * L)
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-12)


def test_borzoi_resolution_is_not_silently_mis_scaled():
    """A bases denominator would understate by ~resolution; bins must not.

    ``scorers.py`` reads no resolution anywhere else, so this is the one place the
    distinction bites — and getting it wrong would stale Borzoi's 1,543 RNA rows.
    """
    exons = [{"chrom": "chr1", "start": PRED_START + 320, "end": PRED_START + 3520}]
    out1 = _rna_effect(_track(1, 20_000, 3.0), _track(1, 20_000, 3.0), exons)
    out32 = _rna_effect(_track(32, 4_096, 3.0), _track(32, 4_096, 3.0), exons)
    # a uniform track has the same per-bin mean at either resolution
    assert out1["ref_value"] == pytest.approx(out32["ref_value"]) == pytest.approx(3.0)


def test_no_usable_exon_returns_none():
    ref = _track(1, 1_000)
    alt = _track(1, 1_000)
    assert _rna_effect(ref, alt, [{"chrom": "chr9", "start": 5, "end": 9}]) is None
    assert _rna_effect(ref, alt, []) is None
