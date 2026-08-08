"""A clamped percentile must not be the last word on how strong an effect is.

``effect_percentile`` is ``min(rank / denominator, 1.0)``. It therefore reaches
exactly 1.0 the moment an effect reaches the largest of the ~10k sampled background
effects for that track, and stays at 1.0 however much further the effect goes. Two
distinct failures follow, and both were measured rather than reasoned about:

1. **No resolution at the top.** At rs12740374 — the SORT1/CELSR2 locus, where the
   variant creates a canonical C/EBP motif — ``CHIP:HepG2:CEBPA:+`` scores +1.865
   against a null maximum of 1.682. It pins at 1.0, identical to what an effect of
   17.0 would report against the same track.

2. **The ceiling itself is unstable.** It is a single extreme order statistic, so
   its position carries large sampling variance. Re-anchoring Enformer's effect null
   made 12 of 12 ``tf_binding`` tracks *wider* at p99 (0.2694 → 0.4402, +63%) while
   11 of 12 reported a *lower* maximum (3.5706 → 3.0272, −15%). Judged on pinning
   alone that reads as a regression; judged on the tail it is a clear improvement.

The fix is read-side and needs no rebuild, because the bound is already in the
shipped artefacts: it is the last (and, for signed rows, the first) entry of the
track's ``effect_cdfs`` row.

Deliberately NOT done here: extrapolating a percentile past the data. A generalised
Pareto fit to Enformer's TF nulls gives shape c = −0.190 — a *bounded* tail whose
endpoint (4.245) lies above the empirical maximum (2.956) but below the observed
effect (4.372), so the fitted model calls the actual measurement impossible. Forcing
an exponential tail does extrapolate monotonically, but it reports a modelling
assumption to eight decimal places. A ratio to the sample maximum is a fact about
the sample.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer
from chorus.analysis.variant_report import TrackScore, _fmt_percentile


@pytest.fixture(scope="module")
def norm():
    return PerTrackNormalizer()


def _first_track(oracle: str):
    """(track_id, lo, hi) for an oracle's first row, or None if not downloaded."""
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / f"{oracle}_pertrack.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as d:
        if "effect_cdfs" not in d:
            return None
        ids = [str(x) for x in d["track_ids"]]
        row = d["effect_cdfs"][0]
    return ids[0], float(row[0]), float(row[-1])


# ---------------------------------------------------------------------------
# The invariant tying the two numbers together
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", [
    "chrombpnet", "cherimoya", "enformer", "borzoi", "alphagenome",
    "sei", "legnet", "epinformerseq",
])
def test_exceedance_is_set_exactly_when_the_percentile_is_clamped(oracle, norm):
    """A reported exceedance must never coexist with an unclamped percentile.

    If a mid-range percentile ever came back with an exceedance, or a pinned one
    without, the report would be showing a contradiction — both are read off the
    same CDF row, so crossing the support's edge is precisely what forces the clamp.
    """
    got = _first_track(oracle)
    if got is None:
        pytest.skip(f"no downloaded background for {oracle}")
    track, lo, hi = got
    signed = lo < 0

    # Probe inside, at, and beyond each live end of the support.
    probes = [hi * 0.5, hi, hi * 1.0001, hi * 2.0]
    if signed:
        probes += [lo * 0.5, lo, lo * 1.0001, lo * 2.0]

    for raw in probes:
        value = raw if signed else abs(raw)
        q = norm.effect_percentile(oracle, track, value, signed=signed)
        ex = norm.effect_exceedance(oracle, track, value, signed=signed)
        if q is None:
            continue
        # The two ends are NOT symmetric, and the asymmetry is inherent rather
        # than a bug. Rank is searchsorted(side="right"), i.e. the count of
        # samples <= value. At exactly the maximum that count is n, so the
        # percentile pins. At exactly the minimum it is 1, so the signed
        # percentile is -0.9998 and does NOT pin -- pinning at the low end needs
        # a value strictly below lo. effect_exceedance uses exactly these
        # comparisons, which is why the two agree below.
        pinned = abs(q) >= 1.0
        expect_pinned = value >= hi or (signed and value < lo)
        assert pinned == expect_pinned, (
            f"{oracle}/{track}: raw={value:+.6f} support=[{lo:.6f},{hi:.6f}] "
            f"pctile={q:+.9f} pinned={pinned} but expected {expect_pinned}"
        )

        # An exceedance never appears without a clamp. The converse has exactly
        # one exception -- value == hi -- where the ratio would be 1.0 and so
        # says nothing; the percentile is pinned there and the field stays None.
        if ex is not None:
            assert pinned, (
                f"{oracle}/{track}: exceedance={ex} reported while the percentile "
                f"({q}) is NOT clamped -- the report would contradict itself"
            )
            assert ex > 1.0, f"{oracle}/{track}: ex={ex} must exceed 1.0"
        elif pinned:
            assert value == hi, (
                f"{oracle}/{track}: percentile is clamped at {q} but no exceedance "
                f"was reported, and raw={value} is not exactly the maximum {hi}"
            )


def test_exceedance_orders_effects_the_clamped_percentile_cannot(norm):
    """The point of the field: monotone resolution above the ceiling."""
    got = _first_track("chrombpnet")
    if got is None:
        pytest.skip("no downloaded background for chrombpnet")
    track, _lo, hi = got

    values = [hi * m for m in (1.01, 1.5, 3.0, 10.0)]
    pctiles = [norm.effect_percentile("chrombpnet", track, v) for v in values]
    exceed = [norm.effect_exceedance("chrombpnet", track, v) for v in values]

    assert len(set(pctiles)) == 1 and pctiles[0] == 1.0, (
        f"expected all four to pin at 1.0, got {pctiles} — if this fails the "
        f"premise of the whole module has changed"
    )
    assert exceed == sorted(exceed), f"exceedance not monotone: {exceed}"
    assert len(set(exceed)) == 4, f"exceedance does not separate them: {exceed}"
    assert exceed[-1] == pytest.approx(10.0, rel=1e-6)


def test_signed_layers_resolve_past_both_ends(norm):
    """Sei and LegNet rows are 100% signed; a strong repressive effect crosses ``lo``.

    Defining the ratio against the maximum alone would silently return ``None``
    for every strongly negative effect — the exact tracks where direction is the
    finding. 12.9% of AlphaGenome's rows and 20.3% of Borzoi's are signed too.
    """
    for oracle in ("sei", "legnet"):
        got = _first_track(oracle)
        if got is None:
            continue
        track, lo, hi = got
        assert lo < 0, f"{oracle} row 0 was expected to be signed, got lo={lo}"

        ex_hi = norm.effect_exceedance(oracle, track, hi * 1.25, signed=True)
        ex_lo = norm.effect_exceedance(oracle, track, lo * 1.25, signed=True)
        assert ex_hi == pytest.approx(1.25, rel=1e-6), f"{oracle} upper: {ex_hi}"
        assert ex_lo == pytest.approx(1.25, rel=1e-6), (
            f"{oracle}: a strongly NEGATIVE effect past lo={lo} returned {ex_lo}; "
            f"the ratio must be taken against the end that was crossed"
        )
        assert ex_lo > 0, "the ratio is a magnitude and must not come back negative"


def test_an_unsigned_row_never_reports_a_low_end_exceedance(norm):
    """Unsigned rows hold ``abs`` effects, so ``lo`` is not a live bound."""
    got = _first_track("enformer")
    if got is None:
        pytest.skip("no downloaded background for enformer")
    track, lo, _hi = got
    assert lo >= 0.0, (
        f"enformer row 0 was expected unsigned, got lo={lo}. Unsigned rows hold "
        f"abs() effects so lo is the smallest sampled magnitude -- a small positive "
        f"number, not a structural zero."
    )
    # signed=False is how unsigned layers are queried; abs() is applied upstream.
    assert norm.effect_exceedance("enformer", track, 0.0, signed=False) is None


def test_unknown_track_and_oracle_return_none_rather_than_raising(norm):
    assert norm.effect_null_support("chrombpnet", "NOT:A:TRACK") is None
    assert norm.effect_exceedance("chrombpnet", "NOT:A:TRACK", 99.0) is None
    assert norm.effect_null_support("not_an_oracle", "x") is None
    assert norm.effect_exceedance("not_an_oracle", "x", 99.0) is None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_the_table_shows_the_ratio_beside_the_pinned_bucket():
    """The bucket appears exactly when the percentile is clamped, and not otherwise.

    A clamped percentile carries no ordering information -- 1.11x past the null's
    maximum and 10x past it both saturate at the same end of the same CDF row -- so
    the bucket plus the ratio is the honest rendering.

    An *un*clamped percentile is a real rank, and gets real digits. This is the
    part that changed with exact retention: while the nulls were thinned the top of
    the scale was an artefact of a subsample, so bucketing 0.9995 and 0.9998
    together was correct. Now they order, and the report must say so.
    """
    # Clamped: bucket + ratio, no fabricated decimals.
    assert _fmt_percentile(1.0, 1.109) == "≥99th (1.11× null max)"
    assert _fmt_percentile(-1.0, 1.25) == "≤1st (1.25× null max)"
    # Inside support: four decimals, because two cannot separate the tail.
    assert _fmt_percentile(1.0) == "1.0000"
    assert _fmt_percentile(0.9998) == "0.9998"
    assert _fmt_percentile(0.9995) == "0.9995"
    assert _fmt_percentile(0.9998) != _fmt_percentile(0.9995)
    assert _fmt_percentile(0.005) == "0.0050"
    # Mid-range keeps two decimals -- an exceedance there would be contradictory.
    assert _fmt_percentile(0.5) == "0.50"
    # Suppressed (below the noise floor) stays legible.
    assert _fmt_percentile(None) == "near-zero"
    assert _fmt_percentile(None, 2.0) == "near-zero"


def test_a_signed_layers_negative_half_is_not_the_bottom_percentile():
    """A signed percentile near -0.8 is a strong DOWN effect, not "≤1st".

    Signed layers span [-1, 1]: the sign is direction, the magnitude is how
    unusual. The old rule tested ``q <= 0.01``, which is true for the entire
    negative half, so the C/EBP vignette rendered nine ``gene_expression`` rows
    as "≤1st" whose real percentiles were -0.7374 to -0.9634 -- moderately to
    strongly down-regulated. Beside a "≥99th" three rows above, that reads as a
    variant which both strongly represses and is indistinguishable from noise.

    Unsigned layers keep both ends as tails, because for them ``q`` near zero
    really is the bottom of the scale. The same number needs opposite treatment
    depending on the layer, which is why the layer has to be passed.
    """
    # Signed: mid-magnitude negatives are body values, shown with two decimals.
    for q in (-0.7374, -0.8410, -0.8700, -0.9054, -0.9634):
        out = _fmt_percentile(q, layer="gene_expression")
        assert out == f"{q:.2f}", f"signed {q} rendered as {out!r}"
        assert "1st" not in out, f"signed {q} still reads as a bottom-percentile bucket"
    # Signed: only |q| >= 0.99 is a tail, and gets four decimals.
    assert _fmt_percentile(-0.9950, layer="gene_expression") == "-0.9950"
    assert _fmt_percentile(0.9950, layer="gene_expression") == "0.9950"
    # Unsigned: the low end IS a tail.
    assert _fmt_percentile(0.0050, layer="chromatin_accessibility") == "0.0050"
    assert _fmt_percentile(0.9998, layer="chromatin_accessibility") == "0.9998"
    # Unsigned mid-range keeps two decimals.
    assert _fmt_percentile(0.42, layer="chromatin_accessibility") == "0.42"


def test_an_unknown_or_absent_layer_falls_back_to_the_unsigned_reading():
    """Most layers are unsigned, so that is the safer default.

    Asserted rather than left implicit: a caller that forgets the argument should
    degrade to imprecision, never to the sign confusion above.
    """
    assert _fmt_percentile(0.0050) == "0.0050"
    assert _fmt_percentile(0.0050, layer="no_such_layer") == "0.0050"
    assert _fmt_percentile(0.9998, layer=None) == "0.9998"


def test_a_clamped_signed_row_still_buckets_by_sign():
    """Past the ceiling, direction is all that is left to report."""
    assert _fmt_percentile(1.0, 1.5, layer="gene_expression") == "≥99th (1.50× null max)"
    assert _fmt_percentile(-1.0, 1.5, layer="gene_expression") == "≤1st (1.50× null max)"


def test_the_cebp_vignette_ordering_is_visible_in_the_rendered_string():
    """The regression this change exists to prevent, in the numbers that caused it.

    CEBPA outranks CEBPB at rs12740374 on a *smaller* raw effect, because each is
    ranked against its own track's null. Both previously rendered as a bare
    "≥99th", so the report contradicted the walkthrough README explaining the
    contrast.
    """
    cebpa, cebpb, cebpg = 0.9998, 0.9995, 0.9997
    shown = [_fmt_percentile(q) for q in (cebpa, cebpb, cebpg)]
    assert len(set(shown)) == 3, f"three distinct percentiles still render alike: {shown}"
    assert shown[0] > shown[1], "CEBPA must read above CEBPB as displayed text"


def test_the_field_survives_a_json_round_trip():
    """A field dropped by ``to_dict``/``from_dict`` is a field the reports lose.

    ``append_tracks`` forwarding only its canonical keys is the same failure this
    repo already paid for twice — most recently ``layers_per_row``, which every
    shard wrote, ``union_shards`` silently dropped, and the guard test SKIPPED on
    absence so the suite stayed green.
    """
    ts = TrackScore(
        assay_id="CHIP:HepG2:CEBPA:+", assay_type="CHIP", cell_type="HepG2",
        layer="tf_binding", ref_value=1.0, alt_value=2.0, raw_score=1.865,
        quantile_score=1.0, effect_exceedance=1.109,
    )
    d = ts.to_dict()
    assert d["effect_exceedance"] == 1.109, f"dropped by to_dict: {sorted(d)}"

    from chorus.analysis.variant_report import VariantReport
    report = VariantReport(
        chrom="chr1", position=109274968, ref_allele="G", alt_alleles=["T"],
        oracle_name="chrombpnet", gene_name="SORT1", allele_scores={"T": [ts]},
    )
    back = VariantReport.from_dict(report.to_dict())
    assert back.allele_scores["T"][0].effect_exceedance == 1.109, (
        "effect_exceedance did not survive to_dict -> from_dict"
    )


def test_none_is_omitted_so_existing_artefacts_do_not_all_churn():
    ts = TrackScore(
        assay_id="x", assay_type="ATAC", cell_type="K562", layer="chromatin",
        ref_value=1.0, alt_value=1.1, raw_score=0.1, quantile_score=0.5,
    )
    assert "effect_exceedance" not in ts.to_dict()


def test_the_dataframe_carries_the_column():
    """Programmatic consumers read the TSV, not the markdown."""
    from chorus.analysis.variant_report import VariantReport

    ts = TrackScore(
        assay_id="CHIP:HepG2:CEBPA:+", assay_type="CHIP", cell_type="HepG2",
        layer="tf_binding", ref_value=1.0, alt_value=2.0, raw_score=1.865,
        quantile_score=1.0, effect_exceedance=1.109,
    )
    df = VariantReport(
        chrom="chr1", position=1, ref_allele="G", alt_alleles=["T"],
        oracle_name="chrombpnet", gene_name=None, allele_scores={"T": [ts]},
    ).to_dataframe()
    assert "effect_exceedance" in df.columns
    assert float(df["effect_exceedance"].iloc[0]) == pytest.approx(1.109)
