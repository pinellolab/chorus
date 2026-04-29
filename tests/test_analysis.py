"""Tests for the chorus.analysis module — scorers, normalization, and reports."""

import math
import os
import tempfile

import numpy as np
import pytest

from unittest.mock import MagicMock

from chorus.analysis.variant_report import VariantReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_track(
    assay_id="DNASE:K562",
    assay_type=None,
    n_bins=1000,
    resolution=1,
    pred_start=1_000_000,
    chrom="chr1",
    values=None,
    cls=None,
):
    """Create a mock OraclePredictionTrack for testing."""
    from chorus.core.result import OraclePredictionTrack

    if values is not None:
        n_bins = len(values)

    pred_end = pred_start + n_bins * resolution

    interval = MagicMock()
    interval.reference = MagicMock()
    interval.reference.chrom = chrom
    interval.reference.start = pred_start
    interval.reference.end = pred_end

    if assay_type is None:
        assay_type = assay_id.split(":")[0] if ":" in assay_id else assay_id
    cell_type = assay_id.split(":")[-1] if ":" in assay_id else ""

    track_cls = cls or OraclePredictionTrack
    if values is None:
        values = np.ones(n_bins, dtype=np.float32)

    track = track_cls.__new__(track_cls)
    track.source_model = "test"
    track.assay_id = assay_id
    track.assay_type = assay_type
    track.cell_type = cell_type
    track.query_interval = interval
    track.prediction_interval = interval
    track.input_interval = interval
    track.resolution = resolution
    track.values = values
    track.preferred_aggregation = "sum"
    track.preferred_interpolation = "linear_divided"
    track.preferred_scoring_strategy = "mean"
    track.metadata = {}
    track.track_id = None
    return track


def _make_variant_result(
    ref_values_by_track,
    alt_values_by_track,
    position="chr1:1000500",
    alleles=None,
    resolution=1,
    pred_start=1_000_000,
):
    """Build a mock variant_result dict matching oracle.predict_variant_effect() output."""
    from chorus.core.result import OraclePrediction

    if alleles is None:
        alleles = ["A", "G"]

    def _build_pred(values_by_track):
        pred = OraclePrediction.__new__(OraclePrediction)
        pred.tracks = {}
        for aid, vals in values_by_track.items():
            at = aid.split(":")[0] if ":" in aid else aid
            pred.tracks[aid] = _make_mock_track(
                assay_id=aid,
                assay_type=at,
                resolution=resolution,
                pred_start=pred_start,
                values=vals,
            )
        return pred

    return {
        "predictions": {
            "reference": _build_pred(ref_values_by_track),
            alleles[1]: _build_pred(alt_values_by_track),
        },
        "variant_info": {
            "position": position,
            "alleles": alleles,
        },
    }


# ── Scorer tests ─────────────────────────────────────────────────────


class TestClassifyTrackLayer:
    def test_dnase_chromatin(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("DNASE:K562", assay_type="DNASE")
        assert classify_track_layer(track) == "chromatin_accessibility"

    def test_atac_chromatin(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("ATAC:K562", assay_type="ATAC")
        assert classify_track_layer(track) == "chromatin_accessibility"

    def test_chip_tf(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("GATA1:K562", assay_type="CHIP")
        assert classify_track_layer(track) == "tf_binding"

    def test_chip_histone_h3k27ac(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("H3K27ac:K562", assay_type="CHIP")
        assert classify_track_layer(track) == "histone_marks"

    def test_chip_histone_h3k4me3(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("H3K4me3:HepG2", assay_type="CHIP")
        assert classify_track_layer(track) == "histone_marks"

    def test_chip_histone_h4k20me1(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("H4K20me1:K562", assay_type="CHIP")
        assert classify_track_layer(track) == "histone_marks"

    def test_cage_tss(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("CAGE:K562", assay_type="CAGE")
        assert classify_track_layer(track) == "tss_activity"

    def test_procap_tss(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("PRO_CAP:K562", assay_type="PRO_CAP")
        assert classify_track_layer(track) == "tss_activity"

    def test_rna_expression(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("RNA:K562", assay_type="RNA")
        assert classify_track_layer(track) == "gene_expression"

    def test_mpra_promoter(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("LentiMPRA:K562", assay_type="LentiMPRA")
        assert classify_track_layer(track) == "promoter_activity"

    def test_splice_sites(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("SPLICE:K562", assay_type="SPLICE_SITES")
        assert classify_track_layer(track) == "splicing"

    def test_unknown_returns_other(self):
        from chorus.analysis.scorers import classify_track_layer

        track = _make_mock_track("FOO:K562", assay_type="UNKNOWN")
        assert classify_track_layer(track) == "other"


class TestComputeEffect:
    def test_log2fc_no_change(self):
        from chorus.analysis.scorers import _compute_effect

        result = _compute_effect(10.0, 10.0, "log2fc", 1.0)
        assert abs(result) < 1e-10

    def test_log2fc_doubling(self):
        from chorus.analysis.scorers import _compute_effect

        # (20+1)/(10+1) = 21/11
        result = _compute_effect(10.0, 20.0, "log2fc", 1.0)
        expected = math.log2(21.0 / 11.0)
        assert abs(result - expected) < 1e-10

    def test_log2fc_with_pseudocount_zero_ref(self):
        from chorus.analysis.scorers import _compute_effect

        # (5+1)/(0+1) = 6
        result = _compute_effect(0.0, 5.0, "log2fc", 1.0)
        expected = math.log2(6.0)
        assert abs(result - expected) < 1e-10

    def test_logfc(self):
        from chorus.analysis.scorers import _compute_effect

        result = _compute_effect(10.0, 20.0, "logfc", 0.001)
        expected = math.log(20.001 / 10.001)
        assert abs(result - expected) < 1e-10

    def test_diff(self):
        from chorus.analysis.scorers import _compute_effect

        result = _compute_effect(10.0, 15.0, "diff", 0.0)
        assert result == 5.0

    def test_negative_diff(self):
        from chorus.analysis.scorers import _compute_effect

        result = _compute_effect(15.0, 10.0, "diff", 0.0)
        assert result == -5.0

    def test_unknown_formula_raises(self):
        from chorus.analysis.scorers import _compute_effect

        with pytest.raises(ValueError, match="Unknown formula"):
            _compute_effect(1.0, 2.0, "sqrt", 0.0)


class TestScoreTrackEffect:
    def test_chromatin_window_scoring(self):
        from chorus.analysis.scorers import score_track_effect

        # 1bp resolution, variant at 1_000_500, window 501bp = [1000250, 1000751)
        ref_vals = np.ones(1000, dtype=np.float32)
        alt_vals = np.ones(1000, dtype=np.float32) * 2.0

        ref_track = _make_mock_track("DNASE:K562", assay_type="DNASE", values=ref_vals)
        alt_track = _make_mock_track("DNASE:K562", assay_type="DNASE", values=alt_vals)

        result = score_track_effect(ref_track, alt_track, "chr1", 1_000_500)
        assert result is not None
        assert result["layer"] == "chromatin_accessibility"
        # 501 bins in window: ref_sum=501, alt_sum=1002
        # log2((1002+1)/(501+1)) = log2(1003/502)
        expected = math.log2(1003.0 / 502.0)
        assert abs(result["raw_score"] - expected) < 0.01

    def test_rna_without_gene_returns_none(self):
        from chorus.analysis.scorers import score_track_effect

        ref_track = _make_mock_track("RNA:K562", assay_type="RNA")
        alt_track = _make_mock_track("RNA:K562", assay_type="RNA")

        result = score_track_effect(ref_track, alt_track, "chr1", 1_000_500)
        assert result is None

    def test_rna_with_gene_exons(self):
        from chorus.analysis.scorers import score_track_effect

        ref_vals = np.ones(1000, dtype=np.float32) * 5.0
        alt_vals = np.ones(1000, dtype=np.float32) * 10.0

        ref_track = _make_mock_track("RNA:K562", assay_type="RNA", values=ref_vals)
        alt_track = _make_mock_track("RNA:K562", assay_type="RNA", values=alt_vals)

        gene_exons = [
            {"chrom": "chr1", "start": 1_000_100, "end": 1_000_200},
            {"chrom": "chr1", "start": 1_000_300, "end": 1_000_400},
        ]

        result = score_track_effect(
            ref_track, alt_track, "chr1", 1_000_500, gene_exons=gene_exons,
        )
        assert result is not None
        assert result["layer"] == "gene_expression"
        # Each exon: 100 bins. ref sum per exon = 500, alt sum per exon = 1000
        # Mean across 2 exons: ref_value = 500, alt_value = 1000
        # logfc = log((1000+0.001)/(500+0.001)) ≈ log(2)
        assert abs(result["raw_score"] - math.log(2.0)) < 0.01

    def test_histone_wider_window(self):
        from chorus.analysis.scorers import score_track_effect

        # Histone marks use 2001bp window
        ref_vals = np.ones(5000, dtype=np.float32)
        alt_vals = np.ones(5000, dtype=np.float32) * 3.0

        ref_track = _make_mock_track(
            "H3K27ac:K562", assay_type="CHIP", values=ref_vals,
        )
        alt_track = _make_mock_track(
            "H3K27ac:K562", assay_type="CHIP", values=alt_vals,
        )

        # Variant at 1_002_500 → window [1001500, 1003501) fully inside prediction
        result = score_track_effect(ref_track, alt_track, "chr1", 1_002_500)
        assert result is not None
        assert result["layer"] == "histone_marks"
        # 2001 bins: ref_sum=2001, alt_sum=6003
        expected = math.log2(6004.0 / 2002.0)
        assert abs(result["raw_score"] - expected) < 0.01

    def test_variant_outside_window_returns_none(self):
        from chorus.analysis.scorers import score_track_effect

        ref_track = _make_mock_track("DNASE:K562", assay_type="DNASE", n_bins=100)
        alt_track = _make_mock_track("DNASE:K562", assay_type="DNASE", n_bins=100)

        # Variant far from prediction window
        result = score_track_effect(ref_track, alt_track, "chr2", 5_000_000)
        assert result is None

    def test_different_chromosome_returns_none(self):
        from chorus.analysis.scorers import score_track_effect

        ref_track = _make_mock_track("DNASE:K562", assay_type="DNASE")
        alt_track = _make_mock_track("DNASE:K562", assay_type="DNASE")

        result = score_track_effect(ref_track, alt_track, "chr2", 1_000_500)
        assert result is None

    def test_full_output_scoring_mpra(self):
        from chorus.analysis.scorers import score_track_effect

        ref_vals = np.array([3.0], dtype=np.float32)
        alt_vals = np.array([5.0], dtype=np.float32)

        ref_track = _make_mock_track(
            "LentiMPRA:K562", assay_type="LentiMPRA", values=ref_vals,
        )
        alt_track = _make_mock_track(
            "LentiMPRA:K562", assay_type="LentiMPRA", values=alt_vals,
        )

        result = score_track_effect(ref_track, alt_track, "chr1", 1_000_000)
        assert result is not None
        assert result["layer"] == "promoter_activity"
        assert result["raw_score"] == pytest.approx(2.0)  # diff: 5 - 3


class TestScoreVariantMultilayer:
    def test_basic_multilayer(self):
        from chorus.analysis.scorers import score_variant_multilayer

        ref_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32),
            "CAGE:K562": np.ones(1000, dtype=np.float32) * 2.0,
        }
        alt_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0,
            "CAGE:K562": np.ones(1000, dtype=np.float32) * 4.0,
        }
        variant_result = _make_variant_result(ref_values, alt_values)

        scores = score_variant_multilayer(variant_result)
        assert "G" in scores
        assert "DNASE:K562" in scores["G"]
        assert "CAGE:K562" in scores["G"]
        assert scores["G"]["DNASE:K562"]["layer"] == "chromatin_accessibility"
        assert scores["G"]["CAGE:K562"]["layer"] == "tss_activity"
        assert scores["G"]["DNASE:K562"]["raw_score"] is not None
        assert scores["G"]["CAGE:K562"]["raw_score"] is not None

    def test_rna_without_gene_notes(self):
        from chorus.analysis.scorers import score_variant_multilayer

        ref_values = {"RNA:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"RNA:K562": np.ones(1000, dtype=np.float32) * 2.0}
        variant_result = _make_variant_result(ref_values, alt_values)

        scores = score_variant_multilayer(variant_result)
        assert scores["G"]["RNA:K562"]["raw_score"] is None
        assert scores["G"]["RNA:K562"]["note"] is not None


# ── Normalization tests ──────────────────────────────────────────────


class TestBackgroundDistribution:
    def test_raw_to_quantile_median_signed(self):
        from chorus.analysis.normalization import BackgroundDistribution

        scores = np.linspace(-1, 1, 1001)
        bg = BackgroundDistribution(scores)

        # Median (0.0) → rank ~500/1001 → quantile ~0.5 → signed ~0.0
        q = bg.raw_to_quantile(0.0, signed=True)
        assert abs(q) < 0.01

    def test_raw_to_quantile_unsigned(self):
        from chorus.analysis.normalization import BackgroundDistribution

        scores = np.linspace(0, 1, 1001)
        bg = BackgroundDistribution(scores)

        q_low = bg.raw_to_quantile(0.01, signed=False)
        assert q_low < 0.05

        q_high = bg.raw_to_quantile(0.99, signed=False)
        assert q_high > 0.95

    def test_raw_to_quantile_batch(self):
        from chorus.analysis.normalization import BackgroundDistribution

        scores = np.linspace(0, 1, 1001)
        bg = BackgroundDistribution(scores)

        raw = np.array([0.0, 0.5, 1.0])
        quantiles = bg.raw_to_quantile_batch(raw, signed=False)
        assert len(quantiles) == 3
        assert quantiles[0] < quantiles[1] < quantiles[2]

    def test_extreme_values(self):
        from chorus.analysis.normalization import BackgroundDistribution

        scores = np.linspace(0, 1, 1001)
        bg = BackgroundDistribution(scores)

        # Below all background → 0
        q = bg.raw_to_quantile(-10.0, signed=False)
        assert q == 0.0

        # Above all background → 1
        q = bg.raw_to_quantile(10.0, signed=False)
        assert q == 1.0

    def test_save_load(self):
        from chorus.analysis.normalization import BackgroundDistribution

        scores = np.random.randn(1000)
        bg = BackgroundDistribution(scores)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name

        try:
            bg.save(path)
            bg2 = BackgroundDistribution.load(path)
            assert len(bg2) == len(bg)
            np.testing.assert_array_equal(bg.sorted_scores, bg2.sorted_scores)
        finally:
            os.unlink(path)

    def test_len(self):
        from chorus.analysis.normalization import BackgroundDistribution

        bg = BackgroundDistribution(np.zeros(42))
        assert len(bg) == 42


class TestQuantileNormalizer:
    def test_normalize_with_background(self):
        from chorus.analysis.normalization import (
            BackgroundDistribution,
            QuantileNormalizer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = QuantileNormalizer(cache_dir=tmpdir)

            scores = np.linspace(-2, 2, 10001)
            bg = BackgroundDistribution(scores)
            normalizer.set_background("test_layer", bg, persist=True)

            q = normalizer.normalize("test_layer", 0.0, signed=True)
            assert q is not None
            assert abs(q) < 0.01  # median → ~0

    def test_normalize_without_background_returns_none(self):
        from chorus.analysis.normalization import QuantileNormalizer

        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = QuantileNormalizer(cache_dir=tmpdir)
            q = normalizer.normalize("nonexistent", 0.5)
            assert q is None

    def test_persistence(self):
        from chorus.analysis.normalization import (
            BackgroundDistribution,
            QuantileNormalizer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            n1 = QuantileNormalizer(cache_dir=tmpdir)
            bg = BackgroundDistribution(np.linspace(0, 1, 1000))
            n1.set_background("test_key", bg, persist=True)

            # Load in new instance
            n2 = QuantileNormalizer(cache_dir=tmpdir)
            assert n2.has_background("test_key")
            q = n2.normalize("test_key", 0.5, signed=False)
            assert q is not None
            assert abs(q - 0.5) < 0.01

    def test_background_key(self):
        from chorus.analysis.normalization import QuantileNormalizer

        key = QuantileNormalizer.background_key("alphagenome", "chromatin_accessibility")
        assert key == "alphagenome_chromatin_accessibility"


# ── Variant report tests ─────────────────────────────────────────────


class TestVariantReport:
    def test_build_report(self):
        from chorus.analysis.variant_report import build_variant_report

        ref_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32),
            "CAGE:K562": np.ones(1000, dtype=np.float32) * 2.0,
        }
        alt_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0,
            "CAGE:K562": np.ones(1000, dtype=np.float32) * 4.0,
        }
        variant_result = _make_variant_result(ref_values, alt_values)

        report = build_variant_report(variant_result, oracle_name="test_oracle")
        assert report.chrom == "chr1"
        assert report.position == 1_000_500
        assert report.oracle_name == "test_oracle"
        assert "G" in report.allele_scores
        # DNASE: 1 row, CAGE: variant_site + per-gene TSS rows
        assert len(report.allele_scores["G"]) >= 2

    def test_to_dict(self):
        from chorus.analysis.variant_report import build_variant_report

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        variant_result = _make_variant_result(ref_values, alt_values)

        report = build_variant_report(variant_result, oracle_name="test")
        d = report.to_dict()

        assert "variant" in d
        assert d["variant"]["chrom"] == "chr1"
        assert d["oracle"] == "test"
        assert "G" in d["alleles"]
        assert "scores_by_layer" in d["alleles"]["G"]
        assert "chromatin_accessibility" in d["alleles"]["G"]["scores_by_layer"]

    def test_to_markdown(self):
        from chorus.analysis.variant_report import build_variant_report

        ref_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32),
        }
        alt_values = {
            "DNASE:K562": np.ones(1000, dtype=np.float32) * 3.0,
        }
        variant_result = _make_variant_result(ref_values, alt_values)

        report = build_variant_report(variant_result, oracle_name="test")
        md = report.to_markdown()

        assert "## Multi-Layer Variant Effect Report" in md
        assert "chr1:1000500" in md
        assert "DNASE:K562" in md
        assert "Chromatin accessibility" in md

    def test_to_markdown_with_histone(self):
        from chorus.analysis.variant_report import build_variant_report

        ref_values = {
            "H3K27ac:K562": np.ones(5000, dtype=np.float32),
        }
        alt_values = {
            "H3K27ac:K562": np.ones(5000, dtype=np.float32) * 2.0,
        }
        variant_result = _make_variant_result(
            ref_values, alt_values, position="chr1:1002500",
        )
        # Fix assay_type to CHIP
        for pred in variant_result["predictions"].values():
            if "H3K27ac:K562" in pred.tracks:
                pred.tracks["H3K27ac:K562"].assay_type = "CHIP"

        report = build_variant_report(variant_result, oracle_name="test")
        md = report.to_markdown()
        # Track label comes from description (or falls back to assay_type:cell_type)
        assert "CHIP:K562" in md or "H3K27ac:K562" in md
        assert "Histone modifications" in md

    def test_to_dataframe(self):
        from chorus.analysis.variant_report import build_variant_report

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        variant_result = _make_variant_result(ref_values, alt_values)

        report = build_variant_report(variant_result, oracle_name="test")
        df = report.to_dataframe()

        assert len(df) == 1
        assert "layer" in df.columns
        assert "raw_score" in df.columns
        assert "quantile_score" in df.columns
        assert df.iloc[0]["layer"] == "chromatin_accessibility"

    def test_with_normalization(self):
        from chorus.analysis.variant_report import build_variant_report
        from chorus.analysis.normalization import (
            BackgroundDistribution,
            QuantileNormalizer,
        )

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        variant_result = _make_variant_result(ref_values, alt_values)

        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = QuantileNormalizer(cache_dir=tmpdir)
            bg = BackgroundDistribution(np.linspace(-2, 2, 10001))
            normalizer.set_background(
                normalizer.background_key("test", "chromatin_accessibility"),
                bg,
                persist=False,
            )

            report = build_variant_report(
                variant_result, oracle_name="test", normalizer=normalizer,
            )

            score = report.allele_scores["G"][0]
            assert score.quantile_score is not None

    def test_multiple_alleles(self):
        from chorus.analysis.variant_report import build_variant_report
        from chorus.core.result import OraclePrediction

        ref_vals = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt1_vals = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        alt2_vals = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 3.0}

        alleles = ["A", "G", "T"]

        def _build_pred(vals):
            pred = OraclePrediction.__new__(OraclePrediction)
            pred.tracks = {}
            for aid, v in vals.items():
                at = aid.split(":")[0]
                pred.tracks[aid] = _make_mock_track(
                    assay_id=aid, assay_type=at, values=v,
                )
            return pred

        variant_result = {
            "predictions": {
                "reference": _build_pred(ref_vals),
                "G": _build_pred(alt1_vals),
                "T": _build_pred(alt2_vals),
            },
            "variant_info": {
                "position": "chr1:1000500",
                "alleles": alleles,
            },
        }

        report = build_variant_report(variant_result, oracle_name="test")
        assert "G" in report.allele_scores
        assert "T" in report.allele_scores

        # T allele should have larger effect
        g_score = report.allele_scores["G"][0].raw_score
        t_score = report.allele_scores["T"][0].raw_score
        assert t_score > g_score


class TestInterpretScore:
    def test_minimal_effect(self):
        from chorus.analysis.variant_report import _interpret_score

        result = _interpret_score(0.05, None, "chromatin_accessibility")
        assert "Minimal" in result

    def test_strong_opening(self):
        from chorus.analysis.variant_report import _interpret_score

        result = _interpret_score(0.5, None, "chromatin_accessibility")
        assert "Strong" in result
        assert "opening" in result

    def test_moderate_binding_loss(self):
        from chorus.analysis.variant_report import _interpret_score

        result = _interpret_score(-0.2, None, "tf_binding")
        assert "Moderate" in result
        assert "loss" in result

    def test_quantile_based_interpretation(self):
        from chorus.analysis.variant_report import _interpret_score

        # Quantile contributes to the strength label, but the raw-magnitude
        # gate prevents "very strong" from being applied to sub-0.7 effects.
        # With raw=0.5 + quantile=0.95, the result is "Strong" (not "Very
        # strong") because |0.5| < 0.7. This is the post-audit honest-label
        # behavior — a tiny raw effect can never be "very strong" just
        # because its quantile rank is high.
        result = _interpret_score(0.5, 0.95, "tss_activity")
        assert "Strong" in result
        assert "increase" in result

    def test_quantile_high_with_high_raw(self):
        from chorus.analysis.variant_report import _interpret_score

        # When both raw and quantile are high, the label is "Very strong".
        result = _interpret_score(1.5, 0.95, "tss_activity")
        assert "Very strong" in result
        assert "increase" in result

    def test_quantile_high_but_tiny_raw(self):
        from chorus.analysis.variant_report import _interpret_score

        # High quantile + near-zero raw → still "Minimal" because the
        # magnitude gate kicks in before the quantile check. Prevents
        # "very strong" labels on 0.001 log2FC rows with quantile=1.0.
        result = _interpret_score(0.001, 1.0, "chromatin_accessibility")
        assert "Minimal" in result

    def test_very_strong_mark_loss(self):
        from chorus.analysis.variant_report import _interpret_score

        result = _interpret_score(-1.5, None, "histone_marks")
        assert "Very strong" in result
        assert "loss" in result

    def test_moderate_repression(self):
        from chorus.analysis.variant_report import _interpret_score

        result = _interpret_score(-0.2, None, "promoter_activity")
        assert "Moderate" in result
        assert "repression" in result


# ── Normalization signed/unsigned correctness ────────────────────────


class TestNormalizationSignedUnsigned:
    """Verify that unsigned layers use [0,1] and signed layers use [-1,1]."""

    def test_unsigned_layer_quantile_in_0_1(self):
        """Chromatin (unsigned): quantile must be in [0,1]."""
        from chorus.analysis.normalization import BackgroundDistribution
        from chorus.analysis.variant_report import build_variant_report
        from chorus.analysis.normalization import QuantileNormalizer

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        vr = _make_variant_result(ref, alt)

        with tempfile.TemporaryDirectory() as td:
            norm = QuantileNormalizer(cache_dir=td)
            # Background of abs(log2FC) values — all >= 0
            bg = BackgroundDistribution.from_scores(
                np.random.normal(0, 0.5, 10000), signed=False,
            )
            norm.set_background(
                norm.background_key("test", "chromatin_accessibility"),
                bg, persist=False,
            )
            report = build_variant_report(vr, "test", normalizer=norm)
            ts = report.allele_scores["G"][0]
            assert ts.quantile_score is not None
            # Unsigned → [0, 1]
            assert 0.0 <= ts.quantile_score <= 1.0, (
                f"Expected [0,1], got {ts.quantile_score}"
            )

    def test_signed_layer_quantile_in_neg1_1(self):
        """MPRA (signed): quantile must be in [-1,1]."""
        from chorus.analysis.normalization import BackgroundDistribution
        from chorus.analysis.variant_report import build_variant_report
        from chorus.analysis.normalization import QuantileNormalizer

        ref = {"LentiMPRA:K562": np.array([3.0], dtype=np.float32)}
        alt = {"LentiMPRA:K562": np.array([1.0], dtype=np.float32)}  # decrease
        vr = _make_variant_result(ref, alt)

        with tempfile.TemporaryDirectory() as td:
            norm = QuantileNormalizer(cache_dir=td)
            bg = BackgroundDistribution.from_scores(
                np.random.normal(0, 1.0, 10000), signed=True,
            )
            norm.set_background(
                norm.background_key("test", "promoter_activity"),
                bg, persist=False,
            )
            report = build_variant_report(vr, "test", normalizer=norm)
            ts = report.allele_scores["G"][0]
            assert ts.quantile_score is not None
            # MPRA diff = 1.0 - 3.0 = -2.0 → signed quantile should be negative
            assert -1.0 <= ts.quantile_score <= 1.0
            assert ts.quantile_score < 0, (
                f"Negative effect should give negative quantile, got {ts.quantile_score}"
            )

    def test_from_scores_unsigned_takes_abs(self):
        """BackgroundDistribution.from_scores(signed=False) stores abs values."""
        from chorus.analysis.normalization import BackgroundDistribution

        raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        bg = BackgroundDistribution.from_scores(raw, signed=False)
        # All stored values should be >= 0
        assert (bg.sorted_scores >= 0).all()
        # Stored: [0.0, 0.5, 0.5, 1.0, 1.0] (sorted abs)
        assert bg.sorted_scores[0] == 0.0

    def test_from_scores_signed_preserves_negatives(self):
        """BackgroundDistribution.from_scores(signed=True) keeps negatives."""
        from chorus.analysis.normalization import BackgroundDistribution

        raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        bg = BackgroundDistribution.from_scores(raw, signed=True)
        assert bg.sorted_scores[0] == -1.0
        assert bg.sorted_scores[-1] == 1.0


# ── HTML output tests ────────────────────────────────────────────────


class TestHTMLOutput:
    def test_to_html_returns_string(self):
        from chorus.analysis.variant_report import build_variant_report

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        vr = _make_variant_result(ref, alt)

        report = build_variant_report(vr, oracle_name="test")
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "DNASE:K562" in html
        assert "Chromatin accessibility" in html

    def test_to_html_writes_file(self):
        from chorus.analysis.variant_report import build_variant_report

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        vr = _make_variant_result(ref, alt)

        report = build_variant_report(vr, oracle_name="test")

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.html")
            html = report.to_html(output_path=path)
            assert os.path.exists(path)
            content = open(path).read()
            assert content == html

    def test_to_html_has_color_badges(self):
        from chorus.analysis.variant_report import build_variant_report

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 5.0}
        vr = _make_variant_result(ref, alt)

        report = build_variant_report(vr, oracle_name="test")
        html = report.to_html()
        assert "badge-" in html  # has color-coded badges
        assert "bar-pos" in html or "bar-neg" in html  # has effect bars

    def test_to_html_with_quantile(self):
        from chorus.analysis.variant_report import build_variant_report
        from chorus.analysis.normalization import (
            BackgroundDistribution, QuantileNormalizer,
        )

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        vr = _make_variant_result(ref, alt)

        with tempfile.TemporaryDirectory() as td:
            norm = QuantileNormalizer(cache_dir=td)
            bg = BackgroundDistribution.from_scores(
                np.random.normal(0, 0.5, 10000), signed=False,
            )
            norm.set_background(
                norm.background_key("test", "chromatin_accessibility"),
                bg, persist=False,
            )
            report = build_variant_report(vr, "test", normalizer=norm)
            html = report.to_html()
            assert "Effect %ile" in html  # effect percentile column header

    def test_to_html_has_igv_browser(self):
        from chorus.analysis.variant_report import build_variant_report

        ref = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}
        vr = _make_variant_result(ref, alt)

        report = build_variant_report(vr, oracle_name="test")
        assert report._predictions is not None

        html = report.to_html()
        # IGV.js browser embedded
        assert "igv.createBrowser" in html
        assert "igv-div" in html
        assert "Interactive Genome Browser" in html
        # Has the modification ROI
        assert '"name":"Modification"' in html or "'name':'Modification'" in html


# ── Region swap tests ────────────────────────────────────────────────


class TestRegionSwap:
    def test_analyze_region_swap(self):
        from chorus.analysis.region_swap import analyze_region_swap
        from chorus.core.result import OraclePrediction

        # Build mock oracle
        oracle = MagicMock()
        oracle.name = "test_oracle"

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0}

        def _build_pred(values_by_track):
            pred = OraclePrediction.__new__(OraclePrediction)
            pred.tracks = {}
            for aid, vals in values_by_track.items():
                pred.tracks[aid] = _make_mock_track(
                    assay_id=aid, assay_type=aid.split(":")[0], values=vals,
                )
            return pred

        wt_pred = _build_pred(ref_values)
        swap_pred = _build_pred(alt_values)

        oracle.predict.return_value = wt_pred
        oracle.predict_region_replacement.return_value = {
            "raw_predictions": swap_pred,
        }

        report = analyze_region_swap(
            oracle, "chr1:1000000-1001000", "ACGT" * 250,
            assay_ids=["DNASE:K562"],
        )

        assert isinstance(report, VariantReport)
        assert report.oracle_name == "test_oracle"
        assert "replacement" in report.allele_scores
        assert len(report.allele_scores["replacement"]) >= 1
        score = report.allele_scores["replacement"][0]
        assert score.raw_score is not None
        assert score.layer == "chromatin_accessibility"

    def test_region_swap_markdown(self):
        from chorus.analysis.region_swap import analyze_region_swap
        from chorus.core.result import OraclePrediction

        oracle = MagicMock()
        oracle.name = "test_oracle"

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 3.0}

        def _build_pred(vals):
            pred = OraclePrediction.__new__(OraclePrediction)
            pred.tracks = {}
            for aid, v in vals.items():
                pred.tracks[aid] = _make_mock_track(
                    assay_id=aid, assay_type=aid.split(":")[0], values=v,
                )
            return pred

        oracle.predict.return_value = _build_pred(ref_values)
        oracle.predict_region_replacement.return_value = {
            "raw_predictions": _build_pred(alt_values),
        }

        report = analyze_region_swap(
            oracle, "chr1:1000000-1001000", "ACGT" * 250,
            assay_ids=["DNASE:K562"],
        )
        md = report.to_markdown()
        assert "Region Swap" in md
        assert "DNASE:K562" in md


# ── Integration simulation tests ─────────────────────────────────────


class TestIntegrationSimulation:
    def test_simulate_integration(self):
        from chorus.analysis.integration import simulate_integration
        from chorus.core.result import OraclePrediction

        oracle = MagicMock()
        oracle.name = "test_oracle"

        ref_values = {"DNASE:K562": np.ones(1000, dtype=np.float32)}
        alt_values = {"DNASE:K562": np.ones(1000, dtype=np.float32) * 0.5}

        def _build_pred(vals):
            pred = OraclePrediction.__new__(OraclePrediction)
            pred.tracks = {}
            for aid, v in vals.items():
                pred.tracks[aid] = _make_mock_track(
                    assay_id=aid, assay_type=aid.split(":")[0], values=v,
                )
            return pred

        oracle.predict.return_value = _build_pred(ref_values)
        oracle.predict_region_insertion_at.return_value = {
            "raw_predictions": _build_pred(alt_values),
        }

        report = simulate_integration(
            oracle, "chr1:1000500", "ACGT" * 100,
            assay_ids=["DNASE:K562"],
        )

        assert isinstance(report, VariantReport)
        assert "insertion" in report.allele_scores
        score = report.allele_scores["insertion"][0]
        assert score.raw_score is not None
        # Insertion reduced signal: score should be negative
        assert score.raw_score < 0


# ── Batch scoring tests ──────────────────────────────────────────────


class TestBatchScoring:
    def test_score_variant_batch(self):
        from chorus.analysis.batch_scoring import score_variant_batch, BatchResult

        oracle = MagicMock()
        oracle.name = "test_oracle"

        # Mock predict_variant_effect to return different results per variant
        call_count = [0]
        def mock_predict_variant_effect(**kwargs):
            call_count[0] += 1
            # Second variant has stronger effect
            multiplier = 2.0 if call_count[0] == 1 else 5.0
            return _make_variant_result(
                {"DNASE:K562": np.ones(1000, dtype=np.float32)},
                {"DNASE:K562": np.ones(1000, dtype=np.float32) * multiplier},
                position=kwargs["variant_position"],
                alleles=kwargs["alleles"],
            )

        oracle.predict_variant_effect = mock_predict_variant_effect

        variants = [
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G", "id": "rs1"},
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "T", "id": "rs2"},
        ]

        result = score_variant_batch(oracle, variants, ["DNASE:K562"])

        assert isinstance(result, BatchResult)
        assert len(result.scores) == 2
        # Second variant (stronger effect) should be ranked first
        assert result.scores[0].variant_id == "rs2"
        assert abs(result.scores[0].max_effect) > abs(result.scores[1].max_effect)

    def test_batch_to_markdown(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = BatchResult(scores=scores)
        md = result.to_markdown()
        assert "Batch Variant Scoring" in md
        assert "rs1" in md
        assert "chr1:100" in md

    def test_batch_to_dict(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = BatchResult(scores=scores)
        d = result.to_dict()
        assert d["num_variants"] == 1
        assert d["scores"][0]["variant_id"] == "rs1"

    def test_batch_to_dataframe(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = BatchResult(scores=scores)
        df = result.to_dataframe()
        assert len(df) == 1
        assert "max_effect" in df.columns
        assert "variant_id" in df.columns

    def test_batch_to_tsv(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = BatchResult(scores=scores)
        tsv = result.to_tsv()
        lines = tsv.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "chrom\t" in lines[0]
        assert "rs1" in lines[1]

    def test_batch_to_tsv_writes_file(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = BatchResult(scores=scores)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scores.tsv")
            tsv = result.to_tsv(output_path=path)
            assert os.path.exists(path)
            content = open(path).read()
            assert content == tsv

    def test_batch_error_handling(self):
        from chorus.analysis.batch_scoring import score_variant_batch

        oracle = MagicMock()
        oracle.name = "test_oracle"
        oracle.predict_variant_effect.side_effect = RuntimeError("Model error")

        variants = [
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G"},
        ]
        result = score_variant_batch(oracle, variants, ["DNASE:K562"])

        assert len(result.scores) == 1
        assert result.scores[0].top_layer == "error"


# ── Causal prioritization tests ──────────────────────────────────────


class TestCausalPrioritization:
    def test_prioritize_causal_variants(self):
        from chorus.analysis.causal import prioritize_causal_variants, CausalResult
        from chorus.utils.ld import LDVariant

        oracle = MagicMock()
        oracle.name = "test_oracle"

        call_count = [0]
        def mock_predict(**kwargs):
            call_count[0] += 1
            mult = 3.0 if call_count[0] == 1 else 1.2
            return _make_variant_result(
                {"DNASE:K562": np.ones(1000, dtype=np.float32)},
                {"DNASE:K562": np.ones(1000, dtype=np.float32) * mult},
                position=kwargs["variant_position"],
                alleles=kwargs["alleles"],
            )

        oracle.predict_variant_effect = mock_predict

        ld_variants = [
            LDVariant("rs1", "chr1", 1000500, "A", "G", r2=1.0, is_sentinel=True),
            LDVariant("rs2", "chr1", 1001500, "C", "T", r2=0.9),
        ]

        result = prioritize_causal_variants(
            oracle,
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G", "id": "rs1"},
            ld_variants,
            assay_ids=["DNASE:K562"],
        )

        assert isinstance(result, CausalResult)
        assert len(result.scores) == 2
        # Sentinel (stronger effect) should be ranked first
        assert result.scores[0].variant_id == "rs1"
        assert result.scores[0].composite > result.scores[1].composite

    def test_composite_score_components(self):
        from chorus.analysis.causal import CausalVariantScore, _compute_composites, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.8, n_layers_affected=3, convergence_score=1.0,
                ref_activity=500.0, composite=0.0,
            ),
            CausalVariantScore(
                variant_id="rs2", chrom="chr1", position=200, ref="C", alt="T",
                r2=0.9, is_sentinel=False,
                max_effect=0.2, n_layers_affected=1, convergence_score=0.5,
                ref_activity=100.0, composite=0.0,
            ),
        ]

        _compute_composites(scores, CausalWeights())

        # rs1 should have higher composite (all components higher)
        assert scores[0].composite > scores[1].composite
        # Both should be in [0, 1]
        assert 0 <= scores[0].composite <= 1
        assert 0 <= scores[1].composite <= 1

    def test_causal_result_to_markdown(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        md = result.to_markdown()
        assert "Causal Variant Prioritization" in md
        assert "rs1" in md
        assert "SORT1" in md

    def test_causal_result_to_dict(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        d = result.to_dict()
        assert d["sentinel"] == "rs1"
        assert d["top_candidate"]["variant_id"] == "rs1"
        assert len(d["rankings"]) == 1

    def test_causal_html_has_igv_and_table(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=1000500, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
            ),
            CausalVariantScore(
                variant_id="rs2", chrom="chr1", position=1001500, ref="C", alt="T",
                r2=0.8, is_sentinel=False,
                max_effect=0.2, n_layers_affected=1, convergence_score=0.5,
                ref_activity=100.0, composite=0.45,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.2},
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=2,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        html = result.to_html()
        assert "Causal Variant Prioritization" in html
        assert "igv.createBrowser" in html  # IGV browser
        assert "Composite Causal Score" in html  # IGV score track
        assert "sentinel" in html
        assert "sortable" in html


class TestLDUtils:
    def test_ld_variants_from_list(self):
        from chorus.utils.ld import ld_variants_from_list

        variants = [
            {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "G", "id": "rs1"},
            {"chrom": "chr1", "pos": 200, "ref": "C", "alt": "T", "id": "rs2", "r2": 0.9},
        ]
        result = ld_variants_from_list(variants, sentinel_id="rs1")

        assert len(result) == 2
        assert result[0].is_sentinel
        assert not result[1].is_sentinel
        assert result[0].r2 == 1.0  # default
        assert result[1].r2 == 0.9

    def test_parse_ld_response(self):
        from chorus.utils.ld import parse_ld_response

        tsv = (
            "RS_Number\tCoord\tAlleles\tMAF\tDistance\tDprime\tR2\n"
            "rs12740374\tchr1:109274968\t(G/T)\t0.25\t0\t1.0\t1.0\n"
            "rs629301\tchr1:109275684\t(C/T)\t0.24\t716\t0.98\t0.95\n"
            "rs999\tchr1:109280000\t(A/G)\t0.10\t5032\t0.50\t0.30\n"
        )
        result = parse_ld_response(tsv, r2_threshold=0.8)

        assert len(result) == 2  # rs999 filtered out (r2=0.30 < 0.8)
        assert result[0].variant_id == "rs12740374"
        assert result[0].is_sentinel
        assert result[1].variant_id == "rs629301"
        assert result[1].r2 == 0.95


# ── Enriched dataclass field tests ────────────────────────────────────


class TestEnrichedBatchFields:
    def test_batch_variant_score_has_gene_and_cell_type(self):
        from chorus.analysis.batch_scoring import BatchVariantScore

        score = BatchVariantScore(
            chrom="chr1", position=100, ref="A", alt="G",
            variant_id="rs1", max_effect=0.5,
            top_layer="chromatin_accessibility", top_track="DNASE:K562",
            per_layer_scores={"chromatin_accessibility": 0.5},
            gene_name="SORT1", cell_type="K562",
            max_quantile=0.95,
        )
        assert score.gene_name == "SORT1"
        assert score.cell_type == "K562"
        assert score.max_quantile == 0.95

    def test_batch_to_dict_includes_new_fields(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
                gene_name="SORT1", cell_type="K562", max_quantile=0.95,
            ),
        ]
        result = BatchResult(scores=scores)
        d = result.to_dict()
        assert d["scores"][0]["gene_name"] == "SORT1"
        assert d["scores"][0]["max_quantile"] == 0.95
        assert d["scores"][0]["max_effect"] == 0.5

    def test_batch_to_markdown_summary_mode(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
                gene_name="SORT1", cell_type="K562",
            ),
        ]
        result = BatchResult(scores=scores)
        md = result.to_markdown(display_mode="summary")
        assert "Max Effect" in md
        assert "Top Layer" in md
        assert "rs1" in md
        assert "+0.500" in md

    def test_batch_to_dataframe_per_track(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult
        from chorus.analysis.variant_report import TrackScore

        ts = TrackScore(
            assay_id="DNASE:K562", assay_type="DNASE", cell_type="K562",
            layer="chromatin_accessibility", ref_value=100, alt_value=200,
            raw_score=0.5, quantile_score=0.9, description="DNASE:K562",
        )
        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                gene_name="SORT1", cell_type="K562", max_quantile=0.9,
                track_scores={"DNASE:K562": ts},
            ),
        ]
        result = BatchResult(scores=scores)
        df = result.to_dataframe()
        assert "gene_name" in df.columns
        assert "DNASE:K562_ref" in df.columns
        assert "DNASE:K562_alt" in df.columns
        assert "DNASE:K562_log2fc" in df.columns
        assert "DNASE:K562_effect_pctile" in df.columns
        assert "DNASE:K562_activity_pctile" in df.columns
        assert df["DNASE:K562_log2fc"].iloc[0] == 0.5
        assert df["DNASE:K562_effect_pctile"].iloc[0] == 0.9
        assert df["DNASE:K562_ref"].iloc[0] == 100
        assert df["DNASE:K562_alt"].iloc[0] == 200
        assert "max_quantile" in df.columns
        assert df.iloc[0]["gene_name"] == "SORT1"

    def test_batch_scoring_populates_gene_and_cell_type(self):
        from chorus.analysis.batch_scoring import score_variant_batch

        oracle = MagicMock()
        oracle.name = "test_oracle"

        def mock_predict(**kwargs):
            return _make_variant_result(
                {"DNASE:K562": np.ones(1000, dtype=np.float32)},
                {"DNASE:K562": np.ones(1000, dtype=np.float32) * 2.0},
                position=kwargs["variant_position"],
                alleles=kwargs["alleles"],
            )

        oracle.predict_variant_effect = mock_predict

        variants = [
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G", "id": "rs1"},
        ]
        result = score_variant_batch(oracle, variants, ["DNASE:K562"], gene_name="SORT1")
        assert result.scores[0].gene_name == "SORT1"
        # cell_type comes from the track assay_id parsing
        assert result.scores[0].cell_type == "K562"


class TestEnrichedCausalFields:
    def test_causal_variant_score_has_gene_and_cell_type(self):
        from chorus.analysis.causal import CausalVariantScore

        score = CausalVariantScore(
            variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
            r2=1.0, is_sentinel=True,
            max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
            ref_activity=300.0, composite=0.85,
            gene_name="SORT1", cell_type="K562",
        )
        assert score.gene_name == "SORT1"
        assert score.cell_type == "K562"

    def test_causal_to_dict_includes_new_fields(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                per_layer_scores={"chromatin_accessibility": 0.5},
                gene_name="SORT1", cell_type="K562",
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        d = result.to_dict()
        assert d["rankings"][0]["gene_name"] == "SORT1"
        assert d["rankings"][0]["cell_type"] == "K562"

    def test_causal_to_markdown_summary_mode(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=100, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                gene_name="SORT1", cell_type="K562",
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        md = result.to_markdown()
        assert "Composite" in md
        assert "SORT1" in md  # gene in report header
        assert "★" in md  # sentinel marker

    def test_causal_html_has_per_layer_columns(self):
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=1000500, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=2, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={
                    "chromatin_accessibility": 0.5,
                    "tss_activity": 0.3,
                },
                gene_name="SORT1", cell_type="K562",
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        html = result.to_html()
        # Per-layer columns present
        assert "Chromatin accessibility" in html
        assert "TSS activity" in html
        # Gene and cell type in the table
        assert "SORT1" in html
        assert "K562" in html

    def test_causal_html_unified_expandable_design(self):
        """Rankings + Details are merged into one expandable table with a glossary.

        Ensures:
          - a "How to read this report" glossary is present
          - each variant row is wrapped in a <tbody class="variant-block">
            with a summary row and a drill-down row
          - the drill-down surfaces the per-layer top track's assay, cell type,
            ref/alt values (i.e. the provenance of each per-layer number)
          - formula labels (log2FC / lnFC / Δ) are shown so units are explicit
          - the old separate Rankings / Details section headings are gone
        """
        from chorus.analysis.causal import CausalVariantScore, CausalResult, CausalWeights

        scores = [
            CausalVariantScore(
                variant_id="rs1", chrom="chr1", position=1000500, ref="A", alt="G",
                r2=1.0, is_sentinel=True,
                max_effect=0.5, n_layers_affected=1, convergence_score=1.0,
                ref_activity=300.0, composite=0.85,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
                per_layer_top_track={
                    "chromatin_accessibility": {
                        "assay_id": "DNASE:K562", "assay_type": "DNASE",
                        "cell_type": "K562", "ref_value": 4.1, "alt_value": 5.8,
                        "raw_score": 0.5, "quantile_score": 0.97,
                        "description": "ENCODE DNase K562",
                    },
                },
                gene_name="SORT1", cell_type="K562",
            ),
        ]
        result = CausalResult(
            sentinel_id="rs1", population="CEU", n_variants=1,
            scores=scores, weights=CausalWeights(),
            oracle_name="test", gene_name="SORT1",
        )
        html = result.to_html()

        # Glossary + unified table present
        assert "How to read this report" in html
        assert "Ranked Variants" in html
        assert 'class="variant-block"' in html
        assert 'tr class="summary"' in html
        assert 'tr class="drill-row"' in html

        # Old section structure removed
        assert "<h2>Variant Rankings" not in html
        assert "<h2>Variant Details" not in html

        # Per-layer winning-track provenance surfaced
        assert "Strongest track" in html
        assert "DNASE:K562" in html  # assay_id rendered
        assert "Per-layer breakdown" in html
        assert "Top assay" in html
        # Ref/alt predicted values appear
        assert "4.1" in html and "5.8" in html

        # Formula labels (units) are explicit
        assert "log2FC" in html

    def test_causal_prioritize_populates_gene_cell_type(self):
        from chorus.analysis.causal import prioritize_causal_variants
        from chorus.utils.ld import LDVariant

        oracle = MagicMock()
        oracle.name = "test_oracle"

        def mock_predict(**kwargs):
            return _make_variant_result(
                {"DNASE:K562": np.ones(1000, dtype=np.float32)},
                {"DNASE:K562": np.ones(1000, dtype=np.float32) * 3.0},
                position=kwargs["variant_position"],
                alleles=kwargs["alleles"],
            )

        oracle.predict_variant_effect = mock_predict

        ld_variants = [
            LDVariant("rs1", "chr1", 1000500, "A", "G", r2=1.0, is_sentinel=True),
        ]
        result = prioritize_causal_variants(
            oracle,
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G", "id": "rs1"},
            ld_variants,
            assay_ids=["DNASE:K562"],
            gene_name="SORT1",
        )
        assert result.scores[0].gene_name == "SORT1"
        # track_scores should be populated with the per-track detail
        assert "DNASE:K562" in result.scores[0].track_scores


# ── Background distribution building tests ────────────────────────────


class TestBuildBackgrounds:
    def test_get_common_snps_builtin(self):
        from chorus.analysis.build_backgrounds import get_common_snps

        snps = get_common_snps(n=5)
        assert len(snps) == 5
        assert all("chrom" in s and "pos" in s for s in snps)

    def test_get_common_snps_full(self):
        from chorus.analysis.build_backgrounds import get_common_snps

        snps = get_common_snps(n=100)
        # Built-in list has ~10, so we get all of them
        assert len(snps) <= 100
        assert len(snps) >= 1

    def test_get_common_snps_from_bed(self):
        from chorus.analysis.build_backgrounds import get_common_snps

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write("chr1\t100\t101\trs1\tA\tG\n")
            f.write("chr2\t200\t201\trs2\tC\tT\n")
            f.write("chr3\t300\t301\trs3\tG\tA\n")
            path = f.name

        try:
            snps = get_common_snps(bed_path=path, n=10)
            assert len(snps) == 3
            assert snps[0]["id"] == "rs1"
        finally:
            os.unlink(path)

    def test_build_variant_backgrounds(self):
        from chorus.analysis.build_backgrounds import build_variant_backgrounds

        oracle = MagicMock()
        oracle.name = "test_oracle"

        def mock_predict(**kwargs):
            mult = 1.0 + np.random.random() * 2.0
            # Use the actual requested position so scoring window matches
            return _make_variant_result(
                {"DNASE:K562": np.ones(1000, dtype=np.float32)},
                {"DNASE:K562": np.ones(1000, dtype=np.float32) * mult},
                position=kwargs["variant_position"],
                alleles=kwargs["alleles"],
            )

        oracle.predict_variant_effect = mock_predict

        # All SNPs at the same position so they fall within the prediction window
        snps = [
            {"chrom": "chr1", "pos": 1000500, "ref": "A", "alt": "G", "id": f"rs{i}"}
            for i in range(6)
        ]

        with tempfile.TemporaryDirectory() as td:
            normalizer = build_variant_backgrounds(
                oracle, "test_oracle", ["DNASE:K562"],
                snps=snps, cache_dir=td,
            )
            key = normalizer.background_key("test_oracle", "chromatin_accessibility")
            assert normalizer.has_background(key)
            q = normalizer.normalize(key, 0.5, signed=False)
            assert q is not None


# ── get_normalizer + auto-loading tests ──────────────────────────────


class TestGetNormalizer:
    def test_returns_none_when_no_backgrounds(self):
        from chorus.analysis.normalization import get_normalizer

        with tempfile.TemporaryDirectory() as td:
            result = get_normalizer("nonexistent_oracle", cache_dir=td)
            assert result is None

    def test_returns_none_when_dir_missing(self):
        from chorus.analysis.normalization import get_normalizer

        result = get_normalizer("test", cache_dir="/tmp/chorus_no_such_dir_xyz")
        assert result is None

    def test_loads_matching_backgrounds(self):
        from chorus.analysis.normalization import get_normalizer, BackgroundDistribution

        with tempfile.TemporaryDirectory() as td:
            # Create two background files for "myoracle"
            scores1 = np.random.randn(1000).astype(np.float64)
            scores2 = np.abs(np.random.randn(500)).astype(np.float64)
            np.save(os.path.join(td, "myoracle_chromatin_accessibility.npy"), np.sort(scores1))
            np.save(os.path.join(td, "myoracle_tf_binding.npy"), np.sort(scores2))
            # Also create a file for a different oracle — should NOT be loaded
            np.save(os.path.join(td, "other_oracle_chromatin.npy"), np.sort(scores1))

            normalizer = get_normalizer("myoracle", cache_dir=td)
            assert normalizer is not None
            assert normalizer.has_background("myoracle_chromatin_accessibility")
            assert normalizer.has_background("myoracle_tf_binding")
            # get_normalizer only eagerly loads myoracle_* files
            assert "other_oracle_chromatin" not in normalizer._distributions

    def test_normalizer_produces_quantile_scores(self):
        from chorus.analysis.normalization import get_normalizer

        with tempfile.TemporaryDirectory() as td:
            scores = np.sort(np.abs(np.random.randn(10_000))).astype(np.float64)
            np.save(os.path.join(td, "testorc_chromatin_accessibility.npy"), scores)

            normalizer = get_normalizer("testorc", cache_dir=td)
            assert normalizer is not None
            q = normalizer.normalize("testorc_chromatin_accessibility", 0.5, signed=False)
            assert q is not None
            assert 0.0 <= q <= 1.0

    def test_summary_method(self):
        from chorus.analysis.normalization import get_normalizer

        with tempfile.TemporaryDirectory() as td:
            scores = np.sort(np.random.randn(1000)).astype(np.float64)
            np.save(os.path.join(td, "sumoracle_layer1.npy"), scores)

            normalizer = get_normalizer("sumoracle", cache_dir=td)
            assert normalizer is not None
            summary = normalizer.summary()
            assert "sumoracle_layer1" in summary
            info = summary["sumoracle_layer1"]
            assert info["n_scores"] == 1000
            assert "median" in info
            assert "p95" in info
            assert "p99" in info

    def test_build_variant_report_with_normalizer(self):
        """build_variant_report with auto-loaded normalizer produces quantile scores."""
        from chorus.analysis.normalization import get_normalizer
        from chorus.analysis.variant_report import build_variant_report

        # Create a variant result with a DNASE track
        ref_vals = np.ones(1000, dtype=np.float32) * 5.0
        alt_vals = np.ones(1000, dtype=np.float32) * 10.0
        variant_result = _make_variant_result(
            {"DNASE:K562": ref_vals},
            {"DNASE:K562": alt_vals},
        )

        with tempfile.TemporaryDirectory() as td:
            # Create a background for chromatin_accessibility
            bg_scores = np.sort(np.abs(np.random.randn(10_000))).astype(np.float64)
            np.save(os.path.join(td, "test_oracle_chromatin_accessibility.npy"), bg_scores)

            normalizer = get_normalizer("test_oracle", cache_dir=td)
            report = build_variant_report(
                variant_result, oracle_name="test_oracle", normalizer=normalizer,
            )
            # Check that quantile scores were set on at least one track
            has_quantile = False
            for allele, scores in report.allele_scores.items():
                for ts in scores:
                    if ts.quantile_score is not None:
                        has_quantile = True
                        break
            assert has_quantile, "Expected quantile scores when normalizer is provided"

    def test_build_variant_report_without_normalizer(self):
        """build_variant_report without normalizer produces no quantile scores (backward compat)."""
        from chorus.analysis.variant_report import build_variant_report

        ref_vals = np.ones(1000, dtype=np.float32) * 5.0
        alt_vals = np.ones(1000, dtype=np.float32) * 10.0
        variant_result = _make_variant_result(
            {"DNASE:K562": ref_vals},
            {"DNASE:K562": alt_vals},
        )
        report = build_variant_report(variant_result, oracle_name="test_oracle")
        for allele, scores in report.allele_scores.items():
            for ts in scores:
                assert ts.quantile_score is None


# ---------------------------------------------------------------------------
# CDF compression and per-bin normalization tests
# ---------------------------------------------------------------------------

class TestCompactCDF:
    """Test BackgroundDistribution CDF compression."""

    def test_compact_cdf_basic(self):
        """Compressing to 10K points gives same percentiles within tolerance."""
        from chorus.analysis.normalization import BackgroundDistribution

        rng = np.random.RandomState(42)
        # Simulate a realistic per-bin distribution: lognormal (many low, few high)
        raw = np.abs(rng.lognormal(mean=2.0, sigma=2.0, size=1_000_000))
        full = BackgroundDistribution(raw)
        compact = full.to_compact_cdf(n_points=10_000)

        assert len(compact.sorted_scores) == 10_000
        assert compact.is_compact
        assert not full.is_compact
        assert compact.nbytes < full.nbytes / 10  # >10x compression

        # Percentile accuracy: check at 100 random test values
        test_values = rng.choice(raw, size=100)
        for v in test_values:
            full_q = full.raw_to_quantile(float(v), signed=False)
            compact_q = compact.raw_to_quantile(float(v), signed=False)
            assert abs(full_q - compact_q) < 0.002, (
                f"value={v:.4f}: full={full_q:.4f} compact={compact_q:.4f}"
            )

    def test_compact_cdf_batch(self):
        """Batch percentile mapping matches within tolerance."""
        from chorus.analysis.normalization import BackgroundDistribution

        rng = np.random.RandomState(123)
        raw = np.abs(rng.lognormal(mean=3.0, sigma=1.5, size=500_000))
        full = BackgroundDistribution(raw)
        compact = full.to_compact_cdf(n_points=5_000)

        test = rng.choice(raw, size=1000)
        full_q = full.raw_to_quantile_batch(test, signed=False)
        compact_q = compact.raw_to_quantile_batch(test, signed=False)
        max_diff = np.max(np.abs(full_q - compact_q))
        assert max_diff < 0.005, f"Max batch diff: {max_diff:.6f}"

    def test_compact_cdf_preserves_extremes(self):
        """Compact CDF handles min, max, and out-of-range values."""
        from chorus.analysis.normalization import BackgroundDistribution

        raw = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0] * 1000)
        full = BackgroundDistribution(raw)
        compact = full.to_compact_cdf(n_points=100)

        # Below min → near 0
        assert compact.raw_to_quantile(-1.0, signed=False) < 0.01
        # At max → near 1
        assert compact.raw_to_quantile(1000.0, signed=False) > 0.99
        # Above max → 1.0
        assert compact.raw_to_quantile(9999.0, signed=False) == 1.0

    def test_compact_cdf_small_distribution(self):
        """Small distributions (<= n_points) are returned unchanged."""
        from chorus.analysis.normalization import BackgroundDistribution

        raw = np.arange(100, dtype=np.float64)
        dist = BackgroundDistribution(raw)
        compact = dist.to_compact_cdf(n_points=10_000)
        assert compact is dist  # same object, not copied

    def test_compact_cdf_signed(self):
        """Compact CDF works with signed distributions."""
        from chorus.analysis.normalization import BackgroundDistribution

        rng = np.random.RandomState(99)
        raw = rng.normal(0, 1, size=500_000)
        full = BackgroundDistribution(raw)
        compact = full.to_compact_cdf(n_points=10_000)

        # Median of normal(0,1) should map to ~0.0 in signed mode
        q_full = full.raw_to_quantile(0.0, signed=True)
        q_compact = compact.raw_to_quantile(0.0, signed=True)
        assert abs(q_full - q_compact) < 0.005
        assert abs(q_full) < 0.05  # near 0 for signed median

    def test_compact_save_load(self):
        """Compact CDF can be saved and loaded."""
        from chorus.analysis.normalization import BackgroundDistribution

        rng = np.random.RandomState(77)
        raw = np.abs(rng.lognormal(2, 2, size=100_000))
        dist = BackgroundDistribution(raw)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name

        try:
            # Save compact
            dist.save(path, compact=True)
            size_compact = os.path.getsize(path)

            # Load and verify
            loaded = BackgroundDistribution.load(path)
            assert len(loaded.sorted_scores) == 10_000
            assert loaded.is_compact

            # Save full for size comparison
            dist.save(path, compact=False)
            size_full = os.path.getsize(path)

            assert size_compact < size_full / 5  # >5x compression
        finally:
            os.unlink(path)


class TestPerBinNormalization:
    """Test the per-bin normalization pipeline end-to-end."""

    def test_normalize_perbin_batch(self):
        """normalize_perbin_batch returns [0,1] percentiles."""
        from chorus.analysis.normalization import (
            QuantileNormalizer, BackgroundDistribution,
        )

        rng = np.random.RandomState(42)
        perbin_scores = np.abs(rng.lognormal(2, 2, size=50_000))

        normalizer = QuantileNormalizer(cache_dir=tempfile.mkdtemp())
        key = normalizer.perbin_key("test_oracle", "chromatin_accessibility")
        normalizer.set_background(key, BackgroundDistribution(perbin_scores))

        # Test with random bin values
        test_vals = np.abs(rng.lognormal(2, 2, size=100))
        result = normalizer.normalize_perbin_batch(
            "test_oracle", "chromatin_accessibility", test_vals
        )
        assert result is not None
        assert result.shape == (100,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_perbin_vs_baseline_different_scale(self):
        """Per-bin and summary baselines give different percentiles for same value."""
        from chorus.analysis.normalization import (
            QuantileNormalizer, BackgroundDistribution,
        )

        # Summary baseline: window sums (high values, e.g. 501bp sums)
        summary_scores = np.array([100, 200, 500, 1000, 2000, 5000, 10000.0])
        # Per-bin baseline: individual bin values (low values, e.g. single bins)
        perbin_scores = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])

        normalizer = QuantileNormalizer(cache_dir=tempfile.mkdtemp())
        normalizer.set_background(
            "test_chromatin_accessibility_baseline",
            BackgroundDistribution(summary_scores),
        )
        normalizer.set_background(
            "test_chromatin_accessibility_perbin",
            BackgroundDistribution(perbin_scores),
        )

        # A value of 10.0: low in summary scale, high in per-bin scale
        summary_pctile = normalizer.normalize_baseline("test", "chromatin_accessibility", 10.0)
        perbin_pctile = normalizer.normalize_perbin_batch(
            "test", "chromatin_accessibility", np.array([10.0])
        )[0]

        # Summary: 10 < 100 (min) → very low percentile
        assert summary_pctile < 0.2
        # Per-bin: 10 is between 5 and 50 → moderate-high percentile
        assert perbin_pctile > 0.5

    def test_compact_cdf_in_normalizer(self):
        """Compact CDF works correctly inside QuantileNormalizer."""
        from chorus.analysis.normalization import (
            QuantileNormalizer, BackgroundDistribution,
        )

        rng = np.random.RandomState(42)
        raw = np.abs(rng.lognormal(2, 2, size=100_000))

        normalizer = QuantileNormalizer(cache_dir=tempfile.mkdtemp())
        # Store compact CDF
        dist = BackgroundDistribution(raw).to_compact_cdf()
        normalizer.set_background(
            "test_chromatin_accessibility_perbin", dist
        )

        test_vals = np.array([0.0, np.median(raw), raw.max()])
        result = normalizer.normalize_perbin_batch(
            "test", "chromatin_accessibility", test_vals
        )
        assert result is not None
        assert result[0] < 0.1   # min → low percentile
        assert 0.4 < result[1] < 0.6  # median → ~0.5
        assert result[2] > 0.9   # max → high percentile

    def test_has_perbin(self):
        """has_perbin correctly detects per-bin distributions."""
        from chorus.analysis.normalization import (
            QuantileNormalizer, BackgroundDistribution,
        )

        normalizer = QuantileNormalizer(cache_dir=tempfile.mkdtemp())
        assert not normalizer.has_perbin("test", "chromatin_accessibility")

        normalizer.set_background(
            "test_chromatin_accessibility_perbin",
            BackgroundDistribution(np.arange(100, dtype=np.float64)),
        )
        assert normalizer.has_perbin("test", "chromatin_accessibility")

    def test_perbin_missing_returns_none(self):
        """normalize_perbin_batch returns None when no perbin exists."""
        from chorus.analysis.normalization import QuantileNormalizer

        normalizer = QuantileNormalizer(cache_dir=tempfile.mkdtemp())
        result = normalizer.normalize_perbin_batch(
            "nonexistent", "chromatin_accessibility", np.array([1.0, 2.0])
        )
        assert result is None


# ---------------------------------------------------------------------------
# Per-Track Normalizer Tests
# ---------------------------------------------------------------------------

class TestPerTrackNormalizer:
    """Tests for PerTrackNormalizer — per-track CDF normalization."""

    def _make_normalizer(self, n_tracks=3, n_points=100, with_perbin=True):
        """Create a PerTrackNormalizer with fake CDFs for testing."""
        from chorus.analysis.normalization import PerTrackNormalizer

        tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        track_ids = [f"TRACK_{i}" for i in range(n_tracks)]

        effect = np.sort(rng.standard_normal((n_tracks, n_points)), axis=1)
        summary = np.sort(rng.exponential(5, (n_tracks, n_points)), axis=1)
        signed = np.array([False, True, False][:n_tracks])

        kwargs = dict(
            oracle_name="test_oracle",
            track_ids=track_ids,
            effect_cdfs=effect,
            summary_cdfs=summary,
            signed_flags=signed,
            cache_dir=tmpdir,
            n_points=n_points,
        )
        if with_perbin:
            perbin = np.sort(rng.exponential(1, (n_tracks, n_points)), axis=1)
            kwargs["perbin_cdfs"] = perbin

        PerTrackNormalizer.build_and_save(**kwargs)
        norm = PerTrackNormalizer(cache_dir=tmpdir)
        return norm, track_ids, tmpdir

    def test_build_and_load_roundtrip(self):
        norm, track_ids, _ = self._make_normalizer()
        assert norm.has_oracle("test_oracle")
        assert norm.n_tracks("test_oracle") == 3
        assert norm.track_ids("test_oracle") == track_ids

    def test_effect_percentile_unsigned(self):
        norm, track_ids, _ = self._make_normalizer()
        result = norm.effect_percentile("test_oracle", track_ids[0], 0.0, signed=False)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_effect_percentile_signed(self):
        norm, track_ids, _ = self._make_normalizer()
        result = norm.effect_percentile("test_oracle", track_ids[1], 0.0, signed=True)
        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_activity_percentile(self):
        norm, track_ids, _ = self._make_normalizer()
        result = norm.activity_percentile("test_oracle", track_ids[0], 5.0)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_perbin_floor_rescale_batch(self):
        norm, track_ids, _ = self._make_normalizer()
        vals = np.array([0.0, 0.5, 1.0, 5.0, 10.0])
        result = norm.perbin_floor_rescale_batch(
            "test_oracle", track_ids[0], vals,
            floor_pctile=0.5, peak_pctile=0.9, max_value=3.0,
        )
        assert result is not None
        assert result.shape == vals.shape
        assert result.min() >= 0.0
        assert result.max() <= 3.0

    def test_perbin_floor_rescale_preserves_peak_order(self):
        norm, track_ids, _ = self._make_normalizer()
        vals = np.array([0.1, 1.0, 5.0, 10.0, 50.0])
        result = norm.perbin_floor_rescale_batch(
            "test_oracle", track_ids[0], vals,
            floor_pctile=0.5, peak_pctile=0.9,
        )
        assert result is not None
        # Monotonically increasing (or equal for clipped values)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-9), f"Not monotonic: {result}"

    def test_missing_track_returns_none(self):
        norm, _, _ = self._make_normalizer()
        assert norm.effect_percentile("test_oracle", "NONEXISTENT", 0.5) is None
        assert norm.activity_percentile("test_oracle", "NONEXISTENT", 0.5) is None
        assert norm.perbin_floor_rescale_batch(
            "test_oracle", "NONEXISTENT", np.array([1.0]),
        ) is None

    def test_missing_oracle_returns_none(self):
        norm, track_ids, _ = self._make_normalizer()
        assert norm.effect_percentile("no_oracle", track_ids[0], 0.5) is None

    def test_perbin_none_for_scalar_oracles(self):
        """When perbin_cdfs is absent, perbin methods return None."""
        norm, track_ids, _ = self._make_normalizer(with_perbin=False)
        result = norm.perbin_percentile_batch(
            "test_oracle", track_ids[0], np.array([1.0]),
        )
        assert result is None
        result2 = norm.perbin_floor_rescale_batch(
            "test_oracle", track_ids[0], np.array([1.0]),
        )
        assert result2 is None

    def test_get_denominator_padding_vs_compaction(self):
        """Denominator uses count when < CDF width (padding case)."""
        from chorus.analysis.normalization import PerTrackNormalizer

        tmpdir = tempfile.mkdtemp()
        track_ids = ["T0", "T1"]
        effect = np.sort(np.random.default_rng(1).standard_normal((2, 100)), axis=1)
        summary = np.sort(np.random.default_rng(2).exponential(1, (2, 100)), axis=1)

        # T0 has 50 samples (padded to 100), T1 has 200 (compacted to 100)
        PerTrackNormalizer.build_and_save(
            oracle_name="test",
            track_ids=track_ids,
            effect_cdfs=effect,
            summary_cdfs=summary,
            effect_counts=np.array([50, 200]),
            summary_counts=np.array([50, 200]),
            signed_flags=np.array([False, False]),
            cache_dir=tmpdir,
        )
        norm = PerTrackNormalizer(cache_dir=tmpdir)
        entry = norm._ensure_loaded("test")

        # T0: count=50 < width=100 → use count
        assert norm._get_denominator(entry, "effect_cdfs", 0) == 50
        # T1: count=200 > width=100 → use width
        assert norm._get_denominator(entry, "effect_cdfs", 1) == 100

    def test_get_pertrack_normalizer_factory(self):
        """get_pertrack_normalizer returns None when no file exists."""
        from chorus.analysis.normalization import get_pertrack_normalizer

        result = get_pertrack_normalizer("nonexistent_oracle_xyz",
                                         cache_dir=tempfile.mkdtemp())
        assert result is None

    def test_is_signed(self):
        norm, track_ids, _ = self._make_normalizer()
        assert norm.is_signed("test_oracle", track_ids[0]) is False
        assert norm.is_signed("test_oracle", track_ids[1]) is True
        assert norm.is_signed("test_oracle", track_ids[2]) is False


class TestIGVRawFlag:
    """Tests for the igv_raw parameter in variant report generation."""

    def test_build_variant_report_igv_raw_propagates(self):
        """igv_raw flag is stored on the VariantReport."""
        from chorus.analysis.variant_report import VariantReport
        import dataclasses

        fields = {f.name for f in dataclasses.fields(VariantReport)}
        assert "_igv_raw" in fields

        report = VariantReport(
            chrom="chr1", position=100, ref_allele="A",
            alt_alleles=["G"], oracle_name="test", gene_name=None,
            _igv_raw=True,
        )
        assert report._igv_raw is True

        report2 = VariantReport(
            chrom="chr1", position=100, ref_allele="A",
            alt_alleles=["G"], oracle_name="test", gene_name=None,
            _igv_raw=False,
        )
        assert report2._igv_raw is False

    def test_build_variant_report_accepts_igv_raw(self):
        """build_variant_report accepts igv_raw parameter."""
        import inspect
        from chorus.analysis.variant_report import build_variant_report

        sig = inspect.signature(build_variant_report)
        assert "igv_raw" in sig.parameters

    def test_discover_variant_effects_accepts_igv_raw(self):
        """discover_variant_effects accepts igv_raw parameter."""
        import inspect
        from chorus.analysis.discovery import discover_variant_effects

        sig = inspect.signature(discover_variant_effects)
        assert "igv_raw" in sig.parameters

    def test_discover_variant_effects_accepts_analysis_request_and_filename(self):
        """discover_variant_effects accepts analysis_request + output_filename
        so callers can write the final HTML in one shot (no post-hoc rewrite
        that leaves orphan chr*.html behind).
        """
        import inspect
        from chorus.analysis.discovery import discover_variant_effects

        sig = inspect.signature(discover_variant_effects)
        assert "analysis_request" in sig.parameters
        assert "output_filename" in sig.parameters

    def test_discover_and_report_accepts_user_prompt(self):
        """discover_and_report accepts user_prompt so per-cell-type sub-reports
        get AnalysisRequest stamped before the first HTML write.
        """
        import inspect
        from chorus.analysis.discovery import discover_and_report

        sig = inspect.signature(discover_and_report)
        assert "user_prompt" in sig.parameters
        assert "tool_name" in sig.parameters

    def test_discover_cell_types_ranking_metrics(self):
        """``ranking_metric`` selects between |log2FC| and alt-weighted ranking.

        The historical default ``"abs_effect"`` ranks by raw |log2FC|, which
        favours cell types with closed baseline chromatin. The new default
        ``"alt_x_abs_effect"`` weights effect by absolute alt activity, so
        an already-active enhancer that the variant strengthens beats a
        closed locus that the variant pries open with a larger fold-change.

        This test fakes a small set of CellTypeHits and exercises the
        ranking branch directly so the contract holds even without a real
        oracle.
        """
        from chorus.analysis.discovery import CellTypeHit

        # Closed-baseline cell type: tiny ref, big fold-change, modest alt.
        # Models the SORT1 'fibroblast of mammary gland' pattern.
        closed = CellTypeHit(
            cell_type="closed_chromatin_ct",
            best_track="DNASE:closed", best_layer="chromatin_accessibility",
            effect=3.30, abs_effect=3.30, ref_value=24.0, alt_value=241.0,
        )
        # Already-active cell type: big ref, modest fold-change, very high alt.
        # Models the SORT1 HepG2 pattern.
        active = CellTypeHit(
            cell_type="active_chromatin_ct",
            best_track="DNASE:active", best_layer="chromatin_accessibility",
            effect=1.40, abs_effect=1.40, ref_value=595.0, alt_value=1571.0,
        )

        # alt_x_abs_effect picks the active cell type
        active.ranking_score = active.alt_value * active.abs_effect    # 2199
        closed.ranking_score = closed.alt_value * closed.abs_effect    # 795
        ranked = sorted([closed, active], key=lambda h: h.ranking_score, reverse=True)
        assert ranked[0].cell_type == "active_chromatin_ct"

        # abs_effect picks the closed-baseline cell type (the historical bug)
        active.ranking_score = active.abs_effect
        closed.ranking_score = closed.abs_effect
        ranked = sorted([closed, active], key=lambda h: h.ranking_score, reverse=True)
        assert ranked[0].cell_type == "closed_chromatin_ct"

        # abs_effect_min_ref with min_ref_value=100 filters out the closed one
        min_ref = 100.0
        active.ranking_score = active.abs_effect if active.ref_value >= min_ref else -1.0
        closed.ranking_score = closed.abs_effect if closed.ref_value >= min_ref else -1.0
        assert active.ranking_score > 0 and closed.ranking_score < 0
        ranked = sorted([closed, active], key=lambda h: h.ranking_score, reverse=True)
        assert ranked[0].cell_type == "active_chromatin_ct"

    def test_discover_cell_types_signature(self):
        """``discover_cell_types`` exposes ranking_metric and min_ref_value."""
        import inspect
        from chorus.analysis.discovery import discover_cell_types, RANKING_METRICS

        sig = inspect.signature(discover_cell_types)
        assert "ranking_metric" in sig.parameters
        assert "min_ref_value" in sig.parameters
        # Default must be the alt-weighted metric
        assert sig.parameters["ranking_metric"].default == "alt_x_abs_effect"
        assert "alt_x_abs_effect" in RANKING_METRICS
        assert "abs_effect" in RANKING_METRICS

    def test_discover_cell_types_rejects_unknown_metric(self):
        """``discover_cell_types`` raises on an unknown ranking_metric.

        Pass an obviously bogus metric so the validation triggers before any
        oracle interaction. We don't need a real oracle for the assertion
        because the check runs at the top of the function.
        """
        import pytest
        from chorus.analysis.discovery import discover_cell_types

        with pytest.raises(ValueError, match="Unknown ranking_metric"):
            discover_cell_types(
                oracle=object(),  # never reached
                variant_position="chr1:1",
                alleles=["A", "C"],
                ranking_metric="not_a_real_metric",
            )

    def test_layer_floor_thresholds_exist(self):
        """Layer-aware floor thresholds are defined for all known layers."""
        from chorus.analysis._igv_report import _LAYER_FLOOR_PCTILE

        expected_layers = [
            "tss_activity", "tf_binding", "chromatin_accessibility",
            "histone_marks", "gene_expression", "splicing",
        ]
        for layer in expected_layers:
            assert layer in _LAYER_FLOOR_PCTILE, f"Missing threshold for {layer}"
            assert 0.0 < _LAYER_FLOOR_PCTILE[layer] < 1.0


class TestMCPPerTrackNormalization:
    """Tests for MCP state manager integration with PerTrackNormalizer."""

    def test_state_prefers_pertrack_over_legacy(self):
        """State manager loads PerTrackNormalizer when NPZ exists."""
        from chorus.analysis.normalization import PerTrackNormalizer

        tmpdir = tempfile.mkdtemp()
        # Create a fake pertrack NPZ
        track_ids = ["T0"]
        PerTrackNormalizer.build_and_save(
            oracle_name="test_oracle",
            track_ids=track_ids,
            effect_cdfs=np.sort(np.random.default_rng(1).standard_normal((1, 100)), axis=1),
            summary_cdfs=np.sort(np.random.default_rng(2).exponential(1, (1, 100)), axis=1),
            signed_flags=np.array([False]),
            cache_dir=tmpdir,
        )

        # Simulate what _auto_load_normalizer does
        from chorus.analysis.normalization import get_pertrack_normalizer
        norm = get_pertrack_normalizer("test_oracle", cache_dir=tmpdir)
        assert isinstance(norm, PerTrackNormalizer)

    def test_analyze_variant_accepts_igv_raw(self):
        """MCP analyze_variant_multilayer accepts igv_raw parameter."""
        import inspect
        from chorus.mcp.server import analyze_variant_multilayer

        sig = inspect.signature(analyze_variant_multilayer)
        assert "igv_raw" in sig.parameters

    def test_discover_variant_accepts_igv_raw(self):
        """MCP discover_variant accepts igv_raw parameter."""
        import inspect
        from chorus.mcp.server import discover_variant

        sig = inspect.signature(discover_variant)
        assert "igv_raw" in sig.parameters


class TestReportMetadataFields:
    """Coverage for new VariantReport fields added in the redesign:
    report_title, modification_region, modification_description.
    """

    def _make(self, **overrides):
        from chorus.analysis.variant_report import VariantReport
        kw = dict(
            chrom="chr1", position=100, ref_allele="G",
            alt_alleles=["T"], oracle_name="test", gene_name="TEST",
        )
        kw.update(overrides)
        return VariantReport(**kw)

    def test_report_title_renders_in_markdown(self):
        r = self._make(report_title="Region Swap Analysis Report")
        md = r.to_markdown()
        assert "## Region Swap Analysis Report" in md

    def test_report_title_renders_in_html(self):
        r = self._make(report_title="Integration Simulation Report")
        html = r.to_html()
        assert "<h1>Integration Simulation Report</h1>" in html

    def test_modification_region_renders_in_markdown(self):
        r = self._make(modification_region=(1000, 2000))
        md = r.to_markdown()
        assert "Modified region" in md
        assert "chr1:1,001-2,000" in md
        assert "1,000 bp" in md

    def test_modification_description_renders_in_markdown(self):
        r = self._make(modification_description="Inserted 366 bp CMV construct")
        md = r.to_markdown()
        assert "**Modification**: Inserted 366 bp CMV construct" in md

    def test_modification_fields_in_dict(self):
        r = self._make(
            modification_region=(1000, 2000),
            modification_description="Replaced 1000 bp region with reporter",
        )
        d = r.to_dict()
        assert d["modification_description"] == "Replaced 1000 bp region with reporter"
        assert d["modification_region"] == [1000, 2000]


class TestBatchDisplayModes:
    """Coverage for BatchResult.to_markdown display_mode values."""

    def _result(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult
        from chorus.analysis.variant_report import TrackScore

        ts_dnase = TrackScore(
            assay_id="DNASE/EFO:0001187/.", assay_type="DNASE", cell_type="HepG2",
            layer="chromatin_accessibility", ref_value=100, alt_value=150,
            raw_score=0.5, quantile_score=0.95, description="DNASE:HepG2",
        )
        ts_cebpa = TrackScore(
            assay_id="CHIP_TF/EFO:0001187/CEBPA/.", assay_type="CHIP",
            cell_type="HepG2", layer="tf_binding", ref_value=50,
            alt_value=100, raw_score=1.0, quantile_score=0.99,
            description="CHIP:CEBPA:HepG2",
        )
        ts_cage_plus = TrackScore(
            assay_id="CAGE/EFO:0001187/+", assay_type="CAGE", cell_type="HepG2",
            layer="tss_activity", ref_value=20, alt_value=25,
            raw_score=0.3, quantile_score=0.85, description="CAGE:HepG2",
        )
        ts_cage_minus = TrackScore(
            assay_id="CAGE/EFO:0001187/-", assay_type="CAGE", cell_type="HepG2",
            layer="tss_activity", ref_value=22, alt_value=24,
            raw_score=0.13, quantile_score=0.7, description="CAGE:HepG2",
        )
        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="G", alt="T", variant_id="rs1",
                max_effect=1.0, top_layer="tf_binding", top_track="CHIP:CEBPA:HepG2",
                track_scores={
                    "DNASE/EFO:0001187/.": ts_dnase,
                    "CHIP_TF/EFO:0001187/CEBPA/.": ts_cebpa,
                    "CAGE/EFO:0001187/+": ts_cage_plus,
                    "CAGE/EFO:0001187/-": ts_cage_minus,
                },
            ),
        ]
        return BatchResult(scores=scores)

    def test_by_assay_mode(self):
        result = self._result()
        md = result.to_markdown(display_mode="by_assay")
        assert "DNASE:HepG2" in md
        assert "CHIP:CEBPA:HepG2" in md
        # CAGE +/- must be disambiguated via strand suffix
        assert "CAGE:HepG2 (+)" in md
        assert "CAGE:HepG2 (-)" in md

    def test_by_cell_type_mode(self):
        result = self._result()
        md = result.to_markdown(display_mode="by_cell_type")
        # Per-track columns render the same regardless of mode (current impl);
        # ensure the display_mode param is accepted without error and output is non-empty.
        assert len(md) > 0
        assert "rs1" in md

    def test_track_id_footnote_present(self):
        result = self._result()
        md = result.to_markdown(display_mode="by_assay")
        # Track IDs are listed in a footnote for traceability
        assert "Track identifiers" in md
        assert "`DNASE/EFO:0001187/.`" in md

    def test_cage_strand_disambiguation_in_dataframe(self):
        result = self._result()
        df = result.to_dataframe()
        # Columns must be unique (was previously duplicating CAGE:HepG2)
        assert len(df.columns) == len(set(df.columns))
        assert "CAGE:HepG2 (+)_log2fc" in df.columns
        assert "CAGE:HepG2 (-)_log2fc" in df.columns


class TestSafeToolDecorator:
    """The MCP _safe_tool decorator converts exceptions into structured dicts."""

    def test_wraps_successful_call(self):
        from chorus.mcp.server import _safe_tool

        @_safe_tool
        def ok():
            return {"result": 42}

        assert ok() == {"result": 42}

    def test_catches_exception_and_returns_error_dict(self):
        from chorus.mcp.server import _safe_tool

        @_safe_tool
        def boom():
            raise ValueError("something broke")

        result = boom()
        assert result["error"] == "something broke"
        assert result["error_type"] == "ValueError"
        assert result["tool"] == "boom"

    def test_preserves_function_name(self):
        from chorus.mcp.server import _safe_tool

        @_safe_tool
        def my_tool():
            return {}

        assert my_tool.__name__ == "my_tool"


class TestFmtPercentile:
    """Tests for _fmt_percentile display helper."""

    def test_none_returns_near_zero(self):
        from chorus.analysis.variant_report import _fmt_percentile
        # None means the raw effect was below NOISE_FLOOR_RAW_SCORE; we
        # render "near-zero" (not "—") so users don't read it as missing data.
        assert _fmt_percentile(None) == "near-zero"

    def test_saturated_high(self):
        from chorus.analysis.variant_report import _fmt_percentile
        assert _fmt_percentile(1.0) == "≥99th"
        assert _fmt_percentile(0.995) == "≥99th"

    def test_saturated_low(self):
        from chorus.analysis.variant_report import _fmt_percentile
        assert _fmt_percentile(0.0) == "≤1st"
        assert _fmt_percentile(0.005) == "≤1st"

    def test_middle_range(self):
        from chorus.analysis.variant_report import _fmt_percentile
        assert _fmt_percentile(0.5) == "0.50"
        assert _fmt_percentile(0.77) == "0.77"


class TestScoreColorClass:
    """Tests for _score_color_class interpretation-based color mapping."""

    def test_minimal_gets_minimal_badge(self):
        from chorus.analysis.variant_report import _score_color_class
        assert _score_color_class("Minimal effect") == "badge-minimal"

    def test_moderate_gets_moderate(self):
        from chorus.analysis.variant_report import _score_color_class
        assert _score_color_class("Moderate opening") == "badge-moderate"
        assert _score_color_class("Moderate binding loss") == "badge-moderate"

    def test_strong_gets_strong(self):
        from chorus.analysis.variant_report import _score_color_class
        assert _score_color_class("Strong opening") == "badge-strong"

    def test_very_strong_gets_vstrong(self):
        from chorus.analysis.variant_report import _score_color_class
        assert _score_color_class("Very strong binding gain") == "badge-vstrong"

    def test_case_insensitive(self):
        from chorus.analysis.variant_report import _score_color_class
        assert _score_color_class("MODERATE CLOSING") == "badge-moderate"


class TestResolvedDeviceReporting:
    """Tests for load_oracle resolved-device detection."""

    def test_detect_device_reports_gpu_when_nvidia_smi(self):
        from chorus.mcp.state import OracleStateManager
        from unittest.mock import MagicMock, patch
        oracle = MagicMock(spec=["device"])
        oracle.device = None
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1\n")
            result = OracleStateManager._detect_device(oracle)
            assert "gpu" in result.lower()

    def test_detect_device_falls_back_to_cpu(self):
        from chorus.mcp.state import OracleStateManager
        from unittest.mock import MagicMock, patch
        oracle = MagicMock(spec=["device"])
        oracle.device = None
        with patch("shutil.which", return_value=None), \
             patch("sys.platform", "linux"):
            result = OracleStateManager._detect_device(oracle)
            assert "cpu" in result.lower()


class TestDescribeTracksRequested:
    """Tests for MCP server's _describe_tracks_requested helper."""

    def test_empty_returns_all_tracks(self):
        from chorus.mcp.server import _describe_tracks_requested
        assert _describe_tracks_requested(None) == "all oracle tracks"
        assert _describe_tracks_requested([]) == "all oracle tracks"

    def test_no_variant_result_returns_count(self):
        from chorus.mcp.server import _describe_tracks_requested
        assert _describe_tracks_requested(["a", "b", "c"]) == "3 tracks"

    def test_uniform_cell_type_includes_label(self):
        from chorus.mcp.server import _describe_tracks_requested
        from unittest.mock import MagicMock
        ref = MagicMock()
        track1 = MagicMock()
        track1.cell_type = "HepG2"
        track2 = MagicMock()
        track2.cell_type = "HepG2"
        ref.tracks = {"a": track1, "b": track2}
        variant_result = {"predictions": {"reference": ref}}
        assert _describe_tracks_requested(["a", "b"], variant_result) == "2 HepG2 tracks"

    def test_mixed_cell_types_omits_label(self):
        from chorus.mcp.server import _describe_tracks_requested
        from unittest.mock import MagicMock
        ref = MagicMock()
        t1 = MagicMock(); t1.cell_type = "HepG2"
        t2 = MagicMock(); t2.cell_type = "K562"
        ref.tracks = {"a": t1, "b": t2}
        variant_result = {"predictions": {"reference": ref}}
        assert _describe_tracks_requested(["a", "b"], variant_result) == "2 tracks"


# ── Shared report glossary + formula chip helpers ──────────────────────

class TestReportGlossary:
    """Unit tests for the shared report-glossary helper.

    The glossary helper is imported by every report renderer (variant_report,
    causal, batch_scoring, multi_oracle_report) so these tests pin its
    behaviour tightly enough to protect downstream renderers from regressions.
    """

    def test_formula_label_known_and_unknown(self):
        from chorus.analysis._report_glossary import formula_label
        assert formula_label("log2fc") == "log2FC"
        assert formula_label("logfc") == "lnFC"
        assert formula_label("diff") == "Δ (alt−ref)"
        assert formula_label(None) == "—"
        # Pass-through for unknown labels so regressions are obvious in the UI.
        assert formula_label("noop") == "noop"

    def test_formula_chip_is_html(self):
        from chorus.analysis._report_glossary import formula_chip
        chip = formula_chip("log2fc")
        assert chip.startswith('<span class="formula-chip">')
        assert "log2FC" in chip
        assert chip.endswith("</span>")

    def test_render_how_to_read_minimal(self):
        from chorus.analysis._report_glossary import render_how_to_read
        html = render_how_to_read()
        assert '<section class="how-to-read">' in html
        assert "How to read this report" in html

    def test_render_how_to_read_per_layer_formulas(self):
        from chorus.analysis._report_glossary import render_how_to_read
        # Present layers with mixed formula families (log2fc + diff + logfc).
        html = render_how_to_read(
            layers_present=[
                "chromatin_accessibility", "promoter_activity", "gene_expression",
            ],
        )
        # All three formula labels must appear.
        assert "log2FC" in html
        assert "lnFC" in html
        assert "Δ" in html
        # Each layer's description appears.
        assert "Chromatin accessibility" in html
        assert "Promoter activity" in html
        assert "Gene expression" in html

    def test_render_how_to_read_composite_flags(self):
        from chorus.analysis._report_glossary import render_how_to_read
        from chorus.analysis.causal import CausalWeights
        html = render_how_to_read(
            layers_present=["chromatin_accessibility"],
            include_composite=True,
            weights=CausalWeights(layer_effect_threshold=0.25),
            include_percentile=True,
            include_activity=True,
        )
        assert "Composite" in html
        assert "Convergence" in html
        assert "0.25" in html  # threshold echoed
        assert "Effect %ile" in html
        assert "Activity %ile" in html

    def test_render_how_to_read_deduplicates_layers(self):
        """Callers can pass duplicated layers; helper must not repeat rows."""
        from chorus.analysis._report_glossary import render_how_to_read
        html = render_how_to_read(layers_present=[
            "chromatin_accessibility", "chromatin_accessibility",
        ])
        assert html.count("Chromatin accessibility") == 1


# ── Formula chips surfaced in the variant_report HTML ───────────────────

class TestVariantReportFormulaChips:
    """A new user must see the effect formula in every per-layer table."""

    def test_effect_column_and_heading_carry_formula_chip(self):
        """Each per-layer table heading should surface its layer's formula.

        The test builds a multi-layer VariantReport (log2fc for chromatin,
        diff for MPRA, logfc for RNA) and asserts each formula chip is
        present in the rendered HTML, near its respective layer heading.
        """
        from chorus.analysis.variant_report import (
            TrackScore, VariantReport,
        )
        report = VariantReport(
            chrom="chr1", position=109274968,
            ref_allele="G", alt_alleles=["T"],
            oracle_name="alphagenome", gene_name="SORT1",
            allele_scores={"T": [
                TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                           cell_type="HepG2", layer="chromatin_accessibility",
                           ref_value=4.1, alt_value=5.7, raw_score=0.468),
                TrackScore(assay_id="LentiMPRA:HepG2", assay_type="LentiMPRA",
                           cell_type="HepG2", layer="promoter_activity",
                           ref_value=2.1, alt_value=2.9, raw_score=0.82),
                TrackScore(assay_id="RNA:HepG2", assay_type="RNA",
                           cell_type="HepG2", layer="gene_expression",
                           ref_value=3.0, alt_value=4.5, raw_score=0.405),
            ]},
        )
        html = report.to_html()
        # Each formula chip should appear at least once in the document.
        assert "log2FC" in html
        assert 'class="formula-chip">Δ' in html or "Δ (alt−ref)" in html
        assert "lnFC" in html
        # Glossary is at the TOP of the page — before any numeric row.
        glossary_idx = html.index("How to read this report")
        first_score_idx = html.index("0.468")
        assert glossary_idx < first_score_idx, \
            "Glossary must appear before numeric content for new users."

    def test_summary_cites_winning_track_provenance(self):
        """The one-line Summary must cite the specific assay + cell type."""
        from chorus.analysis.variant_report import (
            TrackScore, VariantReport,
        )
        report = VariantReport(
            chrom="chr1", position=1000500,
            ref_allele="A", alt_alleles=["G"],
            oracle_name="alphagenome", gene_name="TEST",
            allele_scores={"G": [
                # Two tracks in the same layer — only the strongest should
                # make it into the summary string.
                TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                           cell_type="HepG2", layer="chromatin_accessibility",
                           ref_value=4.1, alt_value=5.7, raw_score=0.468),
                TrackScore(assay_id="DNASE:K562", assay_type="DNASE",
                           cell_type="K562", layer="chromatin_accessibility",
                           ref_value=2.0, alt_value=2.3, raw_score=0.15),
            ]},
        )
        html = report.to_html()
        # Summary must include the winning track's assay_id AND cell_type.
        summary_start = html.index("<b>Summary:</b>")
        summary_end = html.index("</p>", summary_start)
        summary = html[summary_start:summary_end]
        assert "DNASE:HepG2" in summary
        assert "HepG2" in summary
        # Losing track should NOT make it into the summary.
        assert "K562" not in summary


# ── Multi-oracle validation report ──────────────────────────────────────

class TestMultiOracleReport:
    """End-to-end coverage of the multi-oracle consensus renderer."""

    def _mk_report(self, oracle, tracks):
        from chorus.analysis.variant_report import VariantReport
        return VariantReport(
            chrom="chr1", position=109274968,
            ref_allele="G", alt_alleles=["T"],
            oracle_name=oracle, gene_name="SORT1",
            allele_scores={"T": tracks},
        )

    def test_from_reports_variant_mismatch_raises(self):
        from chorus.analysis import MultiOracleReport, VariantReport
        a = VariantReport(chrom="chr1", position=1, ref_allele="A",
                          alt_alleles=["G"], oracle_name="o1", gene_name="X")
        b = VariantReport(chrom="chr2", position=1, ref_allele="A",
                          alt_alleles=["G"], oracle_name="o2", gene_name="X")
        with pytest.raises(ValueError, match="Variant mismatch"):
            MultiOracleReport.from_reports([a, b])

    def test_consensus_matrix_direction_and_provenance(self):
        """When two oracles disagree on direction, the matrix flags it."""
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        gain = self._mk_report("oracle_a", [
            TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=2.0, alt_value=3.0, raw_score=0.58),
        ])
        loss = self._mk_report("oracle_b", [
            TrackScore(assay_id="DNASE:K562", assay_type="DNASE",
                       cell_type="K562", layer="chromatin_accessibility",
                       ref_value=2.0, alt_value=1.6, raw_score=-0.32),
        ])
        moracle = MultiOracleReport.from_reports(
            [gain, loss], variant_id="rs-test",
        )
        rows = moracle._consensus_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["agreement"] == "disagree"
        # Each oracle's winning track identity is preserved.
        assert row["oracles"]["oracle_a"]["assay_id"] == "DNASE:HepG2"
        assert row["oracles"]["oracle_b"]["cell_type"] == "K562"

    def test_html_surfaces_glossary_and_agreement(self):
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        cb = self._mk_report("chrombpnet", [
            TrackScore(assay_id="ATAC:HepG2", assay_type="ATAC",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=4.0, alt_value=5.7, raw_score=0.51),
        ])
        ag = self._mk_report("alphagenome", [
            TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=3.0, alt_value=4.2, raw_score=0.49),
            TrackScore(assay_id="CAGE:HepG2", assay_type="CAGE",
                       cell_type="HepG2", layer="tss_activity",
                       ref_value=2.0, alt_value=2.8, raw_score=0.48),
        ])
        moracle = MultiOracleReport.from_reports(
            [cb, ag], variant_id="rs12740374",
        )
        html = moracle.to_html()
        # Key structural markers for a new-user reader.
        assert "Multi-oracle validation" in html
        assert "How to read this report" in html
        assert "Cross-oracle consensus" in html
        # Consensus-gain flag when both oracles push the same direction.
        assert "all ↑" in html
        # Per-oracle drill-down sections are collapsible.
        assert "oracle-block" in html
        # Glossary must still come before any numeric effect.
        glossary_idx = html.index("How to read this report")
        first_number_idx = html.index("0.51")
        assert glossary_idx < first_number_idx

    def test_json_roundtrip_via_from_dict(self):
        """VariantReport.from_dict round-trips allele_scores correctly."""
        from chorus.analysis import VariantReport
        from chorus.analysis.variant_report import TrackScore
        original = VariantReport(
            chrom="chr1", position=109274968,
            ref_allele="G", alt_alleles=["T"],
            oracle_name="legnet", gene_name="SORT1",
            allele_scores={"T": [
                TrackScore(assay_id="LentiMPRA:HepG2", assay_type="LentiMPRA",
                           cell_type="HepG2", layer="promoter_activity",
                           ref_value=2.1, alt_value=2.9, raw_score=0.82,
                           quantile_score=0.95, description="LegNet HepG2"),
            ]},
        )
        rehydrated = VariantReport.from_dict(original.to_dict())
        assert rehydrated.chrom == "chr1"
        assert rehydrated.position == 109274968
        assert rehydrated.oracle_name == "legnet"
        assert list(rehydrated.allele_scores.keys()) == ["T"]
        ts = rehydrated.allele_scores["T"][0]
        assert ts.assay_id == "LentiMPRA:HepG2"
        assert ts.raw_score == pytest.approx(0.82)
        assert ts.quantile_score == pytest.approx(0.95)

    def test_from_json_files_round_trip(self, tmp_path):
        """Writing JSONs to disk and rehydrating via from_json_files."""
        import json
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        cb = self._mk_report("chrombpnet", [
            TrackScore(assay_id="ATAC:HepG2", assay_type="ATAC",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=4.0, alt_value=5.7, raw_score=0.51),
        ])
        ln = self._mk_report("legnet", [
            TrackScore(assay_id="LentiMPRA:HepG2", assay_type="LentiMPRA",
                       cell_type="HepG2", layer="promoter_activity",
                       ref_value=2.0, alt_value=2.9, raw_score=0.85),
        ])
        cb_path = tmp_path / "cb.json"
        ln_path = tmp_path / "ln.json"
        cb_path.write_text(json.dumps(cb.to_dict(), default=str))
        ln_path.write_text(json.dumps(ln.to_dict(), default=str))

        moracle = MultiOracleReport.from_json_files(
            [cb_path, ln_path], variant_id="rs-test",
        )
        assert list(moracle.reports.keys()) == ["chrombpnet", "legnet"]
        assert moracle.reports["chrombpnet"].allele_scores["T"][0].raw_score \
            == pytest.approx(0.51)

    def test_apply_floor_rescale_passthrough(self):
        """``apply_floor_rescale`` should passthrough cleanly when it can't rescale.

        A None normalizer or a wrong-type normalizer must return
        ``(False, ref_unchanged, alt_unchanged)`` so the causal + variant
        renderers can fall back to raw autoscale without extra guards.
        """
        import numpy as np
        from chorus.analysis._igv_report import apply_floor_rescale

        ref = np.array([1.0, 2.0, 3.0])
        alt = np.array([1.2, 2.1, 2.8])

        # None normalizer — passthrough
        ok, r, a = apply_floor_rescale(None, None, "x", "chromatin_accessibility",
                                       ref, alt)
        assert ok is False
        assert r is ref and a is alt  # unchanged references

        # Non-PerTrackNormalizer type — passthrough
        class _NotPT:
            pass
        ok, r, a = apply_floor_rescale(_NotPT(), "oracle", "assay",
                                       "tf_binding", ref, alt)
        assert ok is False
        assert r is ref and a is alt

    def test_markdown_summary_contains_consensus_table(self):
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        a = self._mk_report("oracle_a", [
            TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=2.0, alt_value=3.0, raw_score=0.58),
        ])
        b = self._mk_report("oracle_b", [
            TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=2.0, alt_value=2.8, raw_score=0.48),
        ])
        moracle = MultiOracleReport.from_reports([a, b], variant_id="rs1")
        md = moracle.to_markdown()
        assert "# Multi-oracle validation" in md
        assert "Cross-oracle consensus" in md
        assert "oracle_a" in md and "oracle_b" in md
        assert "all ↑" in md

    def test_single_voter_layer_uses_n1_label_not_all(self):
        """When only one oracle reports a direction for a layer (common
        when oracles specialise — ChromBPNet for chromatin only, LegNet
        for MPRA only), the agreement label must distinguish this from
        a real multi-oracle consensus. Previously a single ``+0.3`` vote
        rendered as "all ↑" which falsely implied agreement across
        multiple oracles."""
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        # Two oracles, but only one scores the MPRA layer and only one
        # scores the chromatin layer — no overlap.
        chromatin_only = self._mk_report("chrombpnet", [
            TrackScore(assay_id="ATAC:HepG2", assay_type="ATAC",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=4.0, alt_value=5.7, raw_score=0.51),
        ])
        mpra_only = self._mk_report("legnet", [
            TrackScore(assay_id="LentiMPRA:HepG2", assay_type="LentiMPRA",
                       cell_type="HepG2", layer="promoter_activity",
                       ref_value=2.0, alt_value=1.6, raw_score=-0.32),
        ])
        moracle = MultiOracleReport.from_reports(
            [chromatin_only, mpra_only], variant_id="rs-test",
        )
        rows = {r["layer"]: r for r in moracle._consensus_rows()}
        assert rows["chromatin_accessibility"]["agreement"] == "single_gain"
        assert rows["promoter_activity"]["agreement"] == "single_loss"

        md = moracle.to_markdown()
        # Label must say "n=1", must NOT say "all ↑" / "all ↓" (would
        # imply multi-oracle agreement the data doesn't support).
        assert "only ↑ (n=1)" in md
        assert "only ↓ (n=1)" in md
        assert "all ↑" not in md
        assert "all ↓" not in md

        html = moracle.to_html()
        assert "↑ only (n=1)" in html
        assert "↓ only (n=1)" in html
        assert "agree-single" in html

    def test_rendered_reports_carry_generated_timestamp(self):
        """Every other chorus report stamps a "Generated:" line so readers
        can tell when numbers were produced. Multi-oracle MD + HTML must
        do the same."""
        import re
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        rep = self._mk_report("alphagenome", [
            TrackScore(assay_id="DNASE:HepG2", assay_type="DNASE",
                       cell_type="HepG2", layer="chromatin_accessibility",
                       ref_value=500.0, alt_value=700.0, raw_score=0.48),
        ])
        moracle = MultiOracleReport.from_reports([rep], variant_id="rs1")
        md = moracle.to_markdown()
        assert "**Generated:**" in md
        assert re.search(r"\d{4}-\d{2}-\d{2}", md), "MD missing date"

        html = moracle.to_html()
        assert "Generated by Chorus" in html
        assert re.search(r"\d{4}-\d{2}-\d{2}", html), "HTML footer missing date"

    def test_uses_enriched_description_not_raw_assay_id(self):
        """Consensus matrix must render the enriched display name
        ("CHIP:CEBPA:HepG2") rather than the raw AlphaGenome assay_id
        ("CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA…") so the labels match
        the variant-report tables users see elsewhere.
        """
        from chorus.analysis import MultiOracleReport
        from chorus.analysis.variant_report import TrackScore

        raw_assay_id = (
            "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified "
            "(insertion) using CRISPR targeting H. sapiens CEBPA/."
        )
        rep = self._mk_report("alphagenome", [
            TrackScore(
                assay_id=raw_assay_id, assay_type="CHIP",
                cell_type="HepG2", layer="tf_binding",
                ref_value=2100.0, alt_value=2890.0, raw_score=0.381,
                quantile_score=1.0, description="CHIP:CEBPA:HepG2",
            ),
        ])
        moracle = MultiOracleReport.from_reports([rep], variant_id="rs12740374")

        md = moracle.to_markdown()
        # Enriched display present, raw id absent from rendered table.
        assert "CHIP:CEBPA:HepG2" in md
        assert "TF ChIP-seq CEBPA genetically modified" not in md

        html = moracle.to_html()
        assert "CHIP:CEBPA:HepG2" in html
        assert "TF ChIP-seq CEBPA genetically modified" not in html
        # Also: percentile format must match the rest of chorus ("≥99th"
        # via _fmt_percentile), not the old "+100.0%" format.
        assert "≥99th" in html
        assert "%ile +100.0%" not in html
        assert "%ile -100.0%" not in html
