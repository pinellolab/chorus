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
        assert "H3K27ac:K562" in md
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

        # Quantile takes precedence over raw score thresholds
        result = _interpret_score(0.5, 0.95, "tss_activity")
        assert "Very strong" in result
        assert "increase" in result

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
            assert "Quantile" in html  # quantile column header

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
        # Has the variant ROI
        assert '"name":"Variant"' in html or "'name':'Variant'" in html


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
        assert "Multi-Layer" in md
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
        assert d["scores"][0]["cell_type"] == "K562"
        assert d["scores"][0]["max_quantile"] == 0.95

    def test_batch_to_markdown_includes_gene_cell_type(self):
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
        md = result.to_markdown()
        assert "Gene" in md
        assert "Cell Type" in md
        assert "SORT1" in md
        assert "K562" in md

    def test_batch_to_dataframe_includes_new_columns(self):
        from chorus.analysis.batch_scoring import BatchVariantScore, BatchResult

        scores = [
            BatchVariantScore(
                chrom="chr1", position=100, ref="A", alt="G",
                variant_id="rs1", max_effect=0.5,
                top_layer="chromatin_accessibility", top_track="DNASE:K562",
                per_layer_scores={"chromatin_accessibility": 0.5},
                gene_name="SORT1", cell_type="K562", max_quantile=0.9,
            ),
        ]
        result = BatchResult(scores=scores)
        df = result.to_dataframe()
        assert "gene_name" in df.columns
        assert "cell_type" in df.columns
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

    def test_causal_to_markdown_includes_gene_cell_type(self):
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
        assert "Gene" in md
        assert "Cell Type" in md
        assert "SORT1" in md
        assert "K562" in md

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
        assert result.scores[0].cell_type == "K562"


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
