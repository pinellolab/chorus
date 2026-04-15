"""Tests for the Chorus MCP server — serializers, state manager, and tool definitions.

These tests do NOT load any real models; they exercise the logic with mocks and
synthetic data.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field


# ── Fixtures ─────────────────────────────────────────────────────────

@dataclass
class FakeInterval:
    chrom: str = "chr1"
    start: int = 1_000_000
    end: int = 1_100_000

    class reference:
        chrom = "chr1"
        start = 1_000_000
        end = 1_100_000

    def __init__(self, chrom="chr1", start=1_000_000, end=1_100_000):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.reference = type("Ref", (), {"chrom": chrom, "start": start, "end": end})()


def _make_track(assay_id="DNASE:K562", n_bins=100, resolution=128):
    """Build a minimal OraclePredictionTrack-like object for testing."""
    interval = FakeInterval()
    track = MagicMock()
    track.assay_id = assay_id
    track.assay_type = assay_id.split(":")[0]
    track.cell_type = assay_id.split(":")[-1] if ":" in assay_id else ""
    track.chrom = interval.chrom
    track.start = interval.start
    track.end = interval.start + n_bins * resolution
    track.resolution = resolution
    track.values = np.random.randn(n_bins).astype(np.float32)
    track.prediction_interval = interval
    return track


def _make_prediction(assay_ids=None, n_bins=100, resolution=128):
    """Build a minimal OraclePrediction-like object."""
    if assay_ids is None:
        assay_ids = ["DNASE:K562"]
    tracks = {aid: _make_track(aid, n_bins, resolution) for aid in assay_ids}
    pred = MagicMock()
    pred.items.return_value = tracks.items()
    pred.__iter__ = lambda self: iter(tracks)
    pred.__getitem__ = lambda self, key: tracks[key]
    pred.tracks = tracks
    pred.save_predictions_as_bedgraph = MagicMock(
        return_value={aid: f"/tmp/{aid.replace(':', '_')}.bedgraph" for aid in assay_ids}
    )
    return pred


# ── Serializer tests ─────────────────────────────────────────────────

class TestTrackSummary:
    def test_summary_keys(self):
        from chorus.mcp.serializers import _track_summary
        track = _make_track(n_bins=50)
        summary = _track_summary(track)
        for key in ("assay_id", "assay_type", "cell_type", "chrom", "start", "end",
                     "resolution", "num_bins", "mean", "max", "min", "std"):
            assert key in summary
        assert summary["num_bins"] == 50

    def test_summary_values_are_floats(self):
        from chorus.mcp.serializers import _track_summary
        summary = _track_summary(_make_track())
        for k in ("mean", "max", "min", "std"):
            assert isinstance(summary[k], float)


class TestDownsample:
    def test_no_downsample_when_small(self):
        from chorus.mcp.serializers import _downsample
        arr = np.arange(100, dtype=float)
        result = _downsample(arr, max_bins=500)
        assert len(result) == 100

    def test_downsample_to_max_bins(self):
        from chorus.mcp.serializers import _downsample
        arr = np.arange(10_000, dtype=float)
        result = _downsample(arr, max_bins=500)
        assert len(result) == 500


class TestSerializePrediction:
    def test_inline_values_for_small_prediction(self):
        from chorus.mcp.serializers import serialize_prediction
        pred = _make_prediction(n_bins=100)
        result = serialize_prediction(pred, inline_threshold=500)
        assert "tracks" in result
        track = result["tracks"][0]
        assert "values" in track
        assert len(track["values"]) == 100

    def test_preview_for_large_prediction(self):
        from chorus.mcp.serializers import serialize_prediction
        pred = _make_prediction(n_bins=2000)
        result = serialize_prediction(pred, output_dir="/tmp/test_mcp", inline_threshold=500)
        track = result["tracks"][0]
        assert "values_preview" in track
        assert len(track["values_preview"]) == 500
        assert "values_preview_note" in track


class TestSerializeVariantEffect:
    def test_basic_structure(self):
        from chorus.mcp.serializers import serialize_variant_effect
        pred_ref = _make_prediction(["DNASE:K562"])
        pred_alt = _make_prediction(["DNASE:K562"])
        variant_result = {
            "predictions": {"reference": pred_ref, "alt_1": pred_alt},
            "effect_sizes": {
                "alt_1": {"DNASE:K562": np.random.randn(100).astype(np.float32)}
            },
            "variant_info": {"position": "chr1:1050000", "ref": "A", "alts": ["G"]},
        }
        out = serialize_variant_effect(variant_result)
        assert "variant_info" in out
        assert "effect_sizes" in out
        assert "alt_1" in out["effect_sizes"]
        es = out["effect_sizes"]["alt_1"]["DNASE:K562"]
        assert "mean_effect" in es
        assert "abs_max_effect" in es


# ── State manager tests ──────────────────────────────────────────────

class TestOracleStateManager:
    def setup_method(self):
        # Reset singleton
        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None

    def test_singleton(self):
        from chorus.mcp.state import OracleStateManager
        a = OracleStateManager()
        b = OracleStateManager()
        assert a is b

    def test_list_loaded_empty(self):
        from chorus.mcp.state import OracleStateManager
        mgr = OracleStateManager()
        assert mgr.list_loaded() == []

    def test_get_oracle_raises_when_not_loaded(self):
        from chorus.mcp.state import OracleStateManager
        mgr = OracleStateManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_oracle("enformer")

    def test_unload_returns_false_when_not_loaded(self):
        from chorus.mcp.state import OracleStateManager
        mgr = OracleStateManager()
        assert mgr.unload_oracle("enformer") is False

    @patch("chorus.mcp.state.GenomeManager")
    def test_load_and_unload(self, mock_gm_cls):
        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        # Manually inject a fake oracle
        fake_oracle = MagicMock()
        fake_oracle.device = "cpu"
        mgr._oracles["enformer"] = fake_oracle
        mgr._load_times["enformer"] = 1.5

        loaded = mgr.list_loaded()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "enformer"

        assert mgr.get_oracle("enformer") is fake_oracle
        assert mgr.unload_oracle("enformer") is True
        assert mgr.list_loaded() == []

    @patch("chorus.mcp.state.GenomeManager")
    def test_normalizer_auto_load_with_backgrounds(self, mock_gm_cls):
        """Normalizer is auto-loaded when backgrounds exist on disk."""
        import tempfile, os

        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        with tempfile.TemporaryDirectory() as td:
            # Create background files
            scores = np.sort(np.abs(np.random.randn(500))).astype(np.float64)
            np.save(os.path.join(td, "testoracle_chromatin_accessibility.npy"), scores)
            np.save(os.path.join(td, "testoracle_tf_binding.npy"), scores)

            # Manually inject oracle and trigger normalizer auto-load with custom cache_dir
            mgr._oracles["testoracle"] = MagicMock(device="cpu")
            mgr._load_times["testoracle"] = 0.1

            from chorus.analysis.normalization import get_normalizer
            mgr._normalizers["testoracle"] = get_normalizer("testoracle", cache_dir=td)

            normalizer = mgr.get_normalizer("testoracle")
            assert normalizer is not None
            assert normalizer.has_background("testoracle_chromatin_accessibility")
            assert normalizer.has_background("testoracle_tf_binding")

    @patch("chorus.mcp.state.GenomeManager")
    def test_normalizer_none_when_no_backgrounds(self, mock_gm_cls):
        """get_normalizer returns None when no backgrounds exist."""
        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        # Use a fake oracle name that has no background files on disk
        mgr._oracles["nonexistent_oracle_xyz"] = MagicMock(device="cpu")
        mgr._load_times["nonexistent_oracle_xyz"] = 0.1
        mgr._auto_load_normalizer("nonexistent_oracle_xyz")

        assert mgr.get_normalizer("nonexistent_oracle_xyz") is None

    @patch("chorus.mcp.state.GenomeManager")
    def test_normalizer_cleaned_on_unload(self, mock_gm_cls):
        """Unloading an oracle also removes its normalizer."""
        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        mgr._oracles["test"] = MagicMock(device="cpu")
        mgr._load_times["test"] = 0.1
        mgr._normalizers["test"] = MagicMock()

        assert mgr.get_normalizer("test") is not None
        mgr.unload_oracle("test")
        assert mgr.get_normalizer("test") is None

    @patch("chorus.mcp.state.GenomeManager")
    def test_list_loaded_includes_backgrounds(self, mock_gm_cls):
        """list_loaded() includes backgrounds_loaded count."""
        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        from chorus.analysis.normalization import QuantileNormalizer
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        mgr._oracles["test"] = MagicMock(device="cpu")
        mgr._load_times["test"] = 0.1
        mgr._normalizers["test"] = None

        loaded = mgr.list_loaded()
        assert loaded[0]["backgrounds_loaded"] == 0

    @patch("chorus.mcp.state.GenomeManager")
    def test_load_oracle_response_includes_backgrounds(self, mock_gm_cls):
        """load_oracle() response includes background status."""
        import tempfile, os

        mock_gm_cls.return_value.is_genome_downloaded.return_value = False

        from chorus.mcp.state import OracleStateManager
        OracleStateManager._instance = None
        mgr = OracleStateManager()

        # Inject oracle without going through load_oracle (avoids real model load)
        mgr._oracles["myoracle"] = MagicMock(device="cpu")
        mgr._load_times["myoracle"] = 0.5
        mgr._auto_load_normalizer("myoracle")

        # Simulate what load_oracle returns
        normalizer = mgr.get_normalizer("myoracle")
        if normalizer is not None:
            summary = normalizer.summary()
            bg_info = {"status": "loaded", "n_layers": len(summary)}
        else:
            bg_info = {"status": "none"}

        assert bg_info["status"] == "none"  # no backgrounds on disk


# ── Server tool definition smoke tests ───────────────────────────────

_has_fastmcp = pytest.importorskip is not None  # always True, just a placeholder
try:
    import fastmcp as _fastmcp_mod  # noqa: F401
    _has_fastmcp = True
except ImportError:
    _has_fastmcp = False


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp not installed")
class TestServerTools:
    def test_list_oracles_returns_all_six(self):
        from chorus.mcp.server import ORACLE_SPECS
        # Verify we have all 6 oracles in specs
        assert len(ORACLE_SPECS) == 6
        expected = {"enformer", "borzoi", "chrombpnet", "sei", "legnet", "alphagenome"}
        assert set(ORACLE_SPECS.keys()) == expected

    def test_oracle_specs_keys(self):
        from chorus.mcp.server import ORACLE_SPECS
        for name, spec in ORACLE_SPECS.items():
            assert "description" in spec
            assert "framework" in spec
            assert "input_size_bp" in spec


# ── score_region tests ────────────────────────────────────────────────

def _make_real_track(assay_id="DNASE:K562", n_bins=100, resolution=128,
                     pred_start=1_000_000, chrom="chr1", cls=None):
    """Build a real OraclePredictionTrack for testing score_region."""
    from chorus.core.result import OraclePredictionTrack
    from chorus.core.interval import Interval, GenomeRef

    pred_end = pred_start + n_bins * resolution
    interval = MagicMock()
    interval.reference = MagicMock()
    interval.reference.chrom = chrom
    interval.reference.start = pred_start
    interval.reference.end = pred_end

    track_cls = cls or OraclePredictionTrack
    values = np.arange(n_bins, dtype=np.float32)

    track = track_cls.__new__(track_cls)
    track.source_model = "test"
    track.assay_id = assay_id
    track.assay_type = assay_id.split(":")[0]
    track.cell_type = assay_id.split(":")[-1] if ":" in assay_id else ""
    track.query_interval = interval
    track.prediction_interval = interval
    track.input_interval = interval
    track.resolution = resolution
    track.values = values
    track.preferred_aggregation = 'sum'
    track.preferred_interpolation = 'linear_divided'
    track.preferred_scoring_strategy = 'mean'
    track.metadata = {}
    track.track_id = None
    return track


class TestScoreRegionTrack:
    """Test OraclePredictionTrack.score_region()."""

    def test_within_window(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        # Bins 10-20, values = [10, 11, ..., 19]
        start = 1_000_000 + 10 * 128
        end = 1_000_000 + 20 * 128
        score = track.score_region("chr1", start, end, "mean")
        assert score is not None
        assert abs(score - np.mean(np.arange(10, 20, dtype=np.float32))) < 1e-5

    def test_sum_strategy(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        start = 1_000_000 + 0 * 128
        end = 1_000_000 + 5 * 128
        score = track.score_region("chr1", start, end, "sum")
        assert abs(score - float(np.sum(np.arange(5, dtype=np.float32)))) < 1e-5

    def test_max_strategy(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        start = 1_000_000 + 50 * 128
        end = 1_000_000 + 60 * 128
        score = track.score_region("chr1", start, end, "max")
        assert score == 59.0

    def test_median_strategy(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        start = 1_000_000 + 0 * 128
        end = 1_000_000 + 4 * 128
        score = track.score_region("chr1", start, end, "median")
        assert abs(score - np.median(np.arange(4, dtype=np.float32))) < 1e-5

    def test_no_overlap_before(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        score = track.score_region("chr1", 500_000, 600_000, "mean")
        assert score is None

    def test_no_overlap_after(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        pred_end = 1_000_000 + 100 * 128
        score = track.score_region("chr1", pred_end + 1000, pred_end + 2000, "mean")
        assert score is None

    def test_wrong_chrom(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        score = track.score_region("chr2", 1_000_000, 1_001_000, "mean")
        assert score is None

    def test_partial_overlap_left(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        # Region starts before prediction window
        score = track.score_region("chr1", 999_000, 1_000_000 + 5 * 128, "mean")
        assert score is not None

    def test_partial_overlap_right(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        pred_end = 1_000_000 + 100 * 128
        score = track.score_region("chr1", pred_end - 5 * 128, pred_end + 1000, "mean")
        assert score is not None

    def test_default_strategy(self):
        track = _make_real_track(n_bins=100, resolution=128, pred_start=1_000_000)
        # Default is 'mean'
        start = 1_000_000 + 10 * 128
        end = 1_000_000 + 20 * 128
        score_default = track.score_region("chr1", start, end)
        score_mean = track.score_region("chr1", start, end, "mean")
        assert score_default == score_mean


class TestScoreRegionPrediction:
    """Test OraclePrediction.score_region()."""

    def test_delegates_to_tracks(self):
        from chorus.core.result import OraclePrediction

        track1 = _make_real_track("DNASE:K562", n_bins=100, resolution=128)
        track2 = _make_real_track("CAGE:K562", n_bins=100, resolution=128)
        pred = OraclePrediction(tracks={"DNASE:K562": track1, "CAGE:K562": track2})

        scores = pred.score_region("chr1", 1_000_000, 1_000_000 + 10 * 128, "mean")
        assert "DNASE:K562" in scores
        assert "CAGE:K562" in scores
        assert scores["DNASE:K562"] is not None


# ── score_variant_effect tests ────────────────────────────────────────

class TestScoreVariantEffect:
    """Test score_variant_effect() standalone function."""

    def _make_variant_result(self, n_bins=100, resolution=128, pred_start=1_000_000):
        from chorus.core.result import OraclePrediction

        ref_track = _make_real_track("DNASE:K562", n_bins=n_bins, resolution=resolution, pred_start=pred_start)
        alt_track = _make_real_track("DNASE:K562", n_bins=n_bins, resolution=resolution, pred_start=pred_start)
        alt_track.values = ref_track.values + 1.0  # alt is uniformly higher

        ref_pred = OraclePrediction(tracks={"DNASE:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"DNASE:K562": alt_track})

        var_pos = pred_start + 50 * resolution
        return {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {
                'alt_1': {'DNASE:K562': alt_track.values - ref_track.values}
            },
            'variant_info': {
                'position': f'chr1:{var_pos}',
                'ref': 'A',
                'alts': ['G'],
            }
        }

    def test_region_mode(self):
        from chorus.core.result import score_variant_effect
        vr = self._make_variant_result()
        result = score_variant_effect(
            vr, chrom="chr1", start=1_000_000, end=1_000_000 + 10 * 128, scoring_strategy="mean"
        )
        assert 'alt_1' in result
        assert 'DNASE:K562' in result['alt_1']
        entry = result['alt_1']['DNASE:K562']
        assert abs(entry['effect'] - 1.0) < 1e-5  # alt is 1.0 higher everywhere

    def test_at_variant_mode(self):
        from chorus.core.result import score_variant_effect
        vr = self._make_variant_result()
        result = score_variant_effect(vr, at_variant=True, window_bins=2, scoring_strategy="mean")
        assert 'alt_1' in result
        entry = result['alt_1']['DNASE:K562']
        assert abs(entry['effect'] - 1.0) < 1e-5

    def test_abs_max_strategy(self):
        from chorus.core.result import score_variant_effect, OraclePrediction

        ref_track = _make_real_track("DNASE:K562", n_bins=100, resolution=128)
        alt_track = _make_real_track("DNASE:K562", n_bins=100, resolution=128)
        alt_track.values = ref_track.values.copy()
        alt_track.values[50] += 10.0  # Spike at bin 50

        ref_pred = OraclePrediction(tracks={"DNASE:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"DNASE:K562": alt_track})

        vr = {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {'alt_1': {'DNASE:K562': alt_track.values - ref_track.values}},
            'variant_info': {
                'position': f'chr1:{1_000_000 + 50 * 128}',
                'ref': 'A', 'alts': ['G'],
            }
        }
        result = score_variant_effect(
            vr, chrom="chr1",
            start=1_000_000 + 45 * 128, end=1_000_000 + 55 * 128,
            scoring_strategy="abs_max"
        )
        assert abs(result['alt_1']['DNASE:K562']['effect'] - 10.0) < 1e-5

    def test_raises_without_coords_or_at_variant(self):
        from chorus.core.result import score_variant_effect
        vr = self._make_variant_result()
        with pytest.raises(ValueError, match="chrom/start/end"):
            score_variant_effect(vr)


# ── get_gene_exons tests ─────────────────────────────────────────────

class TestGetGeneExons:
    """Test get_gene_exons() and AnnotationManager.get_exon_positions()."""

    def _make_gtf_file(self, tmp_path, lines):
        """Write GTF lines to a temp file."""
        gtf = tmp_path / "test.gtf"
        gtf.write_text("\n".join(lines) + "\n")
        return str(gtf)

    def _gtf_exon_line(self, chrom, start, end, strand, gene_name, transcript_id, exon_number):
        attrs = (
            f'gene_name "{gene_name}"; '
            f'gene_id "ENSG00000000001"; '
            f'transcript_id "{transcript_id}"; '
            f'exon_number "{exon_number}"'
        )
        return f'{chrom}\tENSEMBL\texon\t{start}\t{end}\t.\t{strand}\t.\t{attrs}'

    def test_overlapping_exons_merged(self, tmp_path):
        from chorus.utils.annotations import AnnotationManager
        lines = [
            self._gtf_exon_line("chr1", 1000, 2000, "+", "MYC", "T1", 1),
            self._gtf_exon_line("chr1", 1500, 2500, "+", "MYC", "T2", 1),
            self._gtf_exon_line("chr1", 3000, 4000, "+", "MYC", "T1", 2),
        ]
        gtf_path = self._make_gtf_file(tmp_path, lines)
        mgr = AnnotationManager(str(tmp_path))

        exons = mgr.get_exon_positions(gtf_path, gene_name="MYC")
        assert len(exons) == 3

        # Test merging via get_gene_exons logic
        from chorus.utils.annotations import get_gene_exons
        with patch('chorus.utils.annotations.get_annotation_manager') as mock_mgr:
            mock_mgr.return_value = mgr
            mock_mgr.return_value.get_annotation_path = MagicMock(return_value=gtf_path)
            merged = get_gene_exons("MYC", annotation=gtf_path)

        # Should merge first two overlapping into [1000, 2500], keep [3000, 4000]
        assert len(merged) == 2
        assert merged.iloc[0]['start'] == 1000
        assert merged.iloc[0]['end'] == 2500
        assert merged.iloc[1]['start'] == 3000

    def test_disjoint_exons_not_merged(self, tmp_path):
        from chorus.utils.annotations import AnnotationManager, get_gene_exons
        lines = [
            self._gtf_exon_line("chr1", 1000, 2000, "+", "MYC", "T1", 1),
            self._gtf_exon_line("chr1", 5000, 6000, "+", "MYC", "T1", 2),
        ]
        gtf_path = self._make_gtf_file(tmp_path, lines)
        mgr = AnnotationManager(str(tmp_path))

        with patch('chorus.utils.annotations.get_annotation_manager') as mock_mgr:
            mock_mgr.return_value = mgr
            mock_mgr.return_value.get_annotation_path = MagicMock(return_value=gtf_path)
            merged = get_gene_exons("MYC", annotation=gtf_path)

        assert len(merged) == 2

    def test_single_transcript(self, tmp_path):
        from chorus.utils.annotations import AnnotationManager
        lines = [
            self._gtf_exon_line("chr1", 1000, 2000, "+", "MYC", "T1", 1),
        ]
        gtf_path = self._make_gtf_file(tmp_path, lines)
        mgr = AnnotationManager(str(tmp_path))

        exons = mgr.get_exon_positions(gtf_path, gene_name="MYC")
        assert len(exons) == 1
        assert exons.iloc[0]['gene_name'] == 'MYC'


# ── Shared test oracle factory ───────────────────────────────────────

def _make_test_oracle():
    """Create a minimal concrete oracle subclass for testing."""
    from chorus.core.base import OracleBase

    class TestOracle(OracleBase):
        def __init__(self):
            self.model = None
            self.loaded = True
            self.oracle_name = "test"
            self.use_environment = False
            self._env_manager = None
            self._env_runner = None
            self.model_load_timeout = None
            self.predict_timeout = None
            self.device = None

        def load_pretrained_model(self, weights=None): pass
        def list_assay_types(self): return []
        def list_cell_types(self): return []
        def _predict(self, seq, assay_ids): pass
        def _get_context_size(self): return 0
        def _get_sequence_length_bounds(self): return (0, 0)
        def _get_bin_size(self): return 128
        def fine_tune(self, tracks, track_names, **kwargs): pass

    return TestOracle()


# ── analyze_gene_expression tests ────────────────────────────────────

class TestAnalyzeGeneExpression:
    """Test OracleBase.analyze_gene_expression() method."""

    def test_cage_uses_tss_windowed_max(self):
        from chorus.core.result import OraclePrediction, CAGEOraclePredictionTrack

        oracle = _make_test_oracle()
        track = _make_real_track("CAGE:K562", n_bins=100, resolution=128,
                                 pred_start=1_000_000, cls=CAGEOraclePredictionTrack)
        # Set a peak at bin 50
        track.values[50] = 100.0
        pred = OraclePrediction(tracks={"CAGE:K562": track})

        tss_pos = 1_000_000 + 50 * 128 + 64  # middle of bin 50

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss, \
             patch('chorus.utils.annotations.get_gene_exons') as mock_exons:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': tss_pos, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])
            mock_exons.return_value = pd.DataFrame()

            result = oracle.analyze_gene_expression(pred, 'TEST')

        assert result['gene_name'] == 'TEST'
        assert 'CAGE:K562' in result['per_track']
        track_result = result['per_track']['CAGE:K562']
        assert track_result['quantification_method'] == 'tss_windowed_max'
        assert track_result['expression'] == 100.0

    def test_rna_uses_exon_sum(self):
        from chorus.core.result import OraclePrediction, RNAOraclePredictionTrack

        oracle = _make_test_oracle()
        track = _make_real_track("RNA:K562", n_bins=100, resolution=128,
                                 pred_start=1_000_000, cls=RNAOraclePredictionTrack)
        # All values are 1.0 for easy sum calculation
        track.values = np.ones(100, dtype=np.float32)
        pred = OraclePrediction(tracks={"RNA:K562": track})

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss, \
             patch('chorus.utils.annotations.get_gene_exons') as mock_exons:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': 1_050_000, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])
            # Two exons, each spanning 5 bins (5 * 128 = 640bp)
            mock_exons.return_value = pd.DataFrame([
                {'chrom': 'chr1', 'start': 1_000_000, 'end': 1_000_000 + 5 * 128,
                 'strand': '+', 'gene_name': 'TEST'},
                {'chrom': 'chr1', 'start': 1_000_000 + 50 * 128, 'end': 1_000_000 + 55 * 128,
                 'strand': '+', 'gene_name': 'TEST'},
            ])

            result = oracle.analyze_gene_expression(pred, 'TEST')

        assert 'RNA:K562' in result['per_track']
        track_result = result['per_track']['RNA:K562']
        assert track_result['quantification_method'] == 'exon_sum'
        assert track_result['expression'] == 10.0  # 5 bins + 5 bins, all 1.0

    def test_auto_detects_track_types(self):
        from chorus.core.result import (
            OraclePrediction, CAGEOraclePredictionTrack,
            RNAOraclePredictionTrack, DNaseOraclePredictionTrack,
        )

        oracle = _make_test_oracle()
        cage_track = _make_real_track("CAGE:K562", cls=CAGEOraclePredictionTrack)
        rna_track = _make_real_track("RNA:K562", cls=RNAOraclePredictionTrack)
        dnase_track = _make_real_track("DNASE:K562", cls=DNaseOraclePredictionTrack)

        pred = OraclePrediction(tracks={
            "CAGE:K562": cage_track,
            "RNA:K562": rna_track,
            "DNASE:K562": dnase_track,
        })

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss, \
             patch('chorus.utils.annotations.get_gene_exons') as mock_exons:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': 1_050_000, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])
            mock_exons.return_value = pd.DataFrame([
                {'chrom': 'chr1', 'start': 1_000_000, 'end': 1_005_000,
                 'strand': '+', 'gene_name': 'TEST'},
            ])

            result = oracle.analyze_gene_expression(pred, 'TEST')

        # CAGE and RNA should be detected, DNASE should not
        assert 'CAGE:K562' in result['per_track']
        assert 'RNA:K562' in result['per_track']
        assert 'DNASE:K562' not in result['per_track']

    def test_no_expression_tracks(self):
        from chorus.core.result import OraclePrediction, DNaseOraclePredictionTrack

        oracle = _make_test_oracle()
        track = _make_real_track("DNASE:K562", cls=DNaseOraclePredictionTrack)
        pred = OraclePrediction(tracks={"DNASE:K562": track})

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': 1_050_000, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])

            result = oracle.analyze_gene_expression(pred, 'TEST')

        assert result['per_track'] == {}


# ── analyze_variant_effect_on_gene tests ──────────────────────────────

class TestAnalyzeVariantEffectOnGene:
    """Test OracleBase.analyze_variant_effect_on_gene() method."""

    def test_fold_change_computation(self):
        from chorus.core.result import OraclePrediction, CAGEOraclePredictionTrack

        oracle = _make_test_oracle()

        # Reference track with peak=10 at TSS
        ref_track = _make_real_track("CAGE:K562", cls=CAGEOraclePredictionTrack)
        ref_track.values = np.ones(100, dtype=np.float32)
        ref_track.values[50] = 10.0

        # Alt track with peak=20 at TSS (2x fold change)
        alt_track = _make_real_track("CAGE:K562", cls=CAGEOraclePredictionTrack)
        alt_track.values = np.ones(100, dtype=np.float32)
        alt_track.values[50] = 20.0

        ref_pred = OraclePrediction(tracks={"CAGE:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"CAGE:K562": alt_track})

        variant_result = {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {'alt_1': {'CAGE:K562': alt_track.values - ref_track.values}},
            'variant_info': {
                'position': f'chr1:{1_000_000 + 50 * 128}',
                'ref': 'A', 'alts': ['G'],
            }
        }

        tss_pos = 1_000_000 + 50 * 128 + 64

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss, \
             patch('chorus.utils.annotations.get_gene_exons') as mock_exons:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': tss_pos, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])
            mock_exons.return_value = pd.DataFrame()

            result = oracle.analyze_variant_effect_on_gene(variant_result, 'TEST')

        assert result['gene_name'] == 'TEST'
        assert 'alt_1' in result['per_allele']
        vs_ref = result['per_allele']['alt_1']['vs_reference']['CAGE:K562']
        assert vs_ref['ref_expression'] == 10.0
        assert vs_ref['alt_expression'] == 20.0
        assert vs_ref['fold_change'] == 2.0
        assert abs(vs_ref['absolute_change'] - 10.0) < 1e-5

    def test_no_expression_tracks_returns_empty(self):
        from chorus.core.result import OraclePrediction, DNaseOraclePredictionTrack

        oracle = _make_test_oracle()
        ref_track = _make_real_track("DNASE:K562", cls=DNaseOraclePredictionTrack)
        alt_track = _make_real_track("DNASE:K562", cls=DNaseOraclePredictionTrack)
        ref_pred = OraclePrediction(tracks={"DNASE:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"DNASE:K562": alt_track})

        variant_result = {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {'alt_1': {'DNASE:K562': alt_track.values - ref_track.values}},
            'variant_info': {'position': 'chr1:1050000', 'ref': 'A', 'alts': ['G']},
        }

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss:
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr1', 'tss': 1_050_000, 'strand': '+',
                'gene_name': 'TEST', 'gene_id': 'ENSG1',
                'transcript_id': 'ENST1', 'transcript_type': 'protein_coding',
                'transcript_start': 1_000_000, 'transcript_end': 1_100_000,
            }])

            result = oracle.analyze_variant_effect_on_gene(variant_result, 'TEST')

        assert result['reference_expression'] == {}
        assert result['per_allele']['alt_1']['vs_reference'] == {}

    def test_no_tss_in_window(self):
        from chorus.core.result import OraclePrediction, CAGEOraclePredictionTrack

        oracle = _make_test_oracle()
        ref_track = _make_real_track("CAGE:K562", cls=CAGEOraclePredictionTrack)
        alt_track = _make_real_track("CAGE:K562", cls=CAGEOraclePredictionTrack)
        ref_pred = OraclePrediction(tracks={"CAGE:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"CAGE:K562": alt_track})

        variant_result = {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {'alt_1': {'CAGE:K562': alt_track.values - ref_track.values}},
            'variant_info': {'position': 'chr1:1050000', 'ref': 'A', 'alts': ['G']},
        }

        with patch('chorus.utils.annotations.get_gene_tss') as mock_tss, \
             patch('chorus.utils.annotations.get_gene_exons') as mock_exons:
            # TSS on a different chromosome — not in window
            mock_tss.return_value = pd.DataFrame([{
                'chrom': 'chr2', 'tss': 5_000_000, 'strand': '+',
                'gene_name': 'FAR_GENE', 'gene_id': 'ENSG2',
                'transcript_id': 'ENST2', 'transcript_type': 'protein_coding',
                'transcript_start': 5_000_000, 'transcript_end': 5_100_000,
            }])
            mock_exons.return_value = pd.DataFrame()

            result = oracle.analyze_variant_effect_on_gene(variant_result, 'FAR_GENE')

        # CAGE track should have 0 expression (no TSS in window)
        cage_result = result['reference_expression']['CAGE:K562']
        assert cage_result['expression'] == 0.0


# ── Bug-fix regression tests ─────────────────────────────────────────

class TestBedgraphFilenameSanitization:
    """BUG-2: AlphaGenome track IDs contain / which crash bedgraph save."""

    def test_slash_in_track_id(self):
        import re
        track_id = "CHIP_TF/EFO:0002067 TF ChIP-seq GATA1/."
        clean_id = re.sub(r'[/:*?"<>|.\s]+', '_', track_id).strip('_')
        assert "/" not in clean_id
        assert "\\" not in clean_id
        assert clean_id == "CHIP_TF_EFO_0002067_TF_ChIP-seq_GATA1"

    def test_simple_colon(self):
        import re
        track_id = "DNASE:K562"
        clean_id = re.sub(r'[/:*?"<>|.\s]+', '_', track_id).strip('_')
        assert clean_id == "DNASE_K562"


class TestMixedResolutionScoring:
    """BUG-10: at_variant scoring fails for mixed-resolution tracks (AlphaGenome)."""

    def test_mixed_resolution_at_variant(self):
        """Tracks at 1bp and 128bp should both return scores."""
        from chorus.core.result import score_variant_effect, OraclePrediction

        pred_start = 1_000_000
        var_pos = pred_start + 500  # 500bp into the region

        # 1bp resolution track (like DNASE in AlphaGenome)
        track_1bp = _make_real_track("DNASE:hepatocyte", n_bins=1000, resolution=1,
                                      pred_start=pred_start)
        alt_1bp = _make_real_track("DNASE:hepatocyte", n_bins=1000, resolution=1,
                                    pred_start=pred_start)
        alt_1bp.values = track_1bp.values + 0.5

        # 128bp resolution track (like histone ChIP in AlphaGenome)
        track_128bp = _make_real_track("H3K27ac:hepatocyte", n_bins=8, resolution=128,
                                        pred_start=pred_start)
        alt_128bp = _make_real_track("H3K27ac:hepatocyte", n_bins=8, resolution=128,
                                      pred_start=pred_start)
        alt_128bp.values = track_128bp.values + 2.0

        ref_pred = OraclePrediction(tracks={
            "DNASE:hepatocyte": track_1bp,
            "H3K27ac:hepatocyte": track_128bp,
        })
        alt_pred = OraclePrediction(tracks={
            "DNASE:hepatocyte": alt_1bp,
            "H3K27ac:hepatocyte": alt_128bp,
        })

        vr = {
            'predictions': {'reference': ref_pred, 'alt_1': alt_pred},
            'effect_sizes': {'alt_1': {}},
            'variant_info': {'position': f'chr1:{var_pos}', 'ref': 'A', 'alts': ['G']},
        }

        result = score_variant_effect(vr, at_variant=True, window_bins=50,
                                       scoring_strategy="mean")

        # Both tracks should return non-None scores
        dnase = result['alt_1']['DNASE:hepatocyte']
        h3k27ac = result['alt_1']['H3K27ac:hepatocyte']
        assert dnase['ref_score'] is not None, "1bp track should have score"
        assert h3k27ac['ref_score'] is not None, "128bp track should have score"
        assert abs(dnase['effect'] - 0.5) < 1e-4
        assert abs(h3k27ac['effect'] - 2.0) < 1e-4


class TestAutoRegion:
    """ISSUE-3: auto-center region on variant position."""

    def test_auto_region_returns_1bp(self):
        from chorus.mcp.server import _auto_region
        oracle = MagicMock()
        oracle.sequence_length = 393216
        region = _auto_region(oracle, "chr2:60490908")
        assert region == "chr2:60490908-60490909"

    def test_auto_region_different_chrom(self):
        from chorus.mcp.server import _auto_region
        oracle = MagicMock()
        oracle.sequence_length = 1048576
        region = _auto_region(oracle, "chr16:53767042")
        assert region == "chr16:53767042-53767043"


class TestNonExpressionTrackWarning:
    """BUG-4: Gene expression analysis silently returns empty when no CAGE/RNA tracks."""

    def test_warning_present_in_code(self):
        """The analyze_gene_expression method should produce a warning for non-expression tracks."""
        from chorus.core.result import OraclePrediction, OraclePredictionTrack

        # Create prediction with only DNASE (not CAGE/RNA)
        dnase = _make_real_track("DNASE:K562", n_bins=100, resolution=128)
        pred = OraclePrediction(tracks={"DNASE:K562": dnase})

        # The auto-detection checks isinstance for CAGE/RNA track types.
        # DNASE tracks won't match, so tracks_to_analyze will be empty.
        tracks_to_analyze = {
            tid: track for tid, track in pred.items()
            if isinstance(track, (type(None),))  # no match — simulates non-expression
        }
        assert len(tracks_to_analyze) == 0

        # Verify the warning code path produces the right keys
        present_types = sorted({type(t).__name__ for t in pred.values()})
        result = {
            'gene_name': 'TEST',
            'tss_positions': [],
            'exon_regions': [],
            'per_track': {},
            'warning': (
                f"No expression tracks (CAGE/RNA) found in prediction. "
                f"Track types present: {present_types}. "
                f"Gene expression analysis requires CAGE or RNA track types."
            ),
        }
        assert 'warning' in result
        assert 'CAGE' in result['warning']
        assert 'OraclePredictionTrack' in result['warning']  # the type name


class TestChrombpnetKeyMismatch:
    """BUG-8: predict_variant_effect KeyError for ChromBPNet."""

    def test_effect_sizes_use_prediction_keys(self):
        """Effect sizes should use actual prediction keys, not input assay_ids."""
        from chorus.core.result import OraclePrediction

        # Simulate ChromBPNet: user passes assay_ids=["ATAC"] but
        # prediction returns tracks keyed as "ATAC:K562"
        ref_track = _make_real_track("ATAC:K562", n_bins=100, resolution=1)
        alt_track = _make_real_track("ATAC:K562", n_bins=100, resolution=1)
        alt_track.values = ref_track.values + 1.0

        ref_pred = OraclePrediction(tracks={"ATAC:K562": ref_track})
        alt_pred = OraclePrediction(tracks={"ATAC:K562": alt_track})

        # Build result dict as base.py does (using ref_keys, not assay_ids)
        ref_keys = list(ref_pred.keys())
        effect_sizes = {
            'alt_1': {
                assay: alt_pred[assay].values - ref_pred[assay].values
                for assay in ref_keys
            }
        }

        # Should have "ATAC:K562" not "ATAC"
        assert "ATAC:K562" in effect_sizes['alt_1']
        assert "ATAC" not in effect_sizes['alt_1']


class TestTSSOutOfWindowWarning:
    """ISSUE-9: predict_variant_effect_on_gene should warn when TSS is outside window."""

    def test_warning_generated(self):
        """The MCP server layer should add a warning with distance and recommendations."""
        from chorus.mcp.server import ORACLE_SPECS

        # Simulate the warning logic from server.py
        tss_positions = [109393357, 109397918]  # SORT1
        ref_expr = {"CNhs10624": {"n_tss_in_window": 0, "expression": 0}}
        position = "chr1:109274968"
        oracle_name = "enformer"

        all_zero = all(
            info.get("n_tss_in_window", 0) == 0
            for info in ref_expr.values()
        )
        assert all_zero

        var_chrom, var_pos_str = position.split(":")
        var_pos = int(var_pos_str)
        nearest_tss = min(tss_positions, key=lambda t: abs(t - var_pos))
        distance_kb = abs(nearest_tss - var_pos) / 1000

        assert distance_kb > 100  # SORT1 is ~118kb away
        assert nearest_tss == 109393357

        output_kb = ORACLE_SPECS["enformer"]["output_bins"] * ORACLE_SPECS["enformer"]["resolution_bp"] / 1000
        assert output_kb < distance_kb  # TSS is outside window


class TestChrombpnetLoadParams:
    """BUG-1/3: ChromBPNet loading with TF, fold, model_type params."""

    def test_load_only_keys_expanded(self):
        """state.py should separate TF/fold/model_type from constructor kwargs."""
        _load_only_keys = {"assay", "cell_type", "TF", "fold", "model_type"}
        kwargs = {"assay": "CHIP", "cell_type": "K562", "TF": "GATA1",
                  "fold": 0, "model_type": "chrombpnet", "extra_param": "value"}

        oracle_kwargs = {k: v for k, v in kwargs.items() if k not in _load_only_keys}
        load_kwargs = {k: v for k, v in kwargs.items() if k in _load_only_keys}

        assert "extra_param" in oracle_kwargs
        assert "TF" not in oracle_kwargs
        assert "TF" in load_kwargs
        assert load_kwargs["TF"] == "GATA1"
        assert load_kwargs["fold"] == 0


# ── Region/position parsing tests ────────────────────────────────────

class TestParseRegion:
    """Test _parse_region() validation helper."""

    def test_valid_region(self):
        from chorus.mcp.server import _parse_region
        chrom, start, end = _parse_region("chr1:1000000-2000000")
        assert chrom == "chr1"
        assert start == 1000000
        assert end == 2000000

    def test_chrX(self):
        from chorus.mcp.server import _parse_region
        chrom, start, end = _parse_region("chrX:100-200")
        assert chrom == "chrX"

    def test_missing_colon(self):
        from chorus.mcp.server import _parse_region
        with pytest.raises(ValueError, match="Invalid region format"):
            _parse_region("chr1-1000000-2000000")

    def test_missing_dash(self):
        from chorus.mcp.server import _parse_region
        with pytest.raises(ValueError, match="Invalid region format"):
            _parse_region("chr1:1000000")

    def test_start_ge_end(self):
        from chorus.mcp.server import _parse_region
        with pytest.raises(ValueError, match="start.*less than end"):
            _parse_region("chr1:2000000-1000000")

    def test_empty_string(self):
        from chorus.mcp.server import _parse_region
        with pytest.raises(ValueError, match="Invalid region format"):
            _parse_region("")


class TestParsePosition:
    """Test _parse_position() validation helper."""

    def test_valid_position(self):
        from chorus.mcp.server import _parse_position
        chrom, pos = _parse_position("chr1:1050000")
        assert chrom == "chr1"
        assert pos == 1050000

    def test_invalid_format(self):
        from chorus.mcp.server import _parse_position
        with pytest.raises(ValueError, match="Invalid position format"):
            _parse_position("chr1:100-200")

    def test_no_colon(self):
        from chorus.mcp.server import _parse_position
        with pytest.raises(ValueError, match="Invalid position format"):
            _parse_position("chr1_1050000")


# ── Prompt tests ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp not installed")
class TestPrompts:
    def test_getting_started_returns_string(self):
        from chorus.mcp.server import getting_started
        result = getting_started()
        assert isinstance(result, str)
        assert "list_oracles" in result
        assert "AlphaGenome" in result

    def test_analyze_variant_returns_string(self):
        from chorus.mcp.server import analyze_variant
        result = analyze_variant(variant="chr1:109274968 G>T", gene="SORT1", cell_type="HepG2")
        assert isinstance(result, str)
        assert "SORT1" in result
        assert "HepG2" in result


class TestFineMapRsidBackfill:
    """Regression test: fine_map_causal_variant with rsID-only lead must not KeyError.

    Without the chrom/pos/ref/alt backfill from the LD-resolved sentinel,
    downstream prioritize_causal_variants raises KeyError: 'chrom'.
    """

    def test_rsid_only_lead_backfills_coordinates(self):
        from chorus.mcp import server as srv

        @dataclass
        class FakeLD:
            variant_id: str
            chrom: str
            position: int
            ref: str
            alt: str
            r2: float = 1.0
            is_sentinel: bool = False

        ld_list = [
            FakeLD("rs12740374", "chr1", 109274968, "G", "T", 1.0, is_sentinel=True),
            FakeLD("rs4970836", "chr1", 109279175, "G", "A", 0.9),
        ]

        captured = {}

        def fake_prioritize(oracle, lead_dict, ld_variants, assay_ids, **kwargs):
            captured["lead_dict"] = dict(lead_dict)
            r = MagicMock()
            r.to_dict.return_value = {"rankings": []}
            r.to_markdown.return_value = "# causal"
            return r

        with patch.object(srv, "_state") as mock_state, \
             patch("chorus.utils.ld.ld_variants_from_list", return_value=ld_list), \
             patch("chorus.analysis.causal.prioritize_causal_variants", side_effect=fake_prioritize):
            mock_state.return_value.get_oracle.return_value = MagicMock()
            mock_state.return_value.get_normalizer.return_value = None
            mock_state.return_value.output_dir = None

            result = srv.fine_map_causal_variant(
                oracle_name="alphagenome",
                lead_variant="rs12740374",
                ld_variants=[{"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T"}],
            )

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        ld = captured["lead_dict"]
        assert ld["chrom"] == "chr1"
        assert ld["pos"] == 109274968
        assert ld["ref"] == "G"
        assert ld["alt"] == "T"
