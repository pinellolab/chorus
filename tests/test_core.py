"""Tests for core classes."""

import pytest
import numpy as np
import pandas as pd
from chorus.core import Track, OracleBase
from chorus.core.exceptions import InvalidSequenceError, InvalidRegionError


class TestTrack:
    """Test Track class functionality."""
    
    def test_track_creation(self):
        """Test basic track creation."""
        data = pd.DataFrame({
            'chrom': ['chr1'] * 5,
            'start': [0, 100, 200, 300, 400],
            'end': [100, 200, 300, 400, 500],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        track = Track(
            name="test_track",
            assay_type="DNase",
            cell_type="K562",
            data=data
        )
        
        assert track.name == "test_track"
        assert track.assay_type == "DNase"
        assert track.cell_type == "K562"
        assert len(track.data) == 5
    
    def test_track_validation(self):
        """Test track data validation."""
        # Missing required column
        with pytest.raises(ValueError):
            data = pd.DataFrame({'chrom': ['chr1'], 'start': [0]})
            Track("test", "DNase", "K562", data)
    
    def test_track_normalization(self):
        """Test track normalization methods."""
        data = pd.DataFrame({
            'chrom': ['chr1'] * 5,
            'start': [0, 100, 200, 300, 400],
            'end': [100, 200, 300, 400, 500],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        track = Track("test", "DNase", "K562", data)
        
        # Test z-score normalization
        norm_track = track.normalize('zscore')
        assert abs(norm_track.data['value'].mean()) < 1e-10
        assert abs(norm_track.data['value'].std() - 1.0) < 1e-10
        
        # Test min-max normalization
        norm_track = track.normalize('minmax')
        assert norm_track.data['value'].min() == 0.0
        assert norm_track.data['value'].max() == 1.0
    
    def test_get_region_values(self):
        """Test extracting values for a specific region."""
        data = pd.DataFrame({
            'chrom': ['chr1'] * 5 + ['chr2'] * 5,
            'start': list(range(0, 500, 100)) * 2,
            'end': list(range(100, 600, 100)) * 2,
            'value': range(10)
        })
        
        track = Track("test", "DNase", "K562", data)
        
        # Get chr1 region — bins overlapping [150, 350) are [100-200, 200-300, 300-400]
        region_data = track.get_region_values('chr1', 150, 350)
        assert len(region_data) == 3
        assert all(region_data['chrom'] == 'chr1')
    
    def test_aggregate_by_bins(self):
        """Test bin aggregation."""
        data = pd.DataFrame({
            'chrom': ['chr1'] * 10,
            'start': range(0, 100, 10),
            'end': range(10, 110, 10),
            'value': range(10)
        })
        
        track = Track("test", "DNase", "K562", data)
        binned = track.aggregate_by_bins(50)
        
        assert len(binned.data) == 2
        assert binned.data.iloc[0]['end'] - binned.data.iloc[0]['start'] == 50


class TestOracleBase:
    """Test OracleBase abstract class."""
    
    def test_parse_region(self):
        """Test genomic region parsing."""
        # Create a concrete implementation for testing
        class TestOracle(OracleBase):
            def load_pretrained_model(self, weights): pass
            def list_assay_types(self): return []
            def list_cell_types(self): return []
            def _predict(self, seq, assay_ids): return np.zeros((100, len(assay_ids)))
            def fine_tune(self, tracks, track_names, **kwargs): pass
            def _get_context_size(self): return 1000
            def _get_sequence_length_bounds(self): return (10, 10000)
            def _get_bin_size(self): return 128
        
        oracle = TestOracle()
        
        # Test string format
        chrom, start, end = oracle._parse_region("chr1:1000-2000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000
        
        # Test DataFrame format
        df = pd.DataFrame([{'chrom': 'chr2', 'start': 5000, 'end': 6000}])
        chrom, start, end = oracle._parse_region(df)
        assert chrom == "chr2"
        assert start == 5000
        assert end == 6000
        
        # Test invalid format
        with pytest.raises(InvalidRegionError):
            oracle._parse_region("invalid_format")
    
    def test_parse_position(self):
        """Test genomic position parsing."""
        class TestOracle(OracleBase):
            def load_pretrained_model(self, weights): pass
            def list_assay_types(self): return []
            def list_cell_types(self): return []
            def _predict(self, seq, assay_ids): return np.zeros((100, len(assay_ids)))
            def fine_tune(self, tracks, track_names, **kwargs): pass
            def _get_context_size(self): return 1000
            def _get_sequence_length_bounds(self): return (10, 10000)
            def _get_bin_size(self): return 128
        
        oracle = TestOracle()
        
        # Test string format
        chrom, pos = oracle._parse_position("chr1:1000")
        assert chrom == "chr1"
        assert pos == 1000
        
        # Test invalid format
        with pytest.raises(InvalidRegionError):
            oracle._parse_position("chr1-1000")
    
    def test_validate_sequence(self):
        """Test sequence validation."""
        class TestOracle(OracleBase):
            def load_pretrained_model(self, weights): pass
            def list_assay_types(self): return []
            def list_cell_types(self): return []
            def _predict(self, seq, assay_ids): return np.zeros((100, len(assay_ids)))
            def fine_tune(self, tracks, track_names, **kwargs): pass
            def _get_context_size(self): return 1000
            def _get_sequence_length_bounds(self): return (10, 10000)
            def _get_bin_size(self): return 128
        
        oracle = TestOracle()
        
        # Valid sequence (min length is 10 per _get_sequence_length_bounds)
        oracle._validate_sequence("ATCGATCGATCG")
        oracle._validate_sequence("ATCGATCGATCGN")

        # Invalid characters
        with pytest.raises(InvalidSequenceError):
            oracle._validate_sequence("ATCGATCGATCGX")

        # Too short (min=10)
        with pytest.raises(InvalidSequenceError):
            oracle._validate_sequence("ATG")


class TestOraclePredictionTrack:
    """Test OraclePredictionTrack properties and methods."""

    def _make_track(self, assay_id="DNase:K562", num_bins=896):
        from chorus.core.result import OraclePredictionTrack
        from chorus.core.interval import Interval, Sequence

        seq = "A" * 10000
        interval = Interval.make(Sequence(sequence=seq))

        return OraclePredictionTrack.create(
            source_model="mock",
            assay_id=assay_id,
            track_id=0,
            assay_type="DNase",
            cell_type="K562",
            query_interval=interval,
            prediction_interval=interval,
            input_interval=interval,
            resolution=128,
            values=np.random.rand(num_bins).astype(np.float32),
        )

    def test_score_mean(self):
        track = self._make_track()
        s = track.score("mean")
        assert isinstance(s, float)
        assert np.isclose(s, float(np.mean(track.values)))

    def test_score_max(self):
        track = self._make_track()
        s = track.score("max")
        assert np.isclose(s, float(np.max(track.values)))

    def test_score_sum(self):
        track = self._make_track()
        s = track.score("sum")
        assert np.isclose(s, float(np.sum(track.values)), rtol=1e-5)

    def test_score_default(self):
        track = self._make_track()
        # preferred_scoring_strategy defaults to 'mean'
        assert np.isclose(track.score(), track.score("mean"))

    def test_score_unknown_strategy(self):
        track = self._make_track()
        with pytest.raises(ValueError, match="Unknown scoring strategy"):
            track.score("nonexistent")

    def test_track_properties(self):
        track = self._make_track()
        assert isinstance(track.chrom, str)
        assert isinstance(track.start, int)
        assert isinstance(track.end, int)
        assert track.end >= track.start

    def test_positions(self):
        track = self._make_track(num_bins=10)
        pos = track.positions
        assert len(pos) == 10
        assert pos[1] - pos[0] == 128  # resolution

    def test_len(self):
        track = self._make_track(num_bins=42)
        assert len(track) == 42


class TestOraclePrediction:
    """Test OraclePrediction properties and methods."""

    def _make_prediction(self, assay_ids=None):
        from chorus.core.result import OraclePrediction, OraclePredictionTrack
        from chorus.core.interval import Interval, Sequence

        if assay_ids is None:
            assay_ids = ["DNase:K562", "RNA-seq:HepG2"]

        seq = "A" * 10000
        interval = Interval.make(Sequence(sequence=seq))
        pred = OraclePrediction()

        for i, aid in enumerate(assay_ids):
            parts = aid.split(":")
            track = OraclePredictionTrack.create(
                source_model="mock",
                assay_id=aid,
                track_id=i,
                assay_type=parts[0],
                cell_type=parts[1] if len(parts) > 1 else "UNKNOWN",
                query_interval=interval,
                prediction_interval=interval,
                input_interval=interval,
                resolution=128,
                values=np.random.rand(896).astype(np.float32),
            )
            pred.add(aid, track)
        return pred

    def test_getitem(self):
        pred = self._make_prediction()
        track = pred["DNase:K562"]
        assert track.assay_id == "DNase:K562"

    def test_iter(self):
        pred = self._make_prediction()
        keys = list(pred)
        assert "DNase:K562" in keys
        assert "RNA-seq:HepG2" in keys

    def test_items_keys_values(self):
        pred = self._make_prediction()
        assert len(list(pred.keys())) == 2
        assert len(list(pred.values())) == 2
        assert len(list(pred.items())) == 2

    def test_chrom(self):
        pred = self._make_prediction()
        assert isinstance(pred.chrom, str)

    def test_start_end(self):
        pred = self._make_prediction()
        assert isinstance(pred.start, int)
        assert isinstance(pred.end, int)
        assert pred.end >= pred.start

    def test_add_duplicate_raises(self):
        from chorus.core.result import OraclePrediction
        pred = self._make_prediction(["DNase:K562"])
        with pytest.raises(Exception, match="already exists"):
            # Try to add same assay_id again
            pred.add("DNase:K562", pred["DNase:K562"])

    def test_subset(self):
        pred = self._make_prediction(["A:X", "B:Y", "C:Z"])
        sub = pred.subset(["A:X", "C:Z"])
        assert len(list(sub.keys())) == 2
        assert "A:X" in sub.keys()
        assert "C:Z" in sub.keys()
        assert "B:Y" not in sub.keys()


class TestPlatformAdaptation:
    """Test platform detection and adaptation system."""

    def test_detect_platform(self):
        from chorus.core.platform import detect_platform, PlatformInfo
        info = detect_platform()
        assert isinstance(info, PlatformInfo)
        assert info.system in ("Darwin", "Linux", "Windows")

    def test_platform_key_format(self):
        from chorus.core.platform import PlatformInfo
        # macOS ARM
        info = PlatformInfo(system="Darwin", machine="arm64", is_arm=True, is_macos=True)
        assert info.key == "macos_arm64"

        # Linux x86_64 no CUDA
        info = PlatformInfo(system="Linux", machine="x86_64", is_linux=True)
        assert info.key == "linux_x86_64"

        # Linux x86_64 with CUDA
        info = PlatformInfo(system="Linux", machine="x86_64", is_linux=True, has_cuda=True)
        assert info.key == "linux_x86_64_cuda"

    def test_platform_properties(self):
        from chorus.core.platform import PlatformInfo
        mac = PlatformInfo(system="Darwin", machine="arm64", is_arm=True, is_macos=True)
        assert mac.is_macos
        assert mac.is_arm
        assert not mac.is_linux

        linux = PlatformInfo(system="Linux", machine="x86_64", is_linux=True, has_cuda=True)
        assert linux.is_linux
        assert not linux.is_arm
        assert not linux.is_macos

    def test_adapt_environment_config_no_adaptation(self):
        from chorus.core.platform import adapt_environment_config, PlatformInfo
        config = {"name": "test", "dependencies": ["numpy"]}
        # Use a platform key that has no adaptations (linux_x86_64 for enformer)
        info = PlatformInfo(system="Linux", machine="x86_64", is_linux=True)
        result_config, post_install, notes = adapt_environment_config(config, "enformer", info)
        assert result_config["name"] == "test"

    def test_adapt_environment_config_cuda_fallback(self):
        from chorus.core.platform import adapt_environment_config, PlatformInfo, PLATFORM_ADAPTATIONS
        if "alphagenome" in PLATFORM_ADAPTATIONS:
            ag_adaptations = PLATFORM_ADAPTATIONS["alphagenome"]
            if "linux_x86_64_cuda" in ag_adaptations:
                config = {
                    "name": "chorus-alphagenome",
                    "dependencies": ["numpy", {"pip": ["jax[cpu]"]}],
                }
                info = PlatformInfo(system="Linux", machine="x86_64", is_linux=True, has_cuda=True)
                result_config, post_install, notes = adapt_environment_config(config, "alphagenome", info)
                # jax[cpu] should be replaced with jax[cuda12]
                pip_deps = None
                for dep in result_config["dependencies"]:
                    if isinstance(dep, dict) and "pip" in dep:
                        pip_deps = dep["pip"]
                assert pip_deps is not None
                assert "jax[cuda12]" in pip_deps
                assert "jax[cpu]" not in pip_deps


if __name__ == "__main__":
    pytest.main([__file__])