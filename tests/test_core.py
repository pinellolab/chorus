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
        
        # Get chr1 region
        region_data = track.get_region_values('chr1', 150, 350)
        assert len(region_data) == 2
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
        
        # Valid sequence
        oracle._validate_sequence("ATCGATCG")
        oracle._validate_sequence("ATCGATCGN")
        
        # Invalid sequence
        with pytest.raises(InvalidSequenceError):
            oracle._validate_sequence("ATCGATCGX")
        
        # Invalid length
        with pytest.raises(InvalidSequenceError):
            oracle._validate_sequence("ATG")  # Too short


if __name__ == "__main__":
    pytest.main([__file__])