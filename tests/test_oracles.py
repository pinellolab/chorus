"""Tests for oracle implementations."""

import pytest
import numpy as np
from chorus.oracles import (
    EnformerOracle,
    BorzoiOracle,
    ChromBPNetOracle,
    SeiOracle,
    get_oracle
)


class TestOracleFactory:
    """Test oracle factory functions."""
    
    def test_get_oracle(self):
        """Test getting oracle classes by name."""
        assert get_oracle('enformer') == EnformerOracle
        assert get_oracle('borzoi') == BorzoiOracle
        assert get_oracle('chrombpnet') == ChromBPNetOracle
        assert get_oracle('sei') == SeiOracle
        
        # Test case insensitive
        assert get_oracle('ENFORMER') == EnformerOracle
        
        # Test invalid name
        with pytest.raises(ValueError):
            get_oracle('invalid_oracle')


class TestEnformerOracle:
    """Test Enformer oracle implementation."""
    
    def test_initialization(self):
        """Test Enformer initialization."""
        oracle = EnformerOracle()
        
        assert oracle.target_length == 896
        assert oracle.bin_size == 128
        assert oracle.sequence_length == 393216
        assert oracle.center_length == 196608
        assert not oracle.loaded
    
    def test_list_assays(self):
        """Test listing available assays."""
        oracle = EnformerOracle()
        
        assays = oracle.list_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0
        assert "DNase" in assays
        assert "CAGE" in assays
        
        cell_types = oracle.list_cell_types()
        assert isinstance(cell_types, list)
        assert len(cell_types) > 0
        assert "K562" in cell_types
    
    def test_one_hot_encoding(self):
        """Test DNA sequence one-hot encoding."""
        oracle = EnformerOracle()
        
        # Test basic encoding
        seq = "ATCG"
        one_hot = oracle._one_hot_encode(seq)
        
        assert one_hot.shape == (4, 4)
        assert np.array_equal(one_hot[0], [1, 0, 0, 0])  # A
        assert np.array_equal(one_hot[1], [0, 0, 0, 1])  # T
        assert np.array_equal(one_hot[2], [0, 1, 0, 0])  # C
        assert np.array_equal(one_hot[3], [0, 0, 1, 0])  # G
        
        # Test with N
        seq_n = "ATCGN"
        one_hot_n = oracle._one_hot_encode(seq_n)
        assert np.array_equal(one_hot_n[4], [0, 0, 0, 0])  # N
    
    def test_prepare_sequence(self):
        """Test sequence preparation."""
        oracle = EnformerOracle()
        
        # Test short sequence (padding)
        short_seq = "ATCG" * 100  # 400 bp
        prepared = oracle._prepare_sequence(short_seq)
        assert len(prepared) == oracle.sequence_length
        assert prepared.count('N') == oracle.sequence_length - 400
        
        # Test long sequence (trimming)
        long_seq = "ATCG" * 200000  # 800,000 bp
        prepared = oracle._prepare_sequence(long_seq)
        assert len(prepared) == oracle.sequence_length
        
        # Test exact length
        exact_seq = "A" * oracle.sequence_length
        prepared = oracle._prepare_sequence(exact_seq)
        assert prepared == exact_seq
    
    def test_get_parameters(self):
        """Test parameter getter methods."""
        oracle = EnformerOracle()
        
        assert oracle._get_context_size() == oracle.sequence_length
        assert oracle._get_bin_size() == oracle.bin_size
        
        min_len, max_len = oracle._get_sequence_length_bounds()
        assert min_len == 1000
        assert max_len == oracle.sequence_length


class TestBorzoiOracle:
    """Test Borzoi oracle implementation."""
    
    def test_initialization(self):
        """Test Borzoi initialization."""
        oracle = BorzoiOracle()
        
        assert oracle.sequence_length == 524288
        assert oracle.target_length == 5313
        assert oracle.bin_size == 32
        assert not oracle.loaded
    
    def test_not_implemented(self):
        """Test that unimplemented methods raise NotImplementedError."""
        oracle = BorzoiOracle()
        
        with pytest.raises(NotImplementedError):
            oracle.load_pretrained_model("dummy_path")
        
        with pytest.raises(NotImplementedError):
            oracle._predict("ATCG", ["DNase"])


class TestChromBPNetOracle:
    """Test ChromBPNet oracle implementation."""
    
    def test_initialization(self):
        """Test ChromBPNet initialization."""
        oracle = ChromBPNetOracle()
        
        assert oracle.sequence_length == 2114
        assert oracle.output_length == 1000
        assert oracle.bin_size == 1
        assert not oracle.loaded


class TestSeiOracle:
    """Test Sei oracle implementation."""
    
    def test_initialization(self):
        """Test Sei initialization."""
        oracle = SeiOracle()
        
        assert oracle.sequence_length == 4096
        assert oracle.n_targets == 21907
        assert oracle.bin_size == 1
        assert not oracle.loaded
    
    def test_assay_types(self):
        """Test Sei assay types."""
        oracle = SeiOracle()
        
        groups = oracle.list_group_types()
        assert "Promoter" in groups
        assert "Enhancer" in groups
        assert "Transcription" in groups


if __name__ == "__main__":
    pytest.main([__file__])