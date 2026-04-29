"""Tests for oracle implementations."""

import pytest
import numpy as np
from chorus.oracles import (
    EnformerOracle,
    BorzoiOracle,
    ChromBPNetOracle,
    SeiOracle,
    LegNetOracle,
    AlphaGenomeOracle,
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
        assert get_oracle('legnet') == LegNetOracle
        assert get_oracle('alphagenome') == AlphaGenomeOracle

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
        assert "DNASE" in assays
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
    
    def test_one_hot_roundtrip(self):
        """Test one-hot encoding properties."""
        oracle = EnformerOracle()

        seq = "ACGTACGT"
        one_hot = oracle._one_hot_encode(seq)
        assert one_hot.shape == (8, 4)
        # Each row should have exactly one 1 (for valid bases)
        assert np.all(one_hot.sum(axis=1) == 1)
    
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
        assert oracle.target_length == 6144
        assert oracle.bin_size == 32
        assert not oracle.loaded

    def test_parameters(self):
        """Test Borzoi parameter getter methods."""
        oracle = BorzoiOracle()

        assert oracle._get_context_size() == oracle.sequence_length
        assert oracle._get_bin_size() == oracle.bin_size

        min_len, max_len = oracle._get_sequence_length_bounds()
        assert min_len == 1000
        assert max_len == oracle.sequence_length


class TestChromBPNetOracle:
    """Test ChromBPNet oracle implementation."""

    def test_initialization(self):
        """Test ChromBPNet initialization."""
        oracle = ChromBPNetOracle()

        assert oracle.sequence_length == 2114
        assert oracle.output_length == 1000
        assert oracle.bin_size == 1
        assert not oracle.loaded

    def test_default_model_type_is_nobias(self):
        """v0.3 breaking change: default ``model_type`` is now
        ``'chrombpnet_nobias'`` (bias-corrected). Regression guard so we
        don't accidentally flip back to the bias-aware variant."""
        import inspect
        sig = inspect.signature(ChromBPNetOracle.load_pretrained_model)
        assert sig.parameters["model_type"].default == "chrombpnet_nobias", (
            "ChromBPNet default model_type must stay 'chrombpnet_nobias' "
            "for chorus ≥ 0.3 — flipping back to 'chrombpnet' would silently "
            "regress every shipped notebook + report."
        )

    def test_hf_slim_helpers_exist(self):
        """v0.3: HF slim mirror helpers must be wired so the oracle
        falls through to HF before ENCODE tarball downloads."""
        oracle = ChromBPNetOracle()
        assert hasattr(oracle, "_try_slim_hf_chrombpnet"), \
            "missing _try_slim_hf_chrombpnet helper"
        assert hasattr(oracle, "_try_slim_hf_bpnet"), \
            "missing _try_slim_hf_bpnet helper"
        # Smoke: helpers should return None gracefully when called
        # without configured assay/cell_type (no network roundtrip).
        oracle.assay = None
        oracle.cell_type = None
        oracle.fold = 0
        oracle.tf = None
        assert oracle._try_slim_hf_chrombpnet() is None
        assert oracle._try_slim_hf_bpnet() is None


class TestSeiOracle:
    """Test Sei oracle implementation."""
    
    def test_initialization(self):
        """Test Sei initialization."""
        oracle = SeiOracle()

        assert oracle.sequence_length == 4096
        assert oracle.n_targets == 21907
        assert not oracle.loaded

    def test_assay_types(self):
        """Test Sei assay types."""
        oracle = SeiOracle()

        assays = oracle.list_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0
        # Sei assay types are histone marks, TFs, etc.
        assert "H3K4me3" in assays

    def test_class_types(self):
        """Test Sei class types."""
        oracle = SeiOracle()

        classes = oracle.list_class_types()
        assert isinstance(classes, list)
        assert "Promoter" in classes
        assert "Transcription" in classes


class TestLegNetOracle:
    """Test LegNet oracle implementation."""

    def test_initialization(self):
        """Test LegNet initialization."""
        oracle = LegNetOracle()

        assert oracle.sequence_length == 200
        assert oracle.bin_size == 50  # default step_size
        assert oracle.oracle_name == "legnet"
        assert not oracle.loaded

    def test_assay_types(self):
        """Test LegNet assay types."""
        oracle = LegNetOracle()

        assays = oracle.list_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0

    def test_get_oracle_lookup(self):
        """Test that LegNet is registered in the oracle factory."""
        assert get_oracle('legnet') == LegNetOracle


class TestAlphaGenomeOracle:
    """Test AlphaGenome oracle implementation."""

    def test_initialization(self):
        """Test AlphaGenome initialization."""
        oracle = AlphaGenomeOracle()

        assert oracle.sequence_length == 1_048_576
        assert oracle.bin_size == 1
        assert oracle.target_length == 1_048_576
        assert oracle.oracle_name == "alphagenome"
        assert not oracle.loaded

    def test_parameters(self):
        """Test AlphaGenome parameter getter methods."""
        oracle = AlphaGenomeOracle()

        assert oracle._get_context_size() == oracle.sequence_length
        assert oracle._get_bin_size() == oracle.bin_size
        assert oracle.output_size == oracle.target_length * oracle.bin_size

        min_len, max_len = oracle._get_sequence_length_bounds()
        assert min_len == 1000
        assert max_len == oracle.sequence_length

    def test_assay_types(self):
        """Test AlphaGenome assay type listing."""
        oracle = AlphaGenomeOracle()

        assays = oracle.list_assay_types()
        assert isinstance(assays, list)
        assert len(assays) > 0
        assert "ATAC" in assays
        assert "CAGE" in assays

    def test_get_oracle_lookup(self):
        """Test that AlphaGenome is registered in the oracle factory."""
        assert get_oracle('alphagenome') == AlphaGenomeOracle


if __name__ == "__main__":
    pytest.main([__file__])