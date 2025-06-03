"""Tests for the three main prediction methods in Chorus."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from chorus.core.base import OracleBase
from chorus.core.track import Track


class MockOracle(OracleBase):
    """Mock oracle for testing base prediction methods."""
    
    def __init__(self, reference_fasta=None):
        super().__init__(use_environment=False)  # Don't use environment for tests
        self.loaded = True
        self._context_size = 393216  # Enformer context size
        self._output_size = 114688   # Enformer output size
        self._num_tracks = 5313      # Enformer tracks
        self._bin_size = 128         # Enformer bin size
        self.reference_fasta = reference_fasta  # Store reference separately
        
    def _get_context_size(self):
        return self._context_size
        
    def _get_output_size(self):
        return self._output_size
        
    def _get_num_tracks(self):
        return self._num_tracks
        
    def _get_bin_size(self):
        return self._bin_size
        
    def _get_sequence_length_bounds(self):
        """Return min and max sequence lengths."""
        return (self._context_size, self._context_size)  # Enformer requires exact context size
    
    def load_pretrained_model(self, weights=None):
        """Mock model loading."""
        self.loaded = True
        
    def list_assay_types(self):
        """Return mock assay types."""
        return ['DNase', 'RNA-seq', 'ChIP-seq']
        
    def list_cell_types(self):
        """Return mock cell types."""
        return ['K562', 'HepG2', 'GM12878']
        
    def _get_assay_ids(self, track_indices):
        """Convert track indices to assay IDs."""
        return [f"track_{i}" for i in track_indices]
        
    def _get_track_indices(self, assay_ids):
        """Convert assay IDs to track indices."""
        indices = []
        for aid in assay_ids:
            if aid == 'DNase:K562':
                indices.append(0)
            elif aid == 'RNA-seq:HepG2':
                indices.append(1)
            elif aid.startswith('track_'):
                indices.append(int(aid.split('_')[1]))
            else:
                raise ValueError(f"Unknown assay ID: {aid}")
        return indices
        
    def _predict(self, seq, assay_ids=None):
        """Mock prediction returning random data."""
        # Calculate expected dimensions
        num_bins = self._output_size // self._bin_size  # 896 bins
        
        if assay_ids is None:
            num_tracks = self._num_tracks
        else:
            num_tracks = len(self._get_track_indices(assay_ids))
            
        # Return mock predictions
        return np.random.rand(num_bins, num_tracks)
        
    def fine_tune(self, dataset, epochs=10):
        """Mock fine-tuning."""
        pass


class TestPredictionMethods:
    """Test suite for prediction methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary fasta file
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"
        
        # Write a simple test genome
        with open(self.fasta_path, 'w') as f:
            f.write(">chr1\n")
            f.write("A" * 500000 + "\n")  # 500kb of A's
            f.write(">chr2\n") 
            f.write("T" * 300000 + "\n")  # 300kb of T's
            
        # Create oracle with reference genome
        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_predict_with_sequence(self):
        """Test prediction with a sequence."""
        # Test sequence (must be context_size)
        test_seq = "ACGT" * (self.oracle._context_size // 4)
        
        # Call predict with sequence
        results = self.oracle.predict(
            input_data=test_seq,
            assay_ids=['DNase:K562', 'RNA-seq:HepG2']
        )
        
        # Verify results structure - predict returns dict of assay_id -> predictions
        assert isinstance(results, dict)
        assert 'DNase:K562' in results
        assert 'RNA-seq:HepG2' in results
        
        # Check dimensions
        assert results['DNase:K562'].shape == (896,)
        assert results['RNA-seq:HepG2'].shape == (896,)
        
    def test_predict_region_replacement_with_sequence(self):
        """Test region replacement providing both region and sequence."""
        # Use pysam to create index
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Define a region that fits within chr1
        center = 250000  # Center of chr1
        half_context = self.oracle._context_size // 2
        start = center - half_context
        end = center + half_context
        
        # New sequence to replace with
        new_seq = "ACGT" * (self.oracle._context_size // 4)
        
        # Call with genomic region and replacement sequence
        results = self.oracle.predict_region_replacement(
            genomic_region=f"chr1:{start}-{end}",
            seq=new_seq,
            assay_ids=['DNase:K562']
        )
        
        # Verify results
        assert results['raw_predictions'].shape == (896, 1)
        assert results['normalized_scores'].shape == (896, 1)
        
    def test_predict_invalid_inputs(self):
        """Test prediction with invalid inputs."""
        # Invalid sequence length
        with pytest.raises(ValueError, match="must be exactly"):
            self.oracle.predict(
                input_data="ACGT",  # Too short
                assay_ids=['DNase:K562']
            )
            
    def test_predict_region_insertion_at(self):
        """Test sequence insertion at position."""
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Insert a 1kb sequence at position 250000
        insert_seq = "ACGT" * 250  # 1kb
        
        results = self.oracle.predict_region_insertion_at(
            genomic_position="chr1:250000",
            seq=insert_seq,
            assay_ids=['DNase:K562', 'RNA-seq:HepG2'],
            genome=str(self.fasta_path)
        )
        
        # Verify results
        assert results['raw_predictions'].shape == (896, 2)
        assert results['normalized_scores'].shape == (896, 2)
        assert len(results['track_objects']) == 2
        
    def test_predict_region_insertion_edge_cases(self):
        """Test insertion at chromosome boundaries."""
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Try insertion near start of chromosome
        insert_seq = "ACGT" * 250
        
        # This should work - padding will be added
        results = self.oracle.predict_region_insertion_at(
            genomic_position="chr1:1000",
            seq=insert_seq,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert results['raw_predictions'].shape == (896, 1)
        
    def test_predict_variant_effect_snp(self):
        """Test variant effect prediction for SNP."""
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Define a SNP variant
        variant = {
            'chrom': 'chr1',
            'pos': 250000,
            'ref': 'A',
            'alt': ['C']
        }
        
        results = self.oracle.predict_variant_effect(
            variant=variant,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        # Verify results structure
        assert 'predictions' in results
        assert 'effect_sizes' in results
        assert 'tracks' in results
        assert 'variant_info' in results
        
        # Check predictions
        assert 'ref' in results['predictions']
        assert 'alt_0' in results['predictions']
        
        # Check effect sizes
        assert 'alt_0' in results['effect_sizes']
        
        # Verify dimensions
        assert results['predictions']['ref'].shape == (896, 1)
        assert results['predictions']['alt_0'].shape == (896, 1)
        assert results['effect_sizes']['alt_0'].shape == (896, 1)
        
    def test_predict_variant_effect_multiallelic(self):
        """Test variant effect prediction for multi-allelic variant."""
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Define a multi-allelic variant
        variant = {
            'chrom': 'chr1',
            'pos': 250000,
            'ref': 'A',
            'alt': ['C', 'G', 'T']
        }
        
        results = self.oracle.predict_variant_effect(
            variant=variant,
            assay_ids=['DNase:K562', 'RNA-seq:HepG2'],
            genome=str(self.fasta_path)
        )
        
        # Check all alleles are predicted
        assert 'ref' in results['predictions']
        assert 'alt_0' in results['predictions']
        assert 'alt_1' in results['predictions']
        assert 'alt_2' in results['predictions']
        
        # Check effect sizes for all alts
        assert len(results['effect_sizes']) == 3
        
    def test_predict_variant_effect_indel(self):
        """Test variant effect prediction for indels."""
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        # Test deletion
        deletion = {
            'chrom': 'chr1',
            'pos': 250000,
            'ref': 'AAAA',
            'alt': ['A']
        }
        
        results = self.oracle.predict_variant_effect(
            variant=deletion,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert 'predictions' in results
        assert 'effect_sizes' in results
        
        # Test insertion
        insertion = {
            'chrom': 'chr1',
            'pos': 250000,
            'ref': 'A',
            'alt': ['ACCC']
        }
        
        results = self.oracle.predict_variant_effect(
            variant=insertion,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert 'predictions' in results
        assert 'effect_sizes' in results
        
    def test_create_tracks_functionality(self):
        """Test track creation functionality."""
        test_seq = "ACGT" * (self.oracle._context_size // 4)
        
        # Test with track creation
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.oracle.predict_region_replacement(
                genomic_region=None,
                seq=test_seq,
                assay_ids=['DNase:K562'],
                create_tracks=True,
                output_dir=tmpdir
            )
            
            # Check track objects
            assert len(results['track_objects']) == 1
            track = results['track_objects'][0]
            assert isinstance(track, Track)
            assert track.name == 'DNase:K562'
            
            # Check track files were created
            assert len(results['track_files']) == 1
            assert Path(results['track_files'][0]).exists()
            
    def test_normalization_methods(self):
        """Test different normalization methods."""
        test_seq = "ACGT" * (self.oracle._context_size // 4)
        
        # Test different normalization methods
        for method in ['quantile', 'minmax', 'zscore']:
            results = self.oracle.predict_region_replacement(
                genomic_region=None,
                seq=test_seq,
                assay_ids=['DNase:K562'],
                normalization_method=method
            )
            
            # Check normalized scores are in expected range
            if method == 'minmax':
                assert results['normalized_scores'].min() >= 0
                assert results['normalized_scores'].max() <= 1
            elif method == 'zscore':
                # Should be roughly centered around 0
                assert abs(results['normalized_scores'].mean()) < 1
                
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with unloaded model
        unloaded_oracle = MockOracle()
        unloaded_oracle.loaded = False
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            unloaded_oracle.predict_region_replacement(
                genomic_region=None,
                seq="ACGT" * 100,
                assay_ids=['DNase:K562']
            )
            
        # Test with invalid assay ID
        test_seq = "ACGT" * (self.oracle._context_size // 4)
        with pytest.raises(ValueError, match="Unknown assay ID"):
            self.oracle.predict_region_replacement(
                genomic_region=None,
                seq=test_seq,
                assay_ids=['InvalidAssay']
            )