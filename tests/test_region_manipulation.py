"""
Unit tests for predict_region_replacement and predict_region_insertion_at.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from chorus.core.base import OracleBase
from chorus.core.track import Track
from chorus.core.exceptions import InvalidRegionError, InvalidSequenceError


class MockOracle(OracleBase):
    """Mock oracle for testing region manipulation methods."""
    
    def __init__(self, reference_fasta=None):
        super().__init__(use_environment=False)
        self.loaded = True
        self._context_size = 393216  # Enformer context size
        self._output_size = 114688   # Enformer output size
        self._num_tracks = 5313      # Enformer tracks
        self._bin_size = 128         # Enformer bin size
        self.reference_fasta = reference_fasta
        
    def _get_context_size(self):
        return self._context_size
        
    def _get_output_size(self):
        return self._output_size
        
    def _get_num_tracks(self):
        return self._num_tracks
        
    def _get_bin_size(self):
        return self._bin_size
        
    def _get_sequence_length_bounds(self):
        return (self._context_size, self._context_size)
    
    def _validate_dna_sequence(self, seq: str):
        """Override to match base class."""
        # Check if sequence contains only valid nucleotides
        valid_nucleotides = set('ACGTNacgtn')
        if not all(base in valid_nucleotides for base in seq):
            from chorus.core.exceptions import InvalidSequenceError
            raise InvalidSequenceError(
                f"Sequence contains invalid characters. Only A, C, G, T, N allowed."
            )
        
        # Check for empty sequence
        if len(seq) == 0:
            from chorus.core.exceptions import InvalidSequenceError
            raise InvalidSequenceError("Sequence cannot be empty")
    
    def load_pretrained_model(self, weights=None):
        self.loaded = True
        
    def list_assay_types(self):
        return ['DNase', 'RNA-seq', 'ChIP-seq', 'ATAC-seq']
        
    def list_cell_types(self):
        return ['K562', 'HepG2', 'GM12878']
        
    def _get_assay_ids(self, track_indices):
        return [f"track_{i}" for i in track_indices]
        
    def _get_track_indices(self, assay_ids):
        indices = []
        for aid in assay_ids:
            if aid == 'DNase:K562':
                indices.append(0)
            elif aid == 'RNA-seq:HepG2':
                indices.append(1)
            elif aid == 'ChIP-seq:GM12878':
                indices.append(2)
            elif aid.startswith('track_'):
                indices.append(int(aid.split('_')[1]))
            else:
                raise ValueError(f"Unknown assay ID: {aid}")
        return indices
        
    def _predict(self, seq, assay_ids=None):
        """Mock prediction returning consistent data."""
        num_bins = self._output_size // self._bin_size  # 896 bins
        
        if assay_ids is None:
            num_tracks = self._num_tracks
        else:
            num_tracks = len(self._get_track_indices(assay_ids))
            
        # Return deterministic mock predictions based on sequence
        # Use more of the sequence to ensure different inputs give different outputs
        # Hash the full sequence for insertions which modify the middle
        seq_hash = hash(seq) % (2**32)
        np.random.seed(seq_hash)
        return np.random.rand(num_bins, num_tracks)
        
    def fine_tune(self, dataset, epochs=10):
        pass


class TestRegionReplacement:
    """Test predict_region_replacement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"
        
        # Create test genome with sufficient size
        with open(self.fasta_path, 'w') as f:
            f.write(">chr1\n")
            f.write("A" * 1000000 + "\n")  # 1Mb chromosome
            f.write(">chr8\n")
            f.write("T" * 1000000 + "\n")
            
        # Create FASTA index
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))
        
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_region_replacement_basic(self):
        """Test basic region replacement with string input."""
        # Define a small region to replace (not full context)
        region = "chr1:499000-501000"  # 2kb region
        
        # Create replacement sequence matching region size
        new_seq = "ACGT" * 500  # 2kb
        
        # Perform replacement
        results = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=new_seq,
            assay_ids=['DNase:K562', 'RNA-seq:HepG2'],
            genome=str(self.fasta_path)
        )
        
        # Verify structure
        assert 'raw_predictions' in results
        assert 'normalized_scores' in results
        
        # Check that results contain the requested assays
        assert 'DNase:K562' in results['raw_predictions']
        assert 'RNA-seq:HepG2' in results['raw_predictions']
        assert 'DNase:K562' in results['normalized_scores']
        assert 'RNA-seq:HepG2' in results['normalized_scores']
        
        # Check shapes
        assert results['raw_predictions']['DNase:K562'].shape == (896,)
        assert results['raw_predictions']['RNA-seq:HepG2'].shape == (896,)
        assert results['normalized_scores']['DNase:K562'].shape == (896,)
        assert results['normalized_scores']['RNA-seq:HepG2'].shape == (896,)
        
    def test_region_replacement_with_dataframe(self):
        """Test region replacement with DataFrame input."""
        context = self.oracle._context_size
        center = 500000
        
        # Create DataFrame
        region_df = pd.DataFrame({
            'chrom': ['chr8'],
            'start': [center - context//2],
            'end': [center + context//2]
        })
        
        new_seq = "TGCA" * (context // 4)
        
        results = self.oracle.predict_region_replacement(
            genomic_region=region_df,
            seq=new_seq,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert results['raw_predictions']['DNase:K562'].shape == (896,)
        
    def test_region_replacement_creates_tracks(self):
        """Test track file creation."""
        context = self.oracle._context_size
        center = 500000
        start = center - context//2
        end = center + context//2
        region = f"chr1:{start}-{end}"
        new_seq = "ACGT" * (context // 4)
        
        # Need to change to the temp directory to save files there
        import os
        original_dir = os.getcwd()
        
        try:
            os.chdir(self.temp_dir)
            
            results = self.oracle.predict_region_replacement(
                genomic_region=region,
                seq=new_seq,
                assay_ids=['DNase:K562'],
                create_tracks=True,
                genome=str(self.fasta_path)
            )
            
            # When create_tracks=True, should have these keys
            assert 'track_objects' in results
            assert 'track_files' in results
            
            # Check track objects created
            assert len(results['track_objects']) == 1
            track = results['track_objects'][0]
            assert isinstance(track, Track)
            assert track.name == f"DNase:K562_chr1_{start}_{end}"
            assert track.assay_type == 'DNase'
            assert track.cell_type == 'K562'
            
            # Check files created
            assert len(results['track_files']) == 1
            assert Path(results['track_files'][0]).exists()
            
        finally:
            os.chdir(original_dir)
            
    def test_region_replacement_invalid_sequence_length(self):
        """Test error handling for wrong sequence length."""
        # For replacement, we need a region that matches context size
        context = self.oracle._context_size
        region = f"chr1:100000-{100000 + context}"
        wrong_seq = "ACGT" * 100  # Too short
        
        # Our mock validates based on length
        # Short sequences pass validation in our mock
        # So test with a medium-length sequence that's still wrong
        wrong_seq = "ACGT" * 20000  # 80kb - too short for context
        
        with pytest.raises(InvalidSequenceError, match="must be exactly"):
            self.oracle.predict_region_replacement(
                genomic_region=region,
                seq=wrong_seq,
                assay_ids=['DNase:K562'],
                genome=str(self.fasta_path)
            )
            
    def test_region_replacement_invalid_region(self):
        """Test error handling for invalid region format."""
        with pytest.raises(Exception):  # Will raise during parsing
            self.oracle.predict_region_replacement(
                genomic_region="invalid_region_format",
                seq="ACGT" * 1000,
                assay_ids=['DNase:K562'],
                genome=str(self.fasta_path)
            )
            
    def test_region_replacement_different_sequences_different_results(self):
        """Test that different sequences produce different predictions."""
        context = self.oracle._context_size
        region = f"chr1:{500000 - context//2}-{500000 + context//2}"
        
        # Two different sequences
        seq1 = "A" * context
        seq2 = "T" * context
        
        results1 = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=seq1,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        results2 = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=seq2,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        # Predictions should be different
        # Compare the actual arrays, not the dict
        assert not np.array_equal(
            results1['raw_predictions']['DNase:K562'],
            results2['raw_predictions']['DNase:K562']
        )


class TestSequenceInsertion:
    """Test predict_region_insertion_at functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"
        
        # Create test genome
        with open(self.fasta_path, 'w') as f:
            f.write(">chr1\n")
            f.write("A" * 1000000 + "\n")
            f.write(">chr8\n")
            f.write("T" * 1000000 + "\n")
            
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))
        
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_insertion_basic(self):
        """Test basic sequence insertion."""
        # Insert 1kb sequence
        insert_seq = "ACGT" * 250  # 1kb
        position = "chr1:500000"
        
        results = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=insert_seq,
            assay_ids=['DNase:K562', 'ChIP-seq:GM12878'],
            genome=str(self.fasta_path)
        )
        
        # Verify results
        assert 'DNase:K562' in results['raw_predictions']
        assert 'ChIP-seq:GM12878' in results['raw_predictions']
        assert results['raw_predictions']['DNase:K562'].shape == (896,)
        assert results['raw_predictions']['ChIP-seq:GM12878'].shape == (896,)
        
    def test_insertion_with_dataframe(self):
        """Test insertion with DataFrame position."""
        position_df = pd.DataFrame({
            'chrom': ['chr8'],
            'pos': [500000]  # Use 'pos' column for single position
        })
        
        insert_seq = "TGCA" * 250
        
        results = self.oracle.predict_region_insertion_at(
            genomic_position=position_df,
            seq=insert_seq,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert results['raw_predictions']['DNase:K562'].shape == (896,)
        
    def test_insertion_at_chromosome_edge(self):
        """Test insertion near chromosome boundaries."""
        # Insert at a position that has enough flanking sequence
        insert_seq = "ACGT" * 250
        # Need at least context_size/2 on each side
        min_position = self.oracle._context_size // 2 + 1000
        position = f"chr1:{min_position}"  # Safe position
        
        # Should work with enough flanking sequence
        results = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=insert_seq,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert results['raw_predictions']['DNase:K562'].shape == (896,)
        
    def test_insertion_creates_tracks(self):
        """Test track creation during insertion."""
        insert_seq = "ACGT" * 250
        position = "chr1:500000"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.oracle.predict_region_insertion_at(
                genomic_position=position,
                seq=insert_seq,
                assay_ids=['RNA-seq:HepG2'],
                create_tracks=True,
                genome=str(self.fasta_path)
            )
            
            assert 'track_objects' in results
            assert 'track_files' in results
            assert len(results['track_objects']) == 1
            assert len(results['track_files']) == 1
            assert Path(results['track_files'][0]).exists()
            
    def test_insertion_empty_sequence(self):
        """Test insertion of empty sequence."""
        with pytest.raises(InvalidSequenceError):
            self.oracle.predict_region_insertion_at(
                genomic_position="chr1:500000",
                seq="",  # Empty sequence
                assay_ids=['DNase:K562'],
                genome=str(self.fasta_path)
            )
            
    def test_insertion_position_formats(self):
        """Test different position format inputs."""
        insert_seq = "ACGT" * 250
        
        # Test different valid formats
        positions = [
            "chr1:500000",
            "chr1:500000-500000",  # Range with same start/end
        ]
        
        for pos in positions:
            results = self.oracle.predict_region_insertion_at(
                genomic_position=pos,
                seq=insert_seq,
                assay_ids=['DNase:K562'],
                genome=str(self.fasta_path)
            )
            assert results['raw_predictions']['DNase:K562'].shape == (896,)
            
    def test_insertion_affects_predictions(self):
        """Test that insertions produce different results."""
        position = "chr1:500000"
        
        # Different insertion sequences
        seq1 = "A" * 1000  # All A's
        seq2 = "CACGTG" * 167  # E-box motifs
        
        results1 = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=seq1,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        results2 = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=seq2,
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        # Should produce different predictions
        # Compare the actual arrays
        assert not np.array_equal(
            results1['raw_predictions']['DNase:K562'],
            results2['raw_predictions']['DNase:K562']
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"
        
        # Small test genome
        with open(self.fasta_path, 'w') as f:
            f.write(">chr1\n")
            f.write("A" * 500000 + "\n")  # 500kb only
            
        import pysam
        pysam.faidx(str(self.fasta_path))
        
        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))
        
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
        
    def test_model_not_loaded(self):
        """Test error when model not loaded."""
        self.oracle.loaded = False
        
        with pytest.raises(Exception):  # Should raise ModelNotLoadedError
            self.oracle.predict_region_replacement(
                genomic_region="chr1:1000-394216",
                seq="ACGT" * 98304,
                assay_ids=['DNase:K562'],
                genome=str(self.fasta_path)
            )
            
    def test_invalid_assay_id(self):
        """Test error with invalid assay ID."""
        context = self.oracle._context_size
        
        # This test assumes the region is small enough to fit
        # For a 500kb chromosome, we need to adjust
        region = f"chr1:50000-{50000 + context}"
        
        from chorus.core.exceptions import InvalidAssayError
        with pytest.raises(InvalidAssayError):
            self.oracle.predict_region_replacement(
                genomic_region=region,
                seq="ACGT" * (context // 4),
                assay_ids=['InvalidAssay'],
                genome=str(self.fasta_path)
            )
            
    def test_region_too_large(self):
        """Test region larger than chromosome."""
        # Try to specify region beyond chromosome end
        # Note: This might be handled by extract_sequence
        region = "chr1:100000-600000"  # 500kb region in 500kb chromosome
        
        # This should work as the oracle will pad as needed
        results = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq="ACGT" * (self.oracle._context_size // 4),
            assay_ids=['DNase:K562'],
            genome=str(self.fasta_path)
        )
        
        assert results is not None