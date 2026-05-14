"""Tests for utility functions."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from chorus.utils import (
    extract_sequence,
    parse_vcf,
    apply_variant,
    reverse_complement,
    validate_sequence,
    get_gc_content,
    quantile_normalize,
    minmax_normalize,
    zscore_normalize,
    normalize_allele,
    classify_variant,
)
from chorus.core.exceptions import InvalidRegionError, FileFormatError


class TestSequenceUtils:
    """Test sequence utility functions."""
    
    def test_reverse_complement(self):
        """Test reverse complement function."""
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("atcg") == "cgat"
        assert reverse_complement("ATCGN") == "NCGAT"
        assert reverse_complement("") == ""
    
    def test_validate_sequence(self):
        """Test sequence validation."""
        assert validate_sequence("ATCGATCG") == True
        assert validate_sequence("ATCGATCGN") == True
        assert validate_sequence("ATCGATCGX") == False
        assert validate_sequence("ATCGATCG", strict=True) == True
        assert validate_sequence("ATCGATCGN", strict=True) == False
    
    def test_get_gc_content(self):
        """Test GC content calculation."""
        assert get_gc_content("AAAA") == 0.0
        assert get_gc_content("CCCC") == 1.0
        assert get_gc_content("ATCG") == 0.5
        assert get_gc_content("ATCGNN") == 0.5  # N's excluded
        assert get_gc_content("NNNN") == 0.0
    
    def test_apply_variant(self):
        """Test variant application."""
        ref_seq = "ATCGATCGATCG"
        
        # SNP
        result = apply_variant(ref_seq, 5, "T", "A")
        assert result == "ATCGAACGATCG"
        
        # Deletion
        result = apply_variant(ref_seq, 5, "TCG", "T")
        assert result == "ATCGATATCG"
        
        # Insertion
        result = apply_variant(ref_seq, 5, "T", "TAAA")
        assert result == "ATCGATAAACGATCG"
        
        # Invalid position
        with pytest.raises(ValueError):
            apply_variant(ref_seq, 100, "A", "T")
        
        # Mismatch
        with pytest.raises(ValueError):
            apply_variant(ref_seq, 5, "A", "T")  # Position 5 is T, not A

    def test_normalize_allele_table(self):
        """LDlink ``"-"`` and ``None`` collapse to ``""``; bases uppercase + validated."""
        assert normalize_allele("-") == ""
        assert normalize_allele("") == ""
        assert normalize_allele(None) == ""
        assert normalize_allele("A") == "A"
        assert normalize_allele("a") == "A"
        assert normalize_allele("  CT  ") == "CT"
        assert normalize_allele("GAAAAAATAAAAA") == "GAAAAAATAAAAA"

        for bad in ("X", "AT?", "-A"):
            with pytest.raises(InvalidRegionError):
                normalize_allele(bad)

    def test_classify_variant(self):
        assert classify_variant("A", "G") == "snv"
        assert classify_variant("A", "") == "deletion"
        assert classify_variant("", "CT") == "insertion"
        assert classify_variant("AT", "GC") == "mnv"
        assert classify_variant("A", "AT") == "complex"  # VCF-style anchored
        assert classify_variant("TA", "TAC") == "complex"

    def test_get_centered_window_indels(self):
        """Indels: alt_seq is `length` bp with the variant at the centre.

        Uses a synthetic single-chrom FASTA so the test stays self-
        contained and doesn't depend on hg38 being present.
        """
        from chorus.utils import get_centered_window
        import pysam

        # All-A test genome on chr1, 10 kb long.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">chr1\n" + "A" * 10000 + "\n")
            fa = f.name
        try:
            pysam.faidx(fa)
            length = 100

            # SNV (backward compat): ref/alt both 1 bp, equal lengths
            r, a = get_centered_window(fa, "chr1", 5000, length, "A", "C")
            assert len(r) == length == len(a)
            assert a[length // 2] == "C"

            # Pure deletion: ref="A" alt=""
            r, a = get_centered_window(fa, "chr1", 5000, length, "A", "")
            assert len(r) == length == len(a)
            # alt has one fewer A at the centre, but is padded back to length
            assert r.count("A") == length
            assert a.count("A") == length  # still all A's since genome is uniform

            # Pure insertion: ref="" alt="GT"
            r, a = get_centered_window(fa, "chr1", 5000, length, "", "GT")
            assert len(r) == length == len(a)
            # The inserted "GT" lives at the centre of alt_seq
            assert a[length // 2 : length // 2 + 2] == "GT"

            # 4-bp deletion (LDlink dash notation)
            r, a = get_centered_window(fa, "chr1", 5000, length, "AAAA", "-")
            assert len(r) == length == len(a)
            assert r[length // 2 : length // 2 + 4] == "AAAA"

            # VCF-style anchored insertion: ref="A" alt="AGT"
            r, a = get_centered_window(fa, "chr1", 5000, length, "A", "AGT")
            assert len(r) == length == len(a)
            assert a[length // 2 : length // 2 + 3] == "AGT"

            # MNV equal length
            r, a = get_centered_window(fa, "chr1", 5000, length, "AA", "GT")
            assert len(r) == length == len(a)
            assert a[length // 2 : length // 2 + 2] == "GT"
        finally:
            os.unlink(fa)
            os.unlink(fa + ".fai")

    def test_parse_vcf(self):
        """Test VCF parsing."""
        # Create temporary VCF file
        vcf_content = """##fileformat=VCFv4.3
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\trs123\tA\tG\t30\tPASS\t.
chr1\t200\trs124\tC\tT,G\t40\tPASS\t.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write(vcf_content)
            vcf_file = f.name
        
        try:
            # Parse VCF
            variants = parse_vcf(vcf_file)
            
            assert len(variants) == 3  # 2 variants, but one has 2 alts
            assert variants.iloc[0]['chrom'] == 'chr1'
            assert variants.iloc[0]['pos'] == 100
            assert variants.iloc[0]['ref'] == 'A'
            assert variants.iloc[0]['alt'] == 'G'
            
            # Check multi-allelic variant
            assert variants.iloc[1]['alt'] == 'T'
            assert variants.iloc[2]['alt'] == 'G'
        
        finally:
            os.unlink(vcf_file)
        
        # Test non-existent file
        with pytest.raises(FileFormatError):
            parse_vcf("nonexistent.vcf")


class TestNormalizationUtils:
    """Test normalization functions."""
    
    def test_quantile_normalize(self):
        """Test quantile normalization."""
        # Test with pandas Series
        values = pd.Series([1, 2, 3, 4, 5])
        normalized = quantile_normalize(values)
        
        assert isinstance(normalized, pd.Series)
        assert len(normalized) == len(values)
        
        # Test with numpy array
        values_np = np.array([1, 2, 3, 4, 5])
        normalized_np = quantile_normalize(values_np)
        
        assert isinstance(normalized_np, np.ndarray)
        assert len(normalized_np) == len(values_np)
        
        # Test with NaN values
        values_nan = pd.Series([1, 2, np.nan, 4, 5])
        normalized_nan = quantile_normalize(values_nan)
        assert pd.isna(normalized_nan[2])
    
    def test_minmax_normalize(self):
        """Test min-max normalization."""
        values = pd.Series([1, 2, 3, 4, 5])
        normalized = minmax_normalize(values)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized[0] == 0.0
        assert normalized[4] == 1.0
        
        # Test with constant values
        constant = pd.Series([3, 3, 3, 3])
        normalized_const = minmax_normalize(constant)
        assert all(normalized_const == 0.5)
    
    def test_zscore_normalize(self):
        """Test z-score normalization."""
        values = pd.Series([1, 2, 3, 4, 5])
        normalized = zscore_normalize(values)
        
        assert abs(normalized.mean()) < 1e-10
        assert abs(normalized.std() - 1.0) < 1e-10
        
        # Test with numpy array
        values_np = np.array([1, 2, 3, 4, 5])
        normalized_np = zscore_normalize(values_np)
        
        assert abs(np.mean(normalized_np)) < 1e-10
        assert abs(np.std(normalized_np) - 1.0) < 1e-10


class TestTrackNormalization:
    """Test track normalization functions."""
    
    def test_normalize_tracks(self):
        """Test normalizing BEDGraph tracks."""
        # Create temporary BEDGraph files
        track1_content = """chr1\t0\t100\t1.0
chr1\t100\t200\t2.0
chr1\t200\t300\t3.0
"""
        track2_content = """chr1\t0\t100\t5.0
chr1\t100\t200\t10.0
chr1\t200\t300\t15.0
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write track files
            track1_file = os.path.join(tmpdir, "track1.bedgraph")
            track2_file = os.path.join(tmpdir, "track2.bedgraph")
            
            with open(track1_file, 'w') as f:
                f.write(track1_content)
            with open(track2_file, 'w') as f:
                f.write(track2_content)
            
            # Normalize tracks
            from chorus.utils.normalization import normalize_tracks
            
            normalized_files = normalize_tracks(
                [track1_file, track2_file],
                ["Track1", "Track2"],
                normalization='minmax'
            )
            
            assert len(normalized_files) == 2
            
            # Check normalized values
            for norm_file in normalized_files:
                assert os.path.exists(norm_file)
                
                # Read normalized track
                data = pd.read_csv(
                    norm_file,
                    sep='\t',
                    skiprows=1,
                    names=['chrom', 'start', 'end', 'value']
                )
                
                # Check min-max normalization
                assert data['value'].min() >= 0.0
                assert data['value'].max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])