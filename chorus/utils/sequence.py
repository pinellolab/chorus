"""Sequence manipulation utilities."""

import pysam
from typing import Tuple, Optional, List
import pandas as pd
import re
from ..core.exceptions import InvalidRegionError, FileFormatError


def extract_sequence(
    genomic_region: str,
    genome: str = "hg38.fa"
) -> str:
    """
    Extract DNA sequence from reference genome.
    
    Args:
        genomic_region: Format "chr1:1000-2000" or BED entry
        genome: Path to indexed FASTA file
        
    Returns:
        DNA sequence string
    """
    # Parse region
    if ":" in genomic_region and "-" in genomic_region:
        # Format: chr1:1000-2000
        match = re.match(r'(\w+):(\d+)-(\d+)', genomic_region)
        if not match:
            raise InvalidRegionError(f"Invalid region format: {genomic_region}")
        chrom, start, end = match.groups()
        start, end = int(start), int(end)
    elif "\t" in genomic_region:
        # BED format
        parts = genomic_region.strip().split("\t")
        if len(parts) < 3:
            raise InvalidRegionError(f"Invalid BED format: {genomic_region}")
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
    else:
        raise InvalidRegionError(
            f"Invalid region format: {genomic_region}. "
            "Expected 'chr1:1000-2000' or BED format"
        )
    
    # Validate coordinates
    if start < 0:
        raise InvalidRegionError(f"Start position cannot be negative: {start}")
    if end <= start:
        raise InvalidRegionError(f"End position must be greater than start: {start}-{end}")
    
    # Extract sequence
    try:
        fasta = pysam.FastaFile(genome)
        
        # Check if chromosome exists
        if chrom not in fasta.references:
            fasta.close()
            raise InvalidRegionError(f"Chromosome {chrom} not found in {genome}")
        
        # Check if coordinates are within chromosome bounds
        chrom_length = fasta.get_reference_length(chrom)
        if end > chrom_length:
            raise InvalidRegionError(
                f"End position {end} exceeds chromosome {chrom} length {chrom_length}"
            )
        
        sequence = fasta.fetch(chrom, start, end)
        fasta.close()
        
        return sequence.upper()
    
    except FileNotFoundError:
        raise FileFormatError(f"Genome file not found: {genome}")
    except Exception as e:
        if "index" in str(e).lower():
            raise FileFormatError(
                f"Genome file {genome} is not indexed. "
                "Please run 'samtools faidx {genome}' to create index."
            )
        raise


def parse_vcf(vcf_file: str) -> pd.DataFrame:
    """
    Parse VCF file into DataFrame.
    
    Args:
        vcf_file: Path to VCF file
        
    Returns:
        DataFrame with columns: chrom, pos, id, ref, alt, qual, filter, info
    """
    vcf_data = []
    
    try:
        with open(vcf_file, 'r') as f:
            for line in f:
                # Skip header lines
                if line.startswith('#'):
                    continue
                
                # Parse variant line
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                
                # Handle multiple alternate alleles
                alts = parts[4].split(',')
                for alt in alts:
                    vcf_data.append({
                        'chrom': parts[0],
                        'pos': int(parts[1]),
                        'id': parts[2],
                        'ref': parts[3],
                        'alt': alt,
                        'qual': float(parts[5]) if parts[5] != '.' else None,
                        'filter': parts[6],
                        'info': parts[7] if len(parts) > 7 else ''
                    })
    
    except FileNotFoundError:
        raise FileFormatError(f"VCF file not found: {vcf_file}")
    except Exception as e:
        raise FileFormatError(f"Error parsing VCF file: {str(e)}")
    
    if not vcf_data:
        raise FileFormatError(f"No variants found in VCF file: {vcf_file}")
    
    return pd.DataFrame(vcf_data)


def apply_variant(reference_seq: str, position: int, ref: str, alt: str) -> str:
    """
    Apply variant to reference sequence.
    
    Args:
        reference_seq: Reference DNA sequence
        position: 0-based position of variant
        ref: Reference allele
        alt: Alternate allele
        
    Returns:
        Modified sequence with variant applied
    """
    # Validate inputs
    if position < 0 or position >= len(reference_seq):
        raise ValueError(f"Position {position} is outside sequence bounds [0, {len(reference_seq)})")
    
    # Check if reference allele matches
    ref_len = len(ref)
    if position + ref_len > len(reference_seq):
        raise ValueError(
            f"Reference allele '{ref}' at position {position} extends beyond sequence"
        )
    
    seq_ref = reference_seq[position:position + ref_len]
    if seq_ref.upper() != ref.upper():
        raise ValueError(
            f"Reference allele mismatch at position {position}: "
            f"expected '{ref}', found '{seq_ref}'"
        )
    
    # Apply variant
    new_seq = reference_seq[:position] + alt + reference_seq[position + ref_len:]
    
    return new_seq


def reverse_complement(seq: str) -> str:
    """
    Get reverse complement of DNA sequence.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Reverse complement sequence
    """
    complement = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(complement)[::-1]


def validate_sequence(seq: str, strict: bool = False) -> bool:
    """
    Validate DNA sequence.
    
    Args:
        seq: DNA sequence to validate
        strict: If True, only ACGT allowed. If False, N also allowed.
        
    Returns:
        True if sequence is valid, False otherwise
    """
    if strict:
        valid_chars = set('ACGTacgt')
    else:
        valid_chars = set('ACGTNacgtn')
    
    return all(base in valid_chars for base in seq)


def get_gc_content(seq: str) -> float:
    """
    Calculate GC content of sequence.
    
    Args:
        seq: DNA sequence
        
    Returns:
        GC content as fraction (0-1)
    """
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    total_count = len(seq) - seq.count('N')
    
    if total_count == 0:
        return 0.0
    
    return gc_count / total_count


def split_sequence_into_windows(
    seq: str,
    window_size: int,
    step_size: Optional[int] = None
) -> List[str]:
    """
    Split sequence into overlapping or non-overlapping windows.
    
    Args:
        seq: DNA sequence
        window_size: Size of each window
        step_size: Step between windows (default: window_size for non-overlapping)
        
    Returns:
        List of sequence windows
    """
    if step_size is None:
        step_size = window_size
    
    windows = []
    for i in range(0, len(seq) - window_size + 1, step_size):
        windows.append(seq[i:i + window_size])
    
    return windows


def pad_sequence(seq: str, target_length: int, pad_char: str = 'N') -> str:
    """
    Pad sequence to target length.
    
    Args:
        seq: DNA sequence
        target_length: Desired length
        pad_char: Character to use for padding
        
    Returns:
        Padded sequence
    """
    if len(seq) >= target_length:
        return seq[:target_length]
    
    pad_needed = target_length - len(seq)
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    
    return pad_char * pad_left + seq + pad_char * pad_right