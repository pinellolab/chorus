"""Sequence manipulation utilities."""

import pysam
from typing import Tuple, Optional, List, Literal
import pandas as pd
import re
from ..core.exceptions import InvalidRegionError, FileFormatError


_VALID_BASES = frozenset("ACGTN")


def normalize_allele(a: object) -> str:
    """Normalize a single allele string to a canonical empty/ACGTN form.

    Accepts the LDlink-style ``"-"`` deletion notation and ``None`` and
    returns the empty string. Strips whitespace, uppercases, and
    validates that the remaining characters are all ``A``/``C``/``G``/
    ``T``/``N``. Raises ``InvalidRegionError`` on anything else.

    Examples
    --------
    >>> normalize_allele("-")
    ''
    >>> normalize_allele("a")
    'A'
    >>> normalize_allele(" CT ")
    'CT'
    >>> normalize_allele(None)
    ''
    """
    if a is None:
        return ""
    if not isinstance(a, str):
        raise InvalidRegionError(
            f"Allele must be a string (or None / '-'); got {type(a).__name__}: {a!r}"
        )
    s = a.strip()
    if s in ("", "-"):
        return ""
    s = s.upper()
    bad = [c for c in s if c not in _VALID_BASES]
    if bad:
        raise InvalidRegionError(
            f"Allele {a!r} contains non-ACGTN characters: {''.join(sorted(set(bad)))}. "
            f"Use '-' or '' for indels; ACGTN bases otherwise."
        )
    return s


def classify_variant(
    ref: str, alt: str,
) -> Literal["snv", "insertion", "deletion", "mnv", "complex"]:
    """Classify a (ref, alt) pair after :func:`normalize_allele`.

    - ``"snv"``: 1-bp substitution (``len(ref) == len(alt) == 1``)
    - ``"insertion"``: ``ref == ""``, alt non-empty
    - ``"deletion"``: alt ``== ""``, ref non-empty
    - ``"mnv"``: same-length multi-base substitution (``len > 1``)
    - ``"complex"``: anything else (different non-zero lengths, VCF-style
      anchored indels like ``A`` → ``AT``)
    """
    if ref == "" and alt == "":
        return "complex"
    if ref == "":
        return "insertion"
    if alt == "":
        return "deletion"
    if len(ref) == len(alt) == 1:
        return "snv"
    if len(ref) == len(alt):
        return "mnv"
    return "complex"


def extract_sequence(
    genomic_region: str | tuple[str, int, int],
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
    if isinstance(genomic_region, tuple):
        chrom, start, end = genomic_region
    elif ":" in genomic_region and "-" in genomic_region:
        # Format: chr1:1000-2000 (1-based inclusive coordinates)
        match = re.match(r'(\w+):(\d+)-(\d+)', genomic_region)
        if not match:
            raise InvalidRegionError(f"Invalid region format: {genomic_region}")
        chrom, start, end = match.groups()
        start, end = int(start), int(end)
        # Convert from 1-based inclusive to 0-based half-open for pysam
        start = start - 1
    elif "\t" in genomic_region:
        # BED format (already 0-based half-open)
        parts = genomic_region.strip().split("\t")
        if len(parts) < 3:
            raise InvalidRegionError(f"Invalid BED format: {genomic_region}")
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
        # BED format is already 0-based, no conversion needed
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


def extract_sequence_with_padding(
    fasta_path: str,
    chrom: str,
    start: int,
    end: int,
    total_length: int,
    return_meta: bool = False
) -> str | Tuple[str, dict[str, int]]:
    """
    Extract a genomic sequence with padding to reach total_length.
    
    Args:
        fasta_path: Path to indexed FASTA file
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (exclusive)
        total_length: Desired total length including padding
        
    Returns:
        DNA sequence padded to total_length with flanking genomic sequence
    """
    try:
        with pysam.FastaFile(fasta_path) as fasta:
            # Get chromosome length
            chrom_length = fasta.get_reference_length(chrom)
            
            # Calculate region length
            region_length = end - start
            
            if region_length >= total_length:
                # If region is already long enough, trim from center
                trim_amount = region_length - total_length
                trim_left = trim_amount // 2

                meta =  {'start_change': trim_left, 'end_change': -(trim_amount-trim_left), 'leftN': 0, 'rightN': 0}
                seq = fasta.fetch(chrom, start + trim_left, start + trim_left + total_length).upper()
                
                if return_meta:
                    return seq, meta
                else:
                    return seq
            
            # Calculate padding needed
            padding_needed = total_length - region_length
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            
            # Calculate extraction coordinates with bounds checking
            extract_start = max(0, start - pad_left)
            extract_end = min(chrom_length, end + pad_right)

            meta = {'start_change': extract_start-start, 'end_change': extract_end-end}
            
            # Extract sequence
            extracted_seq = fasta.fetch(chrom, extract_start, extract_end).upper()
            
            # If we couldn't get enough sequence from the chromosome, pad with N's
            if len(extracted_seq) < total_length:
                if extract_start == 0:
                    # Hit start of chromosome, pad on the left
                    n_pad_left = total_length - len(extracted_seq)
                    extracted_seq = 'N' * n_pad_left + extracted_seq
                    meta['leftN'] = n_pad_left
                    meta['rightN'] = 0
                else:
                    # Hit end of chromosome, pad on the right
                    n_pad_right = total_length - len(extracted_seq)
                    extracted_seq = extracted_seq + 'N' * n_pad_right
                    meta['leftN'] = 0
                    meta['rightN'] = n_pad_right
            else:
                meta['leftN'] = 0
                meta['rightN'] = 0
            
            if return_meta:
                return extracted_seq, meta
            else: 
                return extracted_seq
    except FileNotFoundError:
        raise FileFormatError(f"Genome file not found: {fasta_path}")
    except Exception as e:
        if "index" in str(e).lower():
            raise FileFormatError(
                f"Genome file {fasta_path} is not indexed. "
                f"Please run 'samtools faidx {fasta_path}' to create index."
            )
        raise


def get_centered_window(
    fasta_path: str,
    chrom: str,
    pos_1based: int,
    length: int,
    ref: str,
    alt: str,
    strict: bool = True,
) -> Tuple[str, str]:
    """Build a ref/alt sequence pair centred on a 1-based variant position.

    Chorus coordinates are 1-based inclusive throughout (matches
    dbSNP / UCSC / IGV — see ``chorus/core/base.py`` for the canonical
    docstring). pyfaidx / pysam fetch is 0-based half-open, so this
    helper handles the conversion in one place to avoid the off-by-one
    bugs that plagued hand-rolled variant-sweep scripts.

    Supports SNVs, MNVs, insertions, deletions, and VCF-style anchored
    indels. ``ref`` and ``alt`` may differ in length; ``"-"`` (LDlink
    notation) and ``""`` both mean "no bases on this side". The
    variant starts at position ``length // 2`` in both returned
    sequences. ``alt_seq`` is always exactly ``length`` bp — for a
    deletion, extra genomic flank is fetched on the right to fill the
    window; for an insertion, the right flank is trimmed.

    Args:
        fasta_path: Path to indexed reference FASTA.
        chrom: Chromosome name (e.g. ``"chr1"``).
        pos_1based: 1-based variant position (as reported by dbSNP / gnomAD).
            For pure insertions this is the position of the base
            *immediately to the right of* the insertion (matches LDlink).
        length: Desired output length (e.g. 2114 for ChromBPNet).
        ref: Reference allele. Single base, multi-base (MNV), or
            empty (``""`` / ``"-"`` for pure insertion).
        alt: Alternate allele. Single base, multi-base, or empty
            (``""`` / ``"-"`` for pure deletion).
        strict: If True (default) raise ``ValueError`` on reference-allele
                mismatch against the FASTA. If False, log a warning
                and return the FASTA-true sequence as ``ref_seq``.

    Returns:
        ``(ref_seq, alt_seq)``, each of length ``length``, both upper-case.

    Raises:
        ValueError: when ``strict=True`` and the FASTA bases at
            ``pos_1based:pos_1based+len(ref)`` do not match ``ref``,
            or when both ref and alt are empty.
    """
    import logging
    log = logging.getLogger(__name__)

    ref = normalize_allele(ref)
    alt = normalize_allele(alt)
    if ref == "" and alt == "":
        raise ValueError(
            "Both ref and alt alleles are empty after normalization — "
            "nothing to score."
        )

    # Convert 1-based variant position to 0-based half-open. The variant
    # starts at offset `half` inside each returned window.
    pos_0based = pos_1based - 1
    half = length // 2

    # When alt is shorter than ref (deletion), the alt_seq window pulls
    # extra genomic flank on the right; fetch enough to cover that.
    extra_right = max(0, len(ref) - len(alt))
    fetch_len = length + extra_right
    fetch_start = pos_0based - half
    fetch_end = fetch_start + fetch_len

    big_window = extract_sequence_with_padding(
        fasta_path, chrom, fetch_start, fetch_end, fetch_len,
    )

    # Sanity-check the ref allele against the FASTA when present.
    if len(ref) > 0:
        seq_ref_base = big_window[half:half + len(ref)].upper()
        if seq_ref_base != ref:
            msg = (
                f"Reference allele mismatch at {chrom}:{pos_1based}: "
                f"expected {ref!r}, found {seq_ref_base!r} in FASTA. "
                f"Check coordinate convention (chorus uses 1-based inclusive) "
                f"or strand."
            )
            if strict:
                raise ValueError(msg)
            log.warning(msg)

    # ref_seq: trim the (possibly oversized) big_window down to `length`
    # bp, with the ref allele anchored at `half`.
    ref_seq = big_window[:length]

    # alt_seq: splice. Left flank is the first `half` bp of the genomic
    # context; then alt; then right flank starting after the ref's
    # footprint, padded out to `length`.
    left_flank = big_window[:half]
    right_flank_genome_start = half + len(ref)
    right_flank_needed = length - half - len(alt)
    right_flank = big_window[
        right_flank_genome_start:right_flank_genome_start + right_flank_needed
    ]
    alt_seq = left_flank + alt + right_flank

    if len(alt_seq) != length:
        raise ValueError(
            f"Internal error building alt_seq: got len={len(alt_seq)}, "
            f"expected {length}. ref={ref!r} alt={alt!r} pos={pos_1based}"
        )

    return ref_seq, alt_seq
