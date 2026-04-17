"""Gene annotation utilities for Chorus.

This module provides functionality for working with gene annotations,
particularly for visualizing genomic regions in the context of genes
and for analyzing effects on gene expression.

Performance notes:
    The GTF file (~1GB) is loaded into memory once on first use and cached
    for the lifetime of the process.  Subsequent queries use fast DataFrame
    filtering instead of re-scanning the file.  If pysam is available and
    a tabix-indexed GTF exists, region queries use O(1) tabix lookups.
"""

import os
import gzip
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import pandas as pd
import requests
from tqdm import tqdm
import shlex
import subprocess
import shutil
logger = logging.getLogger(__name__)

from ..core.globals import CHORUS_ANNOTATIONS_DIR


class AnnotationManager:
    """Manager for gene annotations (GTF files).

    Caches parsed GTF data in memory after first load for fast repeated queries.
    """

    # Default annotation sources
    ANNOTATION_SOURCES = {
        'gencode_v48_basic': {
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.basic.annotation.gtf.gz',
            'description': 'GENCODE v48 basic gene annotation (hg38)',
            'genome': 'hg38'
        },
        'gencode_v48_comprehensive': {
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz',
            'description': 'GENCODE v48 comprehensive gene annotation (hg38)',
            'genome': 'hg38'
        },
        'gencode_v47_basic': {
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz',
            'description': 'GENCODE v47 basic gene annotation (hg38)',
            'genome': 'hg38'
        }
    }

    def __init__(self, annotations_dir: Optional[str] = None):
        """Initialize annotation manager.

        Args:
            annotations_dir: Directory to store annotation files.
                           Defaults to chorus/annotations/
        """
        if annotations_dir is None:
            # Default to annotations directory in chorus root
            annotations_dir = CHORUS_ANNOTATIONS_DIR

        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches keyed by gtf_path
        self._gene_cache: dict[str, pd.DataFrame] = {}
        self._exon_cache: dict[str, pd.DataFrame] = {}
        self._transcript_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # GTF loading with caching
    # ------------------------------------------------------------------

    def _load_gtf_features(self, gtf_path: Union[str, Path],
                           feature_types: list[str]) -> pd.DataFrame:
        """Parse a GTF file and return a DataFrame of matching features.

        Results are cached in memory so subsequent calls are instant.
        """
        gtf_path = str(gtf_path)
        cache_key = f"{gtf_path}:{'|'.join(sorted(feature_types))}"

        # Check memory cache
        if cache_key in self._gene_cache:
            return self._gene_cache[cache_key]

        logger.info("Loading GTF features (%s) from %s (one-time)...",
                     ", ".join(feature_types), Path(gtf_path).name)

        if gtf_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        feature_set = set(feature_types)
        rows = []

        with open_func(gtf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split('\t', 9)
                if len(parts) < 9:
                    continue
                if parts[2] not in feature_set:
                    continue

                attr_dict = {}
                for attr in parts[8].strip().split(';'):
                    attr = attr.strip()
                    if attr:
                        kv = attr.split(' ', 1)
                        if len(kv) == 2:
                            attr_dict[kv[0]] = kv[1].strip('"')

                rows.append({
                    'chrom': parts[0],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'strand': parts[6],
                    'feature': parts[2],
                    'gene_name': attr_dict.get('gene_name', ''),
                    'gene_id': attr_dict.get('gene_id', ''),
                    'gene_type': attr_dict.get('gene_type', ''),
                    'transcript_id': attr_dict.get('transcript_id', ''),
                    'transcript_type': attr_dict.get('transcript_type', ''),
                    'exon_number': attr_dict.get('exon_number', ''),
                    'level': attr_dict.get('level', ''),
                })

        df = pd.DataFrame(rows)
        self._gene_cache[cache_key] = df
        logger.info("Cached %d %s features from GTF", len(df),
                     "/".join(feature_types))
        return df

    def _get_genes_df(self, gtf_path: Union[str, Path]) -> pd.DataFrame:
        """Get cached DataFrame of gene features."""
        return self._load_gtf_features(gtf_path, ['gene'])

    def _get_exons_df(self, gtf_path: Union[str, Path]) -> pd.DataFrame:
        """Get cached DataFrame of exon features."""
        return self._load_gtf_features(gtf_path, ['exon'])

    def _get_transcripts_df(self, gtf_path: Union[str, Path]) -> pd.DataFrame:
        """Get cached DataFrame of transcript features."""
        return self._load_gtf_features(gtf_path, ['transcript'])

    def annotation_exists(self, annotation_path: Path) -> str | None:
        if annotation_path.exists():
            return str(annotation_path)
        elif Path(str(annotation_path).replace('.gtf.gz', '.gtf')).exists():
            return str(Path(str(annotation_path).replace('.gtf.gz', '.gtf')))
        else:
            return None
  
    
    def download_annotation(self, annotation_id: str = 'gencode_v48_basic', 
                          force: bool = False) -> Path:
        """Download gene annotation file if it doesn't exist.
        
        Args:
            annotation_id: ID of annotation to download from ANNOTATION_SOURCES
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded annotation file
        """
        if annotation_id not in self.ANNOTATION_SOURCES:
            raise ValueError(f"Unknown annotation ID: {annotation_id}. "
                           f"Available: {list(self.ANNOTATION_SOURCES.keys())}")
        
        annotation_info = self.ANNOTATION_SOURCES[annotation_id]
        url = annotation_info['url']
        filename = os.path.basename(url)
        filepath = self.annotations_dir / filename
        
        # Check if already downloaded

        existing_path = self.annotation_exists(filepath)
        if existing_path is not None and not force:
            logger.info(f"Annotation file already exists: {existing_path}")
            return existing_path
        
        # Download with progress bar
        logger.info(f"Downloading {annotation_info['description']}...")
        logger.info(f"URL: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=f"Downloading {filename}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded annotation to: {filepath}")
        logger.info(f"Sorting annotation...")
        filepath = self.sort_annotation(filepath)
        logger.info(f"Sorted annotation to: {filepath}")
        return filepath

    def sort_annotation(self, annotation_path: Path) -> Path:
        gtf_path_no_gz = Path(str(annotation_path).replace('.gtf.gz', '.gtf'))
        with gzip.open(annotation_path, 'rb') as f_in:
            with open(gtf_path_no_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(annotation_path)
        sorted_gtf_path = sort_gtf(gtf_path_no_gz, gtf_path_no_gz.with_suffix('.sorted.gtf'))
        shutil.move(sorted_gtf_path, gtf_path_no_gz)
        return gtf_path_no_gz
    
    def get_annotation_path(self, annotation_id: str = 'gencode_v48_basic',
                          auto_download: bool = True) -> str | None:
        """Get path to annotation file, downloading if necessary.
        
        Args:
            annotation_id: ID of annotation
            auto_download: Whether to download if not found
            
        Returns:
            Path to annotation file or None if not available
        """
        if annotation_id not in self.ANNOTATION_SOURCES:
            # Check if it's a direct filename in annotations dir
            filepath = self.annotations_dir / annotation_id
            if filepath.exists():
                return filepath
            return None
        
        # Check if file exists
        annotation_info = self.ANNOTATION_SOURCES[annotation_id]
        url = annotation_info['url']
        filename = os.path.basename(url)
        filepath = self.annotations_dir / filename
        
        if filepath.exists():
            return filepath
        
        if auto_download:
            return self.download_annotation(annotation_id)
        
        return None

    def list_annotations(self) -> Dict[str, Dict]:
        """List available annotations.
        
        Returns:
            Dictionary of available annotations with their info
        """
        available = {}
        
        # Check predefined sources
        for ann_id, info in self.ANNOTATION_SOURCES.items():
            filename = os.path.basename(info['url'])
            filepath = self.annotations_dir / filename
            info_copy = info.copy()
            info_copy['downloaded'] = filepath.exists()
            info_copy['path'] = str(filepath) if filepath.exists() else None
            available[ann_id] = info_copy
        
        # Check for other GTF files in directory
        for gtf_file in self.annotations_dir.glob("*.gtf*"):
            if gtf_file.name not in [os.path.basename(info['url']) 
                                     for info in self.ANNOTATION_SOURCES.values()]:
                available[gtf_file.stem] = {
                    'description': f'Local GTF file: {gtf_file.name}',
                    'downloaded': True,
                    'path': str(gtf_file),
                    'genome': 'unknown'
                }
        
        return available
    
    def extract_genes_in_region(self, gtf_path: Union[str, Path],
                               chrom: str, start: int, end: int,
                               feature_types: List[str] = ['gene']) -> pd.DataFrame:
        """Extract genes in a specific genomic region from GTF.

        Uses cached in-memory DataFrame for fast repeated queries.

        Args:
            gtf_path: Path to GTF file (can be gzipped)
            chrom: Chromosome name
            start: Start position
            end: End position
            feature_types: Types of features to extract (default: ['gene'])

        Returns:
            DataFrame with gene information
        """
        df = self._load_gtf_features(gtf_path, feature_types)
        if len(df) == 0:
            return df
        mask = (
            (df['chrom'] == chrom) &
            (df['end'] >= start) &
            (df['start'] <= end)
        )
        return df[mask].reset_index(drop=True)
    
    def get_exon_positions(self, gtf_path: Union[str, Path],
                          gene_name: Optional[str] = None,
                          gene_id: Optional[str] = None,
                          chrom: Optional[str] = None) -> pd.DataFrame:
        """Extract exon coordinates from GTF for a gene.

        Uses cached in-memory DataFrame with gene_name index for fast lookups.

        Args:
            gtf_path: Path to GTF file
            gene_name: Filter by gene name (e.g., 'MYC')
            gene_id: Filter by gene ID (e.g., 'ENSG00000136997')
            chrom: Filter by chromosome

        Returns:
            DataFrame with chrom, start, end, strand, gene_name, gene_id,
            transcript_id, exon_number columns.
        """
        gtf_path = str(gtf_path)

        # Build gene_name-indexed lookup on first call
        if gtf_path not in self._exon_cache:
            df = self._get_exons_df(gtf_path)
            if len(df) > 0:
                self._exon_cache[gtf_path] = df.groupby('gene_name')
            else:
                return df

        grouped = self._exon_cache[gtf_path]

        if gene_name:
            try:
                result = grouped.get_group(gene_name)
            except KeyError:
                return pd.DataFrame()
            if chrom:
                result = result[result['chrom'] == chrom]
            if gene_id:
                result = result[result['gene_id'] == gene_id]
            return result.reset_index(drop=True)

        # Fallback: no gene_name filter
        df = self._get_exons_df(gtf_path)
        mask = pd.Series(True, index=df.index)
        if chrom:
            mask &= df['chrom'] == chrom
        if gene_id:
            mask &= df['gene_id'] == gene_id
        return df[mask].reset_index(drop=True)

    def get_tss_positions(self, gtf_path: Union[str, Path],
                         gene_name: Optional[str] = None,
                         gene_id: Optional[str] = None,
                         chrom: Optional[str] = None) -> pd.DataFrame:
        """Extract TSS (Transcription Start Site) positions for genes.

        Uses cached in-memory DataFrame for fast repeated queries.

        Args:
            gtf_path: Path to GTF file
            gene_name: Filter by gene name (e.g., 'GATA1')
            gene_id: Filter by gene ID (e.g., 'ENSG00000102145')
            chrom: Filter by chromosome

        Returns:
            DataFrame with TSS positions
        """
        df = self._get_transcripts_df(gtf_path)
        if len(df) == 0:
            return df

        mask = pd.Series(True, index=df.index)
        if chrom:
            mask &= df['chrom'] == chrom
        if gene_name:
            mask &= df['gene_name'] == gene_name
        if gene_id:
            mask &= df['gene_id'] == gene_id

        filtered = df[mask].copy()
        if len(filtered) == 0:
            return pd.DataFrame()

        # Compute TSS based on strand
        filtered['tss'] = filtered.apply(
            lambda r: r['start'] if r['strand'] == '+' else r['end'],
            axis=1,
        )
        filtered['transcript_start'] = filtered['start']
        filtered['transcript_end'] = filtered['end']

        return filtered[['chrom', 'tss', 'strand', 'gene_name', 'gene_id',
                          'transcript_id', 'transcript_type',
                          'transcript_start', 'transcript_end']].reset_index(drop=True)


# Convenience functions
_manager = None

def get_annotation_manager() -> AnnotationManager:
    """Get the global annotation manager instance."""
    global _manager
    if _manager is None:
        _manager = AnnotationManager()
    return _manager


def download_gencode(version: str = 'v48', annotation_type: str = 'basic') -> Path:
    """Download GENCODE annotation.
    
    Args:
        version: GENCODE version (e.g., 'v48', 'v47')
        annotation_type: 'basic' or 'comprehensive'
        
    Returns:
        Path to downloaded annotation file
    """
    annotation_id = f'gencode_{version}_{annotation_type}'
    gtf_path = get_annotation_manager().download_annotation(annotation_id)
    
    return gtf_path


def sort_gtf(gtf_path: str, output_path: str) -> str:
    """Sort GTF file using gtfsort (Linux) or Python fallback (macOS).

    Args:
        gtf_path: Path to GTF file
        output_path: Path for sorted output

    Returns:
        Path to sorted GTF file
    """
    if shutil.which("gtfsort"):
        cmd = shlex.split(f"gtfsort --input {gtf_path} --output {output_path}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to sort GTF file: {res.stderr}")
        return output_path

    # Fallback: sort by chromosome and position using Python
    logger.info("gtfsort not found (Linux-only); using Python fallback for GTF sorting")
    header_lines = []
    data_lines = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line)

    def _sort_key(line):
        parts = line.split('\t', 5)
        chrom = parts[0]
        # Numeric sort for chr1-22, then chrX, chrY, chrM, others
        chrom_order = chrom.replace('chr', '')
        try:
            chrom_num = int(chrom_order)
        except ValueError:
            chrom_num = {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(chrom_order, 100)
        return (chrom_num, int(parts[3]) if len(parts) > 3 else 0)

    data_lines.sort(key=_sort_key)
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.writelines(data_lines)
    return output_path

def get_genes_in_region(chrom: str, start: int, end: int,
                       annotation: str = 'gencode_v48_basic') -> pd.DataFrame:
    """Get genes in a genomic region.
    
    Args:
        chrom: Chromosome
        start: Start position
        end: End position
        annotation: Annotation to use
        
    Returns:
        DataFrame with gene information
    """
    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")
    
    return manager.extract_genes_in_region(gtf_path, chrom, start, end)


def get_gene_tss(gene_name: str, annotation: str = 'gencode_v48_basic') -> pd.DataFrame:
    """Get TSS positions for a gene.

    Args:
        gene_name: Gene name (e.g., 'GATA1')
        annotation: Annotation to use

    Returns:
        DataFrame with TSS positions
    """
    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")
    return manager.get_tss_positions(gtf_path, gene_name=gene_name)


def get_gene_exons(gene_name: str, annotation: str = 'gencode_v48_basic',
                   merge: bool = True) -> pd.DataFrame:
    """Get exon coordinates for a gene, optionally merged across transcripts.

    When merge=True (default), overlapping exons from different transcripts are
    merged into a union to avoid double-counting when summing RNA-seq signal.

    Args:
        gene_name: Gene symbol (e.g., 'MYC', 'TP53')
        annotation: Annotation to use
        merge: Whether to merge overlapping exons across transcripts

    Returns:
        DataFrame with chrom, start, end, strand, gene_name columns.
    """
    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")
    exons = manager.get_exon_positions(gtf_path, gene_name=gene_name)

    if len(exons) == 0 or not merge:
        return exons

    # Merge overlapping exons: sort by start, then merge intervals
    merged_rows = []
    for (chrom, strand, gname), group in exons.groupby(['chrom', 'strand', 'gene_name']):
        intervals = sorted(zip(group['start'], group['end']), key=lambda x: x[0])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            merged_rows.append({
                'chrom': chrom,
                'start': s,
                'end': e,
                'strand': strand,
                'gene_name': gname,
            })

    return pd.DataFrame(merged_rows)


# ---------------------------------------------------------------------------
# ENCODE SCREEN cCRE utilities
# ---------------------------------------------------------------------------

_CCRE_URL = "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"
_CCRE_FILENAME = "GRCh38-cCREs.bed"

_ccre_cache: pd.DataFrame | None = None


def get_screen_ccres(cache_dir: str | None = None) -> pd.DataFrame:
    """Load ENCODE SCREEN cCREs (candidate cis-Regulatory Elements).

    Downloads the Registry V4 BED file on first call and caches it.
    Returns a DataFrame with columns: chrom, start, end, ccre_id, element_id, category.

    Categories: PLS (promoter-like), pELS (proximal enhancer-like),
    dELS (distal enhancer-like), CA-CTCF, CA-H3K4me3, CA-TF, CA, TF.
    """
    global _ccre_cache
    if _ccre_cache is not None:
        return _ccre_cache

    if cache_dir is None:
        cache_dir = str(CHORUS_ANNOTATIONS_DIR)
    bed_path = Path(cache_dir) / _CCRE_FILENAME

    if not bed_path.exists():
        logger.info("Downloading SCREEN cCREs from %s ...", _CCRE_URL)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(_CCRE_URL, str(bed_path))
        logger.info("Downloaded %s", bed_path)

    logger.info("Loading %s ...", bed_path)
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "end", "ccre_id", "element_id", "category"],
        dtype={"chrom": str, "start": int, "end": int},
    )
    # Filter to main chromosomes
    valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
    df = df[df["chrom"].isin(valid_chroms)].copy()
    _ccre_cache = df
    logger.info("Loaded %d cCREs across %d categories", len(df), df["category"].nunique())
    return df


def sample_ccre_positions(
    n_per_category: dict[str, int] | None = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Sample genomic positions from SCREEN cCREs with stratification.

    Args:
        n_per_category: Dict mapping category -> number of positions.
            Default: balanced across PLS, dELS, pELS, CA-CTCF, CA-H3K4me3, CA-TF, CA, TF.
        seed: Random seed.

    Returns:
        List of (chrom, center_position) tuples.
    """
    import random

    if n_per_category is None:
        n_per_category = {
            "PLS": 5000,
            "dELS": 5000,
            "pELS": 3000,
            "CA-CTCF": 2000,
            "CA-H3K4me3": 2000,
            "CA-TF": 1500,
            "CA": 1500,
            "TF": 1000,
        }

    df = get_screen_ccres()
    rng = random.Random(seed)
    positions = []

    for category, n in n_per_category.items():
        cat_df = df[df["category"] == category]
        if len(cat_df) == 0:
            logger.warning("No cCREs for category '%s'", category)
            continue
        indices = rng.sample(range(len(cat_df)), min(n, len(cat_df)))
        for idx in indices:
            row = cat_df.iloc[idx]
            center = (row["start"] + row["end"]) // 2
            positions.append((row["chrom"], center))

    rng.shuffle(positions)
    logger.info("Sampled %d positions from %d cCRE categories",
                len(positions), len(n_per_category))
    return positions