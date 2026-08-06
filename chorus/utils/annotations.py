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
import random
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

        # Clean up any stale coolbox/tabix artefacts pointing at the old
        # GTF. coolbox re-bgzips + re-indexes the GTF on first use; if a
        # leftover ``.bgz``/``.tbi`` pair from a previous download lingers,
        # its tabix index points at byte offsets in the old .bgz that no
        # longer match the new one. The next ``tabix -p gff <file>.bgz``
        # call then fails with "index file exists" (tabix requires ``-f``
        # to overwrite). Deleting them here lets coolbox regenerate a
        # consistent pair on its first GTF() call.
        for suffix in (".bgz", ".bgz.tbi", ".gz.tbi"):
            stale = filepath.with_suffix(filepath.suffix + suffix)
            if stale.exists():
                stale.unlink()
                logger.info(f"Removed stale index artefact: {stale}")

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


# ---------------------------------------------------------------------------
# Coolbox gene-track helper — workaround for a coolbox/oxbow interop bug
# ---------------------------------------------------------------------------

def make_gene_track(gtf_path: Union[str, Path], **kwargs):
    """Return a coolbox ``GTF`` track configured for reliable rendering.

    coolbox 0.4.x auto-selects the oxbow-backed tab reader for bgzipped
    GTF files. That reader declares nine columns (including ``attributes``)
    but only yields the first eight, so ``GTF.fetch_intervals`` then blows
    up with ``KeyError: 'attributes'`` while trying to extract
    ``gene_name`` for the plot. Notebook cells plotting gene annotations
    produced the traceback verbatim in their output.

    Fix: construct the ``GTF`` track as usual, then swap its reader for
    :class:`TabFileReaderInMemory`, which parses all nine GTF columns
    correctly via pandas. The in-memory reader loads the GTF on
    construction (~80 MB for gencode basic); acceptable for notebook use
    and one-shot rendering — queries are plain pandas filters afterwards.

    Parameters
    ----------
    gtf_path:
        Path to a sorted + bgzipped + tabix-indexed GENCODE GTF (as
        produced by :func:`download_gencode`).
    **kwargs:
        Forwarded to ``coolbox.api.GTF`` (``row_filter``, ``color``,
        ``height``, ``name_attr`` — see coolbox docs).

    Returns
    -------
    ``coolbox.api.GTF`` instance usable as ``frame + make_gene_track(path)``.
    """
    from coolbox.api import GTF
    from coolbox.utilities.reader.tab import (
        TabFileReaderInMemory, FMT2COLUMNS,
    )

    track = GTF(str(gtf_path), **kwargs)
    track.reader = TabFileReaderInMemory(
        str(gtf_path), columns=FMT2COLUMNS["gtf"],
    )
    return track


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


# ---------------------------------------------------------------------------
# Meuleman et al. DHS index (hg38, ~3.6 M peaks)
# ---------------------------------------------------------------------------

_DHS_VOCAB_CACHE: "pd.DataFrame | None" = None


_DHS_VOCAB_HF_REPO = "lucapinello/chorus-backgrounds"
_DHS_VOCAB_HF_FILENAME = "dhs_vocabulary_hg38.txt.gz"
_DHS_VOCAB_SHA256 = "0a4d215026744780ce7f562244e6f46b6387bab9875ca56ad543d30a024c1c48"


def load_dhs_vocabulary(dhs_path: "str | None" = None) -> "pd.DataFrame":
    """Load the Meuleman et al. DHS Index vocabulary (hg38).

    Columns returned: ``seqname, start, end, identifier, mean_signal,
    numsamples, summit, component``.  Filtered to autosomes chr1–22.
    Cached in process so repeated CDF builds don't re-parse the 90 MB
    bgzip every call.

    The file is auto-downloaded from
    ``huggingface.co/datasets/lucapinello/chorus-backgrounds`` on first
    use (~90 MB, sha256 ``0a4d2150…1c1c48``) and cached at
    ``annotations/dhs_vocabulary_hg38.txt.gz`` in the repo root.  This
    keeps every chorus install (every shard of a multi-GPU CDF rebuild,
    every fresh-clone audit) working from byte-identical input — no
    manual ``gdown`` step required.

    The original distribution lives at
    ``meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz``;
    our HF mirror is a verbatim copy.
    """
    global _DHS_VOCAB_CACHE
    if _DHS_VOCAB_CACHE is not None:
        return _DHS_VOCAB_CACHE

    if dhs_path is None:
        path = Path(CHORUS_ANNOTATIONS_DIR) / "dhs_vocabulary_hg38.txt.gz"
    else:
        path = Path(dhs_path)

    if not path.exists():
        # Auto-fetch from HuggingFace.  Same pattern as per-track NPZ
        # downloads — no external Google Drive dependency, no manual
        # gdown step needed.
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"DHS vocabulary not found at {path} and huggingface_hub "
                f"is not installed.  Install it (`pip install huggingface_hub`) "
                f"to enable auto-fetch from {_DHS_VOCAB_HF_REPO}, or download "
                f"manually with: gdown --id 16wbuNmHnwsek3USWM04nR535vPavNZka "
                f"-O {path}."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DHS vocabulary not cached at %s — downloading from "
            "huggingface.co/datasets/%s ...",
            path, _DHS_VOCAB_HF_REPO,
        )
        downloaded = hf_hub_download(
            repo_id=_DHS_VOCAB_HF_REPO,
            filename=_DHS_VOCAB_HF_FILENAME,
            repo_type="dataset",
            local_dir=str(path.parent),
        )
        # hf_hub_download usually returns the same path; if not, move it.
        dl_path = Path(downloaded)
        if dl_path.resolve() != path.resolve() and dl_path.exists():
            dl_path.replace(path)
        if not path.exists():
            raise FileNotFoundError(
                f"DHS vocabulary download finished but file not found at "
                f"{path}. Tried HF mirror {_DHS_VOCAB_HF_REPO}."
            )
        logger.info("DHS vocabulary cached at %s", path)
    df = pd.read_csv(
        path, sep="\t", compression="gzip", low_memory=False,
        usecols=["seqname", "start", "end", "identifier",
                 "mean_signal", "numsamples", "summit", "component"],
    )
    valid = {f"chr{i}" for i in range(1, 23)}
    df = df[df["seqname"].isin(valid)].copy()
    _DHS_VOCAB_CACHE = df
    logger.info("Loaded %d DHS peaks across %d components",
                len(df), df["component"].nunique())
    return df


def sample_dhs_positions(
    n: int,
    dhs_path: "str | None" = None,
    min_numsamples: "int | None" = None,
    max_numsamples: "int | None" = None,
    min_signal_quantile: "float | None" = None,
    max_signal_quantile: "float | None" = None,
    seed: int = 42,
) -> "list[tuple[str, int]]":
    """Sample (chrom, summit) positions from the Meuleman DHS vocabulary.

    Simple seeded random sampling.  Optional filters:

    - ``min_numsamples`` / ``max_numsamples`` constrain cell-type
      specificity (1 = highly specific, 733 = ubiquitous).
    - ``min_signal_quantile`` / ``max_signal_quantile`` constrain peak
      strength via the empirical ``mean_signal`` distribution.

    Returns at most ``n`` positions.
    """
    df = load_dhs_vocabulary(dhs_path)
    if min_numsamples is not None:
        df = df[df["numsamples"] >= min_numsamples]
    if max_numsamples is not None:
        df = df[df["numsamples"] <= max_numsamples]
    if min_signal_quantile is not None:
        lo = df["mean_signal"].quantile(min_signal_quantile)
        df = df[df["mean_signal"] >= lo]
    if max_signal_quantile is not None:
        hi = df["mean_signal"].quantile(max_signal_quantile)
        df = df[df["mean_signal"] <= hi]
    rng = random.Random(seed)
    k = min(n, len(df))
    chosen = rng.sample(range(len(df)), k)
    return [
        (df.iloc[i]["seqname"], int(df.iloc[i]["summit"]))
        for i in chosen
    ]

# ---------------------------------------------------------------------------
# Per-gene exon index, shared by the builders and the query path (#144 inst. 3)
# ---------------------------------------------------------------------------

def build_gene_exon_index(
    annotation: str = 'gencode_v48_basic',
) -> dict:
    """Per-**gene** merged exon unions, keyed by chromosome and sorted by start.

    The background builders had no structure like this. ``load_exon_index`` in
    ``scripts/build_backgrounds_alphagenome.py`` merged exons across **every gene
    on the chromosome**, which discards gene identity — so the builder aggregated
    RNA signal over every protein-coding exon in its ~1 Mb window (median
    24,325 exonic bp) while the query aggregates over **one gene's** exons
    (median 3,328 bp). Different quantities, so the percentile ranked a
    gene-scoped statistic against a genome-scoped null. #144 instance 3.

    Built from :func:`get_gene_exons` per gene, so the mask a builder scores is
    the *same object* the query scores rather than a lookalike — that equivalence
    is asserted in ``tests/test_gene_exon_index.py`` rather than assumed.

    Returns ``{chrom: [(gene_start, gene_end, gene_name, [(exon_start, exon_end), ...]), ...]}``
    with genes sorted by ``gene_start`` so :func:`genes_overlapping` can bisect.
    """
    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")

    exons = manager._get_exons_df(gtf_path)
    genes = manager._get_genes_df(gtf_path)
    pc_names = set(genes[genes['gene_type'] == 'protein_coding']['gene_name'])
    pc_exons = exons[exons['gene_name'].isin(pc_names)]

    index: dict = {}
    for (chrom, gene_name), group in pc_exons.groupby(['chrom', 'gene_name']):
        intervals = sorted(zip(group['start'].tolist(), group['end'].tolist()))
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        spans = [(int(s), int(e)) for s, e in merged]
        index.setdefault(chrom, []).append(
            (spans[0][0], spans[-1][1], str(gene_name), spans)
        )
    for chrom in index:
        index[chrom].sort(key=lambda g: g[0])
    return index


def genes_overlapping(index: dict, chrom: str, start: int, end: int) -> list:
    """Genes whose span overlaps ``[start, end)``.

    **OVERLAP, not containment** — matching what the query's ``_find_nearby_genes``
    does, deliberately. AlphaGenome's own ``GeneQueryType`` is
    ``INTERVAL_CONTAINED``, but 68 protein-coding genes can never be contained in
    a 1,048,576 bp window (RBFOX1 spans 2.47 Mb; also CNTNAP2, PTPRD, DMD,
    CSMD1), and ``variant_report.py`` force-inserts a user-named gene even when
    it is absent from the window. A containment rule in the builder would
    therefore build a null over a gene population the query does not use.
    """
    out = []
    for g_start, g_end, name, spans in index.get(chrom, []):
        if g_start >= end:
            break
        if g_end <= start:
            continue
        out.append((g_start, g_end, name, spans))
    return out


def exon_bins_for_gene(
    spans: list, pred_start: int, pred_end: int, n_bins: int, resolution: int,
) -> "np.ndarray":
    """Bin indices covered by one gene's exon union, clipped to the prediction.

    Clipping is why the count has to be measured rather than derived from the
    gene's annotated length: a gene straddling the window edge contributes only
    the bins actually predicted, and those are the only ones that may be summed.
    """
    import numpy as _np

    bins = set()
    for es, ee in spans:
        if es >= pred_end:
            break
        if ee <= pred_start:
            continue
        s = max(es, pred_start)
        e = min(ee, pred_end)
        b0 = (s - pred_start) // resolution
        b1 = (e - pred_start + resolution - 1) // resolution
        bins.update(range(max(0, b0), min(n_bins, b1)))
    return _np.array(sorted(bins), dtype=_np.int64)


# ---------------------------------------------------------------------------
# Gene-anchored effect region set (#83, and the reason CAGE/RNA are degenerate)
# ---------------------------------------------------------------------------

# The stratum mixture. THIS IS A JUDGMENT CALL, not derived from anything, and it
# is exposed as parameters so it can be retuned without touching code.
#
# What it fixes: the effect nulls are currently drawn from ~1,900-2,000
# UNIFORMLY RANDOM genomic positions (build_backgrounds_alphagenome.py's
# `random.randint(5_000_000, max_pos)`). For a TSS-peaked assay like CAGE, or an
# exon-scoped one like RNA, a random position carries almost no signal, so the
# null collapses toward zero and every real effect reads >= 99th percentile.
# AlphaGenome RNA's effect null tops out at 0.0417 — anything >= 0.05 saturates
# at exactly 1.0000.
#
# Why 15% stays uniformly random, and why that fraction is load-bearing: without
# a near-zero mass the null loses its lower body, and small real effects would
# get artificially LOW percentiles — the exact mirror of today's failure. The
# random stratum is not filler.
#
# Why NOT an eQTL/GWAS-derived position set: "variants somebody chose to test" is
# a worse reference class than "variants near genes", because the selection is
# correlated with effect size. That would bias the null toward large effects and
# make everything look unremarkable.
DEFAULT_REGION_STRATA = {
    # One third of the positions reproduce the originally shipped gene-anchored set
    # EXACTLY, in its original internal proportions (0.20 / 0.20 / 0.33 / 0.12 / 0.15).
    "tss_near": 1 / 15,   # within +/- 1 kb of a protein-coding TSS   -> 1,200
    "tss_far": 1 / 15,    # 1-10 kb from a TSS                        -> 1,200
    "junction": 0.11,     # +/- 100 bp of an exon/intron boundary     -> 1,980
    "gene_body": 0.04,    # elsewhere inside a protein-coding gene    ->   720
    "random": 0.05,       # uniform, to keep the null's lower body    ->   900
    # The second third, added 2026-08-03, purely ADDITIVE.
    "ccre": 1 / 3,        # inside an ENCODE SCREEN cCRE              -> 6,000
    # The final third, added 2026-08-06, also purely ADDITIVE. DHS summits from the
    # Meuleman index concentrate transcription-factor footprints, so a single-base
    # change there perturbs TF and histone tracks far more than one in a gene body --
    # which lengthens the tail of exactly the layers that pin most.
    #
    # Justified by tail width and by max(union) = max(max_a, max_b), NOT by a
    # calibration measurement, and it does NOT fix motif-creation saturation: the
    # DHS-anchored null ChromBPNet has always used still pins on rs12740374 CEBPA at
    # 1.11x. A null over random positions -- DHS, cCRE or gene-anchored alike --
    # contains few single-base changes that complete a specific factor's full motif.
    "dhs": 1 / 3,         # +/-150 bp of a Meuleman DHS summit        -> 6,000
}

# The intended total. Grown with each added component rather than re-divided, so
# every pre-existing stratum keeps its EXACT absolute count: at n=18,000 the strata
# above yield tss_near 1,200, tss_far 1,200, junction 1,980, gene_body 720,
# random 900 -- the same counts the original gene-anchored build used -- plus 6,000
# cCRE and 6,000 DHS positions on top.
#
# Re-dividing instead of growing would DILUTE. Measured when the cCRE half was first
# tried at a fixed N: TF-track saturation went from 25% to 92%, because each
# component got half the draws and so a shorter tail. Additivity is what makes
# "nothing that already worked can get worse" true.
DEFAULT_N_EFFECT_POSITIONS = 18_000

# ONE region set, shared by every layer and every oracle, and it is a UNION rather
# than a mixture. That distinction is the whole finding, and it was learned the hard
# way, so it is written down at length.
#
# WHY NOT PER-LAYER REFERENCE SETS
# --------------------------------
# Composing a different population per layer means keying on a per-row layer field,
# and the builders did not agree on a vocabulary: Enformer wrote its internal
# ``spec_key`` (``DNASE``, ``ATAC``, ``CHIP_HIST``, ``CHIP_TF``, ``CAGE``) while
# AlphaGenome wrote canonical names. A composition keyed on
# ``chromatin_accessibility`` matched 472 of AlphaGenome's rows and **0 of Enformer's
# 5,313** -- silently, for the one oracle where the change had been measured to help.
# Same defect class as #122 and #144: two producers, two conventions, nothing
# comparing them. (Fixed separately by ``scorers.canonical_layer``, which raises.)
#
# It is also the wrong shape of solution, because layers are shared far more widely
# than per-layer treatment assumes. Measured row counts:
#
#     layer                     AG    enformer  borzoi  chrombpnet  cherimoya  epinf
#     chromatin_accessibility   472    684        906      9          1,518      11
#     histone_marks           1,116  1,890     (subset)  (subset)      -         22
#     tf_binding              1,617  2,101     (subset)  (subset)      -          -
#     tss_activity              558    638      1,276      -           -          -
#     gene_expression           667      -      1,543      -           -          -
#
# Accessibility exists in SIX oracles and already had TWO reference classes across
# them. A third for only some of them would make "0.98 accessibility percentile"
# mean three things depending on which oracle answered, and the multi-oracle
# walkthrough prints them side by side.
#
# WHY A UNION AT 2N, NOT A MIXTURE AT N
# -------------------------------------
# The first attempt held the total position count fixed and gave cCRE 25 % of it.
# It made things WORSE, and the reason is worth internalising: the statistic that
# decides whether a percentile still discriminates is the null MAXIMUM, and a maximum
# grows with the number of draws. Splitting a fixed budget gives every component
# fewer draws, so every component's tail SHORTENS. Measured on one Enformer
# accessibility track and one TF track:
#
#     reference set                              accessibility    tf_binding
#     gene-anchored, 5,949 positions                    1.653         3.539
#     cCRE-only,     5,986 positions                    2.754         3.301
#     25/75 mixture, 5,962 total (1,500 cCRE)           1.697         2.937
#
# The mixture's maximum came out below BOTH full-size components for TF binding, and
# saturation there went from 25 % of rows to 92 %. Mixing does not combine the best
# of both; at fixed N it dilutes both.
#
# Keeping each component at full size instead makes the union's maximum exactly
# ``max(max_gene, max_ccre)``, so the union is **provably never worse than the better
# component, for every layer**. That is a guarantee, not a lucky measurement:
#
#     layer                     gene only   cCRE only   union at 2N
#     chromatin_accessibility        50 %        0 %          0 %
#     tf_binding                     25 %       50 %         25 %
#     histone_marks                   0 %        0 %          0 %
#     tss_activity                    0 %        0 %          0 %
#     all committed rows             11 %        7 %          4 %
#
# Because the gene-anchored half reproduces the shipped counts exactly, the cCRE half
# is purely additive: nothing that already worked can get worse.
#
# STILL NOT FIXED, and not claimed to be: AlphaGenome ``histone_marks`` and Enformer
# ``tf_binding`` keep whatever their better component gives (20 % and 25 %). Both
# would need a *per-track* population -- that mark's own broad domains, that factor's
# own ChIP peaks -- which is a different design, not a different fraction.
DEFAULT_CCRE_STRATUM = "ccre"


def load_chrom_sizes(fai_path: Union[str, Path]) -> dict:
    """Chromosome lengths from a ``.fai``, so sampling cannot run off a contig."""
    sizes = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    return sizes


def sample_gene_anchored_positions(
    n: int,
    *,
    chrom_sizes: dict,
    annotation: str = 'gencode_v48_basic',
    strata: Optional[dict] = None,
    seed: int = 12345,
    tss_near_bp: int = 1_000,
    tss_far_bp: Tuple[int, int] = (1_000, 10_000),
    junction_bp: int = 100,
    dhs_jitter_bp: int = 150,
    margin_bp: int = 5_000_000,
) -> List[Tuple[str, int, str]]:
    """``[(chrom, pos, stratum), ...]`` for building an effect background.

    Anchored on gene structure rather than drawn uniformly, because a uniformly
    random position carries no CAGE signal and sits in no exon — which is why
    those layers' nulls are degenerate today (see ``DEFAULT_REGION_STRATA`` for
    the full reasoning and the exact fractions).

    Every position is tagged with its stratum so the builder can record the
    composition in provenance; a background whose reference class cannot be
    recovered from the file is a background nobody can re-derive (#124).

    ``margin_bp`` keeps positions away from contig ends so a 1 Mb prediction
    window always fits, matching the existing builders' guard.
    """
    import random as _random

    strata = dict(strata or DEFAULT_REGION_STRATA)
    total = sum(strata.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"strata must sum to 1.0, got {total}")

    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")

    genes = manager._get_genes_df(gtf_path)
    pc = genes[genes['gene_type'] == 'protein_coding']
    exons = manager._get_exons_df(gtf_path)
    pc_exons = exons[exons['gene_name'].isin(set(pc['gene_name']))]

    usable = {c: L for c, L in chrom_sizes.items()
              if L > 2 * margin_bp and c in set(pc['chrom'])}

    # Every anchored source population is filtered to the usable interval, NOT just to
    # a usable chromosome. The cCRE pool below always did this; these three did not,
    # and the difference was a real defect in the shipped backgrounds.
    #
    # 2,515 of 20,083 protein-coding TSS (12.5%) lie within 5 Mb of a contig end. They
    # passed the `r.chrom in usable` test, then `_clamp` moved them onto the margin
    # boundary -- up to 5 Mb from the TSS they were labelled as being within 1 kb of.
    # Measured over 6,000 positions: 12.1% of `tss_near`, 12.2% of `junction`, 13.0%
    # of `tss_far` and 14.6% of `gene_body` landed EXACTLY on a boundary coordinate,
    # only 5,265 of 6,000 positions were distinct, and chr16:5,000,000 alone appeared
    # 64 times.
    #
    # Two harms, and the second is the one that matters for the null. The reference
    # class was mislabelled -- a position 5 Mb from any TSS tagged `tss_near`. And
    # duplicate positions yield *identical* effect values, which inflate the sample
    # count without adding information and manufacture tied runs in the CDF: exactly
    # the degeneracy `_rank_with_tie_breaking` exists to paper over, injected by the
    # sampler rather than by the biology.
    def _in_margin(chrom, pos):
        return chrom in usable and margin_bp <= pos <= usable[chrom] - margin_bp

    # TSS is strand-aware: start for +, end for -. Getting this backwards would
    # anchor CAGE on transcript 3' ends, where there is no promoter signal.
    tss = [(r.chrom, int(r.start) if r.strand == '+' else int(r.end))
           for r in pc.itertuples()]
    tss = [(c, p) for c, p in tss if _in_margin(c, p)]
    # Exon boundaries are splice junctions; both edges of every exon count.
    junctions = [(r.chrom, int(r.start)) for r in pc_exons.itertuples()]
    junctions += [(r.chrom, int(r.end)) for r in pc_exons.itertuples()]
    junctions = [(c, p) for c, p in junctions if _in_margin(c, p)]
    # A gene body is kept only where the whole span is usable, so a position drawn
    # uniformly inside it never needs clamping.
    bodies = [(r.chrom, int(r.start), int(r.end)) for r in pc.itertuples()
              if _in_margin(r.chrom, int(r.start)) and _in_margin(r.chrom, int(r.end))]
    chrom_list = sorted(usable)

    rng = _random.Random(seed)

    def _clamp(chrom, pos):
        return min(max(pos, margin_bp), usable[chrom] - margin_bp)

    # cCRE positions come from the SAME sampler the baseline path uses, drawn once
    # up front rather than per-position, because get_screen_ccres() parses a
    # genome-wide BED. Drawn generously: the loop below rejects any that fall inside
    # the contig-end margin.
    ccre_pool: List[Tuple[str, int]] = []
    if strata.get("ccre"):
        want = int(round(n * strata["ccre"]))
        per_class = {
            "PLS": 0.22, "dELS": 0.28, "pELS": 0.15, "CA-CTCF": 0.10,
            "CA-H3K4me3": 0.08, "CA-TF": 0.07, "CA": 0.06, "TF": 0.04,
        }
        raw = sample_ccre_positions(
            n_per_category={k2: max(1, int(round(want * v * 2.0)))
                            for k2, v in per_class.items()},
            seed=seed,
        )
        ccre_pool = [(c, int(p)) for c, p in raw
                     if c in usable and margin_bp <= p <= usable[c] - margin_bp]
        rng.shuffle(ccre_pool)

    # DHS summits, drawn once like the cCRE pool. Generously (2x) so the cursor below
    # never has to wrap and emit a duplicate position. Jittered by +/-150 bp to match
    # the convention the ChromBPNet and Cherimoya builders already use, so the "dhs"
    # stratum means the same thing in every oracle rather than merely having the same
    # name.
    dhs_pool: List[Tuple[str, int]] = []
    if strata.get("dhs"):
        want = int(round(n * strata["dhs"]))
        raw = sample_dhs_positions(max(1, want * 2), seed=seed)
        dhs_pool = [(c, int(p)) for c, p in raw
                    if c in usable and margin_bp <= p <= usable[c] - margin_bp]
        rng.shuffle(dhs_pool)

    # Per-stratum cursors, NOT a shared index into the pools.
    #
    # This used to be ``ccre_pool[len(out) % len(ccre_pool)]`` -- indexed by the TOTAL
    # number of positions emitted so far. Inserting any new stratum therefore shifted
    # which cCREs got drawn, silently, which would have broken the additivity this
    # mixture is built on the moment "dhs" was added. The promoter sampler already
    # used a per-stratum counter (``pls[i % len(pls)]``); the two disagreed, which is
    # the #144 shape again -- two code paths computing the same thing differently.
    #
    # Note this does change WHICH cCRE positions are drawn relative to the currently
    # shipped build. That is fine and expected: the additivity guarantee is
    # ``max(union) >= max(component)``, which holds for any valid uniform draw from
    # the pool, not for one specific draw.
    _ANCHORED = {
        "ccre": ccre_pool, "dhs": dhs_pool, "tss_near": tss, "tss_far": tss,
        "junction": junctions, "gene_body": bodies,
    }

    out: List[Tuple[str, int, str]] = []
    for name, frac in strata.items():
        if name != "random" and name not in _ANCHORED:
            # The landmine this replaces: the old ``else`` branch was simultaneously
            # the "random" handler, the empty-pool fallback AND the catch-all for
            # unrecognised names. Adding "dhs" to the strata dict without a handler
            # would have emitted 6,000 UNIFORMLY RANDOM positions, tagged them
            # "dhs", tallied them as DHS in the build log and stamped them as DHS in
            # provenance -- an invisibly wrong reference class in every artefact.
            raise ValueError(
                f"unhandled stratum {name!r}: no sampler branch exists for it. Add "
                f"one rather than letting it fall through to uniform random "
                f"positions, which would be tagged and stamped with this name. "
                f"Known: {sorted(set(_ANCHORED) | {'random'})}"
            )
        if name != "random" and not _ANCHORED[name]:
            # Also previously silent: an empty pool fell through to random. A missing
            # or unreadable annotation source must be loud, not quietly substituted.
            raise ValueError(
                f"stratum {name!r} was requested but its source population is empty "
                f"(no usable positions after the {margin_bp} bp contig-end margin). "
                f"Refusing to substitute uniform random positions under this label."
            )

        k = int(round(n * frac))
        for i in range(k):
            if name == "ccre":
                c, p = ccre_pool[i % len(ccre_pool)]
            elif name == "dhs":
                c, p = dhs_pool[i % len(dhs_pool)]
                p += rng.randint(-dhs_jitter_bp, dhs_jitter_bp)
            elif name == "tss_near":
                c, p = rng.choice(tss)
                p += rng.randint(-tss_near_bp, tss_near_bp)
            elif name == "tss_far":
                c, p = rng.choice(tss)
                off = rng.randint(*tss_far_bp)
                p += off if rng.random() < 0.5 else -off
            elif name == "junction":
                c, p = rng.choice(junctions)
                p += rng.randint(-junction_bp, junction_bp)
            elif name == "gene_body":
                c, s, e = rng.choice(bodies)
                p = rng.randint(s, e) if e > s else s
            else:                                    # name == "random"
                c = rng.choice(chrom_list)
                p = rng.randint(margin_bp, usable[c] - margin_bp)
            out.append((c, _clamp(c, p), name))

    rng.shuffle(out)
    return out


# Per-layer effect region sets. A percentile answers "how unusual is this effect
# against a reference population of variants", and the right population depends on
# what the assay measures.
#
# Measured 2026-08-04 over every committed walkthrough row, comparing each raw
# effect against its own track's null maximum on the gene-anchored null:
#
#     oracle        layer                      rows   above null max
#     enformer      chromatin_accessibility      12       50.0 %
#     alphagenome   histone_marks                10       30.0 %
#     enformer      tf_binding                   12       25.0 %
#     alphagenome   gene_expression             100        7.0 %
#     alphagenome   tss_activity                263        8.0 %
#     enformer      tss_activity                 48        0.0 %
#
# The *peak* layers saturate and the rest do not, and the reason is structural: most
# gene-anchored positions are not inside a peak, and a variant in closed chromatin
# cannot move an accessibility or ChIP signal much. So the null's upper tail is too
# short — at SORT1, enformer accessibility effects are 1.14-1.45x its null maximum,
# which is exactly why they pin at 1.0000 and stop discriminating.
#
# CAGE needs nothing. That was measured too, and the result was the opposite of what
# was expected: sweeping the variant's distance from an annotated TSS gives a
# monotone curve in that one parameter (eQTL percentile p50 0.323 at the TSS itself,
# 0.411 at +/-500 bp, 0.526 at 1 kb, 0.604 at 2 kb, 0.654 at 5 kb, 0.729 at 10 kb),
# and the shipped gene-anchored mixture sits at 0.659 — i.e. it already behaves like
# a "+/-5 kb of a TSS" null for CAGE. There is no qualitative gain available, only a
# choice of distance scale that no principle fixes.
EFFECT_REGION_SETS = ('gene-anchored', 'ccre')

# Layers whose null should come from cCREs rather than the gene-anchored mixture.
# Keyed by the layer names in scorers.LAYER_CONFIGS.
#
# This set is MEASURED, one layer at a time, not inferred from "peak layers behave
# alike" — they do not. Enformer, 5,986 cCRE positions against 5,949 gene-anchored
# (a 1.006 count ratio, so the comparison is not confounded by sample size):
#
#     layer                     saturated, gene-anchored -> cCRE
#     chromatin_accessibility            50 %  ->   0 %    ACCEPTED
#     tf_binding                         25 %  ->  50 %    REJECTED
#     histone_marks                       0 %  ->   0 %    no change for enformer
#     tss_activity                        0 %  ->   0 %    no change
#
# Accessibility is fixed outright. TF binding gets WORSE, and the reason is that a
# cCRE is *defined* by accessibility, H3K4me3 or CTCF signal — a randomly chosen cCRE
# is often not bound by the particular TF a given ChIP track measures, so its ChIP
# signal there is low and a variant cannot move it. Gene-anchored positions include
# promoters where many TFs are bound, which is a better reference class for TF ChIP.
#
# So the rule is: a cCRE-anchored null helps the layer whose signal *defines* a cCRE,
# and hurts layers that merely correlate with it.
CCRE_ANCHORED_LAYERS = frozenset({
    'chromatin_accessibility',
})


def sample_ccre_anchored_positions(
    n: int,
    *,
    chrom_sizes: dict,
    seed: int = 12345,
    margin_bp: int = 600_000,
) -> list:
    """Positions inside ENCODE SCREEN cCREs, stratified by element class.

    The matched reference class for a peak assay: "a variant inside a candidate
    cis-regulatory element". Reuses :func:`sample_ccre_positions`, which the
    *baseline* path has always used — the effect path never did, which is the whole
    defect. Sharing it means the two paths cannot drift.

    Returns ``(chrom, pos, stratum)`` triples so the caller's tally, logging and
    provenance stamp work unchanged against ``sample_gene_anchored_positions``.

    ``margin_bp`` keeps a position far enough from a contig end that the oracle's
    input window fits; 600 kb clears AlphaGenome's 1,048,576 bp half-window.
    """
    per_category = {
        # Roughly SCREEN's own class proportions, so no single element type
        # dominates the null. PLS and dELS carry most real regulatory variation.
        'PLS': 0.22, 'dELS': 0.28, 'pELS': 0.15, 'CA-CTCF': 0.10,
        'CA-H3K4me3': 0.08, 'CA-TF': 0.07, 'CA': 0.06, 'TF': 0.04,
    }
    # Oversample, because the margin filter below rejects some.
    counts = {k: max(1, int(round(n * v * 1.4))) for k, v in per_category.items()}
    raw = sample_ccre_positions(n_per_category=counts, seed=seed)

    usable = {c: L for c, L in chrom_sizes.items() if L > 2 * margin_bp}
    out = []
    for chrom, pos in raw:
        if chrom not in usable:
            continue
        if not (margin_bp <= pos <= usable[chrom] - margin_bp):
            continue
        out.append((chrom, int(pos), 'ccre'))

    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


# LegNet is a promoter MPRA model: 200 bp input, ``window_bp=None``, so the sampled
# position IS the whole thing being modelled. A uniformly random 200 bp window is
# almost entirely non-promoter sequence, which makes it the worst-anchored effect null
# in the fleet -- the null answers "what does a variant do to random DNA" while every
# query asks about a promoter.
#
# Deliberately NOT the generic cCRE mix, and not DHS summits. Both are
# accessibility-general: the SCREEN catalogue is 62 % dELS (1,469,205 of 2,348,854
# distal enhancer-like) against 2 % PLS (47,532 promoter-like), and DHS summits track
# accessibility rather than promoter identity. Anchoring a promoter model on either
# would give it a null made mostly of enhancers -- right family, wrong member.
#
# PLS *is* "promoter-like signature", i.e. LegNet's own estimand, so it leads. pELS
# (proximal enhancer-like) is promoter-adjacent and included at a lower weight. The
# 15 % uniform tail is kept for the same reason the gene-anchored set keeps one:
# without near-zero mass, genuinely small effects receive artificially LOW percentiles.
# 2026-08-06: DHS is now included after all, and the paragraph above needs its
# qualification rather than deletion. The objection stands for a *re-weighting* --
# giving DHS a share of a fixed N would dilute the promoter component and hand a
# promoter model a null made mostly of enhancers. It does not stand for an ADDITIVE
# union at scaled N: every stratum below keeps its exact absolute count, N grows from
# 12,000 to 18,000, and max(union) = max(max_promoter, max_dhs), so the promoter
# component cannot be weakened by the addition. That is the only construction under
# which the two are reconcilable, and it is the one used here.
PROMOTER_REGION_STRATA = {
    "tss_promoter": 4 / 15,  # +/- 250 bp of a PC TSS, so a 200 bp window overlaps
                             # the core promoter                        -> 4,800
    "ccre_pls": 0.20,        # SCREEN promoter-like signature           -> 3,600
    "ccre_pels": 0.10,       # SCREEN proximal enhancer-like            -> 1,800
    "random": 0.10,          # uniform, keeps the null's lower body     -> 1,800
    "dhs": 1 / 3,            # +/-150 bp of a Meuleman DHS summit       -> 6,000
}


def sample_promoter_anchored_positions(
    n: int,
    *,
    chrom_sizes: dict,
    annotation: str = 'gencode_v48_basic',
    strata: Optional[dict] = None,
    seed: int = 12345,
    margin_bp: int = 100_000,
    tss_jitter_bp: int = 250,
    dhs_jitter_bp: int = 150,
) -> list:
    """Positions anchored on promoters, for promoter-activity models.

    Returns ``(chrom, pos, stratum)`` triples, matching
    :func:`sample_gene_anchored_positions` so a builder's tally, logging and
    provenance stamp work against either without branching.
    """
    strata = dict(strata or PROMOTER_REGION_STRATA)
    usable = {c: L for c, L in chrom_sizes.items() if L > 2 * margin_bp}
    if not usable:
        raise ValueError("no chromosome long enough for the requested margin")
    rng = random.Random(seed)

    manager = get_annotation_manager()
    gtf = manager.get_annotation_path(annotation)
    genes = manager._get_genes_df(gtf)
    pc = genes[genes["gene_type"] == "protein_coding"]
    tss = []
    for row in pc.itertuples():
        chrom = str(row.chrom)
        if chrom not in usable:
            continue
        pos = int(row.start) if row.strand == "+" else int(row.end)
        if margin_bp <= pos <= usable[chrom] - margin_bp:
            tss.append((chrom, pos))
    if not tss:
        raise ValueError("no usable protein-coding TSS found")

    def _ccre_pool(classes, want):
        raw = sample_ccre_positions(
            n_per_category={c: max(1, int(round(want * 2.0 / len(classes))))
                            for c in classes},
            seed=seed,
        )
        pool = [(c, int(p)) for c, p in raw
                if c in usable and margin_bp <= p <= usable[c] - margin_bp]
        rng.shuffle(pool)
        return pool

    pls = _ccre_pool(["PLS"], int(round(n * strata.get("ccre_pls", 0))))         if strata.get("ccre_pls") else []
    pels = _ccre_pool(["pELS"], int(round(n * strata.get("ccre_pels", 0))))         if strata.get("ccre_pels") else []

    dhs_pool = []
    if strata.get("dhs"):
        want = int(round(n * strata["dhs"]))
        raw = sample_dhs_positions(max(1, want * 2), seed=seed)
        dhs_pool = [(c, int(pp)) for c, pp in raw
                    if c in usable and margin_bp <= pp <= usable[c] - margin_bp]
        rng.shuffle(dhs_pool)

    def _clamp(chrom, pos):
        return min(max(pos, margin_bp), usable[chrom] - margin_bp)

    # Same discipline as sample_gene_anchored_positions: an unrecognised stratum name
    # must RAISE rather than fall through to uniform random positions that then get
    # tagged, tallied and stamped with that name.
    _ANCHORED = {"tss_promoter": tss, "ccre_pls": pls, "ccre_pels": pels,
                 "dhs": dhs_pool}

    out = []
    for name, frac in strata.items():
        if name != "random" and name not in _ANCHORED:
            raise ValueError(
                f"unhandled stratum {name!r}: no sampler branch exists for it. "
                f"Known: {sorted(set(_ANCHORED) | {'random'})}"
            )
        if name != "random" and not _ANCHORED[name]:
            raise ValueError(
                f"stratum {name!r} was requested but its source population is empty; "
                f"refusing to substitute uniform random positions under this label"
            )
        k = int(round(n * frac))
        for i in range(k):
            if name == "tss_promoter":
                c, p = rng.choice(tss)
                p += rng.randint(-tss_jitter_bp, tss_jitter_bp)
            elif name == "ccre_pls":
                c, p = pls[i % len(pls)]
            elif name == "ccre_pels":
                c, p = pels[i % len(pels)]
            elif name == "dhs":
                c, p = dhs_pool[i % len(dhs_pool)]
                p += rng.randint(-dhs_jitter_bp, dhs_jitter_bp)
            else:                                    # name == "random"
                c = rng.choice(list(usable))
                p = rng.randint(margin_bp, usable[c] - margin_bp)
            out.append((c, _clamp(c, p), name))

    rng.shuffle(out)
    return out


def build_transcript_exon_index(
    annotation: str = 'gencode_v48_basic',
    gene_types: Optional[set] = frozenset({'protein_coding'}),
) -> dict:
    """Per-gene, per-**transcript** exon spans plus each transcript's TSS.

    Needed because AlphaGenome selects genes by **TSS-in-window**, not by gene-body
    overlap, and then unions the exons of *only those transcripts*
    (``gene_mask_extractor.py:326, 357-371``). A gene whose body overlaps the
    window but whose TSS lies outside contributes **nothing** under that rule,
    where a gene-body-overlap rule contributes all of its exons.

    TSS is strand-aware — ``Start`` for ``+``, ``End`` for ``-``
    (``alphagenome/data/gene_annotation.py:94``). Getting that backwards anchors on
    transcript 3' ends, and the gene counts would still look right.

    ``gene_types`` defaults to protein-coding only, which is a **deliberate
    divergence**: AlphaGenome applies no gene-type filter at all, but chorus's
    query does (``variant_report.py:825``), and a null built over lncRNAs and
    pseudogenes would be a different population from the numerator — #144 in the
    other direction. Pass ``None`` for no filter. The choice is recorded in
    provenance rather than left implicit.

    Returns ``{chrom: [(gene_start, gene_end, gene_name, [(tss, [(es, ee), ...]), ...]), ...]}``.
    """
    manager = get_annotation_manager()
    gtf_path = manager.get_annotation_path(annotation)
    if not gtf_path:
        raise ValueError(f"Could not find annotation: {annotation}")

    exons = manager._get_exons_df(gtf_path)
    if gene_types is not None:
        genes = manager._get_genes_df(gtf_path)
        keep = set(genes[genes['gene_type'].isin(gene_types)]['gene_name'])
        exons = exons[exons['gene_name'].isin(keep)]

    # exon rows -> per-transcript spans
    per_tx: dict = {}
    for row in exons.itertuples():
        entry = per_tx.get(row.transcript_id)
        if entry is None:
            per_tx[row.transcript_id] = entry = [
                str(row.chrom), str(row.strand), str(row.gene_name), [],
            ]
        entry[3].append((int(row.start), int(row.end)))

    by_gene: dict = {}
    for chrom, strand, gene_name, spans in per_tx.values():
        spans.sort()
        # A transcript's TSS is its outermost exon boundary on the 5' side.
        tss = spans[0][0] if strand == '+' else spans[-1][1]
        by_gene.setdefault((chrom, gene_name), []).append((tss, spans))

    index: dict = {}
    for (chrom, gene_name), transcripts in by_gene.items():
        transcripts.sort(key=lambda t: t[0])
        starts = [s for _tss, spans in transcripts for s, _e in spans]
        ends = [e for _tss, spans in transcripts for _s, e in spans]
        index.setdefault(chrom, []).append(
            (min(starts), max(ends), gene_name, transcripts)
        )
    for chrom in index:
        index[chrom].sort(key=lambda g: g[0])
    return index


def genes_with_tss_in_window(index: dict, chrom: str, start: int, end: int) -> list:
    """``[(gene_name, merged_exon_spans), ...]`` for AlphaGenome's selection rule.

    A gene is included iff at least one of its transcripts has its TSS in
    ``[start, end)`` — semi-open, matching ``_PositionExtractor``. Its mask is the
    union of the exons of **only** those transcripts, which is what
    ``gene_mask |= exon_mask`` builds.
    """
    out = []
    for _g_start, _g_end, gene_name, transcripts in index.get(chrom, []):
        spans: list = []
        for tss, tx_spans in transcripts:
            if start <= tss < end:
                spans.extend(tx_spans)
        if not spans:
            continue
        spans.sort()
        merged = [list(spans[0])]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out.append((gene_name, [(int(s), int(e)) for s, e in merged]))
    return out
