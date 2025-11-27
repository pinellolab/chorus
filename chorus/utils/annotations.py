"""Gene annotation utilities for Chorus.

This module provides functionality for working with gene annotations,
particularly for visualizing genomic regions in the context of genes
and for analyzing effects on gene expression.
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
    """Manager for gene annotations (GTF files)."""
    
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
        
        Args:
            gtf_path: Path to GTF file (can be gzipped)
            chrom: Chromosome name
            start: Start position
            end: End position
            feature_types: Types of features to extract (default: ['gene'])
            
        Returns:
            DataFrame with gene information
        """
        gtf_path = Path(gtf_path)
        
        # Open file (handle gzip if needed)
        if str(gtf_path).endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        genes = []
        
        with open_func(gtf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                # Parse GTF fields
                seqname = parts[0]
                feature = parts[2]
                feat_start = int(parts[3])
                feat_end = int(parts[4])
                strand = parts[6]
                attributes = parts[8]
                
                # Check if in region and correct feature type
                if (seqname == chrom and 
                    feature in feature_types and
                    feat_end >= start and 
                    feat_start <= end):
                    
                    # Parse attributes
                    attr_dict = {}
                    for attr in attributes.strip().split(';'):
                        if attr.strip():
                            key_value = attr.strip().split(' ', 1)
                            if len(key_value) == 2:
                                key = key_value[0]
                                value = key_value[1].strip('"')
                                attr_dict[key] = value
                    
                    genes.append({
                        'chrom': seqname,
                        'start': feat_start,
                        'end': feat_end,
                        'strand': strand,
                        'feature': feature,
                        'gene_name': attr_dict.get('gene_name', ''),
                        'gene_id': attr_dict.get('gene_id', ''),
                        'gene_type': attr_dict.get('gene_type', ''),
                        'transcript_type': attr_dict.get('transcript_type', ''),
                        'level': attr_dict.get('level', ''),
                        'attributes': attr_dict
                    })
        
        return pd.DataFrame(genes)
    
    def get_tss_positions(self, gtf_path: Union[str, Path],
                         gene_name: Optional[str] = None,
                         gene_id: Optional[str] = None,
                         chrom: Optional[str] = None) -> pd.DataFrame:
        """Extract TSS (Transcription Start Site) positions for genes.
        
        Args:
            gtf_path: Path to GTF file
            gene_name: Filter by gene name (e.g., 'GATA1')
            gene_id: Filter by gene ID (e.g., 'ENSG00000102145')
            chrom: Filter by chromosome
            
        Returns:
            DataFrame with TSS positions
        """
        gtf_path = Path(gtf_path)
        
        # Open file (handle gzip if needed)
        if str(gtf_path).endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        transcripts = []
        
        with open_func(gtf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                # Only look at transcript features
                if parts[2] != 'transcript':
                    continue
                
                # Parse GTF fields
                seqname = parts[0]
                feat_start = int(parts[3])
                feat_end = int(parts[4])
                strand = parts[6]
                attributes = parts[8]
                
                # Apply chromosome filter if specified
                if chrom and seqname != chrom:
                    continue
                
                # Parse attributes
                attr_dict = {}
                for attr in attributes.strip().split(';'):
                    if attr.strip():
                        key_value = attr.strip().split(' ', 1)
                        if len(key_value) == 2:
                            key = key_value[0]
                            value = key_value[1].strip('"')
                            attr_dict[key] = value
                
                # Apply gene filters if specified
                if gene_name and attr_dict.get('gene_name') != gene_name:
                    continue
                if gene_id and attr_dict.get('gene_id') != gene_id:
                    continue
                
                # Determine TSS based on strand
                tss = feat_start if strand == '+' else feat_end
                
                transcripts.append({
                    'chrom': seqname,
                    'tss': tss,
                    'strand': strand,
                    'gene_name': attr_dict.get('gene_name', ''),
                    'gene_id': attr_dict.get('gene_id', ''),
                    'transcript_id': attr_dict.get('transcript_id', ''),
                    'transcript_type': attr_dict.get('transcript_type', ''),
                    'transcript_start': feat_start,
                    'transcript_end': feat_end
                })
        
        return pd.DataFrame(transcripts)


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
    """Sort GTF file.
    
    Args:
        gtf_path: Path to GTF file
        
    Returns:
        Path to sorted GTF file
    """
    cmd = f"gtfsort --input {gtf_path} --output {output_path}"
    cmd = shlex.split(cmd)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to sort GTF file: {res.stderr}")
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