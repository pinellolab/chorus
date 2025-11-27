"""Genome management utilities for downloading and managing reference genomes."""

import gzip
import urllib.request
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
import subprocess

from ..core.globals import CHORUS_GENOMES_DIR

logger = logging.getLogger(__name__)

# UCSC genome URLs
GENOME_URLS = {
    'hg38': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
    'hg19': 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
    'mm10': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz',
    'mm9': 'https://hgdownload.soe.ucsc.edu/goldenPath/mm9/bigZips/mm9.fa.gz',
    'dm6': 'https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz',
    'ce11': 'https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz',
}

# Genome descriptions
GENOME_DESCRIPTIONS = {
    'hg38': 'Human genome assembly GRCh38/hg38',
    'hg19': 'Human genome assembly GRCh37/hg19',
    'mm10': 'Mouse genome assembly GRCm38/mm10',
    'mm9': 'Mouse genome assembly NCBI37/mm9',
    'dm6': 'Drosophila melanogaster genome assembly BDGP6/dm6',
    'ce11': 'C. elegans genome assembly WBcel235/ce11',
}


class GenomeManager:
    """Manages reference genome downloads and storage."""
    
    def __init__(self, genomes_dir: Optional[Path] = None):
        """Initialize genome manager.
        
        Args:
            genomes_dir: Directory to store genomes. Defaults to chorus/genomes/
        """
        if genomes_dir is None:
            # Default to genomes directory in project root
            genomes_dir = CHORUS_GENOMES_DIR
        
        self.genomes_dir = Path(genomes_dir)
        self.genomes_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_genomes(self) -> Dict[str, str]:
        """List all available genomes for download.
        
        Returns:
            Dictionary mapping genome ID to description
        """
        return GENOME_DESCRIPTIONS.copy()
    
    def list_downloaded_genomes(self) -> List[str]:
        """List all downloaded genomes.
        
        Returns:
            List of genome IDs that have been downloaded
        """
        downloaded = []
        for genome_id in GENOME_URLS:
            if self.is_genome_downloaded(genome_id):
                downloaded.append(genome_id)
        return downloaded
    
    def get_genome_path(self, genome_id: str) -> Path:
        """Get the path to a genome file.
        
        Args:
            genome_id: Genome identifier (e.g., 'hg38')
            
        Returns:
            Path to the genome FASTA file
        """
        return self.genomes_dir / f"{genome_id}.fa"
    
    def is_genome_downloaded(self, genome_id: str) -> bool:
        """Check if a genome has been downloaded.
        
        Args:
            genome_id: Genome identifier
            
        Returns:
            True if genome is downloaded and valid
        """
        fasta_path = self.get_genome_path(genome_id)
        fai_path = Path(str(fasta_path) + '.fai')
        
        # Check if both FASTA and index exist
        return fasta_path.exists() and fai_path.exists()
    
    def download_genome(self, genome_id: str, force: bool = False) -> bool:
        """Download a reference genome from UCSC.
        
        Args:
            genome_id: Genome identifier (e.g., 'hg38')
            force: Force re-download even if genome exists
            
        Returns:
            True if download successful
        """
        if genome_id not in GENOME_URLS:
            logger.error(f"Unknown genome: {genome_id}")
            logger.info(f"Available genomes: {', '.join(GENOME_URLS.keys())}")
            return False
        
        fasta_path = self.get_genome_path(genome_id)
        
        # Check if already downloaded
        if self.is_genome_downloaded(genome_id) and not force:
            logger.info(f"Genome {genome_id} already downloaded at {fasta_path}")
            return True
        
        url = GENOME_URLS[genome_id]
        gz_path = self.genomes_dir / f"{genome_id}.fa.gz"
        
        try:
            # Download compressed file
            logger.info(f"Downloading {genome_id} from {url}...")
            logger.info("This may take several minutes depending on your connection speed...")
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", 
                      end='', flush=True)
            
            urllib.request.urlretrieve(url, gz_path, reporthook=download_progress)
            print()  # New line after progress
            
            # Decompress
            logger.info(f"Decompressing {genome_id}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(fasta_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file
            gz_path.unlink()
            
            # Create FASTA index
            logger.info(f"Creating FASTA index for {genome_id}...")
            if not self._create_fasta_index(fasta_path):
                logger.error("Failed to create FASTA index")
                return False
            
            logger.info(f"Successfully downloaded {genome_id} to {fasta_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {genome_id}: {e}")
            # Clean up partial downloads
            if gz_path.exists():
                gz_path.unlink()
            if fasta_path.exists() and not self.is_genome_downloaded(genome_id):
                fasta_path.unlink()
            return False
    
    def _create_fasta_index(self, fasta_path: Path) -> bool:
        """Create FASTA index using samtools faidx.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            True if index created successfully
        """
        try:
            # Try to use samtools
            result = subprocess.run(
                ['samtools', 'faidx', str(fasta_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
            else:
                logger.warning(f"samtools faidx failed: {result.stderr}")
                logger.info("Falling back to pyfaidx...")
                
        except FileNotFoundError:
            logger.warning("samtools not found, using pyfaidx...")
        
        # Fall back to pyfaidx
        try:
            import pyfaidx
            pyfaidx.Faidx(str(fasta_path))
            return True
        except ImportError:
            logger.error("pyfaidx not installed. Install with: pip install pyfaidx")
            return False
        except Exception as e:
            logger.error(f"Failed to create FASTA index: {e}")
            return False
    
    def remove_genome(self, genome_id: str) -> bool:
        """Remove a downloaded genome.
        
        Args:
            genome_id: Genome identifier
            
        Returns:
            True if removal successful
        """
        fasta_path = self.get_genome_path(genome_id)
        fai_path = Path(str(fasta_path) + '.fai')
        
        removed = False
        if fasta_path.exists():
            fasta_path.unlink()
            removed = True
            logger.info(f"Removed {fasta_path}")
        
        if fai_path.exists():
            fai_path.unlink()
            logger.info(f"Removed {fai_path}")
        
        if not removed:
            logger.warning(f"Genome {genome_id} not found")
            return False
        
        return True
    
    def get_genome_info(self, genome_id: str) -> Optional[Dict]:
        """Get information about a genome.
        
        Args:
            genome_id: Genome identifier
            
        Returns:
            Dictionary with genome information or None if not found
        """
        if not self.is_genome_downloaded(genome_id):
            return None
        
        fasta_path = self.get_genome_path(genome_id)
        fai_path = Path(str(fasta_path) + '.fai')
        
        info = {
            'id': genome_id,
            'description': GENOME_DESCRIPTIONS.get(genome_id, 'Unknown'),
            'path': str(fasta_path),
            'size_mb': fasta_path.stat().st_size / (1024 * 1024),
        }
        
        # Read chromosome info from index
        if fai_path.exists():
            chromosomes = []
            total_length = 0
            with open(fai_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        chrom_name = parts[0]
                        chrom_length = int(parts[1])
                        chromosomes.append({
                            'name': chrom_name,
                            'length': chrom_length
                        })
                        total_length += chrom_length
            
            info['chromosomes'] = chromosomes
            info['total_length'] = total_length
            info['num_chromosomes'] = len(chromosomes)
        
        return info
    
    def get_genome(self, genome_id: str = 'hg38', auto_download: bool = True) -> Optional[Path]:
        """Get path to a genome, downloading if necessary.
        
        Args:
            genome_id: Genome identifier (defaults to 'hg38')
            auto_download: Automatically download if not present
            
        Returns:
            Path to genome FASTA file or None if not available
        """
        if self.is_genome_downloaded(genome_id):
            return self.get_genome_path(genome_id)
        
        if auto_download:
            logger.info(f"Genome {genome_id} not found. Downloading...")
            if self.download_genome(genome_id):
                return self.get_genome_path(genome_id)
            else:
                logger.error(f"Failed to download {genome_id}")
                return None
        else:
            logger.warning(f"Genome {genome_id} not found and auto_download is disabled")
            return None


def download_genome(genome_id: str, genomes_dir: Optional[Path] = None, 
                   force: bool = False) -> Optional[Path]:
    """Convenience function to download a genome.
    
    Args:
        genome_id: Genome identifier (e.g., 'hg38')
        genomes_dir: Directory to store genomes
        force: Force re-download even if genome exists
        
    Returns:
        Path to downloaded genome or None if failed
    """
    manager = GenomeManager(genomes_dir)
    if manager.download_genome(genome_id, force):
        return manager.get_genome_path(genome_id)
    return None


def list_genomes(genomes_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """List all available and downloaded genomes.
    
    Args:
        genomes_dir: Directory containing genomes
        
    Returns:
        Dictionary with 'available' and 'downloaded' keys
    """
    manager = GenomeManager(genomes_dir)
    return {
        'available': manager.list_available_genomes(),
        'downloaded': manager.list_downloaded_genomes()
    }


def get_genome(genome_id: str = 'hg38', genomes_dir: Optional[Path] = None,
               auto_download: bool = True) -> Optional[Path]:
    """Convenience function to get a genome path, downloading if necessary.
    
    Args:
        genome_id: Genome identifier (defaults to 'hg38')
        genomes_dir: Directory to store genomes
        auto_download: Automatically download if not present
        
    Returns:
        Path to genome FASTA file or None if not available
    """
    manager = GenomeManager(genomes_dir)
    return manager.get_genome(genome_id, auto_download)