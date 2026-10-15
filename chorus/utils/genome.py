"""Genome management utilities for downloading and managing reference genomes."""

import gzip
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
import subprocess

from ..core.globals import CHORUS_GENOMES_DIR
from .http import download_with_resume

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
            # Download compressed file.
            #
            # Use the chunked+resumable helper rather than urllib.urlretrieve —
            # UCSC's server occasionally cuts long connections mid-download
            # (observed on macOS during the 2026-04-14 v2 audit: stall at ~36%
            # with "retrieval incomplete: got only 363743871 out of 983659424
            # bytes"). download_with_resume will pick up from the partial file
            # on the next call via an HTTP Range request.
            logger.info(f"Downloading {genome_id} from {url}...")
            logger.info("This may take several minutes depending on your connection speed...")
            download_with_resume(url, gz_path, label=f"{genome_id} genome")

            # Decompress — guard against concurrent runs where another
            # process may have already decompressed and deleted the .gz.
            if fasta_path.exists():
                gz_path.unlink(missing_ok=True)
            elif gz_path.exists():
                logger.info(f"Decompressing {genome_id}...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(fasta_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                gz_path.unlink(missing_ok=True)
            else:
                raise FileNotFoundError(
                    f"{gz_path} missing after download and {fasta_path} "
                    "not present — re-run with --force to retry."
                )
            
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


# ----------------------------------------------------------------------
# Assembly identity
# ----------------------------------------------------------------------

#: Length of chromosome 1 in each assembly chorus knows how to name.
#:
#: The cheapest fingerprint that is also *provider-independent*: a UCSC
#: ``hg38.fa``, an Ensembl ``GRCh38.dna.primary_assembly.fa`` and a GENCODE
#: release disagree about chromosome naming (``chr1`` vs ``1``), line width,
#: soft-masking and which scaffolds are included, but they all agree about how
#: long chromosome 1 is. Hashing the file instead — which is what
#: ``build_config.fasta_sha256_prefix64mb`` records — distinguishes *files*, not
#: assemblies, so it would reject a correct Ensembl GRCh38 as loudly as it
#: rejects mm10.
ASSEMBLY_CHR1_LENGTH: Dict[str, int] = {
    'hg38': 248_956_422,   # GRCh38
    'hg19': 249_250_621,   # GRCh37
    'mm39': 195_154_279,   # GRCm39
    'mm10': 195_471_971,   # GRCm38
    'mm9':  197_195_432,
}

#: Reverse lookup. hg38 and hg19 differ by ~294 kb on chr1, mm10 and mm39 by
#: ~318 kb, so there is no collision to resolve.
_CHR1_LENGTH_TO_ASSEMBLY: Dict[int, str] = {
    v: k for k, v in ASSEMBLY_CHR1_LENGTH.items()
}


def chr1_length(fasta_path) -> Optional[int]:
    """Length of chromosome 1 in *fasta_path*, or None if it can't be read.

    Reads the ``.fai`` index when there is one — a few hundred bytes and no
    dependency — and only falls back to opening the FASTA through pysam when
    there isn't. Accepts both ``chr1`` and ``1`` so an Ensembl reference works.
    """
    fasta_path = Path(fasta_path)
    fai = Path(str(fasta_path) + '.fai')
    if fai.exists():
        try:
            for line in fai.read_text().splitlines():
                name, _, rest = line.partition('\t')
                if name in ('chr1', '1'):
                    return int(rest.split('\t')[0])
        except (OSError, ValueError, IndexError):
            pass
    try:
        import pysam
        with pysam.FastaFile(str(fasta_path)) as fa:
            for name in ('chr1', '1'):
                if name in fa.references:
                    return int(fa.get_reference_length(name))
    except Exception:                                   # unreadable / unindexed
        return None
    return None


def detect_assembly(fasta_path) -> Optional[str]:
    """Identify the assembly of *fasta_path*, or None if it is unrecognised.

    None means "no claim": an unreadable reference, one with no chromosome 1, or
    a genuine assembly not in :data:`ASSEMBLY_CHR1_LENGTH` (dm6, ce11, a custom
    build). Callers must treat that differently from a *wrong* answer — see
    :func:`require_assembly`.
    """
    length = chr1_length(fasta_path)
    if length is None:
        return None
    return _CHR1_LENGTH_TO_ASSEMBLY.get(length)


def require_assembly(fasta_path, expected: str, *, context: str = "") -> Optional[str]:
    """Raise unless *fasta_path* is the *expected* assembly. Returns what it found.

    The asymmetry is deliberate and is the whole design:

    * a **recognised, different** assembly raises
      :class:`~chorus.core.exceptions.GenomeAssemblyMismatchError`, because every
      coordinate in mm10 also exists in hg38 and so the only symptom of the
      mistake is a plausible number about the wrong piece of DNA;
    * an **unrecognised** reference warns and returns None, because refusing it
      would break anyone using a legitimate assembly chorus has no chr1 length
      for, to enforce a lookup table's completeness.

    ``expected`` itself being unknown is a programming error and raises, so a
    typo (``"GRCh38"`` for ``"hg38"``) fails at the guard rather than silently
    disabling it.
    """
    return _require_assembly_from_detected(
        detect_assembly(fasta_path), expected, context=context, source=str(fasta_path),
    )


def _require_assembly_from_detected(
    found: Optional[str], expected: str, *, context: str = "", source: str,
) -> Optional[str]:
    """Shared raise/warn body for :func:`require_assembly` and its chrom-sizes sibling.

    Factored out so the FASTA path (``.fai``/pysam) and the chrom-sizes path (a
    bigwig's ``bw.chroms()``) can never drift apart in wording or in the
    raise-on-confident-mismatch / warn-on-unrecognized asymmetry — see
    :func:`require_assembly`'s docstring for why that asymmetry is the whole design.
    """
    if expected not in ASSEMBLY_CHR1_LENGTH:
        raise ValueError(
            f"require_assembly(expected={expected!r}) is not an assembly chorus can "
            f"identify; known: {sorted(ASSEMBLY_CHR1_LENGTH)}. A typo here would "
            f"silently disable the check it was added to perform."
        )
    where = f" ({context})" if context else ""
    if found is None:
        logger.warning(
            "Could not identify the assembly of %s%s (no chr1, or a build chorus has "
            "no length for). Expected %s; proceeding unverified.",
            source, where, expected,
        )
        return None
    if found != expected:
        from ..core.exceptions import GenomeAssemblyMismatchError
        raise GenomeAssemblyMismatchError(
            f"{source} is {found}, not {expected}{where}. chr1 is "
            f"{ASSEMBLY_CHR1_LENGTH[found]:,} bp; {expected} has "
            f"{ASSEMBLY_CHR1_LENGTH[expected]:,}. Every coordinate in one assembly "
            f"also exists in the other, so this would not fail -- it would return "
            f"predictions about different DNA than the ones asked for."
        )
    return found


def chr1_length_from_chrom_sizes(chrom_sizes: Dict[str, int]) -> Optional[int]:
    """Length of chr1 in a ``{chrom: length}`` dict (e.g. a bigwig's ``bw.chroms()``).

    Accepts both ``chr1`` and ``1`` keys, mirroring :func:`chr1_length`'s FASTA behaviour.
    """
    for name in ('chr1', '1'):
        if name in chrom_sizes:
            return int(chrom_sizes[name])
    return None


def detect_assembly_from_chrom_sizes(chrom_sizes: Dict[str, int]) -> Optional[str]:
    """Same contract as :func:`detect_assembly`, fingerprinting from a chrom-sizes dict.

    ``None`` means "no claim": no ``chr1``/``1`` key, or a length not in
    :data:`ASSEMBLY_CHR1_LENGTH`.
    """
    length = chr1_length_from_chrom_sizes(chrom_sizes)
    if length is None:
        return None
    return _CHR1_LENGTH_TO_ASSEMBLY.get(length)


def require_assembly_from_chrom_sizes(
    chrom_sizes: Dict[str, int], expected: str, *, context: str = "",
) -> Optional[str]:
    """Same raise/warn/ValueError asymmetry as :func:`require_assembly`, sourced from
    a chrom-sizes dict (e.g. a bigwig's ``bw.chroms()``) rather than opening a FASTA.
    """
    return _require_assembly_from_detected(
        detect_assembly_from_chrom_sizes(chrom_sizes), expected,
        context=context, source=f"chrom-sizes {dict(chrom_sizes)!r}",
    )


def require_assembly_for_bigwig(bigwig_path, expected: str, *, context: str = "") -> Optional[str]:
    """Convenience: open *bigwig_path* with pyBigWig and verify its assembly.

    Extracts ``bw.chroms()`` and delegates to :func:`require_assembly_from_chrom_sizes`,
    so a confident mismatch raises :class:`~chorus.core.exceptions.GenomeAssemblyMismatchError`
    and an unrecognized build only warns — same asymmetry as :func:`require_assembly`.
    """
    import pyBigWig

    bigwig_path = str(bigwig_path)
    with pyBigWig.open(bigwig_path) as bw:
        chrom_sizes = dict(bw.chroms())
    return _require_assembly_from_detected(
        detect_assembly_from_chrom_sizes(chrom_sizes), expected,
        context=context, source=bigwig_path,
    )


def missing_reference_fasta_error(oracle_name: str = "") -> ValueError:
    """The error for a coordinate query with no reference, phrased so it can be acted on.

    Ten sites across nine oracles raised this, in two wordings that differed only by
    punctuation, and neither named the argument to pass or the command that produces the file:

        "Reference FASTA required for genomic coordinate input"
        "Reference FASTA required for genomic coordinates."

    A first-time user reading either has to go and find out that the kwarg is
    ``reference_fasta`` and that chorus can fetch hg38 itself. Both facts belong in the
    message. Audit finding F3, 2026-08-12; the same N-copies-of-one-string shape as #125.
    """
    who = f"{oracle_name} " if oracle_name else ""
    return ValueError(
        f"{who}needs a reference genome to turn coordinates into sequence, and none was "
        f"given. Pass reference_fasta='<path to hg38.fa>' when constructing the oracle "
        f"(chorus.create_oracle(..., reference_fasta=...)), or run `chorus genome download "
        f"hg38` to fetch and index one — `chorus config data-dir` shows where it will land. "
        f"Passing a DNA string instead of coordinates needs no reference."
    )


#: Spellings of "human" an `organism=` argument may use.
_HUMAN_ALIASES = frozenset({'human', 'homo_sapiens', 'homo sapiens', 'hsapiens', 'hg38'})


def require_human_organism(organism: str, *, oracle: str) -> str:
    """Accept a spelling of human; raise on anything else. Returns the normalised value.

    ``AlphaGenomeOracle(organism="mouse")`` used to be accepted, stored on
    ``self.organism``, and never read by anything: the metadata loader hardcodes
    ``Organism.HOMO_SAPIENS`` and the PyTorch port passes ``organism_index=0``.
    So the one oracle whose upstream API genuinely supports mouse had a parameter
    that looked functional, returned human predictions, and labelled them mouse.

    Of the three ways out — make it work, remove it, raise — raising is the only
    one that is both honest and cheap. Making it work is not a parameter fix: it
    needs an mm10/mm39 FASTA in the genome manager, an mm10 reference class for
    the background (SCREEN publishes mm10 cCREs; the Meuleman DHS vocabulary has
    no mouse equivalent), and a full background pass over ~4,300 mouse tracks.
    Removing it would silently change a public signature. So the parameter stays,
    documents the gap, and refuses to pretend.
    """
    if str(organism).strip().lower() in _HUMAN_ALIASES:
        return 'human'
    raise NotImplementedError(
        f"{oracle}(organism={organism!r}) is not supported: chorus is human-only. "
        f"This parameter used to be accepted and silently ignored, which returned "
        f"human predictions labelled as {organism!r}. Mouse needs an mm10 reference "
        f"in the genome manager, an mm10 reference class for the background null "
        f"(the hg38 DHS vocabulary has no mouse equivalent), and a background pass "
        f"over the ~4,300 mouse tracks -- tracked in issue #124. Pass "
        f"organism='human' or omit it."
    )


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