"""Base class for all oracle implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
import pandas as pd
import re
import os
import logging
from ..core.interval import Interval, GenomeRef

from ..core.track import Track
from ..core.exceptions import (
    ModelNotLoadedError,
    InvalidSequenceError,
    InvalidAssayError,
    InvalidRegionError
)
from ..core.result import OraclePrediction

logger = logging.getLogger(__name__)


class OracleBase(ABC):
    """Abstract base class for all oracle implementations."""
    
    def __init__(self, use_environment: bool = True, 
                 model_load_timeout: Optional[int] = 600,
                 predict_timeout: Optional[int] = 300,
                 device: Optional[str] = None):
        self.model = None
        self.loaded = False
        self._assay_types = []
        self._cell_types = []
        # Environment management
        self.use_environment = use_environment
        # Remember whether the user asked for env isolation so we can
        # distinguish a later silent flip to False ("env setup failed")
        # from a legitimate test/library-internal use_environment=False.
        # Only the former should raise EnvironmentNotReadyError.
        self._user_asked_for_env = use_environment
        self._env_setup_error: Optional[str] = None
        self._env_manager = None
        self._env_runner = None
        
        # Timeout settings (in seconds)
        # Check for global timeout disable
        if os.environ.get('CHORUS_NO_TIMEOUT', '').lower() in ('1', 'true', 'yes'):
            logger.info("CHORUS_NO_TIMEOUT is set - all timeouts disabled")
            self.model_load_timeout = None
            self.predict_timeout = None
        else:
            self.model_load_timeout = model_load_timeout  # Default 10 minutes
            self.predict_timeout = predict_timeout  # Default 5 minutes
        
        # Device settings
        # Default: None (auto-detect GPU if available)
        # Options: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.device = device or os.environ.get('CHORUS_DEVICE')
        if self.device:
            logger.info(f"Device set to: {self.device}")
        else:
            logger.info("Device: auto-detect (GPU if available, else CPU)")
        
        # Set oracle name if not already set by subclass
        if not hasattr(self, 'oracle_name'):
            self.oracle_name = self.__class__.__name__.lower().replace('oracle', '')
        
        # Initialize environment if requested
        if self.use_environment:
            self._setup_environment()

    
    @abstractmethod
    def load_pretrained_model(self, weights: str) -> None:
        """Load pre-trained model weights."""
        pass
    
    @abstractmethod
    def list_assay_types(self) -> List[str]:
        """Return list of available assay types."""
        pass
    
    @abstractmethod
    def list_cell_types(self) -> List[str]:
        """Return list of available cell types."""
        pass
    
    def _setup_environment(self):
        """Set up environment management for the oracle.

        On success, ``self.use_environment`` remains True and the
        ``_env_manager`` / ``_env_runner`` are attached. On any failure
        (missing env, invalid env, import error, unexpected exception)
        we flip ``use_environment`` to False **and** record a deferred
        reason in ``self._env_setup_error``. The next call to a user-
        facing method (``load_pretrained_model`` / ``predict``) raises
        :class:`EnvironmentNotReadyError` with that reason — so users
        who opted into ``use_environment=True`` don't silently fall
        back to running in the base env and hit a confusing
        ``ModuleNotFoundError`` for a framework-specific package.
        Regression for v26 P1 #11.
        """
        self._env_setup_error: Optional[str] = None
        try:
            from ..core.environment import EnvironmentManager, EnvironmentRunner

            self._env_manager = EnvironmentManager()
            self._env_runner = EnvironmentRunner(self._env_manager)

            # Check if environment exists
            if not self._env_manager.environment_exists(self.oracle_name):
                msg = (
                    f"Environment for {self.oracle_name} does not exist. "
                    f"Run 'chorus setup --oracle {self.oracle_name}' to create it."
                )
                logger.warning(msg)
                self._env_setup_error = msg
                self.use_environment = False
            else:
                # Validate environment
                is_valid, issues = self._env_manager.validate_environment(self.oracle_name)
                if not is_valid:
                    # Distinguish transient timeouts (slow NFS, cold-cache
                    # `import jax`) from genuine missing-dep failures.
                    # Genuine failures return fast and are real signal;
                    # timeouts are not — the actual subprocess invocation
                    # has its own per-call timeout and will surface a real
                    # ModuleNotFoundError if the env is truly broken.
                    # Issue #64.
                    timeout_only = bool(issues) and all(
                        i.startswith("Timeout while checking dependency")
                        for i in issues
                    )
                    if timeout_only:
                        logger.warning(
                            f"Environment validation for {self.oracle_name} "
                            f"timed out ({'; '.join(issues)}). Proceeding "
                            f"with use_environment=True; the actual run "
                            f"will surface any real issue."
                        )
                        # Keep use_environment=True; do NOT set _env_setup_error.
                    else:
                        msg = (
                            f"Environment validation failed for {self.oracle_name}: "
                            f"{'; '.join(issues)}. Run 'chorus health --oracle "
                            f"{self.oracle_name}' to diagnose, or 'chorus setup "
                            f"--oracle {self.oracle_name} --force' to rebuild."
                        )
                        logger.warning(msg)
                        self._env_setup_error = msg
                        self.use_environment = False
                else:
                    logger.info(f"Using conda environment: chorus-{self.oracle_name}")
        except ImportError as e:
            msg = f"Environment management not available ({e}). Oracle will run in current environment."
            logger.warning(msg)
            self._env_setup_error = msg
            self.use_environment = False
        except Exception as e:
            msg = f"Failed to set up environment: {e}. Oracle will run in current environment."
            logger.warning(msg)
            self._env_setup_error = msg
            self.use_environment = False

    def _check_env_ready(self) -> None:
        """Raise ``EnvironmentNotReadyError`` if the caller asked for
        ``use_environment=True`` but setup failed.

        Called by :meth:`predict` and friends so users get a clear
        error instead of a confusing downstream ``ModuleNotFoundError``
        when ``_setup_environment`` flipped ``use_environment`` to
        False silently.
        """
        from .exceptions import EnvironmentNotReadyError
        err = getattr(self, "_env_setup_error", None)
        # Only raise when the user explicitly asked for environment
        # isolation. use_environment=False at construction is a
        # legitimate test/library-internal path.
        if err and not self.use_environment and self._user_asked_for_env:
            raise EnvironmentNotReadyError(err)
    
    def run_in_environment(self, func: Any, *args, **kwargs) -> Any:
        """Run a function in the oracle's environment if available."""
        if self.use_environment and self._env_runner:
            return self._env_runner.run_in_environment(
                self.oracle_name, func, args, kwargs
            )
        else:
            # Run directly in current environment
            return func(*args, **kwargs)
    
    def run_code_in_environment(self, code: str, timeout: Optional[int] = None) -> Any:
        """Run code in the oracle's environment and return the result."""
        if self.use_environment and self._env_runner:
            return self._env_runner.run_code_in_environment(
                self.oracle_name, code, timeout
            )
        else:
            # Run directly in current environment
            local_vars = {}
            exec(code, {'__builtins__': __builtins__}, local_vars)
            return local_vars.get('result')
    
    def get_environment_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the oracle's environment."""
        if self._env_manager:
            return self._env_manager.get_environment_info(self.oracle_name)
        return None
    
    def predict(
        self,
        input_data: Union[str, Tuple[str, int, int]],
        assay_ids: List[str] | None = None,
    ) -> OraclePrediction:
        """
        Predict regulatory activity for a sequence or genomic region.
        
        Args:
            input_data: Either a DNA sequence string or a tuple of (chrom, start, end)
            assay_ids: List of assay identifiers (e.g., ['ENCFF413AHU'] or ['DNase:K562'])
            create_tracks: Whether to create track files (not implemented yet)
            
        Returns:
            Dictionary mapping assay IDs to prediction arrays
            
        Example:
            >>> # Using sequence
            >>> predictions = oracle.predict('ACGT...', ['DNase:K562'])
            >>> 
            >>> # Using genomic coordinates (requires reference_fasta)
            >>> predictions = oracle.predict(('chrX', 48780505, 48785229), ['ENCFF413AHU'])
        """
        # Validate inputs
        self._check_env_ready()

        self._validate_loaded()
        self._validate_assay_ids(assay_ids)

        # Get raw predictions
        predictions = self._predict(input_data, assay_ids)
        return predictions

    def get_output_window_offsets(self) -> Tuple[int, int]:
        """
           by default we assume that model predicts for the same size of window it accepts
        """
        return 0, 0
       
    def predict_region_replacement(
        self,
        genomic_region: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str],
        genome: Optional[str] = None
    ) ->  dict[str, OraclePrediction]:
        """
        Replace a genomic region with a new sequence and predict activity.
        
        Args:
            genomic_region: BED format string "chr1:1000-2000" or DataFrame
            seq: DNA sequence to insert (replaces only the specified region)
            assay_ids: List of assay identifiers
            create_tracks: Whether to save tracks as files
            genome: Path to reference genome FASTA
            
        Returns:
            Dictionary with raw_predictions, normalized_scores, 
            track_objects, and track_files
        """
        # Validate inputs
        self._check_env_ready()

        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        self._validate_dna_sequence(seq)  # Only validate DNA content, not length
        
        # Use instance reference_fasta if no genome provided
        if genome is None:
            if hasattr(self, 'reference_fasta') and self.reference_fasta:
                genome = self.reference_fasta
            else:
                raise ValueError(
                    "No reference genome provided. Either pass genome parameter or "
                    "initialize oracle with reference_fasta."
                )

        # Parse region to replace
        chrom, start, end = self._parse_region(genomic_region)
        region_interval = Interval.make(GenomeRef(chrom=chrom, start=start, end=end, fasta=genome))
        region_interval = region_interval.replace(seq=seq, start=0, end=end-start)
        
        # Get predictions for the full context
        predictions = self._predict(region_interval, 
                                    assay_ids)
        
        results = {
            'raw_predictions': predictions,
            'normalized_scores': predictions.normalize()
        }

        return results 
    
    def predict_region_insertion_at(
        self,
        genomic_position: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str],
        genome: Optional[str] = None
    ) -> dict[str, OraclePrediction]:
        """Insert sequence at a specific position and predict."""
        # Validate inputs
        self._check_env_ready()

        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        self._validate_dna_sequence(seq)  # Only validate DNA content, not length
        
        # Use instance reference_fasta if no genome provided
        if genome is None:
            if hasattr(self, 'reference_fasta') and self.reference_fasta:
                genome = self.reference_fasta
            else:
                raise ValueError(
                    "No reference genome provided. Either pass genome parameter or "
                    "initialize oracle with reference_fasta."
                )
        
        # Parse position
        chrom, position = self._parse_position(genomic_position)

        position_interval = Interval.make(GenomeRef(chrom=chrom, 
                                                    start=position,
                                                    end=position, 
                                                    fasta=genome))
        insertion_interval = position_interval.insert(seq=seq, pos=0)
        
        # Get predictions
        predictions = self._predict(insertion_interval, assay_ids) # contains prediction without correct intervals 
    
        results = {
            'raw_predictions': predictions,
            'normalized_scores': predictions.normalize()
        }

        return results
    
    def predict_variant_effect(
        self,
        genomic_region: Union[str, pd.DataFrame],
        variant_position: Union[str, pd.DataFrame],
        alleles: Union[List[str], pd.DataFrame],
        assay_ids: List[str],
        genome: Optional[str] = None
    ) -> Dict:
        """Predict effects of variants."""
        # Validate inputs
        self._check_env_ready()

        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        
        # Use instance reference_fasta if no genome provided
        if genome is None:
            if hasattr(self, 'reference_fasta') and self.reference_fasta:
                genome = self.reference_fasta
            else:
                raise ValueError(
                    "No reference genome provided. Either pass genome parameter or "
                    "initialize oracle with reference_fasta."
                )
        
        # Parse inputs.
        #
        # Convention: string-form `genomic_region` ("chr1:S-E") and
        # `variant_position` ("chr1:P") are 1-based inclusive, matching
        # dbSNP/UCSC/IGV/`extract_sequence`. GenomeRef is 0-based
        # half-open (see chorus/core/interval.py:26), so we convert when
        # building the interval and when looking up the variant base.
        # Without this conversion the ref-allele check reads the base one
        # position to the right of what the user intended (rs12740374 at
        # chr1:109274968 returned 'T' instead of 'G').
        region_chrom, region_start, region_end = self._parse_region(genomic_region)
        var_chrom, var_pos = self._parse_position(variant_position)

        if region_chrom != var_chrom:
            raise InvalidRegionError("Variant and region must be on the same chromosome")

        if not (region_start <= var_pos <= region_end):
            raise InvalidRegionError("Variant position must be within the specified region")

        region_start_0based = region_start - 1  # 1-based inclusive → 0-based
        var_pos_0based = var_pos - 1            # 1-based → 0-based

        region_interval = Interval.make(GenomeRef(chrom=region_chrom,
                                                  start=region_start_0based,
                                                  end=region_end,
                                                  fasta=genome))

        # Parse alleles
        if isinstance(alleles, pd.DataFrame):
            ref_allele = alleles.iloc[0]['ref']
            alt_alleles = alleles['alt'].tolist()
        else:
            ref_allele = alleles[0]
            alt_alleles = alleles[1:]

        # SNV-only validation. `predict_variant_effect` substitutes a
        # single base at `real_pos` (see lines ~342 / ~347 below), so an
        # indel (ref='G' / alt='GT' or ref='GT' / alt='G') would silently
        # be treated as a 1-base swap — giving a semantically nonsense
        # score. Reject up-front with a clear message naming the bad
        # allele, so the user doesn't waste an oracle run on invalid
        # input (v20 §14.2 finding).
        all_alleles = [ref_allele, *alt_alleles]
        for a in all_alleles:
            if not isinstance(a, str) or len(a) != 1 or a.upper() not in "ACGTN":
                raise InvalidRegionError(
                    f"Allele {a!r} is not a single-nucleotide variant. "
                    f"predict_variant_effect currently supports SNVs only "
                    f"(ref/alt each one of A, C, G, T, N). For indels or "
                    f"multi-base substitutions, use predict_region_replacement "
                    f"with explicit genomic_region + seq."
                )

        intervals = {}
        real_pos = region_interval.ref2query(var_pos_0based, ref_global=True)
        # Case-insensitive compare: pyfaidx returns lowercase for softmasked
        # (repetitive) regions, users always pass uppercase.
        if region_interval[real_pos].upper() != ref_allele.upper():
            logger.warning(
                'Provided reference allele %r does not match the genome at this position (%r). Chorus will use the provided allele.',
                ref_allele, region_interval[real_pos],
            )
            intervals['reference'] = region_interval.replace(seq=ref_allele, start=real_pos, end=real_pos+1)
        else:
            intervals['reference'] = region_interval
        
        for i, alt in enumerate(alt_alleles):
            intervals[f'alt_{i+1}'] = region_interval.replace(seq=alt, start=real_pos, end=real_pos+1)

        
        # Get predictions for each sequence
        all_predictions = {}
        
        for allele_name, interval in intervals.items():
            predictions = self._predict(interval, assay_ids)
            all_predictions[allele_name] = predictions
    
        # Calculate effect sizes using actual prediction keys (may differ from
        # input assay_ids, e.g. ChromBPNet returns "ATAC:K562" not "ATAC")
        ref_keys = list(all_predictions['reference'].keys())
        effect_sizes = {}
        for allele_name in ['alt_' + str(i+1) for i in range(len(alt_alleles))]:
            effect_sizes[allele_name] = {
                assay: all_predictions[allele_name][assay].values - all_predictions['reference'][assay].values
                for assay in ref_keys
            }
        
        return {
            'predictions': all_predictions,
            'effect_sizes': effect_sizes,
            'variant_info': {
                'position': f"{var_chrom}:{var_pos}",
                'ref': ref_allele,
                'alts': alt_alleles
            }
        }
    
    @abstractmethod
    def fine_tune(
        self,
        tracks: List[Track],
        track_names: List[str],
        **kwargs
    ) -> None:
        """Fine-tune model on new tracks."""
        pass
    
    # Helper methods
    def _validate_loaded(self):
        """Check if model is loaded."""
        if not self.loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_pretrained_model first.")
    
    def _validate_assay_ids(self, assay_ids: List[str] | None = None):
        return True # for base class it is not needed to validate assay ids
    
    def _validate_sequence(self, seq: str):
        """Validate DNA sequence."""
        # Check if sequence contains only valid nucleotides
        valid_nucleotides = set('ACGTNacgtn')
        if not all(base in valid_nucleotides for base in seq):
            raise InvalidSequenceError(
                f"Sequence contains invalid characters. Only A, C, G, T, N allowed."
            )
        
        # Check sequence length
        min_len, max_len = self._get_sequence_length_bounds()
        if not (min_len <= len(seq) <= max_len):
            raise InvalidSequenceError(
                f"Sequence length {len(seq)} is outside valid range [{min_len}, {max_len}]"
            )
    
    def _validate_dna_sequence(self, seq: str):
        """Validate DNA sequence content only (not length)."""
        # Check if sequence contains only valid nucleotides
        valid_nucleotides = set('ACGTNacgtn')
        if not all(base in valid_nucleotides for base in seq):
            raise InvalidSequenceError(
                f"Sequence contains invalid characters. Only A, C, G, T, N allowed."
            )
        
        # Check for empty sequence
        if len(seq) == 0:
            raise InvalidSequenceError("Sequence cannot be empty")
    
    def _parse_region(self, genomic_region: Union[str, pd.DataFrame]) -> Tuple[str, int, int]:
        """Parse genomic region into chromosome, start, end."""
        if isinstance(genomic_region, pd.DataFrame):
            # Assume first row contains the region
            row = genomic_region.iloc[0]
            return str(row['chrom']), int(row['start']), int(row['end'])
        else:
            # Parse string format "chr1:1000-2000"
            match = re.match(r'(\w+):(\d+)-(\d+)', genomic_region)
            if match:
                chrom, start, end = match.groups()
                return chrom, int(start), int(end)
            else:
                raise InvalidRegionError(f"Invalid region format: {genomic_region}")
    
    def _parse_position(self, genomic_position: Union[str, pd.DataFrame]) -> Tuple[str, int]:
        """Parse genomic position into chromosome and position."""
        if isinstance(genomic_position, pd.DataFrame):
            row = genomic_position.iloc[0]
            return str(row['chrom']), int(row['pos'])
        else:
            # Parse string format "chr1:1000"
            match = re.match(r'(\w+):(\d+)', genomic_position)
            if match:
                chrom, pos = match.groups()
                return chrom, int(pos)
            else:
                raise InvalidRegionError(f"Invalid position format: {genomic_position}")
    
    def analyze_gene_expression(
        self,
        predictions: OraclePrediction,
        gene_name: str,
        expression_track_ids: Optional[List[str]] = None,
        cage_window_bin_size: int = 5,
    ) -> Dict:
        """Analyze predicted gene expression using CAGE and/or RNA-seq signal.

        Auto-detects expression tracks in the prediction by track type:
        - CAGEOraclePredictionTrack / LentiMPRAOraclePredictionTrack → TSS windowed max
        - RNAOraclePredictionTrack → sum signal over merged exonic regions

        Args:
            predictions: OraclePrediction from predict()
            gene_name: Gene symbol (e.g. 'MYC', 'TP53')
            expression_track_ids: Override auto-detection with specific track IDs
            cage_window_bin_size: Bins around TSS for CAGE quantification (default ±5)

        Returns:
            Dict with gene_name, tss_positions, exon_regions, and per-track
            expression quantification.
        """
        from ..utils.annotations import get_gene_tss, get_gene_exons
        from ..core.result import (
            CAGEOraclePredictionTrack,
            LentiMPRAOraclePredictionTrack,
            RNAOraclePredictionTrack,
        )

        # Get TSS positions
        tss_df = get_gene_tss(gene_name)

        # Auto-detect or filter expression tracks
        if expression_track_ids is not None:
            tracks_to_analyze = {
                tid: predictions[tid] for tid in expression_track_ids
            }
        else:
            tracks_to_analyze = {
                tid: track for tid, track in predictions.items()
                if isinstance(track, (CAGEOraclePredictionTrack,
                                      LentiMPRAOraclePredictionTrack,
                                      RNAOraclePredictionTrack))
            }

        if not tracks_to_analyze:
            present_types = sorted({
                type(t).__name__ for t in predictions.values()
            })
            return {
                'gene_name': gene_name,
                'tss_positions': tss_df['tss'].tolist() if len(tss_df) > 0 else [],
                'exon_regions': [],
                'per_track': {},
                'warning': (
                    f"No expression tracks (CAGE/RNA) found in prediction. "
                    f"Track types present: {present_types}. "
                    f"Gene expression analysis requires CAGE or RNA track types."
                ),
            }

        # Check if any RNA tracks are present — only fetch exons if needed
        has_rna = any(
            isinstance(t, RNAOraclePredictionTrack) for t in tracks_to_analyze.values()
        )
        exon_df = get_gene_exons(gene_name) if has_rna else pd.DataFrame()

        per_track = {}
        for track_id, track in tracks_to_analyze.items():
            chrom = track.prediction_interval.reference.chrom

            if isinstance(track, (CAGEOraclePredictionTrack, LentiMPRAOraclePredictionTrack)):
                # CAGE / LentiMPRA: windowed max around TSS
                tss_in_window = tss_df[
                    (tss_df['chrom'] == chrom) &
                    (tss_df['tss'] >= track.start) &
                    (tss_df['tss'] <= track.end)
                ]
                if len(tss_in_window) == 0:
                    per_track[track_id] = {
                        'expression': 0.0,
                        'quantification_method': 'tss_windowed_max',
                        'n_tss_in_window': 0,
                    }
                    continue

                tss_signals = []
                for _, row in tss_in_window.iterrows():
                    tss_bin = track.pos2bin(row['chrom'], row['tss'])
                    if tss_bin is None:
                        continue
                    s = max(0, tss_bin - cage_window_bin_size)
                    e = min(len(track.values), tss_bin + cage_window_bin_size + 1)
                    if s < e:
                        tss_signals.append(float(np.max(track.values[s:e])))

                expression = max(tss_signals) if tss_signals else 0.0
                per_track[track_id] = {
                    'expression': expression,
                    'quantification_method': 'tss_windowed_max',
                    'n_tss_in_window': len(tss_in_window),
                    'tss_signals': tss_signals,
                }

            elif isinstance(track, RNAOraclePredictionTrack):
                # RNA: sum signal over merged exons
                if len(exon_df) == 0:
                    per_track[track_id] = {
                        'expression': 0.0,
                        'quantification_method': 'exon_sum',
                        'n_exons_in_window': 0,
                    }
                    continue

                exon_sum = 0.0
                n_exons = 0
                for _, exon in exon_df.iterrows():
                    score = track.score_region(exon['chrom'], exon['start'], exon['end'], 'sum')
                    if score is not None:
                        exon_sum += score
                        n_exons += 1

                per_track[track_id] = {
                    'expression': exon_sum,
                    'quantification_method': 'exon_sum',
                    'n_exons_in_window': n_exons,
                }

        return {
            'gene_name': gene_name,
            'tss_positions': tss_df['tss'].tolist() if len(tss_df) > 0 else [],
            'exon_regions': exon_df[['chrom', 'start', 'end']].to_dict('records') if len(exon_df) > 0 else [],
            'per_track': per_track,
        }

    def analyze_variant_effect_on_gene(
        self,
        variant_result: Dict,
        gene_name: str,
        expression_track_ids: Optional[List[str]] = None,
        cage_window_bin_size: int = 5,
    ) -> Dict:
        """Predict how a variant affects expression of a nearby gene.

        Calls analyze_gene_expression() on reference and each alternate allele,
        then computes fold change, log2 fold change, and absolute change.

        Args:
            variant_result: Return value of predict_variant_effect().
            gene_name: Gene symbol (e.g. 'MYC', 'TP53')
            expression_track_ids: Override auto-detection with specific track IDs
            cage_window_bin_size: Bins around TSS for CAGE quantification

        Returns:
            Dict with gene_name, variant_info, reference_expression,
            and per-allele expression with fold changes.
        """
        ref_expr = self.analyze_gene_expression(
            variant_result['predictions']['reference'],
            gene_name,
            expression_track_ids=expression_track_ids,
            cage_window_bin_size=cage_window_bin_size,
        )

        per_allele = {}
        for allele_name, pred in variant_result['predictions'].items():
            if allele_name == 'reference':
                continue
            alt_expr = self.analyze_gene_expression(
                pred, gene_name,
                expression_track_ids=expression_track_ids,
                cage_window_bin_size=cage_window_bin_size,
            )

            allele_vs_ref = {}
            for track_id in ref_expr['per_track']:
                ref_val = ref_expr['per_track'][track_id]['expression']
                alt_val = alt_expr['per_track'].get(track_id, {}).get('expression', 0.0)
                abs_change = alt_val - ref_val
                if ref_val != 0:
                    fold_change = alt_val / ref_val
                    log2fc = float(np.log2(alt_val / ref_val)) if alt_val > 0 and ref_val > 0 else None
                else:
                    fold_change = None
                    log2fc = None

                allele_vs_ref[track_id] = {
                    'ref_expression': ref_val,
                    'alt_expression': alt_val,
                    'absolute_change': abs_change,
                    'fold_change': fold_change,
                    'log2_fold_change': log2fc,
                }

            per_allele[allele_name] = {
                'expression': alt_expr['per_track'],
                'vs_reference': allele_vs_ref,
            }

        return {
            'gene_name': gene_name,
            'variant_info': variant_result['variant_info'],
            'tss_positions': ref_expr['tss_positions'],
            'exon_regions': ref_expr['exon_regions'],
            'reference_expression': ref_expr['per_track'],
            'per_allele': per_allele,
        }

    @abstractmethod
    def _predict(self, seq: str, assay_ids: List[str]) -> OraclePrediction:
        """Internal prediction method to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        pass
    
    @abstractmethod
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths accepted by the model."""
        pass
    
    @abstractmethod
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        pass
