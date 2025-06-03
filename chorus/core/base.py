"""Base class for all oracle implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
import pandas as pd
import re
import os
import logging
from pathlib import Path

from ..core.track import Track
from ..core.exceptions import (
    ModelNotLoadedError,
    InvalidSequenceError,
    InvalidAssayError,
    InvalidRegionError
)

logger = logging.getLogger(__name__)


class OracleBase(ABC):
    """Abstract base class for all oracle implementations."""
    
    def __init__(self, use_environment: bool = True):
        self.model = None
        self.loaded = False
        self._assay_types = []
        self._cell_types = []
        
        # Environment management
        self.use_environment = use_environment
        self._env_manager = None
        self._env_runner = None
        
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
        """Set up environment management for the oracle."""
        try:
            from ..core.environment import EnvironmentManager, EnvironmentRunner
            
            self._env_manager = EnvironmentManager()
            self._env_runner = EnvironmentRunner(self._env_manager)
            
            # Check if environment exists
            if not self._env_manager.environment_exists(self.oracle_name):
                logger.warning(
                    f"Environment for {self.oracle_name} does not exist. "
                    f"Run 'chorus setup --oracle {self.oracle_name}' to create it."
                )
                self.use_environment = False
            else:
                # Validate environment
                is_valid, issues = self._env_manager.validate_environment(self.oracle_name)
                if not is_valid:
                    logger.warning(
                        f"Environment validation failed for {self.oracle_name}: "
                        f"{'; '.join(issues)}"
                    )
                    self.use_environment = False
                else:
                    logger.info(f"Using conda environment: chorus-{self.oracle_name}")
        except ImportError:
            logger.warning("Environment management not available. Running in current environment.")
            self.use_environment = False
        except Exception as e:
            logger.warning(f"Failed to set up environment: {e}")
            self.use_environment = False
    
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
        assay_ids: List[str],
        create_tracks: bool = False
    ) -> Dict[str, np.ndarray]:
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
        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        
        # Get raw predictions
        predictions = self._predict(input_data, assay_ids)
        
        # Return as dictionary
        result = {}
        for i, assay_id in enumerate(assay_ids):
            result[assay_id] = predictions[:, i]
        
        return result
    
    def predict_region_replacement(
        self,
        genomic_region: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str],
        create_tracks: bool = False,
        genome: str = "hg38.fa"
    ) -> Dict:
        """
        Replace a genomic region with a new sequence and predict activity.
        
        Args:
            genomic_region: BED format string "chr1:1000-2000" or DataFrame
            seq: DNA sequence to insert
            assay_ids: List of assay identifiers
            create_tracks: Whether to save tracks as files
            genome: Path to reference genome FASTA
            
        Returns:
            Dictionary with raw_predictions, normalized_scores, 
            track_objects, and track_files
        """
        # Validate inputs
        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        self._validate_sequence(seq)
        
        # Parse region
        chrom, start, end = self._parse_region(genomic_region)
        
        # Get predictions
        predictions = self._predict(seq, assay_ids)
        
        # Format results
        return self._format_results(
            predictions, assay_ids, chrom, start, end, create_tracks
        )
    
    def predict_region_insertion_at(
        self,
        genomic_position: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str],
        create_tracks: bool = False,
        genome: str = "hg38.fa"
    ) -> Dict:
        """Insert sequence at a specific position and predict."""
        # Validate inputs
        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        self._validate_sequence(seq)
        
        # Parse position
        chrom, position = self._parse_position(genomic_position)
        
        # Extract flanking sequences
        from ..utils.sequence import extract_sequence
        
        # Get context window size based on model requirements
        context_size = self._get_context_size()
        flank_size = (context_size - len(seq)) // 2
        
        # Extract flanking sequences
        left_flank = extract_sequence(
            f"{chrom}:{position-flank_size}-{position}", genome
        )
        right_flank = extract_sequence(
            f"{chrom}:{position}-{position+flank_size}", genome
        )
        
        # Construct full sequence
        full_seq = left_flank + seq + right_flank
        
        # Get predictions
        predictions = self._predict(full_seq, assay_ids)
        
        # Format results with insertion coordinates
        return self._format_results(
            predictions, assay_ids, chrom, position, position + len(seq), create_tracks
        )
    
    def predict_variant_effect(
        self,
        genomic_region: Union[str, pd.DataFrame],
        variant_position: Union[str, pd.DataFrame],
        alleles: Union[List[str], pd.DataFrame],
        assay_ids: List[str],
        create_tracks: bool = False,
        genome: str = "hg38.fa"
    ) -> Dict:
        """Predict effects of variants."""
        # Validate inputs
        self._validate_loaded()
        self._validate_assay_ids(assay_ids)
        
        # Parse inputs
        region_chrom, region_start, region_end = self._parse_region(genomic_region)
        var_chrom, var_pos = self._parse_position(variant_position)
        
        if region_chrom != var_chrom:
            raise InvalidRegionError("Variant and region must be on the same chromosome")
        
        if not (region_start <= var_pos < region_end):
            raise InvalidRegionError("Variant position must be within the specified region")
        
        # Parse alleles
        if isinstance(alleles, pd.DataFrame):
            ref_allele = alleles.iloc[0]['ref']
            alt_alleles = alleles['alt'].tolist()
        else:
            ref_allele = alleles[0]
            alt_alleles = alleles[1:]
        
        # Extract reference sequence
        from ..utils.sequence import extract_sequence, apply_variant
        ref_seq = extract_sequence(genomic_region, genome)
        
        # Create sequences for each allele
        sequences = {'reference': ref_seq}
        relative_pos = var_pos - region_start
        
        for i, alt in enumerate(alt_alleles):
            alt_seq = apply_variant(ref_seq, relative_pos, ref_allele, alt)
            sequences[f'alt_{i+1}'] = alt_seq
        
        # Get predictions for each sequence
        all_predictions = {}
        all_tracks = {}
        all_files = {}
        
        for allele_name, seq in sequences.items():
            predictions = self._predict(seq, assay_ids)
            results = self._format_results(
                predictions, assay_ids, region_chrom, region_start, region_end, create_tracks
            )
            
            all_predictions[allele_name] = results['raw_predictions']
            if 'track_objects' in results:
                all_tracks[allele_name] = results['track_objects']
            if 'track_files' in results:
                all_files[allele_name] = results['track_files']
        
        # Calculate effect sizes
        effect_sizes = {}
        for allele_name in ['alt_' + str(i+1) for i in range(len(alt_alleles))]:
            effect_sizes[allele_name] = {
                assay: all_predictions[allele_name][assay] - all_predictions['reference'][assay]
                for assay in assay_ids
            }
        
        return {
            'predictions': all_predictions,
            'effect_sizes': effect_sizes,
            'track_objects': all_tracks if all_tracks else None,
            'track_files': all_files if all_files else None,
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
    
    def _validate_assay_ids(self, assay_ids: List[str]):
        """Validate assay IDs."""
        valid_assays = self.list_assay_types()
        valid_cells = self.list_cell_types()
        valid_ids = valid_assays + valid_cells
        
        # Also check for combined format like "DNASE:K562" and ENCODE identifiers
        for assay_id in assay_ids:
            # Skip validation for ENCODE identifiers (start with ENCFF)
            if assay_id.startswith('ENCFF'):
                continue
            elif ':' in assay_id:
                assay, cell = assay_id.split(':', 1)
                if assay not in valid_assays or cell not in valid_cells:
                    raise InvalidAssayError(f"Invalid assay ID: {assay_id}")
            elif assay_id not in valid_ids:
                raise InvalidAssayError(f"Invalid assay ID: {assay_id}")
    
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
    
    @abstractmethod
    def _predict(self, seq: str, assay_ids: List[str]) -> np.ndarray:
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
    
    def _format_results(
        self,
        predictions: np.ndarray,
        assay_ids: List[str],
        chrom: str,
        start: int,
        end: int,
        create_tracks: bool
    ) -> Dict:
        """Format prediction results."""
        results = {
            'raw_predictions': {},
            'normalized_scores': {}
        }
        
        track_objects = []
        track_files = []
        
        # Get bin size for the model
        bin_size = self._get_bin_size()
        
        for i, assay_id in enumerate(assay_ids):
            # Store raw predictions
            results['raw_predictions'][assay_id] = predictions[:, i]
            
            # Normalize scores
            results['normalized_scores'][assay_id] = self._normalize_predictions(
                predictions[:, i]
            )
            
            # Create track if requested
            if create_tracks:
                # Create track data
                num_bins = predictions.shape[0]
                track_data = []
                
                for j in range(num_bins):
                    track_data.append({
                        'chrom': chrom,
                        'start': start + j * bin_size,
                        'end': start + (j + 1) * bin_size,
                        'value': float(predictions[j, i])
                    })
                
                # Parse assay and cell type
                if ':' in assay_id:
                    assay_type, cell_type = assay_id.split(':', 1)
                else:
                    assay_type = assay_id
                    cell_type = "unknown"
                
                # Create Track object
                track = Track(
                    name=f"{assay_id}_{chrom}_{start}_{end}",
                    assay_type=assay_type,
                    cell_type=cell_type,
                    data=pd.DataFrame(track_data)
                )
                track_objects.append(track)
                
                # Save to file
                filename = f"{assay_id}_{chrom}_{start}_{end}.bedgraph"
                track.to_bedgraph(filename)
                track_files.append(filename)
        
        if create_tracks:
            results['track_objects'] = track_objects
            results['track_files'] = track_files
        
        return results
    
    @abstractmethod
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        pass
    
    def _normalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Normalize predictions to a standard scale."""
        # Default implementation: min-max normalization
        min_val = np.min(predictions)
        max_val = np.max(predictions)
        if max_val > min_val:
            return (predictions - min_val) / (max_val - min_val)
        else:
            return predictions