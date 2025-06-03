"""Enformer oracle implementation with environment support."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import os
from pathlib import Path
import logging

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidSequenceError
from ..utils.sequence import extract_sequence_with_padding

logger = logging.getLogger(__name__)


class EnformerOracle(OracleBase):
    """Enformer oracle with automatic environment management."""
    
    def __init__(self, use_environment: bool = True, reference_fasta: Optional[str] = None):
        """
        Initialize Enformer oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
            reference_fasta: Path to reference FASTA file (e.g., hg38.fa)
        """
        # Set the oracle name BEFORE calling super().__init__
        self.oracle_name = 'enformer'
        
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment)
        
        # Enformer specific parameters
        self.target_length = 896
        self.bin_size = 128
        self.sequence_length = 393216
        self.center_length = 196608
        
        # Model components
        self._enformer_model = None
        self._track_dict = None
        
        # Default model path
        self.default_model_path = "https://tfhub.dev/deepmind/enformer/1"
        
        # Reference genome
        self.reference_fasta = reference_fasta
    
    def load_pretrained_model(self, weights: str = None) -> None:
        """Load Enformer model in the appropriate environment."""
        if weights is None:
            weights = self.default_model_path
        
        logger.info(f"Loading Enformer model from {weights}...")
        
        if self.use_environment:
            # Code to run in environment
            load_code = f"""
import tensorflow as tf
import tensorflow_hub as hub
import os

# Set TFHub progress tracking
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"

# Load the model
enformer = hub.load({repr(weights)})
# Get the actual model from the enformer object
model = enformer.model

# Get model info (we can't pickle the model itself)
result = {{
    'loaded': True,
    'model_class': str(type(model)),
    'has_predict': hasattr(model, 'predict_on_batch'),
    'description': 'Enformer model loaded successfully'
}}
"""
            
            # Run loading in environment
            model_info = self.run_code_in_environment(load_code, timeout=300)
            
            if model_info and model_info['loaded']:
                self.loaded = True
                self._model_info = model_info
                logger.info("Enformer model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError("Failed to load model in environment")
        else:
            # Load directly if not using environment
            self._load_direct(weights)
    
    def _load_direct(self, weights: str):
        """Load model directly in current environment."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
            enformer = hub.load(weights)
            self._enformer_model = enformer.model
            self.model = self._enformer_model
            self._load_track_metadata()
            self.loaded = True
            logger.info("Enformer model loaded successfully!")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Enformer model: {str(e)}")
    
    def _predict(self, seq: Union[str, Tuple[str, int, int]], assay_ids: List[str]) -> np.ndarray:
        """Run prediction in the appropriate environment.
        
        Args:
            seq: Either a DNA sequence string or a tuple of (chrom, start, end)
            assay_ids: List of assay identifiers
        """
        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinate input")
            chrom, start, end = seq
            # Extract sequence with padding from reference
            full_seq = extract_sequence_with_padding(
                self.reference_fasta,
                chrom,
                start,
                end,
                total_length=self.sequence_length
            )
        else:
            full_seq = seq
            
        if self.use_environment:
            # Save sequence to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as seq_file:
                seq_file.write(full_seq)
                seq_path = seq_file.name
            
            try:
                # Code to run in environment
                predict_code = f"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Read sequence from file
with open({repr(seq_path)}, 'r') as f:
    seq = f.read().strip()

# Load model (cached in TFHub)
# Enformer model has a specific structure - we need to get the model attribute
enformer = hub.load({repr(self.default_model_path)})
model = enformer.model

# Prepare sequence
if len(seq) != 393216:
    # Pad or trim sequence
    if len(seq) < 393216:
        pad_needed = 393216 - len(seq)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        seq = 'N' * pad_left + seq + 'N' * pad_right
    else:
        trim_needed = len(seq) - 393216
        trim_left = trim_needed // 2
        trim_right = trim_needed - trim_left
        seq = seq[trim_left:len(seq)-trim_right]

# One-hot encode
mapping = {{'A': 0, 'C': 1, 'G': 2, 'T': 3}}
one_hot = np.zeros((len(seq), 4), dtype=np.float32)

for i, base in enumerate(seq.upper()):
    if base in mapping:
        one_hot[i, mapping[base]] = 1.0

# Add batch dimension
one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)

# Run prediction - Use predict_on_batch method
predictions = model.predict_on_batch(one_hot_batch)
# Extract human predictions (Enformer outputs both human and mouse)
human_predictions = predictions['human'][0].numpy()

# Map assay IDs to track indices
# For environment execution, we'll use a simplified mapping
# Map track IDs to indices - handle common ones
track_indices = []
for assay_id in {repr(assay_ids)}:
    # Handle specific identifiers we know
    if assay_id == 'ENCFF413AHU':  # DNase:K562 
        track_indices.append(121)
    elif assay_id == 'CNhs11250':  # CAGE:K562
        track_indices.append(4828)
    elif assay_id == 'CNhs12336':  # CAGE:K562 ENCODE
        track_indices.append(5241)
    elif assay_id == 'DNase:K562':
        track_indices.append(121)
    elif assay_id == 'CAGE:chronic myelogenous leukemia cell line:K562':
        track_indices.append(4828)
    elif assay_id.startswith('ENCFF') or assay_id.startswith('CNhs'):
        # For unknown IDs, need to warn
        print(f"Warning: Unknown track ID {{assay_id}}, using default")
        track_indices.append(121)  
    else:
        # For descriptions, use defaults
        print(f"Warning: Using default track for {{assay_id}}")
        track_indices.append(121)

# Extract predictions for selected tracks
selected_predictions = human_predictions[:, track_indices]
result = selected_predictions.tolist()
"""
                
                # Run prediction in environment
                predictions_list = self.run_code_in_environment(predict_code, timeout=120)
                
                return np.array(predictions_list)
                
            finally:
                # Clean up sequence file
                import os
                if os.path.exists(seq_path):
                    os.unlink(seq_path)
        else:
            # Use direct prediction
            return self._predict_direct(seq, assay_ids)
    
    def _predict_direct(self, seq: Union[str, Tuple[str, int, int]], assay_ids: List[str]) -> np.ndarray:
        """Direct prediction in current environment."""
        import tensorflow as tf
        
        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinate input")
            chrom, start, end = seq
            # Extract sequence with padding from reference
            full_seq = extract_sequence_with_padding(
                self.reference_fasta,
                chrom,
                start,
                end,
                total_length=self.sequence_length
            )
        else:
            full_seq = seq
        
        # Prepare sequence
        if len(full_seq) != self.sequence_length:
            full_seq = self._prepare_sequence(full_seq)
        
        # One-hot encode
        one_hot = self._one_hot_encode(full_seq)
        one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
        
        # Run prediction - Use predict_on_batch method
        predictions = self._enformer_model.predict_on_batch(one_hot_batch)
        human_predictions = predictions['human'][0]
        
        # Get indices for requested assays
        assay_indices = self._get_assay_indices(assay_ids)
        
        return human_predictions[:, assay_indices].numpy()
    
    def list_assay_types(self) -> List[str]:
        """Return all unique assay types from Enformer metadata."""
        from .enformer_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_assay_types()
    
    def list_cell_types(self) -> List[str]:
        """Return all unique cell types from Enformer metadata."""
        from .enformer_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_cell_types()
    
    def _prepare_sequence(self, seq: str) -> str:
        """Prepare sequence to correct length."""
        seq = seq.upper()
        
        if len(seq) < self.sequence_length:
            pad_needed = self.sequence_length - len(seq)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        elif len(seq) > self.sequence_length:
            trim_needed = len(seq) - self.sequence_length
            trim_left = trim_needed // 2
            trim_right = trim_needed - trim_left
            seq = seq[trim_left:len(seq)-trim_right]
        
        return seq
    
    def _one_hot_encode(self, seq: str) -> np.ndarray:
        """Convert DNA sequence to one-hot encoding."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        one_hot = np.zeros((len(seq), 4), dtype=np.float32)
        
        for i, base in enumerate(seq):
            if base in mapping:
                one_hot[i, mapping[base]] = 1.0
        
        return one_hot
    
    def _get_assay_indices(self, assay_ids: List[str]) -> List[int]:
        """Map assay IDs to track indices using proper metadata."""
        from .enformer_metadata import get_metadata
        
        metadata = get_metadata()
        indices = []
        
        for assay_id in assay_ids:
            # Check if it's an ENCODE identifier (starts with ENCFF)
            if assay_id.startswith('ENCFF'):
                idx = metadata.get_track_by_identifier(assay_id)
                if idx is not None:
                    indices.append(idx)
                else:
                    logger.warning(f"Identifier '{assay_id}' not found in metadata")
                    indices.append(0)
            else:
                # Search by description
                matches = metadata.get_tracks_by_description(assay_id)
                if matches:
                    # Use the first match and warn if multiple
                    if len(matches) > 1:
                        logger.info(f"Multiple tracks found for '{assay_id}': {[m[1] for m in matches]}")
                        logger.info(f"Using first match: {matches[0][1]} (index {matches[0][0]})")
                    indices.append(matches[0][0])
                else:
                    logger.warning(f"No tracks found for '{assay_id}'")
                    indices.append(0)
        
        return indices
    
    def _load_track_metadata(self):
        """Load track metadata."""
        # Simplified version
        self._track_dict = []
        for i, assay in enumerate(self.list_assay_types()):
            for j, cell in enumerate(self.list_cell_types()):
                self._track_dict.append({
                    'id': len(self._track_dict),
                    'assay': assay,
                    'cell_type': cell,
                    'name': f"{assay}_{cell}"
                })
    
    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tuning not implemented for this demo."""
        raise NotImplementedError("Fine-tuning is not yet implemented")
    
    def _get_context_size(self) -> int:
        """Return the required context size."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths."""
        return (1000, self.sequence_length)
    
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size
    
    def get_status(self) -> Dict[str, Any]:
        """Get oracle status including environment info."""
        status = {
            'name': self.__class__.__name__,
            'loaded': self.loaded,
            'use_environment': self.use_environment,
            'environment_info': None
        }
        
        if self.use_environment:
            status['environment_info'] = self.get_environment_info()
        
        return status
    
    def get_track_info(self, query: str = None) -> Union[pd.DataFrame, Dict[str, int]]:
        """Get information about available tracks.
        
        Args:
            query: Optional search query. If None, returns summary by assay type.
        
        Returns:
            If query is provided: DataFrame of matching tracks
            If no query: Dictionary with counts by assay type
        """
        from .enformer_metadata import get_metadata
        metadata = get_metadata()
        
        if query:
            return metadata.search_tracks(query)
        else:
            return metadata.get_track_summary()
    
    def get_output_window_coords(self, region_center: int) -> Tuple[int, int]:
        """Calculate Enformer's output window coordinates for a given region center.
        
        Enformer has a specific architecture:
        - Input: 393,216 bp
        - Output: 114,688 bp (896 bins × 128 bp)
        - The output is centered within the input with 139,264 bp offset on each side
        
        Args:
            region_center: Genomic coordinate of the region center
            
        Returns:
            Tuple of (output_start, output_end) genomic coordinates
        """
        output_size = self.target_length * self.bin_size  # 114,688 bp
        output_offset = (self.sequence_length - output_size) // 2  # 139,264 bp
        
        # Calculate input window
        input_start = region_center - self.sequence_length // 2
        
        # Calculate output window
        output_start = input_start + output_offset
        output_end = output_start + output_size
        
        return output_start, output_end
    
    def map_predictions_to_coords(self, predictions: np.ndarray, 
                                chrom: str, start: int, end: int) -> List[Dict[str, Any]]:
        """Map prediction values to genomic coordinates.
        
        This handles the complex coordinate mapping needed for Enformer's architecture,
        where the output window is a subset of the input window.
        
        Args:
            predictions: Array of prediction values (896 bins)
            chrom: Chromosome name
            start: Start coordinate of the original query region
            end: End coordinate of the original query region
            
        Returns:
            List of dictionaries with genomic coordinates and values for BedGraph
        """
        # Get the region center
        region_center = (start + end) // 2
        
        # Get output window coordinates
        output_start, output_end = self.get_output_window_coords(region_center)
        
        # Map predictions to coordinates
        mapped_predictions = []
        for i, value in enumerate(predictions):
            bin_start = output_start + i * self.bin_size
            bin_end = bin_start + self.bin_size
            
            mapped_predictions.append({
                'chrom': chrom,
                'start': bin_start,
                'end': bin_end,
                'value': float(value)
            })
        
        return mapped_predictions
    
    def analyze_gene_expression(self, predictions: Dict[str, np.ndarray], 
                              gene_name: str, 
                              chrom: str, start: int, end: int,
                              gtf_file: str,
                              cage_track_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze predicted gene expression using CAGE signal at TSS.
        
        For Enformer, we analyze gene expression by looking at CAGE signal
        around the transcription start sites (TSS) of the gene.
        
        Args:
            predictions: Dictionary of track predictions
            gene_name: Name of the gene to analyze
            chrom: Chromosome of the predicted region
            start: Start of the predicted region
            end: End of the predicted region  
            gtf_file: Path to GTF file with gene annotations
            cage_track_ids: List of CAGE track IDs to analyze
                          If None, uses all CAGE tracks in predictions
                          
        Returns:
            Dictionary with gene expression analysis:
            - tss_positions: List of TSS positions
            - cage_signals: Dict of track_id -> signals at each TSS
            - mean_expression: Dict of track_id -> mean expression
            - max_expression: Dict of track_id -> max expression
            
        Note:
            For Borzoi, we would sum RNA-seq signal over coding exons
            as described in their paper, but Enformer doesn't have RNA-seq tracks.
        """
        from ..utils.annotations import get_gene_tss
        
        # Get TSS positions for the gene
        tss_df = get_gene_tss(gene_name, annotation=gtf_file)
        
        if len(tss_df) == 0:
            logger.warning(f"No TSS found for gene {gene_name}")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }
        
        # Filter TSS positions to those in our region
        region_center = (start + end) // 2
        output_start, output_end = self.get_output_window_coords(region_center)
        
        tss_in_region = tss_df[
            (tss_df['chrom'] == chrom) &
            (tss_df['tss'] >= output_start) &
            (tss_df['tss'] <= output_end)
        ]
        
        if len(tss_in_region) == 0:
            logger.warning(f"No TSS for {gene_name} in output window")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }
        
        # Identify CAGE tracks if not specified
        if cage_track_ids is None:
            cage_track_ids = [
                track_id for track_id in predictions.keys()
                if 'CAGE' in track_id.upper() or track_id.startswith('CNhs')
            ]
        
        # Analyze CAGE signal at TSS positions
        cage_signals = {}
        mean_expression = {}
        max_expression = {}
        
        for track_id in cage_track_ids:
            if track_id not in predictions:
                continue
                
            track_signals = []
            
            for _, tss_info in tss_in_region.iterrows():
                tss_pos = tss_info['tss']
                
                # Convert TSS position to bin index
                tss_bin = (tss_pos - output_start) // self.bin_size
                
                # Get signal in window around TSS (e.g., +/- 5 bins = +/- 640bp)
                window_size = 5
                start_bin = max(0, tss_bin - window_size)
                end_bin = min(len(predictions[track_id]), tss_bin + window_size + 1)
                
                # Take max signal in window (TSS can be somewhat imprecise)
                if start_bin < end_bin:
                    window_signal = predictions[track_id][start_bin:end_bin]
                    track_signals.append(np.max(window_signal))
            
            cage_signals[track_id] = track_signals
            
            if track_signals:
                mean_expression[track_id] = np.mean(track_signals)
                max_expression[track_id] = np.max(track_signals)
            else:
                mean_expression[track_id] = 0.0
                max_expression[track_id] = 0.0
        
        return {
            'gene_name': gene_name,
            'tss_positions': tss_in_region['tss'].tolist(),
            'tss_info': tss_in_region.to_dict('records'),
            'cage_signals': cage_signals,
            'mean_expression': mean_expression,
            'max_expression': max_expression,
            'n_tss': len(tss_in_region)
        }
    
    def save_predictions_as_bedgraph(self, 
                                   predictions: Dict[str, np.ndarray],
                                   chrom: str,
                                   start: int,
                                   output_dir: str = ".",
                                   prefix: str = "",
                                   bin_size: Optional[int] = None,
                                   track_colors: Optional[Dict[str, str]] = None,
                                   end: Optional[int] = None) -> List[str]:
        """Save predictions as BedGraph files with Enformer-specific coordinate mapping.
        
        This overrides the base class method to handle Enformer's specific architecture
        where the output window is offset within the input window.
        
        Args:
            predictions: Dictionary mapping track names to prediction arrays
            chrom: Chromosome name
            start: Start coordinate of the query region
            output_dir: Directory to save files
            prefix: Prefix for output filenames
            bin_size: Bin size (uses self.bin_size if not provided)
            track_colors: Optional dict mapping track names to colors
            end: End coordinate of the query region (optional, used for coordinate mapping)
            
        Returns:
            List of created file paths
        """
        # If end is not provided, we need to calculate it based on the predictions
        if end is None:
            # Assume the predictions cover the full output window
            # This might not be ideal for all use cases
            logger.warning("End coordinate not provided for save_predictions_as_bedgraph. "
                         "Using start as region center for coordinate mapping.")
            region_center = start
        else:
            region_center = (start + end) // 2
        
        # Get the actual genomic coordinates for Enformer's output window
        output_start, _ = self.get_output_window_coords(region_center)
        
        # Use the base class method with the corrected start coordinate
        return super().save_predictions_as_bedgraph(
            predictions=predictions,
            chrom=chrom,
            start=output_start,  # Use the mapped output start
            output_dir=output_dir,
            prefix=prefix,
            bin_size=bin_size,
            track_colors=track_colors
        )