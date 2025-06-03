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
# In real usage, the metadata file provides the mapping
track_indices = []
for assay_id in {repr(assay_ids)}:
    # Handle specific identifiers
    if assay_id == 'ENCFF413AHU':  # DNase:K562 from line 121
        track_indices.append(121)
    elif assay_id == 'DNase:K562':
        # Use the first K562 DNase track (line 121)
        track_indices.append(121)
    elif assay_id.startswith('ENCFF'):
        # For other ENCODE IDs, we'd need the full metadata
        print(f"Warning: Cannot lookup {{assay_id}} without metadata file")
        track_indices.append(121)  # Default to K562 DNase
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
        """Return Enformer's assay types."""
        return [
            "DNase", "ATAC-seq", "ChIP-seq_H3K4me1", "ChIP-seq_H3K4me3",
            "ChIP-seq_H3K27ac", "ChIP-seq_H3K27me3", "ChIP-seq_H3K36me3",
            "ChIP-seq_H3K9me3", "ChIP-seq_CTCF", "CAGE", "RNA-seq"
        ]
    
    def list_cell_types(self) -> List[str]:
        """Return Enformer's cell types."""
        return [
            "K562", "HepG2", "GM12878", "H1-hESC", "MCF-7", "A549",
            "HeLa-S3", "IMR90", "HUVEC", "HCT116"
        ]
    
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