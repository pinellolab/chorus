"""Enformer oracle implementation with environment support."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import json
import os
from pathlib import Path
import logging

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidSequenceError

logger = logging.getLogger(__name__)


class EnformerOracleEnv(OracleBase):
    """Enformer oracle with automatic environment management."""
    
    def __init__(self, use_environment: bool = True):
        """
        Initialize Enformer oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
        """
        # Set oracle name before calling super().__init__
        self.oracle_name = 'enformer'
        
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
    
    def load_pretrained_model(self, weights: str = None) -> None:
        """Load Enformer model in the appropriate environment."""
        if weights is None:
            weights = self.default_model_path
        
        logger.info(f"Loading Enformer model from {weights}...")
        
        if self.use_environment:
            # Define loading function to run in environment
            def _load_in_env(weights_path):
                import tensorflow as tf
                import tensorflow_hub as hub
                import os
                
                # Set TFHub progress tracking
                os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
                
                # Load the model
                model = hub.load(weights_path).model
                
                # Return model info (we can't pickle the model itself)
                return {
                    'loaded': True,
                    'input_shape': model.inputs[0].shape.as_list(),
                    'output_keys': list(model.output_shapes.keys()) if hasattr(model, 'output_shapes') else ['human', 'mouse']
                }
            
            # Run loading in environment
            model_info = self.run_in_environment(_load_in_env, weights)
            
            if model_info['loaded']:
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
            self._enformer_model = hub.load(weights).model
            self.model = self._enformer_model
            self._load_track_metadata()
            self.loaded = True
            logger.info("Enformer model loaded successfully!")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Enformer model: {str(e)}")
    
    def _predict(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Run prediction in the appropriate environment."""
        if self.use_environment:
            # Define prediction function to run in environment
            def _predict_in_env(seq, assay_ids, model_path):
                import tensorflow as tf
                import tensorflow_hub as hub
                import numpy as np
                
                # Load model (cached in TFHub)
                model = hub.load(model_path).model
                
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
                mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                one_hot = np.zeros((len(seq), 4), dtype=np.float32)
                
                for i, base in enumerate(seq.upper()):
                    if base in mapping:
                        one_hot[i, mapping[base]] = 1.0
                
                # Add batch dimension
                one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
                
                # Run prediction
                predictions = model.predict_on_batch(one_hot_batch)
                human_predictions = predictions['human'][0].numpy()
                
                # For demo, return random subset of tracks
                # In real implementation, would map assay_ids to track indices
                n_tracks = len(assay_ids)
                selected_indices = np.random.choice(human_predictions.shape[1], n_tracks, replace=False)
                
                return human_predictions[:, selected_indices].tolist()
            
            # Run prediction in environment
            predictions_list = self.run_in_environment(
                _predict_in_env, 
                seq, 
                assay_ids, 
                self.default_model_path
            )
            
            return np.array(predictions_list)
        else:
            # Use direct prediction
            return self._predict_direct(seq, assay_ids)
    
    def _predict_direct(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Direct prediction in current environment."""
        import tensorflow as tf
        
        # Prepare sequence
        if len(seq) != self.sequence_length:
            seq = self._prepare_sequence(seq)
        
        # One-hot encode
        one_hot = self._one_hot_encode(seq)
        one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
        
        # Run prediction
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
        """Map assay IDs to track indices."""
        # Simplified version - in real implementation would use proper mapping
        indices = []
        for i, assay_id in enumerate(assay_ids):
            # Just return sequential indices for demo
            indices.append(i)
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