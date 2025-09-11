"""Enformer oracle implementation with environment support."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import os
import logging
from copy import copy

from ..core.base import OracleBase
from ..core.result import OraclePrediction, OraclePrediction, OraclePredictionTrack
from ..core.track import Track
from ..core.new_interval import Interval, GenomeRef, Sequence 
from ..core.exceptions import ModelNotLoadedError
from ..utils.sequence import extract_sequence_with_padding

logger = logging.getLogger(__name__)

class EnformerOracle(OracleBase):
    """Enformer oracle with automatic environment management."""
    
    def __init__(self, 
                 use_environment: bool = True, 
                 reference_fasta: Optional[str] = None,
                 model_load_timeout: Optional[int] = 600,
                 predict_timeout: Optional[int] = 300,
                 device: Optional[str] = None):
        """
        Initialize Enformer oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
            reference_fasta: Path to reference FASTA file (e.g., hg38.fa)
            model_load_timeout: Timeout for model loading in seconds (default: 600s/10min)
                               Set to None to disable timeout
            predict_timeout: Timeout for predictions in seconds (default: 300s/5min)
                            Set to None to disable timeout
            device: Device to use for computation. Options:
                   - None: Auto-detect (GPU if available, else CPU)
                   - 'cpu': Force CPU usage
                   - 'cuda' or 'gpu': Use default GPU
                   - 'cuda:0', 'cuda:1', etc.: Use specific GPU
        """
        # Set the oracle name BEFORE calling super().__init__
        self.oracle_name = 'enformer'
        
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment, 
                         model_load_timeout=model_load_timeout,
                         predict_timeout=predict_timeout,
                         device=device)
        
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

# Configure device
device = {repr(self.device)}
if device:
    if device == 'cpu':
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Forcing CPU usage")
    elif device.startswith('cuda:'):
        # Use specific GPU
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Using GPU {{gpu_id}}")
    elif device in ['cuda', 'gpu']:
        # Use default GPU (don't change CUDA_VISIBLE_DEVICES)
        print("Using default GPU")
else:
    # Auto-detect - TensorFlow will use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Auto-detected {{len(gpus)}} GPU(s), using first available")
    else:
        print("No GPU detected, using CPU")

# Load the model
enformer = hub.load({repr(weights)})
# Get the actual model from the enformer object
model = enformer.model

# Get device info
if device == 'cpu' or not tf.config.list_physical_devices('GPU'):
    actual_device = 'CPU'
else:
    actual_device = f'GPU ({{len(tf.config.list_physical_devices("GPU"))}} available)'

# Get model info (we can't pickle the model itself)
result = {{
    'loaded': True,
    'model_class': str(type(model)),
    'has_predict': hasattr(model, 'predict_on_batch'),
    'description': 'Enformer model loaded successfully',
    'device': actual_device
}}
"""
            
            # Run loading in environment
            # Use the instance timeout setting (can be customized or disabled)
            model_info = self.run_code_in_environment(load_code, timeout=self.model_load_timeout)
            
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
            
            # Configure device
            if self.device:
                if self.device == 'cpu':
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    logger.info("Forcing CPU usage")
                elif self.device.startswith('cuda:'):
                    gpu_id = self.device.split(':')[1]
                    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                    logger.info(f"Using GPU {gpu_id}")
                elif self.device in ['cuda', 'gpu']:
                    logger.info("Using default GPU")
            else:
                # Auto-detect
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Auto-detected {len(gpus)} GPU(s)")
                else:
                    logger.info("No GPU detected, using CPU")
            
            os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
            enformer = hub.load(weights)
            self._enformer_model = enformer.model
            self.model = self._enformer_model
            self._load_track_metadata()
            self.loaded = True
            logger.info("Enformer model loaded successfully!")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Enformer model: {str(e)}")
    
    def _predict(self, seq: str | Tuple[str, int, int] | Interval, assay_ids: List[str] | None = None) -> OraclePrediction:
        """Run prediction in the appropriate environment.
        
        Args:
            seq: Either a DNA sequence string or a tuple of (chrom, start, end)
            assay_ids: List of assay identifiers
        """

        if assay_ids is None:
            assay_ids =  self.get_all_assay_ids()

        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinate input")
            chrom, start, end = seq
            query_interval = Interval.make(GenomeRef(chrom=chrom, 
                                                     start=start, 
                                                     end=end, 
                                                     fasta=self.reference_fasta))
        elif isinstance(seq, str):
            query_interval = Interval.make(Sequence(sequence=seq))
        elif isinstance(seq, Interval):
            query_interval = seq
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")

        input_interval = query_interval.extend(self.sequence_length)
        prediction_interval = query_interval.extend(self.output_size)
        
        full_seq = input_interval.sequence

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
import os

# Configure device
device = {repr(self.device)}
if device:
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device.startswith('cuda:'):
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# Read sequence from file
with open({repr(seq_path)}, 'r') as f:
    seq = f.read().strip()

# Load model (cached in TFHub)
# Enformer model has a specific structure - we need to get the model attribute
enformer = hub.load({repr(self.default_model_path)})
model = enformer.model

# Prepare sequence
#if len(seq) != 393216:
#    # Pad or trim sequence
#    if len(seq) < 393216:
#        pad_needed = 393216 - len(seq)
#        pad_left = pad_needed // 2
#        pad_right = pad_needed - pad_left
#        seq = 'N' * pad_left + seq + 'N' * pad_right
#    else:
#        trim_needed = len(seq) - 393216
#        trim_left = trim_needed // 2
#        trim_right = trim_needed - trim_left
#        seq = seq[trim_left:len(seq)-trim_right]

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
                # Use the instance timeout setting (can be customized or disabled)
                predictions_list = self.run_code_in_environment(predict_code, timeout=self.predict_timeout)
                
                predictions =  np.array(predictions_list)
                
            finally:
                # Clean up sequence file
                import os
                if os.path.exists(seq_path):
                    os.unlink(seq_path)
        else:
            # Use direct prediction
            predictions = self._predict_direct(seq, assay_ids)
        
        # for now we have all predictions

        from .enformer_metadata import get_metadata
        metadata = get_metadata()

        final_prediction = OraclePrediction()
        for ind, assay_id in enumerate(assay_ids):

            track_id = metadata.get_track_by_identifier(assay_id)
            info = metadata.get_track_info(track_id)

            types_info = metadata.parse_description(info['description']) 

            track = OraclePredictionTrack(
                source_model="enformer",
                assay_id=assay_id, 
                assay_type=types_info['assay_type'],
                cell_type=types_info['cell_type'],
                query_interval=query_interval,
                prediction_interval=prediction_interval,
                input_interval=input_interval,
                resolution=self.bin_size,
                values=predictions[:, ind],
                metadata=info,
                preferred_aggregation='mean',
                preferred_deconvolution='repeat',
                preferred_scoring_strategy='mean'
            )
            final_prediction.add(assay_id, track)

        return final_prediction

    def _predict_direct(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Direct prediction in current environment."""
        import tensorflow as tf
        
        # One-hot encode
        one_hot = self._one_hot_encode(seq)
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
    
    def _prepare_sequence(self, seq: str) -> Tuple[str, dict[str, int]]:
        """Prepare sequence to correct length."""
        seq = seq.upper()
        
        if len(seq) < self.sequence_length:
            pad_needed = self.sequence_length - len(seq)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
            meta = dict(leftN=pad_left, rightN=pad_right, start_change=0, end_change=0)

        elif len(seq) > self.sequence_length:
            trim_needed = len(seq) - self.sequence_length
            trim_left = trim_needed // 2
            trim_right = trim_needed - trim_left
            seq = seq[trim_left:len(seq)-trim_right]
            meta = dict(leftN=0, rightN=0, start_change=trim_left, end_change=-trim_right)
        
        return seq, meta
    
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

    def get_all_assay_ids(self) -> list[str]:
        from .enformer_metadata import get_metadata
        metadata = get_metadata()
        
        return list(metadata._track_index_map.keys())

    @property
    def output_size(self) -> int:
        return self.target_length * self.bin_size

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