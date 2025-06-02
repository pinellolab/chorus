"""Enformer oracle implementation."""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import json
import os
from tqdm import tqdm

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidSequenceError


class EnformerOracle(OracleBase):
    """Enformer oracle implementation for genomic sequence prediction."""
    
    def __init__(self):
        super().__init__()
        self.target_length = 896  # Enformer's output length in bins
        self.bin_size = 128  # Base pairs per bin
        self.sequence_length = 393216  # Enformer's receptive field (196608 * 2)
        self.center_length = 196608  # Central prediction window
        
        # Model components
        self._enformer_model = None
        self._track_dict = None
        
        # Default model path
        self.default_model_path = "https://tfhub.dev/deepmind/enformer/1"
        
    def load_pretrained_model(self, weights: str = None) -> None:
        """
        Load Enformer model.
        
        Args:
            weights: Path to model weights or TFHub URL. 
                    If None, uses default Enformer model from TFHub.
        """
        if weights is None:
            weights = self.default_model_path
        
        print(f"Loading Enformer model from {weights}...")
        
        # Set TFHub progress tracking
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
        
        try:
            # Load the model
            self._enformer_model = hub.load(weights).model
            self.model = self._enformer_model  # Set base class attribute
            
            # Load track metadata
            self._load_track_metadata()
            
            self.loaded = True
            print("Enformer model loaded successfully!")
            
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Enformer model: {str(e)}")
    
    def _load_track_metadata(self):
        """Load track metadata for Enformer predictions."""
        # This would ideally load from a JSON file with track descriptions
        # For now, we'll create a basic structure
        self._track_dict = self._get_default_track_dict()
        
        # Extract assay types and cell types
        self._assay_types = list(set(track['assay'] for track in self._track_dict))
        self._cell_types = list(set(track['cell_type'] for track in self._track_dict 
                                    if track['cell_type'] != 'unknown'))
    
    def list_assay_types(self) -> List[str]:
        """Return Enformer's assay types."""
        if not self.loaded:
            # Return known assay types even if model not loaded
            return [
                "DNase", "ATAC-seq", "ChIP-seq_H3K4me1", "ChIP-seq_H3K4me3",
                "ChIP-seq_H3K27ac", "ChIP-seq_H3K27me3", "ChIP-seq_H3K36me3",
                "ChIP-seq_H3K9me3", "ChIP-seq_CTCF", "CAGE", "RNA-seq"
            ]
        return self._assay_types
    
    def list_cell_types(self) -> List[str]:
        """Return Enformer's cell types."""
        if not self.loaded:
            # Return common cell types
            return [
                "K562", "HepG2", "GM12878", "H1-hESC", "MCF-7", "A549",
                "HeLa-S3", "IMR90", "HUVEC", "HCT116"
            ]
        return self._cell_types
    
    def _predict(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """
        Run Enformer prediction.
        
        Args:
            seq: DNA sequence (must be exactly 393,216 bp)
            assay_ids: List of assay identifiers
            
        Returns:
            Predictions array of shape (896, len(assay_ids))
        """
        # Validate sequence length
        if len(seq) != self.sequence_length:
            # Pad or trim sequence to correct length
            seq = self._prepare_sequence(seq)
        
        # One-hot encode sequence
        one_hot = self._one_hot_encode(seq)
        
        # Add batch dimension
        one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
        
        # Run prediction
        predictions = self._enformer_model.predict_on_batch(one_hot_batch)
        
        # Extract human predictions (Enformer outputs both human and mouse)
        human_predictions = predictions['human'][0]  # Shape: (896, 5313)
        
        # Get indices for requested assays
        assay_indices = self._get_assay_indices(assay_ids)
        
        # Extract specific assays
        selected_predictions = human_predictions[:, assay_indices].numpy()
        
        return selected_predictions
    
    def _prepare_sequence(self, seq: str) -> str:
        """Prepare sequence to correct length by padding or trimming."""
        seq = seq.upper()
        
        if len(seq) < self.sequence_length:
            # Pad with N's
            pad_needed = self.sequence_length - len(seq)
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            seq = 'N' * pad_left + seq + 'N' * pad_right
        elif len(seq) > self.sequence_length:
            # Trim from ends
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
            # N's remain as all zeros
        
        return one_hot
    
    def _get_assay_indices(self, assay_ids: List[str]) -> List[int]:
        """Get track indices for requested assays."""
        indices = []
        
        for assay_id in assay_ids:
            if ':' in assay_id:
                # Format: "assay:cell_type"
                assay, cell_type = assay_id.split(':', 1)
                matching_tracks = [
                    i for i, track in enumerate(self._track_dict)
                    if track['assay'] == assay and track['cell_type'] == cell_type
                ]
            else:
                # Just assay type - return first matching track
                matching_tracks = [
                    i for i, track in enumerate(self._track_dict)
                    if track['assay'] == assay_id
                ]
            
            if matching_tracks:
                indices.append(matching_tracks[0])
            else:
                # Try to find partial matches
                matching_tracks = [
                    i for i, track in enumerate(self._track_dict)
                    if assay_id.lower() in track['assay'].lower() or
                       assay_id.lower() in track['cell_type'].lower()
                ]
                if matching_tracks:
                    indices.append(matching_tracks[0])
        
        return indices
    
    def fine_tune(
        self,
        tracks: List[Track],
        track_names: List[str],
        learning_rate: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 1,
        **kwargs
    ) -> None:
        """
        Fine-tune Enformer on new tracks.
        
        Args:
            tracks: List of Track objects with training data
            track_names: Names for the tracks
            learning_rate: Learning rate for fine-tuning
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # This is a placeholder for fine-tuning implementation
        # In practice, this would require:
        # 1. Preparing training data from tracks
        # 2. Setting up optimizer and loss function
        # 3. Running training loop with gradient updates
        
        raise NotImplementedError("Fine-tuning is not yet implemented for Enformer")
    
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths accepted by the model."""
        # Enformer can handle various lengths through padding
        return (1000, self.sequence_length)
    
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size
    
    def _get_default_track_dict(self) -> List[Dict]:
        """Get default track dictionary for Enformer."""
        # This is a simplified version - in practice, load from JSON
        tracks = []
        
        # Common assay types and cell types
        assay_types = [
            "DNase", "ATAC-seq", "CAGE", "RNA-seq",
            "ChIP-seq_H3K4me1", "ChIP-seq_H3K4me3", "ChIP-seq_H3K27ac",
            "ChIP-seq_H3K27me3", "ChIP-seq_H3K36me3", "ChIP-seq_H3K9me3",
            "ChIP-seq_CTCF"
        ]
        
        cell_types = ["K562", "HepG2", "GM12878", "H1-hESC", "MCF-7"]
        
        # Create track entries
        track_id = 0
        for assay in assay_types:
            for cell_type in cell_types:
                tracks.append({
                    'id': track_id,
                    'assay': assay,
                    'cell_type': cell_type,
                    'name': f"{assay}_{cell_type}",
                    'description': f"{assay} in {cell_type} cells"
                })
                track_id += 1
        
        return tracks
    
    def predict_from_bed_file(
        self,
        bed_file: str,
        assay_ids: List[str],
        genome: str = "hg38.fa",
        output_dir: str = "./predictions",
        batch_predict: bool = True
    ) -> Dict[str, Track]:
        """
        Predict for multiple regions from a BED file.
        
        Args:
            bed_file: Path to BED file with regions
            assay_ids: List of assay identifiers  
            genome: Path to reference genome
            output_dir: Directory to save predictions
            batch_predict: Whether to batch predictions
            
        Returns:
            Dictionary mapping region names to Track objects
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load regions
        regions = pd.read_csv(bed_file, sep='\t', header=None,
                             names=['chrom', 'start', 'end', 'name'])
        
        results = {}
        
        # Process each region
        for _, region in tqdm(regions.iterrows(), total=len(regions), 
                             desc="Processing regions"):
            region_str = f"{region['chrom']}:{region['start']}-{region['end']}"
            
            # Get predictions for region
            pred_results = self.predict_region_replacement(
                genomic_region=region_str,
                seq="",  # Will extract from genome
                assay_ids=assay_ids,
                create_tracks=True,
                genome=genome
            )
            
            # Store tracks
            if 'track_objects' in pred_results:
                for track in pred_results['track_objects']:
                    track.name = f"{region['name']}_{track.assay_type}"
                    results[track.name] = track
                    
                    # Save to file
                    output_file = os.path.join(output_dir, f"{track.name}.bedgraph")
                    track.to_bedgraph(output_file)
        
        return results
    
    @tf.function
    def compute_contribution_scores(
        self,
        input_sequence: tf.Tensor,
        target_mask: tf.Tensor,
        output_head: str = "human"
    ) -> tf.Tensor:
        """
        Compute contribution scores using integrated gradients.
        
        Args:
            input_sequence: One-hot encoded sequence
            target_mask: Mask for target positions
            output_head: 'human' or 'mouse'
            
        Returns:
            Contribution scores
        """
        input_sequence = input_sequence[tf.newaxis]
        target_mask_mass = tf.reduce_sum(target_mask)
        
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = self._enformer_model(input_sequence)[output_head]
            masked_prediction = (
                tf.reduce_sum(target_mask[tf.newaxis] * prediction) / target_mask_mass
            )
        
        input_grad = tape.gradient(masked_prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        
        return tf.reduce_sum(input_grad, axis=-1)