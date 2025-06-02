"""Sei oracle implementation."""

from ..core.base import OracleBase
from ..core.track import Track
from typing import List, Tuple
import numpy as np


class SeiOracle(OracleBase):
    """Sei oracle implementation for sequence regulatory activities."""
    
    def __init__(self):
        super().__init__()
        # Sei-specific parameters
        self.sequence_length = 4096  # Sei input length
        self.n_targets = 21907  # Number of regulatory features
        self.bin_size = 1  # Sequence-level predictions
        
    def load_pretrained_model(self, weights: str) -> None:
        """Load Sei model weights."""
        # TODO: Implement Sei model loading
        raise NotImplementedError("Sei oracle implementation coming soon")
    
    def list_assay_types(self) -> List[str]:
        """Return Sei's assay types."""
        return [
            "Promoter", "Enhancer", "Transcription", "Insulator",
            "Repressed", "Open_chromatin"
        ]
    
    def list_cell_types(self) -> List[str]:
        """Return Sei's cell types."""
        # Sei is trained on many cell types
        return ["Multiple", "Cell-type-agnostic"]
    
    def _predict(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Run Sei prediction."""
        raise NotImplementedError("Sei prediction not yet implemented")
    
    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tune Sei on new tracks."""
        raise NotImplementedError("Sei fine-tuning not yet implemented")
    
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths."""
        return (100, self.sequence_length)
    
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size
    
    def get_sequence_class_scores(self, seq: str) -> Dict[str, float]:
        """
        Get sequence class scores for regulatory activities.
        
        Args:
            seq: DNA sequence
            
        Returns:
            Dictionary mapping class names to scores
        """
        # TODO: Implement when model is loaded
        raise NotImplementedError("Sequence classification not yet implemented")