"""Borzoi oracle implementation"""

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError
from typing import List, Tuple, Union, Optional, Dict
import os
import logging
import json
import numpy as np 
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict, Any

from ..core.base import OracleBase
from ..core.result import OraclePrediction, OraclePrediction, OraclePredictionTrack
from ..core.track import Track
from ..core.interval import Interval, GenomeRef, Sequence 
from ..core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class BorzoiOracle(OracleBase):
    """Borzoi oracle with automatic environment management."""
    
    def __init__(self, 
                 use_environment: bool = True, 
                 reference_fasta: Optional[str] = None,
                 model_load_timeout: Optional[int] = 600,
                 predict_timeout: Optional[int] = 300,
                 device: Optional[str] = None,
                 fold: int = 0):
        """
        Initialize Borzoi oracle.
        
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
        self.oracle_name = 'borzoi'
        
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment, 
                         model_load_timeout=model_load_timeout,
                         predict_timeout=predict_timeout,
                         device=device)
        
        # Borzoi specific parameters
        self.sequence_length = 524288  # 524kb input
        self.target_length = 6144      # Will be updated from actual model
        self.bin_size = 32             # 32bp bins
        self.fold = fold
       
        # Model components
        self._borzoi_model = None
        self._track_dict = None
        
        
        # Reference genome
        self.reference_fasta = reference_fasta
        self.model_dir = None
    
    def get_model_weights_path(self) -> str:
        # no need for that as transformers will automatically handle weights 
        return ""
    
    def load_pretrained_model(self, weights: str = None) -> None:
        logger.info(f"Loading Borzoi model fold {self.fold}")
        
        if self.use_environment:
            self._load_in_environment(weights)
        else:
            # Load directly if not using environment
            self._load_direct(weights)


    def get_model_dir_path(self) -> str:
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "borzoi_source")
        return self.model_dir

    def get_templates_dir(self) -> str:
        return os.path.join(self.get_model_dir_path(), "templates")

    def get_load_template(self) -> str:
        d = self.get_templates_dir()
        path = os.path.join(d, 'load_template.py')
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    def get_predict_template(self) -> str:
        d = self.get_templates_dir()
        path = os.path.join(d, 'predict_template.py')
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
        
    def _get_metadata_path(self) -> str:
        return os.path.join(self.get_model_dir_path(), "borzoi_human_targets.txt")

    def _load_direct(self, weights: str):
        """Load model directly in current environment."""
        try:
            from borzoi_pytorch import Borzoi
            import torch
            flashzoi = Borzoi.from_pretrained(f'johahi/borzoi-replicate-{self.fold}')
            
            if self.device is None:
                if torch.cuda.is_available():
                    device = 'cuda:0'
                else:
                    device = 'cpu'
            device = torch.device(device)

            flashzoi.to(device)

            self._borzoi_model = flashzoi
            self.model = flashzoi
            self._load_track_metadata()
            self.loaded = True
            logger.info("Borzoi model loaded successfully!")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Borzoi model: {str(e)}")

    def _load_in_environment(self, weights: str) -> None:
        args = {
            'device': self.device,
            'fold': self.fold,
        }

        # Save arguments to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg = self.get_load_template()
            template = template.replace(arg, arg_file.name)
            model_info = self.run_code_in_environment(template, timeout=self.model_load_timeout)
            
            if model_info and model_info['loaded']:
                self.loaded = True
                self._model_info = model_info
                logger.info("Borzoi model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError("Failed to load Borzoi model in environment")

    
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
            predictions = self._predict_in_environment(full_seq, assay_ids)
        else:
            # Use direct prediction
            predictions = self._predict_direct(full_seq, assay_ids)
        
        # for now we have all predictions

        from .borzoi_source.borzoi_metadata import get_metadata
        metadata = get_metadata()

        final_prediction = OraclePrediction()
        for ind, assay_id in enumerate(assay_ids):
            track_id = metadata.get_track_by_identifier(assay_id)
            info = metadata.get_track_info(track_id)

            types_info = metadata.parse_description(info['description']) 

            track = OraclePredictionTrack.create(
                source_model="borzoi",
                assay_id=assay_id, 
                track_id=track_id,
                assay_type=types_info['assay_type'],
                cell_type=types_info['cell_type'],
                query_interval=query_interval,
                prediction_interval=prediction_interval,
                input_interval=input_interval,
                resolution=self.bin_size,
                values=predictions[:, ind],
                metadata=info,
                preferred_aggregation='sum',
                preferred_interpolation='linear_divided',
                preferred_scoring_strategy='mean'
            )
            final_prediction.add(assay_id, track)

        return final_prediction
    
    def _predict_in_environment(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        args = {
            'device': self.device,
            'fold': self.fold,
            'length': self.sequence_length,
            'sequence': seq,
            'assay_ids': assay_ids,
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as arg_file:
            args['metainfo_path'] = self._get_metadata_path()
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg1 = self.get_predict_template()
            template = template.replace(arg1, arg_file.name)
            predictions_list = self.run_code_in_environment(template, timeout=self.predict_timeout)    
            predictions = np.array(predictions_list)
        return predictions

    def _predict_direct(self, seq: str, assay_ids: List[str]) -> np.ndarray:
        """Direct prediction in current environment."""
        from .borzoi_source.helpers import perform_prediction  
        import torch

        if self.device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'

        device = torch.device(device)

        pred = perform_prediction(self.model, seq, self.sequence_length, device)
        
        # Get indices for requested assays
        assay_indices = self._get_assay_indices(assay_ids)
        
        return pred[:, assay_indices].numpy()
    
    def list_assay_types(self) -> List[str]:
        """Return all unique assay types from Borzoi metadata."""
        from .borzoi_source.borzoi_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_assay_types()
    
    def list_cell_types(self) -> List[str]:
        """Return all unique cell types from Borzoi metadata."""
        from .borzoi_source.borzoi_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_cell_types()
    
    def _get_assay_indices(self, assay_ids: List[str]) -> List[int]:
        """Map assay IDs to track indices using proper metadata."""
        from .borzoi_source.borzoi_metadata import get_metadata
        
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
        from .borzoi_source.borzoi_metadata import get_metadata
        metadata = get_metadata()
        
        if query:
            return metadata.search_tracks(query)
        else:
            return metadata.get_track_summary()

    def get_all_assay_ids(self) -> list[str]:
        from .borzoi_source.borzoi_metadata import get_metadata
        metadata = get_metadata()
        
        return list(metadata._track_index_map.keys())

    @property
    def output_size(self) -> int:
        return self.target_length * self.bin_size