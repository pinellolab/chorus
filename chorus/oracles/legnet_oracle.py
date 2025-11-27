"""LegNet oracle implementation."""

from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import os 
import json
import logging
from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidAssayError
from ..utils.sequence import extract_sequence_with_padding

from .legnet.legnet_globals import LEGNET_WINDOW, LEGNET_STEP, LEGNET_AVAILABLE_CELLTYPES
from .legnet.exceptions import LegNetError
from .legnet.metainfo import MetaInfoArray, MetaInfoDict
from .legnet.agarwal_meta import LEFT_MPRA_FLANK, RIGHT_MPRA_FLANK

logger = logging.getLogger(__name__)


class LegNetOracle(OracleBase):
    """LegNet oracle implementation for sequence regulatory activities."""
    
    def __init__(self, 
                 cell_line: str,
                 step_size: int = LEGNET_STEP,
                 sliding_predict: bool = False,
                 batch_size: int = 1,
                 left_flank: str = LEFT_MPRA_FLANK,
                 right_flank: str = RIGHT_MPRA_FLANK,
                 use_environment: bool = True, 
                 reference_fasta: str | None = None,
                 model_load_timeout: int | None = 600,
                 predict_timeout: int | None  = 300,
                 device: str | None = None,
                 average_reverse: bool = False, # In general, averaging predictions only slightly improves quality
                 model_dir: str | None = None):
        
        self.oracle_name = 'legnet'
        if cell_line not in LEGNET_AVAILABLE_CELLTYPES:
            raise LegNetError(f"Cell line {cell_line} not in available cell types: {LEGNET_AVAILABLE_CELLTYPES}")
        self.cell_line = cell_line
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment, 
                         model_load_timeout=model_load_timeout,
                         predict_timeout=predict_timeout,
                         device=device)
        if self.device is None:
            self.device = 'cpu'

        # Sei-specific parameters
        self.sequence_length = LEGNET_WINDOW # Sei input length
        self.n_targets = 1  # Number of regulatory features
        self.sliding_predict = sliding_predict
        
        self.bin_size = step_size if self.sliding_predict else self.sequence_length # Sequence-level predictions
        self.model_dir = model_dir 
        self.average_reverse = average_reverse
        self.reference_fasta = reference_fasta
        self.batch_size = batch_size
        self.left_flank = left_flank
        self.right_flank = right_flank
        self._model = None # Predictor model

    def get_model_dir_path(self):
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "legnet")
        return self.model_dir

    def get_model_weight_dir(self):
        d = self.get_model_dir_path()
        return  os.path.join(d, "models", self.cell_line)

    def get_model_weights_path(self):
        d = self.get_model_weight_dir()
        return os.path.join(d, "weights.ckpt")

    def get_training_config_path(self):
        d = self.get_model_weight_dir()
        return os.path.join(d, "config.json")

    def get_templates_dir(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "templates")
    
    def get_load_template(self):
        d = self.get_templates_dir()
        path = os.path.join(d, 'load_template.py')
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
    
    def get_predict_template(self):
        d = self.get_templates_dir()
        path = os.path.join(d, 'predict_template.py')
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
    
    def load_pretrained_model(self, weights: str | None) -> None:
        """Load LegNet model weights."""
        if weights is not None:
            self.model_dir = weights

        if self.use_environment:
            self._load_in_environment()
        else:
            self._load_direct()
    
    def _load_in_environment(self):
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'model_weights': self.get_model_weights_path(),
            'cell_line': self.cell_line,
            'config_path': self.get_training_config_path(),
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
                logger.info("LegNet model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError("Failed to load LegNet model in environment")
    
    def _load_direct(self):
        try:
            import torch 
            from .legnet.model_usage import load_model

            model = load_model(self.get_training_config_path(), self.get_model_weights_path())
            device = torch.device(self.device)
            model.to(device)
            model.eval()
            self._model = model # Predictor model

        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load LegNet model: {str(e)}")
    
    def list_assay_types(self) -> List[str]:
        """Return LegNet's assay types."""
        return ["MPRA"]

    def list_cell_types(self) -> List[str]:
        """Return LegNet's cell types."""       
        return [self.cell_line]
 
    def _validate_loaded(self):
        """Check if model is loaded."""
        if not self.loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_pretrained_model first.")
    
    def _validate_assay_ids(self, assay_ids: List[str] | None):
        if assay_ids is None or (len(assay_ids) == 1 and assay_ids[0] == self.cell_line):
            return 
        raise InvalidAssayError(f"Instantiated LegNet oracle can only predict for assay {self.cell_line}")

    def _refine_total_length(self, total_length: int) -> int:
        if not self.sliding_predict:
            return self.sequence_length

        div, mod = divmod(total_length, self.bin_size)
        total_length = div * self.bin_size + self.bin_size * (mod > 0)
        return total_length

    def predict(
        self,
        input_data: Union[str, Tuple[str, int, int]],
        assay_ids: list[str] | None = None,
        create_tracks: bool = False
    ) -> Dict[str, MetaInfoArray]:
        """
        Predict regulatory activity for a sequence or genomic region.
        
        Args:
            input_data: Either a DNA sequence string or a tuple of (chrom, start, end)
            create_tracks: Whether to create track files (not implemented yet)
            
        Returns:
            Dictionary mapping assay IDs to prediction arrays
        """
        
        # Validate inputs
        self._validate_loaded()
        
        # Get raw predictions
        predictions = self._predict(input_data, assay_ids=assay_ids)
        
        # Return as dictionary
        result = MetaInfoDict({self.cell_line: predictions[0]}, metainfo={'positions': predictions.metainfo['positions']})
        return result 
    
    def _predict(self,
                 seq: Union[str, Tuple[str, int, int]],
                 assay_ids: list[str] | None = None) -> MetaInfoArray:
        self._validate_assay_ids(assay_ids)
        
        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinate input")
            chrom, start, end = seq

            # Extract sequence with padding from reference
            total_length =  self._refine_total_length(end - start)
            full_seq = extract_sequence_with_padding(
                self.reference_fasta,
                chrom,
                start,
                end,
                total_length=total_length
            )

            center = (start + end) // 2 
            real_start = center - total_length // 2
        else:
            full_seq = seq        
            real_start = 0
        
        if self.use_environment:
            preds, offsets = self._predict_in_environment(
                seq=full_seq, 
                reverse_aug=self.average_reverse)
            
        else:
            preds, offsets = self._predict_direct(
                seq=full_seq, 
                reverse_aug=self.average_reverse)

        positions = real_start + offsets 
        preds = preds[None, :] # Add assay dimension 

        return MetaInfoArray(preds, metainfo={'positions': positions})
        
    
    def _predict_in_environment(self,
                                seq: str,
                                reverse_aug: bool = True) -> Tuple[np.ndarray, np.ndarray]:
 
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'model_weights': self.get_model_weights_path(),
            'config_path': self.get_training_config_path(),
            'seq': seq,
            'reverse_aug': reverse_aug,
            'batch_size': self.batch_size,
            'bin_size': self.bin_size,
            'left_flank': self.left_flank,
            'right_flank': self.right_flank,
        }

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg = self.get_predict_template()
            template = template.replace(arg, arg_file.name)
            model_predictions = self.run_code_in_environment(template, timeout=self.model_load_timeout)
            predictions = np.array(model_predictions['preds'], dtype=np.float32)
            offsets = np.array(model_predictions['offsets'], dtype=np.int64)
        return predictions, offsets
        
        
    def _predict_direct(self,
                        seq: str,
                        reverse_aug: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Direct prediction in current environment."""

        if self._model is None:
            raise ModelNotLoadedError()
        from .legnet.model_usage import predict_bigseq
        preds, offsets = predict_bigseq(self._model, 
                                        seq=seq, 
                                        reverse_aug=reverse_aug,
                                        window_size=self.sequence_length,
                                        step=self.bin_size,
                                        left_flank=self.left_flank,
                                        right_flank=self.right_flank,
                                        batch_size=self.batch_size)

        return preds, offsets 

    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tune Sei on new tracks."""
        # TODO: for now we decided not to implement this functionality
        raise NotImplementedError("Sei fine-tuning not yet implemented")
    
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths."""
        # LegNet can be used for sequences of any length, but we use the same window size for all sequences
        return (self.sequence_length, self.sequence_length) 
    
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size
    
    def get_status(self) -> Dict[str, Any] | None:
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

    def predict_region_replacement(
        self,
        genomic_region: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str] | None = None,
        create_tracks: bool = False,
        genome: str | None = None
    ) -> MetaInfoDict:
        if assay_ids is None:
            assay_ids = [self.cell_line]

        dt = super().predict_region_replacement(genomic_region, seq, assay_ids, create_tracks, genome)
        
        try:
            val = next(iter(dt['raw_predictions'].values()))
            metainfo = {'positions': val.metainfo['positions']}
        except StopIteration:
            metainfo = {}

        return MetaInfoDict(dt, metainfo=metainfo)

    def predict_region_insertion_at(
        self,
        genomic_position: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str] | None = None,
        create_tracks: bool = False,
        genome: str | None = None
    ) -> MetaInfoDict:
        if assay_ids is None:
            assay_ids = [self.cell_line]

        dt = super().predict_region_insertion_at(genomic_position=genomic_position, 
                                                   seq=seq, 
                                                   assay_ids=assay_ids, 
                                                   create_tracks=create_tracks, 
                                                   genome=genome)
        try:
            val = next(iter(dt['raw_predictions'].values()))
            metainfo = {'positions': val.metainfo['positions']}
        except StopIteration:
            metainfo = {}

        return MetaInfoDict(dt, metainfo=metainfo)
        

    def predict_variant_effect(
        self,
        genomic_region: Union[str, pd.DataFrame],
        variant_position: Union[str, pd.DataFrame],
        alleles: Union[List[str], pd.DataFrame],
        assay_ids: List[str] | None = None,
        create_tracks: bool = False,
        genome: str | None = None
    ) -> Dict:
        if assay_ids is None:
            assay_ids = [self.cell_line]

        return super().predict_variant_effect(genomic_region=genomic_region, 
                                              variant_position=variant_position,
                                              alleles=alleles,
                                              assay_ids=assay_ids, 
                                              create_tracks=create_tracks, 
                                              genome=genome)