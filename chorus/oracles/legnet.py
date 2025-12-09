"""LegNet oracle implementation."""

from typing import List, Tuple, Dict, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import os 
import json
import logging
from ..core.base import OracleBase
from ..core.exceptions import ModelNotLoadedError, InvalidAssayError
from ..core.globals import CHORUS_DOWNLOADS_DIR
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.interval import Interval, GenomeRef, Sequence

from .legnet_source.legnet_globals import LEGNET_WINDOW, LEGNET_DEFAULT_STEP, LEGNET_AVAILABLE_CELLTYPES
from .legnet_source.exceptions import LegNetError
from .legnet_source.agarwal_meta import LEFT_MPRA_FLANK, RIGHT_MPRA_FLANK


logger = logging.getLogger(__name__)

LEGNET_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "legnet"
LEGNET_MODELS_DIR.mkdir(exist_ok=True, parents=True)

class LegNetOracle(OracleBase):
    """LegNet oracle implementation for sequence regulatory activities."""

    def __init__(self, 
                 cell_type: str,
                 assay: str = 'LentiMPRA',
                 model_id: str = 'example',
                 step_size: int = LEGNET_DEFAULT_STEP,
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
        if cell_type not in LEGNET_AVAILABLE_CELLTYPES:
            raise LegNetError(f"Cell line {cell_type} not in available cell types: {LEGNET_AVAILABLE_CELLTYPES}")
        self.cell_type = cell_type
        self.assay = assay
        self.model_id = model_id
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment, 
                         model_load_timeout=model_load_timeout,
                         predict_timeout=predict_timeout,
                         device=device)
        if self.device is None:
            self.device = 'cpu'

        self.download_dir = LEGNET_MODELS_DIR

        self.sequence_length = LEGNET_WINDOW
        self.n_targets = 1  # Number of regulatory features
        
        self.bin_size = step_size
        self.model_dir = model_dir 
        self.average_reverse = average_reverse
        self.reference_fasta = reference_fasta
        self.batch_size = batch_size
        self.left_flank = left_flank
        self.right_flank = right_flank
        self._model = None # Predictor model

    def get_model_weights_dir(self) -> Path:
        if self.model_dir is not None:
            self.download_dir = Path(self.model_dir)
        
        path = self.download_dir / f"{self.assay}_{self.cell_type}"
        if not path.exists():
            self._download_legnet_model()
        return path

    def set_bin_size(self, bin_size: int):
        self.bin_size = bin_size

    def get_model_weights_path(self) -> Path:
        path = self.get_model_weights_dir() / self.model_id / 'weights.ckpt'
        return path

    def get_training_config_path(self) -> Path:
        path = self.get_model_weights_dir() / 'config.json'
        return path

    def get_model_dir_path(self) -> Path:
        path = Path(__file__).parent / "legnet_source"
        return path

    def get_templates_dir(self) -> Path:
        path = self.get_model_dir_path() / "templates"
        return path
    
    def get_load_template(self):
        d = self.get_templates_dir()
        path = d / 'load_template.py'
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
    
    def get_predict_template(self):
        d = self.get_templates_dir()
        path = d / 'predict_template.py'
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
    
    def load_pretrained_model(self, weights: str | None = None) -> None:
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
            'model_weights': str(self.get_model_weights_path()),
            'cell_type': self.cell_type,
            'assay': self.assay,
            'config_path': str(self.get_training_config_path()),
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
            from .legnet_source.model_usage import load_model

            model = load_model(self.get_training_config_path(), self.get_model_weights_path())
            device = torch.device(self.device)
            model.to(device)
            model.eval()
            self._model = model # Predictor model

        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load LegNet model: {str(e)}")
    
    def list_assay_types(self) -> List[str]:
        """Return LegNet's assay types."""
        return ["LentiMPRA"]

    def list_cell_types(self) -> List[str]:
        """Return LegNet's cell types."""       
        return [self.cell_type]
 
    def _validate_loaded(self):
        """Check if model is loaded."""
        if not self.loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_pretrained_model first.")
    
    def _validate_assay_ids(self, assay_ids: List[str] | None):
        if assay_ids is None or (len(assay_ids) == 1 and assay_ids[0] == self._cell_type):
            return 
        raise InvalidAssayError(f"Instantiated LegNet oracle can only predict for assay {self.cell_type}")

    def _refine_total_length(self, total_length: int) -> int:
        div, mod = divmod(total_length, self.bin_size)
        total_length = div * self.bin_size + self.bin_size * (mod > 0)
        return total_length

    def _predict(self, seq: str | Tuple[str, int, int] | Interval, assay_ids: List[str] = None) -> OraclePrediction:
        """Run ChromBPNet prediction in the appropriate environment.
        
        Args:
            seq: Either a DNA sequence string or a tuple of (chrom, start, end)
            assay_ids: List of assay identifiers. In case of chrombnet this parameter is ignored.
        """

        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinates.")
            chrom, start, end = seq
            query_interval = Interval.make(GenomeRef(
                chrom=chrom,
                start=start,
                end=end,
                fasta=self.reference_fasta
            ))
        elif isinstance(seq, str):
            query_interval = Interval.make(Sequence(sequence=seq))
        elif isinstance(seq, Interval):
            query_interval = seq
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")

        input_interval = query_interval.extend(self.sequence_length)
        prediction_interval = query_interval.extend(self.sequence_length)

        full_seq = input_interval.sequence
        
        if self.use_environment:
            preds = self._predict_in_environment(
                seq=full_seq, 
                reverse_aug=self.average_reverse)
            
        else:
            preds = self._predict_direct(
                seq=full_seq, 
                reverse_aug=self.average_reverse)

        final_prediction = OraclePrediction()

        # Create a Prediction Object
        track = OraclePredictionTrack.create(
            source_model="legnet",
            assay_id=None, 
            track_id=None,
            assay_type=self.assay,
            cell_type=self.cell_type,
            query_interval=query_interval,
            prediction_interval=prediction_interval,
            input_interval=input_interval,
            resolution=self.bin_size,
            values=preds,
            metadata=None,
            preferred_aggregation='mean',
            preferred_interpolation='linear_divided',
            preferred_scoring_strategy='mean'
        )
        final_prediction.add(f"{self.assay}:{self.cell_type}", track)
        
        return final_prediction
        
    
    def _predict_in_environment(self,
                                seq: str,
                                reverse_aug: bool = True) -> np.ndarray:
 
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'model_weights': str(self.get_model_weights_path()),
            'config_path': str(self.get_training_config_path()),
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
        return predictions
        
        
    def _predict_direct(self,
                        seq: str,
                        reverse_aug: bool = True) -> np.ndarray:
        """Direct prediction in current environment."""

        if self._model is None:
            raise ModelNotLoadedError()
        from .legnet_source.model_usage import predict_bigseq
        preds, _ = predict_bigseq(self._model, 
                                        seq=seq, 
                                        reverse_aug=reverse_aug,
                                        window_size=self.sequence_length,
                                        step=self.bin_size,
                                        left_flank=self.left_flank,
                                        right_flank=self.right_flank,
                                        batch_size=self.batch_size)

        return preds

    def fine_tune(self, tracks: List[OraclePredictionTrack], track_names: List[str], **kwargs) -> None:
        """Fine-tune LegNet on new tracks."""
        # TODO: for now we decided not to implement this functionality
        raise NotImplementedError("LegNet fine-tuning not yet implemented")
    
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

    def get_zenodo_link(self) -> str:
        return "https://zenodo.org/records/17863550/files/chorus_small_legnet.zip"
    
    def _download_legnet_model(self): 
        import zipfile
        import urllib.request
        import shutil
       
        # Create download link
        download_link = self.get_zenodo_link()
        download_path = self.download_dir

        logger.info(f"Dowloading LegNet into {download_path}...")

        download_file_path = os.path.join(
            download_path, 
            os.path.basename(download_link)
        )

        if not Path(download_file_path).exists():
            urllib.request.urlretrieve(download_link, filename=download_file_path)    
            logger.info("Dowload completed!")
        
        # Now extract the file in the same download folder
        extract_folder = download_path
        with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        extracted_folder = download_file_path.replace('.zip', '')
        for file in Path(extracted_folder).glob('*'):
            dest = download_path / file.name
            if Path(dest).exists():
                shutil.rmtree(dest)
            shutil.move(file, dest)
        shutil.rmtree(extracted_folder)
        os.remove(download_file_path)
        logger.info("Extraction completed!")