"""ChromBPNet oracle implementation."""

from ..core.base import OracleBase
from ..core.track import Track
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.interval import Interval, GenomeRef, Sequence
from ..core.exceptions import ModelNotLoadedError
from ..core.globals import CHORUS_DOWNLOADS_DIR

from typing import List, Tuple, Optional, ClassVar
import numpy as np
import subprocess
import logging
import json
import os
import tarfile
from pathlib import Path


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ChromBPNetOracle(OracleBase):
    """ChromBPNet oracle implementation for TF binding and chromatin accessibility."""
    CHROMBPNET_MODELS_DICT: ClassVar[dict[str, dict[str, str]]] = {
        "ATAC": {
            "K562": "ENCFF984RAF",
            "HepG2": "ENCFF137WCM",
            "GM12878": "ENCFF142IOR",
            "IMR-90": "ENCFF113GSV"
        },
        "DNASE": {
            "HepG2": "ENCFF615AKY",
            "IMR-90": "ENCFF515HBV",
            "GM12878": "ENCFF673TIN",
            "K562": "ENCFF574YLK"
        },
        "CHIP": {}
    }

    def __init__(self,
                 use_environment: bool = True, 
                 reference_fasta: Optional[str] = None,
                 model_load_timeout: Optional[int] = 600,
                 predict_timeout: Optional[int] = 300,
                 device: Optional[str] = None):
    
        # ChromBPNet-specific parameters
        self.sequence_length = 2114  # ChromBPNet input length
        self.output_length = 1000  # Profile output length
        self.bin_size = 1  # Base-pair resolution

        # Set oracle name
        self.oracle_name = 'chrombpnet'

        super().__init__(use_environment=use_environment,
                       model_load_timeout=model_load_timeout,
                       predict_timeout=predict_timeout,
                       device=device)
        
        # Store Reference Genome
        self.reference_fasta = reference_fasta

        self.download_dir = CHORUS_DOWNLOADS_DIR / "chrombpnet"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = None # will be set when the model is downloaded
        self.model_dir = None # Where templates and utils are stored
        self.JASPAR_metadata = None
        self.assay = None
        self.cell_type = None
        self.fold = 0

    def get_encode_link(self, idx: str) -> str:
        return f"https://www.encodeproject.org/files/{idx}/@@download/{idx}.tar.gz"

    def get_model_weights_dir(self, assay: str, cell_type: str, tf: Optional[str] = None) -> Path:
        weights_dir = (
            f"{assay}_{cell_type}" if tf is None
            else f"{assay}_{cell_type}_{tf}"
        )
        path = self.download_dir / weights_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_weights_path(self, assay: str, cell_type: str, fold: int, tf: Optional[str] = None, model_type: str = 'chrombpnet') -> Path:
        if tf is None:  
            path = self.get_model_weights_dir(assay, cell_type) / 'models' / f"fold_{fold}" / model_type / 'chrombpnet'
        else:
            weights_name = os.path.basename(self.JASPAR_metadata.get_weights_by_cell_and_tf(tf, cell_type))
            path = self.get_model_weights_dir(assay, cell_type, tf) / weights_name
        return path
    
    def get_model_dir_path(self) -> str:
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "chrombpnet_source")
        return self.model_dir
    
    def get_templates_dir(self) -> str:
        return os.path.join(self.get_model_dir_path(), "templates")
    
    def get_load_template(self) -> str:
        d = self.get_templates_dir()
        path = os.path.join(d, "load_template.py")
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
        
    def get_predict_template(self) -> str:
        d = self.get_templates_dir()
        path = os.path.join(d, "predict_template.py")
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"
        
    def _get_JASPAR_path(self) -> str:
        return os.path.join(self.get_model_dir_path(), "chrombpnet_JASPAR_metadata.tsv")

    #from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
    @staticmethod
    def multinomial_nll(true_counts, logits):
        """Compute multinomial negative log likelihood"""
        import tensorflow as tf
        import tensorflow_probability as tfp

        counts_per_example = tf.reduce_sum(true_counts, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,logits=logits)

        return (-tf.reduce_sum(dist.log_prob(true_counts)) /
                tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))
    
    def _download_model_from_JASPAR(self):

        # Get link from metadata
        download_link = self.JASPAR_metadata.get_weights_by_cell_and_tf(self.tf, self.cell_type)
        download_path = self.get_model_weights_dir(self.assay, self.cell_type, self.tf)

        logger.info(f"Downloading BPNet into {download_path}...")

        download_file_path = os.path.join(
            download_path,
            os.path.basename(download_link)
        )

        if not os.path.exists(download_file_path):

            # Download from JASPAR
            result = subprocess.run(
                ["wget", "-P", download_path, download_link],
                text=True,
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Execution failed: {result.stderr}")
            
            logger.info("Download completed!")


    def _download_chrombpnet_model(self): 
        
        # Get model's ENCODE idx
        idx = self.CHROMBPNET_MODELS_DICT[self.assay][self.cell_type]

        # Create download link
        download_link = self.get_encode_link(idx)
        download_path = self.get_model_weights_dir(self.assay, self.cell_type)

        logger.info(f"Dowloading ChromBPNet into {download_path}...")

        download_file_path = os.path.join(
            download_path, 
            os.path.basename(download_link)
        )

        if not os.path.exists(download_file_path):
            
            # Download from ENCODE (tar file)
            result = subprocess.run(
                ["wget", "-P", download_path, download_link],
                text=True,
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Execution Failed: {result.stderr}")
            
            logger.info("Dowload completed!")
        


        # Now extract the file in the same download folder
        extract_folder = os.path.join(
            download_path,
            "models"
        )

        with tarfile.open(download_file_path, "r:gz") as tar:
            tar.extractall(path=extract_folder)

        for fold in range(5):
            # Now select model coming from fold 0 (ChromBPNet was trained with CV)
            models_dir = os.path.join(
                extract_folder,
                f"fold_{fold}"
            )
            tar_mappings = {
                f"model.bias_scaled.fold_{fold}.*.tar": 'bias_scaled',
                f"model.chrombpnet.fold_{fold}.*.tar": 'chrombpnet',
                f"model.chrombpnet_nobias.fold_{fold}.*.tar": 'chrombpnet_nobias'
            }
            for t_name, t_type in tar_mappings.items():
                t_pattern = os.path.join(models_dir, t_name)
                import glob
                t_path =glob.glob(t_pattern)[0] # one file for pattern
                t_out = os.path.join(models_dir, t_type)
                with tarfile.open(t_path, "r:") as tar:
                    tar.extractall(path=t_out)
                

    def load_pretrained_model(
            self,
            assay: Optional[str] = None,
            cell_type: Optional[str] = None,
            weights: Optional[str] = None,
            TF: Optional[str] = None,
            fold: int = 0,
            model_type: str = 'chrombpnet'
        ) -> None:
        """Load ChromBPNet model weights."""

        if assay is None or cell_type is None:
            raise ValueError("You must provide both assay and cell-type if weights are None.")
        
        if assay not in self.CHROMBPNET_MODELS_DICT:
            raise ValueError(f"ChromBPNet supports only the following assays: {list(self.CHROMBPNET_MODELS_DICT.keys())}")
        
        if assay != "CHIP":            
            # Check if the combination is valid
            if cell_type not in self.CHROMBPNET_MODELS_DICT[assay]:
                raise ValueError(f"ChromBPNet {assay} predictions can only be done on the following cell types: {list(self.CHROMBPNET_MODELS_DICT[assay].keys())}")

            if fold not in range(5):
                raise ValueError(f"ChromBPNet fold must be an integer between 0 and 4, got {fold}")
        else:
            if TF is None:
                 raise ValueError(f"You must provide TF for which you want ChIP-seq model.")
            
        # Store assay and celltype
        self.assay = assay
        self.cell_type = cell_type
        self.fold = fold
        self.tf = TF

        # Load JASPAR metadata if TF is set
        if TF is not None:
            from .chrombpnet_source.metadata import BPNetMetadata
            self.JASPAR_metadata = BPNetMetadata()

        if weights is None:
            # Check whether the user has provided assay and cell-type
            # Download weights and return path to them

            weights = self.get_model_weights_path(assay, cell_type, fold, TF, model_type)
            if not os.path.exists(weights):
                if assay != "CHIP":
                    self._download_chrombpnet_model()
                else:
                    self._download_model_from_JASPAR()
            if not os.path.exists(weights):
                raise FileNotFoundError(f"Weights file {weights} not found even after downloading from ENCODE")
            self.model_path = weights
        else:
            # Use directly the specified path
            self.model_path = weights            

        # Now load the model
        logger.info(f"Loading ChromBPNet model...")

        if self.use_environment:
            print('Loading in environment')
            self._load_in_environment(self.model_path)
        else:
            print('Loading directly')
            self._load_direct(self.model_path)
 

    def _load_in_environment(self, weights: str):
        args = {
            "device": self.device,
            "model_weights": str(weights),
            "is_CHIP": self.assay == "CHIP",
            "BPNet_dir": self.get_templates_dir()
        }

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg = self.get_load_template()
            template = template.replace(arg, arg_file.name)
            model_info = self.run_code_in_environment(template, timeout=self.model_load_timeout)

            if model_info and model_info["loaded"]:
                self.loaded = True
                self._model_info = model_info
                logger.info("ChromBPNet model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError("Failed to load ChromBPNet model in environment")
    
    def _load_direct(self, weights: str):
        """Load model directly in current environment"""
        try:
            import tensorflow as tf
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

            # Load the model using custom objects for loss function
            model = tf.keras.models.load_model(
                weights,
                compile=False,
                custom_objects={"multinomial_nll": self.multinomial_nll, "tf": tf}
            )
            self.model = model

            self.loaded = True
            logger.info("ChromBPNet model loaded successfully!")

        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load ChromBPNet model: {str(e)}")
    
    def list_assay_types(self) -> List[str]:
        """Return ChromBPNet's assay types."""
        return ["ATAC", "DNASE", "CHIP"]
    
    def list_cell_types(self) -> List[str]:
        """Return ChromBPNet's cell types."""
        return ["IMR-90", "GM12878", "HepG2", "K562"]
    
    def _transform_predictions_to_tracks(self, probabilities: np.array, counts: np.array, seq_len: int) -> np.array:

        # Allocate space for output tensor
        out = np.zeros(seq_len)

        # Predictions should represent probabilities and should be multiplied 
        # by the predicted log counts
        norm_prob = probabilities - np.mean(probabilities, axis=1, keepdims=True)
        softmax_probs = np.exp(norm_prob) / np.sum(np.exp(norm_prob), axis=1, keepdims=True)

        predictions = softmax_probs * (np.expand_dims(np.exp(counts)[:, 0], axis=1)) # (B, 1000)

        # Stack into 1D array
        predictions = predictions.reshape(-1, 1).squeeze()

        # Insert into final output
        start_insertion = (self.sequence_length - self.output_length) // 2 # ChromBPNet padding

        # Define insertion boundaries
        end_insertion_out = min(start_insertion + len(predictions), seq_len)
        end_insertion_pred = min(len(predictions), (end_insertion_out - start_insertion))
        
        out[start_insertion:end_insertion_out] = predictions[:end_insertion_pred]

        return out
    
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
        prediction_interval = query_interval.extend(self.output_size)

        full_seq = input_interval.sequence

        if self.use_environment:

            # Allocate space for output
            seq_len = max(len(full_seq), self.sequence_length)
            
            predictions = self._predict_in_environment(full_seq, assay_ids)

            # Extract track and counts
            probabilities, counts = predictions

            predictions_list = []
            if self.assay == "CHIP":
                # You have plus and minus strand predictions
                plus = self._transform_predictions_to_tracks(probabilities[..., 0], counts, seq_len)
                minus = self._transform_predictions_to_tracks(probabilities[..., 1], counts, seq_len)

                # Append to the list
                predictions_list.append(plus)
                predictions_list.append(minus)
            else:
                # You have only one prediction
                prediction = self._transform_predictions_to_tracks(probabilities, counts, seq_len)
                predictions_list.append(prediction)

        else:
            raise NotImplementedError("ChromBPNet direct prediction not yet implemented")
        
        final_prediction = OraclePrediction()

        # Create a Prediction Object
        strands = ["+", "-"]
        for i, pred in enumerate(predictions_list):
            track = OraclePredictionTrack.create(
                source_model="chrombpnet",
                assay_id=None, 
                track_id=None,
                assay_type=self.assay,
                cell_type=self.cell_type,
                query_interval=query_interval,
                prediction_interval=prediction_interval,
                input_interval=input_interval,
                resolution=self.bin_size,
                values=pred,
                metadata=None,
                preferred_aggregation='mean',
                preferred_interpolation='linear_divided',
                preferred_scoring_strategy='mean'
            )
            track_id = (
                f"{self.assay}:{self.cell_type}" if self.assay != "CHIP"
                else f"{self.assay}:{self.cell_type}:{self.tf}:{strands[i]}"
            )
            final_prediction.add(track_id, track)
        
        return final_prediction
    
    def _predict_in_environment(self, seq: str, assay_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        args = {
            "device": self.device,
            "model_weights": str(self.get_model_weights_path(self.assay, self.cell_type, self.fold, self.tf)),
            "sequence": seq,
            "assay_ids": assay_ids,
            "sequence_length": self.sequence_length,
            "output_length": self.output_length,
            "is_CHIP": self.assay == "CHIP",
            "BPNet_dir": self.get_templates_dir()
        }

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix="txt", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg1 = self.get_predict_template()
            template = template.replace(arg1, arg_file.name)
            predictions = self.run_code_in_environment(template, timeout=self.predict_timeout)
            probabilities, counts = predictions

        return probabilities, counts
    
    @property
    def output_size(self):
        return self.bin_size * self.output_length

    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tune ChromBPNet on new tracks."""
        raise NotImplementedError("ChromBPNet fine-tuning not yet implemented")
    
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths."""
        return (500, self.sequence_length)
    
    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size