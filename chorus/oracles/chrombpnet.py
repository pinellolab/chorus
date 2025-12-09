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
        }
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
        self.assay = None
        self.cell_type = None
        self.fold = 0

    def get_encode_link(self, idx: str) -> str:
        return f"https://www.encodeproject.org/files/{idx}/@@download/{idx}.tar.gz"

    def get_model_weights_dir(self, assay: str, cell_type: str) -> Path:
        path = self.download_dir / f"{assay}_{cell_type}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_weights_path(self, assay: str, cell_type: str, fold: int, model_type: str = 'chrombpnet') -> Path:
        path = self.get_model_weights_dir(assay, cell_type) / 'models' / f"fold_{fold}" / model_type / 'chrombpnet'
        return path

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
            fold: int = 0,
            model_type: str = 'chrombpnet'
        ) -> None:
        """Load ChromBPNet model weights."""

        if assay is None or cell_type is None:
            raise ValueError("You must provide both assay and cell-type if weights are None.")
            
        # Check whether the assay is valid
        if assay not in self.CHROMBPNET_MODELS_DICT:
            raise ValueError(f"ChromBPNet supports only the following assays: {list(self.CHROMBPNET_MODELS_DICT.keys())}")
        
        # Check if the combination is valid
        if cell_type not in self.CHROMBPNET_MODELS_DICT[assay]:
            raise ValueError(f"ChromBPNet {assay} predictions can only be done on the following cell types: {list(self.CHROMBPNET_MODELS_DICT[assay].keys())}")

        if fold not in range(5):
            raise ValueError(f"ChromBPNet fold must be an integer between 0 and 4, got {fold}")
            
        # Store assay and celltype
        self.assay = assay
        self.cell_type = cell_type
        self.fold = fold

        if weights is None:
            # Check whether the user has provided assay and cell-type
            # Download weights and return path to them

            weights = self.get_model_weights_path(assay, cell_type, fold, model_type)
            if not os.path.exists(weights):
                self._download_chrombpnet_model()
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
        load_code = f"""
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import tensorflow_probability as tfp

def multinomial_nll(true_counts, logits):
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,logits=logits)

    return (-tf.reduce_sum(dist.log_prob(true_counts)) /
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))

# Configure device
device = {repr(self.device)}
if device:
    if device=='cpu':
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

# Load model
model = tf.keras.models.load_model(
    '{weights}',
    compile=False,
    custom_objects={{"multinomial_nll": multinomial_nll, "tf": tf}}
)

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
    'description': 'ChromBPNet model loaded successfully',
    'device': actual_device
}}
"""

        # Run loading in environment
        model_info = self.run_code_in_environment(load_code, timeout=self.model_load_timeout)

        if model_info and model_info['loaded']:
            self.loaded = True
            self._model_info = model_info
            logger.info("ChromBPNet model loaded successfully in environment!")
        else:
            raise ModelNotLoadedError("Failed to load model in environment.")
    
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
        return ["ATAC", "DNASE"]
    
    def list_cell_types(self) -> List[str]:
        """Return ChromBPNet's cell types."""
        return ["IMR-90", "GM12878", "HepG2", "K562"]
    
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
            # Save sequence to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as seq_file:
                seq_file.write(full_seq)
                seq_path = seq_file.name

            # Allocate space for output
            seq_len = max(len(full_seq), self.sequence_length)
            out = np.zeros(seq_len)
            
            try:
                # Code to run in environment
                predict_code = f"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def multinomial_nll(true_counts, logits):
        counts_per_example = tf.reduce_sum(true_counts, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,logits=logits)

        return (-tf.reduce_sum(dist.log_prob(true_counts)) /
                tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))

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

# Load model
model = tf.keras.models.load_model(
    '{self.model_path}',
    compile=False,
    custom_objects={{"multinomial_nll": multinomial_nll, "tf": tf}}
)

# Mapping dict
MAPPING = {{'A': 0, 'C': 1, 'G': 2, 'T': 3}}

if len(seq) > {self.sequence_length}:
    num_windows_stride_one = (len(seq) - {self.sequence_length} + 1)
    num_windows = (num_windows_stride_one + {self.output_length} - 1) // {self.output_length} + 1

    # Define seq_len and flag
    seq_len = ({self.output_length} * (num_windows - 1)) + {self.sequence_length}
    trimmed = False
else:
    seq_len = len(seq)
    trimmed = True

# One hot encoding
one_hot = np.zeros((seq_len, 4), dtype=np.float32)
for i, base in enumerate(seq.upper()):
    if base in MAPPING:
        one_hot[i, MAPPING[base]] = 1.0

# Add batch dimension
if trimmed:
    one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
else:
    # Compute windows of 2114 with a stride of 1000 to extend the prediction
    new_shape = (num_windows, {self.sequence_length}, 4)
    stride_x, stride_y = one_hot.strides
    new_stride = (stride_x * {self.output_length}, stride_x, stride_y)

    one_hot_batch = np.lib.stride_tricks.as_strided(one_hot, shape=new_shape, strides=new_stride)

# Extract predictions
result = model.predict_on_batch(one_hot_batch)
"""
                
                # Run predictions in environment
                predictions = self.run_code_in_environment(predict_code, timeout=self.predict_timeout)

                # Extract track and counts
                probabilities, counts = predictions

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

            finally:
                # Clean up the sequence file
                import os
                if os.path.exists(seq_path):
                    os.unlink(seq_path)

        else:
            raise NotImplementedError("ChromBPNet direct prediction not yet implemented")
        
        final_prediction = OraclePrediction()

        # Create a Prediction Object
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
            values=out,
            metadata=None,
            preferred_aggregation='mean',
            preferred_interpolation='linear_divided',
            preferred_scoring_strategy='mean'
        )
        final_prediction.add(f"{self.assay}:{self.cell_type}", track)
        
        return final_prediction
    
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