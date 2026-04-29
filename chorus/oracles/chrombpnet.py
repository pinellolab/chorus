"""ChromBPNet oracle implementation."""

from ..core.base import OracleBase
from ..core.track import Track
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.interval import Interval, GenomeRef, Sequence
from ..core.exceptions import ModelNotLoadedError, InvalidAssayError
from ..core.globals import CHORUS_DOWNLOADS_DIR
from .chrombpnet_source.chrombpnet_globals import CHROMBPNET_MODELS_DICT

from typing import List, Tuple, Optional, ClassVar
import numpy as np
import subprocess
import logging
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# HuggingFace slim mirror for ChromBPNet/BPNet weights.
# Contains only fold-0 chrombpnet_nobias h5's per cell-type (42 models)
# and the 744 BPNet/CHIP h5's. ~1.5 GB total vs ~100 GB for the full
# ENCODE-tarball prefetch. See docs/plans/chrombpnet-hf-slim-mirror.md
# and audits/2026-04-28_chrombpnet_slim_mirror/ for the design and
# verification trail.
HF_SLIM_REPO = "lucapinello/chorus-chrombpnet-slim"
HF_SLIM_REPO_TYPE = "model"


class ChromBPNetOracle(OracleBase):
    """ChromBPNet oracle implementation for TF binding and chromatin accessibility."""

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
        
        # Paths to model weights
        self.download_dir = CHORUS_DOWNLOADS_DIR / "chrombpnet"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = None # will be set when the model is downloaded
        self.model_dir = None # Where templates and utils are stored
        self.model = None

        # Metadata path
        self.JASPAR_metadata = None

        # Model specific parameters
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
            path = self.get_model_weights_dir(assay, cell_type) / 'models' / f"fold_{fold}" / model_type
            # Models can have inner dirs to descend
            if path.exists() and path.is_dir():
                subdirs = [p for p in path.iterdir() if p.is_dir()]
                if len(subdirs) == 1:
                    path = subdirs[0]
    
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

            # Download from JASPAR. Same resumable+locked helper as the
            # ChromBPNet model download — see _download_chrombpnet_model.
            from chorus.utils.http import download_with_resume
            os.makedirs(download_path, exist_ok=True)
            logger.info(f"Downloading {download_link}...")
            download_with_resume(
                download_link, download_file_path,
                label=f"JASPAR motif {os.path.basename(download_link)}",
            )

            logger.info("Download completed!")


    def _download_chrombpnet_model(self):
        
        # Get model's ENCODE idx
        idx = CHROMBPNET_MODELS_DICT[self.assay][self.cell_type]

        # Create download link
        download_link = self.get_encode_link(idx)
        download_path = self.get_model_weights_dir(self.assay, self.cell_type)

        logger.info(f"Dowloading ChromBPNet into {download_path}...")

        download_file_path = os.path.join(
            download_path, 
            os.path.basename(download_link)
        )

        if not os.path.exists(download_file_path):

            # Download from ENCODE (tar file).
            #
            # Use chunked+resumable+locked helper rather than urllib.urlretrieve.
            # Two separate callers (e.g. pytest smoke fixture + a background
            # build_backgrounds_chrombpnet job) racing the same URL were
            # observed during the 2026-04-14 v2 audit to each write to the
            # same partial .tar.gz path and the loser's tarfile.extractall()
            # hit `EOFError: Compressed file ended before the end-of-stream
            # marker was reached`. The new helper holds an fcntl lock on
            # a sibling .lock file so only one writer proceeds at a time,
            # and the second caller resumes (or returns) once the first
            # finishes.
            from chorus.utils.http import download_with_resume
            os.makedirs(download_path, exist_ok=True)
            logger.info(f"Downloading {download_link}...")
            download_with_resume(
                download_link,
                download_file_path,
                label=f"ChromBPNet {self.assay}:{self.cell_type}",
            )

            logger.info("Download completed!")
        


        # Now extract the file in the same download folder. Guard against
        # two callers racing the same extraction: tarfile.extractall creates
        # subdirectories without exist_ok=True, so the loser hits
        # FileExistsError on paths the winner already wrote. Use an fcntl
        # lock on <extract_folder>.lock and skip if extraction has already
        # produced fold_0 weights.
        import fcntl
        extract_folder = os.path.join(download_path, "models")
        os.makedirs(extract_folder, exist_ok=True)
        lock_path = extract_folder + ".extract.lock"
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                fold0_dir = os.path.join(extract_folder, "fold_0")
                if not os.path.isdir(fold0_dir):
                    with tarfile.open(download_file_path, "r:gz") as tar:
                        tar.extractall(path=extract_folder)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)

        # Nested tar extraction (one tar per model_type per fold) — must be
        # inside the same lock as the outer extraction so two concurrent
        # callers don't race on the inner directories. Re-acquire the lock
        # for the nested step and skip any t_out that is already populated.
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                for fold in range(5):
                    models_dir = os.path.join(extract_folder, f"fold_{fold}")
                    tar_mappings = {
                        f"model.bias_scaled.fold_{fold}.*.tar": 'bias_scaled',
                        f"model.chrombpnet.fold_{fold}.*.tar": 'chrombpnet',
                        f"model.chrombpnet_nobias.fold_{fold}.*.tar": 'chrombpnet_nobias'
                    }
                    for t_name, t_type in tar_mappings.items():
                        t_pattern = os.path.join(models_dir, t_name)
                        import glob
                        matches = glob.glob(t_pattern)
                        if not matches:
                            continue
                        t_path = matches[0]
                        t_out = os.path.join(models_dir, t_type)

                        # Skip if already extracted (by concurrent caller
                        # or by a previous run). Populated = directory that
                        # exists and is non-empty.
                        if os.path.isdir(t_out) and os.listdir(t_out):
                            continue

                        try:
                            with tarfile.open(t_path, "r:") as tar:
                                tar.extractall(path=t_out)
                        except Exception:
                            # "tar file" is actually a pre-extracted
                            # directory — rename it in place.
                            if os.path.isdir(t_path) and not os.path.exists(t_out):
                                os.rename(t_path, t_out)
                            else:
                                raise
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
                

    # ── HF slim-mirror helpers (added in 0.3.0) ──────────────────────

    def _try_slim_hf_chrombpnet(self) -> Optional[Path]:
        """Try to fetch the fold-0 ``chrombpnet_nobias`` h5 from the
        HF slim mirror (``lucapinello/chorus-chrombpnet-slim``).

        Returns the local cached path on success, ``None`` on miss or
        when the mirror isn't applicable (CHIP, fold ≠ 0, custom
        model, missing huggingface_hub, network down, repo error).
        Callers fall back to the existing ENCODE-tarball flow.
        """
        if self.assay == "CHIP" or self.fold != 0:
            return None
        encff = CHROMBPNET_MODELS_DICT.get(self.assay, {}).get(self.cell_type)
        if not encff:
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.warning(
                "huggingface_hub not available in this Python environment; "
                "ChromBPNet slim mirror disabled, falling back to the "
                "~700 MB ENCODE tarball flow. Add 'huggingface_hub>=0.20.0' "
                "to environments/chorus-chrombpnet.yml to fix."
            )
            return None
        try:
            manifest_local = hf_hub_download(
                repo_id=HF_SLIM_REPO,
                filename="manifest.json",
                repo_type=HF_SLIM_REPO_TYPE,
            )
        except Exception as exc:  # pragma: no cover — network / repo missing
            logger.info("HF slim mirror unavailable (%s); falling back to ENCODE.", exc)
            return None
        try:
            with open(manifest_local) as f:
                manifest = json.load(f)
            entry = manifest.get(encff)
            if not entry or entry.get("kind") != "ChromBPNet":
                return None
            slim_path = entry["slim_path"]  # e.g. DNASE/HepG2/fold_0/model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5
            h5_local = hf_hub_download(
                repo_id=HF_SLIM_REPO,
                filename=slim_path,
                repo_type=HF_SLIM_REPO_TYPE,
            )
            logger.info("Resolved %s:%s via HF slim mirror (%s)", self.assay, self.cell_type, encff)
            return Path(h5_local)
        except Exception as exc:  # pragma: no cover
            logger.info("HF slim fetch failed (%s); falling back to ENCODE.", exc)
            return None

    def _try_slim_hf_bpnet(self) -> Optional[Path]:
        """Try to fetch a BPNet/CHIP h5 by JASPAR ``BP_BASE_ID`` from
        the slim mirror. Same semantics as the ChromBPNet variant —
        returns ``None`` on miss; caller falls back to JASPAR.
        """
        if self.assay != "CHIP" or self.tf is None:
            return None
        try:
            model_url = self.JASPAR_metadata.get_weights_by_cell_and_tf(
                self.tf, self.cell_type
            )
        except Exception:
            return None
        # URL ends with /<BP_BASE_ID>/<BP_BASE_ID>_model.h5
        bp_id = os.path.basename(os.path.dirname(model_url))
        if not bp_id.startswith("BP"):
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.warning(
                "huggingface_hub not available in this Python environment; "
                "BPNet slim mirror disabled, falling back to JASPAR mencius. "
                "Add 'huggingface_hub>=0.20.0' to environments/chorus-chrombpnet.yml to fix."
            )
            return None
        try:
            h5_local = hf_hub_download(
                repo_id=HF_SLIM_REPO,
                filename=f"BPNet/{bp_id}.h5",
                repo_type=HF_SLIM_REPO_TYPE,
            )
            logger.info("Resolved CHIP:%s:%s via HF slim mirror (%s)", self.cell_type, self.tf, bp_id)
            return Path(h5_local)
        except Exception as exc:  # pragma: no cover
            logger.info("HF slim BPNet fetch failed (%s); falling back to JASPAR.", exc)
            return None

    def load_pretrained_model(
            self,
            assay: Optional[str] = None,
            cell_type: Optional[str] = None,
            weights: Optional[str] = None,
            TF: Optional[str] = None,
            fold: int = 0,
            model_type: str = 'chrombpnet_nobias',
            is_custom: bool = False
        ) -> None:
        """Load ChromBPNet model weights.

        ``model_type`` default in chorus ≥ 0.3 is ``'chrombpnet_nobias'``
        (bias-corrected, what the Kundaje paper recommends for variant
        analysis and what the HF slim mirror ships). Pass
        ``model_type='chrombpnet'`` explicitly for the bias-aware
        variant; that path falls back to the full ENCODE tarball.
        """

        if assay is None or cell_type is None:
            raise InvalidAssayError(
                "You must provide both `assay` and `cell_type` when `weights` is None."
            )

        if not is_custom:
            if assay not in CHROMBPNET_MODELS_DICT:
                raise InvalidAssayError(
                    f"ChromBPNet supports only the following assays: "
                    f"{list(CHROMBPNET_MODELS_DICT.keys())}."
                )

            if assay != "CHIP":
                if cell_type not in CHROMBPNET_MODELS_DICT[assay]:
                    raise InvalidAssayError(
                        f"ChromBPNet {assay} predictions can only be done on the following "
                        f"cell types: {list(CHROMBPNET_MODELS_DICT[assay].keys())}."
                    )

                if fold not in range(5):
                    raise InvalidAssayError(
                        f"ChromBPNet fold must be an integer between 0 and 4, got {fold}."
                    )
            else:
                if TF is None:
                    raise InvalidAssayError(
                        "You must provide `TF` when requesting a ChIP-seq ChromBPNet model."
                    )
        else:
            if weights is None:
                raise InvalidAssayError(
                    "You must provide `weights` when `is_custom=True`."
                )
            
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
            # HF slim mirror first (chorus ≥ 0.3) — covers the common
            # case (fold=0 + chrombpnet_nobias for ATAC/DNase, or any
            # CHIP/BPNet model). On miss / network failure we fall
            # through to the existing ENCODE / JASPAR tarball flow so
            # users on alternate folds or the bias-aware variant keep
            # working.
            slim_path = None
            if not is_custom:
                if assay != "CHIP" and fold == 0 and model_type == "chrombpnet_nobias":
                    slim_path = self._try_slim_hf_chrombpnet()
                elif assay == "CHIP":
                    slim_path = self._try_slim_hf_bpnet()
            if slim_path is not None:
                self.model_path = str(slim_path)
            else:
                # Existing ENCODE/JASPAR tarball flow
                weights = self.get_model_weights_path(assay, cell_type, fold, TF, model_type)
                if not os.path.exists(weights):
                    if assay != "CHIP":
                        self._download_chrombpnet_model()
                        # Rechecking due to download
                        weights = self.get_model_weights_path(assay, cell_type, fold, TF, model_type)
                    else:
                        self._download_model_from_JASPAR()
                if not os.path.exists(weights):
                    raise FileNotFoundError(
                        f"Weights file {weights} not found even after downloading from ENCODE"
                    )
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
                raise ModelNotLoadedError(
                    "Failed to load ChromBPNet model in the chorus-chrombpnet environment. "
                    "Run `chorus health --oracle chrombpnet` to diagnose."
                )
    
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

            # Load model
            if self.assay == "CHIP":
                sys.path.insert(0, self.get_templates_dir()) # to get BPNet visible
                
                from BPNet.arch import BPNet
                
                # Open JSON for architecture
                with open(os.path.join(self.get_templates_dir(), "input_data.json")) as fopen:
                    tasks_raw = json.load(fopen)
                tasks = {int(k): v for k, v in tasks_raw.items()}

                # Create model
                model = BPNet(tasks, {}, name_prefix="main")

                # Load weights
                model.load_weights(weights)
            else:
                model = tf.keras.models.load_model(
                    weights,
                    compile=False,
                    custom_objects={"multinomial_nll": self.multinomial_nll, "tf": tf}
                )
        
            self.model = model

            self.loaded = True
            logger.info("ChromBPNet model loaded successfully!")

        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load ChromBPNet model: {e}.")
    
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

        # Allocate space for output
        seq_len = max(len(full_seq), self.sequence_length)

        if self.use_environment:
            predictions = self._predict_in_environment(
                full_seq, 
                assay_ids
            )

        else:
            predictions = self._predict_direct(
                full_seq, 
                assay_ids
            )

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
            "model_weights": str(self.model_path),
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
    
    def _predict_direct(self, seq: str, assay_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf
        # Mapping dict
        MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        if len(seq) > self.sequence_length:
            num_windows_stride_one = (len(seq) - self.sequence_length + 1)
            num_windows = (num_windows_stride_one + self.sequence_length - 1) // self.sequence_length + 1

            # Define seq_len and flag
            seq_len = (self.output_length * (num_windows - 1)) + self.sequence_length
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
            new_shape = (num_windows, self.sequence_length, 4)
            stride_x, stride_y = one_hot.strides
            new_stride = (stride_x * self.output_length, stride_x, stride_y)

            one_hot_batch = np.lib.stride_tricks.as_strided(one_hot, shape=new_shape, strides=new_stride)

        # Extract predictions
        if self.assay == "CHIP":
            # JASPAR models require bias profile and counts
            profile_bias = (
                np.zeros((num_windows, self.output_length, 2), dtype="float32") if not trimmed
                else np.zeros((1, self.output_length, 2), dtype="float32")
            )
            count_bias = (
                np.zeros((num_windows, 1), dtype="float32") if not trimmed
                else np.zeros((1, 1), dtype="float32")
            )
            result = self.model.predict_on_batch(
                [one_hot_batch, profile_bias, count_bias]
            ) 
        else:
            result = self.model.predict_on_batch(one_hot_batch)

        return result
            
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