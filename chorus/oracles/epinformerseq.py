"""EPInformer-seq per-cell oracle: 2114-bp sequence → scalar DNase activity.

The oracle's public ``predict`` path returns one scalar per region. The
underlying ``PerCellProfileNetWide + BiasNet`` does emit a full per-bp profile
internally (see ``model_usage.predict_profile``), but only the aggregated
scalar is exposed through the chorus track interface.

Architecture:

* **PerCellProfileNetWide** — one model per cell type (no FiLM, no cell
  embedding). Dilated CNN; a **2114-bp** input is run through the body, then
  the central **1024 bp** is cropped for the profile + count heads
  (ChromBPNet-style "valid" geometry, full real-sequence receptive field per
  output base).
* **Per-cell frozen BiasNet** (ChromBPNet-style, 1024-bp) — subtracts Tn5 /
  MNase sequence preference in logit space, recovering cell specificity. It
  runs on the central 1024 bp of the 2114-bp input.
* Trained on two channels at DNase peak summits across 11 Roadmap cells,
  fold-10 leave-chrom-out CV split: **ch0 = 5′ DNase cut-sites**,
  **ch1 = H3K27ac coverage**.

Scalar definition (must match the background CDF builder at
``scripts/build_backgrounds_epinformerseq_v2_percell.py``), per-bp peak max over
the **central 256 bp** of the 1024-bp output:

* ``Enhancer_DNase`` (default): max DNase.
* ``Enhancer_H3K27ac``: max H3K27ac.
* ``Enhancer_H3K27ac_DNase``: composite ``sqrt(max DNase × max H3K27ac)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.base import OracleBase
from ..core.exceptions import InvalidAssayError, ModelNotLoadedError
from ..core.globals import CHORUS_DOWNLOADS_DIR
from ..core.interval import GenomeRef, Interval, Sequence
from ..core.result import OraclePrediction, OraclePredictionTrack

from .epinformerseq_source.globals import (
    EPINFORMERSEQ_AVAILABLE_ASSAYS,
    EPINFORMERSEQ_AVAILABLE_CELLTYPES,
    EPINFORMERSEQ_DEFAULT_ASSAY,
    EPINFORMERSEQ_DEFAULT_STEP,
    EPINFORMERSEQ_WIDE_WINDOW,
)
from .epinformerseq_source.exceptions import EPInformerSeqError


logger = logging.getLogger(__name__)

EPINFORMERSEQ_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "epinformerseq"
EPINFORMERSEQ_MODELS_DIR.mkdir(exist_ok=True, parents=True)


class EPInformerSeqOracle(OracleBase):
    """EPInformer-seq per-cell oracle: profile output + bias-correction.

    Layout under ``~/.chorus/downloads/epinformerseq/``:
        per_cell_widewin/{cell_type}/main.pt (PerCellProfileNetWide)
        bias/{cell_type}/bias.pt             (frozen per-cell BiasNet)
    """

    def __init__(
        self,
        cell_type: str = "K562",
        assay: str = EPINFORMERSEQ_DEFAULT_ASSAY,
        step_size: int = EPINFORMERSEQ_DEFAULT_STEP,
        batch_size: int = 1,
        use_environment: bool = True,
        reference_fasta: str | None = None,
        model_load_timeout: int | None = 600,
        predict_timeout: int | None = 300,
        device: str | None = None,
        average_reverse: bool = False,
        model_dir: str | None = None,
    ):
        self.oracle_name = "epinformerseq"
        # Single architecture: PerCellProfileNetWide, 2114-bp input -> central
        # 1024-bp profile crop (ChromBPNet-style geometry). DNase-only.
        self.in_window = EPINFORMERSEQ_WIDE_WINDOW
        if cell_type not in EPINFORMERSEQ_AVAILABLE_CELLTYPES:
            raise EPInformerSeqError(
                f"Cell type {cell_type!r} not available. Choose from: "
                f"{EPINFORMERSEQ_AVAILABLE_CELLTYPES}"
            )
        if assay not in EPINFORMERSEQ_AVAILABLE_ASSAYS:
            raise EPInformerSeqError(
                f"Assay {assay!r} not supported. Available: "
                f"{EPINFORMERSEQ_AVAILABLE_ASSAYS}"
            )
        self.cell_type = cell_type
        self.assay = assay
        self.assay_id = f"{self.assay}:{self.cell_type}"

        super().__init__(
            use_environment=use_environment,
            model_load_timeout=model_load_timeout,
            predict_timeout=predict_timeout,
            device=device,
        )
        if self.device is None:
            self.device = "auto"

        self.download_dir = EPINFORMERSEQ_MODELS_DIR
        # widewin needs a 2114-bp input window; the profile output stays 1024 bp.
        self.sequence_length = self.in_window
        self.n_targets = 1
        self.bin_size = step_size
        self.model_dir = model_dir
        self.average_reverse = average_reverse
        self.reference_fasta = reference_fasta
        self.batch_size = batch_size
        self._main_model = None
        self._bias_model = None

    # ------------------------------------------------------------------
    # Weight path resolution
    # ------------------------------------------------------------------

    def get_root_dir(self) -> Path:
        if self.model_dir is not None:
            return Path(self.model_dir)
        return self.download_dir

    def get_main_weights_path(self) -> Path:
        root = self.get_root_dir()
        path = root / "per_cell_widewin" / self.cell_type / "main.pt"
        if not path.exists():
            self._download_model()
        return path

    def get_bias_weights_path(self) -> Path:
        root = self.get_root_dir()
        path = root / "bias" / self.cell_type / "bias.pt"
        if not path.exists():
            self._download_model()
        return path

    def get_model_dir_path(self) -> Path:
        return Path(__file__).parent / "epinformerseq_source"

    def get_templates_dir(self) -> Path:
        return self.get_model_dir_path() / "templates"

    def get_load_template(self):
        path = self.get_templates_dir() / "load_template.py"
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    def get_predict_template(self):
        path = self.get_templates_dir() / "predict_template.py"
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pretrained_model(
        self,
        weights: str | None = None,
        cell_type: str | None = None,
    ) -> None:
        """Load the shared main model + this cell's bias model.

        ``cell_type`` lets callers switch cells without re-instantiating.
        """
        self._check_env_ready()
        if weights is not None:
            self.model_dir = weights
        if cell_type is not None and cell_type != self.cell_type:
            if cell_type not in EPINFORMERSEQ_AVAILABLE_CELLTYPES:
                raise EPInformerSeqError(
                    f"Cell type {cell_type!r} not available. Choose from: "
                    f"{EPINFORMERSEQ_AVAILABLE_CELLTYPES}"
                )
            self.cell_type = cell_type
            self.assay_id = f"{self.assay}:{self.cell_type}"
            self._main_model = None
            self._bias_model = None
            self.loaded = False
            self._model_info = None
        if self.use_environment:
            self._load_in_environment()
        else:
            self._load_direct()

    def _load_in_environment(self):
        args = {
            "device": self.device,
            "sequence_length": self.sequence_length,
            "main_weights": str(self.get_main_weights_path()),
            "bias_weights": str(self.get_bias_weights_path()),
            "cell_type": self.cell_type,
            "assay": self.assay,
        }
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()
            template, arg = self.get_load_template()
            template = template.replace(arg, arg_file.name)
            model_info = self.run_code_in_environment(template, timeout=self.model_load_timeout)
            if model_info and model_info.get("loaded"):
                self.loaded = True
                self._model_info = model_info
                logger.info("EPInformer-seq v2 model loaded successfully in environment.")
            else:
                raise ModelNotLoadedError(
                    "Failed to load EPInformer-seq v2 model in chorus-epinformerseq env."
                )

    def _load_direct(self):
        try:
            import torch

            from .epinformerseq_source.model_usage import load_main_model, load_bias_model

            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                ):
                    self.device = "mps"
                else:
                    self.device = "cpu"
                logger.info(f"EPInformer-seq v2 auto-detected device: {self.device}")

            device = torch.device(self.device)
            self._main_model = load_main_model(
                str(self.get_main_weights_path()), device=device
            )
            self._bias_model = load_bias_model(str(self.get_bias_weights_path()), device=device)
            self.loaded = True
            logger.info("EPInformer-seq v2 model loaded successfully.")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load EPInformer-seq v2 model: {e}.")

    # ------------------------------------------------------------------
    # Public API contracts (OracleBase abstract methods)
    # ------------------------------------------------------------------

    def list_assay_types(self) -> List[str]:
        return list(EPINFORMERSEQ_AVAILABLE_ASSAYS)

    def list_cell_types(self) -> List[str]:
        return [self.cell_type]

    def _validate_loaded(self):
        if not self.loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_pretrained_model first.")

    def _validate_assay_ids(self, assay_ids: List[str] | None):
        if assay_ids is None or (len(assay_ids) == 1 and assay_ids[0] == self.assay_id):
            return
        raise InvalidAssayError(
            f"Instantiated EPInformer-seq v2 oracle can only predict for assay {self.assay_id}"
        )

    def _get_context_size(self) -> int:
        return self.sequence_length

    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        return (self.sequence_length, self.sequence_length)

    def _get_bin_size(self) -> int:
        return self.bin_size

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(
        self,
        seq: str | Tuple[str, int, int] | Interval,
        assay_ids: List[str] = None,
    ) -> OraclePrediction:
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinates.")
            chrom, start, end = seq
            query_interval = Interval.make(
                GenomeRef(chrom=chrom, start=start, end=end, fasta=self.reference_fasta)
            )
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
            preds = self._predict_in_environment(seq=full_seq, reverse_aug=self.average_reverse)
        else:
            preds = self._predict_direct(seq=full_seq, reverse_aug=self.average_reverse)

        final_prediction = OraclePrediction()
        track = OraclePredictionTrack.create(
            cls_name=self.assay,
            source_model="epinformerseq",
            assay_id=self.assay_id,
            track_id=self.assay_id,
            assay_type=self.assay,
            cell_type=self.cell_type,
            query_interval=query_interval,
            prediction_interval=prediction_interval,
            input_interval=input_interval,
            resolution=self.bin_size,
            values=preds,
            metadata=None,
            preferred_aggregation="mean",
            preferred_interpolation="linear_divided",
            preferred_scoring_strategy="mean",
        )
        final_prediction.add(self.assay_id, track)
        return final_prediction

    def _predict_in_environment(self, seq: str, reverse_aug: bool = False) -> np.ndarray:
        args = {
            "device": self.device,
            "sequence_length": self.sequence_length,
            "main_weights": str(self.get_main_weights_path()),
            "bias_weights": str(self.get_bias_weights_path()),
            "cell_type": self.cell_type,
            "assay": self.assay,
            "seq": seq,
            "reverse_aug": reverse_aug,
            "batch_size": self.batch_size,
            "in_window": self.in_window,
        }
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()
            template, arg = self.get_predict_template()
            template = template.replace(arg, arg_file.name)
            model_predictions = self.run_code_in_environment(template, timeout=self.predict_timeout)
            return np.array(model_predictions["preds"], dtype=np.float32)

    def _predict_direct(self, seq: str, reverse_aug: bool = False) -> np.ndarray:
        if self._main_model is None or self._bias_model is None:
            raise ModelNotLoadedError()
        from .epinformerseq_source.model_usage import predict_activity

        preds, _ = predict_activity(
            self._main_model, self._bias_model,
            seq=seq, cell_type=self.cell_type,
            assay=self.assay,
            average_reverse=reverse_aug,
            device=self.device,
            in_window=self.in_window,
        )
        return preds

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def fine_tune(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "EPInformer-seq v2 fine-tuning is not supported through this oracle. "
            "Train a new checkpoint externally with the legnet_profile_v2 pipeline."
        )

    def get_status(self) -> Dict[str, Any] | None:
        status = {
            "name": self.__class__.__name__,
            "loaded": self.loaded,
            "use_environment": self.use_environment,
            "environment_info": None,
        }
        if self.use_environment:
            status["environment_info"] = self.get_environment_info()
        return status

    def get_zenodo_link(self) -> str:
        return ""

    def _try_hf_mirror(self) -> bool:
        """Fetch per_cell/{cell}/main.pt + bias/{cell}/bias.pt from the HF mirror.

        Returns True on success.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.info("huggingface_hub not available; cannot fetch EPInformer-seq per-cell weights.")
            return False
        try:
            root = self.get_root_dir()
            root.mkdir(parents=True, exist_ok=True)
            (root / "per_cell_widewin" / self.cell_type).mkdir(parents=True, exist_ok=True)
            (root / "bias" / self.cell_type).mkdir(parents=True, exist_ok=True)
            import shutil as _shutil

            main_local = hf_hub_download(
                repo_id="lucapinello/chorus-epinformerseq-v2",
                filename=f"per_cell_widewin/{self.cell_type}/main.pt",
                repo_type="model",
            )
            _shutil.copyfile(main_local, root / "per_cell_widewin" / self.cell_type / "main.pt")
            bias_local = hf_hub_download(
                repo_id="lucapinello/chorus-epinformerseq-v2",
                filename=f"bias/{self.cell_type}/bias.pt",
                repo_type="model",
            )
            _shutil.copyfile(bias_local, root / "bias" / self.cell_type / "bias.pt")
            logger.info(f"Fetched EPInformer-seq per-cell weights for {self.cell_type} from HF mirror.")
            return True
        except Exception as exc:
            logger.info(f"chorus-epinformerseq-v2 HF mirror unavailable ({exc}).")
            return False

    def _download_model(self):
        if not self._try_hf_mirror():
            raise EPInformerSeqError(
                "Could not download EPInformer-seq per-cell weights. "
                "Either provide a local --model-dir pointing at a directory containing "
                "per_cell_widewin/{cell_type}/main.pt and bias/{cell_type}/bias.pt, or wait "
                "for the HF mirror at lucapinello/chorus-epinformerseq-v2 to be populated."
            )
