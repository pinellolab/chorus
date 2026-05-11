"""EPInformer-seq oracle: 256bp sequence → scalar enhancer activity.

The model (``enhancer_predictor_256bp``) is the per-cell sequence encoder
component of EPInformer (Pinello Lab). It takes a one-hot 256-bp DNA
window and returns a single scalar in
``log2(0.1 + sqrt(DNase × H3K27ac))`` space.

Trained per cell on H3K27ac peak-summit windows (5 × 256 bp at offsets
{-2, -1, 0, 1, 2} × 156 bp from each ENCODE H3K27ac narrowPeak summit).
This is the "Enhancer_H3K27ac_DNase" assay — a combined accessibility + active-mark
signal, not separate H3K27ac/DNase tracks.

Used in chorus for variant effect prediction: REF and ALT 256-bp windows
are scored separately; the effect = ALT − REF (signed).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.base import OracleBase
from ..core.exceptions import InvalidAssayError, ModelNotLoadedError
from ..core.globals import CHORUS_DOWNLOADS_DIR
from ..core.interval import GenomeRef, Interval, Sequence
from ..core.result import OraclePrediction, OraclePredictionTrack

from .epinformerseq_source.epinformerseq_globals import (
    EPINFORMERSEQ_AVAILABLE_ASSAYS,
    EPINFORMERSEQ_AVAILABLE_CELLTYPES,
    EPINFORMERSEQ_DEFAULT_ASSAY,
    EPINFORMERSEQ_DEFAULT_STEP,
    EPINFORMERSEQ_WINDOW,
)
from .epinformerseq_source.exceptions import EPInformerSeqError


logger = logging.getLogger(__name__)

EPINFORMERSEQ_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "epinformerseq"
EPINFORMERSEQ_MODELS_DIR.mkdir(exist_ok=True, parents=True)


class EPInformerSeqOracle(OracleBase):
    """EPInformer-seq oracle for 256-bp enhancer activity prediction."""

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
        self.sequence_length = EPINFORMERSEQ_WINDOW
        self.n_targets = 1
        self.bin_size = step_size
        self.model_dir = model_dir
        self.average_reverse = average_reverse
        self.reference_fasta = reference_fasta
        self.batch_size = batch_size
        self._model = None

    # ------------------------------------------------------------------
    # Weights / templates path resolution
    # ------------------------------------------------------------------

    def get_model_weights_dir(self) -> Path:
        if self.model_dir is not None:
            self.download_dir = Path(self.model_dir)
        path = self.download_dir / self.cell_type
        if not path.exists():
            self._download_model()
        return path

    def get_model_weights_path(self) -> Path:
        return self.get_model_weights_dir() / "weights.pt"

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
        """Load the per-cell weights checkpoint.

        ``cell_type`` lets callers (e.g. ``discover_variant_effects``)
        switch which Roadmap cell line the oracle is pointed at without
        re-instantiating. Validated against
        ``EPINFORMERSEQ_AVAILABLE_CELLTYPES``; resets ``assay_id`` and
        forces a fresh load on the next predict.
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
            self._model = None
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
            "model_weights": str(self.get_model_weights_path()),
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
                logger.info("EPInformer-seq model loaded successfully in environment.")
            else:
                raise ModelNotLoadedError(
                    "Failed to load EPInformer-seq model in chorus-epinformerseq env. "
                    "Run `chorus health --oracle epinformerseq` to diagnose."
                )

    def _load_direct(self):
        try:
            import torch

            from .epinformerseq_source.model_usage import load_model

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
                logger.info(f"EPInformer-seq auto-detected device: {self.device}")

            device = torch.device(self.device)
            model = load_model(str(self.get_model_weights_path()), device=device)
            self._model = model
            self.loaded = True
            logger.info("EPInformer-seq model loaded successfully.")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load EPInformer-seq model: {e}.")

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
            f"Instantiated EPInformer-seq oracle can only predict for assay {self.assay_id}"
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
            "model_weights": str(self.get_model_weights_path()),
            "seq": seq,
            "reverse_aug": reverse_aug,
            "batch_size": self.batch_size,
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
        if self._model is None:
            raise ModelNotLoadedError()
        from .epinformerseq_source.model_usage import predict_activity

        preds, _ = predict_activity(
            self._model, seq=seq, average_reverse=reverse_aug, device=self.device
        )
        return preds

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def fine_tune(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "EPInformer-seq fine-tuning is not supported. Train a new "
            "checkpoint externally with EPInformer's train_seqEncoder.py "
            "if you need a different cell type."
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
        # Placeholder. The HF mirror at lucapinello/chorus-epinformerseq is the
        # primary source; this is a fallback that does not yet exist.
        return ""

    def _try_hf_mirror(self, dest_dir: Path) -> bool:
        """Fetch one cell's weights.pt from the HF mirror.

        Returns True on success (file copied to dest_dir / 'weights.pt').
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.info("huggingface_hub not available; cannot fetch EPInformer-seq weights")
            return False
        try:
            local = hf_hub_download(
                repo_id="lucapinello/chorus-epinformerseq",
                filename=f"{self.cell_type}/weights.pt",
                repo_type="model",
            )
            import shutil as _shutil

            dest_dir.mkdir(parents=True, exist_ok=True)
            _shutil.copyfile(local, dest_dir / "weights.pt")
            logger.info(f"Fetched EPInformer-seq {self.cell_type} weights from HF mirror.")
            return True
        except Exception as exc:
            logger.info(f"chorus-epinformerseq HF mirror unavailable ({exc}).")
            return False

    def _download_model(self):
        cell_dir = self.download_dir / self.cell_type
        if not self._try_hf_mirror(cell_dir):
            raise EPInformerSeqError(
                f"Could not download EPInformer-seq {self.cell_type} weights. "
                "Either provide a local --model-dir pointing at a directory of "
                f"per-cell subdirs containing weights.pt, or wait for the HF "
                "mirror at lucapinello/chorus-epinformerseq to be populated."
            )
