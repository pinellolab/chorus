"""AlphaGenome PyTorch backend oracle (spike).

Wraps the upstream PyTorch port at ``genomicsxai/alphagenome-pytorch``.
Track schema, assay identifiers, and the ``alphagenome_tracks.json`` cache
are shared with the JAX-backed :class:`AlphaGenomeOracle`; only the
load + forward path differs.

The PyTorch port adds:

- Native MPS support on Apple Silicon (the JAX path forces CPU on macOS).
- Turnkey variant scoring via ``alphagenome_pytorch.variant_scoring``
  (not yet wired through chorus — exposed only via the upstream API).
- LoRA / linear-probe fine-tuning hooks (also not yet wired through chorus).

This is an opt-in side-by-side backend pending equivalence and benchmark
verification. Use ``chorus.create_oracle('alphagenome_pt', ...)``.
"""

from ..core.base import OracleBase
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.track import Track
from ..core.interval import Interval, GenomeRef, Sequence
from ..core.exceptions import ModelNotLoadedError

from typing import List, Tuple, Union, Optional, Dict, Any
import os
import logging
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaGenomePTOracle(OracleBase):
    """AlphaGenome PyTorch-port oracle (opt-in backend).

    Mirrors :class:`AlphaGenomeOracle` shape; differs only in load + forward.
    Shares the 5,731-track metadata cache and `Track` conversion conventions
    so percentile CDFs, walkthroughs, and report templates port over without
    schema changes.
    """

    HF_REPO = "gtca/alphagenome_pytorch"
    WEIGHTS_FILENAME = "model_all_folds.safetensors"

    def __init__(
        self,
        use_environment: bool = True,
        reference_fasta: Optional[str] = None,
        model_load_timeout: Optional[int] = 900,
        predict_timeout: Optional[int] = 600,
        device: Optional[str] = None,
        organism: str = "human",
        hf_repo: Optional[str] = None,
        weights_filename: Optional[str] = None,
    ):
        self.oracle_name = "alphagenome_pt"

        super().__init__(
            use_environment=use_environment,
            model_load_timeout=model_load_timeout,
            predict_timeout=predict_timeout,
            device=device,
        )

        self.sequence_length = 1_048_576
        self.target_length = 1_048_576
        self.bin_size = 1
        self.organism = organism
        self.hf_repo = hf_repo or self.HF_REPO
        self.weights_filename = weights_filename or self.WEIGHTS_FILENAME

        self._model = None
        self._track_dict = None

        self.reference_fasta = reference_fasta
        self.model_dir = None

    # ------------------------------------------------------------------
    # Model paths
    # ------------------------------------------------------------------
    def get_model_weights_path(self) -> str:
        return ""

    def get_model_dir_path(self) -> str:
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "alphagenome_pt_source")
        return self.model_dir

    def get_templates_dir(self) -> str:
        return os.path.join(self.get_model_dir_path(), "templates")

    def get_load_template(self) -> Tuple[str, str]:
        path = os.path.join(self.get_templates_dir(), "load_template.py")
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    def get_predict_template(self) -> Tuple[str, str]:
        path = os.path.join(self.get_templates_dir(), "predict_template.py")
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_pretrained_model(self, weights: str = None) -> None:
        logger.info("Loading AlphaGenome PyTorch port")
        if self.use_environment:
            self._load_in_environment(weights)
        else:
            self._load_direct(weights)

    def _resolve_torch_device(self):
        import torch

        if self.device == "mps":
            return torch.device("mps")
        if self.device == "cuda" or (self.device and self.device.startswith("cuda:")):
            return torch.device(self.device)
        if self.device == "cpu":
            return torch.device("cpu")
        # Auto: MPS > CUDA > CPU. CUDA wins on Linux boxes; MPS wins on macOS.
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_direct(self, weights: str) -> None:
        try:
            import huggingface_hub
            from .alphagenome_pt_source import _mps_compat  # noqa: F401
            from alphagenome_pytorch import AlphaGenome

            try:
                huggingface_hub.whoami()
            except huggingface_hub.errors.LocalTokenNotFoundError:
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
                    "HUGGING_FACE_HUB_TOKEN"
                )
                if hf_token:
                    huggingface_hub.login(
                        token=hf_token, add_to_git_credential=False
                    )

            weights_path = huggingface_hub.hf_hub_download(
                repo_id=self.hf_repo, filename=self.weights_filename
            )
            device = self._resolve_torch_device()
            model = AlphaGenome.from_pretrained(weights_path, device=device)
            model.eval()

            self._model = model
            self.model = model
            self.loaded = True
            logger.info(
                "AlphaGenome PyTorch model loaded on %s (weights: %s)",
                device,
                weights_path,
            )
        except Exception as e:
            raise ModelNotLoadedError(
                f"Failed to load AlphaGenome PyTorch model: {e}."
            )

    def _load_in_environment(self, weights: str) -> None:
        args = {
            "device": self.device,
            "hf_repo": self.hf_repo,
            "weights_filename": self.weights_filename,
        }
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

        try:
            template, placeholder = self.get_load_template()
            template = template.replace(placeholder, arg_file.name)
            model_info = self.run_code_in_environment(
                template, timeout=self.model_load_timeout
            )
            if model_info and model_info.get("loaded"):
                self.loaded = True
                self._model_info = model_info
                logger.info(
                    "AlphaGenome PyTorch model loaded in environment (device=%s)",
                    model_info.get("device"),
                )
            else:
                raise ModelNotLoadedError(
                    "Failed to load AlphaGenome PyTorch model in environment"
                )
        finally:
            os.unlink(arg_file.name)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict(
        self,
        seq: Union[str, Tuple[str, int, int], Interval],
        assay_ids: Optional[List[str]] = None,
    ) -> OraclePrediction:
        if assay_ids is None:
            assay_ids = self.get_all_assay_ids()

        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError(
                    "Reference FASTA required for genomic coordinate input"
                )
            chrom, start, end = seq
            query_interval = Interval.make(
                GenomeRef(
                    chrom=chrom, start=start, end=end, fasta=self.reference_fasta
                )
            )
        elif isinstance(seq, str):
            query_interval = Interval.make(Sequence(sequence=seq))
        elif isinstance(seq, Interval):
            query_interval = seq
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")

        input_interval = query_interval.extend(self.sequence_length)
        prediction_interval = query_interval.extend(self.output_size)

        full_seq = input_interval.sequence

        # Same N-padding handling as the JAX path — AlphaGenome architecture
        # accepts variable-length input but rejects sequences padded with N.
        from .alphagenome import AlphaGenomeOracle

        full_seq = AlphaGenomeOracle._strip_n_padding(full_seq)

        if self.use_environment:
            raw_result = self._predict_in_environment(full_seq, assay_ids)
        else:
            raw_result = self._predict_direct(full_seq, assay_ids)

        from .alphagenome_source.alphagenome_metadata import get_metadata

        metadata = get_metadata()

        final_prediction = OraclePrediction()
        for ind, assay_id in enumerate(assay_ids):
            track_id = metadata.get_track_by_identifier(assay_id)
            if track_id is None:
                raise ValueError(f"Assay ID not found in metadata: {assay_id}")
            info = metadata.get_track_info(track_id)
            if info is None:
                raise ValueError(
                    f"No track info for index {track_id} (assay {assay_id})"
                )
            types_info = metadata.parse_description(info["description"])
            resolution = info.get("resolution", 1)

            values = np.array(raw_result["values"][ind], dtype=np.float32)

            track = OraclePredictionTrack.create(
                source_model="alphagenome_pt",
                assay_id=assay_id,
                track_id=track_id,
                assay_type=types_info["assay_type"],
                cell_type=types_info["cell_type"],
                query_interval=query_interval,
                prediction_interval=prediction_interval,
                input_interval=input_interval,
                resolution=resolution,
                values=values,
                metadata=info,
                preferred_aggregation="sum",
                preferred_interpolation="linear_divided",
                preferred_scoring_strategy="mean",
            )
            final_prediction.add(assay_id, track)

        return final_prediction

    def _predict_in_environment(self, seq: str, assay_ids: List[str]) -> dict:
        args = {
            "device": self.device,
            "hf_repo": self.hf_repo,
            "weights_filename": self.weights_filename,
            "length": self.sequence_length,
            "sequence": seq,
            "assay_ids": assay_ids,
        }
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

        try:
            template, placeholder = self.get_predict_template()
            template = template.replace(placeholder, arg_file.name)
            result = self.run_code_in_environment(
                template, timeout=self.predict_timeout
            )
        finally:
            os.unlink(arg_file.name)
        return result

    def _predict_direct(self, seq: str, assay_ids: List[str]) -> dict:
        from .alphagenome_source.alphagenome_metadata import (
            get_metadata,
            SKIPPED_OUTPUT_TYPES,
        )
        import torch

        metadata = get_metadata()

        needed_output_types = set()
        for aid in assay_ids:
            idx = metadata.get_track_by_identifier(aid)
            if idx is None:
                raise ValueError(f"Assay ID not found in metadata: {aid}")
            info = metadata.get_track_info(idx)
            if info is None:
                raise ValueError(f"No track info for index {idx} (assay {aid})")
            needed_output_types.add(info["output_type"])

        _OUTPUT_TYPE_TO_PT_KEY = {
            "ATAC": "atac",
            "DNASE": "dnase",
            "CAGE": "cage",
            "RNA_SEQ": "rna_seq",
            "CHIP_HISTONE": "chip_histone",
            "CHIP_TF": "chip_tf",
            "PROCAP": "procap",
            "SPLICE_SITES": "splice_sites",
            "SPLICE_SITE_USAGE": "splice_site_usage",
        }
        heads = tuple(
            _OUTPUT_TYPE_TO_PT_KEY[ot]
            for ot in needed_output_types
            if ot in _OUTPUT_TYPE_TO_PT_KEY and ot not in SKIPPED_OUTPUT_TYPES
        )

        device = next(self._model.parameters()).device
        _BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq_arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, b in enumerate(seq):
            j = _BASE_TO_IDX.get(b.upper(), -1)
            if j >= 0:
                seq_arr[i, j] = 1.0
        dna_onehot = torch.from_numpy(seq_arr).unsqueeze(0).to(device)

        with torch.no_grad():
            # Use forward() instead of predict() so we can pass heads= to
            # skip computation for output types we don't need. predict()
            # is a thin wrapper without that filter.
            output = self._model(
                dna_onehot,
                organism_index=0,
                heads=heads if heads else None,
            )

        collected = []
        resolutions = []
        for aid in assay_ids:
            idx = metadata.get_track_by_identifier(aid)
            info = metadata.get_track_info(idx)
            ot_name = info["output_type"]
            local_idx = info["local_index"]
            res = info.get("resolution", 1)
            pt_key = _OUTPUT_TYPE_TO_PT_KEY.get(ot_name)
            if pt_key is None or pt_key not in output:
                raise ValueError(
                    f"Output type {ot_name} not produced by PyTorch port "
                    f"(key={pt_key})"
                )
            head_out = output[pt_key]
            if isinstance(head_out, dict):
                if res not in head_out:
                    res = next(iter(head_out.keys()))
                tensor = head_out[res]
            else:
                tensor = head_out
            arr = tensor.detach().cpu().numpy()
            if arr.ndim == 3:
                track_values = arr[0, :, local_idx]
            elif arr.ndim == 2:
                track_values = arr[:, local_idx]
            else:
                raise ValueError(
                    f"Unexpected output array shape {arr.shape} for {ot_name}"
                )
            collected.append(track_values.astype(np.float32).tolist())
            resolutions.append(int(res))

        return {"values": collected, "resolutions": resolutions}

    # ------------------------------------------------------------------
    # Metadata helpers (delegate to shared cache)
    # ------------------------------------------------------------------
    def list_assay_types(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata
        return get_metadata().list_assay_types()

    def list_cell_types(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata
        return get_metadata().list_cell_types()

    def get_all_assay_ids(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        metadata = get_metadata()
        ids = []
        for aid, idx in metadata._track_index_map.items():
            if "/Padding/" in aid:
                continue
            info = metadata.get_track_info(idx)
            if info and info.get("name", "").lower() == "padding":
                continue
            ids.append(aid)
        return ids

    def get_track_info(
        self, query: str = None
    ) -> Union[pd.DataFrame, Dict[str, int]]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        metadata = get_metadata()
        if query:
            return metadata.search_tracks(query)
        return metadata.get_track_summary()

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------
    def fine_tune(
        self, tracks: List[Track], track_names: List[str], **kwargs
    ) -> None:
        raise NotImplementedError(
            "Fine-tuning via chorus is not yet wired for the PyTorch backend. "
            "Upstream supports LoRA + linear-probe via "
            "alphagenome_pytorch.scripts.finetune."
        )

    def _get_context_size(self) -> int:
        return self.sequence_length

    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        return (1000, self.sequence_length)

    def _get_bin_size(self) -> int:
        return self.bin_size

    @property
    def output_size(self) -> int:
        return self.target_length * self.bin_size

    def get_status(self) -> Dict[str, Any]:
        status = {
            "name": self.__class__.__name__,
            "loaded": self.loaded,
            "use_environment": self.use_environment,
            "environment_info": None,
        }
        if self.use_environment:
            status["environment_info"] = self.get_environment_info()
        return status
