"""AlphaGenome oracle implementation."""

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


class AlphaGenomeOracle(OracleBase):
    """AlphaGenome oracle with automatic environment management.

    AlphaGenome (Google DeepMind, Nature 2026) predicts 5,930 human functional
    genomic tracks at single base-pair resolution from up to 1 MB of DNA
    sequence using a JAX-based model.

    Track identifiers use the AlphaGenome naming convention, e.g.
    ``"CL:0000084 ATAC-seq"`` for T-cell ATAC-seq. Use
    ``list_assay_types()`` and ``get_track_info(query)`` to discover
    available tracks.
    """

    def __init__(
        self,
        use_environment: bool = True,
        reference_fasta: Optional[str] = None,
        model_load_timeout: Optional[int] = 900,
        predict_timeout: Optional[int] = 600,
        device: Optional[str] = None,
        fold: str = "all_folds",
        organism: str = "human",
    ):
        self.oracle_name = "alphagenome"

        super().__init__(
            use_environment=use_environment,
            model_load_timeout=model_load_timeout,
            predict_timeout=predict_timeout,
            device=device,
        )

        # AlphaGenome specific parameters
        self.sequence_length = 1_048_576  # 1 MB input window
        self.target_length = 1_048_576    # single bp resolution output
        self.bin_size = 1                 # default (most modalities are 1bp)
        self.fold = fold
        self.organism = organism

        # Model state
        self._model = None
        self._track_dict = None

        # Reference genome
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
            self.model_dir = os.path.join(parent, "alphagenome_source")
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
    # Model loading
    # ------------------------------------------------------------------
    def load_pretrained_model(self, weights: str = None) -> None:
        logger.info("Loading AlphaGenome model")

        if self.use_environment:
            self._load_in_environment(weights)
        else:
            self._load_direct(weights)

    def _load_direct(self, weights: str) -> None:
        try:
            import os
            import jax
            import huggingface_hub
            from alphagenome_research.model.dna_model import create_from_huggingface

            # Ensure HuggingFace auth (required for gated AlphaGenome model)
            try:
                huggingface_hub.whoami()
            except huggingface_hub.errors.LocalTokenNotFoundError:
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    huggingface_hub.login(token=hf_token, add_to_git_credential=False)
                else:
                    raise ModelNotLoadedError(
                        "AlphaGenome requires HuggingFace authentication. "
                        "Set the HF_TOKEN environment variable or run 'huggingface-cli login'. "
                        "You must also accept the model license at "
                        "https://huggingface.co/google/alphagenome"
                    )

            # Determine JAX device — AlphaGenome requires explicit device when no GPU
            # NOTE: Metal is skipped in auto-detect (too experimental for AlphaGenome)
            if self.device and self.device.startswith("cpu"):
                jax_device = jax.devices("cpu")[0]
            elif self.device and self.device.startswith("gpu"):
                jax_device = jax.devices("gpu")[0]
            elif self.device and self.device.startswith("metal"):
                jax_device = jax.devices("METAL")[0]
            else:
                # Auto-detect: prefer CUDA GPU > CPU (Metal skipped)
                available_platforms = {d.platform for d in jax.devices()}
                if "gpu" in available_platforms:
                    jax_device = jax.devices("gpu")[0]
                else:
                    jax_device = jax.devices("cpu")[0]

            model = create_from_huggingface(self.fold, device=jax_device)
            self._model = model
            self.model = model
            self.loaded = True
            logger.info("AlphaGenome model loaded successfully on %s", jax_device)
        except ModelNotLoadedError:
            raise
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load AlphaGenome model: {e}")

    def _load_in_environment(self, weights: str) -> None:
        args = {
            "device": self.device,
            "fold": self.fold,
        }

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as arg_file:
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
                logger.info("AlphaGenome model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError(
                    "Failed to load AlphaGenome model in environment"
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

        # Build query interval
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
                raise ValueError(f"No track info for index {track_id} (assay {assay_id})")
            types_info = metadata.parse_description(info["description"])
            resolution = info.get("resolution", 1)

            values = np.array(raw_result["values"][ind], dtype=np.float32)

            track = OraclePredictionTrack.create(
                source_model="alphagenome",
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

    def _predict_in_environment(
        self, seq: str, assay_ids: List[str]
    ) -> dict:
        args = {
            "device": self.device,
            "fold": self.fold,
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
        from alphagenome.models.dna_output import OutputType

        metadata = get_metadata()

        # Determine which output types we need
        needed_output_types = set()
        for aid in assay_ids:
            idx = metadata.get_track_by_identifier(aid)
            if idx is None:
                raise ValueError(f"Assay ID not found in metadata: {aid}")
            info = metadata.get_track_info(idx)
            if info is None:
                raise ValueError(f"No track info for index {idx} (assay {aid})")
            needed_output_types.add(info["output_type"])

        requested_outputs = [
            ot
            for ot in OutputType
            if ot.name in needed_output_types
            and ot.name not in SKIPPED_OUTPUT_TYPES
        ]

        output = self._model.predict_sequence(
            seq,
            requested_outputs=requested_outputs,
            ontology_terms=None,
        )

        # Extract per-assay values
        collected = []
        resolutions = []
        for aid in assay_ids:
            idx = metadata.get_track_by_identifier(aid)
            if idx is None:
                raise ValueError(f"Assay ID not found in metadata: {aid}")
            info = metadata.get_track_info(idx)
            if info is None:
                raise ValueError(f"No track info for index {idx} (assay {aid})")
            ot_name = info["output_type"]
            local_idx = info["local_index"]
            ot_enum = OutputType[ot_name]
            track_data = output.get(ot_enum)
            if track_data is None:
                raise ValueError(
                    f"No prediction data for output type {ot_name} (assay {aid})"
                )
            values = np.asarray(track_data.values)[:, local_idx]
            collected.append(values.tolist())
            resolutions.append(info["resolution"])

        return {"values": collected, "resolutions": resolutions}

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def list_assay_types(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        return get_metadata().list_assay_types()

    def list_cell_types(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        return get_metadata().list_cell_types()

    def get_all_assay_ids(self) -> List[str]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        return list(get_metadata()._track_index_map.keys())

    def get_track_info(
        self, query: str = None
    ) -> Union[pd.DataFrame, Dict[str, int]]:
        from .alphagenome_source.alphagenome_metadata import get_metadata

        metadata = get_metadata()
        if query:
            return metadata.search_tracks(query)
        else:
            return metadata.get_track_summary()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    def fine_tune(
        self, tracks: List[Track], track_names: List[str], **kwargs
    ) -> None:
        raise NotImplementedError(
            "Fine-tuning is not yet implemented for AlphaGenome"
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
