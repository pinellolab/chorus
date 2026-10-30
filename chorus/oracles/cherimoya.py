"""Cherimoya oracle: the CATv1 (accessibility) and BATv1 (ChIP-seq) atlases.

Cherimoya is a compact convolutional sequence-to-function model in the BPNet /
ChromBPNet family.  Two atlases of per-experiment models are served here, in
the way ChromBPNet serves ATAC, DNASE and CHIP from one oracle:

* **CATv1** — the Cherimoya Accessibility aTlas — 1,518 chromatin accessibility
  models (~0.6 M parameters each) covering 1,149 ENCODE DNase-seq and 369
  ATAC-seq experiments, assays ``ATAC`` and ``DNASE``.
* **BATv1** — the Cherimoya Binding ATlas — 3,429 transcription-factor and
  DNA-associated-protein binding models (~46 K parameters each) covering 1,139
  targets in 183 biosamples, assay ``CHIP``.  These take a control track the
  oracle cannot supply and predict two strands; see
  :mod:`chorus.oracles.cherimoya_source.batv1_globals` for how both are handled.

Every model is trained across the same five chromosome-held-out folds.

This oracle is the closest analog to :class:`~chorus.oracles.chrombpnet.ChromBPNetOracle`
and shares its exact geometry (2114 bp input, 1000 bp base-resolution output),
so the offset arithmetic and the 501 bp central scoring window are ported
rather than reinvented.  Two things differ substantively:

* **Track ids are ``ASSAY:ENCSR``**, e.g. ``DNASE:ENCSR000EOT`` or
  ``CHIP:ENCSR000EGM``, not ``ASSAY:cell_type``.  ``(assay, biosample)`` is
  ambiguous for 1,188 of the 1,518 CATv1 experiments — even ``ATAC:K562`` maps
  to four — and ``(biosample, target)`` for 437 of BATv1's 2,474 pairs, so the
  ENCODE experiment accession is the identifier.  See
  :func:`~chorus.oracles.cherimoya_source.catv1_globals.catv1_track_id`.
* **The count head predicts ``log(count + 1)``**, so recovering counts uses
  ``expm1``.  That transform lives in
  :mod:`chorus.oracles.cherimoya_source.scoring`, shared with the background
  builder so the two cannot drift.

Both atlases are GRCh38-only; unlike ChromBPNet they ship no mouse models.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy

from ..core.base import OracleBase
from ..core.exceptions import InvalidAssayError, ModelNotLoadedError
from ..core.globals import CHORUS_DOWNLOADS_DIR
from ..core.interval import GenomeRef, Interval, Sequence
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.track import Track
from ..utils.genome import missing_reference_fasta_error
from .cherimoya_source.catv1_globals import (
    CATV1_ASSAY_TYPES,
    CATV1_BIN_SIZE,
    CATV1_CHECKPOINT_TEMPLATE,
    CATV1_DEFAULT_FOLD,
    CATV1_HF_REPO,
    CATV1_INPUT_LENGTH,
    CATV1_N_FOLDS,
    CATV1_ENSEMBLE,
    CATV1_OUTPUT_LENGTH,
    CATV1_TRIMMING,
    catv1_track_id,
)
from .cherimoya_source.batv1_globals import (
    BATV1_ASSAY_TYPES,
    BATV1_CHECKPOINT_TEMPLATE,
    BATV1_HF_REPO,
    BATV1_N_CONTROL_TRACKS,
    BATV1_SIGNAL_GROUPS,
)
from .cherimoya_source.catv1_metadata import get_metadata
from .cherimoya_source.scoring import (
    expected_counts_profile,
    heads_equivalent_to_profile,
)

logger = logging.getLogger(__name__)

# The one-time advisory assembly check now reads chorus.utils.genome's table rather
# than a local copy of hg38's chr1 length -- CATv1 is hg38-only and silently scoring
# an hg19 FASTA produces plausible-looking nonsense, and two copies of the same
# constant is how that kind of check drifts out of agreement with the guard beside it.


class CherimoyaOracle(OracleBase):
    """Cherimoya oracle: CATv1 chromatin accessibility (DNase / ATAC) and BATv1 ChIP-seq binding."""

    #: Weights are trained on GRCh38. Enforced, not assumed -- see #124.
    training_genome = "hg38"

    def __init__(
        self,
        use_environment: bool = True,
        reference_fasta: Optional[str] = None,
        model_load_timeout: Optional[int] = 600,
        predict_timeout: Optional[int] = 300,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        """Initialise the oracle.

        Args:
            use_environment: Run model code in the isolated
                ``chorus-cherimoya`` conda environment.
            reference_fasta: Path to a GRCh38 FASTA, required for
                coordinate-based queries.
            model_load_timeout: Seconds before a load is abandoned.
            predict_timeout: Seconds before a prediction is abandoned.
            device: ``'cpu'``, ``'cuda'``, ``'cuda:N'``, or None to auto-detect.
            batch_size: Sliding-window batch size for wide queries.
        """
        # Geometry, set before super().__init__() alongside oracle_name so
        # that base-class helpers (e.g. predict_variant_effect's window
        # widening, which reads self.sequence_length) see real values.
        self.sequence_length = CATV1_INPUT_LENGTH
        self.output_length = CATV1_OUTPUT_LENGTH
        self.bin_size = CATV1_BIN_SIZE
        self.batch_size = batch_size

        self.oracle_name = "cherimoya"

        super().__init__(
            use_environment=use_environment,
            model_load_timeout=model_load_timeout,
            predict_timeout=predict_timeout,
            device=device,
        )

        self.reference_fasta = reference_fasta
        self._assembly_checked = False

        self.download_dir = CHORUS_DOWNLOADS_DIR / "cherimoya"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.model_path = None
        # Every loaded checkpoint. Length > 1 means the 5-fold ensemble;
        # _forward_windows dispatches on this in BOTH execution modes.
        self.model_paths = []
        self._model_info = None

        # Set by load_pretrained_model.
        self.assay = None
        self.cell_type = None
        self.encode_id = None
        self.tf = None
        self.fold = CATV1_DEFAULT_FOLD

    # ── paths / templates ────────────────────────────────────────────

    def get_model_dir_path(self) -> str:
        """Directory holding the vendored metadata and env templates."""
        parent = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(parent, "cherimoya_source")

    def get_templates_dir(self) -> str:
        """Directory holding the load/predict environment templates."""
        return os.path.join(self.get_model_dir_path(), "templates")

    def _read_template(self, name: str) -> Tuple[str, str]:
        path = os.path.join(self.get_templates_dir(), name)
        with open(path) as inp:
            return inp.read(), "__ARGS_FILE_NAME__"

    def get_load_template(self) -> Tuple[str, str]:
        """Return ``(source, placeholder)`` for the load template."""
        return self._read_template("load_template.py")

    def get_predict_template(self) -> Tuple[str, str]:
        """Return ``(source, placeholder)`` for the predict template."""
        return self._read_template("predict_template.py")

    # ── model loading ────────────────────────────────────────────────

    @property
    def atlas(self) -> Optional[str]:
        """``"CATv1"`` or ``"BATv1"``, from the loaded assay; ``None`` before a load."""
        if self.assay is None:
            return None
        return "BATv1" if self.assay in BATV1_ASSAY_TYPES else "CATv1"

    @property
    def normalization_key(self) -> str:
        """Which per-track CDF key ranks predictions from this oracle's current atlas and fold.

        A percentile is a rank against a null, so the null has to come from the same model.
        chorus ships four Cherimoya artefacts -- ``cherimoya_pertrack.npz`` built on CATv1
        fold 0 (the default), ``cherimoya_ensemble_pertrack.npz`` on CATv1's 5-fold mean, and
        ``batv1_pertrack.npz`` / ``batv1_ensemble_pertrack.npz`` for the ChIP-seq atlas -- and
        this is how a caller asks for the right one:

            get_normalizer(oracle.normalization_key)

        Returning a *key* rather than threading a ``variant=`` argument through every call
        site is deliberate: ``PerTrackNormalizer`` is already keyed by a name that need not be
        an oracle (see ``_CDF_ALIASES``, where ``alphagenome_pt`` resolves to
        ``alphagenome``'s CDF), so this rides an existing seam instead of widening twenty
        signatures. The load-time guard in ``_ensure_loaded`` then checks the artefact's
        stamped fold against the key, so a stale or mislabelled file fails loudly.
        """
        from ..analysis.normalization import normalization_key

        return normalization_key(self.oracle_name, fold=self.fold, atlas=self.atlas)

    def load_pretrained_model(
        self,
        assay: Optional[str] = None,
        cell_type: Optional[str] = None,
        encode_id: Optional[str] = None,
        fold: int = CATV1_DEFAULT_FOLD,
        weights: Optional[str] = None,
        TF: Optional[str] = None,
    ) -> None:
        """Load one experiment's checkpoint, from CATv1 or BATv1.

        A Cherimoya model is specific to a single ENCODE experiment, so exactly
        one experiment is loaded at a time — the same one-model-per-instance
        contract ChromBPNet uses.

        Because ``(assay, cell_type)`` is ambiguous for most of CATv1, and
        ``(cell_type, TF)`` for a fifth of BATv1, both resolve through committed
        defaults tables (``cherimoya_source/catv1_defaults.py`` and
        ``batv1_defaults.py``); ambiguous pairs log which experiment was
        chosen.  Pass ``encode_id`` to pin one exactly.

        Args:
            assay: ``'ATAC'`` or ``'DNASE'`` (CATv1), or ``'CHIP'`` (BATv1).
            cell_type: Biosample term name, e.g. ``'K562'``.
            encode_id: ENCODE experiment accession, e.g. ``'ENCSR000EOT'``.
                Overrides ``cell_type`` and ``TF``.
            TF: ChIP-seq target as ENCODE names it, e.g. ``'GATA1'``; required
                with ``assay='CHIP'`` unless ``encode_id`` is given. Named as
                ChromBPNet names it so ``load_oracle`` passes it through unchanged.
            fold: Cross-validation fold ``0``–``4``, or ``CATV1_ENSEMBLE``
                (``"ensemble"``) to average the predictions of all five, which is
                the **default** and what the shipped background CDFs are built on.
                CATv1's model card offers both; the ensemble is used because one
                checkpoint is a sample rather than the model — at rs12740374 the
                five folds give accessibility ratios spanning 2.39–3.47 for the
                identical sequence. Pinning a single fold ranks it against an
                ensemble-built null, so prefer the default unless you are
                deliberately comparing folds.
            weights: Path to a local ``.torch`` checkpoint, bypassing
                HuggingFace. ``assay`` is still required so the emitted
                track gets the right type; ``encode_id`` names the track.

        Raises:
            InvalidAssayError: bad assay, fold, or missing identifiers.
            ModelNotLoadedError: the checkpoint failed to load.
        """
        self._check_env_ready()

        ensemble = fold == CATV1_ENSEMBLE
        if not ensemble and fold not in range(CATV1_N_FOLDS):
            raise InvalidAssayError(
                f"CATv1 fold must be an integer in 0..{CATV1_N_FOLDS - 1} or "
                f"{CATV1_ENSEMBLE!r}, got {fold!r}."
            )
        if not ensemble and fold != 0:
            # Refused rather than allowed-with-a-warning, because the failure would be a
            # silently wrong number rather than a missing one: a percentile is a rank against
            # a null, chorus ships nulls for fold 0 and the ensemble only, and the folds are
            # not interchangeable -- on DNASE:ENCSR149XIL their peaks span 7.65 to 15.47
            # (2.02x). Ranking fold 3 against fold 0's null returns a percentile that looks
            # perfectly normal and is wrong.
            raise InvalidAssayError(
                f"Cherimoya fold {fold} has no matching background null, so its percentiles "
                f"would be ranked against a different model's distribution. chorus ships "
                f"nulls for fold 0 (the default) and {CATV1_ENSEMBLE!r} only. The five folds "
                f"disagree by ~2x on the same sequence, so this is not an approximation. "
                f"To use fold {fold}, build its null first: "
                f"scripts/build_backgrounds_cherimoya.py --fold {fold}"
            )

        if weights is not None:
            if assay is None:
                raise InvalidAssayError(
                    "Pass `assay` alongside `weights` so the emitted track "
                    "gets the correct assay type."
                )
            resolved_assay = assay.upper()
            if resolved_assay not in CATV1_ASSAY_TYPES + BATV1_ASSAY_TYPES:
                raise InvalidAssayError(
                    f"Cherimoya covers {sorted(CATV1_ASSAY_TYPES + BATV1_ASSAY_TYPES)}, "
                    f"not {assay!r}."
                )
            resolved_id = encode_id or Path(weights).stem
            resolved_cell_type = cell_type
            resolved_tf = TF
            self.model_path = str(weights)
            self.model_paths = [self.model_path]
        else:
            try:
                resolved_assay, resolved_id = get_metadata().resolve(
                    assay=assay, cell_type=cell_type, encode_id=encode_id, tf=TF,
                )
            except (KeyError, ValueError) as exc:
                raise InvalidAssayError(str(exc)) from exc
            row = get_metadata().describe(resolved_id)
            resolved_cell_type = row["biosample"]
            resolved_tf = row["target"] or None
            if ensemble:
                self.model_paths = [
                    self._download_checkpoint(resolved_id, f, resolved_assay)
                    for f in range(CATV1_N_FOLDS)
                ]
                self.model_path = self.model_paths[0]
            else:
                self.model_path = self._download_checkpoint(resolved_id, fold, resolved_assay)
                self.model_paths = [self.model_path]

        self.assay = resolved_assay
        self.encode_id = resolved_id
        self.cell_type = resolved_cell_type
        self.tf = resolved_tf
        self.fold = fold

        logger.info(
            "Loading Cherimoya %s (%s, %s)...",
            self.track_id,
            f"{resolved_tf} in {resolved_cell_type}" if resolved_tf else resolved_cell_type,
            "5-fold ensemble" if ensemble else f"fold {fold}",
        )

        if self.use_environment:
            self._load_in_environment(self.model_path)
        else:
            self._load_direct(self.model_path)
            self._models = [self.model]
            for extra in self.model_paths[1:]:
                self._load_direct(extra)
                self._models.append(self.model)
            # self.model stays the FIRST fold so anything reaching past the
            # oracle API sees a real single model rather than a tuple.
            self.model = self._models[0]

    def _download_checkpoint(self, encode_id: str, fold: int, assay: str) -> str:
        """Fetch one checkpoint from HuggingFace, returning the cached path.

        The atlas, and so the repository, follows the assay: ``CHIP`` experiments
        live in BATv1, ``ATAC`` and ``DNASE`` in CATv1. Both lay out checkpoints
        the same way.
        """
        if assay in BATV1_ASSAY_TYPES:
            repo, template = BATV1_HF_REPO, BATV1_CHECKPOINT_TEMPLATE
        else:
            repo, template = CATV1_HF_REPO, CATV1_CHECKPOINT_TEMPLATE
        filename = template.format(encode_id=encode_id, fold=fold)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ModelNotLoadedError(
                "huggingface_hub is required to fetch Cherimoya weights. Add "
                "'huggingface_hub>=0.20' to environments/chorus-cherimoya.yml."
            ) from exc

        try:
            return hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Failed to download {filename} from {repo}: {exc}"
            ) from exc

    def _load_in_environment(self, weights: str) -> None:
        args = {"device": self.device, "model_weights": str(weights)}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()
            template, placeholder = self.get_load_template()
            template = template.replace(placeholder, arg_file.name)
            model_info = self.run_code_in_environment(
                template, timeout=self.model_load_timeout,
            )

        if not (model_info and model_info.get("loaded")):
            raise ModelNotLoadedError(
                "Failed to load the Cherimoya model in the chorus-cherimoya "
                "environment. Run `chorus health --oracle cherimoya` to "
                "diagnose."
            )

        self._check_geometry(
            trimming=model_info.get("trimming"),
            n_control_tracks=model_info.get("n_control_tracks"),
            signal_groups=model_info.get("signal_groups"),
        )
        self._model_info = model_info
        self.loaded = True
        logger.info("Cherimoya model loaded in environment (%s).", model_info.get("device"))

    def _load_direct(self, weights: str) -> None:
        """Load in the current interpreter (used by tests and the builder)."""
        try:
            import torch

            # Before `import cherimoya`: the knob is read when Triton's
            # autotune decorators are evaluated, which happens on import.
            from .cherimoya_source._triton_autotune import enable_autotune_cache
            enable_autotune_cache()

            from cherimoya import Cherimoya
        except ImportError as exc:
            raise ModelNotLoadedError(
                f"Cherimoya and torch must be importable to load directly "
                f"({exc}). Either install them here or use "
                f"use_environment=True."
            ) from exc

        device = self.device
        if device in (None, "", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"

        try:
            # compile=False is mandatory: Cherimoya.load defaults to
            # compile=True with mode='max-autotune', which would add a
            # multi-minute warmup per model -- ruinous for the background
            # build, which loads thousands of them.
            model = Cherimoya.load(weights, device=device, compile=False)
            model = model.eval()
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Failed to load Cherimoya model from {weights}: {exc}"
            ) from exc

        self._check_geometry(
            trimming=int(model.trimming),
            n_control_tracks=int(model.n_control_tracks),
            signal_groups=[int(g) for g in model.signal_groups],
        )
        if model.n_control_tracks:
            # A BATv1 checkpoint takes the experiment's control track as a second
            # input, which chorus cannot supply. ControlWrapper feeds an all-zero
            # control (the model card's setting for sequence-only predictions), so
            # the wrapped model takes the sequence alone like a CATv1 one. The
            # predict template does the same in environment mode.
            from cherimoya import ControlWrapper

            model = ControlWrapper(model)
        self.model = model
        self._resolved_device = device
        self.loaded = True
        logger.info("Cherimoya model loaded directly on %s.", device)

    def _check_geometry(self, trimming, n_control_tracks, signal_groups) -> None:
        """Fail loudly if a checkpoint's geometry isn't what we assume.

        Every offset in this oracle, and every background CDF, assumes a
        2114 bp input trimmed by 557 to a 1000 bp output window.  A CATv1
        checkpoint has no control tracks and one output track; a BATv1 one
        has two control tracks and one group of two strands.  A checkpoint
        that differs from what its atlas declares would otherwise produce
        silently misaligned or mis-scaled tracks, so the atlas is checked
        against the checkpoint rather than inferred from it.
        """
        if trimming is not None and int(trimming) != CATV1_TRIMMING:
            raise ModelNotLoadedError(
                f"Checkpoint has trimming={trimming}, expected "
                f"{CATV1_TRIMMING} ({CATV1_INPUT_LENGTH} -> "
                f"{CATV1_OUTPUT_LENGTH}). This oracle's offsets and the "
                f"background CDFs assume Cherimoya atlas geometry."
            )
        if self.atlas == "BATv1":
            expected_controls, expected_groups, atlas = BATV1_N_CONTROL_TRACKS, BATV1_SIGNAL_GROUPS, "BATv1"
        else:
            expected_controls, expected_groups, atlas = 0, [1], "CATv1"
        if n_control_tracks is not None and int(n_control_tracks) != expected_controls:
            raise ModelNotLoadedError(
                f"Checkpoint expects {n_control_tracks} control tracks; {atlas} "
                f"models have {expected_controls}. Is assay={self.assay!r} the "
                f"right atlas for this checkpoint?"
            )
        if signal_groups is not None and list(signal_groups) != expected_groups:
            raise ModelNotLoadedError(
                f"Checkpoint has signal_groups={signal_groups}; {atlas} "
                f"models have {expected_groups}. Is assay={self.assay!r} the "
                f"right atlas for this checkpoint?"
            )

    # ── track identity / metadata ────────────────────────────────────

    @property
    def track_id(self) -> Optional[str]:
        """Canonical ``ASSAY:ENCSR`` id for the loaded experiment."""
        if self.assay is None or self.encode_id is None:
            return None
        return catv1_track_id(self.assay, self.encode_id)

    @property
    def output_size(self) -> int:
        """Width in bp of the predicted window."""
        return self.bin_size * self.output_length

    def list_assay_types(self) -> List[str]:
        """Return the assay types across both atlases: ``ATAC``, ``CHIP``, ``DNASE``."""
        return sorted(CATV1_ASSAY_TYPES + BATV1_ASSAY_TYPES)

    def list_cell_types(self, assay: Optional[str] = None) -> List[str]:
        """Return biosample term names.

        Like ChromBPNet's, the default is the accessibility (CATv1) biosamples,
        407 of them; ``assay="CHIP"`` gives BATv1's 183.
        """
        return get_metadata().list_cell_types(assay)

    def list_targets(self) -> List[str]:
        """Return every ChIP-seq target in BATv1 (1,139 of them)."""
        return get_metadata().list_targets()

    def list_tracks(self) -> List[str]:
        """Return every canonical track id across both atlases (4,947 of them)."""
        return get_metadata().list_track_ids()

    def _describe_tracks(self) -> list:
        """Both atlases — 1,518 CATv1 + 3,429 BATv1 tracks. Available before load; `track_id` is None until then."""
        from ..core.tracks import TrackRecord

        meta = get_metadata()
        out = []
        for d in meta.tracks_df.sort_values("track_id").to_dict(orient="records"):
            tid = d["track_id"]
            summary = d.get("biosample_summary")
            target = d.get("target") or None
            extra = {k: d[k] for k in ("experiment_accession", "biosample_classification", "atlas") if k in d}
            if target:
                # The same shape ChromBPNet's CHIP records carry, so a consumer that reads
                # `extra["tf"]` or the description sees one vocabulary for both oracles.
                extra["tf"] = target
                extra["quality_flags"] = d.get("quality_flags") or ""
                description = f"{target} binding in {d['biosample']}"
            else:
                # `biosample` is the key CATv1-metadata.tsv actually uses. The first version looked
                # for "biosample_term_name", so cell_type was None on all 1,518 records -- counts were
                # right, so the conformance test passed, and the field the catalogue exists to let you
                # filter on was empty.
                description = summary if summary and str(summary) != "nan" else None
            out.append(TrackRecord(
                track_id=str(tid),
                assay=d.get("assay") or (str(tid).split(":")[0] if ":" in str(tid) else None),
                cell_type=d.get("biosample"),
                description=description,
                extra=extra,
            ))
        return out

    def describe_track(self, track_or_accession: str) -> Dict:
        """Return atlas, assay, target, biosample, and fold-0 metrics for a track.

        Track ids carry only assay and accession, so this is how a caller
        recovers the biosample, the ChIP-seq target and provenance.

        Args:
            track_or_accession: ``'DNASE:ENCSR000EOT'``, ``'CHIP:ENCSR000EGM'``
                or a bare accession.
        """
        return get_metadata().describe(track_or_accession)

    def search_tracks(self, query: str):
        """Substring-search the atlas; returns a DataFrame."""
        return get_metadata().search_tracks(query)

    # ── prediction ───────────────────────────────────────────────────

    def _check_assembly(self) -> None:
        """Warn once if the reference isn't the assembly the atlases were trained on.

        Advisory, unlike the build-side ``require_reference_assembly``: refusing here
        would change behaviour for anyone deliberately scoring a non-reference FASTA,
        which is a separate decision from #124's build and load guards. It does now
        *name* the assembly it found rather than printing two chromosome lengths, and
        it reads the shared table instead of a local copy of hg38's chr1 length.
        """
        if self._assembly_checked or not self.reference_fasta:
            return
        self._assembly_checked = True
        from ..utils.genome import chr1_length, detect_assembly

        found = detect_assembly(self.reference_fasta)
        if found == self.training_genome:
            return
        length = chr1_length(self.reference_fasta)
        if length is None:
            return                      # unreadable or no chr1 -- no claim to make
        logger.warning(
            "Reference %s looks like %s (chr1 is %s bp), not %s. Every Cherimoya model is "
            "trained on %s; predictions against another assembly resolve fine and are "
            "silently about different DNA.",
            self.reference_fasta, found or "an assembly chorus does not recognise",
            f"{length:,}", self.training_genome, self.training_genome,
        )

    def _track_metadata(self, **extra) -> Dict:
        """Metadata stamped on every emitted track.

        ``fold`` and ``atlas`` are read back by ``normalization.normalization_key`` to pick
        the null a prediction is ranked against; ``description`` is what
        ``classify_chip_layer`` reads to split CHIP tracks into TF binding and histone
        marks, and no BATv1 target is a histone mark, so every CHIP track lands in
        ``tf_binding`` with the 501 bp window -- the same as ChromBPNet's BPNet models.
        """
        meta = {
            "encode_id": self.encode_id,
            "fold": self.fold,
            "atlas": self.atlas,
        }
        if self.tf:
            meta["target"] = self.tf
            meta["description"] = f"{self.tf} binding in {self.cell_type}"
        meta.update(extra)
        return meta

    def _transform_predictions_to_tracks(
        self,
        profile_logits: numpy.ndarray,
        log_counts: numpy.ndarray,
        seq_len: int,
    ) -> numpy.ndarray:
        """Stitch per-window head outputs into one base-resolution track.

        Mirrors ``ChromBPNetOracle._transform_predictions_to_tracks``: the
        concatenated per-window outputs are laid into a ``seq_len`` array
        starting at the trim offset, so the values line up with the
        prediction interval.

        Args:
            profile_logits: ``(n_windows, 1, output_length)``.
            log_counts: ``(n_windows, 1)`` predicted ``log(count + 1)``.
            seq_len: Length of the output array.

        Returns:
            ``(seq_len,)`` expected counts per base pair.
        """
        profiles = expected_counts_profile(profile_logits, log_counts)
        flat = profiles.reshape(-1)

        out = numpy.zeros(seq_len)
        start = CATV1_TRIMMING
        end = min(start + len(flat), seq_len)
        out[start:end] = flat[: end - start]
        return out

    def _predict(
        self,
        seq: "str | Tuple[str, int, int] | Interval",
        assay_ids: Optional[List[str]] = None,
    ) -> OraclePrediction:
        """Predict accessibility or binding for a sequence, region, or interval.

        Args:
            seq: DNA string, ``(chrom, start, end)``, or an
                :class:`~chorus.core.interval.Interval`.
            assay_ids: Ignored — a Cherimoya instance holds exactly one
                experiment, so the loaded model determines the track. Kept
                for interface compatibility with the other oracles.

        Returns:
            An :class:`~chorus.core.result.OraclePrediction` with a single
            ``ASSAY:ENCSR`` track.
        """
        self._validate_loaded()
        self._check_assembly()

        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise missing_reference_fasta_error(self.oracle_name)
            chrom, start, end = seq
            query_interval = Interval.make(GenomeRef(
                chrom=chrom, start=start, end=end, fasta=self.reference_fasta,
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
        seq_len = max(len(full_seq), self.sequence_length)

        windows = self._window_sequences(full_seq)
        profile_logits, log_counts = self._forward_windows(windows)

        values = self._transform_predictions_to_tracks(
            profile_logits, log_counts, seq_len,
        )

        track = OraclePredictionTrack.create(
            source_model="cherimoya",
            # The CANONICAL id, ASSAY:ENCSR -- which is what the background CDF
            # rows are keyed on, and what this module's docstring says a Cherimoya
            # track id is. A bare accession here silently loses BOTH percentiles:
            # PerTrackNormalizer looks up by assay_id, and 'ENCSR149XIL' matches no
            # row, so effect_percentile and activity_percentile both return None
            # with no warning. Verified: 'ENCSR149XIL' -> None/None,
            # 'DNASE:ENCSR149XIL' -> 0.9997/0.947.
            #
            # The committed walkthrough hid this because its runner passes the
            # prefixed id in assay_ids explicitly; a user calling predict() or
            # predict_variant_effect() without naming tracks got no percentiles at
            # all. self.track_id is the same catv1_track_id() value the dict key
            # already uses.
            assay_id=self.track_id,
            track_id=None,
            assay_type=self.assay,
            cell_type=self.cell_type,
            query_interval=query_interval,
            prediction_interval=prediction_interval,
            input_interval=input_interval,
            resolution=self.bin_size,
            values=values,
            metadata=self._track_metadata(),
            preferred_aggregation="mean",
            preferred_interpolation="linear_divided",
            preferred_scoring_strategy="mean",
        )

        prediction = OraclePrediction()
        prediction.add(self.track_id, track)
        return prediction

    def _window_sequences(self, seq: str, step: Optional[int] = None) -> List[str]:
        """Cut a sequence into ``sequence_length`` windows spaced by ``step``.

        All geometry lives here rather than in the env template, so the
        sliding-window formula exists in exactly one place — the template
        just does a batched forward pass. The final window is right-padded
        with ``N`` if the sequence runs short, which the model treats as an
        all-zero one-hot column.

        Args:
            seq: Input sequence, any length.
            step: Spacing between window starts. Defaults to
                ``output_length`` (adjacent, non-overlapping outputs).

        Returns:
            List of sequences, each exactly ``sequence_length`` long.
        """
        if step is None:
            step = self.output_length

        if len(seq) > self.sequence_length:
            n_windows = (len(seq) - self.sequence_length) // step + 1
        else:
            n_windows = 1

        windows = []
        for k in range(n_windows):
            start = k * step
            chunk = seq[start:start + self.sequence_length]
            if len(chunk) < self.sequence_length:
                chunk = chunk + "N" * (self.sequence_length - len(chunk))
            windows.append(chunk)
        return windows

    def _forward_windows(self, windows: List[str]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Run the model over pre-cut windows, in-process or in the env.

        Args:
            windows: Sequences, each exactly ``sequence_length`` long.

        Returns:
            ``(profile_logits, log_counts)`` with shapes
            ``(n_windows, 1 or 2, output_length)`` and ``(n_windows, 1)``; two
            for BATv1's strands.
        """
        if len(getattr(self, "model_paths", []) or []) > 1:
            return self._forward_ensemble(windows)
        if self.use_environment:
            return self._forward_in_environment(windows)
        return self._forward_direct(windows)

    def _forward_ensemble(self, windows) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Average the expected-counts predictions across the loaded folds.

        The CATv1 and BATv1 model cards: "use a single fold (e.g. fold_0), or
        average the predictions of all five folds for a more robust estimate."

        The mean is over the **expected-counts profiles** -- that is what
        "predictions" means here. Not over the two raw heads: both enter
        :func:`expected_counts_profile` non-linearly (softmax across positions,
        ``expm1`` on the count head), so averaging heads computes a different and
        meaningless quantity. Nor over per-fold log2FCs: measured at
        rs12740374 / ENCSR149XIL, averaging predictions gives log2FC 1.4576 while
        averaging per-fold log2FCs gives 1.4849, and only the former is what the
        card describes.

        The averaged profile is mapped back onto equivalent heads so that every
        caller downstream keeps the two-head contract -- see
        :func:`heads_equivalent_to_profile`.

        Both execution modes are handled, and that is deliberate rather than
        incidental. ``use_environment=True`` is the **default** for users, and an
        earlier version of this method keyed off ``self._models``, which only the
        in-process loader populates -- so in env mode the dispatch fell through and
        an ensemble request silently returned fold 0 alone, with no warning. That
        is precisely the class of defect this release exists to remove, so the
        dispatch now keys off ``model_paths``, which both modes set.
        """
        if self.use_environment:
            # The predict template takes `model_weights` per call, so swapping
            # the path is enough -- no reload. Costs one subprocess per fold.
            keys, attr, runner = list(self.model_paths), "model_path", self._forward_in_environment
        else:
            keys, attr, runner = list(self._models), "model", self._forward_direct

        saved = getattr(self, attr)
        acc = None
        try:
            for key in keys:
                setattr(self, attr, key)
                logits, log_counts = runner(windows)
                profile = expected_counts_profile(logits, log_counts)
                acc = profile if acc is None else acc + profile
        finally:
            setattr(self, attr, saved)

        return heads_equivalent_to_profile(acc / len(keys))

    def _forward_in_environment(self, windows: List[str]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        args = {
            "device": self.device,
            "model_weights": str(self.model_path),
            "windows": windows,
            "batch_size": self.batch_size,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()
            template, placeholder = self.get_predict_template()
            template = template.replace(placeholder, arg_file.name)
            result = self.run_code_in_environment(template, timeout=self.predict_timeout)

        if result is None:
            raise ModelNotLoadedError(
                "Prediction in the chorus-cherimoya environment returned "
                "nothing. Run `chorus health --oracle cherimoya` to diagnose."
            )
        profile_logits, log_counts, device_used = result
        self._record_device(device_used)
        return numpy.asarray(profile_logits), numpy.asarray(log_counts)

    def _record_device(self, device_used: str) -> None:
        """Track the device predictions actually ran on, and warn on drift.

        Cherimoya's Triton kernels and its pure-PyTorch CPU fallback agree
        only to ~1e-2 on the profile logits, so a run that silently lands
        on CPU produces subtly different numbers as well as being far
        slower. Two calls that disagree on device would make the
        builder-vs-``predict()`` invariant fail for a reason that has
        nothing to do with the code under test — so surface it.
        """
        previous = getattr(self, "_last_device", None)
        self._last_device = device_used
        if previous is not None and previous != device_used:
            logger.warning(
                "Cherimoya predictions changed device between calls (%s -> "
                "%s). Results are not numerically comparable across the "
                "Triton and CPU paths; pin device= explicitly.",
                previous, device_used,
            )
        elif previous is None and device_used == "cpu" and self.device in (None, "", "auto"):
            logger.warning(
                "Cherimoya auto-detected no CUDA device and is running on "
                "CPU. This is ~50x slower and numerically differs from the "
                "Triton path by ~1e-2 on the logits. Pass device= "
                "explicitly if that is not what you intended."
            )

    def _forward_direct(self, windows: List[str]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Batched forward pass in the current interpreter."""
        import torch

        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        encoded = []
        for chunk in windows:
            one_hot = numpy.zeros((4, len(chunk)), dtype=numpy.float32)
            for i, base in enumerate(chunk.upper()):
                j = mapping.get(base)
                if j is not None:
                    one_hot[j, i] = 1.0
            encoded.append(one_hot)

        device = getattr(self, "_resolved_device", None) or "cpu"
        self._record_device(device)

        all_logits, all_counts = [], []
        with torch.no_grad():
            for i in range(0, len(encoded), self.batch_size):
                batch = numpy.stack(encoded[i:i + self.batch_size])
                X = torch.from_numpy(batch).to(device)
                profile_logits, log_counts = self.model(X)
                all_logits.append(profile_logits.float().cpu().numpy())
                all_counts.append(log_counts.float().cpu().numpy())

        return numpy.concatenate(all_logits), numpy.concatenate(all_counts)

    def predict_sliding(
        self,
        seq: "str | Tuple[str, int, int] | Interval",
        assay_ids: Optional[List[str]] = None,
        step: Optional[int] = None,
    ) -> OraclePrediction:
        """Predict across a region wider than the model's input window.

        Runs the model at ``step``-spaced centres and stitches the central
        ``output_length`` of each window into one continuous track covering
        *exactly* the requested interval, averaging any overlap. Ported from
        :meth:`~chorus.oracles.chrombpnet.ChromBPNetOracle.predict_sliding`,
        whose geometry is identical.

        Unlike the ChromBPNet version this works under
        ``use_environment=True`` as well, because the forward pass goes
        through :meth:`_forward_windows` rather than touching ``self.model``
        directly.

        Args:
            seq: DNA string, ``(chrom, start, end)``, or an ``Interval``.
            assay_ids: Ignored; the loaded model determines the track.
            step: Window spacing, in ``(0, output_length]``. Defaults to
                ``output_length`` (no overlap).

        Returns:
            An :class:`~chorus.core.result.OraclePrediction` whose values
            cover the query interval at 1 bp resolution.
        """
        self._validate_loaded()
        self._check_assembly()

        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise missing_reference_fasta_error(self.oracle_name)
            chrom, start, end = seq
            query_interval = Interval.make(GenomeRef(
                chrom=chrom, start=start, end=end, fasta=self.reference_fasta,
            ))
        elif isinstance(seq, str):
            query_interval = Interval.make(Sequence(sequence=seq))
        elif isinstance(seq, Interval):
            query_interval = seq
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")

        query_length = len(query_interval)
        if query_length <= self.sequence_length:
            return self._predict(seq, assay_ids)

        if step is None:
            step = self.output_length
        if not (0 < step <= self.output_length):
            raise ValueError(
                f"step must be in (0, output_length={self.output_length}]"
            )

        # Pad each side so every window's central output lands inside the
        # query: the first window starts CATV1_TRIMMING bases to the left.
        side_pad = CATV1_TRIMMING
        n_windows = (query_length + step - 1) // step
        needed = (n_windows - 1) * step + self.sequence_length
        target_len = max(needed, self.sequence_length)

        # Extend left by exactly side_pad, then right to target_len. Doing
        # it in two directional steps -- rather than a single
        # extend(target_len, how="both") -- is what makes the left offset
        # exactly side_pad in *both* the extendible and non-extendible
        # cases, which the stitch offset below depends on:
        #
        #   * extendible: slop() clamps at the chromosome edge, and
        #     pad2length(how='left') makes up any shortfall with N, so the
        #     left offset is side_pad even at position 0 of a chromosome.
        #   * non-extendible: extend() falls through to
        #     pad2length(how='left'), giving lpad == side_pad exactly.
        #
        # ChromBPNet's predict_sliding uses extend(how="both") on the
        # non-extendible branch while still assuming left_pad == side_pad;
        # for a bare-string query wider than the input window that
        # mis-registers the stitched track by
        # ((target_len - Q) // 2 - side_pad) bases.
        staged = query_interval.extend(query_length + side_pad, how="left")
        input_interval = staged.extend(target_len, how="right")

        full_seq = input_interval.sequence
        windows = self._window_sequences(full_seq, step=step)
        profile_logits, log_counts = self._forward_windows(windows)
        profiles = expected_counts_profile(profile_logits, log_counts)

        # Window k's central output covers query bases
        # [k*step - side_pad + side_pad, ... + output_length).
        out = numpy.zeros(query_length, dtype=numpy.float64)
        weight = numpy.zeros(query_length, dtype=numpy.float64)
        for k, profile in enumerate(profiles):
            q_start = k * step
            q_end = q_start + self.output_length
            lo, hi = max(q_start, 0), min(q_end, query_length)
            if hi <= lo:
                continue
            out[lo:hi] += profile[lo - q_start:lo - q_start + (hi - lo)]
            weight[lo:hi] += 1.0

        weight[weight == 0] = 1.0
        out /= weight

        track = OraclePredictionTrack.create(
            source_model="cherimoya",
            # The CANONICAL id, ASSAY:ENCSR -- which is what the background CDF
            # rows are keyed on, and what this module's docstring says a Cherimoya
            # track id is. A bare accession here silently loses BOTH percentiles:
            # PerTrackNormalizer looks up by assay_id, and 'ENCSR149XIL' matches no
            # row, so effect_percentile and activity_percentile both return None
            # with no warning. Verified: 'ENCSR149XIL' -> None/None,
            # 'DNASE:ENCSR149XIL' -> 0.9997/0.947.
            #
            # The committed walkthrough hid this because its runner passes the
            # prefixed id in assay_ids explicitly; a user calling predict() or
            # predict_variant_effect() without naming tracks got no percentiles at
            # all. self.track_id is the same catv1_track_id() value the dict key
            # already uses.
            assay_id=self.track_id,
            track_id=None,
            assay_type=self.assay,
            cell_type=self.cell_type,
            query_interval=query_interval,
            prediction_interval=query_interval,
            input_interval=input_interval,
            resolution=self.bin_size,
            values=out,
            metadata=self._track_metadata(sliding_step=step),
            preferred_aggregation="mean",
            preferred_interpolation="linear_divided",
            preferred_scoring_strategy="mean",
        )

        prediction = OraclePrediction()
        prediction.add(self.track_id, track)
        return prediction

    # ── OracleBase contract ──────────────────────────────────────────

    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Not implemented.

        Cherimoya does expose a real ``fit()``, so unlike the other oracles
        this is a deliberate omission rather than a hard limitation —
        wiring chorus ``Track`` objects through to it is tracked separately.
        """
        raise NotImplementedError(
            "Cherimoya fine-tuning is not wired into chorus yet. Train "
            "directly with the `cherimoya` CLI or `Cherimoya.fit()`."
        )

    def _get_context_size(self) -> int:
        """Return the model's input length in bp."""
        return self.sequence_length

    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max accepted sequence lengths."""
        return (500, self.sequence_length)

    def _get_bin_size(self) -> int:
        """Return the output resolution in bp."""
        return self.bin_size
