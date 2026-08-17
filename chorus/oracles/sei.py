"""Sei oracle implementation."""

from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import os 
import json
import logging

from pathlib import Path

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidAssayError
from ..utils.sequence import extract_sequence_with_padding

from .sei_source.annotations import SeiClass, SeiTarget, SeiClassesList, SeiTargetList
from .sei_source.sei_globals import SEI_WINDOW, SEI_DEFAULT_STEP, SEI_TARGETS, SEI_CLASSES
from ..core.result import OraclePrediction, OraclePredictionTrack
from ..core.interval import Interval, GenomeRef, Sequence
from ..core.globals import CHORUS_DOWNLOADS_DIR
from ..utils.genome import missing_reference_fasta_error


logger = logging.getLogger(__name__)

SEI_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "sei"
SEI_MODELS_DIR.mkdir(exist_ok=True, parents=True)

class SeiOracle(OracleBase):
    """Sei oracle implementation for sequence regulatory activities."""

    #: Weights are trained on GRCh38. Enforced, not assumed -- see #124.
    training_genome = "hg38"
    
    def __init__(self, 
                 step_size: int = SEI_DEFAULT_STEP,
                 sliding_predict: bool = True,
                 batch_size: int = 1,
                 use_environment: bool = True, 
                 reference_fasta: str | None = None,
                 model_load_timeout: int | None = 600,
                 predict_timeout: int | None  = 300,
                 device: str | None = None,
                 average_reverse: bool = True, # in original implementation, Sei average predictions for both strands
                 model_dir: str | None = None):
        
        self.oracle_name = 'sei'
        
        # Now initialize base class with correct oracle name
        super().__init__(use_environment=use_environment, 
                         model_load_timeout=model_load_timeout,
                         predict_timeout=predict_timeout,
                         device=device)
        # Sentinel; resolved to a real torch device inside _load_direct, where
        # torch is importable (chorus-sei env). 'auto' means: prefer cuda > mps > cpu.
        if self.device is None:
            self.device = 'auto'

        # Sei-specific parameters
        self.sequence_length = SEI_WINDOW # Sei input length
        self.n_targets = SEI_TARGETS  # Number of regulatory features
        self.n_classes = SEI_CLASSES # Number of high-level classes 
        self.sliding_predict = sliding_predict
        
        self.bin_size = step_size if self.sliding_predict else self.sequence_length # Sequence-level predictions
        self.model_dir = model_dir 
        self.average_reverse = average_reverse
        self.reference_fasta = reference_fasta
        self.batch_size = batch_size

        self._model = None # Predictor model
        self._normalizer = None # Model to correct model histone scores for nucleosome occupancy
        self._projector = None # Model to get high-level classes 
        self._target_list = None
        self._classes_list = None 
        self.download_dir = SEI_MODELS_DIR
        self._model_info = None

        # NB: the 3.3 GB Zenodo tarball is no longer pulled at construction
        # time. Loading the weights (via load_pretrained_model) now triggers
        # the download if necessary; this keeps SeiOracle() itself cheap
        # and consistent with the other 5 oracles. `chorus setup --oracle sei`
        # pre-downloads the archive so users don't hit the delay
        # on their first predict call.

    def get_model_dir_path(self) -> Path:
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "sei_source")
        return Path(self.model_dir)

    def get_model_weights_path(self) -> Path:
        return self.download_dir / 'model' / "sei.pth"
    
    def get_projector_weights(self) -> Path:
        return self.download_dir / 'model'/ "projvec_targets.npy"
    
    def get_adjustor_params(self) -> Path:
        return self.download_dir / 'model'/"histone_inds.npy"
    
    def get_target_names(self) -> Path:
        cached = self.download_dir / 'model' / "target.names"
        # Fall back to the copy packaged with the source so
        # list_assay_types() works without the 3.3 GB Zenodo archive.
        if cached.exists():
            return cached
        return self.get_model_dir_path() / "target.names"

    def get_classes_names(self) -> Path:
        cached = self.download_dir / 'model' / "seqclass_info.txt"
        if cached.exists():
            return cached
        return self.get_model_dir_path() / "seqclass_info.txt"

    def get_templates_dir(self) -> Path:
        d = self.get_model_dir_path()
        return d / "templates"
    
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
    
    def load_pretrained_model(self, weights: str | None = None) -> None:
        """Load Sei model weights."""
        # Raise EnvironmentNotReadyError up-front if env setup failed and
        # the user explicitly asked for use_environment=True (issue #64).
        self._check_env_ready()
        if weights is not None:
            self.model_dir = weights

        # Lazy download here (not in __init__) to keep SeiOracle()
        # construction cheap for tests/metadata calls and aligned with
        # the other 5 oracles' behavior.
        if (
            not self.get_model_weights_path().exists()
            or not self.get_projector_weights().exists()
            or not self.get_adjustor_params().exists()
            or not self.get_target_names().exists()
            or not self.get_classes_names().exists()
        ):
            self._download_sei_model()

        # Separately materialise the cached seqclass_info.txt even when
        # the Zenodo tarball was already extracted on a prior run —
        # `get_classes_names()` has a packaged-source fallback so the
        # guard above can be satisfied by the source copy alone, leaving
        # downloads/sei/model/seqclass_info.txt missing and `chorus
        # health` reporting sei as Not installed (v23 scorched-earth
        # audit finding).
        self._materialize_cached_seqclass_info()

        if self.use_environment:
            self._load_in_environment()
        else:
            self._load_direct()
    
    def _load_in_environment(self):
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'n_genomic_features': self.n_targets,
            'model_weights': str(self.get_model_weights_path()),
            'projector_weights': str(self.get_projector_weights()),
            'n_classes': self.n_classes,
            'histone_inds': str(self.get_adjustor_params()),
            'targets': str(self.get_target_names()),
            'classes': str(self.get_classes_names())
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
                logger.info("Sei model loaded successfully in environment!")
            else:
                raise ModelNotLoadedError(
                    "Failed to load Sei model in the chorus-sei environment. "
                    "Run `chorus health --oracle sei` to diagnose."
                )
    
    def _load_direct(self):
        try:
            import torch
            from .sei_source.sei import Sei, SeiProjector, SeiNormalizer
            from .sei_source.annotations import SeiClassesList, SeiTargetList

            # Resolve 'auto' sentinel: cuda > mps > cpu.
            if self.device == 'auto':
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
                logger.info(f"Sei auto-detected device: {self.device}")
            device = torch.device(self.device)
            model = Sei(sequence_length=self.sequence_length, n_genomic_features=self.n_targets)
            # Load weights to CPU first so map_location works regardless of target
            # device (mps doesn't accept arbitrary state dicts loaded with
            # map_location='mps' across torch versions); then move to device.
            model_weights = torch.load(self.get_model_weights_path(), map_location='cpu', weights_only=True)
            model_weights = {key.replace("module.model.", ""): value for key, value in model_weights.items()}
            model.load_state_dict(model_weights)
            model.eval()
            model.to(device)

            projector = SeiProjector(weights=self.get_projector_weights(), 
                                     n_classes=self.n_classes)

            normalizer = SeiNormalizer(histone_inds=self.get_adjustor_params())

            targets = SeiTargetList.load(self.get_target_names())
            classes = SeiClassesList.load(self.get_classes_names())


            self._model = model # Predictor model
            self._normalizer = normalizer # Model to correct model histone scores for nucleosome occupancy
            self._projector = projector
            self._target_list = targets
            self._classes_list = classes
            self.loaded = True
            logger.info("Sei model loaded successfully!")
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Sei model: {e}.")

    
    def list_assay_types(self) -> List[str]:
        """Return Sei's assay types."""
        if self._model_info is not None: # model is loaded through environment 
            return self._model_info['assays']
        elif self._target_list is not None: # model is loaded in current environment
            return self._target_list.list_assay_types()
        else:
            from .sei_source.annotations import SeiTargetList
            targets = SeiTargetList.load(self.get_target_names())
            return targets.list_assay_types()

    def list_class_types(self) -> List[str]:
        """Return Sei's classes"""
        if self._model_info is not None: # model is loaded through environment 
            return self._model_info['classes']
        elif self._classes_list is not None: # model is loaded in current environment
            return self._classes_list.list_class_types()
        else:
            classes = SeiClassesList.load(self.get_classes_names())
            return classes.list_class_types()   
    
    def list_cell_types(self) -> List[str]:
        """Return Sei's cell types."""       
        if self._model_info is not None: # model is loaded through environment 
            return self._model_info['celltypes']
        elif self._target_list is not None: # model is loaded in current environment
            return self._target_list.list_cell_types()
        else:
            targets = SeiTargetList.load(self.get_target_names())
            return targets.list_cell_types()

    def list_group_types(self) -> List[str]:
        """Return Sei's group types."""
        if self._model_info is not None: # model is loaded through environment 
            return self._model_info['groups']
        elif self._classes_list is not None: # model is loaded in current environment
            return self._classes_list.list_group_types()
        else:
            classes = SeiClassesList.load(self.get_classes_names())
            return classes.list_group_types()
        
    
    def select_classes(self,
                      pats: list[Tuple[str | None, str | None]] | str,
                      exact: bool=False, 
                      regex: bool=True, 
                      case: bool=False,
                      convert2str: bool = True) -> list[str] | list[SeiClass]:
        if self._classes_list is not None:
            classes = self._classes_list
        else:
            classes = SeiClassesList.load(self.get_classes_names())
        
        selected = classes.select_classes(pats, 
                                      exact=exact, 
                                      regex=regex, 
                                      case=case)
        if not convert2str:
            return selected
        
        selected_ids = [str(cl) for cl in selected]
        return selected_ids
    
    def _cl2ind(self, cls_lst: list[SeiClass]) -> list[int]:
        if self._classes_list is not None:
            classes = self._classes_list
        else:
            classes = SeiClassesList.load(self.get_classes_names())

        return classes.cl2ind(cls_lst)
    
    def select_targets(self,
                       pats: list[Tuple[str | None, str | None]] | str,
                       exact: bool=False, 
                       regex: bool=True, 
                       case: bool=False,
                       convert2str: bool = True) -> list[SeiTarget] | list[SeiClass] | list[str]:
        if self._target_list is not None:
            targets = self._target_list
        else:
            targets = SeiTargetList.load(self.get_target_names())
        
        selected = targets.select_targets(pats, exact=exact, regex=regex, case=case)
        if not convert2str:
            return selected
        
        selected_ids = [str(ta) for ta in selected]
        return selected_ids
    
    def _targets2inds(self, cls_lst: list[SeiTarget]) -> list[int]:
        if self._target_list is not None:
            targets = self._target_list
        else:
            targets = SeiTargetList.load(self.get_target_names())
        return targets.targets2inds(cls_lst)

    def _target_assays_ids(self) -> list[str]:
        if self._target_list is not None:
            targets = self._target_list
        else:
            targets = SeiTargetList.load(self.get_target_names())
        return [str(ta) for ta in targets.targets.keys()]

    def _class_assays_ids(self) -> list[str]:
        if self._classes_list is not None:
            classes = self._classes_list
        else:
            classes = SeiClassesList.load(self.get_classes_names())
        return [str(cl) for cl in classes.classes.keys()]

    def _get_all_assay_ids(self) -> list[str]:
        return self._target_assays_ids() + self._class_assays_ids()

    def _validate_loaded(self):
        """Check if model is loaded."""
        if not self.loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_pretrained_model first.")
    
    #: What `predict()` scores when the caller names nothing.
    #:
    #: Every other oracle defaults to all of its tracks; Sei raised `TypeError: 'NoneType' object is
    #: not iterable` instead, because `_validate_assay_ids` took `List[str]` with no `| None` and
    #: iterated it. That inconsistency blocked a determinism probe that passed `None` exactly as it did
    #: for the other eight.
    #:
    #: Sei defaults to its **40 projected sequence classes**, not all 21,947 tracks. Strict consistency
    #: with enformer would mean an unnamed call scoring 21,907 chromatin profiles as well — a very
    #: large accident to make easy. The classes are also the interpretable output the projection exists
    #: to produce. Profiles remain available by naming them.
    DEFAULT_ASSAY_KIND = "classes"

    def _default_assay_ids(self) -> list:
        """The 40 sequence classes — see `DEFAULT_ASSAY_KIND` for why not all 21,947."""
        return list(self._class_assays_ids())

    def _validate_assay_ids(self, assay_ids=None):
        if assay_ids is None:
            return True
        available_assay_ids = set(self._get_all_assay_ids())
        for ai in assay_ids:
            if ai not in available_assay_ids:
                raise InvalidAssayError(f"Invalid assay ID: {ai}")
        return True

    def _refine_total_length(self, total_length: int) -> int:
        if not self.sliding_predict:
            return self.sequence_length

        div, mod = divmod(total_length, self.bin_size)
        total_length = div * self.bin_size + self.bin_size * (mod > 0)
        return total_length

    
    def _assemble_prediction(self, assay_ids, parsed, target_preds, class_preds,
                             query_interval, prediction_interval, input_interval):
        """Build the OraclePrediction for one sequence from its selected target/class arrays.

        Shared by the single-sequence path and the multi-allele variant path so both produce
        identical track metadata; the only difference between them is whether the raw profiles were
        histone-equalised against the other alleles first.
        """
        mapping = parsed["mapping"]
        sei_targets, sei_classes = parsed["sei_targets"], parsed["sei_classes"]

        final_prediction = OraclePrediction()

        for ind, assay_id in enumerate(assay_ids):
            source, source_ind = mapping[ind]
            if source == 't':
                info = sei_targets[source_ind]
                values = target_preds[:, source_ind]
                assay_type = info.assay
                cell_type = info.celltype
                metadata = None

            elif source == 'c':
                info = sei_classes[source_ind]
                values = class_preds[:, source_ind]
                cell_type = info.group
                # Sei's 40 sequence classes are ONE layer -- regulatory_classification
                # -- not 40 distinct assay types. ``classify_track_layer`` dispatches
                # on the literal string "sequence-class" (scorers.py:297), so assigning
                # ``info.name`` here (e.g. "Polycomb-repressed") fell through to
                # "other", whose LAYER_CONFIGS entry is None, so ``score_track_effect``
                # returned None and EVERY Sei track scored raw_score=None. Sei
                # consequently appeared in no committed example output and its 40
                # background rows were unreachable from the query path -- a whole
                # oracle silently dark, with a built and shipped null behind it.
                # The class name is preserved as the description rather than dropped.
                assay_type = "sequence-class"
                metadata = {"description": info.name}
            else:
                raise ValueError(f"Invalid mapping: {mapping[ind]}")
            

            track = OraclePredictionTrack.create(
                source_model="sei",
                assay_id=assay_id, 
                track_id=source_ind,
                assay_type=assay_type,
                cell_type=cell_type,
                query_interval=query_interval,
                prediction_interval=prediction_interval,
                input_interval=input_interval,
                resolution=self.bin_size,
                values=values,
                metadata=metadata,
                preferred_aggregation='sum',
                preferred_interpolation='linear_divided',
                preferred_scoring_strategy='mean'
            )
            final_prediction.add(assay_id, track)

        return final_prediction

    def _describe_tracks(self) -> list:
        """All 21,947 tracks: 21,907 chromatin profiles plus the 40 projected sequence classes.

        Reads the packaged annotation files, so this works without the 3.3 GB Zenodo archive — the same
        reason `get_target_names()` falls back to the vendored copy.

        Both kinds are included because `predict()` accepts both, and as of the 2026-08-16 rebuild both
        have background rows. Before that rebuild the 21,907 profiles returned real values whose
        percentile was always None, which is what motivated the `has_background` field.
        """
        from ..core.tracks import TrackRecord
        from .sei_source.annotations import SeiClassesList, SeiTargetList

        out = []
        for tg in SeiTargetList.load(self.get_target_names()).targets.keys():
            out.append(TrackRecord(
                track_id=str(tg), assay=tg.assay, cell_type=tg.celltype,
                description=f"{tg.assay} in {tg.celltype}", extra={"kind": "chromatin_profile"},
            ))
        for cl in SeiClassesList.load(self.get_classes_names()).classes.keys():
            out.append(TrackRecord(
                track_id=str(cl), assay="sequence-class", cell_type=None,
                description=cl.name, extra={"kind": "sequence_class", "group": cl.group},
            ))
        return out

    def _parse_assay_ids(self, assay_ids):
        """Split requested ids into Sei targets and sequence classes, with their model indices.

        Extracted from ``_predict`` so the single-sequence and multi-allele paths cannot drift: the
        pairwise nucleosome correction needs the same id resolution, and duplicating it is how the two
        would end up disagreeing about which column a track is.
        """
        targets_ids, classes_ids, mapping = [], [], {}
        for ind, ai in enumerate(assay_ids):
            if SeiTarget.is_id(ai):
                mapping[ind] = ('t', len(targets_ids))
                targets_ids.append(ai)
            elif SeiClass.is_id(ai):
                mapping[ind] = ('c', len(classes_ids))
                classes_ids.append(ai)
            else:
                raise InvalidAssayError(f"Invalid assay ID: {ai}")

        sei_targets = [SeiTarget.from_str(tai) for tai in targets_ids]
        sei_classes = [SeiClass.from_str(cli) for cli in classes_ids]
        return {
            "mapping": mapping,
            "sei_targets": sei_targets,
            "sei_classes": sei_classes,
            "targets_inds": self._targets2inds(sei_targets),
            "classes_inds": self._cl2ind(sei_classes),
        }

    def _predict(self,
                 seq: Union[str, Tuple[str, int, int], Interval],
                 assay_ids: list[str]) -> OraclePrediction:
        if not assay_ids:
            assay_ids = self._default_assay_ids()
        parsed = self._parse_assay_ids(assay_ids)
        mapping = parsed["mapping"]
        sei_targets, sei_classes = parsed["sei_targets"], parsed["sei_classes"]
        targets_inds, classes_inds = parsed["targets_inds"], parsed["classes_inds"]

       # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise missing_reference_fasta_error(self.oracle_name)
            chrom, start, end = seq
            query_interval = Interval.make(GenomeRef(chrom=chrom, 
                                                     start=start, 
                                                     end=end, 
                                                     fasta=self.reference_fasta))
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
            target_preds, class_preds = self._predict_in_environment(
                seq=full_seq, 
                targets_inds=targets_inds, 
                classes_inds=classes_inds,
                reverse_aug=self.average_reverse)
            
        else:
            target_preds, class_preds = self._predict_direct(
                seq=full_seq, 
                targets_inds=targets_inds, 
                classes_inds=classes_inds,
                reverse_aug=self.average_reverse)            


        return self._assemble_prediction(
            assay_ids=assay_ids, parsed=parsed, target_preds=target_preds, class_preds=class_preds,
            query_interval=query_interval, prediction_interval=prediction_interval,
            input_interval=input_interval)

        
    
    def _predict_alleles_direct(self, seqs, targets_inds, classes_inds, reverse_aug=True):
        """Every allele in-process, with the pairwise histone correction applied before projection."""
        if self._model is None or self._projector is None or self._normalizer is None:
            raise ModelNotLoadedError()

        raws = []
        for s in seqs:
            preds, _ = self._model.seq_sliding_predict(
                seq=s, reverse_aug=reverse_aug, window_size=self.sequence_length,
                step=self.bin_size, batch_size=self.batch_size)
            raws.append(np.asarray(preds))

        raws = self._normalizer.equalize(raws)
        return [(r[:, targets_inds], self._projector(r)[:, classes_inds]) for r in raws]

    def _predict_alleles(self, intervals, assay_ids):
        """Predict every allele together, so the nucleosome-occupancy correction can apply.

        Overrides ``OracleBase._predict_alleles``, whose default predicts each allele independently.
        Sei needs the override because upstream's ``sc_hnorm_varianteffect`` equalises the histone
        totals of the ref/alt **pair** on the raw 21,907-profile vectors before projecting -- a
        correction that cannot be expressed one sequence at a time, and which chorus was not applying
        at all (the normalizer was built, required, and never called; see
        ``tests/test_sei_nucleosome_normalization.py``).

        The result schema is deliberately identical to every other oracle's: one entry per allele
        keyed as the base class keys them, and ``effect = alt - ref`` computed by the base. With more
        than one alt, every allele is equalised to their common histone total, which for ref + 1 alt
        is bit-for-bit upstream.
        """
        names = list(intervals.keys())
        if not assay_ids:
            assay_ids = self._default_assay_ids()
        parsed = self._parse_assay_ids(assay_ids)

        seqs, meta = [], []
        for name in names:
            query_interval = intervals[name]
            input_interval = query_interval.extend(self.sequence_length)
            seqs.append(input_interval.sequence)
            meta.append((query_interval, input_interval))

        if self.use_environment:
            per_allele = self._predict_in_environment(
                targets_inds=parsed["targets_inds"], classes_inds=parsed["classes_inds"],
                reverse_aug=self.average_reverse, seqs=seqs, normalize=True)
        else:
            per_allele = self._predict_alleles_direct(
                seqs, parsed["targets_inds"], parsed["classes_inds"],
                reverse_aug=self.average_reverse)

        out = {}
        for name, (query_interval, input_interval), (tp, cp) in zip(names, meta, per_allele):
            out[name] = self._assemble_prediction(
                assay_ids=assay_ids, parsed=parsed, target_preds=tp, class_preds=cp,
                query_interval=query_interval,
                prediction_interval=query_interval.extend(self.sequence_length),
                input_interval=input_interval)
        return out

    def _predict_in_environment(self,
                                seq: str = None,
                                targets_inds: list[int] = None,
                                classes_inds: list[int] = None,
                                reverse_aug: bool = True,
                                seqs: list = None,
                                normalize: bool = False):
        """One sequence, or every allele of a variant in a single child process.

        Passing ``seqs`` runs the multi-allele path, which exists so the nucleosome-occupancy
        correction can be applied to the raw 21,907-profile vectors *before* projection. Those raw
        vectors never cross the subprocess boundary -- only the selected values do -- so the
        correction has to happen in the child.
        """
 
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'n_genomic_features': self.n_targets,
            'model_weights': str(self.get_model_weights_path()),
            'projector_weights': str(self.get_projector_weights()),
            'n_classes': self.n_classes,
            'targets': str(self.get_target_names()),
            'classes': str(self.get_classes_names()),
            'seq': seq,
            'seqs': seqs,
            'histone_inds': str(self.get_adjustor_params()) if (seqs and normalize) else None,
            'targets_inds': targets_inds,
            'classes_inds': classes_inds,
            'reverse_aug': reverse_aug,
            'batch_size': self.batch_size,
            'bin_size': self.bin_size,
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as arg_file:
            json.dump(args, arg_file)
            arg_file.flush()

            template, arg = self.get_predict_template()
            template = template.replace(arg, arg_file.name)
            model_predictions = self.run_code_in_environment(template, timeout=self.predict_timeout)

            if seqs is not None:
                return [
                    (np.array(d['selected_preds'], dtype=np.float32),
                     np.array(d['selected_classes'], dtype=np.float32))
                    for d in model_predictions['per_allele']
                ]
            selected_preds = np.array(model_predictions['selected_preds'], dtype=np.float32)
            selected_classes = np.array(model_predictions['selected_classes'], dtype=np.float32)
        return selected_preds, selected_classes
        
        
    def _predict_direct(self,
                        seq: str,
                        targets_inds: list[int],
                        classes_inds: list[int],
                        reverse_aug: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Direct prediction in current environment."""

        if self._model is None or self._projector is None or self._normalizer is None:
            raise ModelNotLoadedError()
        
        predictions, _ = self._model.seq_sliding_predict(seq=seq, 
                                                               reverse_aug=reverse_aug,
                                                               window_size=self.sequence_length,
                                                               step=self.bin_size,
                                                               batch_size=self.batch_size)

        class_preds = self._projector(predictions)

        selected_preds = predictions[:, targets_inds]
        selected_classes = class_preds[:, classes_inds]

        return selected_preds, selected_classes 

    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tuning is not supported for Sei.

        Sei's 21,907-class classification head is tied to its training
        vocabulary; fine-tuning on user-supplied tracks would require
        re-engineering the head. Use AlphaGenome or Borzoi for
        workflows that need on-the-fly track adaptation.
        """
        raise NotImplementedError(
            "Sei fine-tuning is not supported. Use AlphaGenome or "
            "Borzoi for workflows that need on-the-fly track adaptation."
        )
    
    def _get_context_size(self) -> int:
        """Return the required context size for the model."""
        return self.sequence_length
    
    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return min and max sequence lengths."""
        # Sei uses MLP-layers in the head so there is no way to pass sequence of other length to it directly
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
        return "https://zenodo.org/record/4906997/files/sei_model.tar.gz"

    def _try_hf_mirror(self, dest_tar: "Path") -> bool:
        """Fetch sei_model.tar.gz from the chorus HF mirror at
        lucapinello/chorus-sei. Returns True on success. On any failure
        (no huggingface_hub, network, repo missing, etc.) returns False
        so the caller can fall back to Zenodo."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.info("huggingface_hub not available; using Zenodo fallback for Sei")
            return False
        try:
            local = hf_hub_download(
                repo_id="lucapinello/chorus-sei",
                filename="sei_model.tar.gz",
                repo_type="model",
            )
            import shutil as _shutil
            _shutil.copyfile(local, dest_tar)
            logger.info("Fetched Sei tarball from chorus HF mirror.")
            return True
        except Exception as exc:
            logger.info(f"chorus-sei HF mirror unavailable ({exc}); using Zenodo fallback")
            return False

    def _download_sei_model(self):
        from pathlib import Path
        import tarfile
        import shutil

        # Create download link
        download_link = self.get_zenodo_link()
        download_path = self.download_dir

        logger.info(f"Downloading Sei model into {download_path}...")

        download_file_path = os.path.join(
            download_path,
            os.path.basename(download_link)
        )

        if not Path(download_file_path).exists():
            # Prefer chorus-controlled HF mirror; fall back to Zenodo
            # on any failure.
            if not self._try_hf_mirror(Path(download_file_path)):
                self._download_with_resume(download_link, download_file_path)
            logger.info("Download completed!")
        else:
            logger.info("Sei model archive is already downloaded!")

        # Now extract the file in the same download folder
        try:
            with tarfile.open(download_file_path, "r:gz") as tar:
                tar.extractall(path=download_path)
        except (tarfile.TarError, EOFError) as e:
            logger.warning(f"Archive appears corrupt ({e}), re-downloading...")
            Path(download_file_path).unlink(missing_ok=True)
            self._download_with_resume(download_link, download_file_path)
            with tarfile.open(download_file_path, "r:gz") as tar:
                tar.extractall(path=download_path)
        logger.info("Sei model downloaded and extracted successfully!")

        # Delegate to the always-run helper so the copy happens whether
        # we got here via a fresh Zenodo download, an existing tarball
        # re-extract, or (on re-runs) the load_pretrained_model fast
        # path that doesn't call this function at all.
        self._materialize_cached_seqclass_info()

    def _materialize_cached_seqclass_info(self) -> None:
        """Ensure ``downloads/sei/model/seqclass_info.txt`` exists by
        copying the packaged copy when the cache file is missing.

        `chorus health` probes this exact path (see
        ``chorus/core/weights_probe.py::_probe_sei``) so if the cache
        isn't materialised, Sei reports "Not installed" even after a
        successful setup. Regression for v23 scorched-earth audit.
        """
        import shutil
        info_file_path = self.get_model_dir_path() / "seqclass_info.txt"
        cached_info = self.download_dir / "model" / "seqclass_info.txt"
        if cached_info.exists():
            return
        if not info_file_path.exists():
            return  # nothing to copy; guard rail
        cached_info.parent.mkdir(parents=True, exist_ok=True)
        if info_file_path.resolve() != cached_info.resolve():
            shutil.copy(info_file_path, cached_info)

    @staticmethod
    def _download_with_resume(url: str, dest: str, chunk_bytes: int = 4 * 1024 * 1024) -> None:
        """Streamed HTTP download with ``Range`` resume + single-flight lock.

        Thin compatibility shim around :func:`chorus.utils.http.download_with_resume`.
        Kept so any external callers of ``SeiOracle._download_with_resume`` keep
        working after the helper moved into the shared utility module.
        """
        from chorus.utils.http import download_with_resume
        download_with_resume(url, dest, chunk_bytes=chunk_bytes, label="Sei download")