"""Sei oracle implementation."""

from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import os 
import json
import logging
from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidAssayError
from ..utils.sequence import extract_sequence_with_padding

from .sei.annotations import SeiClass, SeiTarget, SeiClassesList, SeiTargetList
from .sei.sei_globals import SEI_WINDOW, SEI_STEP, SEI_TARGETS, SEI_CLASSES
from .sei.metainfo import MetaInfoArray, MetaInfoDict

logger = logging.getLogger(__name__)


class SeiOracle(OracleBase):
    """Sei oracle implementation for sequence regulatory activities."""
    
    def __init__(self, 
                 step_size: int = SEI_STEP,
                 sliding_predict: bool = False,
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
        if self.device is None:
            self.device = 'cpu'

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

    def get_model_dir_path(self):
        if self.model_dir is None:
            parent = os.path.dirname(os.path.realpath(__file__))
            self.model_dir = os.path.join(parent, "sei")
        return self.model_dir

    def get_model_weights_path(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "sei.pth")
    
    def get_projector_weights(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "projvec_targets.npy")
    
    def get_adjustor_params(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "histone_inds.npy")
    
    def get_target_names(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "target.names")
    
    def get_classes_names(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "seqclass_info.txt")

    def get_templates_dir(self):
        d = self.get_model_dir_path()
        return os.path.join(d, "templates")
    
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
    
    def load_pretrained_model(self, weights: str | None) -> None:
        """Load Sei model weights."""
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
            'n_genomic_features': self.n_targets,
            'model_weights': self.get_model_weights_path(),
            'projector_weights': self.get_projector_weights(),
            'n_classes': self.n_classes,
            'histone_inds': self.get_adjustor_params(),
            'targets': self.get_target_names(),
            'classes': self.get_classes_names()
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
                raise ModelNotLoadedError("Failed to load Sei model in environment")
    
    def _load_direct(self):
        try:
            import torch 
            from .sei.sei import Sei, SeiProjector, SeiNormalizer
            from .sei.annotations import SeiClassesList, SeiTargetList

            device = torch.device(self.device)
            model = Sei(sequence_length=self.sequence_length, n_genomic_features=self.n_targets)
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
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load Enformer model: {str(e)}")

    
    def list_assay_types(self) -> List[str]:
        """Return Sei's assay types."""
        if self._model_info is not None: # model is loaded through environment 
            return self._model_info['assays']
        elif self._target_list is not None: # model is loaded in current environment
            return self._target_list.list_assay_types()
        else:
            from .sei.annotations import SeiTargetList
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
            classes = SeiClassesList.load(self.get_classes_names())
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
    
    def _validate_assay_ids(self, assay_ids: List[str]):
        available_assay_ids = set(self._get_all_assay_ids())
        for ai in assay_ids:
            if ai not in available_assay_ids:
                raise InvalidAssayError(f"Invalid assay ID: {ai}")

    def _refine_total_length(self, total_length: int) -> int:
        if not self.sliding_predict:
            return self.sequence_length

        div, mod = divmod(total_length, self.bin_size)
        total_length = div * self.bin_size + self.bin_size * (mod > 0)
        return total_length

    def predict(
        self,
        input_data: Union[str, Tuple[str, int, int]],
        assay_ids: List[str] | None = None,
        create_tracks: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict regulatory activity for a sequence or genomic region.
        
        Args:
            input_data: Either a DNA sequence string or a tuple of (chrom, start, end)
            assay_ids: List of assay identifiers (e.g., ['ENCFF413AHU'] or ['DNase:K562'])
            create_tracks: Whether to create track files (not implemented yet)
            
        Returns:
            Dictionary mapping assay IDs to prediction arrays
            
        Example:
            >>> # Using sequence
            >>> predictions = oracle.predict('ACGT...', ['DNase:K562'])
            >>> 
            >>> # Using genomic coordinates (requires reference_fasta)
            >>> predictions = oracle.predict(('chrX', 48780505, 48785229), ['ENCFF413AHU'])
        """
        
        # Validate inputs
        self._validate_loaded()
        if assay_ids is None:
            assay_ids = self._get_all_assay_ids()
        else:
            self._validate_assay_ids(assay_ids)
        
        # Get raw predictions
        predictions = self._predict(input_data, assay_ids)
        
        # Return as dictionary
        result = {}
        for i, assay_id in enumerate(assay_ids):
            result[assay_id] = predictions[:, i]
        return MetaInfoDict(result, metainfo={'positions': predictions.metainfo['positions']})
    
    def _predict(self,
                 seq: Union[str, Tuple[str, int, int]],
                 assay_ids: list[str]) -> MetaInfoArray:
        targets_ids = []
        classes_ids = []
        mapping = {}
        
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
        
        targets_inds = self._targets2inds(sei_targets)
        classes_inds = self._cl2ind(sei_classes)

        # Handle genomic coordinates
        if isinstance(seq, tuple):
            if self.reference_fasta is None:
                raise ValueError("Reference FASTA required for genomic coordinate input")
            chrom, start, end = seq

            # Extract sequence with padding from reference
            total_length =  self._refine_total_length(end - start)
            full_seq = extract_sequence_with_padding(
                self.reference_fasta,
                chrom,
                start,
                end,
                total_length=total_length
            )

            center = (start + end) // 2 
            real_start = center - total_length // 2
        else:
            full_seq = seq        
            real_start = 0
        
        if self.use_environment:
            target_preds, class_preds, offsets = self._predict_in_environment(
                seq=full_seq, 
                targets_inds=targets_inds, 
                classes_inds=classes_inds,
                reverse_aug=self.average_reverse)
            
        else:
            target_preds, class_preds, offsets = self._predict_direct(
                seq=full_seq, 
                targets_inds=targets_inds, 
                classes_inds=classes_inds,
                reverse_aug=self.average_reverse)

        positions = real_start + offsets 
        n_bins = target_preds.shape[0]             
        joined_preds = np.zeros((n_bins, len(assay_ids)), dtype=np.float32)
        for ind in range(0, joined_preds.shape[1]):
            if mapping[ind] is not None:
                source, source_ind = mapping[ind]
                if source == 't':
                    joined_preds[:, ind] = target_preds[:, source_ind]
                elif source == 'c':
                    joined_preds[:, ind] = class_preds[:, source_ind]
            else:
                joined_preds[:, ind] = np.nan

        return MetaInfoArray(joined_preds, metainfo={'positions': positions})
        
    
    def _predict_in_environment(self,
                                seq: str,
                                targets_inds: list[int],
                                classes_inds: list[int],
                                reverse_aug: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 
        args = {
            'device': self.device,
            'sequence_length': self.sequence_length,
            'n_genomic_features': self.n_targets,
            'model_weights': self.get_model_weights_path(),
            'projector_weights': self.get_projector_weights(),
            'n_classes': self.n_classes,
            'targets': self.get_target_names(),
            'classes': self.get_classes_names(),
            'seq': seq,
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
            model_predictions = self.run_code_in_environment(template, timeout=self.model_load_timeout)
            
            selected_preds = np.array(model_predictions['selected_preds'], dtype=np.float32)
            selected_classes = np.array(model_predictions['selected_classes'], dtype=np.float32)
            offsets = np.array(model_predictions['offsets'], dtype=np.int64)
        return selected_preds, selected_classes, offsets
        
        
    def _predict_direct(self,
                        seq: str,
                        targets_inds: list[int],
                        classes_inds: list[int],
                        reverse_aug: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct prediction in current environment."""

        if self._model is None or self._projector is None or self._normalizer is None:
            raise ModelNotLoadedError()
        
        predictions, offsets = self._model.seq_sliding_predict(seq=seq, 
                                                               reverse_aug=reverse_aug,
                                                               window_size=self.sequence_length,
                                                               step=self.bin_size,
                                                               batch_size=self.batch_size)

        class_preds = self._projector(predictions)

        selected_preds = predictions[:, targets_inds]
        selected_classes = class_preds[:, classes_inds]

        return selected_preds, selected_classes, offsets 

    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tune Sei on new tracks."""
        # TODO: for now we decided not to implement this functionality
        raise NotImplementedError("Sei fine-tuning not yet implemented")
    
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

    def predict_region_replacement(
        self,
        genomic_region: Union[str, pd.DataFrame],
        seq: str,
        assay_ids: List[str],
        create_tracks: bool = False,
        genome: str | None = None
    ) -> MetaInfoDict:
        dt = super().predict_region_replacement(genomic_region, seq, assay_ids, create_tracks, genome)
        
        try:
            val = next(iter(dt['raw_predictions'].values()))
            metainfo = {'positions': val.metainfo['positions']}
        except StopIteration:
            metainfo = {}

        return MetaInfoDict(dt, metainfo=metainfo)

    def predict_variant_effect(self, 
                               genomic_region: Union[str, pd.DataFrame], 
                               variant_position: Union[str, pd.DataFrame], 
                               alleles: Union[List[str], pd.DataFrame],
                               assay_ids: List[str], 
                               create_tracks: bool = False,
                               genome: str | None = None) -> Dict:
        # TODO: add histone correction for variant effect prediction
        return super().predict_variant_effect(genomic_region, variant_position, alleles, assay_ids, create_tracks, genome)
