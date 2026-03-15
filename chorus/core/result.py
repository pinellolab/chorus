from dataclasses import dataclass, field, asdict
from dataclasses import replace as dt_replace
import numpy as np 
import pandas as pd
import logging
import re
import tempfile
import shutil
import weakref
from pathlib import Path

from copy import deepcopy

from ..core.interval import Interval, GenomeRef, CigarEqual, CigarNotEqual, CigarInsertion
from ..core.aggregation import Aggregation
from ..core.interpolation import Interpolation
from ..core.delete_registry import DeleteOnExitRegistry

from typing import Type

logger = logging.getLogger(__name__)


from typing import ClassVar, Any
from threading import RLock

def default_track_visualization_params() -> dict:
    coolbox_params = {
            'threshold_color': 'lightblue',
            'height': 8,
            'color': 'black',
            'fontsize': 15,
            'highlight_color': 'green',
            'vline_width': 2}
    return coolbox_params


def modify_dict(dt: dict, **kwargs: Any) -> dict:
    dt = deepcopy(dt)
    for key, value in kwargs.items():
        dt[key] = value
    return dt



@dataclass
class OraclePredictionTrack:
    _registry: ClassVar[dict[str, 'OraclePredictionTrack']] = {}
    _lock: ClassVar[RLock] = RLock()

    source_model: str = field(repr=True)
    assay_id: str = field(repr=True)
    assay_type: str = field(repr=True) # experiment type
    cell_type: str = field(repr=True)# celltype or cell line
    query_interval: Interval = field(repr=True)
    prediction_interval: Interval = field(repr=True)
    input_interval: Interval = field(repr=False)
    resolution: int = field(repr=True) # bin size, resolutions can be different for different tracks
    values: np.ndarray = field(repr=False)
    preferred_aggregation : str = field(repr=False, default='sum')
    preferred_interpolation : str = field(repr=False, default='linear_divided')
    preferred_scoring_strategy: str = field(repr=False, default='mean')
    metadata: dict = field(repr=False, default_factory=dict)
    track_id: int | None = field(repr=True, default=None)
    coolbox_params: ClassVar[dict] = default_track_visualization_params()
    PREFIX: ClassVar[str] = 'track-'
    COOLBOX_FILE_NAME: ClassVar[str] = 'coolbox.bedgraph'
    ASSAY_TYPE_COLUMN: ClassVar[str] = 'assay_type'

    def __init_subclass__(cls, *, register: bool = True, name: str | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if not register:
            return  # allow abstract/templates to skip registration

        if name is None:
            raise TypeError(
                f"{cls.__name__}: name must be passed using name argument when subclassing) for registration"
            )
        cls.name = name 

        with OraclePredictionTrack._lock:
            if name in OraclePredictionTrack._registry and OraclePredictionTrack._registry[name] is not cls:
                raise ValueError(
                    f"Duplicate registration for '{name}': "
                    f"{OraclePredictionTrack._registry[name].__name__} already registered"
                )
            OraclePredictionTrack._registry[name] = cls

    def __post_init__(self):
        td = tempfile.TemporaryDirectory(prefix=self.PREFIX)
        self._storage = Path(td.name) # directory to store files associated with the track (e.g for )
        DeleteOnExitRegistry().register(td)


    @classmethod
    def create(cls, cls_name: str | None = None, **kwargs: Any) -> "OraclePredictionTrack":
        if cls_name is None:
            cls_name = kwargs.get(cls.ASSAY_TYPE_COLUMN, None)
        if cls_name is not None:
            try:
                impl = OraclePredictionTrack._registry[cls_name]
            except KeyError as e:
                available = ", ".join(sorted(OraclePredictionTrack._registry))
                logger.warning(f"Unknown implementation '{cls_name}'. Available: {available}")
                impl = cls
        else:
            impl = cls 
        return impl(**kwargs) 

    @classmethod
    def registered(cls) -> tuple[str, ...]:
        return tuple(sorted(OraclePredictionTrack._registry))

    def score(self, scoring_strategy: str | None = None) -> float:
        """
        Convert prediction to single value used for variant effect prediction and other tasks
        """
        if scoring_strategy is None:
            scoring_strategy = self.preferred_scoring_strategy

        if scoring_strategy == 'mean':
            return float(np.mean(self.values))
        elif scoring_strategy == 'max':
            return float(np.max(self.values))
        elif scoring_strategy == 'sum':
            return float(np.sum(self.values))
        elif scoring_strategy == 'median':
            return float(np.median(self.values))
        else:
            raise ValueError(f"Unknown scoring strategy: {scoring_strategy}")

    def score_region(self, chrom: str, start: int, end: int,
                     scoring_strategy: str | None = None) -> float | None:
        """Score prediction values within a genomic sub-region.

        Converts genomic coords to bin indices, slices the values array,
        and applies the scoring strategy (mean/max/sum/median).

        Returns None if the region does not overlap the prediction window.
        """
        if chrom != self.prediction_interval.reference.chrom:
            return None

        pred_start = self.prediction_interval.reference.start
        pred_end = self.prediction_interval.reference.end

        # No overlap
        if start >= pred_end or end <= pred_start:
            return None

        # Clamp to prediction window
        clamped_start = max(start, pred_start)
        clamped_end = min(end, pred_end)

        start_bin = (clamped_start - pred_start) // self.resolution
        end_bin = (clamped_end - pred_start + self.resolution - 1) // self.resolution
        end_bin = min(end_bin, len(self.values))
        start_bin = min(start_bin, end_bin)

        if start_bin >= end_bin:
            return None

        region_values = self.values[start_bin:end_bin]

        if scoring_strategy is None:
            scoring_strategy = self.preferred_scoring_strategy

        if scoring_strategy == 'mean':
            return float(np.mean(region_values))
        elif scoring_strategy == 'max':
            return float(np.max(region_values))
        elif scoring_strategy == 'sum':
            return float(np.sum(region_values))
        elif scoring_strategy == 'median':
            return float(np.median(region_values))
        else:
            raise ValueError(f"Unknown scoring strategy: {scoring_strategy}")

    def pos2bin(self, chrom: str, position: int) -> int | None:
        if chrom != self.prediction_interval.reference.chrom:
            return None
        if position < self.prediction_interval.reference.start:
            return None
        if position > self.prediction_interval.reference.end:
            return None

        return (position - self.prediction_interval.reference.start) // self.resolution

    @property
    def positions(self) ->np.ndarray[int]:
        return np.arange(self.values.shape[0]) * self.resolution + self.prediction_interval.reference.start

    def normalize(self, normalization: str = "minmax"):
        other = deepcopy(self)
        if normalization == 'minmax':
            other.values = minmax(other.values)
        else:
            raise NotImplementedError(normalization)
        return other
    
    @property
    def chrom(self) -> str:
        return self.prediction_interval.reference.chrom
    
    @property
    def start(self) -> int:
        return self.prediction_interval.reference.start
    
    @property
    def end(self) -> int:
        return self.prediction_interval.reference.end       
    
    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __getitem__(self, item) -> np.ndarray:
        return self.values[item]

    def qclamp(self, lower_q: float | None = None, upper_q: float | None = None) -> 'OraclePredictionTrack':
        # TODO: check correctness of qclamp
        orig_v = self.values
        v = np.copy(orig_v)
        if upper_q is not None:
            q = np.quantile(orig_v, upper_q)
            v[v > q] = q 
        if lower_q is not None:
            q = np.quantile(orig_v, lower_q)
            v[v < q] = q 
        return dt_replace(self, values=v)

    def clamp(self, lower: float | None = None, upper: float | None = None) -> 'OraclePredictionTrack':
        other = self.copy()
        other.values[other.values < lower] = lower
        other.values[other.values > upper] = upper
        return other

    def exp(self) -> 'OraclePredictionTrack':
        other = self.copy()
        other.values = np.exp(other.values)
        return other

    def to_dataframe(self, use_reference_interval: bool = True) -> pd.DataFrame:
        track_data = []
        if not use_reference_interval:
            raise NotImplementedError("For now chorus can't store predictions for non-genome reference intervals")


        for i, value in enumerate(self.values):
            bin_start = self.prediction_interval.reference.start + i * self.resolution
            bin_end = bin_start + self.resolution
            track_data.append({
                'chrom': self.prediction_interval.reference.chrom,
                'start': bin_start,
                'end': bin_end,
                'value': float(value)
            })
        return pd.DataFrame(track_data)

    def save_as_bedgraph(self, filepath: str, write_header: bool = True):
        from .track import Track
        df = self.to_dataframe(use_reference_interval=True)
        track = Track(
                name=f"{self.source_model}_{self.assay_id}",
                assay_type=self.assay_type,
                cell_type=self.cell_type,
                data=df,
                color=self.coolbox_params['color']
        )
        track.to_bedgraph(str(filepath), write_header=write_header)

    def get_coolbox_representation(self, 
                                   title: str | None = None,
                                   override_params: dict | None = None, 
                                   signal_threshold: float | None = None, 
                                   add_xaxis: bool = True,
                                   add_highlight: bool = False,
                                   add_vlines: bool = True):
        from coolbox.api import (
            BedGraph,
            TrackHeight,
            Color,
            MinValue,
            MaxValue,
            Title,
            XAxis,
            Vlines,
            HighLights,
        )
        if override_params is None:
            coolbox_params = self.coolbox_params
        else:
            coolbox_params = modify_dict(self.coolbox_params, **override_params)
        if title is None:
            title = self.assay_id
        df = self.to_dataframe(use_reference_interval=True)
        coolbox_file = self._storage / self.COOLBOX_FILE_NAME
        df.to_csv(coolbox_file, sep='\t', index=False, header=False)
        if signal_threshold is None:
            signal_threshold = df['value'].max() + 0.1 
      
        frame = BedGraph(str(coolbox_file),  threshold=signal_threshold, threshold_color=coolbox_params['threshold_color']) +\
            TrackHeight(coolbox_params['height']) +\
            Color(coolbox_params['color']) +\
            MinValue(min(df['value'].min(), 0) ) +\
            MaxValue(df['value'].max()) +\
            Title(title)
       
        if add_vlines:
            frame += Vlines([self.query_interval.to_string()], line_width=coolbox_params['vline_width'])
        if add_highlight:
            frame += HighLights([self.query_interval.to_string()], color=coolbox_params['highlight_color'])
        if add_xaxis:
            frame += XAxis(fontsize=coolbox_params['fontsize'])
        return frame

    def promote(self) -> 'OraclePredictionTrack':
        kwargs = asdict(self)
        promoted = OraclePredictionTrack.create(cls_name=self.ASSAY_TYPE_COLUMN, **kwargs)
        return promoted

    def copy(self) -> 'OraclePredictionTrack':
        other = deepcopy(self)
        other.__post_init__()
        return other

    def rescale(self, 
                resolution: int, 
                interpolation: str | Type[Interpolation] | None = None, 
                aggregation: str | Type[Aggregation] | None = None) -> 'OraclePredictionTrack':
        if resolution == self.resolution:
            return self.copy()
        elif resolution > self.resolution:
            return self.aggregate(resolution=resolution, aggregation=aggregation, interpolation=interpolation)
        else: #resolution < self.resolution
            return self.interpolate(resolution=resolution, method=interpolation)

    def interpolate(self, 
                    resolution: int,
                    method: str | Type[Interpolation] | None = None) -> 'OraclePredictionTrack':
        '''
        Interpolate track to a new resolution.
        '''
        assert resolution < self.resolution, "Target resolution must be smaller than the initial one"

        if method is None:
            method = self.preferred_interpolation
        if isinstance(method, str):
            interp = Interpolation.from_string(method) 
        else:
            interp = method()
        interp.fit(resolution=self.resolution, values=self.values)

        new_vals = interp.predict(resolution=resolution, interval_end=len(self.prediction_interval))

        other = self.copy()
        other.resolution = resolution
        other.values = new_vals
        return other 

    def aggregate(self, 
                resolution: int,
                aggregation: str | Type[Aggregation] | None = None,
                interpolation: str | Type[Interpolation] | None = None) -> 'OraclePredictionTrack':
        '''
        Change track resolution. 
        Uses intermediate interpolation when the target resolution is not an exact multiple of the initial one.
        '''
        assert resolution > self.resolution, "Target resolution must be greater than the initial one"
        if aggregation is None:
            aggregation = self.preferred_aggregation
        if interpolation is None:
            interpolation = self.preferred_interpolation

        if isinstance(aggregation, str):
            aggregation = Aggregation.from_string(aggregation)
        else:
            aggregation = aggregation()
        
        new_vals = aggregation.aggregate(values=self.values,
            resolution=self.resolution,
            new_resolution=resolution, 
            interpolation=interpolation)
        
        other = self.copy()
        other.resolution = resolution
        other.values = new_vals
        return other 

    def calc_ref_values(self, 
                aggregation: str | Aggregation | None = None, 
                interpolation: str| Interpolation | None = None):

        if aggregation is None:
            aggregation = self.preferred_aggregation
        if isinstance(aggregation, str):
            aggregation = Aggregation.from_string(aggregation)
        else:
            aggregation = aggregation()
        if interpolation is None:
            interpolation = self.preferred_interpolation
        if isinstance(interpolation, str):
            interpolation = Interpolation.from_string(interpolation) 
        else:
            interpolation = interpolation()
        
        values = self.values
        interpolation.fit(self.resolution, values)
        
        query_length = len(self.prediction_interval)
        
        q_values = interpolation.predict(1, query_length )
        
        
        ref_start = ref_end = 0
        q_start = q_end = 0 
        
        reference_length = sum(len(ci) for ci in self.prediction_interval.cigar if ci.consumes_ref)
        ref_values = np.zeros(reference_length, dtype=np.float32)
        
        for ci in self.prediction_interval.cigar:
            if ci.consumes_ref:
                ref_end = ref_start + len(ci)
            if ci.consumes_query:
                q_end = q_start + len(ci)
            if isinstance(ci, (CigarEqual, CigarNotEqual)):
                ref_values[ref_start:ref_end] += q_values[q_start:q_end]
            elif isinstance(ci, CigarInsertion):
                cov = aggregation.aggregation_func(q_values[q_start:q_end])
                if ref_start + 1 == reference_length:
                    ref_values[ref_start] += cov
                else:
                    ref_values[ref_start] += cov / 2
                    ref_values[ref_start+1] += cov / 2
            else: # missing reference values 
                ref_values[ref_start:ref_end] = np.nan
        
            ref_start = ref_end
            q_start = q_end
        known_coords = np.where(~np.isnan(ref_values))[0]
        interp_f = interpolation.interpolation_func(coords=known_coords, values=ref_values[known_coords])
        unknown_coords = np.where(np.isnan(ref_values))[0]
        ref_values[unknown_coords] = interp_f(unknown_coords)

        ref_values = aggregation.aggregate(ref_values, 1, self.resolution) # no need to pass interpolation as resolution is equal to 1 
        return ref_values



class DNaseOraclePredictionTrack(OraclePredictionTrack, name='DNASE'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(), 
                                         color='#1f77b4')

class ATACOraclePredictionTrack(OraclePredictionTrack, name='ATAC'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(), 
                                         color='#2ca02c')
    
class CAGEOraclePredictionTrack(OraclePredictionTrack, name='CAGE'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(), 
                                         color='#ff7f0e')
    
class CHIPOraclePredictionTrack(OraclePredictionTrack, name='CHIP'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(), 
                                         color='#d62728')
    
class RNAOraclePredictionTrack(OraclePredictionTrack, name='RNA'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(), 
                                         color='#9467bd')

class LentiMPRAOraclePredictionTrack(OraclePredictionTrack, name='LentiMPRA'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(),
                                         color='#ff7f0e')

class SpliceSitesOraclePredictionTrack(OraclePredictionTrack, name='SPLICE_SITES'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(),
                                         color='#8c564b')

class ProCapOraclePredictionTrack(OraclePredictionTrack, name='PRO_CAP'):
    coolbox_params: ClassVar[dict] = modify_dict(default_track_visualization_params(),
                                         color='#e377c2')

@dataclass
class OraclePrediction:
    tracks: dict[str, OraclePredictionTrack] = field(default_factory=dict)

    @property
    def chrom(self) -> str:
        chroms = []
        for track in self.tracks.values():
            chroms.append(track.chrom)
        chroms = list(set(chroms))
        if len(chroms) != 1:
            raise NotImplementedError("For now chorus can't store predictions for different chromosomes in the same oracleprediction object")
        return chroms[0]
    
    @property
    def start(self) -> int:
        return min(track.start for track in self.tracks.values())

    @property
    def end(self) -> int:
        return max(track.end for track in self.tracks.values())

    def add(self, assay_id: str, track: OraclePredictionTrack):
        if assay_id in self.tracks:
            raise Exception(f"The following assay_id already exists: {assay_id}")
        self.tracks[assay_id] = track

    def __iter__(self):
        return self.tracks.__iter__()

    def __getitem__(self, assay_id: str):
        return self.tracks[assay_id]

    def items(self):
        return self.tracks.items()
    
    def keys(self):
        return self.tracks.keys()

    def values(self):
        return self.tracks.values()

    def subset(self, track_ids: list[str]) -> 'OraclePrediction':
        selected = {ti: self[ti] for ti in track_ids}
        return OraclePrediction(selected)

    def score_region(self, chrom: str, start: int, end: int,
                     scoring_strategy: str | None = None) -> dict[str, float | None]:
        """Score all tracks within a genomic sub-region.

        Returns a dict mapping assay_id to the score (or None if no overlap).
        """
        return {
            assay_id: track.score_region(chrom, start, end, scoring_strategy)
            for assay_id, track in self.tracks.items()
        }

    def save_predictions_as_bedgraph(
        self, 
        output_dir: str = ".",
        prefix: str = "") -> dict[str, str]:
        """Save predictions as BedGraph files.
        
        Args:
            output_dir: Directory to save files
            prefix: Prefix for filenames
            track_colors: Optional dictionary of track_id -> color
            
        Returns:
            List of saved file paths
        """
        from pathlib import Path


        example_track = list(self.tracks.values())[0]
        if not isinstance(example_track.prediction_interval.reference, GenomeRef):
            raise NotImplementedError("For now chorus can't save predictions for non-genome reference intervals")


        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for track_id, track in self.tracks.items():
            # Create track data
            
            # Save to file
            clean_id = re.sub(r'[/:*?"<>|.\s]+', '_', track_id).strip('_')
            filename = f"{prefix}_{clean_id}.bedgraph" if prefix else f"{clean_id}.bedgraph"
            filepath = output_dir / filename
            
            track.save_as_bedgraph(str(filepath), write_header=False)
            saved_files[track_id] = str(filepath)
            
        return saved_files

    def normalize(self, normalization: str = "minmax"):
        norm_tracks = {track_id: track.normalize(normalization) for track_id, track in self.tracks.items()}
        return OraclePrediction(norm_tracks)

    def join(self, other: 'OraclePrediction') -> 'OraclePrediction':
        joined = {**self.tracks, **other.tracks}
        return OraclePrediction(joined)

    def q_normalize(self, points_cnt: int = 10_001) -> 'OraclePrediction':
        from ..utils.quantile_normalization import build_support_distr, quantile_map
        support_distr = build_support_distr({track_id: track.values for track_id, track in self.tracks.items()}, points_cnt=points_cnt)
        new_dt = {}
        for k, v in self.tracks.items():
            v = v.copy()
            v.values = quantile_map(v.values, v.values, support_distr)
            new_dt[k] = v
        return OraclePrediction(new_dt)

def minmax(arr: np.ndarray):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return arr

def score_variant_effect(
    variant_result: dict,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    at_variant: bool = False,
    window_bins: int = 1,
    scoring_strategy: str = 'mean',
) -> dict:
    """Score variant effects focused on a sub-region or variant site.

    Two modes:
    - Region-based (chrom/start/end): scores ref and alt predictions in the
      specified region, returns per-allele scores and effect (alt - ref).
    - at_variant=True: extracts ±window_bins around the variant position.

    Args:
        variant_result: Return value of predict_variant_effect().
        chrom: Chromosome for region-based scoring.
        start: Region start for region-based scoring.
        end: Region end for region-based scoring.
        at_variant: If True, score around the variant position.
        window_bins: Bins on each side when at_variant=True.
        scoring_strategy: mean, max, sum, median, or abs_max.

    Returns:
        Dict of {allele_name: {assay_id: {ref_score, alt_score, effect}}}.
    """
    predictions = variant_result['predictions']
    variant_info = variant_result['variant_info']
    ref_pred = predictions['reference']

    # Determine region
    if at_variant:
        # Parse variant position
        pos_str = variant_info['position']
        var_chrom, var_pos = pos_str.split(':')
        var_pos = int(var_pos)

        # Use the first track to determine bin boundaries
        first_track = next(iter(ref_pred.values()))
        var_bin = first_track.pos2bin(var_chrom, var_pos)
        if var_bin is None:
            raise ValueError(f"Variant position {pos_str} is outside the prediction window")

        pred_start = first_track.prediction_interval.reference.start
        resolution = first_track.resolution
        region_start_bin = max(0, var_bin - window_bins)
        region_end_bin = min(len(first_track.values), var_bin + window_bins + 1)
        chrom = var_chrom
        start = pred_start + region_start_bin * resolution
        end = pred_start + region_end_bin * resolution
    elif chrom is None or start is None or end is None:
        raise ValueError("Either provide chrom/start/end or set at_variant=True")

    # Apply abs_max strategy via post-processing
    base_strategy = scoring_strategy if scoring_strategy != 'abs_max' else 'mean'

    # Helper to score a slice of values using the given strategy
    def _score_array(arr: np.ndarray, strategy: str) -> float:
        if strategy == 'mean':
            return float(np.mean(arr))
        elif strategy == 'max':
            return float(np.max(arr))
        elif strategy == 'sum':
            return float(np.sum(arr))
        elif strategy == 'median':
            return float(np.median(arr))
        else:
            raise ValueError(f"Unknown scoring strategy: {strategy}")

    results = {}
    for allele_name, alt_pred in predictions.items():
        if allele_name == 'reference':
            continue
        allele_scores = {}
        for assay_id in ref_pred.keys():
            ref_track = ref_pred[assay_id]
            alt_track = alt_pred[assay_id]

            if at_variant:
                # Compute per-track bin indices to handle mixed resolutions
                # (e.g. AlphaGenome has 1bp DNASE + 128bp histone tracks)
                t_var_bin = ref_track.pos2bin(var_chrom, var_pos)
                if t_var_bin is not None:
                    # Scale window_bins by resolution ratio so genomic window is consistent
                    genomic_window = window_bins * resolution  # window in bp from first track
                    t_window = max(1, genomic_window // ref_track.resolution)
                    t_start = max(0, t_var_bin - t_window)
                    t_end = min(len(ref_track.values), t_var_bin + t_window + 1)
                    ref_vals = ref_track.values[t_start:t_end]
                    alt_vals = alt_track.values[t_start:t_end]
                else:
                    ref_vals = np.array([])
                    alt_vals = np.array([])

                if len(ref_vals) == 0:
                    ref_score = None
                    alt_score = None
                    effect = None
                elif scoring_strategy == 'abs_max':
                    ref_score = _score_array(ref_vals, 'mean')
                    alt_score = _score_array(alt_vals, 'mean')
                    diff = alt_vals - ref_vals
                    effect = float(diff[np.argmax(np.abs(diff))]) if len(diff) > 0 else 0.0
                else:
                    ref_score = _score_array(ref_vals, base_strategy)
                    alt_score = _score_array(alt_vals, base_strategy)
                    effect = alt_score - ref_score
            else:
                ref_score = ref_track.score_region(chrom, start, end, base_strategy)
                alt_score = alt_track.score_region(chrom, start, end, base_strategy)

                if scoring_strategy == 'abs_max':
                    pred_start_coord = ref_track.prediction_interval.reference.start
                    s_bin = (max(start, pred_start_coord) - pred_start_coord) // ref_track.resolution
                    e_bin = min(
                        (min(end, ref_track.prediction_interval.reference.end) - pred_start_coord + ref_track.resolution - 1) // ref_track.resolution,
                        len(ref_track.values)
                    )
                    if s_bin < e_bin:
                        diff = alt_track.values[s_bin:e_bin] - ref_track.values[s_bin:e_bin]
                        effect = float(diff[np.argmax(np.abs(diff))])
                    else:
                        effect = 0.0
                    ref_score = ref_score if ref_score is not None else 0.0
                    alt_score = alt_score if alt_score is not None else 0.0
                else:
                    if ref_score is not None and alt_score is not None:
                        effect = alt_score - ref_score
                    else:
                        effect = None

            allele_scores[assay_id] = {
                'ref_score': ref_score,
                'alt_score': alt_score,
                'effect': effect,
            }
        results[allele_name] = allele_scores

    return results


def analyze_gene_expression(predictions: OraclePrediction,
                            gene_name: str,
                            #chrom: str, start: int, end: int,
                            gtf_file: str,
                            expresion_track_ids: list[str] | None = None,
                            cage_window_bin_size: int = 5) -> dict[str, Any]:
        """Analyze predicted gene expression using CAGE signal at TSS.

        .. deprecated::
            Use ``oracle.analyze_gene_expression()`` instead, which auto-detects
            expression track types and supports both CAGE and RNA-seq quantification.

        Args:
            predictions: Dictionary of track predictions
            gene_name: Name of the gene to analyze
            gtf_file: Path to GTF file with gene annotations
            expresion_track_ids: List of CAGE track IDs to analyze
                          If None, uses all CAGE tracks in predictions
            cage_window_bin_size: Bins around TSS for windowed max

        Returns:
            Dictionary with gene expression analysis
        """
        import warnings
        warnings.warn(
            "analyze_gene_expression() is deprecated. "
            "Use oracle.analyze_gene_expression() instead, which supports "
            "both CAGE and RNA-seq quantification.",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: add support for RNA-seq signal 
        from ..utils.annotations import get_gene_tss
        
        # Get TSS positions for the gene
        tss_df = get_gene_tss(gene_name, annotation=gtf_file)
        
        if len(tss_df) == 0:
            logger.warning(f"No TSS found for gene {gene_name}")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }

        # Identify CAGE tracks if not specified
        if expresion_track_ids is None:
            expresion_track_ids = [
                track_id for track_id, track in predictions.items() if track.assay_id == 'CAGE'
            ]
        predictions = predictions.subset(expresion_track_ids)

        # Analyze CAGE signal at TSS positions
        cage_signals = {}
        mean_expression = {}
        max_expression = {}
        
        any_tss = False
        for track_id, track in predictions.items():
            if not isinstance(track, (CAGEOraclePredictionTrack, LentiMPRAOraclePredictionTrack)):
                logger.warning(f"Track {track_id} is not a CAGE or LentiMPRA track, skipping")
                continue
            
            # Filter TSS positions to those in our region
            tss_in_region = tss_df[
                (tss_df['chrom'] == track.prediction_interval.reference.chrom) &
                (tss_df['tss'] >= track.prediction_interval.reference.start) &
                (tss_df['tss'] <= track.prediction_interval.reference.end)
            ]
            if len(tss_in_region) == 0:
                continue
            else:
                any_tss = True
                
            track_signals = []
            
            for _, tss_info in tss_in_region.iterrows():
                tss_pos = tss_info['tss']
                
                # Convert TSS position to bin index
                tss_bin = track.pos2bin(tss_info['chrom'], tss_pos)

                # Get signal in window around TSS (e.g., +/- 5 bins = +/- 640bp)
                start_bin = max(0, tss_bin - cage_window_bin_size)
                end_bin = min(track.values.shape[0], tss_bin + cage_window_bin_size + 1)
                
                # Take max signal in window (TSS can be somewhat imprecise)
                if start_bin < end_bin:
                    window_signal = track.values[start_bin:end_bin]
                    track_signals.append(np.max(window_signal))
            
            cage_signals[track_id] = track_signals
            
            if track_signals:
                mean_expression[track_id] = np.mean(track_signals)
                max_expression[track_id] = np.max(track_signals)
            else:
                mean_expression[track_id] = 0.0
                max_expression[track_id] = 0.0

        if any_tss == 0:
            logger.warning(f"No TSS for {gene_name} in output window")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }
        
        
        return {
            'gene_name': gene_name,
            'tss_positions': tss_in_region['tss'].tolist(),
            'tss_info': tss_in_region.to_dict('records'),
            'cage_signals': cage_signals,
            'mean_expression': mean_expression,
            'max_expression': max_expression,
            'n_tss': len(tss_in_region)
        }