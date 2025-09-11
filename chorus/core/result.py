from dataclasses import dataclass, field
import numpy as np 
import logging 
from typing import Any 
from copy import deepcopy

from ..core.new_interval import Interval, GenomeRef


logger = logging.getLogger(__name__)

@dataclass
class OraclePredictionTrack:
    source_model: str = field(repr=True)
    assay_id: str = field(repr=True)
    assay_type: str = field(repr=True) # experiment type
    cell_type: str = field(repr=True)# celltype or cell line
    query_interval: Interval = field(repr=True)
    prediction_interval: Interval = field(repr=True)
    input_interval: Interval = field(repr=False)
    resolution: int = field(repr=True) # bin size, resolutions can be different for different tracks
    values: np.ndarray = field(repr=False)
    preferred_aggregation : str = field(repr=False)
    preferred_deconvolution: str = field(repr=False)
    preferred_scoring_strategy: str = field(repr=False)
    metadata: dict = field(repr=False, default_factory=dict)

    def aggregate(self, target_resolution: int, aggregation_strategy: str | None = None): 
        """
        decrease track resolution using specified aggregation strategy
        if target resolution equals to -1 - provide a single value 
        """
        if aggregation_strategy is None:
            aggregation_strategy = self.preferred_aggregation
        raise NotImplementedError()

    def deconvolve(self, target_resolution: int, deconvolution_strategy: str | None = None):
        """
        increase track resolution using specified aggregation strategy 
        if target resolution equals to -1 - provide a single value 
        """
        if deconvolution_strategy is None:
            deconvolution_strategy = self.preferred_deconvolution
        raise NotImplementedError

    def score(self, scoring_strategy: str | None = None) -> float:
        """
        Convert prediction to single value used for variant effect prediction and other tasks
        """
        if scoring_strategy is None:
            scoring_strategy = self.preferred_scoring_strategy

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
    def start(self) -> str:
        return min(track.start for track in self.tracks.values())

    @property
    def end(self) -> str:
        return max(track.start for track in self.tracks.values())

    def add(self, assay_id: str, track: OraclePredictionTrack):
        if assay_id in self.tracks:
            raise Exception("The following assay_id already exists: {assay_id}")
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

    def subset(self, track_ids: str) -> 'OraclePrediction':
        selected = {ti: self[ti] for ti in track_ids}
        return OraclePrediction(selected)

    def save_predictions_as_bedgraph(
        self, 
        output_dir: str = ".",
        prefix: str = "",
        track_colors: dict[str, str] | None = None) -> dict[str, str]:
        """Save predictions as BedGraph files.
        
        Args:
            output_dir: Directory to save files
            prefix: Prefix for filenames
            track_colors: Optional dictionary of track_id -> color
            
        Returns:
            List of saved file paths
        """
        from pathlib import Path
        from .track import Track
        import pandas as pd


        example_track = list(self.tracks.values())[0]
        if not isinstance(example_track.prediction_interval.reference, GenomeRef):
            raise NotImplementedError("For now chorus can't save predictions for non-genome reference intervals")


        # Default colors for different assay types
        default_colors = {
            'DNASE': '#1f77b4',  # Blue
            'ATAC': '#2ca02c',   # Green  
            'CAGE': '#ff7f0e',   # Orange
            'CHIP': '#d62728',   # Red
            'RNA': '#9467bd'     # Purple
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for track_id, track in self.tracks.items():
            
            # Create track data
            track_data = []
            for i, value in enumerate(track.values):
                bin_start = track.prediction_interval.reference.start + i * track.resolution
                bin_end = bin_start + track.resolution
                track_data.append({
                    'chrom': track.prediction_interval.reference.chrom,
                    'start': bin_start,
                    'end': bin_end,
                    'value': float(value)
                })
            
            df = pd.DataFrame(track_data)
            
            # Determine color
            if track_colors and track_id in track_colors:
                color = track_colors[track_id]
            else:
                # Use default color based on assay type
                color = default_colors.get(track.assay_type.upper(), '#000000')
            
            # Create Track object
            track = Track(
                name=f"{prefix}_{track_id}" if prefix else track_id,
                assay_type=track.assay_type,
                cell_type=track.cell_type,
                data=df,
                color=color
            )
            
            # Save to file
            clean_id = track_id.replace(':', '_')
            filename = f"{prefix}_{clean_id}.bedgraph" if prefix else f"{clean_id}.bedgraph"
            filepath = output_dir / filename
            
            track.to_bedgraph(str(filepath))
            saved_files[track_id] = str(filepath)
            
        return saved_files

    def normalize(self, normalization: str = "minmax"):
        norm_tracks = {track_id: track.normalize(normalization) for track_id, track in self.tracks.items()}
        return OraclePrediction(norm_tracks)
        

def minmax(arr: np.ndarray):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return arr


def analyze_gene_expression(predictions: OraclePrediction, 
                            gene_name: str, 
                            #chrom: str, start: int, end: int,
                            gtf_file: str,
                            cage_track_ids: list[str] | None = None,
                            cage_window_bin_size: int = 5) -> dict[str, Any]:
        """Analyze predicted gene expression using CAGE signal at TSS.
        
        For now, we analyze gene expression by looking at CAGE signal
        around the transcription start sites (TSS) of the gene.
        
        Args:
            predictions: Dictionary of track predictions
            gene_name: Name of the gene to analyze
            chrom: Chromosome of the predicted region
            start: Start of the predicted region
            end: End of the predicted region  
            gtf_file: Path to GTF file with gene annotations
            cage_track_ids: List of CAGE track IDs to analyze
                          If None, uses all CAGE tracks in predictions
                          
        Returns:
            Dictionary with gene expression analysis:
            - tss_positions: List of TSS positions
            - cage_signals: Dict of track_id -> signals at each TSS
            - mean_expression: Dict of track_id -> mean expression
            - max_expression: Dict of track_id -> max expression
            
        Note:
            For Borzoi, we also need to sum RNA-seq signal over coding exons
            as described in their paper, but Enformer doesn't have RNA-seq tracks.
        """
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
        if cage_track_ids is None:
            cage_track_ids = [
                track_id for track_id, track in predictions.items() if track.assay_id == 'CAGE'
            ]
        predictions = predictions.subset(cage_track_ids)

        # Analyze CAGE signal at TSS positions
        cage_signals = {}
        mean_expression = {}
        max_expression = {}
        
        any_tss = False
        for track_id, track in predictions.items():
            
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