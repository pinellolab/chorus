"""Track class for representing genomic tracks."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Track:
    """Represents a genomic track with associated metadata."""
    name: str
    assay_type: str
    cell_type: str
    data: pd.DataFrame  # columns: chrom, start, end, value
    normalization_method: str = "none"
    color: str = "#000000"
    
    def __post_init__(self):
        """Validate track data after initialization."""
        required_columns = ['chrom', 'start', 'end', 'value']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Track data must contain columns: {required_columns}")
        
        # Ensure start and end are integers
        self.data['start'] = self.data['start'].astype(int)
        self.data['end'] = self.data['end'].astype(int)
        
        # Ensure value is numeric
        self.data['value'] = pd.to_numeric(self.data['value'], errors='coerce')
        
        # Remove any rows with NaN values
        self.data = self.data.dropna()
        
        # Sort by chromosome and start position
        self.data = self.data.sort_values(['chrom', 'start'])
    
    def to_bedgraph(self, filepath: str, write_header: bool = False) -> None:
        """Save track as BEDGraph file."""
        with open(filepath, 'w') as f:
            # Write track header
            if write_header:
                f.write(f"track type=bedGraph name=\"{self.name}\" ")
                f.write(f"description=\"{self.assay_type} - {self.cell_type}\" ")
                f.write(f"color={self._color_to_rgb()}\n")
            
            # Write data
            self.data[['chrom', 'start', 'end', 'value']].to_csv(
                f, sep='\t', index=False, header=False, float_format='%.6g'
            )
    
    def normalize(self, method: str = "quantile") -> 'Track':
        """Return a new Track with normalized values."""
        from ..utils.normalization import quantile_normalize, minmax_normalize, zscore_normalize
        
        # Create a copy of the data
        normalized_data = self.data.copy()
        
        # Apply normalization
        if method == "quantile":
            normalized_data['value'] = quantile_normalize(normalized_data['value'])
        elif method == "minmax":
            normalized_data['value'] = minmax_normalize(normalized_data['value'])
        elif method == "zscore":
            normalized_data['value'] = zscore_normalize(normalized_data['value'])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new Track with normalized data
        return Track(
            name=f"{self.name}_normalized",
            assay_type=self.assay_type,
            cell_type=self.cell_type,
            data=normalized_data,
            normalization_method=method,
            color=self.color
        )
    
    def _color_to_rgb(self) -> str:
        """Convert hex color to RGB format for BEDGraph."""
        # Remove # if present
        hex_color = self.color.lstrip('#')
        
        # Convert to RGB
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r},{g},{b}"
        except (ValueError, IndexError):
            return "0,0,0"  # Default to black if color parsing fails
    
    def get_region_values(self, chrom: str, start: int, end: int) -> pd.DataFrame:
        """Get values for a specific genomic region."""
        mask = (
            (self.data['chrom'] == chrom) &
            (self.data['start'] < end) &
            (self.data['end'] > start)
        )
        return self.data[mask].copy()
    
    def aggregate_by_bins(self, bin_size: int) -> 'Track':
        """Aggregate track values into fixed-size bins."""
        aggregated_data = []
        
        for chrom in self.data['chrom'].unique():
            chrom_data = self.data[self.data['chrom'] == chrom]
            
            # Find the range
            min_start = chrom_data['start'].min()
            max_end = chrom_data['end'].max()
            
            # Create bins
            bins = range(min_start, max_end + bin_size, bin_size)
            
            for i in range(len(bins) - 1):
                bin_start = bins[i]
                bin_end = bins[i + 1]
                
                # Find overlapping intervals
                mask = (
                    (chrom_data['start'] < bin_end) &
                    (chrom_data['end'] > bin_start)
                )
                overlapping = chrom_data[mask]
                
                if len(overlapping) > 0:
                    # Calculate weighted average based on overlap
                    weights = []
                    values = []
                    
                    for _, row in overlapping.iterrows():
                        overlap_start = max(row['start'], bin_start)
                        overlap_end = min(row['end'], bin_end)
                        weight = overlap_end - overlap_start
                        weights.append(weight)
                        values.append(row['value'])
                    
                    weights = np.array(weights)
                    values = np.array(values)
                    weighted_avg = np.average(values, weights=weights)
                    
                    aggregated_data.append({
                        'chrom': chrom,
                        'start': bin_start,
                        'end': bin_end,
                        'value': weighted_avg
                    })
        
        # Create new Track with aggregated data
        return Track(
            name=f"{self.name}_binned_{bin_size}bp",
            assay_type=self.assay_type,
            cell_type=self.cell_type,
            data=pd.DataFrame(aggregated_data),
            normalization_method=self.normalization_method,
            color=self.color
        )