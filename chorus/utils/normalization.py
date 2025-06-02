"""Normalization functions for genomic tracks."""

import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Union
from scipy import stats


def normalize_tracks(
    tracks_filenames: List[str],
    track_names: List[str],
    normalization: Literal['quantile', 'minmax', 'zscore'] = 'quantile',
    output_suffix: str = '_normalized'
) -> List[str]:
    """
    Normalize track values using specified method.
    
    Args:
        tracks_filenames: List of BEDGraph files
        track_names: Names for each track
        normalization: Normalization method
        output_suffix: Suffix for output files
        
    Returns:
        List of normalized track filenames
    """
    output_files = []
    
    for filename, name in zip(tracks_filenames, track_names):
        # Load track
        track_data = pd.read_csv(
            filename, 
            sep='\t', 
            comment='#',
            names=['chrom', 'start', 'end', 'value'],
            skiprows=lambda x: x == 0 and 'track' in open(filename).readline()
        )
        
        # Apply normalization
        if normalization == 'quantile':
            track_data['value'] = quantile_normalize(track_data['value'])
        elif normalization == 'minmax':
            track_data['value'] = minmax_normalize(track_data['value'])
        elif normalization == 'zscore':
            track_data['value'] = zscore_normalize(track_data['value'])
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Save normalized track
        output_file = filename.replace('.bedgraph', f'{output_suffix}.bedgraph')
        if not output_file.endswith('.bedgraph'):
            output_file += '.bedgraph'
        
        with open(output_file, 'w') as f:
            # Write track header
            f.write(f'track type=bedGraph name="{name}_normalized" ')
            f.write(f'description="{name} ({normalization} normalized)"\n')
            
            # Write data
            track_data.to_csv(f, sep='\t', index=False, header=False, float_format='%.6g')
        
        output_files.append(output_file)
    
    return output_files


def quantile_normalize(values: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Apply quantile normalization.
    
    Args:
        values: Values to normalize
        
    Returns:
        Quantile normalized values
    """
    if isinstance(values, pd.Series):
        # Handle pandas Series
        sorted_values = np.sort(values.dropna())
        ranks = values.rank(method='average', na_option='keep')
        
        # Interpolate normalized values
        normalized = np.interp(
            ranks.fillna(0), 
            np.arange(1, len(sorted_values) + 1), 
            sorted_values
        )
        
        # Restore NaN values
        result = pd.Series(normalized, index=values.index)
        result[values.isna()] = np.nan
        return result
    else:
        # Handle numpy array
        values = np.asarray(values)
        mask = ~np.isnan(values)
        
        if not np.any(mask):
            return values
        
        sorted_values = np.sort(values[mask])
        ranks = stats.rankdata(values[mask], method='average')
        
        normalized = np.full_like(values, np.nan)
        normalized[mask] = np.interp(
            ranks,
            np.arange(1, len(sorted_values) + 1),
            sorted_values
        )
        
        return normalized


def minmax_normalize(values: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Normalize to [0, 1] range.
    
    Args:
        values: Values to normalize
        
    Returns:
        Min-max normalized values
    """
    if isinstance(values, pd.Series):
        min_val = values.min()
        max_val = values.max()
    else:
        values = np.asarray(values)
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
    
    if max_val > min_val:
        return (values - min_val) / (max_val - min_val)
    else:
        # All values are the same
        if isinstance(values, pd.Series):
            return pd.Series(0.5, index=values.index)
        else:
            return np.full_like(values, 0.5)


def zscore_normalize(values: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Apply z-score normalization.
    
    Args:
        values: Values to normalize
        
    Returns:
        Z-score normalized values
    """
    if isinstance(values, pd.Series):
        mean_val = values.mean()
        std_val = values.std()
    else:
        values = np.asarray(values)
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
    
    if std_val > 0:
        return (values - mean_val) / std_val
    else:
        # All values are the same
        if isinstance(values, pd.Series):
            return pd.Series(0.0, index=values.index)
        else:
            return np.zeros_like(values)


def robust_zscore_normalize(
    values: Union[pd.Series, np.ndarray],
    axis: Optional[int] = None
) -> Union[pd.Series, np.ndarray]:
    """
    Apply robust z-score normalization using median and MAD.
    
    Args:
        values: Values to normalize
        axis: Axis along which to normalize (for arrays)
        
    Returns:
        Robust z-score normalized values
    """
    if isinstance(values, pd.Series):
        median_val = values.median()
        mad_val = np.median(np.abs(values - median_val))
    else:
        values = np.asarray(values)
        median_val = np.nanmedian(values, axis=axis, keepdims=True)
        mad_val = np.nanmedian(np.abs(values - median_val), axis=axis, keepdims=True)
    
    # Scale MAD to approximate standard deviation
    mad_val = mad_val * 1.4826
    
    if np.any(mad_val > 0):
        return (values - median_val) / np.where(mad_val > 0, mad_val, 1.0)
    else:
        if isinstance(values, pd.Series):
            return pd.Series(0.0, index=values.index)
        else:
            return np.zeros_like(values)


def log_transform(
    values: Union[pd.Series, np.ndarray],
    pseudocount: float = 1.0,
    base: float = 2.0
) -> Union[pd.Series, np.ndarray]:
    """
    Apply log transformation with pseudocount.
    
    Args:
        values: Values to transform
        pseudocount: Small value to add before log transformation
        base: Logarithm base (2 for log2, np.e for natural log, 10 for log10)
        
    Returns:
        Log-transformed values
    """
    if base == 2:
        return np.log2(values + pseudocount)
    elif base == 10:
        return np.log10(values + pseudocount)
    else:
        return np.log(values + pseudocount) / np.log(base)


def percentile_normalize(
    values: Union[pd.Series, np.ndarray],
    lower_percentile: float = 5,
    upper_percentile: float = 95
) -> Union[pd.Series, np.ndarray]:
    """
    Normalize values to percentile range.
    
    Args:
        values: Values to normalize
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        
    Returns:
        Percentile normalized values
    """
    if isinstance(values, pd.Series):
        lower_val = values.quantile(lower_percentile / 100)
        upper_val = values.quantile(upper_percentile / 100)
        
        # Clip and normalize
        clipped = values.clip(lower=lower_val, upper=upper_val)
    else:
        values = np.asarray(values)
        lower_val = np.nanpercentile(values, lower_percentile)
        upper_val = np.nanpercentile(values, upper_percentile)
        
        # Clip and normalize
        clipped = np.clip(values, lower_val, upper_val)
    
    return minmax_normalize(clipped)


def normalize_by_reference(
    values: Union[pd.Series, np.ndarray],
    reference_values: Union[pd.Series, np.ndarray],
    method: Literal['ratio', 'difference', 'zscore'] = 'ratio'
) -> Union[pd.Series, np.ndarray]:
    """
    Normalize values relative to a reference.
    
    Args:
        values: Values to normalize
        reference_values: Reference values for normalization
        method: Normalization method
        
    Returns:
        Normalized values relative to reference
    """
    if method == 'ratio':
        # Add small pseudocount to avoid division by zero
        return values / (reference_values + 1e-8)
    elif method == 'difference':
        return values - reference_values
    elif method == 'zscore':
        # Normalize both to z-scores first
        values_z = zscore_normalize(values)
        reference_z = zscore_normalize(reference_values)
        return values_z - reference_z
    else:
        raise ValueError(f"Unknown method: {method}")