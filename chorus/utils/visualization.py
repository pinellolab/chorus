"""Visualization utilities for genomic tracks."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def visualize_tracks(
    tracks_filenames: List[str],
    track_names: List[str],
    scales: Optional[List[Tuple[float, float]]] = None,
    colors: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    genomic_region: Optional[str] = None,
    figure_size: Optional[Tuple[float, float]] = None,
    style: str = 'default'
) -> None:
    """
    Visualize multiple tracks in a single figure.
    
    Args:
        tracks_filenames: List of BEDGraph files
        track_names: Names for each track
        scales: Y-axis scales for each track (min, max)
        colors: Colors for each track
        output_file: Save figure to file
        genomic_region: Specific region to plot (chr:start-end)
        figure_size: Figure size (width, height)
        style: Visualization style ('default', 'minimal', 'publication')
    """
    # Set style
    if style == 'minimal':
        plt.style.use('seaborn-v0_8-whitegrid')
    elif style == 'publication':
        plt.style.use('seaborn-v0_8-paper')
    else:
        plt.style.use('default')
    
    # Default colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracks_filenames)))
    
    # Figure size
    if figure_size is None:
        figure_size = (12, 3 * len(tracks_filenames))
    
    # Create subplots
    fig, axes = plt.subplots(
        len(tracks_filenames), 1, 
        figsize=figure_size,
        sharex=True
    )
    
    if len(tracks_filenames) == 1:
        axes = [axes]
    
    # Parse genomic region if provided
    region_chrom, region_start, region_end = None, None, None
    if genomic_region:
        import re
        match = re.match(r'(\w+):(\d+)-(\d+)', genomic_region)
        if match:
            region_chrom = match.group(1)
            region_start = int(match.group(2))
            region_end = int(match.group(3))
    
    # Plot each track
    for i, (filename, name) in enumerate(zip(tracks_filenames, track_names)):
        ax = axes[i]
        
        # Load track data
        track_data = _load_bedgraph(filename)
        
        # Filter by region if specified
        if region_chrom:
            track_data = track_data[
                (track_data['chrom'] == region_chrom) &
                (track_data['start'] < region_end) &
                (track_data['end'] > region_start)
            ]
        
        if len(track_data) == 0:
            ax.text(0.5, 0.5, 'No data in region', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_ylabel(name)
            continue
        
        # Create step plot
        positions = []
        values = []
        
        for _, row in track_data.iterrows():
            positions.extend([row['start'], row['end']])
            values.extend([row['value'], row['value']])
        
        # Plot
        ax.plot(positions, values, color=colors[i], linewidth=1.5)
        ax.fill_between(positions, values, alpha=0.3, color=colors[i])
        
        # Set scale
        if scales and i < len(scales):
            ax.set_ylim(scales[i])
        
        # Styling
        ax.set_ylabel(name, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if style == 'minimal':
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis label on bottom plot
    axes[-1].set_xlabel('Genomic Position', fontsize=12)
    
    # Set title
    if genomic_region:
        fig.suptitle(f'Track Visualization: {genomic_region}', fontsize=14)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_track_heatmap(
    tracks: List[Union[str, pd.DataFrame]],
    track_names: List[str],
    genomic_regions: List[str],
    output_file: Optional[str] = None,
    cmap: str = 'RdBu_r',
    figsize: Optional[Tuple[float, float]] = None,
    normalize_tracks: bool = True,
    cluster_tracks: bool = False,
    cluster_regions: bool = False
) -> None:
    """
    Create a heatmap of multiple tracks across multiple regions.
    
    Args:
        tracks: List of track files or DataFrames
        track_names: Names for each track
        genomic_regions: List of regions to include (chr:start-end)
        output_file: Save figure to file
        cmap: Colormap to use
        figsize: Figure size
        normalize_tracks: Whether to normalize each track
        cluster_tracks: Whether to cluster tracks
        cluster_regions: Whether to cluster regions
    """
    # Load and process track data
    heatmap_data = []
    
    for track in tracks:
        if isinstance(track, str):
            track_data = _load_bedgraph(track)
        else:
            track_data = track
        
        track_values = []
        for region in genomic_regions:
            # Parse region
            import re
            match = re.match(r'(\w+):(\d+)-(\d+)', region)
            if not match:
                continue
            
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            
            # Get values for region
            region_data = track_data[
                (track_data['chrom'] == chrom) &
                (track_data['start'] < end) &
                (track_data['end'] > start)
            ]
            
            if len(region_data) > 0:
                # Average value across region
                track_values.append(region_data['value'].mean())
            else:
                track_values.append(0)
        
        heatmap_data.append(track_values)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=track_names,
        columns=genomic_regions
    )
    
    # Normalize if requested
    if normalize_tracks:
        from .normalization import zscore_normalize
        heatmap_df = heatmap_df.apply(zscore_normalize, axis=1)
    
    # Set figure size
    if figsize is None:
        figsize = (len(genomic_regions) * 0.5 + 2, len(track_names) * 0.5 + 2)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Cluster if requested
    if cluster_tracks or cluster_regions:
        import seaborn as sns
        g = sns.clustermap(
            heatmap_df,
            cmap=cmap,
            center=0 if normalize_tracks else None,
            row_cluster=cluster_tracks,
            col_cluster=cluster_regions,
            cbar_kws={'label': 'Signal' if not normalize_tracks else 'Z-score'},
            figsize=figsize
        )
        
        if output_file:
            g.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            center=0 if normalize_tracks else None,
            cbar_kws={'label': 'Signal' if not normalize_tracks else 'Z-score'},
            xticklabels=True,
            yticklabels=True
        )
        
        plt.title('Track Signal Heatmap')
        plt.xlabel('Genomic Regions')
        plt.ylabel('Tracks')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def plot_track_comparison(
    track1_file: str,
    track2_file: str,
    track1_name: str,
    track2_name: str,
    genomic_region: Optional[str] = None,
    output_file: Optional[str] = None,
    correlation_method: str = 'pearson'
) -> Dict[str, float]:
    """
    Compare two tracks and plot correlation.
    
    Args:
        track1_file: First track file
        track2_file: Second track file
        track1_name: Name of first track
        track2_name: Name of second track
        genomic_region: Specific region to compare
        output_file: Save figure to file
        correlation_method: Method for correlation ('pearson', 'spearman')
        
    Returns:
        Dictionary with correlation statistics
    """
    # Load tracks
    track1_data = _load_bedgraph(track1_file)
    track2_data = _load_bedgraph(track2_file)
    
    # Filter by region if specified
    if genomic_region:
        import re
        match = re.match(r'(\w+):(\d+)-(\d+)', genomic_region)
        if match:
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            
            track1_data = track1_data[
                (track1_data['chrom'] == chrom) &
                (track1_data['start'] < end) &
                (track1_data['end'] > start)
            ]
            track2_data = track2_data[
                (track2_data['chrom'] == chrom) &
                (track2_data['start'] < end) &
                (track2_data['end'] > start)
            ]
    
    # Merge tracks on overlapping intervals
    merged_data = _merge_track_intervals(track1_data, track2_data)
    
    if len(merged_data) == 0:
        print("No overlapping intervals found between tracks")
        return {'correlation': np.nan, 'p_value': np.nan}
    
    # Calculate correlation
    from scipy import stats
    if correlation_method == 'pearson':
        corr, p_value = stats.pearsonr(merged_data['value1'], merged_data['value2'])
    else:
        corr, p_value = stats.spearmanr(merged_data['value1'], merged_data['value2'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(merged_data['value1'], merged_data['value2'], alpha=0.5, s=10)
    ax1.set_xlabel(track1_name)
    ax1.set_ylabel(track2_name)
    ax1.set_title(f'{correlation_method.capitalize()} correlation: {corr:.3f} (p={p_value:.3e})')
    
    # Add regression line
    z = np.polyfit(merged_data['value1'], merged_data['value2'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_data['value1'].min(), merged_data['value1'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    # Hexbin plot for dense data
    ax2.hexbin(merged_data['value1'], merged_data['value2'], gridsize=50, cmap='Blues')
    ax2.set_xlabel(track1_name)
    ax2.set_ylabel(track2_name)
    ax2.set_title('Density Plot')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'n_intervals': len(merged_data)
    }


def _load_bedgraph(filename: str) -> pd.DataFrame:
    """Load BEDGraph file, handling various formats."""
    # Try to detect if there's a header
    with open(filename, 'r') as f:
        first_line = f.readline()
        has_header = first_line.startswith('track') or first_line.startswith('#')
    
    # Load data
    if has_header:
        data = pd.read_csv(
            filename,
            sep='\t',
            comment='#',
            names=['chrom', 'start', 'end', 'value'],
            skiprows=1 if first_line.startswith('track') else 0
        )
    else:
        data = pd.read_csv(
            filename,
            sep='\t',
            names=['chrom', 'start', 'end', 'value']
        )
    
    # Ensure numeric types
    data['start'] = pd.to_numeric(data['start'], errors='coerce')
    data['end'] = pd.to_numeric(data['end'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    
    # Remove invalid rows
    data = data.dropna()
    
    return data


def _merge_track_intervals(track1: pd.DataFrame, track2: pd.DataFrame) -> pd.DataFrame:
    """Merge two tracks based on overlapping intervals."""
    merged_data = []
    
    # Simple approach: for each interval in track1, find overlapping intervals in track2
    for _, interval1 in track1.iterrows():
        overlapping = track2[
            (track2['chrom'] == interval1['chrom']) &
            (track2['start'] < interval1['end']) &
            (track2['end'] > interval1['start'])
        ]
        
        for _, interval2 in overlapping.iterrows():
            # Calculate overlap
            overlap_start = max(interval1['start'], interval2['start'])
            overlap_end = min(interval1['end'], interval2['end'])
            
            merged_data.append({
                'chrom': interval1['chrom'],
                'start': overlap_start,
                'end': overlap_end,
                'value1': interval1['value'],
                'value2': interval2['value']
            })
    
    return pd.DataFrame(merged_data)