"""Visualization utilities for genomic tracks."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import warnings
import os

from ..core.result import OraclePrediction


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


def plot_tracks_with_pygenometracks(
    track_files: List[str],
    chrom: str,
    start: int,
    end: int,
    output_file: str,
    track_config: Optional[Dict[str, Dict]] = None,
    genome_file: Optional[str] = None,
    gtf_file: Optional[str] = None,
    height_ratios: Optional[List[float]] = None,
    width: float = 10,
    dpi: int = 300
) -> bool:
    """
    Plot tracks using pyGenomeTracks if available.
    
    Args:
        track_files: List of BedGraph files
        genomic_region: Region to plot (chr:start-end)
        output_file: Output image file
        track_config: Optional configuration for each track
        genome_file: Optional genome file for gene annotations
        height_ratios: Height ratio for each track
        width: Figure width in inches
        dpi: DPI for output
        
    Returns:
        True if successful, False if pyGenomeTracks not available
    """
    try:
        import pygenometracks.tracks as pygtk
        from pygenometracks.plotTracks import PlotTracks
        import tempfile
        import os
    except ImportError:
        warnings.warn("pyGenomeTracks not installed. Use pip install pyGenomeTracks")
        return False
    
    # Default track configurations for different assay types
    default_configs = {
        'DNASE': {
            'file_type': 'bedgraph',
            'color': '#1f77b4',
            'height': 2,
            'style': 'fill',
            'max_value': 'auto'
        },
        'CAGE': {
            'file_type': 'bedgraph', 
            'color': '#ff7f0e',
            'height': 2,
            'style': 'line:1.5',
            'max_value': 'auto'
        },
        'ATAC': {
            'file_type': 'bedgraph',
            'color': '#2ca02c',
            'height': 2,
            'style': 'fill',
            'max_value': 'auto'
        },
        'CHIP': {
            'file_type': 'bedgraph',
            'color': '#d62728',
            'height': 2,
            'style': 'fill',
            'max_value': 'auto'
        }
    }
    
    # Create configuration file
    config_content = []
    
    for track_file in track_files:
        # Determine track type from filename or metadata
        track_type = 'Unknown'
        with open(track_file, 'r') as f:
            header = f.readline()
            if 'DNASE' in header.upper():
                track_type = 'DNASE'
            elif 'CAGE' in header.upper():
                track_type = 'CAGE'
            elif 'ATAC' in header.upper():
                track_type = 'ATAC'
            elif 'CHIP' in header.upper():
                track_type = 'CHIP'
        
        # Get configuration
        if track_config and track_file in track_config:
            config = track_config[track_file]
        else:
            config = default_configs.get(track_type, {
                'file_type': 'bedgraph',
                'color': '#000000',
                'height': 2,
                'style': 'fill',
                'max_value': 'auto'
            })
        
        # Extract track name from file
        track_name = os.path.basename(track_file).replace('.bedgraph', '')
        
        # Add track configuration
        config_content.append(f"[{track_name}]")
        config_content.append(f"file = {track_file}")
        config_content.append(f"file_type = {config.get('file_type', 'bedgraph')}")
        config_content.append(f"color = {config.get('color', '#000000')}")
        config_content.append(f"height = {config.get('height', 2)}")
        
        # Style determines how the track is displayed
        style = config.get('style', 'fill')
        if style == 'fill':
            config_content.append("type = fill")
        elif style.startswith('line'):
            config_content.append("type = line")
            if ':' in style:
                line_width = style.split(':')[1]
                config_content.append(f"line_width = {line_width}")
        
        config_content.append(f"max_value = {config.get('max_value', 'auto')}")
        config_content.append(f"title = {track_name}")
        config_content.append("")
    
    # Add gene track if GTF file provided
    if gtf_file or genome_file:
        # Prefer GTF file over genome_file for backwards compatibility
        gene_file = gtf_file if gtf_file else genome_file
        config_content.append("[genes]")
        config_content.append(f"file = {gene_file}")
        config_content.append("file_type = gtf")
        config_content.append("height = 3")
        config_content.append("title = Genes")
        config_content.append("labels = true")
        config_content.append("fontsize = 10")
        config_content.append("gene_rows = 3")
        config_content.append("arrow_length = 1000")
        config_content.append("arrowhead_included = true")
        config_content.append("")
    
    # Write configuration to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('\n'.join(config_content))
        config_file = f.name
    
    try:
        # Create the plot
        
        tracks = PlotTracks(config_file, fig_width=width, dpi=dpi, plot_regions=[[chrom, start, end]])
        tracks.plot(output_file, chrom=chrom, start=start, end=end)
        return True
    #except Exception as e:
    #    warnings.warn(f"Error creating pyGenomeTracks plot: {e}")
    #    return False
    finally:
        # Clean up temp file
        if os.path.exists(config_file):
            os.remove(config_file)


def visualize_chorus_predictions(
    predictions: OraclePrediction,
    track_ids: List[str] = None,
    output_file: Optional[str] = None,
    use_pygenometracks: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    gtf_file: Optional[str] = None,
    show_gene_names: bool = True
) -> None:
    """
    Visualize Chorus predictions with appropriate styling for different assay types.
    
    Args:
        predictions: oracle predictions 
        track_ids: List of track IDs to plot
        output_file: Optional output file
        bin_size: Bin size for predictions
        style: Visualization style ('modern', 'classic', 'minimal')
        use_pygenometracks: Try to use pyGenomeTracks if available
        figsize: Figure size (width, height)
        gtf_file: Optional GTF file for gene annotations
        show_gene_names: Whether to show gene names on the plot
    """
    if track_ids is None:
        track_ids = list(predictions.keys())
    else:
        predictions = predictions.subset(track_ids)

    # First try pyGenomeTracks if requested
    if use_pygenometracks and output_file:
        # Save predictions as temporary BedGraph files
        import tempfile
        temp_files = []
        
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Determine track types and save files
            track_configs = {}

            # account for track_ids
            temp_files = predictions.save_predictions_as_bedgraph(output_dir=temp_dir,
                                                                  prefix='')                                                     
            for track_id, temp_file in temp_files.items():

                # Set configuration

                if predictions[track_id].assay_type == 'DNASE':
                    track_configs[temp_file] = {
                        'style': 'fill',
                        'color': '#1f77b4'
                    }
                elif predictions[track_id].assay_type == 'CAGE':
                    track_configs[temp_file] = {
                        'style': 'line:2',
                        'color': '#ff7f0e' 
                    }
            
            # Calculate genomic region
            interval = predictions[track_ids[0]].prediction_interval
            chrom, start, end = interval.reference.chrom, interval.reference.start, interval.reference.end
            # Try pyGenomeTracks
            success = plot_tracks_with_pygenometracks(
                list(temp_files.values()),
                chrom=chrom,
                start=start,
                end=end,
                output_file=output_file,
                track_config=track_configs,
                gtf_file=gtf_file
            )
            
            if success:
                return
                
        finally:
            # Clean up temp files
            import shutil
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
            pass
    
    # Fall back to matplotlib
    # Add extra subplot for genes if GTF provided
    n_subplots = len(track_ids) + (1 if gtf_file and show_gene_names else 0)
    
    if not figsize:
        figsize = (12, 3 * n_subplots)
    
    # Use white background style
    plt.style.use('default')
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True, facecolor='white')
    
    if n_subplots == 1:
        axes = [axes]
    
    # Plot each track
    for idx, (track_id, track) in enumerate(predictions.items()):
        ax = axes[idx]

        positions = np.arange(track.values.shape[0]) * track.resolution + track.prediction_interval.reference.start
        
        # Determine track color based on type
        if track.assay_type == 'DNASE' :
            color = '#1f77b4'  # Blue for DNase
        elif track.assay_type == 'CAGE':
            color = '#ff7f0e'  # Orange for CAGE
        elif track.assay_type == 'ATAC': 
            color = '#2ca02c'  # Green for ATAC
        elif track.assay_type =='CHIP':
            color = '#d62728'  # Red for ChIP
        else:
            color = '#9467bd'  # Purple for others
        
        # Use consistent style for all tracks: filled area plot
        ax.fill_between(positions, track.values, alpha=0.7, color=color)
        ax.plot(positions, track.values, color=color, linewidth=1)
        
        # Styling
        ax.set_ylabel(track_id, fontsize=10)
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add track statistics
        mean_val = np.mean(track.values)
        max_val = np.max(track.values)
        ax.text(0.02, 0.95, f'Mean: {mean_val:.2f}, Max: {max_val:.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add gene track if GTF file provided
    if gtf_file and show_gene_names:
        from .annotations import AnnotationManager
        
        ax_genes = axes[len(track_ids)]  # Gene track is the last subplot
        interval = predictions[track_ids[0]].prediction_interval
        chrom, start, end = interval.reference.chrom, interval.reference.start, interval.reference.end
        
        try:
            # Extract genes in region
            manager = AnnotationManager()
            genes_df = manager.extract_genes_in_region(gtf_file, chrom, start, end)
            
            if len(genes_df) > 0:
                # Plot genes
                gene_height = 0.3
                used_y_positions = []
                
                for _, gene in genes_df.iterrows():
                    # Find a y position that doesn't overlap
                    y_pos = 0
                    for used_y in used_y_positions:
                        if abs(y_pos - used_y) < gene_height * 1.5:
                            y_pos = used_y + gene_height * 1.5
                    used_y_positions.append(y_pos)
                    
                    # Draw gene as a rectangle
                    gene_start = max(gene['start'], start)
                    gene_end = min(gene['end'], end)
                    
                    # Gene body
                    rect = plt.Rectangle((gene_start, y_pos), 
                                       gene_end - gene_start, 
                                       gene_height,
                                       facecolor='lightblue' if gene['strand'] == '+' else 'lightcoral',
                                       edgecolor='black',
                                       linewidth=1)
                    ax_genes.add_patch(rect)
                    
                    # Gene name
                    if gene['gene_name']:
                        text_x = (gene_start + gene_end) / 2
                        ax_genes.text(text_x, y_pos + gene_height/2, 
                                    gene['gene_name'],
                                    ha='center', va='center',
                                    fontsize=8, weight='bold')
                    
                    # Add arrow for strand
                    arrow_y = y_pos + gene_height / 2
                    if gene['strand'] == '+':
                        ax_genes.annotate('', xy=(gene_end, arrow_y), 
                                        xytext=(gene_end - 2000, arrow_y),
                                        arrowprops=dict(arrowstyle='->', lw=1.5))
                    else:
                        ax_genes.annotate('', xy=(gene_start, arrow_y), 
                                        xytext=(gene_start + 2000, arrow_y),
                                        arrowprops=dict(arrowstyle='->', lw=1.5))
                
                # Set gene track properties
                ax_genes.set_ylim(-gene_height, max(used_y_positions) + gene_height * 1.5)
                ax_genes.set_ylabel('Genes', fontsize=10)
                ax_genes.set_facecolor('white')
                ax_genes.spines['top'].set_visible(False)
                ax_genes.spines['right'].set_visible(False)
                ax_genes.spines['left'].set_visible(False)
                ax_genes.set_yticks([])
                
            else:
                ax_genes.text(0.5, 0.5, 'No genes in region', 
                            transform=ax_genes.transAxes, 
                            ha='center', va='center',
                            fontsize=10, style='italic')
                ax_genes.set_ylabel('Genes', fontsize=10)
                ax_genes.set_ylim(0, 1)
                ax_genes.set_yticks([])
                
        except Exception as e:
            ax_genes.text(0.5, 0.5, f'Could not load genes: {str(e)}', 
                        transform=ax_genes.transAxes, 
                        ha='center', va='center',
                        fontsize=10, style='italic')
            ax_genes.set_ylabel('Genes', fontsize=10)
            ax_genes.set_ylim(0, 1)
            ax_genes.set_yticks([])
    
    # Set x-axis
    axes[-1].set_xlabel('Genomic Position', fontsize=12)
    axes[-1].ticklabel_format(style='plain', axis='x')
    
    # Add title
    interval = predictions[track_ids[0]].prediction_interval
    chrom, start, end = interval.reference.chrom, interval.reference.start, interval.reference.end
    fig.suptitle(f'{chrom}:{start}-{end}', fontsize=14)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure after saving
    else:
        # Always display the figure, even with non-interactive backend
        from IPython.display import display
        display(fig)
        plt.close(fig)  # Close after display to free memory