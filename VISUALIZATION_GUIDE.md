# Chorus Visualization Guide

## Overview

Chorus provides multiple visualization functions for genomic tracks and predictions.

## Main Visualization Functions

### 1. visualize_chorus_predictions()

The primary function for visualizing Chorus predictions with gene annotations.

```python
visualize_chorus_predictions(
    predictions: Dict[str, np.ndarray],
    chrom: str,
    start: int,
    track_ids: List[str],
    output_file: Optional[str] = None,
    bin_size: int = 128,
    style: str = 'modern',
    use_pygenometracks: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    gtf_file: Optional[str] = None,
    show_gene_names: bool = True
) -> None
```

**Features**:
- Automatic track coloring based on assay type
- Gene annotation track (when GTF provided)
- PyGenomeTracks support for publication-quality figures
- Matplotlib fallback for quick visualization

**Example**:
```python
# Basic visualization
visualize_chorus_predictions(
    predictions=oracle_predictions,
    chrom='chrX',
    start=48726820,  # Use output window start for Enformer
    track_ids=['ENCFF413AHU', 'CNhs11250'],
    output_file='predictions.png'
)

# With gene annotations
visualize_chorus_predictions(
    predictions=oracle_predictions,
    chrom='chrX', 
    start=48726820,
    track_ids=['ENCFF413AHU', 'CNhs11250'],
    gtf_file='gencode.v48.gtf.gz',
    show_gene_names=True,
    output_file='predictions_with_genes.png'
)
```

**Track Colors** (automatic assignment):
- DNase/ATAC: Blue (#1f77b4)
- CAGE: Orange (#ff7f0e)
- ChIP-seq: Red (#d62728)
- Others: Purple (#9467bd)

### 2. visualize_tracks()

Visualize multiple BedGraph track files.

```python
visualize_tracks(
    tracks_filenames: List[str],
    track_names: List[str],
    scales: Optional[List[Tuple[float, float]]] = None,
    colors: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    genomic_region: Optional[str] = None,
    figure_size: Optional[Tuple[float, float]] = None,
    style: str = 'default'
) -> None
```

**Example**:
```python
# Visualize saved BedGraph files
visualize_tracks(
    tracks_filenames=[
        'wt_ENCFF413AHU.bedgraph',
        'mutant_ENCFF413AHU.bedgraph'
    ],
    track_names=['Wild-type DNase', 'Mutant DNase'],
    genomic_region='chrX:48780000-48790000',
    scales=[(0, 20), (0, 20)],  # Same scale for comparison
    colors=['blue', 'red'],
    output_file='comparison.png'
)
```

### 3. plot_track_heatmap()

Create heatmaps across multiple regions.

```python
plot_track_heatmap(
    tracks: List[Union[str, pd.DataFrame]],
    track_names: List[str],
    genomic_regions: List[str],
    output_file: Optional[str] = None,
    cmap: str = 'RdBu_r',
    normalize_tracks: bool = True,
    cluster_tracks: bool = False,
    cluster_regions: bool = False
) -> None
```

**Example**:
```python
# Compare signals across promoters
regions = [
    'chr11:5247000-5248000',  # HBB
    'chr11:5269000-5270000',  # HBD
    'chr16:226000-227000'     # HBA1
]

plot_track_heatmap(
    tracks=['dnase_erythroid.bedgraph', 'h3k4me3_erythroid.bedgraph'],
    track_names=['DNase', 'H3K4me3'],
    genomic_regions=regions,
    normalize_tracks=True,
    cluster_regions=True,
    output_file='promoter_heatmap.png'
)
```

### 4. plot_track_comparison()

Compare and correlate two tracks.

```python
plot_track_comparison(
    track1_file: str,
    track2_file: str,
    track1_name: str,
    track2_name: str,
    genomic_region: Optional[str] = None,
    output_file: Optional[str] = None,
    correlation_method: str = 'pearson'
) -> Dict[str, float]
```

**Example**:
```python
# Compare DNase and ATAC
stats = plot_track_comparison(
    track1_file='dnase_k562.bedgraph',
    track2_file='atac_k562.bedgraph',
    track1_name='DNase-seq',
    track2_name='ATAC-seq',
    genomic_region='chr1:1000000-2000000',
    correlation_method='pearson',
    output_file='dnase_vs_atac.png'
)

print(f"Correlation: {stats['correlation']:.3f}")
print(f"P-value: {stats['p_value']:.3e}")
```

### 5. plot_tracks_with_pygenometracks()

High-quality visualization using pyGenomeTracks.

```python
plot_tracks_with_pygenometracks(
    track_files: List[str],
    genomic_region: str,
    output_file: str,
    track_config: Optional[Dict[str, Dict]] = None,
    gtf_file: Optional[str] = None,
    height_ratios: Optional[List[float]] = None,
    width: float = 10,
    dpi: int = 300
) -> bool
```

**Example**:
```python
# Publication-quality figure
success = plot_tracks_with_pygenometracks(
    track_files=[
        'dnase.bedgraph',
        'cage.bedgraph', 
        'h3k4me3.bedgraph'
    ],
    genomic_region='chrX:48780000-48790000',
    output_file='figure_2a.pdf',
    track_config={
        'dnase.bedgraph': {
            'color': '#1f77b4',
            'style': 'fill',
            'height': 3,
            'max_value': 50
        },
        'cage.bedgraph': {
            'color': '#ff7f0e', 
            'style': 'line:2',
            'height': 3,
            'max_value': 200
        }
    },
    gtf_file='gencode.gtf',
    width=8,
    dpi=300
)
```

## Gene Annotation Visualization

When a GTF file is provided, gene tracks show:
- Gene bodies as rectangles
- Strand direction (+ strand: blue, - strand: red)
- Gene names as labels
- Arrows indicating transcription direction

## Coordinate Handling for Enformer

Enformer has asymmetric input/output windows:
- Input: 393,216 bp
- Output: 114,688 bp (centered)

**Important**: When visualizing Enformer predictions, use the output window coordinates:

```python
# Get output window coordinates
region_center = (start + end) // 2
output_start, output_end = oracle.get_output_window_coords(region_center)

# Visualize using output coordinates
visualize_chorus_predictions(
    predictions=predictions,
    chrom='chrX',
    start=output_start,  # NOT the original start
    track_ids=track_ids
)
```

## Visualization Styles

### Modern (default)
- Clean white background
- Colored tracks with transparency
- Grid lines for reference
- Track statistics overlay

### Minimal
- Reduced visual elements
- No spines or ticks
- Focus on data

### Publication
- High contrast
- Suitable for journals
- No decorative elements

## Saving Outputs

### File Formats
- **PNG**: Best for presentations and web
- **PDF**: Vector format for publications
- **SVG**: Editable vector format

### Resolution
```python
# High resolution for print
plt.savefig('figure.png', dpi=300, bbox_inches='tight')

# Screen resolution
plt.savefig('figure.png', dpi=100)
```

## Tips for Publication-Quality Figures

1. **Use PyGenomeTracks** for complex figures:
```bash
pip install pyGenomeTracks
```

2. **Consistent scales** across conditions:
```python
max_val = max(np.max(wt), np.max(mutant))
scales = [(0, max_val), (0, max_val)]
```

3. **Color accessibility**:
- Use colorblind-friendly palettes
- Avoid red-green combinations
- Test with grayscale printing

4. **Font sizes**:
```python
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
```

## Interactive Visualization

For exploring data interactively:

1. **Jupyter notebooks**: Inline plots with zoom/pan
2. **Genome browsers**: Export BedGraph → IGV/UCSC
3. **Plotly** (future support): Interactive web plots

## Common Issues

### "Figure not displaying"
```python
# Ensure matplotlib backend is set
%matplotlib inline  # For Jupyter

# Or use explicit display
from IPython.display import display
display(fig)
```

### "pyGenomeTracks not found"
```bash
# Install separately
pip install pyGenomeTracks

# Or use matplotlib fallback
use_pygenometracks=False
```

### "Memory error with large regions"
- Downsample data for visualization
- Use specific genomic regions
- Save to file instead of displaying

## Examples Gallery

### 1. Before/After Enhancer Insertion
```python
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Plot wild-type
visualize_predictions_on_ax(wt_predictions, axes[0], 'Wild-type')

# Plot with enhancer
visualize_predictions_on_ax(enhancer_predictions, axes[1], 'With Enhancer')

plt.tight_layout()
```

### 2. Multi-Region Comparison
```python
regions = ['chr1:1000-2000', 'chr2:3000-4000', 'chr3:5000-6000']
fig, axes = plt.subplots(len(regions), 1, figsize=(12, 3*len(regions)))

for region, ax in zip(regions, axes):
    predictions = oracle.predict(region, tracks)
    visualize_predictions_on_ax(predictions, ax, region)
```

### 3. Variant Effect Waterfall
```python
# Show effects of all variants in a region
effects = []
positions = []

for pos in range(start, end):
    result = oracle.predict_variant_effect(...)
    max_effect = np.max(np.abs(result['effect_sizes']['alt_1']['ENCFF413AHU']))
    effects.append(max_effect)
    positions.append(pos)

plt.figure(figsize=(12, 4))
plt.bar(positions, effects, width=1)
plt.xlabel('Genomic Position')
plt.ylabel('Max Variant Effect')
```