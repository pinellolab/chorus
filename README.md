# Chorus

A unified interface for genomic sequence oracles - deep learning models that predict genomic regulatory activity from DNA sequences.

## Overview

Chorus provides a consistent, easy-to-use API for working with state-of-the-art genomic deep learning models including:

- **Enformer**: Predicts gene expression and chromatin states from DNA sequences
- **Borzoi**: Enhanced model for regulatory genomics predictions  
- **ChromBPNet**: Predicts TF binding and chromatin accessibility at base-pair resolution
- **Sei**: Sequence regulatory effect predictions across 21,907 chromatin profiles

Key features:
- 🧬 Unified API across different models
- 📊 Built-in visualization tools for genomic tracks
- 🔬 Variant effect prediction
- 🎯 In silico mutagenesis and sequence optimization
- 📈 Track normalization and comparison utilities
- 🚀 Batch prediction support

## Installation

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

# Install in development mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/pinellolab/chorus.git
```

### Dependencies

Chorus requires Python 3.8+ and the following key dependencies:
- TensorFlow 2.8+ (for Enformer)
- PyTorch 1.10+ (for other models)
- NumPy, Pandas, Matplotlib
- pysam (for genomic file handling)

## Quick Start

```python
import chorus

# Create an oracle instance
oracle = chorus.create_oracle('enformer')

# Load pre-trained model
oracle.load_pretrained_model()

# List available assays and cell types
print(oracle.list_assay_types())
print(oracle.list_cell_types())

# Predict regulatory activity for a genomic region
results = oracle.predict_region_replacement(
    genomic_region="chr1:1000000-1200000",
    seq="",  # Use reference sequence
    assay_ids=["DNase:K562", "CAGE:K562"],
    genome="hg38.fa"
)

# Analyze variant effects
variant_results = oracle.predict_variant_effect(
    genomic_region="chr1:1000000-1001000",
    variant_position="chr1:1000500",
    alleles=["A", "G"],  # Reference and alternate
    assay_ids=["DNase:K562"],
    genome="hg38.fa"
)
```

## Core Concepts

### Oracles
Oracles are deep learning models that predict genomic regulatory activity. Each oracle implements a common interface:

```python
# All oracles inherit from OracleBase
oracle = chorus.EnformerOracle()
oracle.load_pretrained_model()

# Common methods across all oracles
oracle.predict_region_replacement(...)
oracle.predict_region_insertion_at(...)
oracle.predict_variant_effect(...)
oracle.fine_tune(...)
```

### Tracks
Tracks represent genomic signal data (e.g., DNase-seq, ChIP-seq):

```python
# Create a track from data
track = chorus.Track(
    name="my_dnase_track",
    assay_type="DNase",
    cell_type="K562",
    data=track_dataframe  # Requires: chrom, start, end, value
)

# Save as BEDGraph
track.to_bedgraph("output.bedgraph")

# Normalize track
normalized = track.normalize(method="quantile")
```

### Sequence Utilities
Work with DNA sequences and variants:

```python
# Extract sequence from reference genome
seq = chorus.extract_sequence("chr1:1000-2000", genome="hg38.fa")

# Apply variants
mutated = chorus.apply_variant(seq, position=500, ref="A", alt="G")

# Parse VCF files
variants = chorus.parse_vcf("variants.vcf")
```

## Examples

### 1. Predict Enhancer Activity

```python
# Define enhancer sequence
enhancer_seq = "GGATCCAAGGCTGCAGCAGAGGGGCAAAGTGAGGC..."

# Test at different positions
for position in ["chr8:128740000", "chr8:128745000"]:
    results = oracle.predict_region_insertion_at(
        genomic_position=position,
        seq=enhancer_seq,
        assay_ids=["H3K27ac:K562", "DNase:K562"],
        genome="hg38.fa"
    )
    print(f"Activity at {position}: {results['normalized_scores']}")
```

### 2. Visualize Predictions

```python
# Visualize multiple tracks
chorus.visualize_tracks(
    tracks_filenames=["dnase.bedgraph", "h3k27ac.bedgraph"],
    track_names=["DNase", "H3K27ac"],
    colors=["blue", "red"],
    output_file="tracks_visualization.png"
)

# Compare two tracks
stats = chorus.plot_track_comparison(
    track1_file="predicted.bedgraph",
    track2_file="experimental.bedgraph",
    track1_name="Predicted",
    track2_name="Experimental",
    correlation_method="pearson"
)
```

### 3. Batch Variant Analysis

```python
# Analyze all variants in a VCF
variants = chorus.parse_vcf("gwas_variants.vcf")

for _, variant in variants.iterrows():
    results = oracle.predict_variant_effect(
        genomic_region=f"{variant['chrom']}:{variant['pos']-1000}-{variant['pos']+1000}",
        variant_position=f"{variant['chrom']}:{variant['pos']}",
        alleles=[variant['ref'], variant['alt']],
        assay_ids=["DNase:K562", "ATAC-seq:K562"],
        genome="hg38.fa"
    )
    
    # Analyze effects
    print(f"{variant['id']}: {results['effect_sizes']}")
```

## Model-Specific Features

### Enformer
- Sequence length: 393,216 bp (196,608 bp prediction window)
- Output: 896 bins × 5,313 tracks (human + mouse)
- Bin size: 128 bp

```python
enformer = chorus.EnformerOracle()
enformer.compute_contribution_scores(...)  # Integrated gradients
```

### ChromBPNet
- Sequence length: 2,114 bp
- Output: Base-pair resolution (1,000 bp)
- Specialized for TF binding and accessibility

### Sei
- Sequence length: 4,096 bp
- Output: 21,907 regulatory features
- Sequence class predictions

## API Reference

### Oracle Methods

#### `predict_region_replacement(genomic_region, seq, assay_ids, create_tracks=False, genome="hg38.fa")`
Replace a genomic region with a new sequence and predict activity.

#### `predict_variant_effect(genomic_region, variant_position, alleles, assay_ids, create_tracks=False, genome="hg38.fa")`
Predict the effects of genetic variants.

#### `fine_tune(tracks, track_names, **kwargs)`
Fine-tune the model on new experimental data.

### Utility Functions

#### Sequence Manipulation
- `extract_sequence()`: Extract DNA from reference genome
- `apply_variant()`: Apply variant to sequence
- `reverse_complement()`: Get reverse complement
- `parse_vcf()`: Parse VCF files

#### Normalization
- `quantile_normalize()`: Quantile normalization
- `minmax_normalize()`: Min-max scaling
- `zscore_normalize()`: Z-score normalization

#### Visualization
- `visualize_tracks()`: Plot genomic tracks
- `plot_track_heatmap()`: Create track heatmaps
- `plot_track_comparison()`: Compare and correlate tracks

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use Chorus in your research, please cite:

```bibtex
@software{chorus2024,
  title = {Chorus: A unified interface for genomic sequence oracles},
  author = {Pinello Lab},
  year = {2024},
  url = {https://github.com/pinellolab/chorus}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Chorus integrates several groundbreaking models:
- Enformer (Avsec et al., 2021)
- Borzoi (Linder et al., 2023)
- ChromBPNet (Agarwal et al., 2021)
- Sei (Chen et al., 2022)