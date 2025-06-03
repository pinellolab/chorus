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
- 🔧 **NEW**: Isolated conda environments for each oracle to avoid dependency conflicts

## ⚠️ Current Status

**This is a work-in-progress implementation.** Currently, only the Enformer oracle is fully implemented with:
- Environment isolation support
- Reference genome integration for biologically accurate predictions
- ENCODE track identifier support
- BedGraph output generation

Other oracles (Borzoi, ChromBPNet, Sei) have placeholder implementations and will be completed in future updates.

## Installation

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

# Install chorus package
pip install -e .

# Install the CLI tool
pip install --upgrade anthropic  # Required for setup assistant
chorus --help
```

### Setting Up Oracle Environments

Chorus uses isolated conda environments for each oracle to avoid dependency conflicts between TensorFlow and PyTorch models:

```bash
# Set up Enformer environment (TensorFlow-based)
chorus setup --oracle enformer

# Check environment status
chorus env status

# List available environments
chorus env list
```

## Quick Start

```python
import chorus

# Create an Enformer oracle with environment isolation
# This ensures TensorFlow dependencies don't conflict with PyTorch
oracle = chorus.create_oracle('enformer', use_environment=True)

# For genomic coordinate predictions, provide reference genome
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta='/path/to/hg38.fa')

# Load pre-trained model
oracle.load_pretrained_model()

# Predict using genomic coordinates (with automatic sequence extraction)
predictions = oracle.predict(
    ('chrX', 48780505, 48785229),  # Genomic coordinates
    ['ENCFF413AHU']  # ENCODE identifier for DNase:K562
)

# Or predict using a DNA sequence directly
sequence = 'ACGT' * 1000  # Your sequence
predictions = oracle.predict(sequence, ['DNase:K562'])
```

## Examples

See the `examples/` directory for comprehensive examples:

1. **`enformer_quick_start.py`** - Minimal example to get started
2. **`enformer_comprehensive_example.py`** - Detailed examples showing all features

```bash
# Run quick start (requires hg38.fa reference genome)
cd examples
python enformer_quick_start.py /path/to/hg38.fa

# Run comprehensive examples
python enformer_comprehensive_example.py /path/to/hg38.fa
```

## Key Features

### 1. Environment Isolation

Each oracle runs in its own conda environment to avoid dependency conflicts:

```python
# TensorFlow-based Enformer runs in isolated environment
enformer = chorus.create_oracle('enformer', use_environment=True)

# Future: PyTorch-based models will have their own environments
borzoi = chorus.create_oracle('borzoi', use_environment=True)  # Coming soon
```

### 2. Reference Genome Integration

For accurate predictions, provide a reference genome to extract proper flanking sequences:

```python
# Enformer requires 393,216 bp of context
# Chorus automatically extracts and pads sequences from the reference
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta='hg38.fa')

# Predict using genomic coordinates
predictions = oracle.predict(('chr1', 1000000, 1001000), ['DNase:K562'])
```

### 3. ENCODE Track Support

Use specific ENCODE identifiers or general track descriptions:

```python
# Using ENCODE identifier (recommended for reproducibility)
predictions = oracle.predict(sequence, ['ENCFF413AHU'])  # Specific DNase:K562 experiment

# Using descriptive name (may return multiple tracks)
predictions = oracle.predict(sequence, ['DNase:K562'])
```

### 4. BedGraph Output

Predictions can be saved as BedGraph tracks for genome browser visualization:

```python
# Predictions are returned as numpy arrays
# Each bin represents 128 bp for Enformer
# See examples for BedGraph generation code
```

## Core Concepts

### Oracles
Oracles are deep learning models that predict genomic regulatory activity. Each oracle implements a common interface while running in isolated environments.

### Tracks
Tracks represent genomic signal data (e.g., DNase-seq, ChIP-seq). Enformer predicts 5,313 human tracks covering various assays and cell types.

### Environment Management
The `chorus` CLI manages conda environments for each oracle:

```bash
# Set up environments
chorus setup --oracle enformer

# Check status
chorus env status

# Clean up
chorus env clean enformer
```

## API Reference

### Oracle Creation
```python
oracle = chorus.create_oracle(
    oracle_name='enformer',
    use_environment=True,  # Use isolated conda environment
    reference_fasta='/path/to/hg38.fa'  # Optional, for coordinate-based predictions
)
```

### Prediction Methods
```python
# Predict from sequence
predictions = oracle.predict(
    'ACGTACGT...',  # DNA sequence
    ['DNase:K562', 'ATAC-seq:K562']  # Track identifiers
)

# Predict from genomic coordinates (requires reference_fasta)
predictions = oracle.predict(
    ('chr1', 1000000, 1001000),  # (chrom, start, end)
    ['ENCFF413AHU']  # ENCODE identifiers
)
```

## Model-Specific Details

### Enformer (Implemented)
- Sequence length: 393,216 bp input, 114,688 bp output window
- Output: 896 bins × 5,313 tracks
- Bin size: 128 bp
- Tracks: Gene expression, chromatin accessibility, histone modifications, etc.
- Track metadata: Included in the package (783KB file with all 5,313 human track definitions)

### Other Models (Coming Soon)
- **Borzoi**: Enhanced Enformer with improved performance
- **ChromBPNet**: Base-pair resolution TF binding predictions  
- **Sei**: Sequence regulatory effect predictions

## Troubleshooting

### Environment Issues
```bash
# Check if environment exists
chorus env status

# Recreate environment
chorus env clean enformer
chorus setup --oracle enformer
```

### Memory Issues
Enformer requires significant memory (~8-16 GB) for predictions. Reduce batch size if needed.

### CUDA/GPU Support
The isolated environments include GPU support. Ensure CUDA is properly installed on your system.

## Contributing

We welcome contributions! Areas needing work:
1. Complete implementations for Borzoi, ChromBPNet, and Sei oracles
2. Add more examples and tutorials
3. Implement batch prediction optimizations
4. Add more visualization utilities

### Adding New Oracles

We've designed Chorus to make it easy to add new genomic prediction models. Each oracle runs in its own isolated conda environment, avoiding dependency conflicts between different frameworks (TensorFlow, PyTorch, JAX, etc.).

**For detailed instructions on implementing a new oracle, see our [Contributing Guide](CONTRIBUTING.md).**

Key steps:
1. Inherit from `OracleBase` and implement required methods
2. Define your conda environment configuration
3. Use the environment isolation system for model loading and predictions
4. Add tests and example notebooks
5. Submit a PR with your implementation

We're particularly interested in:
- **Borzoi** - Enhanced Enformer with improved performance
- **ChromBPNet** - Base-pair resolution TF binding predictions
- **Sei** - Sequence regulatory effect predictions
- **Basset** - Chromatin accessibility predictions
- **DeepSEA** - Variant effect predictions

The contributing guide includes complete code examples and templates to get you started.

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