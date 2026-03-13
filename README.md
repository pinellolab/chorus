# Chorus

A unified interface for genomic sequence oracles - deep learning models that predict genomic regulatory activity from DNA sequences.

## Overview

Chorus provides a consistent, easy-to-use API for working with state-of-the-art genomic deep learning models including:

- **Enformer**: Predicts gene expression and chromatin states from DNA sequences
- **Borzoi**: Enhanced model for regulatory genomics predictions
- **ChromBPNet**: Predicts TF binding and chromatin accessibility at base-pair resolution
- **Sei**: Sequence regulatory effect predictions across 21,907 chromatin profiles
- **LegNet**: Regulatory regions activity prediction using models trained on MPRA data
- **AlphaGenome**: Google DeepMind's model predicting 5,930 genomic tracks at single base-pair resolution from 1MB input

Key features:
- 🧬 Unified API across different models
- 📊 Built-in visualization tools for genomic tracks
- 🔬 Variant effect prediction
- 🎯 In silico mutagenesis and sequence optimization
- 📈 Track normalization and comparison utilities
- 🚀 Enchanced sequence editing logic
- 🔧 **NEW**: Isolated conda environments for each oracle to avoid dependency conflicts

## ⚠️ Current Status

 Currently, Enformer, Sei, Borzoi, ChromBPNet, LegNet and AlphaGenome oracles are fully implemented with:

- Environment isolation support
- Reference genome integration for biologically accurate predictions
- ENCODE track identifier support
- BedGraph output generation

## Installation

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

#create main chorus env
mamba env create -f environment.yml
mamba activate chorus

# Install chorus package
pip install -e .
```

### Setting Up Oracle Environments

Chorus uses isolated conda environments for each oracle to avoid dependency conflicts between TensorFlow and PyTorch models:

```bash
# Set up Enformer environment (TensorFlow-based)
chorus setup --oracle enformer

# Set up AlphaGenome environment (JAX-based, see AlphaGenome section below)
chorus setup --oracle alphagenome

# List available environments
chorus list
```

You can check the correctness of installation using the following command

```bash
# Check environment health
chorus health --timeout 300
```

Note: If you haven’t used Oracle yet, it will need some time to download its weights.

### Managing Reference Genomes

Chorus includes built-in support for downloading and managing reference genomes:

```bash
# List available genomes
chorus genome list

# Download a reference genome (e.g., hg38, hg19, mm10)
chorus genome download hg38

# Get information about a downloaded genome
chorus genome info hg38

# Remove a downloaded genome
chorus genome remove hg38
```

Supported genomes:
- **hg38**: Human genome assembly GRCh38
- **hg19**: Human genome assembly GRCh37
- **mm10**: Mouse genome assembly GRCm38
- **mm9**: Mouse genome assembly NCBI37
- **dm6**: Drosophila melanogaster genome assembly BDGP6
- **ce11**: C. elegans genome assembly WBcel235

Genomes are stored in the `genomes/` directory within your Chorus installation.

## Quick Start

### Basic Setup

```python
import chorus
from chorus.utils import get_genome

# Create oracle with reference genome (auto-downloads if needed)
genome_path = get_genome('hg38')
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))
oracle.load_pretrained_model()

# Define tracks to predict (ENCODE IDs or descriptions)
tracks = ['ENCFF413AHU', 'CNhs11250']  # DNase:K562, CAGE:K562
```

#### Device Selection

By default, Chorus auto-detects and uses GPU if available. You can explicitly control device selection:

```python
# Force CPU usage (useful for testing or GPU memory issues)
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path),
                             device='cpu')

# Use specific GPU (for multi-GPU systems)
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             reference_fasta=str(genome_path),
                             device='cuda:1')  # Use second GPU

# Set default device via environment variable
# export CHORUS_DEVICE=cpu
```

#### Timeout Configuration

For slower systems or CPU-only environments, you may need to adjust timeouts:

```python
# Custom timeouts for slower systems
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path),
                             model_load_timeout=1200,  # 20 minutes
                             predict_timeout=600)      # 10 minutes

# Combine device and timeout settings
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             reference_fasta=str(genome_path),
                             device='cpu',             # Force CPU
                             model_load_timeout=1800,  # 30 minutes for CPU
                             predict_timeout=900)      # 15 minutes for CPU

# Disable all timeouts (use with caution)
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             reference_fasta=str(genome_path),
                             model_load_timeout=None,
                             predict_timeout=None)

# Or set environment variable to disable all timeouts globally
# export CHORUS_NO_TIMEOUT=1
```

### 1. Wild-type Prediction

```python
# Predict from genomic coordinates
predictions = oracle.predict(
    ('chr11', 5247000, 5248000),  # Beta-globin locus
    tracks
)

# Or from DNA sequence
sequence = 'ACGT' * 98304  # 393,216 bp for Enformer
predictions = oracle.predict(sequence, tracks)
```

### 2. Region Replacement

```python
# Replace a 200bp region with enhancer sequence
enhancer = 'GATA' * 50  # 200bp GATA motif repeats
replaced = oracle.predict_region_replacement(
    'chr11:5247400-5247600',  # Region to replace
    enhancer,                  # New sequence
    tracks
)
```

### 3. Sequence Insertion

```python
# Insert enhancer at specific position
inserted = oracle.predict_region_insertion_at(
    'chr11:5247500',  # Insertion point
    enhancer,         # Sequence to insert
    tracks
)
```

### 4. Variant Effect

```python
# Test SNP effects (e.g., A→G mutation)
variant_effects = oracle.predict_variant_effect(
    'chr11:5247000-5248000',  # Region containing variant
    'chr11:5247500',          # Variant position
    ['A', 'G', 'C', 'T'],     # Reference first, then alternates
    tracks
)
```

### 5. Save Predictions

```python
# Save as BedGraph for genome browser
wt_files = predictions.save_predictions_as_bedgraph(output_dir="bedgraph_outputs",
                                                    prefix='a_wt')

```

## Comprehensive Example

For a detailed walkthrough with visualizations and gene annotations, see the comprehensive notebook:

```bash
# Download reference genome and gene annotations
chorus genome download hg38

# Run the comprehensive notebook
jupyter notebook examples/gata1_comprehensive_analysis.ipynb
```

This notebook demonstrates:
- All prediction methods with real genomic data
- Gene annotation and visualization
- Saving outputs for genome browsers
- Performance tips and best practices

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

# Option 1: Using get_genome() - simplest approach
from chorus.utils import get_genome
genome_path = get_genome('hg38')  # Auto-downloads if not present
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))

# Option 2: Using GenomeManager directly
from chorus.utils import GenomeManager
gm = GenomeManager()
genome_path = gm.get_genome('hg38')  # Auto-downloads if needed
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))

# Predict using genomic coordinates
predictions = oracle.predict(('chr1', 1000000, 1001000), ['DNase:K562'])
```

### 3. Track Support

**Note: ENCODE track identifiers and cell type descriptions are specific to Enformer model. Other oracles may use different track naming conventions.**

For Enformer:
```python
# Using ENCODE identifier (recommended for reproducibility)
predictions = oracle.predict(sequence, ['ENCFF413AHU'])  # Specific DNase:K562 experiment

# Using descriptive name
predictions = oracle.predict(sequence, ['DNase:K562'])

# Using CAGE identifiers
predictions = oracle.predict(sequence, ['CNhs11250'])  # CAGE:K562
```

For other oracles (Borzoi, ChromBPNet, Sei, etc.), track specifications will vary based on the model's training data.

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

### Intervals

Class as a unified interface to the reference genome/sequence. This component enables structured access to genomic coordinates while explicitly tracking and managing sequence edits together with their corresponding model predictions, thereby supporting reproducible in silico perturbation workflows and consistent downstream analysis.

### Tracks
Tracks represent genomic signal data (e.g., DNase-seq, ChIP-seq). Enformer predicts 5,313 human tracks covering various assays and cell types.

### Environment Management
The `chorus` CLI manages conda environments for each oracle:

```bash
# Set up environments
chorus setup --oracle enformer

# Check health
chorus health

# Clean up
chorus remove --oracle enformer
```

## Model-Specific Details

### Enformer 

Enformer [6] is a hybrid convolutional–transformer architecture
designed for long-range sequence-to-function modeling of regulatory
genomics, with the primary goal of predicting transcriptional and
epigenomic activity directly from DNA sequence.

- Sequence length: 393,216 bp input, 114,688 bp output window
- Output: 896 bins × 5,313 tracks
- Bin size: 128 bp
- Track types: Gene expression (CAGE), chromatin accessibility (DNase/ATAC), histone modifications (ChIP-seq)
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF413AHU for DNase:K562)
  - CAGE IDs (e.g., CNhs11250 for CAGE:K562)
  - Descriptive names (e.g., 'DNase:K562', 'H3K4me3:HepG2')
- Track metadata: Included in the package (file with all 5,313 human track definitions)

### Borzoi

Enhanced Enformer with improved performance and RNA-tracks predictions.

- Sequence length: 393,216 bp input, 114,688 bp output window
- Output: 896 bins × 5,313 tracks
- Bin size: 128 bp
- Track types: Gene expression (CAGE, RNA-Seq), chromatin accessibility (DNase/ATAC), histone modifications (ChIP-seq)
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF413AHU for DNase:K562)
  - CAGE IDs (e.g., CNhs11250 for CAGE:K562)
  - Descriptive names (e.g., 'DNase:K562', 'H3K4me3:HepG2')
- Track metadata: Included in the package (file with all 7,610 human track definitions)


### ChromBPNet

Base-pair resolution for chromatin accessibility and TF binding predictions (uses TF-specific tracks)

- Sequence length: 2114 bp input
- Output: 1000 bins 
- Bin size: 1 bp
- Track types: DNase accessibility, TF binding (CHIP-Seq) 
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF574YLK for DNase:K562)

### Sei

Sequence regulatory effect predictions (uses custom track naming for 21,907 profiles)

- Sequence length: 4096 bp input
- Output: 1 bin 
- Bin size: 4096 bp
- Track types: DNase accessibility, TF binding (CHIP-Seq), histone modifications 
- Track identifiers: 
  - custom Sei track identifiers
- Track metadata: Included in the package (files with all 21907 human track definitions and 41 Sei-defined classes)


### LegNet

LegNet is a fully convolutional neural network designed for efficient modeling of short regulatory DNA sequences.

- Sequence length: 200 bp input
- Output: 1 bin
- Bin size: 200 bp
- Track types: Element activity in MPRA experiment
- Track identifiers:
  - cell line names

### AlphaGenome

AlphaGenome (Google DeepMind, Nature 2026) predicts 5,930 human functional genomic tracks at single base-pair resolution from up to 1 MB of DNA sequence using a JAX-based model.

- Sequence length: 1,048,576 bp (1 MB) input
- Output: 1,048,576 bins at single base-pair resolution
- Bin size: 1 bp (ATAC, CAGE, DNase, RNA-seq, splice sites, PRO-CAP) or 128 bp (ChIP-seq histone/TF)
- Track types: ATAC, CAGE, ChIP-seq (histone + TF), DNase, RNA-seq, Splice sites, PRO-CAP
- Track identifiers: `{OutputType}/{TrackName}/{Strand}` (e.g., `ATAC/CL:0000084 ATAC-seq/.`)
- Weights: Hosted on HuggingFace (gated repository, requires authentication)

#### AlphaGenome Setup

AlphaGenome weights are hosted on a **gated HuggingFace repository**. You must authenticate before first use:

1. **Create a HuggingFace account** at https://huggingface.co/join

2. **Accept the model license terms** at https://huggingface.co/google/alphagenome-all-folds (click "Agree and access repository")

3. **Log in to HuggingFace CLI**:
```bash
# Activate the AlphaGenome environment first
conda activate chorus-alphagenome

# Log in (will prompt for your HF token)
huggingface-cli login
```

You can generate a token at https://huggingface.co/settings/tokens.

4. **Set up the environment and verify**:
```bash
# Create the environment
chorus setup --oracle alphagenome

# Verify installation
chorus health --oracle alphagenome
```

#### AlphaGenome Usage

```python
import chorus
from chorus.utils import get_genome

genome_path = get_genome('hg38')

# Create and load oracle
oracle = chorus.create_oracle('alphagenome',
                              reference_fasta=str(genome_path),
                              device='cpu')  # or omit for auto-detect GPU
oracle.load_pretrained_model()

# Discover available tracks
print(oracle.list_assay_types())   # ['ATAC', 'CAGE', 'CHIP', 'DNASE', ...]
print(oracle.get_track_info('ATAC'))  # DataFrame of ATAC tracks

# Predict
tracks = ['ATAC/CL:0000084 ATAC-seq/.']  # T-cell ATAC-seq
predictions = oracle.predict(('chr1', 1_000_000, 2_048_576), tracks)
```

#### AlphaGenome GPU Support

AlphaGenome uses JAX, which supports multiple accelerator backends:

- **NVIDIA GPU (Linux)**: Automatically installs `jax[cuda12]` when NVIDIA GPU is detected during `chorus setup`
- **Apple Silicon (macOS)**: Automatically installs `jax-metal` for Metal GPU acceleration during `chorus setup`
- **CPU**: Works everywhere as fallback; pass `device='cpu'` to force CPU

```python
# Auto-detect best available device (GPU > Metal > CPU)
oracle = chorus.create_oracle('alphagenome')

# Force specific device
oracle = chorus.create_oracle('alphagenome', device='cpu')
oracle = chorus.create_oracle('alphagenome', device='gpu')    # NVIDIA CUDA
oracle = chorus.create_oracle('alphagenome', device='metal')  # Apple Metal
```

## Troubleshooting

### Timeout Issues
If you encounter timeout errors on slower systems:

```python
# Increase timeouts
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             model_load_timeout=1800,  # 30 minutes
                             predict_timeout=900)      # 15 minutes

# Or disable timeouts entirely
export CHORUS_NO_TIMEOUT=1
```

Common timeout scenarios:
- **Model loading**: First-time downloads can be slow (~1GB model)
- **CPU predictions**: GPU is 10-100x faster than CPU
- **Network filesystems**: Add 50% to timeouts for NFS/shared storage

### Environment Issues
```bash
# Check if environment exists
chorus health

# Recreate environment
chorus remove --oracle enformer
chorus setup --oracle enformer
```

### Memory Issues
Some oracles require a significant memory (~8-16 GB) for predictions. Solutions:
- Force CPU usage: `device='cpu'`
- Use a different GPU: `device='cuda:1'`
- Reduce batch size if needed

### AlphaGenome Authentication
AlphaGenome weights are hosted on a gated HuggingFace repository. If you see a `GatedRepoError` or 403 error:

```bash
# 1. Accept model terms at https://huggingface.co/google/alphagenome-all-folds
# 2. Log in to HuggingFace
conda activate chorus-alphagenome
huggingface-cli login
```

### CUDA/GPU Support
The isolated environments include GPU support. Ensure CUDA is properly installed on your system.

To check GPU availability:
```python
# In your Python environment
import tensorflow as tf
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
```

To force CPU usage when GPU causes issues:
```python
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             device='cpu')
```

## Contributing

We welcome contributions! Areas needing work:

1. Add more examples and tutorials
2. Implement batch prediction optimizations
3. Add more visualization utilities
4. Add more oracles 

### Adding New Oracles

We've designed Chorus to make it easy to add new genomic prediction models. Each oracle runs in its own isolated conda environment, avoiding dependency conflicts between different frameworks (TensorFlow, PyTorch, JAX, etc.).

**For detailed instructions on implementing a new oracle, see our [Contributing Guide](CONTRIBUTING.md).**

Key steps:
1. Inherit from `OracleBase` and implement required methods
2. Define your conda environment configuration
3. Use the environment isolation system for model loading and predictions
4. Add tests and example notebooks
5. Submit a PR with your implementation

The contributing guide includes complete code examples and templates to get you started.

## Citation

If you use Chorus in your research, please cite:

```bibtex
@software{chorus2026,
  title = {Chorus: A unified interface for genomic sequence oracles},
  author = {Dmitry Penzar , Lorenzo Ruggeri , Rosalba Giugno, Luca Pinello},
  year = {2026},
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
- LegNet (Penzar et al., 2023)
- AlphaGenome (Google DeepMind, 2026)

For vizualization tasks we extensively use [coolbox package](https://github.com/GangCaoLab/CoolBox)