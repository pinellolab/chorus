# Quick Start Guide - Modular Environment System

This guide shows how to use Chorus with the new modular environment system that isolates each oracle's dependencies.

## Initial Setup

### 1. Install the Base Environment

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

# Create and activate the minimal base environment
mamba env create -f environment.yml
mamba activate chorus

# Install Chorus
pip install -e .
```

### 2. Check Available Environments

```bash
# List all available oracle environments
chorus list

# Output:
# Available oracle environments:
# - chorus-enformer: TensorFlow-based environment for Enformer oracle
# - chorus-borzoi: PyTorch-based environment for Borzoi oracle
# - chorus-chrombpnet: TensorFlow-based environment for ChromBPNet oracle
# - chorus-sei: PyTorch-based environment for Sei oracle
```

### 3. Set Up Oracle Environments

```bash
# Set up all environments (this may take a while)
chorus setup --all

# Or set up specific oracles only
chorus setup --oracle enformer
chorus setup --oracle sei

# Check setup status
chorus validate
```

## Managing Reference Genomes

Chorus includes utilities for automatically downloading and managing reference genomes:

```bash
# List available genomes
chorus genome list

# Download a reference genome
chorus genome download hg38

# Get genome information
chorus genome info hg38

# Remove a genome
chorus genome remove hg38
```

In Python, genomes are automatically downloaded when needed:

```python
from chorus.utils import get_genome

# Get genome path (downloads automatically if not present)
genome_path = get_genome('hg38')  # Default genome

# Use with oracles
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))
```

## Using Oracles with Isolated Environments

### Python API

```python
import chorus

# Method 1: Automatic environment management
oracle = chorus.create_oracle('enformer', use_environment=True)
oracle.load_pretrained_model()  # Runs in isolated environment

# Get reference genome (auto-downloads if needed)
from chorus.utils import get_genome
genome_path = get_genome('hg38')

# Make predictions
results = oracle.predict_region_replacement(
    genomic_region="chr1:1000000-1200000",
    seq="",
    assay_ids=["DNase:K562"],
    genome=str(genome_path)
)

# Method 2: Manual environment specification
from chorus.oracles.enformer_env import EnformerOracleEnv
oracle = EnformerOracleEnv(use_environment=True)
```

### Command Line

```bash
# Check environment health
chorus health

# Remove an environment
chorus remove --oracle enformer

# Run tests in isolated environments
chorus test --oracle enformer
```

## Benefits of the Modular System

1. **No Dependency Conflicts**: TensorFlow and PyTorch models can coexist
2. **Smaller Base Install**: Only install what you need
3. **Easy Updates**: Update individual oracle environments without affecting others
4. **Clean Uninstall**: Remove oracle environments individually

## Troubleshooting

### Environment Not Found
```bash
# If you get "Environment chorus-enformer not found"
chorus setup --oracle enformer
```

### Slow First Run
The first time you use an oracle with `use_environment=True`, it may be slower as it activates the conda environment.

### Memory Issues
Each oracle runs in a separate Python process when using isolated environments, which may use more memory.

### Disable Environment Isolation
If you want to run without isolation (requires manual dependency management):
```python
oracle = chorus.create_oracle('enformer', use_environment=False)
```

## Environment Details

- **Base (chorus)**: ~500MB, core functionality only
- **Enformer (chorus-enformer)**: ~2GB, includes TensorFlow
- **Borzoi (chorus-borzoi)**: ~2.5GB, includes PyTorch
- **ChromBPNet (chorus-chrombpnet)**: ~2GB, includes TensorFlow
- **Sei (chorus-sei)**: ~2.5GB, includes PyTorch

## Next Steps

1. Try the example notebooks with the modular system
2. Set up only the oracles you need
3. Use `chorus health` to monitor environment status