# Chorus Modular Environment System

This directory contains conda environment definitions for each oracle in the Chorus library. Each oracle runs in its own isolated environment to avoid dependency conflicts.

> **Which env file should I use?** The root `environment.yml` (at the top of the
> repo) is the one used by the install instructions in the main `README.md` —
> it adds matplotlib/seaborn/jupyter on top of the minimal `chorus-base.yml`
> here, so notebooks work out of the box. The `chorus-{oracle}.yml` files in
> this directory are installed automatically by `chorus setup --oracle {name}`
> and you should not need to touch them directly.

## Environment Files

- `chorus-base.yml`: Minimal base environment (used by `chorus setup` internals — the root `environment.yml` is a superset for general use)
- `chorus-enformer.yml`: Environment for Enformer (TensorFlow-based)
- `chorus-borzoi.yml`: Environment for Borzoi (PyTorch-based)
- `chorus-chrombpnet.yml`: Environment for ChromBPNet (TensorFlow-based)
- `chorus-sei.yml`: Environment for Sei (PyTorch-based)
- `chorus-legnet.yml`: Environment for LegNet (PyTorch-based)
- `chorus-alphagenome.yml`: Environment for AlphaGenome (JAX-based)

## Usage

### Using the CLI

```bash
# List available environments
chorus list

# Set up all oracle environments
chorus setup

# Set up a specific oracle environment
chorus setup --oracle enformer

# Check environment health
chorus health

# Validate environments
chorus validate

# Remove an environment
chorus remove --oracle enformer

# Force recreate an environment
chorus setup --oracle enformer --force
```

### Using in Python

```python
import chorus
from chorus.utils import get_genome

# Create oracle with automatic environment management
genome_path = get_genome('hg38')
oracle = chorus.create_oracle('enformer',
                              use_environment=True,
                              reference_fasta=str(genome_path))

# Load model (runs in isolated environment)
oracle.load_pretrained_model()

# Make predictions (runs in isolated environment)
predictions = oracle.predict(('chr1', 1000000, 1001000), ['ENCFF413AHU'])
```

## Adding New Oracles

To add a new oracle with its own environment:

1. Create `chorus-{oracle_name}.yml` in this directory
2. Include core dependencies plus oracle-specific packages
3. Update the oracle class to inherit from `OracleBase` with `use_environment=True`
4. The environment will be automatically detected by the CLI

## GPU Support

| Oracle | Framework | GPU Support |
|--------|-----------|-------------|
| Enformer | TensorFlow 2.14 | `nvidia-*-cu11` pip packages (in YML) |
| ChromBPNet | TensorFlow 2.8 | `nvidia-*-cu11` pip packages (in YML) |
| Borzoi | PyTorch | Bundled CUDA (automatic) |
| Sei | PyTorch | Bundled CUDA (automatic) |
| LegNet | PyTorch | Bundled CUDA (automatic) |
| AlphaGenome | JAX | Bundled CUDA (automatic) |

**PyTorch/JAX oracles** detect GPUs automatically — no extra setup needed.

**TensorFlow oracles** (Enformer, ChromBPNet) require `nvidia-*-cu11` pip
packages for GPU support. These are included in the YML files and installed
automatically during `chorus setup`. The chorus environment runner sets
`LD_LIBRARY_PATH` to the nvidia package lib directories so TF can find them.

On macOS, the nvidia packages are automatically excluded by the platform
adaptation system (CUDA is not available on Mac).

## HuggingFace Access

Most oracles and all background distributions are **publicly available** and
require no HuggingFace account.

**AlphaGenome** is a gated model from Google DeepMind. To use it:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Request access at [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) (click "Agree and access repository")
3. Set your token: `export HF_TOKEN=hf_your_token_here`

## Troubleshooting

- **Environment creation fails**: Check that mamba is installed and you have sufficient disk space
- **Import errors in environment**: Run `chorus validate --oracle {name}` to check
- **Slow first prediction**: Normal — environment activation adds overhead on first call
- **TF oracle shows "Skipping GPU"**: Check that `nvidia-cudnn-cu11` is installed with version <9.0 (TF 2.8-2.14 need cuDNN 8.x)
- **GPU out of memory**: Use `device='cpu'` parameter or `CUDA_VISIBLE_DEVICES=''`
