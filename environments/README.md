# Chorus Modular Environment System

This directory contains conda environment definitions for each oracle in the Chorus library. Each oracle runs in its own isolated environment to avoid dependency conflicts.

## Environment Files

- `chorus-base.yml`: Minimal base environment with core Chorus dependencies
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

## Troubleshooting

- **Environment creation fails**: Check that mamba is installed and you have sufficient disk space
- **Import errors in environment**: Run `chorus validate --oracle {name}` to check
- **Slow first prediction**: Normal — environment activation adds overhead on first call
