# Chorus Modular Environment System

This directory contains conda environment definitions for each oracle in the Chorus library. Each oracle can have its own isolated environment to avoid dependency conflicts.

## Overview

The modular environment system allows each oracle to run in its own conda environment with specific dependencies. This solves the common problem of conflicting dependencies between different genomic models (e.g., TensorFlow versions for Enformer vs PyTorch versions for Borzoi).

## Environment Files

- `chorus-base.yml`: Minimal base environment with core Chorus dependencies
- `chorus-enformer.yml`: Environment for Enformer (TensorFlow-based)
- `chorus-borzoi.yml`: Environment for Borzoi (PyTorch-based)
- `chorus-sei.yml`: Environment for Sei (PyTorch-based)
- `chorus-chrombpnet.yml`: Environment for ChromBPNet (TensorFlow-based)

## Usage

### Using the CLI

```bash
# List available environments
chorus list

# Set up all oracle environments
chorus setup

# Set up a specific oracle environment
chorus setup --oracle enformer

# Validate environments
chorus validate

# Check environment health
chorus health

# Remove an environment
chorus remove --oracle enformer
```

### Using in Python

```python
from chorus.oracles.enformer_env import EnformerOracleEnv

# Create oracle with automatic environment management
oracle = EnformerOracleEnv(use_environment=True)

# Load model (runs in isolated environment)
oracle.load_pretrained_model()

# Make predictions (runs in isolated environment)
predictions = oracle.predict_region_replacement(
    genomic_region="chr1:1000-2000",
    seq="ATCG" * 1000,
    assay_ids=["DNase", "ATAC-seq"]
)
```

### Direct Environment Management

```python
from chorus.core.environment import EnvironmentManager, EnvironmentRunner

# Create environment manager
manager = EnvironmentManager()

# Set up an environment
manager.create_environment('enformer')

# Run code in environment
runner = EnvironmentRunner(manager)

def my_function():
    import tensorflow as tf
    return tf.__version__

result = runner.run_in_environment('enformer', my_function)
```

## Environment Structure

Each environment includes:
- Core Python (3.10)
- Essential scientific computing packages (numpy, pandas, scipy)
- Bioinformatics tools (pysam, biopython, pyfaidx)
- Oracle-specific dependencies (TensorFlow/PyTorch/etc.)

## Adding New Oracles

To add a new oracle with its own environment:

1. Create `chorus-{oracle_name}.yml` in this directory
2. Include core dependencies plus oracle-specific packages
3. Update the oracle class to inherit from `OracleBase` with `use_environment=True`
4. The environment will be automatically detected by the CLI

## Troubleshooting

### Environment Creation Issues

If environment creation fails:
- Check that conda/mamba is installed and accessible
- Ensure you have sufficient disk space
- Try using mamba instead of conda for faster solving
- Check for proxy/firewall issues

### Import Errors

If imports fail in an environment:
- Run `chorus validate --oracle {name}` to check the environment
- Ensure all required dependencies are in the environment file
- Check for version conflicts

### Performance

Running in isolated environments adds some overhead:
- First prediction may be slower due to environment activation
- Subsequent predictions reuse the activated environment
- For production use, consider running with `use_environment=False` after testing

## Development

To modify environment definitions:
1. Edit the appropriate `.yml` file
2. Run `chorus setup --oracle {name} --force` to recreate
3. Test with `chorus validate --oracle {name}`