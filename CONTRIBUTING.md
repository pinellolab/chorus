# Contributing to Chorus

Thank you for your interest in contributing to Chorus! This guide will walk you through the process of implementing a new oracle (genomic sequence prediction model) step by step.

## Overview

Chorus provides a unified interface for genomic sequence oracles. Each oracle runs in its own isolated conda environment to avoid dependency conflicts. To add a new oracle, you'll need to:

1. Create the oracle implementation
2. Define the conda environment requirements
3. Implement required methods
4. Add tests and examples
5. Submit a pull request

## Step-by-Step Guide to Implementing a New Oracle

### Step 1: Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/chorus.git
cd chorus
pip install -e .
```

### Step 2: Create Your Oracle Implementation

Create a new file in `chorus/oracles/` named after your oracle (e.g., `borzoi.py`):

```python
# chorus/oracles/borzoi.py
"""Borzoi oracle implementation."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging

from ..core.base import OracleBase
from ..core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class BorzoiOracle(OracleBase):
    """Borzoi oracle implementation."""
    
    def __init__(self, use_environment: bool = True, reference_fasta: Optional[str] = None):
        """
        Initialize Borzoi oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
            reference_fasta: Path to reference genome FASTA file
        """
        # Set oracle name before calling super().__init__
        self.oracle_name = 'borzoi'
        
        super().__init__(use_environment=use_environment)
        
        # Model-specific parameters
        self.sequence_length = 524288  # Example: Borzoi uses 524kb sequences
        self.bin_size = 128
        self.num_tracks = 7919  # Example track count
        
        # Store reference genome path
        self.reference_fasta = reference_fasta
        
        # Model components (will be loaded later)
        self._model = None
```

### Step 3: Implement Required Methods

Your oracle must implement these abstract methods from `OracleBase`:

#### 3.1 Model Loading

```python
def load_pretrained_model(self, weights: Optional[str] = None) -> None:
    """Load pre-trained model weights."""
    if weights is None:
        weights = "default_model_path_or_url"
    
    logger.info(f"Loading {self.oracle_name} model from {weights}")
    
    if self.use_environment:
        # Run loading in isolated environment
        load_code = f"""
import torch  # or tensorflow, depending on your model
# Your model loading code here
model = load_your_model('{weights}')
result = {{'loaded': True, 'description': 'Model loaded successfully'}}
"""
        
        result = self.run_code_in_environment(load_code, timeout=300)
        if result and result['loaded']:
            self.loaded = True
            logger.info(f"{self.oracle_name} model loaded successfully!")
        else:
            raise ModelNotLoadedError(f"Failed to load {self.oracle_name} model")
    else:
        # Direct loading if not using environment
        self._load_direct(weights)
```

#### 3.2 Track Information

```python
def list_assay_types(self) -> List[str]:
    """Return list of available assay types."""
    return [
        "DNase", "ATAC-seq", "ChIP-seq", "RNA-seq", 
        # Add your model's supported assay types
    ]

def list_cell_types(self) -> List[str]:
    """Return list of available cell types."""
    return [
        "K562", "GM12878", "HepG2", "H1-hESC",
        # Add your model's supported cell types
    ]
```

#### 3.3 Prediction Method

```python
def _predict(self, seq: Union[str, Tuple[str, int, int]], assay_ids: List[str]) -> np.ndarray:
    """
    Make predictions for given sequence and assays.
    
    Args:
        seq: Either DNA sequence string or (chrom, start, end) tuple
        assay_ids: List of assay identifiers
        
    Returns:
        numpy array of shape (num_bins, num_tracks)
    """
    if not self.loaded:
        raise ModelNotLoadedError("Model not loaded")
    
    # Handle genomic coordinates if provided
    if isinstance(seq, tuple):
        if self.reference_fasta is None:
            raise ValueError("Reference FASTA required for coordinate input")
        chrom, start, end = seq
        # Use the utility function to extract sequence with padding
        from ..utils.sequence import extract_sequence_with_padding
        seq = extract_sequence_with_padding(
            self.reference_fasta, chrom, start, end, self.sequence_length
        )
    
    if self.use_environment:
        # Run prediction in isolated environment
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(seq)
            seq_path = f.name
        
        predict_code = f"""
# Read sequence
with open('{seq_path}', 'r') as f:
    seq = f.read().strip()

# Your prediction code here
import torch  # or tensorflow
model = load_cached_model()  # Load from cache
predictions = model.predict(seq, {repr(assay_ids)})
result = predictions.tolist()
"""
        
        predictions = self.run_code_in_environment(predict_code, timeout=120)
        return np.array(predictions)
    else:
        # Direct prediction
        return self._predict_direct(seq, assay_ids)
```

#### 3.4 Required Helper Methods

```python
def _get_context_size(self) -> int:
    """Return the required context size for the model."""
    return self.sequence_length

def _get_sequence_length_bounds(self) -> Tuple[int, int]:
    """Return min and max sequence lengths accepted by the model."""
    return (1000, self.sequence_length)

def _get_bin_size(self) -> int:
    """Return the bin size for predictions."""
    return self.bin_size
```

### Step 4: Define the Conda Environment

Create an environment configuration that we can integrate into the setup system. Provide us with:

1. **Conda packages needed:**
```yaml
# Example for a PyTorch-based model
channels:
  - pytorch
  - conda-forge
  - bioconda
  - defaults

dependencies:
  - python=3.9
  - pytorch=2.0
  - torchvision
  - numpy
  - pandas
  - scikit-learn
  - pysam
  - bedtools
  - pip
  - pip:
    - your-special-package==1.0.0
```

2. **Installation commands:**
```bash
# Any special setup commands
# For example, downloading model weights:
wget https://example.com/model_weights.pt -O ~/.cache/borzoi/weights.pt
```

### Step 5: Register Your Oracle

Add your oracle to `chorus/oracles/__init__.py`:

```python
from .borzoi import BorzoiOracle

ORACLES = {
    'enformer': EnformerOracle,
    'borzoi': BorzoiOracle,  # Add your oracle
    # ...
}
```

Update `chorus/__init__.py` to support environment isolation:

```python
if oracle_name.lower() == 'borzoi':
    from .oracles.borzoi import BorzoiOracle
    return BorzoiOracle(use_environment=True, **kwargs)
```

### Step 6: Add Tests

Create a test file `tests/test_borzoi.py`:

```python
import pytest
import chorus


def test_borzoi_creation():
    """Test Borzoi oracle creation."""
    oracle = chorus.create_oracle('borzoi', use_environment=False)
    assert oracle.oracle_name == 'borzoi'
    assert oracle.sequence_length == 524288


def test_borzoi_tracks():
    """Test track listing."""
    oracle = chorus.create_oracle('borzoi', use_environment=False)
    assays = oracle.list_assay_types()
    assert 'DNase' in assays
    
    cells = oracle.list_cell_types()
    assert 'K562' in cells


# Add more tests for predictions, etc.
```

### Step 7: Create an Example Notebook

Create `examples/borzoi_example.ipynb` demonstrating your oracle's features:

```python
# Example notebook structure
1. Oracle initialization
2. Model loading
3. Basic sequence prediction
4. Genomic coordinate prediction (if supported)
5. Track visualization
6. Special features of your model
```

### Step 8: Document Your Oracle

Add a section to the README.md describing:
- Model capabilities
- Sequence length requirements
- Number of tracks
- Special features
- Citation information

## Environment Configuration Format

When submitting your oracle, provide the environment configuration in this format:

```python
# In your oracle implementation or a separate config file
BORZOI_ENV_CONFIG = {
    'channels': ['pytorch', 'conda-forge', 'bioconda', 'defaults'],
    'dependencies': [
        'python=3.9',
        'pytorch=2.0',
        'numpy',
        'pandas',
        # ... other conda packages
    ],
    'pip_packages': [
        'special-package==1.0.0',
        # ... other pip packages
    ],
    'post_install_commands': [
        'wget https://example.com/weights.pt -O ~/.cache/borzoi/weights.pt',
        # ... other setup commands
    ]
}
```

## Best Practices

1. **Lazy Imports**: Import model-specific packages inside methods to avoid import errors:
   ```python
   def _load_direct(self, weights):
       import torch  # Import here, not at module level
   ```

2. **Memory Management**: Be mindful of memory usage, especially for large models

3. **Error Handling**: Provide clear error messages for common issues

4. **Logging**: Use the logger for important status updates

5. **Type Hints**: Use proper type annotations for all methods

6. **Documentation**: Include docstrings for all public methods

## Submitting Your Contribution

1. **Create a Pull Request** with:
   - Your oracle implementation
   - Environment configuration
   - Tests
   - Example notebook
   - Documentation updates

2. **PR Description** should include:
   - Model description and capabilities
   - Environment setup instructions
   - Any special requirements
   - Link to model paper/repository

3. **Testing**: Ensure all tests pass and the oracle works in both modes:
   - With environment isolation (`use_environment=True`)
   - Without environment isolation (`use_environment=False`)

## Example PR Structure

```
chorus/
├── oracles/
│   └── borzoi.py          # Your oracle implementation
├── tests/
│   └── test_borzoi.py     # Tests
├── examples/
│   └── borzoi_example.ipynb  # Example notebook
└── README.md              # Updated with your oracle info
```

## Getting Help

- Open an issue for questions
- Join discussions in existing oracle implementation PRs
- Tag maintainers for review: @pinellolab

## Current Priorities

We're particularly interested in implementations for:
1. **Borzoi** - Enhanced Enformer model
2. **ChromBPNet** - Base-pair resolution TF binding
3. **Sei** - Sequence regulatory effects
4. **Basset** - Chromatin accessibility
5. **DeepSEA** - Variant effects

Thank you for contributing to Chorus! Your implementation will help make genomic deep learning models more accessible to the research community.