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
python -m pip install -e .
```

### Step 2: Create Your Oracle Implementation

Create a new file in `chorus/oracles/` named after your oracle (e.g., `mymodel.py`):

```python
# chorus/oracles/mymodel.py
"""MyModel oracle implementation."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging

from ..core.base import OracleBase
from ..core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class MyModelOracle(OracleBase):
    """MyModel oracle implementation."""
    
    def __init__(self, use_environment: bool = True, reference_fasta: Optional[str] = None):
        """
        Initialize MyModel oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
            reference_fasta: Path to reference genome FASTA file
        """
        # Set oracle name before calling super().__init__
        self.oracle_name = 'mymodel'
        
        super().__init__(use_environment=use_environment)
        
        # Model-specific parameters
        self.sequence_length = 524288  # Example: MyModel uses 524kb sequences
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
# Example for a PyTorch-based model. Compare against a real one, e.g.
# environments/chorus-sei.yml, which carries an in-file note on why its old
# pytorch<2.0 + cudatoolkit=11.7 pin was itself the bug.
name: chorus-mymodel   # REQUIRED — EnvironmentManager derives both the env name
                       # and the file path as chorus-{oracle}
channels:
  - pytorch
  - conda-forge
  - bioconda
  - defaults

dependencies:
  - python=3.10
  - pytorch>=2.0.0
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
wget https://example.com/model_weights.pt -O ~/.cache/mymodel/weights.pt
```

### Step 5: Register Your Oracle

> **This guide is the canonical one.** `environments/README.md` and
> `docs/IMPLEMENTATION_GUIDE.md` used to carry their own shorter recipes and both were wrong —
> the latter told contributors to edit an `ORACLE_REGISTRY` that has never existed in this
> codebase. They now point here. If you change the registration surface, change it here.

Registration is **not** automatic. Dropping a `chorus-{name}.yml` into `environments/` makes your
oracle appear in `chorus list`, and that is genuinely all it does — the oracle is not loadable,
not scoreable, and invisible to the MCP server until every site below names it. There is no
single registry; these are hand-edited, and the list is in dependency order.

**Required — without these the oracle does not work at all:**

1. **`chorus/oracles/mymodel.py`** — the class (Steps 2–3 above). It **must** declare
   `training_genome` on the subclass. `OracleBase.training_genome` is deliberately `None` so a new
   oracle cannot inherit `"hg38"` by saying nothing; `tests/test_genome_is_asserted_not_assumed.py`
   enumerates the subclasses and fails if yours is silent.

2. **`chorus/oracles/__init__.py`** — the import, the `ORACLES` dict, and `__all__`:
   ```python
   from .mymodel import MyModelOracle
   ORACLES = {'enformer': EnformerOracle, 'mymodel': MyModelOracle, ...}
   ```

3. **`chorus/__init__.py`** (`create_oracle`) — one `elif` branch **and** the valid-names string in
   the `else`. Miss the string and the error message for a typo silently omits your oracle:
   ```python
   elif oracle_name.lower() == 'mymodel':
       from .oracles.mymodel import MyModelOracle
       return MyModelOracle(use_environment=True, **kwargs)
   ```

4. **`chorus/mcp/server.py`** — `ORACLE_SPECS`. `tests/test_mcp.py` asserts the **exact** key set,
   so the suite goes red until you add yours; that is intentional.

**Required for the CLI to report your oracle honestly:**

5. **`chorus/core/weights_probe.py`** — `_ARTIFACT_PROBES`, so `chorus health` and
   `chorus setup` can tell "not installed" from "unhealthy" without spawning a subprocess.

6. **The two dependency probes** — `chorus/core/environment/runner.py`'s `dependencies` and
   `chorus/core/environment/manager.py`'s `oracle_deps`. These are near-duplicates that have
   drifted apart before: as of this writing `runner` was missing both `cherimoya` and
   `alphagenome_pt`, and `manager` was missing `alphagenome_pt`, so `chorus health` reported
   **Healthy** for those two even with a broken env, because an absent key means an empty
   dependency list rather than an error. Add yours to **both**.

7. **`chorus/cli/_setup_prefetch.py`** — `_DEFAULT_CTOR_KWARGS` / `_DEFAULT_LOAD_KWARGS`, needed
   whenever a bare `load_pretrained_model()` will not work. The file's own comments record LegNet
   raising `TypeError` and Cherimoya raising `InvalidAssayError` from getting this wrong.

8. **`chorus/cli/_backgrounds.py`** `_KNOWN_ORACLES` and **`chorus/cli/_cleanup.py`** — so
   `chorus backgrounds status` lists you and `chorus cleanup --oracle mymodel` removes you.

9. **`environments/chorus-mymodel.yml`** — the filename is load-bearing:
   `EnvironmentManager.list_available_oracles` globs `chorus-*.yml` and strips the prefix. Include
   the `name: chorus-mymodel` key (Step 4).

**Required before any percentile is meaningful:**

10. **Background nulls** — read [`docs/BACKGROUND_NULL_PROTOCOL.md`](docs/BACKGROUND_NULL_PROTOCOL.md)
    **§8, "Adding a new oracle"**, and follow it. This is not optional polish. If
    `classify_track_layer` returns `"other"` for your track ids, every score comes back `None`
    with no error — Sei shipped 40 built, verified, unreachable rows that way for months. You will
    also need `scripts/build_backgrounds_mymodel.py` and an `ACTIVITY_POPULATIONS` entry in
    `scripts/stamp_provenance_v4.py`.

**Three tests pin the registry and will fail until you update them** — that is the design, not an
obstacle: `tests/test_mcp.py` (exact `ORACLE_SPECS` key set),
`tests/test_reference_position_sets.py` (every oracle needs a reference SNP family), and
`tests/test_genome_is_asserted_not_assumed.py` (`training_genome` declared).

`tests/test_registries_cover_every_oracle.py` checks the sites above mechanically, so a missing
entry fails with the site named rather than surfacing as a mystery later.

### Step 6: Add Tests

Create a test file `tests/test_mymodel.py`:

```python
import pytest
import chorus


def test_mymodel_creation():
    """Test MyModel oracle creation."""
    oracle = chorus.create_oracle('mymodel', use_environment=False)
    assert oracle.oracle_name == 'mymodel'
    assert oracle.sequence_length == 524288


def test_mymodel_tracks():
    """Test track listing."""
    oracle = chorus.create_oracle('mymodel', use_environment=False)
    assays = oracle.list_assay_types()
    assert 'DNase' in assays
    
    cells = oracle.list_cell_types()
    assert 'K562' in cells


# Add more tests for predictions, etc.
```

### Step 7: Create an Example Notebook

Create `examples/notebooks/mymodel_example.ipynb` demonstrating your oracle's features (library tutorials live in `examples/notebooks/`; the per-walkthrough `examples/walkthroughs/*/notebook.ipynb` files are code-generated by `scripts/generate_walkthrough_notebooks.py` and should not be hand-written):

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
        'python=3.10',
        'pytorch>=2.0.0',
        'numpy',
        'pandas',
        # ... other conda packages
    ],
    'pip_packages': [
        'special-package==1.0.0',
        # ... other pip packages
    ],
    'post_install_commands': [
        'wget https://example.com/weights.pt -O ~/.cache/mymodel/weights.pt',
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

## Running the tests

From the `chorus` base env, at the repo root. This is the command CI runs, and a guard test enforces
that it stays the same one:

```bash
pytest tests/                             # the fast suite — no GPU, no network, ~5 min
```

No marker flag: `pytest.ini` sets `addopts = -m "not integration"`, so the exclusion is already
applied and lives in one place. That matters for the two heavier suites, because **passing no `-m`
does not mean "run everything"** — you have to override the default explicitly:

```bash
# Integration — needs the per-oracle conda envs, a GPU, and hg38 on disk. ~19 min.
pytest tests/ -m integration

# Browser — renders every committed HTML report in headless Chromium. ~2 min, no GPU.
pip install playwright && playwright install chromium
pytest tests/test_committed_reports_render_in_a_browser.py -m ""
```

The `-m ""` is load-bearing. That file sets `pytestmark = pytest.mark.integration` at module level,
so without it `pytest.ini`'s default deselects the lot and you get `no tests collected (46
deselected)` — which scrolls past looking like success. Set `CHORUS_BROWSER_SMOKE=1` only if you want
CI's reduced 3-report subset (12 tests) rather than all 46.

Both heavier suites **skip cleanly** rather than failing when their prerequisites are missing — the
browser one names exactly what is absent (`playwright not installed`, or `no chromium in <cache>`).
If you are only changing Python, the fast suite is what you need; CI runs the rest.

> **Do not run the notebooks and the integration suite at the same time**, even pinned to different
> `CUDA_VISIBLE_DEVICES`. `scripts/gate_end_to_end_determinism.py` sets its own mask and spawns two
> AlphaGenome processes with JAX preallocating, which produces a false `CUDA_ERROR_OUT_OF_MEMORY`
> in whichever suite loses the race.

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
│   └── mymodel.py          # Your oracle implementation
├── tests/
│   └── test_mymodel.py     # Tests
├── examples/
│   └── notebooks/mymodel_example.ipynb  # Example notebook
└── README.md              # Updated with your oracle info
```

## Getting Help

- Open an issue for questions
- Join discussions in existing oracle implementation PRs
- Tag maintainers for review: @pinellolab

## Current Priorities

All eight core oracles (Enformer, Borzoi, ChromBPNet/BPNet, Sei, LegNet, AlphaGenome, Cherimoya/CATv1, EPInformer-seq) are implemented — nine registered names, since AlphaGenome ships both a JAX and a PyTorch backend. We're interested in contributions for:
1. **Custom fine-tuned models** — models trained on specific tissues or conditions
2. **Species-specific oracles** — mouse, drosophila, etc.
3. **New architectures** — HyenaDNA, Evo, Nucleotide Transformer, etc.

Thank you for contributing to Chorus! Your implementation will help make genomic deep learning models more accessible to the research community.