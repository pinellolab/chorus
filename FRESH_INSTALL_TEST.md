# Chorus: Fresh Install & Validation Guide

Instructions for an agent (or human) to validate the entire Chorus system from scratch. Works on **Linux x86_64** and **macOS (Intel/Apple Silicon)**.

## Prerequisites

- **Linux**: x86_64, optionally with NVIDIA GPU + CUDA drivers (`nvidia-smi` should work)
- **macOS**: Intel or Apple Silicon
- **Miniforge** installed (`mamba` available): https://github.com/conda-forge/miniforge
- **Git** installed
- ~20 GB free disk space (models, genomes, conda environments)
- **HuggingFace account** with access to AlphaGenome weights (see Step 1b)

## Step 1: Clone and Install

```bash
git clone --branch alphagenome-oracle https://github.com/pinellolab/chorus.git
cd chorus

# Create base environment
mamba env create -f environment.yml
mamba activate chorus

# Install chorus in editable mode
pip install -e .

# Verify import
python -c "import chorus; print(f'chorus {chorus.__version__}')"
```

**Expected**: `chorus 0.1.0` printed without errors.

## Step 1b: Set Up HuggingFace Authentication (Required for AlphaGenome)

AlphaGenome model weights are hosted on a **gated HuggingFace repository**. Before running AlphaGenome predictions:

1. **Create a HuggingFace account** at https://huggingface.co/join
2. **Accept the model license** at https://huggingface.co/google/alphagenome-all-folds
3. **Generate a token** at https://huggingface.co/settings/tokens (read access is sufficient)
4. **Authenticate**:

```bash
# Option A: Set environment variable (recommended)
export HF_TOKEN="hf_your_token_here"

# Option B: Interactive login (run after chorus setup --oracle alphagenome)
mamba run -n chorus-alphagenome huggingface-cli login
```

## Step 2: Set Up All Oracle Environments

```bash
chorus setup --oracle enformer
chorus setup --oracle borzoi
chorus setup --oracle chrombpnet
chorus setup --oracle sei
chorus setup --oracle legnet
chorus setup --oracle alphagenome
```

**Expected**: Each command creates a conda environment. Platform-specific adaptations are applied automatically:
- **Linux + CUDA**: `pytorch-cuda` channel for PyTorch oracles, `jax[cuda12]` for AlphaGenome
- **macOS ARM64**: conda-forge PyTorch, `jax` + `jax-metal` for AlphaGenome (but AlphaGenome defaults to CPU at runtime since Metal is too experimental)

Verify:
```bash
chorus list
```
Should show all 6 environments as "Installed".

## Step 3: Health Check All Oracles

```bash
chorus health --timeout 300
```

**Expected**: All 6 oracles report healthy. Sei may take extra time on first run (~2.5 GB model download from Zenodo).

## Step 4: Download Reference Genome

```bash
chorus genome download hg38
```

Verify:
```bash
chorus genome list
python -c "from chorus.utils import get_genome; print(get_genome('hg38'))"
```

## Step 5: Run Full Test Suite

```bash
python -m pytest tests/ -v --tb=short
```

**Expected**: 80/80 tests pass. First run may take 10-30 minutes (model weight downloads).

### Run tests by category if needed:

```bash
# Fast unit tests only (~2 seconds)
python -m pytest tests/test_core.py tests/test_utils.py -v

# Oracle initialization tests (~30 seconds)
python -m pytest tests/test_oracles.py -v

# Prediction method tests (~1 minute)
python -m pytest tests/test_prediction_methods.py tests/test_region_manipulation.py -v

# Smoke tests - loads models and predicts (~5-10 minutes)
python -m pytest tests/test_smoke_predict.py -v
```

## Step 6: Run Comprehensive Notebook

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  examples/comprehensive_oracle_showcase.ipynb \
  --output comprehensive_oracle_showcase_executed.ipynb
```

**Expected**: All 25 code cells execute without errors, producing 8 visualizations.

Verify:
```bash
python3 << 'PYEOF'
import json
with open('examples/comprehensive_oracle_showcase_executed.ipynb') as f:
    nb = json.load(f)
errors = sum(1 for c in nb['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error')
images = sum(1 for c in nb['cells'] for o in c.get('outputs', [])
             if o.get('output_type') in ('display_data', 'execute_result')
             and 'image/png' in o.get('data', {}))
code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
print(f"Code cells: {code_cells}, Errors: {errors}, Visualizations: {images}")
assert errors == 0, f"Found {errors} errors!"
assert images >= 8, f"Expected 8+ visualizations, got {images}"
print("NOTEBOOK PASS")
PYEOF
```

## Step 7: Platform-Specific Validation (Linux GPU only)

Skip this step on macOS. On Linux with NVIDIA GPU:

```bash
python3 << 'PYEOF'
import chorus
from chorus.core.platform import detect_platform

info = detect_platform()
print(f"Platform: {info.key}")
print(f"CUDA available: {info.has_cuda}")
assert info.has_cuda, "CUDA not detected! Check nvidia-smi and CUDA drivers."
assert info.key == "linux_x86_64_cuda", f"Expected linux_x86_64_cuda, got {info.key}"
print("PLATFORM PASS")
PYEOF
```

## Success Criteria

| Check | Expected |
|---|---|
| `chorus list` | All 6 environments installed |
| `chorus health --timeout 300` | All 6 oracles healthy |
| `pytest tests/ -v` | 80/80 passed |
| Notebook execution | 25 cells, 0 errors, 8 visualizations |
| Platform (Linux GPU) | `linux_x86_64_cuda` |
| Platform (macOS ARM) | `macos_arm64` |

## Troubleshooting

- **`mamba env create` fails on macOS**: Ensure you're using the latest `environment.yml` — `coolbox` is installed via pip (conda version lacks ARM64 support).
- **AlphaGenome Metal crash on macOS**: AlphaGenome auto-selects CPU on macOS. If you see `default_memory_space is not supported`, ensure you have the latest templates.
- **`nvidia-smi` not found (Linux)**: Install NVIDIA drivers. CUDA toolkit needed for JAX/PyTorch GPU.
- **Sei download hangs**: Zenodo can be slow. Corrupt archives are automatically re-downloaded.
- **`jax[cuda12]` install fails**: Ensure CUDA 12.x is installed. For CUDA 11, edit `environments/chorus-alphagenome.yml`.
- **Out of GPU memory**: AlphaGenome with 1 MB input needs ~16 GB VRAM. Use `device='cpu'` as fallback.
- **HuggingFace `GatedRepoError`**: Accept license at https://huggingface.co/google/alphagenome-all-folds and set `HF_TOKEN`.
