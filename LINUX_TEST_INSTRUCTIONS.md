# Chorus: Full Linux GPU Test Pass

Instructions for an agent to validate the entire Chorus system from scratch on a Linux machine with NVIDIA GPU + CUDA.

## Prerequisites

- Linux x86_64 (Ubuntu 20.04+ or similar)
- NVIDIA GPU with CUDA drivers installed (`nvidia-smi` should work)
- Miniforge/Mambaforge installed (`mamba` available)
- Git installed
- ~20 GB free disk space (for models, genomes, conda envs)

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
python -c "import chorus; print(f'chorus {chorus.__version__} from {chorus.PACKAGE_DIR}')"
```

## Step 2: Set Up All Oracle Environments

Each oracle runs in an isolated conda environment. Set them all up:

```bash
chorus setup --oracle enformer
chorus setup --oracle borzoi
chorus setup --oracle chrombpnet
chorus setup --oracle sei
chorus setup --oracle legnet
chorus setup --oracle alphagenome
```

**Expected**: Each command creates a conda environment (chorus-enformer, chorus-borzoi, etc.). On Linux with CUDA, the platform adaptation system should automatically:
- Use `pytorch-cuda` channel for PyTorch oracles (Borzoi, Sei, LegNet)
- Use `jax[cuda12]` instead of `jax[cpu]` for AlphaGenome
- Use GPU-enabled TensorFlow for Enformer and ChromBPNet

Verify:
```bash
chorus list
```
Should show all 6 environments as "Installed".

## Step 3: Health Check All Oracles

```bash
chorus health --timeout 300
```

Or individually:
```bash
chorus health --oracle enformer --timeout 300
chorus health --oracle borzoi --timeout 300
chorus health --oracle chrombpnet --timeout 300
chorus health --oracle sei --timeout 300
chorus health --oracle legnet --timeout 300
chorus health --oracle alphagenome --timeout 300
```

**Expected**: All 6 oracles report healthy. If Sei times out, it may still be downloading the model (~2.5 GB from Zenodo on first run).

## Step 4: Download Reference Genome

```bash
chorus genome download hg38
```

This downloads the hg38 reference genome (~3 GB). Required for genomic coordinate predictions.

Verify:
```bash
chorus genome list
python -c "from chorus.utils import get_genome; print(get_genome('hg38'))"
```

## Step 5: Run Full Test Suite

```bash
# Run all tests (unit + smoke + integration)
python -m pytest tests/ -v --tb=short
```

**Expected**: 80/80 tests pass. The smoke tests will load each oracle in its environment and run a prediction. On first run, some oracles may need to download model weights, so this could take 10-30 minutes.

If the Sei model hasn't been downloaded yet, the Sei tests may take extra time (downloading ~2.5 GB). The corrupt archive retry logic should handle partial downloads automatically.

### Run tests by category if needed:

```bash
# Fast unit tests only (~2 seconds)
python -m pytest tests/test_core.py tests/test_utils.py -v

# Oracle initialization tests (~30 seconds)
python -m pytest tests/test_oracles.py -v

# Prediction method tests (~1 minute)
python -m pytest tests/test_prediction_methods.py tests/test_region_manipulation.py -v

# Smoke tests - actually loads models and predicts (~5-10 minutes)
python -m pytest tests/test_smoke_predict.py -v
```

## Step 6: Run Comprehensive Notebook

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  examples/comprehensive_oracle_showcase.ipynb \
  --output comprehensive_oracle_showcase_executed.ipynb
```

**Expected**: All 25 code cells execute without errors, producing 8 visualizations. Timeout is set to 1200s (20 min per cell) to accommodate first-time model downloads.

Verify the output:
```python
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

## Step 7: GPU-Specific Validation

Verify that oracles are actually using the GPU:

```python
python3 << 'PYEOF'
import chorus

# Check platform detection
from chorus.core.platform import detect_platform
info = detect_platform()
print(f"Platform: {info.key}")
print(f"CUDA available: {info.has_cuda}")
assert info.has_cuda, "CUDA not detected! Check nvidia-smi and CUDA drivers."
assert info.key == "linux_x86_64_cuda", f"Expected linux_x86_64_cuda, got {info.key}"

# Enformer (TensorFlow) - should use GPU
oracle = chorus.create_oracle('enformer', use_environment=True)
oracle.load_pretrained_model()
result = oracle.predict('A' * 393216, ['ENCFF413AHU'])
print(f"Enformer: {list(result.keys())[0]} shape={list(result.values())[0].values.shape}")

# AlphaGenome (JAX) - should use GPU
oracle = chorus.create_oracle('alphagenome', use_environment=True)
oracle.load_pretrained_model()
result = oracle.predict('A' * 1048576, [list(oracle.list_assay_types())[0] + ':*'])
print(f"AlphaGenome: prediction successful")

print("\nGPU VALIDATION PASS")
PYEOF
```

## Step 8: Performance Benchmark (Optional)

Compare GPU vs CPU prediction times:

```python
python3 << 'PYEOF'
import time
import chorus

genome_path = str(chorus.utils.get_genome('hg38'))

# Benchmark Enformer
oracle = chorus.create_oracle('enformer', use_environment=True, reference_fasta=genome_path)
oracle.load_pretrained_model()

start = time.time()
result = oracle.predict(("chr1", 1000000, 1100000), ['ENCFF413AHU'])
elapsed = time.time() - start
print(f"Enformer predict: {elapsed:.1f}s")

# Benchmark AlphaGenome
oracle = chorus.create_oracle('alphagenome', use_environment=True, reference_fasta=genome_path)
oracle.load_pretrained_model()

start = time.time()
from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
meta = get_metadata()
atac = meta.search_tracks("ATAC")
result = oracle.predict(("chr1", 1000000, 1100000), [atac.iloc[0]['identifier']])
elapsed = time.time() - start
print(f"AlphaGenome predict: {elapsed:.1f}s")
PYEOF
```

## Success Criteria

| Check | Expected |
|---|---|
| `chorus list` | All 6 environments installed |
| `chorus health --timeout 300` | All 6 oracles healthy |
| `pytest tests/ -v` | 80/80 passed |
| Notebook execution | 25 cells, 0 errors, 8 visualizations |
| Platform detection | `linux_x86_64_cuda` |
| GPU prediction | Enformer + AlphaGenome run on GPU |

## Troubleshooting

- **`nvidia-smi` not found**: Install NVIDIA drivers. CUDA toolkit needed for JAX/PyTorch GPU.
- **Sei download hangs**: Zenodo can be slow. If download exceeds 30 min, cancel and retry. Corrupt archives are automatically re-downloaded.
- **`jax[cuda12]` install fails**: Ensure CUDA 12.x is installed. For CUDA 11, manually edit `environments/chorus-alphagenome.yml` to use `jax[cuda11_pip]`.
- **PyTorch CUDA not found**: Run `python -c "import torch; print(torch.cuda.is_available())"` inside the oracle env. If False, reinstall PyTorch with CUDA: `mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`.
- **Out of GPU memory**: AlphaGenome with 1 Mb input needs ~16 GB VRAM. Use `device='cpu'` as fallback: `chorus.create_oracle('alphagenome', use_environment=True, device='cpu')`.
