# Chorus Modular Environment System

This directory contains conda environment definitions for each oracle in the Chorus library. Each oracle runs in its own isolated environment to avoid dependency conflicts.

> **Which env file should I use?** The root `environment.yml` (at the top of the
> repo) — it is the only one the install instructions in the main `README.md` use,
> and it is what `chorus setup` builds the base env from. The
> `chorus-{oracle}.yml` files in this directory are installed automatically by
> `chorus setup --oracle {name}` and you should not need to touch them directly.
>
> **`chorus-base.yml` is vestigial — do not install it.** Nothing reads it:
> `EnvironmentManager` explicitly *excludes* it when enumerating oracle envs
> (`chorus/core/environment/manager.py:129`), and `tests/test_core.py:393` asserts
> that exclusion. It is also **not** a subset of the root `environment.yml` — it
> declares `coolbox`, `h5py`, `pyyaml`, `setuptools` and `wheel`, which the root
> file does not, while the root file adds `click`, `samtools`, `htslib`,
> `pygenometracks`, `pillow` and `huggingface_hub`, which it does not. Worse, both
> files declare `name: chorus`, so creating an env from this one **collides with
> the documented base env**.

## Environment Files

- `chorus-base.yml`: **unused / vestigial** — no code reads it, and it collides with the root `environment.yml` on the env name `chorus`. Use the root `environment.yml` instead (see the note above)
- `chorus-enformer.yml`: Environment for Enformer (TensorFlow-based)
- `chorus-borzoi.yml`: Environment for Borzoi (PyTorch-based)
- `chorus-chrombpnet.yml`: Environment for ChromBPNet (TensorFlow-based)
- `chorus-sei.yml`: Environment for Sei (PyTorch-based)
- `chorus-legnet.yml`: Environment for LegNet (PyTorch-based)
- `chorus-alphagenome.yml`: Environment for AlphaGenome (JAX-based)
- `chorus-alphagenome_pt.yml`: Environment for the AlphaGenome **PyTorch** backend — installed by
  default alongside the JAX one, so Apple Silicon gets MPS
- `chorus-cherimoya.yml`: Environment for Cherimoya / CATv1 (PyTorch + Triton; CUDA on Linux,
  CPU-only on macOS)
- `chorus-epinformerseq.yml`: Environment for EPInformer-seq (PyTorch)

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

**[`CONTRIBUTING.md` → Step 5](../CONTRIBUTING.md#step-5-register-your-oracle) is the canonical
procedure.** Only the environment-file half belongs here:

1. Create `chorus-{oracle_name}.yml` in this directory. The filename is load-bearing —
   `EnvironmentManager.list_available_oracles` globs `chorus-*.yml` and strips the prefix, which is
   what makes the name appear in `chorus list`. Include the `name: chorus-{oracle_name}` key.
2. Include the oracle's own dependencies; model the file on an existing one such as
   `chorus-sei.yml`. On macOS arm64 `chorus setup` strips the CUDA packages for you
   (`chorus/core/platform.py`), so pin them normally.

> **The environment is *not* enough.** A yml alone gets your oracle into `chorus list` and nowhere
> else — it will not load, score, or appear to any MCP client until roughly ten hand-edited
> registration sites name it, and it returns `None` for every percentile until it has a background
> null. This README used to say "the environment will be automatically detected by the CLI", which
> was true of `chorus list` and misleading about everything else. See
> [CONTRIBUTING.md Step 5](../CONTRIBUTING.md#step-5-register-your-oracle) for the full list and
> [`docs/BACKGROUND_NULL_PROTOCOL.md`](../docs/BACKGROUND_NULL_PROTOCOL.md) §8 for the nulls.

## GPU Support

| Oracle | Framework | GPU Support |
|--------|-----------|-------------|
| Enformer | TensorFlow 2.14 | `nvidia-*-cu11` pip packages (in YML) |
| ChromBPNet | TensorFlow 2.8 | `nvidia-*-cu11` pip packages (in YML) |
| Borzoi | PyTorch | Bundled CUDA (automatic) |
| Sei | PyTorch | Bundled CUDA (automatic) |
| LegNet | PyTorch | Bundled CUDA (automatic) |
| AlphaGenome | JAX | Bundled CUDA (automatic) |
| AlphaGenome (`alphagenome_pt`) | PyTorch | Bundled CUDA (automatic); MPS on Apple Silicon |
| Cherimoya / CATv1 | PyTorch + Triton | Bundled CUDA on Linux. **CPU-only on macOS** — the model has no MPS path and the `triton>=3.5.1` pin ships no macOS wheel |
| EPInformer-seq | PyTorch | Bundled CUDA (automatic) |

**PyTorch/JAX oracles** detect GPUs automatically — no extra setup needed.

**TensorFlow oracles** (Enformer, ChromBPNet) require `nvidia-*-cu11` pip
packages for GPU support. These are included in the YML files and installed
automatically during `chorus setup`. The chorus environment runner sets
`LD_LIBRARY_PATH` to the nvidia package lib directories so TF can find them.

On macOS, the nvidia packages are automatically excluded by the platform
adaptation system (CUDA is not available on Mac).

## HuggingFace Access

Most oracles and all background distributions are **publicly available** and
require no HuggingFace account — but the **default `chorus setup` still needs a
token**, because it sets up AlphaGenome along with everything else and resolves
the token up front. Without one it logs
`a working HuggingFace token is required for AlphaGenome. Nothing was downloaded.`
and exits 1 **before building any environment**, so the eight token-free oracles
are not installed either.

Two ways round it if you do not want an account:

```bash
chorus setup --oracle enformer     # per-oracle setup only gates for alphagenome
chorus setup --no-weights          # skip the weight downloads (and the token gate)
```

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
