# Normalization Guide

Chorus normalizes raw oracle predictions into percentile scores using
empirical CDF (cumulative distribution function) backgrounds. This guide
documents how normalization works, what was built for each oracle, and how
to add new models and extend the backgrounds.

## How normalization works

Every oracle produces raw prediction values (e.g. predicted DNASE signal
in counts, or a log2 fold-change variant effect). These raw values are
hard to interpret across tracks and oracles because each has different
scale and dynamic range. Normalization ranks each raw value against a
pre-computed background distribution to give a percentile.

### Signed vs unsigned tracks

Each track is one of two flavours:

- **Unsigned** layers (chromatin accessibility, TF binding, histone
  marks, TSS activity, splicing): only the **magnitude** of an effect
  matters. Effects are scored as `|log2 fold-change|` and percentiles
  fall in `[0, 1]`. A value of 0.95 means "this variant's effect is
  larger than 95% of common-variant effects on this track".
- **Signed** layers (gene expression, MPRA / promoter activity, Sei
  regulatory class): direction matters too. Effects use the raw
  log-fold-change or `alt − ref` difference, and percentiles fall in
  `[-1, 1]` — sign carries the up/down direction, magnitude carries
  the rank.

The flavour is stored per-track via the `signed_flags` array in each
NPZ. The `LAYER_CONFIGS` table later in this doc lists the formula and
default sign per layer type.

### Three CDF types

For each track in each oracle, we store three empirical CDFs:

| CDF | What it normalizes | How it's built | Output range |
|-----|--------------------|----------------|--------------|
| **effect_cdfs** | Variant effect scores | Score ~10K random SNPs sampled uniformly across chr1–chr22 | [0, 1] or [-1, 1] |
| **summary_cdfs** | Baseline signal levels | Predict ~30K genomic positions (random + cCREs + TSS) | [0, 1] |
| **perbin_cdfs** | Per-bin signal values | Same positions, per-bin (oracle-dependent resolution) | [0, 1] |

**Effect percentile**: "How extreme is this variant's effect compared to
common variants?" A score at the 95th percentile means 95% of common
variants show a smaller effect for this track.

**Activity percentile**: "How active is this region compared to the
genome-wide background?" Used for baseline signal interpretation.

**Per-bin percentile**: Used for IGV-style visualization rescaling so
tracks with different dynamic ranges can be displayed on a common scale.

### Lookup mechanism

Each CDF is a sorted 1-D array of 10,000 background values per track.
Lookup uses binary search (`np.searchsorted`) to find the rank of a raw
value, then divides by the total sample count to get the percentile.

### Storage format

Each oracle stores one compressed NumPy archive:

```
~/.chorus/backgrounds/{oracle}_pertrack.npz
```

NPZ keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `track_ids` | `(N,)` | Track identifier strings |
| `effect_cdfs` | `(N, 10000)` | Sorted effect values per track |
| `summary_cdfs` | `(N, 10000)` | Sorted window-sum signal values |
| `perbin_cdfs` | `(N, 10000)` | Sorted per-bin signal values |
| `signed_flags` | `(N,)` | True if track uses signed normalization |
| `effect_counts` | `(N,)` | Actual number of background samples (effect) |
| `summary_counts` | `(N,)` | Actual number of background samples (summary) |
| `perbin_counts` | `(N,)` | Actual number of background samples (per-bin) |

Files auto-download from `huggingface.co/datasets/lucapinello/chorus-backgrounds`
on first use.

## Per-oracle details

### Enformer

| Property | Value |
|----------|-------|
| Tracks | 5,313 (ENCODE experiments: DNASE, ATAC, histone marks, CAGE, TF ChIP) |
| Model | Single pre-trained model (TFHub `deepmind/enformer/1`) |
| Input length | 393,216 bp |
| Output bins | 896 bins × 128 bp = 114,688 bp |
| Build script | `scripts/build_backgrounds_enformer.py` |
| Conda env | `chorus-enformer` (TensorFlow) |
| NPZ size | ~523 MB |

**Layer types used**: chromatin_accessibility, tf_binding, histone_marks,
tss_activity, gene_expression, splicing.

**Effect scoring**: log2 fold-change of window-sum (501bp for chromatin/TF/
histone/CAGE/splicing; exon-based for gene expression). Pseudocount = 1.0
for window-based, 0.001 for expression.

**Extensibility**: Fixed model — tracks are defined by ENCODE metadata.
No custom model support (single Enformer weights).

---

### Borzoi

| Property | Value |
|----------|-------|
| Tracks | 7,611 (ENCODE + Roadmap: DNASE, ATAC, histone, CAGE, RNA-seq) |
| Model | 4 replicate folds (HuggingFace `johahi/borzoi-replicate-{0..3}`); chorus default = fold 0 |
| Input length | 524,288 bp |
| Output bins | 6,144 bins × 32 bp |
| Build script | `scripts/build_backgrounds_borzoi.py` |
| Conda env | `chorus-borzoi` (PyTorch) |
| NPZ size | ~766 MB |

**Layer types used**: Same as Enformer (chromatin_accessibility, tf_binding,
histone_marks, tss_activity, gene_expression, splicing).

**Extensibility**: Fixed model — single set of weights per fold. Tracks
defined by Borzoi metadata.

---

### ChromBPNet

| Property | Value |
|----------|-------|
| Tracks | 786 (42 ChromBPNet ATAC/DNASE + 744 BPNet/CHIP) |
| Model | One model per track (ENCODE ChromBPNet + JASPAR BPNet) |
| Input length | 2,114 bp (both ChromBPNet and BPNet) |
| Output length | 1,000 bp at 1-bp resolution |
| Build script | `scripts/build_backgrounds_chrombpnet.py` |
| Conda env | `chorus-chrombpnet` (TensorFlow) |
| NPZ size | ~78 MB |

**Track families**:
- **ATAC/DNASE** (42 tracks): ENCODE-published ChromBPNet models.
  Registry: `chrombpnet_globals.py` → `CHROMBPNET_MODELS_DICT`.
- **CHIP** (744 tracks): JASPAR BPNet TF x cell-type models.
  Registry: `chrombpnet_globals.py` → `iter_unique_bpnet_models()` from
  `chrombpnet_JASPAR_metadata.tsv`.

**Layer types used**: chromatin_accessibility (ATAC/DNASE), tf_binding (CHIP).

**Effect scoring**: log2 fold-change of 501-bp window sum at 1-bp resolution.
Pseudocount = 1.0.

**BPNet strand handling**: BPNet CHIP models output 2-stranded profiles
`(B, L, 2)`. Strands are summed before scoring.

**Extensibility**: Supports custom models via `is_custom=True` in
`load_pretrained_model()`. To add a new track and CDF row in one pass,
register it in `chrombpnet_globals.py` and run the build script with
`--only-missing` (see "Bring your own ChromBPNet model" below). The
script auto-merges new rows into the existing NPZ via
`PerTrackNormalizer.append_tracks`.

**Sharded build**: For a full rebuild of the 744 BPNet/CHIP CDFs
across multiple GPUs (~6 hours wall on 6× CUDA cards):

```bash
bash scripts/run_bpnet_cdf_build.sh  # 6-GPU parallel build
```

For a single custom track (BYOM walkthrough below), use
`--only-missing` instead — the 6-GPU script is overkill and would
re-score all 786 tracks.

---

### Sei

| Property | Value |
|----------|-------|
| Tracks | 40 (high-level regulatory element classes — used for percentile scoring) |
| Underlying profiles | 21,907 chromatin profiles internally; Sei aggregates these into 40 classes for variant scoring |
| Model | Single pre-trained model (Zenodo) |
| Input length | 4,096 bp |
| Output | 40 sequence class scores |
| Build script | `scripts/build_backgrounds_sei.py` |
| Conda env | `chorus-sei` (PyTorch) |
| NPZ size | ~2.8 MB |

**Layer types used**: regulatory_classification (signed, [-1, 1]).

**Effect scoring**: Direct difference (alt - ref) of class logits.

**Extensibility**: Fixed model — 40 classes defined by Sei architecture.
No custom model support.

---

### LegNet

| Property | Value |
|----------|-------|
| Tracks | 3 (K562, HepG2, WTC11 MPRA) |
| Model | One model per cell type |
| Input length | 200 bp (`LEGNET_WINDOW`) |
| Output | 1 scalar (promoter activity) |
| Build script | `scripts/build_backgrounds_legnet.py` |
| Conda env | `chorus-legnet` (PyTorch) |
| NPZ size | ~0.2 MB |

**Layer types used**: promoter_activity (signed, [-1, 1]).

**Effect scoring**: Direct difference (alt - ref) of predicted activity.
Pseudocount = 0.0.

**Extensibility**: Cell types defined in `legnet_globals.py` →
`LEGNET_AVAILABLE_CELLTYPES`. New cell types can be added by extending
the list and providing model weights.

---

### AlphaGenome

| Property | Value |
|----------|-------|
| Tracks | 5,168 (human functional genomics: ATAC, DNASE, histone, CAGE, RNA, CHIP) |
| Model | Single gated model (HuggingFace `google/alphagenome-all-folds`, requires auth) |
| Input length | 1,048,576 bp (up to 1 MB) |
| Output | 1-bp resolution for most modalities (some at 128 bp) |
| Build script | `scripts/build_backgrounds_alphagenome.py` |
| Conda env | `chorus-alphagenome` (JAX) |
| NPZ size | ~263 MB |

**Layer types used**: chromatin_accessibility, tf_binding, histone_marks,
tss_activity, gene_expression.

**Extensibility**: Fixed model — tracks defined by AlphaGenome metadata.
Requires HuggingFace authentication (`HF_TOKEN`).

---

### EPInformer-seq

| Property | Value |
|----------|-------|
| Tracks | 11 (K562, GM12878, HepG2, A549, H1, HeLa, HMEC, HSMM, HUVEC, NHEK, NHLF) |
| Model | **One model per cell type** — `PerCellProfileNet` + frozen per-cell `BiasNet` (ChromBPNet-style sequence-bias subtraction) |
| Input length | 1,024 bp (`EPINFORMERSEQ_WINDOW`) |
| Output | 1 scalar per region: `sqrt(max DNase × max H3K27ac)` over the central 256 bp (composite enhancer activity); also exposes `Enhancer_DNase` and `Enhancer_H3K27ac` single-channel scalars |
| Build script | `scripts/build_backgrounds_epinformerseq_v2_percell.py` |
| Conda env | `chorus-epinformerseq` (PyTorch) |
| NPZ size | ~0.8 MB |

**Layer types used**: `Enhancer_H3K27ac_DNase` (custom — chromatin-accessibility-like, unsigned).

**Effect scoring**: `|log2((alt + 1) / (ref + 1))|` on the scalar
`sqrt(D × H)` activity. Pseudocount = 1.0. Unsigned (variants are scored by
the magnitude of their effect on regulatory activity).

**Training**: 11 per-cell PerCellProfileNet checkpoints (no FiLM / no cell
embedding — one independent model per cell). Trained on per-rep ENCODE BAMs
(ENCODE `recommended=true` single rep per assay) over the DNase ∪ H3K27ac
peak summit union, fold-10 leave-chrom-out CV (chr11 + chr21 held out).
ChromBPNet recipe: multinomial-NLL profile loss + MSE on log10(total + 1)
count head; per-cell frozen `BiasNet` subtracts Tn5/MNase sequence
preference in logit space.

**Extensibility**: Add a cell by retraining `PerCellProfileNet` + `BiasNet`
for it (see `percell_v1_vs_v2/v2_1024bp/{train,train_bias}.py` in the
research cluster), staging the new ckpts under
`downloads/epinformerseq/{per_cell,bias}/<cell>/`, and extending
`EPINFORMERSEQ_AVAILABLE_CELLTYPES` in `chorus/oracles/epinformerseq_source/globals.py`.
Per-bin CDFs are absent (scalar-output oracle); IGV display falls back to
`summary_cdfs` via `rescale_for_display`.

## Layer configuration reference

From `chorus/analysis/scorers.py` — `LAYER_CONFIGS`:

| Layer | Window | Formula | Pseudocount | Signed | Used by |
|-------|--------|---------|-------------|--------|---------|
| `chromatin_accessibility` | 501 bp | log2fc | 1.0 | No [0,1] | Enformer, Borzoi, ChromBPNet, AlphaGenome |
| `tf_binding` | 501 bp | log2fc | 1.0 | No [0,1] | Enformer, Borzoi, ChromBPNet, AlphaGenome |
| `histone_marks` | 2001 bp | log2fc | 1.0 | No [0,1] | Enformer, Borzoi, AlphaGenome |
| `tss_activity` | 501 bp | log2fc | 1.0 | No [0,1] | Enformer, Borzoi, AlphaGenome |
| `gene_expression` | exon-based | logfc | 0.001 | Yes [-1,1] | Enformer, Borzoi, AlphaGenome |
| `promoter_activity` | N/A | diff | 0.0 | Yes [-1,1] | LegNet |
| `regulatory_classification` | N/A | diff | 0.0 | Yes [-1,1] | Sei |
| `splicing` | 501 bp | log2fc | 1.0 | No [0,1] | Enformer, Borzoi |

**Splicing layer note**: `splicing` covers Enformer / Borzoi tracks
whose ENCODE assay metadata indicates a splicing-related readout
(e.g. shRNA-knockdown RNA-seq used to derive splice-site usage). It
shares the unsigned-log2FC math with chromatin_accessibility but is
listed separately so future oracle-specific overrides (windowing, mask)
can override only the splicing tracks.

**Formula definitions**:
- `log2fc`: `log2((alt + pseudocount) / (ref + pseudocount))`
- `logfc`: `log((alt + pseudocount) / (ref + pseudocount))`
- `diff`: `alt - ref`

## Build pipeline

The CDF build pipeline for each oracle follows the same pattern:

```
1. Sample variant positions (~10K common SNPs from gnomAD)
2. Sample baseline positions (~30K: random + cCREs + TSS-proximal)
3. For each track/model:
   a. Score all variants → collect |effect| values via reservoir sampling
   b. Score all baselines → collect window-sum + per-bin values
4. Compact reservoirs to 10,000-point CDFs
5. Save as {oracle}_pertrack.npz via PerTrackNormalizer.build_and_save()
```

Build scripts support `--part variants`, `--part baselines`, `--part both`,
and `--part merge` for splitting GPU-intensive scoring from CPU merging.

For ChromBPNet, additional flags support distributed builds:
- `--shard N --shard-of M`: process shard N of M total shards
- `--only-missing`: skip tracks already in the existing NPZ
- `--assay CHIP|ATAC_DNASE`: build only one assay family

## Walkthrough: Bring your own ChromBPNet or BPNet model

You trained a ChromBPNet or BPNet model outside chorus (custom cell
type, your own ChIP-seq, a tweaked architecture) and want to use it
with full percentile normalization.

### Step 1: Locate your `.h5` weights file

Chorus's `weights=` parameter is a **path to the actual `.h5` (or
`.keras`) file**, not a parent directory:

```
# ChromBPNet — output of the chrombpnet training pipeline:
/path/to/your_run/models/chrombpnet_nobias.h5

# BPNet (CHIP/TF) — single-file model:
/path/to/your_run/model.h5
```

If you have a directory layout instead, point `weights=` at the file
inside it; chorus does not search for or auto-resolve filenames.

### Step 2: Smoke-test the load + predict

```python
from chorus.oracles.chrombpnet import ChromBPNetOracle

oracle = ChromBPNetOracle(use_environment=False)

# is_custom=True bypasses the built-in registry — `cell_type` becomes
# part of the track_id ("ATAC:my_cell_line") and is otherwise free-form.
oracle.load_pretrained_model(
    assay="ATAC",                                      # "ATAC" / "DNASE" / "CHIP"
    cell_type="my_cell_line",
    weights="/path/to/your_run/models/chrombpnet_nobias.h5",
    is_custom=True,
)

# Quick check — should return finite values
result = oracle.predict(("chr1", 1000000, 1002114))
print(next(iter(result.items())))  # (track_id, OraclePredictionTrack)
```

For CHIP/BPNet, also pass `TF="MyTF"` so the track_id becomes
`CHIP:my_cell_line:MyTF`:

```python
oracle.load_pretrained_model(
    assay="CHIP", cell_type="my_cell_line", TF="MyTF",
    weights="/path/to/your_run/model.h5",
    is_custom=True,
)
```

### Step 3: Build the per-track CDF for your model

The CDF needs ~10K random SNPs (effect rows) and ~30K baseline positions
(summary + perbin rows) through your model. Rough wall-clock per model:
~22 min for ChromBPNet on Apple M3 Ultra Metal, ~5 min on a CUDA A100;
BPNet (smaller architecture) is ~2× faster either way. The
[full sharded build](#sharded-build) of all 786 ChromBPNet+BPNet
models takes about 6 hours on a single GPU.

**Recommended path — bypass the build script and call its scoring
helpers directly** so you don't have to fork the registry. Save your
results into an NPZ that `chorus backgrounds add-tracks` accepts:

```python
from pathlib import Path
import numpy as np

from chorus.oracles.chrombpnet import ChromBPNetOracle
from chorus.analysis.normalization import PerTrackNormalizer

# ---- 1. Load your custom model ----
oracle = ChromBPNetOracle(use_environment=False)
oracle.load_pretrained_model(
    assay="ATAC", cell_type="my_cell_line",
    weights="/path/to/your_run/models/chrombpnet_nobias.h5",
    is_custom=True,
)

# ---- 2. Reuse the build script's scoring helpers (variants + baselines) ----
# scripts/build_backgrounds_chrombpnet.py exposes:
#   - get_sequence(ref, chrom, pos)
#   - predict_profiles_batch(model, seqs)
#   - score_window_sum(profile)  → 501-bp center sum
#   - compute_effect(ref_val, alt_val)  → log2 fold-change
# Copy or import those + drive your own variant set / baseline set.
# Put effects, summaries, perbin into 1×10000 sorted arrays each.

effect_cdf  = np.sort(np.asarray(your_abs_log2fc_per_snp,  dtype=np.float32))
summary_cdf = np.sort(np.asarray(your_window_sums,         dtype=np.float32))
perbin_cdf  = np.sort(np.asarray(your_per_bin_values,      dtype=np.float32))

# ---- 3. Save in the per-track NPZ schema ----
out = Path("/tmp/my_chrombpnet_track.npz")
np.savez_compressed(
    out,
    track_ids=np.array(["ATAC:my_cell_line"], dtype="U"),
    effect_cdfs=effect_cdf.reshape(1, -1),       # (1, 10000)
    summary_cdfs=summary_cdf.reshape(1, -1),     # (1, 10000)
    perbin_cdfs=perbin_cdf.reshape(1, -1),       # (1, 10000)
    signed_flags=np.array([False]),              # ATAC/DNASE/CHIP are unsigned
    effect_counts=np.array([len(your_abs_log2fc_per_snp)]),
    summary_counts=np.array([len(your_window_sums)]),
    perbin_counts=np.array([len(your_per_bin_values)]),
)
print(f"Wrote {out}")
```

### Step 4: Append your CDF row to the main NPZ

```bash
chorus backgrounds add-tracks --oracle chrombpnet --npz /tmp/my_chrombpnet_track.npz
```

The source NPZ must use the schema written in Step 3 — keys
`track_ids`, `effect_cdfs`, `summary_cdfs`, `perbin_cdfs` (optional),
`signed_flags`, and the matching `_counts` arrays. Duplicates against
existing `track_ids` are skipped automatically.

You can also call the underlying API from Python:

```python
from chorus.analysis.normalization import PerTrackNormalizer

path, n_added = PerTrackNormalizer.append_tracks(
    oracle_name="chrombpnet",
    new_track_ids=["ATAC:my_cell_line"],
    new_effect_cdfs=effect_cdf.reshape(1, -1),
    new_summary_cdfs=summary_cdf.reshape(1, -1),
    new_perbin_cdfs=perbin_cdf.reshape(1, -1),
    new_signed_flags=np.array([False]),
    new_effect_counts=np.array([len(your_abs_log2fc_per_snp)]),
    new_summary_counts=np.array([len(your_window_sums)]),
    new_perbin_counts=np.array([len(your_per_bin_values)]),
)
print(f"Appended {n_added} new tracks → {path}")
```

### Step 5: Verify

```bash
chorus backgrounds status --oracle chrombpnet
# Track count should be one higher (e.g. 786 → 787) and your new
# `ATAC:my_cell_line` row should appear.
```

From now on any chorus call that scores `assay_ids=["ATAC:my_cell_line"]`
through your custom model returns percentile-normalised effects.

## Walkthrough: Bring your own LegNet model

You trained a LegNet MPRA model for a cell type chorus doesn't ship
(default registry covers K562, HepG2, WTC11).

### Step 1: Register your cell type

`LegNetOracle.__init__` validates `cell_type` against
`LEGNET_AVAILABLE_CELLTYPES`, so add yours **before** instantiating:

```python
# chorus/oracles/legnet_source/legnet_globals.py
LEGNET_AVAILABLE_CELLTYPES = ["K562", "HepG2", "WTC11", "my_cell"]
```

### Step 2: Organize your weights into LegNet's expected layout

LegNet resolves model paths as
`{model_dir}/{assay}_{cell_type}/{model_id}/weights.ckpt`. So if you
pass `model_dir="/path/to/my_legnet_models"`, lay your files out like:

```
/path/to/my_legnet_models/
  LentiMPRA_my_cell/
    example/                   # the default model_id
      weights.ckpt
      config.json              # hyperparameters
```

(`assay` defaults to `"LentiMPRA"`; `model_id` defaults to `"example"`.
Override either via the constructor.)

### Step 3: Smoke-test the load + predict

LegNet input length is 200 bp (`LEGNET_WINDOW = 200`):

```python
from chorus.oracles.legnet import LegNetOracle

oracle = LegNetOracle(
    cell_type="my_cell",
    use_environment=False,
    model_dir="/path/to/my_legnet_models",   # parent dir; see layout above
)
oracle.load_pretrained_model()  # picks up weights from model_dir

# 200 bp test sequence
result = oracle.predict("ACGT" * 50)
print(result)
```

### Step 4: Build CDF and append

Same shape as the ChromBPNet walkthrough above (Step 3 there) but with
`signed_flags=np.array([True])` because `promoter_activity` is a
**signed** layer (Δ-formula, [-1, 1]):

```python
import numpy as np
from chorus.analysis.normalization import PerTrackNormalizer

# After scoring your variants and baselines yourself:
path, n_added = PerTrackNormalizer.append_tracks(
    oracle_name="legnet",
    new_track_ids=["LentiMPRA:my_cell"],
    new_effect_cdfs=effect_cdf.reshape(1, -1),
    new_summary_cdfs=summary_cdf.reshape(1, -1),
    new_signed_flags=np.array([True]),   # signed for MPRA
    new_effect_counts=np.array([n_variants]),
    new_summary_counts=np.array([n_baselines]),
)
print(f"Appended {n_added} → {path}")
```

### Step 5: Verify

```bash
chorus backgrounds status --oracle legnet
# Should show 4 tracks now (was 3) including LentiMPRA:my_cell.
```

## Walkthrough: Adding a new oracle

If a collaborator wants to add an entirely new oracle to Chorus, here is
the full checklist. The architecture is designed so that each oracle is
a self-contained module with a standardized interface.

### Step 1: Create the oracle class

Create `chorus/oracles/my_oracle.py` subclassing `OracleBase`:

```python
from typing import List, Tuple
from ..core.base import OracleBase
from ..core.result import OraclePrediction

class MyOracle(OracleBase):
    def __init__(self, use_environment=True, **kwargs):
        super().__init__(use_environment=use_environment, **kwargs)
        self.oracle_name = "myoracle"   # must match the conda env suffix
        # Set any model-specific defaults here (sequence_length, bin_size, …).

    # ── Required abstract methods (6 total) ──

    def load_pretrained_model(self, weights: str = None) -> None:
        """Load model weights from disk or download."""
        # Download weights if not cached, then load into self.model
        self.loaded = True

    def list_assay_types(self) -> List[str]:
        return ["DNASE"]  # whatever your oracle predicts

    def list_cell_types(self) -> List[str]:
        return ["K562", "HepG2"]

    def _predict(self, seq: str, assay_ids: List[str]) -> OraclePrediction:
        """Run the model and return predictions.

        Called by OracleBase.predict() after input validation.
        Return an OraclePrediction with Track objects, one per assay_id.
        """
        # Your inference code here
        ...

    def _get_context_size(self) -> int:
        """Required input length the model expects (in bp)."""
        return 2114

    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Min and max sequence lengths the model accepts (in bp)."""
        return (2114, 2114)
```

**Required methods** (from `OracleBase`, all `@abstractmethod`):
- `load_pretrained_model(weights)` — load model weights
- `list_assay_types()` — return available assay types
- `list_cell_types()` — return available cell types
- `_predict(seq, assay_ids)` — internal prediction; returns `OraclePrediction`
- `_get_context_size()` — required input length in bp
- `_get_sequence_length_bounds()` — `(min, max)` accepted seq length

**Key conventions**:
- `self.oracle_name` is set in `__init__` as a lowercase string matching
  the conda env suffix (e.g. `chorus-myoracle` env → `oracle_name="myoracle"`).
- Model weights go in `~/chorus/downloads/{oracle_name}/`.
- Use `self.device` for GPU/CPU selection (auto-detected or user-set).

### Step 2: Register the oracle

Add it to `chorus/oracles/__init__.py`:

```python
from .my_oracle import MyOracle

ORACLES = {
    # ... existing oracles ...
    'myoracle': MyOracle,
}
```

This is required for `chorus.create_oracle("myoracle")` and the prefetch
helpers to find your class.

### Step 3: Create the conda environment

Create `environments/chorus-myoracle.yml`:

```yaml
name: chorus-myoracle
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pysam
  - pip
  - pip:
    - torch>=2.0   # or tensorflow, jax — whatever your model needs
```

Note: oracle envs **don't** install chorus itself; the build scripts
add the repo to `sys.path` at runtime. Keeps the env minimal and
isolates your model's deps from chorus's.

Then build it:

```bash
mamba env create -f environments/chorus-myoracle.yml
```

`chorus setup --oracle myoracle` will then build this env automatically
(the CLI scans `environments/chorus-*.yml` for available oracles), but
the oracle still has to be registered in `ORACLES` (Step 2) for
prefetch and discovery to work end-to-end.

### Step 4: Write the CDF build script

The CDF build is where most of the layer-specific judgment lives.
A new oracle can't reuse an existing build script verbatim because
**what you sample matters more than how you score**. The same model
will produce wildly different percentiles depending on whether your
"baseline" reservoir was drawn from random genome (mostly silent),
ENCODE cCREs (mostly active), DHS summits (open chromatin),
TSSs (transcription start), exons (gene bodies), or splice junctions.
This section is the recipe.

#### What goes in each CDF

Each oracle ships **three CDF matrices** (`(n_tracks, n_points)` each)
in a single `{oracle}_pertrack.npz`:

| CDF | Sampled from | Used for |
|---|---|---|
| `effect_cdfs` | **|effect|** (or signed effect) of ~10–20 K SNPs against the same track | "How unusual is this variant's effect?" → `Effect %ile` column in reports |
| `summary_cdfs` | **window-sum** (or layer-specific aggregate) at ~30–35 K genomic positions | "How active is this site genome-wide?" → `Activity %ile` column |
| `perbin_cdfs` | **individual bin values** at the same baseline positions, dozens of bins per position | IGV / matplotlib / CoolBox display rescale via `rescale_for_display` |

Each row is a **sorted** array (the empirical CDF). Lookups use binary
search; see [Lookup mechanism](#lookup-mechanism). All three CDFs use the
same baseline position set — you sample the positions once and feed
them into both `summary_cdfs` (window aggregate) and `perbin_cdfs`
(per-bin values within the window).

Rule of thumb for sample counts: **~20 K samples per track** for
effect, **~30–35 K** for summary, **~1 M total** for perbin (tens of
bins × thousands of positions). Below that you start seeing
visible discreteness in the percentile output.

#### Reservoir sampling in practice

For long runs (e.g. 1 M perbin samples × 786 tracks), build the CDF
incrementally using the `ReservoirSampler` pattern from
`scripts/build_backgrounds_chrombpnet.py:ReservoirSampler` —
fixed-capacity uniform sampling so memory stays bounded. The sampler
exposes `to_cdf_matrix(n_points=10000)` which returns the compact
sorted CDF directly.

#### What to sample, layer by layer

Helpers used below all live in `chorus.utils.annotations`:
`sample_ccre_positions()`, `sample_dhs_positions()`,
`load_dhs_vocabulary()`, `get_gene_tss()`, `get_gene_exons()`,
`get_screen_ccres()`. All of them are seeded and produce
deterministic outputs.

**`chromatin_accessibility` (DNASE / ATAC) — sharp, focused peaks.**
Window 501 bp, log2FC, unsigned.

- **Effect**: ~10 K random genome-wide SNPs (existing) + ~10 K SNPs at
  random offsets within ±150 bp of DHS peak summits
  (`sample_dhs_positions(10_000, ...)`). Mixing in DHS-anchored SNPs
  shifts the interpretation from "unusual among all genomic SNPs"
  toward "unusual among SNPs at real regulatory elements" — much more
  discriminating for cell-type-specific peaks. See
  `scripts/build_backgrounds_chrombpnet.py` `--n-dhs-variants`
  for the canonical implementation.
- **Summary / perbin baseline**: ~25 K random + cCRE positions
  (`sample_ccre_positions()`) + ~5 K DHS summits
  (`sample_dhs_positions(5_000, ...)`).

**`tf_binding` (ChIP-TF) — sharp binding peaks.** Window 501 bp,
log2FC, unsigned.

- **Effect**: random SNPs + (optionally) DHS-anchored SNPs as above.
  The 744 BPNet/CHIP tracks in the canonical chrombpnet NPZ use the
  same random+DHS recipe — TF binding sites overlap heavily with
  open chromatin, so DHS sampling is the cheapest "in TF-relevant
  regions" enrichment.
- **Summary / perbin baseline**: random + cCRE (`PLS`, `dELS`, `pELS`,
  `CA-CTCF`, `CA-TF`) + DHS summits.

**`histone_marks` (ChIP-Histone) — broad domains.** Window **2001 bp**
(not 501 — histones cover wider regions), log2FC, unsigned.

- **Effect**: same random + DHS recipe; the wider scoring window
  averages over the histone domain.
- **Summary / perbin baseline**: random + cCRE (especially `pELS`
  and `dELS`) + DHS summits.

**`tss_activity` (CAGE / PRO-CAP) — TSS-localized.** Window 501 bp,
log2FC, unsigned.

- **Effect**: random SNPs is fine — CAGE peaks are sharp, sample
  quality dominates over enrichment.
- **Summary / perbin baseline**: random + cCRE **+ explicitly
  GENCODE TSSs** (use `get_gene_tss()` over a list of protein-coding
  genes, sample ~5–10 K). Without TSSs the summary distribution is
  dominated by silent regions and produces inflated "this CAGE site
  is active" percentiles. See `build_backgrounds_borzoi.py` and
  `build_backgrounds_alphagenome.py` for how the 'CAGE summary
  routing' adds TSSs while skipping cCREs.

**`gene_expression` (RNA-seq) — broad gene bodies; signed effects.**
Window **N/A — exon-based**, logfc, **signed**.

- **Effect**: random SNPs scored as `mean(prediction[exon_bins])` for
  each gene — i.e. for each variant, find the nearest gene's exons,
  average the model's per-bin predictions over those exon bins, then
  compute the log fold change of that exon-mean vs ref. This is
  RNA-specific: gene expression is integrated over exons, not at one
  point. See `build_backgrounds_borzoi.py:exon_bins_for_window` for
  the exon-mask machinery.
- **Summary / perbin baseline**: gene-body midpoints (use
  `get_gene_exons()` per gene, sample midpoint per gene; ~10–15 K
  genes). RNA tracks are signed because alleles can repress
  transcription — effect/summary CDFs include negative values, so
  set `signed_flags[idx] = True` for these tracks. Display rescale
  for signed tracks goes through `signed_floor_rescale_batch`
  (symmetric `[-3, +3]`) instead of the unsigned floor-rescale.

**`splicing` (splice site / SPLICE_SITES) — sharp signals at
splice junctions.** Window 501 bp, log2FC, unsigned.

- **Effect**: random SNPs + (highest signal-to-noise) **GENCODE exon
  boundaries**. Sample ~5–10 K splice donors / acceptors using
  `get_gene_exons()` and taking each exon's `start` and `end` minus
  one (donor / acceptor positions). Random SNPs alone give a
  splicing-effect distribution dominated by intergenic regions where
  the splicing model predicts ~0 anyway.
- **Summary / perbin baseline**: splice sites + cCRE (`PLS`).

**`promoter_activity` (LentiMPRA) — element-level point predictions;
signed.** No window (the model emits one number per ~200 bp element),
diff, **signed**.

- **Effect**: random ~200 bp regulatory regions tested by the model.
  No genome-wide SNP scan applies because the model doesn't have a
  spatial output — it's just `alt_score - ref_score` per element.
- **Summary / perbin baseline**: same set of ~30 K random + cCRE
  elements scored by the model. `perbin_cdfs` is **absent** for
  LegNet (there's no per-bin axis); IGV display falls back to
  `summary_cdfs` via `rescale_for_display`'s `_find_matching_cdf`
  fallback. Since LegNet predictions can be repressive, set
  `signed_flags = ones`.

**`regulatory_classification` (Sei) — sequence-class probabilities;
signed.** No window, diff, **signed**.

- **Effect**: random SNPs + cCRE-anchored SNPs. Each track is a class
  ("CTCF", "Promoter", "Enhancer", …); the effect is `alt_class_prob
  − ref_class_prob`.
- **Summary / perbin baseline**: random + cCRE positions; cCRE
  enrichment matters because most of the genome is "Background"
  class, and you want enough samples in each non-background class
  for the per-class CDFs to be informative. Sei has no spatial
  output, so `perbin_cdfs` is absent; same fallback as LegNet.

#### Common pitfalls

- **All-random baseline** for any layer that has cell-type-specific
  signal: the genome is mostly silent, so your `summary_cdfs` p99
  becomes "the few real peaks that randomly landed in the sample",
  and any moderately-active site looks like a 99th-percentile site.
  Mix in DHS / cCRE / TSS as appropriate.
- **Forgetting `signed_flags=True`** for RNA / MPRA / Sei tracks:
  the rescaler will silently clip negative effects to 0 and the
  repressive half of the distribution disappears from IGV.
- **Mismatched scoring window between build and live**: the build
  script's window (`profile[ws:we]`) must match what
  `oracle.score_track_effect()` does at inference time. ChromBPNet's
  build uses `profile[250:751]` of the 1000-bin output; the live
  scorer uses `score_region(POS-250, POS+251, 'sum')` which lands
  on the same bins thanks to `prediction_interval = input_interval`
  (PR #76). If you change the build window, update the live scorer
  in lockstep — otherwise scores are correct but percentiles are
  miscalibrated.
- **Per-bin samples drawn only from peak centers**: the perbin CDF
  is for *display rescaling*, so it needs bins from peak flanks too
  to characterize the noise floor. Sample 16-32 random bins per
  baseline position rather than one bin at the center.
- **Boundary effects**: skip variants/positions within ~5 Mb of
  chromosome ends. ENCODE peak callers and most reference genomes
  have noisy edges that produce outlier predictions and skew the
  CDFs.

#### Build script skeleton

```python
import argparse, numpy as np
from chorus.analysis.normalization import PerTrackNormalizer
from chorus.utils.annotations import (
    sample_ccre_positions, sample_dhs_positions, get_gene_tss,
)

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "both", "merge"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--n-variants", type=int, default=10_000)
parser.add_argument("--n-dhs-variants", type=int, default=10_000,
    help="DHS-anchored SNPs (summit ±150 bp). 0 to disable.")
parser.add_argument("--n-dhs-peaks", type=int, default=5_000)
args = parser.parse_args()

ORACLE_NAME = "myoracle"
track_ids = ["DNASE:K562", "DNASE:HepG2"]
n_tracks = len(track_ids)

# ── Variant set: random genome SNPs + DHS-anchored SNPs ──
variant_positions = generate_random_snps(args.n_variants, seed=42)
if args.n_dhs_variants > 0:
    variant_positions += dhs_anchored_snps(
        sample_dhs_positions(args.n_dhs_variants, seed=43),
        max_offset_bp=150, seed=44,
    )

# ── Baseline position set: random + cCRE + (layer-specific) ──
baseline_positions = []
baseline_positions.extend(random_genome_positions(20_000, seed=99))
baseline_positions.extend(sample_ccre_positions(seed=42))      # ~20 K
baseline_positions.extend(sample_dhs_positions(args.n_dhs_peaks, seed=567))
# For tss_activity / gene_expression layers, also:
#   baseline_positions.extend(get_gene_tss(...).head(10_000))
# For splicing layer:
#   baseline_positions.extend(get_splice_sites(...))

# ── Score each track ──
effect_sampler = ReservoirSampler(n_tracks, capacity=50_000)
summary_sampler = ReservoirSampler(n_tracks, capacity=50_000)
perbin_sampler = ReservoirSampler(n_tracks, capacity=50_000)
signed_flags = np.zeros(n_tracks, dtype=bool)   # True per-track for signed layers

for ti, track_id in enumerate(track_ids):
    layer = classify_layer(track_id)            # your mapping
    signed_flags[ti] = is_signed_layer(layer)
    for snp in variant_positions:
        ref_v, alt_v = score_track(model, snp, layer)
        eff = compute_effect(ref_v, alt_v, layer)   # log2fc / logfc / diff
        if not signed_flags[ti]:
            eff = abs(eff)
        effect_sampler.add(ti, eff)
    for pos in baseline_positions:
        win_value = score_window(model, pos, layer)
        summary_sampler.add(ti, win_value)
        for bin_value in sample_bins(model, pos, n=32):
            perbin_sampler.add(ti, bin_value)

# ── Save the merged NPZ ──
PerTrackNormalizer.build_and_save(
    oracle_name=ORACLE_NAME,
    track_ids=track_ids,
    effect_cdfs=effect_sampler.to_cdf_matrix(n_points=10_000),
    summary_cdfs=summary_sampler.to_cdf_matrix(n_points=10_000),
    perbin_cdfs=perbin_sampler.to_cdf_matrix(n_points=10_000),
    effect_counts=effect_sampler.get_counts(),
    summary_counts=summary_sampler.get_counts(),
    perbin_counts=perbin_sampler.get_counts(),
    signed_flags=signed_flags,
)
```

The full reference implementation (with sharded multi-GPU support,
incremental rebuilds, and the full DHS hooks) lives in
`scripts/build_backgrounds_chrombpnet.py`. For RNA-style exon-based
scoring see `scripts/build_backgrounds_borzoi.py:exon_bins_for_window`.
For element-level signed scoring (LentiMPRA / Sei) see
`scripts/build_backgrounds_legnet.py` and `_sei.py`.

Run:

```bash
mamba run -n chorus-myoracle python scripts/build_backgrounds_myoracle.py \
  --part both --gpu 0 --n-variants 10000 --n-dhs-variants 10000 --n-dhs-peaks 5000
```

### Step 5: Upload backgrounds to HuggingFace (optional)

If you want other users to auto-download your backgrounds:

```python
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(Path.home() / ".chorus/backgrounds/myoracle_pertrack.npz"),
    path_in_repo="myoracle_pertrack.npz",
    repo_id="lucapinello/chorus-backgrounds",
    repo_type="dataset",
    commit_message="Add myoracle backgrounds",
)
```

After this, any user running `chorus setup --oracle myoracle` will
auto-download the backgrounds on first use.

### Step 6: Verify integration

```bash
# Environment builds
chorus setup --oracle myoracle

# Health check
chorus health --oracle myoracle

# Backgrounds present
chorus backgrounds status --oracle myoracle

# Predictions work
mamba run -n chorus-myoracle python -c "
from chorus.oracles import get_oracle
Oracle = get_oracle('myoracle')
o = Oracle(use_environment=False)
o.load_pretrained_model()
result = o.predict(('chr1', 1000000, 1001000), ['DNASE:K562'])
print(result)
"
```

### Checklist for a complete oracle

| Item | File | Required? |
|------|------|-----------|
| Oracle class | `chorus/oracles/my_oracle.py` | Yes |
| Register in `__init__.py` | `chorus/oracles/__init__.py` | Yes |
| Conda environment | `environments/chorus-myoracle.yml` | Yes |
| CDF build script | `scripts/build_backgrounds_myoracle.py` | Yes (for normalization) |
| Layer config entry | `chorus/analysis/scorers.py` | If new layer type |
| Model weights hosting | ENCODE / Zenodo / HuggingFace | Yes (for distribution) |
| HuggingFace backgrounds | `lucapinello/chorus-backgrounds` | Recommended |
| Metadata class | `chorus/oracles/my_oracle_metadata.py` | Optional (for track discovery) |

## Checking background status

```bash
# Show all oracles
chorus backgrounds status

# Show one oracle with details
chorus backgrounds status --oracle chrombpnet
```

Example output:

```
  enformer         5313 tracks  522.9 MB  2026-04-24 23:53  CDFs: effect_cdfs, summary_cdfs, perbin_cdfs
  borzoi           7611 tracks  766.1 MB  2026-04-24 23:43  CDFs: effect_cdfs, summary_cdfs, perbin_cdfs
  chrombpnet        786 tracks   78.6 MB  2026-04-26 15:04  CDFs: effect_cdfs, summary_cdfs, perbin_cdfs
                         ATAC/DNASE: 42  CHIP: 744
  sei                40 tracks    2.8 MB  2026-04-25 00:29  CDFs: effect_cdfs, summary_cdfs
  legnet              3 tracks    0.2 MB  2026-04-24 23:54  CDFs: effect_cdfs, summary_cdfs
  alphagenome       5168 tracks  262.8 MB  2026-04-24 23:32  CDFs: effect_cdfs, summary_cdfs, perbin_cdfs
```

## Code reference

| Component | Path |
|-----------|------|
| Normalization core | `chorus/analysis/normalization.py` |
| Layer configs | `chorus/analysis/scorers.py` |
| Oracle registry | `chorus/oracles/__init__.py` |
| CLI backgrounds | `chorus/cli/_backgrounds.py` |
| Build: Enformer | `scripts/build_backgrounds_enformer.py` |
| Build: Borzoi | `scripts/build_backgrounds_borzoi.py` |
| Build: ChromBPNet | `scripts/build_backgrounds_chrombpnet.py` |
| Build: Sei | `scripts/build_backgrounds_sei.py` |
| Build: LegNet | `scripts/build_backgrounds_legnet.py` |
| Build: AlphaGenome | `scripts/build_backgrounds_alphagenome.py` |
| Shard orchestrator | `scripts/run_bpnet_cdf_build.sh` |
| HuggingFace repo | `lucapinello/chorus-backgrounds` |
