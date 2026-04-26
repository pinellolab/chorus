# Normalization Guide

Chorus normalizes raw oracle predictions into percentile scores using
empirical CDF (cumulative distribution function) backgrounds. This guide
documents how normalization works, what was built for each oracle, and how
to add new models and extend the backgrounds.

## How normalization works

Every oracle produces raw prediction values (e.g. predicted DNASE signal
in counts, or a log2 fold-change variant effect). These raw values are
hard to interpret across tracks and oracles because each has different
scale and dynamic range. Normalization maps raw values to percentiles
[0, 1] (or [-1, 1] for signed layers) by ranking them against a
pre-computed background distribution.

### Three CDF types

For each track in each oracle, we store three empirical CDFs:

| CDF | What it normalizes | How it's built | Output range |
|-----|--------------------|----------------|--------------|
| **effect_cdfs** | Variant effect scores | Score ~10K common SNPs (gnomAD) | [0, 1] or [-1, 1] |
| **summary_cdfs** | Baseline signal levels | Predict ~30K genomic positions (random + cCREs + TSS) | [0, 1] |
| **perbin_cdfs** | Per-bin signal values | Same positions, per-bin (128bp or 1bp) values | [0, 1] |

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

For **unsigned** layers (chromatin, TF binding, histone marks, CAGE,
splicing): raw effect scores are converted to absolute values, and the
percentile maps to [0, 1] reflecting effect magnitude.

For **signed** layers (gene expression, MPRA, Sei classification): the
sign is preserved, and the percentile maps to [-1, 1] reflecting both
direction and magnitude.

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

**Sharded build**: For large-scale rebuilds across multiple GPUs, use:

```bash
bash scripts/run_bpnet_cdf_build.sh  # 6-GPU parallel build
```

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
| Input length | 230 bp |
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

## Walkthrough: Bring your own ChromBPNet model

This is the most common case — you trained a ChromBPNet or BPNet model
outside Chorus and want to use it with full percentile normalization.

### Step 1: Organize your model weights

Chorus expects a directory containing the ChromBPNet model files (the same
structure that `chrombpnet` produces after training):

```
my_model/
  models/
    chrombpnet_nobias.h5     # or .keras
```

For BPNet (CHIP/TF) models from JASPAR, the structure is:

```
my_bpnet_model/
  model.h5
```

### Step 2: Verify the model loads

```python
from chorus.oracles.chrombpnet import ChromBPNetOracle

oracle = ChromBPNetOracle(use_environment=False)

# Load your custom model — is_custom=True bypasses the built-in registry
oracle.load_pretrained_model(
    assay="ATAC",              # or "DNASE" or "CHIP"
    cell_type="my_cell_line",  # any string you choose — becomes the track ID
    weights="/path/to/my_model",
    is_custom=True,
)

# Quick smoke test
result = oracle.predict(("chr1", 1000000, 1002114), assay_ids=["ATAC:my_cell_line"])
print(result)
```

If using CHIP, also pass `TF="MyTF"`:

```python
oracle.load_pretrained_model(
    assay="CHIP",
    cell_type="K562",
    TF="MyTF",
    weights="/path/to/my_bpnet_model",
    is_custom=True,
)
```

### Step 3: Build the CDF background for your model

The CDF background requires scoring ~10K common variants and ~30K baseline
positions through your model. The recommended path is to register your
custom track in the script's model list and run the existing build pipeline,
which handles batched inference, reservoir sampling, and CDF compaction
for you.

**Step 3a — register your track.** Edit
`chorus/oracles/chrombpnet_source/chrombpnet_globals.py` and add an
entry pointing at your custom weights, e.g.:

```python
# Use a stable identifier for `cell_type` so the resulting track_id
# is "ATAC:my_cell_line" — this becomes the row key in the NPZ.
CHROMBPNET_MODELS_DICT["ATAC"]["my_cell_line"] = "MY_CUSTOM_001"  # arbitrary id
```

For an externally-provided weights path, you'll also want to extend
`ChromBPNetOracle._download_chrombpnet_model` (or
`_download_model_from_JASPAR` for CHIP) to recognise the custom id and
copy / symlink your weights into
`downloads/chrombpnet/ATAC_my_cell_line/`.

**Step 3b — run the build.** With the dict updated, the existing
`--only-missing` pass will pick up just your new track:

```bash
mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
    --part both --only-missing --gpu 0
```

This writes interim files at `~/.chorus/backgrounds/chrombpnet_*_interim.npz`
containing only the new row(s), then auto-merges into the main NPZ via
`merge_to_final_incremental` (which calls `PerTrackNormalizer.append_tracks`
with dedup).

### Step 4: Append your CDF to the main NPZ

If you ran the build script with `--only-missing --part both`, the
merge happens automatically and your row is already in the main NPZ.

**Alternative — bring your own NPZ.** If you have CDF rows you computed
outside the build script (e.g. from a different scoring pipeline),
package them into a small NPZ and append:

```bash
chorus backgrounds add-tracks --oracle chrombpnet --npz my_custom_cdfs.npz
```

The source NPZ must follow the same schema as
`~/.chorus/backgrounds/chrombpnet_pertrack.npz`: keys `track_ids`,
`effect_cdfs`, `summary_cdfs`, `perbin_cdfs` (optional), `signed_flags`,
and the matching `_counts` arrays. See `PerTrackNormalizer.build_and_save`
for the exact contract.

Or call `PerTrackNormalizer.append_tracks(...)` directly from Python —
it handles dedup against existing track_ids and concatenates rows
preserving every counts array.

### Step 5: Verify

```bash
chorus backgrounds status --oracle chrombpnet
# Should show your new track in the count
```

Now your custom model's predictions will be normalized to percentiles
automatically whenever Chorus scores variants through it.

## Walkthrough: Bring your own LegNet model

If you trained a LegNet MPRA model for a new cell type:

### Step 1: Organize weights

LegNet expects a directory with:

```
my_legnet_model/
  example/
    weights.ckpt
  config.json
```

### Step 2: Load and test

```python
from chorus.oracles.legnet import LegNetOracle

oracle = LegNetOracle(use_environment=False)
oracle.cell_type = "my_cell"
oracle.assay = "MPRA"
oracle.load_pretrained_model(weights="/path/to/my_legnet_model")

# Test prediction
result = oracle.predict("ACGT" * 57 + "AC")  # 230 bp input
print(result)
```

### Step 3: Build CDF and append

Use the same approach as ChromBPNet — either the build script or manual
scoring + `PerTrackNormalizer.append_tracks()`.

For LegNet, the key difference is that promoter_activity is a **signed**
layer ([-1, 1]), so:

```python
PerTrackNormalizer.append_tracks(
    oracle_name="legnet",
    new_track_ids=["MPRA:my_cell"],
    new_effect_cdfs=effect_cdf.reshape(1, -1),
    new_summary_cdfs=summary_cdf.reshape(1, -1),
    new_signed_flags=np.array([True]),   # signed for MPRA
    new_effect_counts=np.array([n_variants]),
    new_summary_counts=np.array([n_baselines]),
)
```

### Step 4: Register the cell type (optional)

To make it discoverable via `oracle.list_cell_types()`, add your cell
type to `chorus/oracles/legnet_globals.py`:

```python
LEGNET_AVAILABLE_CELLTYPES = ["K562", "HepG2", "WTC11", "my_cell"]
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

Create `scripts/build_backgrounds_myoracle.py`. Follow the pattern of
any existing build script. The key structure:

```python
import argparse
import numpy as np
from chorus.analysis.normalization import PerTrackNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "both", "merge"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--only-missing", action="store_true")
args = parser.parse_args()

ORACLE_NAME = "myoracle"

# 1. Define your track IDs
track_ids = ["DNASE:K562", "DNASE:HepG2"]

# 2. For each track, score variants and baselines
#    (use ReservoirSampler from existing scripts for memory efficiency)

# 3. Save
PerTrackNormalizer.build_and_save(
    oracle_name=ORACLE_NAME,
    track_ids=track_ids,
    effect_cdfs=effect_matrix,     # (n_tracks, 10000)
    summary_cdfs=summary_matrix,
    perbin_cdfs=perbin_matrix,     # optional
    signed_flags=signed_flags,
    effect_counts=effect_counts,
    summary_counts=summary_counts,
    perbin_counts=perbin_counts,
)
```

Then run:

```bash
mamba run -n chorus-myoracle python scripts/build_backgrounds_myoracle.py --part both --gpu 0
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
