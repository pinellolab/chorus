<div align="center">

## **Chorus**
<img src="logo.png" alt="Chorus" width="250">

*Predict how a genetic variant changes gene regulation — chromatin accessibility, transcription factor binding, histone marks, gene expression — across thousands of cell types. One API over six state-of-the-art deep-learning models; ask in natural language via Claude or call it from Python.*

</div>

## Overview

Chorus provides a consistent, easy-to-use API for working with state-of-the-art genomic deep learning models including:

- **Enformer**: Predicts gene expression and chromatin states from DNA sequences
- **Borzoi**: Enhanced model for regulatory genomics predictions
- **ChromBPNet / BPNet**: Predicts chromatin accessibility (ChromBPNet) and TF binding (BPNet) at base-pair resolution
- **Sei**: Sequence regulatory effect predictions across 21,907 chromatin profiles
- **LegNet**: Regulatory regions activity prediction using models trained on MPRA data
- **AlphaGenome**: Google DeepMind's model predicting 5,731 genomic tracks at single base-pair resolution from 1MB input

Key features:
- 🧬 Unified API across different models
- 📊 Built-in visualization tools for genomic tracks
- 🔬 Variant effect prediction
- 🎯 In silico mutagenesis and sequence optimization
- 📈 Effect-percentile scoring against pre-computed genome-wide backgrounds (auto-downloaded from HuggingFace) — not RNA-seq-style quantile normalization; each variant's effect is ranked against ~10k random SNPs
- 🚀 Enhanced sequence editing logic
- 🔧 Isolated conda environments for each oracle to avoid dependency conflicts
- 🧪 Sub-region scoring, gene expression analysis (CAGE + RNA-seq), and variant-to-gene effect prediction
- 🤖 MCP server for AI assistant integration (Claude, etc.)

## Key terms

| Term | Meaning |
|------|---------|
| **Oracle** | A deep learning model that predicts regulatory activity from DNA sequence (e.g. Enformer, AlphaGenome) |
| **Track** | A single experimental measurement predicted by an oracle (e.g. DNase-seq in K562 cells) |
| **assay_id** | The unique identifier for a track, used in API calls (e.g. `"ENCFF413AHU"` or `"DNASE/EFO:0001187 DNase-seq/."`) |
| **Effect percentile** | How extreme a variant's effect is compared to ~10,000 random SNPs (≥99th = stronger than 99% of random variants) |
| **log2FC** | Log2 fold-change between alternate and reference allele predictions — the raw effect size (most layers). Gene-expression uses **lnFC** (natural log) and MPRA uses **Δ (alt−ref)**; every report states the formula used per layer. |

## 👉 Quick navigation — pick your path

| You are... | Start here |
|---|---|
| **A biologist who wants to ask questions in plain English** | [`docs/MCP_WALKTHROUGH.md`](docs/MCP_WALKTHROUGH.md) — one-line `chorus mcp install`, then type questions to Claude Code |
| **Someone who wants to see what outputs look like first** | [`examples/walkthroughs/`](examples/walkthroughs/) — pre-run concrete examples with HTML, JSON, TSV + embedded IGV browser |
| **A Python developer building a pipeline** | [`examples/notebooks/`](examples/notebooks/) — three end-to-end Jupyter walkthroughs (15 / 45 / 60 min) |
| **Just want to verify the install** | Skip to [Installation](#installation) → [Minimal Working Example](#minimal-working-example) |

## 👉 Worked application examples

Every subfolder under [`examples/walkthroughs/`](examples/walkthroughs/) is a
concrete, ready-to-reproduce use case with full outputs in **Markdown, JSON,
TSV, and HTML** (with an embedded IGV browser):

| I want to... | Example |
|---|---|
| Analyze a GWAS / clinical variant in a specific cell type | [variant_analysis/SORT1_rs12740374](examples/walkthroughs/variant_analysis/SORT1_rs12740374/) |
| I have a variant but don't know the relevant tissue | [discovery/SORT1_cell_type_screen](examples/walkthroughs/discovery/SORT1_cell_type_screen/) |
| Fine-map a GWAS locus to the causal SNP | [causal_prioritization/SORT1_locus](examples/walkthroughs/causal_prioritization/SORT1_locus/) |
| Score a batch of variants from a VCF | [batch_scoring/](examples/walkthroughs/batch_scoring/) |
| Predict the effect of an engineered sequence edit | [sequence_engineering/region_swap](examples/walkthroughs/sequence_engineering/region_swap/) |
| Replicate a published regulatory variant finding | [validation/SORT1_rs12740374_with_CEBP](examples/walkthroughs/validation/SORT1_rs12740374_with_CEBP/) |
| Cross-validate a variant across multiple oracles | [validation/SORT1_rs12740374_multioracle](examples/walkthroughs/validation/SORT1_rs12740374_multioracle/) |

These examples were generated through Claude Code using Chorus's MCP server —
the same way you'll use it. Every report preserves the original prompt at the
top, so you can see exactly what was asked and reproduce it.
See [`examples/walkthroughs/README.md`](examples/walkthroughs/README.md) for
the full list with per-persona ("Geneticist", "Bioinformatician", "Clinician",
"Computational biologist") starting points.

## Prerequisites

- **Miniforge** (provides `mamba`): Install from https://github.com/conda-forge/miniforge
- **Git**
- ~20 GB free disk space (for models, genomes, and conda environments)
- Works on **Linux x86_64** and **macOS (Intel/Apple Silicon)**
- GPU support: NVIDIA CUDA (Linux) is auto-detected. Apple Metal is experimental and not fully supported by all oracles (see AlphaGenome section).

### Hardware matrix per oracle

Pick at most a couple of oracles to start with — you can always add more
with `chorus setup --oracle <name>` later. AlphaGenome is the most
capable but also the heaviest; Enformer is the best CPU-friendly
starter.

| Oracle | Minimum RAM | GPU needed? | Typical cold-predict | Best for |
|---|---|---|---|---|
| **Enformer** | 8 GB | optional | ~10 s (GPU) / ~1 min (CPU) | lightweight multi-track, CPU-friendly starter |
| **Borzoi** | 12 GB | recommended | ~30 s (GPU) | distal gene-expression effects, longer context |
| **ChromBPNet** | 4 GB | optional | ~1 s (CPU ok) | base-pair chromatin / motif disruption |
| **LegNet** | 4 GB | optional | <1 s | MPRA / promoter activity |
| **Sei** | 4 GB | optional | ~2 s | regulatory sequence-class profiling |
| **AlphaGenome** | 16 GB | strongly recommended | ~30 s (GPU) / 2–5 min (CPU) | comprehensive multi-layer (5,731 tracks, 1 Mb window) |

All oracles auto-detect CUDA via `torch.cuda.is_available()` /
`jax.device_get`; respect `CUDA_VISIBLE_DEVICES` to pin to a specific
GPU. Pass `device='cuda'` / `'cpu'` / `'mps'` explicitly if needed.

## Installation

### Fresh Install

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

# 1. Create the base chorus environment (uses the root environment.yml;
#    the per-oracle YAMLs in environments/ are installed for you by
#    `chorus setup` in the next step)
mamba env create -f environment.yml
mamba activate chorus

# 2. Install the chorus package + CLI (registers `chorus` and `chorus-mcp` commands)
pip install -e .

# 3. Register the chorus env as a Jupyter kernel so the example notebooks
#    in examples/notebooks/*.ipynb pick up the right Python (the kernel
#    they ship with is plain `python3`, which on a fresh machine doesn't
#    have chorus installed)
python -m ipykernel install --user --name chorus --display-name "Python 3 (chorus)"

# 4. Set up at least one oracle environment (see below)
chorus setup --oracle enformer   # lightweight CPU-friendly starter

# 5. Download the reference genome your analyses will need
chorus genome download hg38

# Verify installation
python -c "import chorus; print(f'chorus {chorus.__version__}')"
```

> **Two env files, one source of truth.** The root `environment.yml` is
> what you install. The per-oracle files in `environments/` are consumed
> internally by `chorus setup --oracle <name>` — you don't install them
> directly.

### Upgrading

After the first install, to upgrade cleanly:

```bash
cd chorus && git pull
# Remove oracle envs first (while the chorus CLI is still available):
chorus remove --oracle enformer
# Repeat for each oracle you had installed...
# Then remove the base env:
mamba env remove -n chorus -y
```

Then re-run the Fresh Install steps above.

### Setting Up Oracle Environments

Chorus uses isolated conda environments for each oracle to avoid dependency conflicts between TensorFlow, PyTorch, and JAX models.

**Which oracle to start with?** For variant analysis, **AlphaGenome** is the most comprehensive (1 Mb input window, 1 bp prediction resolution, 5,731 tracks) but requires ~16 GB RAM and benefits from a GPU. **Enformer** is a good lightweight alternative that runs comfortably on CPU with ~8 GB RAM (see the table in [examples/walkthroughs/README.md](examples/walkthroughs/README.md#which-oracle-should-i-use) for a full side-by-side comparison).

```bash
# Set up all oracle environments
chorus setup --oracle alphagenome   # JAX-based — recommended primary oracle (see AlphaGenome section below for auth)
chorus setup --oracle enformer      # TensorFlow-based
chorus setup --oracle borzoi        # PyTorch-based
chorus setup --oracle chrombpnet    # TensorFlow-based (includes BPNet for TF binding)
chorus setup --oracle sei           # PyTorch-based
chorus setup --oracle legnet        # PyTorch-based

# List available environments
chorus list
```

You can check the correctness of installation using the following command:

```bash
# Check environment health (use --timeout for first run when models download)
chorus health --timeout 300
```

**Note:** The first health check (or first prediction) for each oracle may take several minutes as model weights are downloaded automatically. Subsequent runs will be much faster.

### Managing Reference Genomes

Chorus includes built-in support for downloading and managing reference genomes:

```bash
# List available genomes
chorus genome list

# Download a reference genome (e.g., hg38, hg19, mm10)
chorus genome download hg38

# Get information about a downloaded genome
chorus genome info hg38

# Remove a downloaded genome
chorus genome remove hg38
```

Supported genomes:
- **hg38**: Human genome assembly GRCh38
- **hg19**: Human genome assembly GRCh37
- **mm10**: Mouse genome assembly GRCm38
- **mm9**: Mouse genome assembly NCBI37
- **dm6**: Drosophila melanogaster genome assembly BDGP6
- **ce11**: C. elegans genome assembly WBcel235

Genomes are stored in the `genomes/` directory within your Chorus installation.

### Per-track background distributions (auto-downloaded)

Chorus converts every raw prediction into an **effect percentile** and
**activity percentile** against ~10,000 random SNPs and ~30,000 genome-wide
positions scored on the same oracle. These pre-computed per-track CDFs are
what let a user interpret a `+0.45` log2FC as `0.962 activity %ile`.

**Nothing to configure.** On the first variant analysis for a given
oracle, the relevant backgrounds are automatically fetched from the public
HuggingFace dataset
[`lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
and cached at `~/.chorus/backgrounds/`.

| Oracle | File size | Tracks covered |
|---|---|---|
| AlphaGenome | ~260 MB | 5,168 |
| Enformer | ~520 MB | 5,313 |
| Borzoi | ~770 MB | 7,611 |
| ChromBPNet | ~2.4 MB | per-model |
| Sei | ~2.8 MB | 40 classes |
| LegNet | ~210 KB | 3 cell types |

> **The backgrounds dataset is public — no HuggingFace token required.**
> `HF_TOKEN` is only needed for the gated AlphaGenome model itself (see
> the AlphaGenome section below). Causal prioritization with auto-LD-fetch
> needs a separate free LDlink token (see Troubleshooting).

To pre-download all backgrounds (optional, avoids the first-use wait):

```python
from chorus.analysis.normalization import download_pertrack_backgrounds
for oracle in ["alphagenome", "enformer", "borzoi", "chrombpnet", "sei", "legnet"]:
    download_pertrack_backgrounds(oracle)
```

## Quick Start

> **Prefer a notebook?** Open [`examples/notebooks/single_oracle_quickstart.ipynb`](examples/notebooks/single_oracle_quickstart.ipynb)
> for a full walkthrough using Enformer + the GATA1 locus. The code below is the
> minimum viable snippet.
>
> **Prerequisite for the snippet:** you've run `chorus setup --oracle enformer`
> and `chorus genome download hg38` (both in the Installation section).

### Minimal Working Example

```python
import chorus
from chorus.utils import get_genome

# 1. Create oracle with reference genome
genome_path = get_genome('hg38')  # auto-downloads if needed
oracle = chorus.create_oracle('enformer', use_environment=True,
                              reference_fasta=str(genome_path))
oracle.load_pretrained_model()

# 2. Predict DNase accessibility at the beta-globin locus.
#    'ENCFF413AHU' is the ENCODE track ID for DNase-seq in K562 cells.
#    See "Discovering Tracks" below to find track IDs for other assays /
#    cell types, or use the `list_tracks` MCP tool from Claude Code.
predictions = oracle.predict(('chr11', 5247000, 5248000), ['ENCFF413AHU'])

# 3. Check the result
track = predictions['ENCFF413AHU']
print(f"Mean signal: {track.values.mean():.2f}, Max: {track.values.max():.2f}")
```

### Discovering Tracks

Each oracle has thousands of tracks. Use the metadata to find the right ones:

```python
# List available assay types
print(oracle.list_assay_types())   # ['ATAC', 'CAGE', 'CHIP', 'DNASE']

# Search for tracks by keyword (e.g. cell type)
from chorus.oracles.enformer_source.enformer_metadata import get_metadata
meta = get_metadata()
k562_tracks = meta.search_tracks('K562')  # Returns DataFrame with 'identifier' column
print(k562_tracks[['identifier', 'description']].head())

# Use the 'identifier' column as track IDs for predictions
tracks = ['ENCFF413AHU', 'CNhs11250']  # DNase:K562, CAGE:K562
```

> **Tip:** Each oracle has different track naming. Enformer and Borzoi use ENCODE identifiers (e.g. `ENCFF413AHU`). ChromBPNet uses assay + cell type. AlphaGenome uses `{OutputType}/{TrackName}/{Strand}`. See the [Model-Specific Details](#model-specific-details) section for each oracle's track format.

### 1. Wild-type Prediction

```python
# Predict from genomic coordinates
predictions = oracle.predict(
    ('chr11', 5247000, 5248000),  # Beta-globin locus
    tracks
)

# Or from DNA sequence
sequence = 'ACGT' * 98304  # 393,216 bp for Enformer
predictions = oracle.predict(sequence, tracks)
```

### 2. Region Replacement

```python
# Replace a 200bp region with enhancer sequence
enhancer = 'GATA' * 50  # 200bp GATA motif repeats
replaced = oracle.predict_region_replacement(
    'chr11:5247400-5247600',  # Region to replace
    enhancer,                  # New sequence
    tracks
)
```

### 3. Sequence Insertion

```python
# Insert enhancer at specific position
inserted = oracle.predict_region_insertion_at(
    'chr11:5247500',  # Insertion point
    enhancer,         # Sequence to insert
    tracks
)
```

### 4. Variant Effect

```python
# Test SNP effects (e.g., A→G mutation)
variant_effects = oracle.predict_variant_effect(
    'chr11:5247000-5248000',  # Region containing variant
    'chr11:5247500',          # Variant position
    ['A', 'G', 'C', 'T'],     # Reference first, then alternates
    tracks
)
```

### 5. Sub-Region Scoring

```python
# Score a specific peak or promoter within the prediction window
# (instead of summarizing the entire 114 kb output)
score = predictions.score_region('chr11', 5247400, 5247600, 'mean')
# Returns {track_id: score} for all tracks

# Also available on individual tracks with different strategies
track = predictions['ENCFF413AHU']
track.score_region('chr11', 5247400, 5247600, 'max')   # peak signal
track.score_region('chr11', 5247400, 5247600, 'sum')   # total signal
```

### 6. Focused Variant Effect Scoring

```python
from chorus.core.result import score_variant_effect

# Score variant effects at the variant site (±N bins)
scores = score_variant_effect(variant_effects, at_variant=True, window_bins=2)
# Returns {allele: {track_id: {ref_score, alt_score, effect}}}

# Or score variant effects at a specific region (e.g. a nearby promoter)
scores = score_variant_effect(
    variant_effects,
    chrom='chr11', start=5247400, end=5247600,
    scoring_strategy='mean'  # mean, max, sum, median, or abs_max
)
```

### 7. Gene Expression Analysis

```python
# Quantify predicted gene expression from CAGE and/or RNA-seq tracks
# Auto-detects expression tracks and uses appropriate quantification:
#   CAGE/LentiMPRA → TSS windowed max
#   RNA-seq → exon sum (Borzoi/AlphaGenome)
expr = oracle.analyze_gene_expression(predictions, 'GATA1')
# Returns per-track expression values with quantification method

# Also available: get exon annotations for a gene
from chorus.utils.annotations import get_gene_exons
exons = get_gene_exons('GATA1')  # merged exon coordinates
```

### 8. Variant Effect on Gene Expression

```python
# The key question: does this variant change expression of a gene?
result = oracle.analyze_variant_effect_on_gene(variant_effects, 'GATA1')
# Returns fold change, log2 fold change, and absolute change per allele per track
```

### 9. Save Predictions

```python
# Save as BedGraph for genome browser
wt_files = predictions.save_predictions_as_bedgraph(output_dir="bedgraph_outputs",
                                                    prefix='a_wt')

```

## Example Notebooks

```bash
# Download reference genome first
chorus genome download hg38
```

Three notebooks are provided, from introductory to advanced:

| Notebook | Oracles | What it covers |
|----------|---------|----------------|
| `examples/notebooks/single_oracle_quickstart.ipynb` | Enformer | Deep single-oracle tutorial: predictions, region replacement, insertion, variant effects, gene expression, coolbox visualization |
| `examples/notebooks/comprehensive_oracle_showcase.ipynb` | All 6 | All oracles side by side, cross-oracle comparison, variant analysis with gene expression, sub-region scoring |
| `examples/notebooks/advanced_multi_oracle_analysis.ipynb` | Enformer + ChromBPNet/BPNet + LegNet | CHIP-seq TF binding, strand-specific tracks, Interval API, effect-percentile normalization, cell-type switching |

## Key Features

### 1. Environment Isolation

Each oracle runs in its own conda environment to avoid dependency conflicts:

```python
# TensorFlow-based Enformer runs in isolated environment
enformer = chorus.create_oracle('enformer', use_environment=True)

# PyTorch-based Borzoi runs in its own isolated environment
borzoi = chorus.create_oracle('borzoi', use_environment=True)
```

### 2. Reference Genome Integration

For accurate predictions, provide a reference genome to extract proper flanking sequences:

```python
# Enformer requires 393,216 bp of context
# Chorus automatically extracts and pads sequences from the reference

# Option 1: Using get_genome() - simplest approach
from chorus.utils import get_genome
genome_path = get_genome('hg38')  # Auto-downloads if not present
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))

# Option 2: Using GenomeManager directly
from chorus.utils import GenomeManager
gm = GenomeManager()
genome_path = gm.get_genome('hg38')  # Auto-downloads if needed
oracle = chorus.create_oracle('enformer', 
                             use_environment=True,
                             reference_fasta=str(genome_path))

# Predict using genomic coordinates
predictions = oracle.predict(('chr1', 1000000, 1001000), ['DNase:K562'])
```

### 3. Track Support

Track identifiers vary by oracle. Use the metadata search (see [Discovering Tracks](#discovering-tracks)) to find the right IDs.

For Enformer and Borzoi, you can use ENCODE identifiers or descriptive names in the Python API:
```python
# ENCODE identifier (recommended — works in both Python API and MCP)
predictions = oracle.predict(sequence, ['ENCFF413AHU'])  # DNase:K562

# Descriptive name (Python API only)
predictions = oracle.predict(sequence, ['DNase:K562'])

# CAGE identifier
predictions = oracle.predict(sequence, ['CNhs11250'])  # CAGE:K562
```

> **MCP users:** The MCP server requires ENCODE identifiers (e.g. `ENCFF413AHU`), not descriptive names. Use `list_tracks(oracle_name, query='K562')` to search and get the `identifier` field.

### 4. BedGraph Output

Predictions can be saved as BedGraph tracks for genome browser visualization:

```python
# Predictions are returned as numpy arrays
# Each bin represents 128 bp for Enformer
# See examples for BedGraph generation code
```

## Core Concepts

### Oracles
Oracles are deep learning models that predict genomic regulatory activity. Each oracle implements a common interface while running in isolated environments.

### Intervals

A unified interface to genomic coordinates and reference sequences. Intervals track sequence edits alongside their corresponding model predictions, supporting reproducible in silico perturbation workflows and consistent downstream analysis.

### Tracks
Tracks represent genomic signal data (e.g., DNase-seq, ChIP-seq). Enformer predicts 5,313 human tracks covering various assays and cell types.

### Environment Management
The `chorus` CLI manages conda environments for each oracle:

```bash
# Set up environments
chorus setup --oracle enformer

# Check health
chorus health

# Clean up
chorus remove --oracle enformer
```

## Model-Specific Details

### Enformer 

Enformer (Avsec et al., 2021) is a hybrid convolutional-transformer architecture
designed for long-range sequence-to-function modeling of regulatory
genomics, with the primary goal of predicting transcriptional and
epigenomic activity directly from DNA sequence.

- Sequence length: 393,216 bp input, 114,688 bp output window
- Output: 896 bins × 5,313 tracks
- Bin size: 128 bp
- Track types: Gene expression (CAGE), chromatin accessibility (DNase/ATAC), histone modifications (ChIP-seq)
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF413AHU for DNase:K562)
  - CAGE IDs (e.g., CNhs11250 for CAGE:K562)
  - Descriptive names (e.g., 'DNase:K562', 'H3K4me3:HepG2')
- Track metadata: Included in the package (file with all 5,313 human track definitions)

### Borzoi

Enhanced Enformer with improved performance and RNA-tracks predictions.

- Sequence length: 524,288 bp input, 196,608 bp output window
- Output: 6,144 bins × 7,611 tracks
- Bin size: 32 bp
- Track types: Gene expression (CAGE, RNA-Seq), chromatin accessibility (DNase/ATAC), histone modifications (ChIP-seq)
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF413AHU for DNase:K562)
  - CAGE IDs (e.g., CNhs11250 for CAGE:K562)
  - Descriptive names (e.g., 'DNase:K562', 'H3K4me3:HepG2')
- Track metadata: Included in the package (file with all 7,611 human track definitions)


### ChromBPNet / BPNet

Base-pair resolution for chromatin accessibility and TF binding predictions. This oracle supports two model types through the same interface:

- **ChromBPNet** (`assay="DNASE"` or `assay="ATAC"`): Predicts chromatin accessibility at base-pair resolution. Models are downloaded from ENCODE.
- **BPNet** (`assay="CHIP"`, `TF="GATA1"`): Predicts transcription factor binding at base-pair resolution. Models are downloaded from JASPAR.

Specs:
- Sequence length: 2,114 bp input
- Output: 1,000 bins at 1 bp resolution
- Track types: DNase/ATAC accessibility, TF binding (ChIP-seq)

```python
# ChromBPNet: chromatin accessibility
oracle = chorus.create_oracle('chrombpnet', use_environment=True,
                              reference_fasta=str(genome_path))
oracle.load_pretrained_model(assay="DNASE", cell_type="K562")

# BPNet: TF-specific binding prediction
oracle.load_pretrained_model(assay="CHIP", cell_type="K562", TF="GATA1")
```

#### Loading Custom Models

You can load your own ChromBPNet/BPNet weights (e.g. trained on a new cell type):

```python
oracle.load_pretrained_model(
    assay="DNASE",                   # DNASE, ATAC, or CHIP
    cell_type="HepG2",              # your cell type label
    weights='path/to/weights',      # path to your model weights
    is_custom=True                  # enables custom weight paths
)
```

### Sei

Sequence regulatory effect predictions (uses custom track naming for 21,907 profiles)

- Sequence length: 4096 bp input
- Output: 1 bin 
- Bin size: 4096 bp
- Track types: DNase accessibility, TF binding (CHIP-Seq), histone modifications 
- Track identifiers: 
  - custom Sei track identifiers
- Track metadata: Included in the package (files with all 21907 human track definitions and 40 Sei-defined classes)


### LegNet

LegNet is a fully convolutional neural network designed for efficient modeling of short regulatory DNA sequences.

- Sequence length: 200 bp input
- Output: 1 bin
- Bin size: 200 bp
- Track types: Element activity in MPRA experiment
- Track identifiers:
  - cell line names

### AlphaGenome

AlphaGenome (Google DeepMind, Nature 2026) predicts 5,731 human functional genomic tracks at single base-pair resolution from up to 1 MB of DNA sequence using a JAX-based model.

- Sequence length: 1,048,576 bp (1 MB) input
- Output: 1,048,576 bins at single base-pair resolution
- Bin size: 1 bp (ATAC, CAGE, DNase, RNA-seq, splice sites, PRO-CAP) or 128 bp (ChIP-seq histone/TF)
- Track types: ATAC, CAGE, ChIP-seq (histone + TF), DNase, RNA-seq, Splice sites, PRO-CAP
- Track identifiers: `{OutputType}/{TrackName}/{Strand}` (e.g., `ATAC/CL:0000084 ATAC-seq/.`)
- Weights: Hosted on HuggingFace (gated repository, requires authentication)

#### AlphaGenome Setup

AlphaGenome weights are hosted on a **gated HuggingFace repository**. You must authenticate before first use:

1. **Create a HuggingFace account** at https://huggingface.co/join

2. **Accept the model license terms** at https://huggingface.co/google/alphagenome-all-folds (click "Agree and access repository")

3. **Generate a token** at https://huggingface.co/settings/tokens (read access is sufficient)

4. **Authenticate** via one of these methods:
```bash
# Option A: Set environment variable (recommended — works with automation and across envs)
export HF_TOKEN="hf_your_token_here"

# Option B: Interactive login (saves token to ~/.cache/huggingface/token)
mamba run -n chorus-alphagenome huggingface-cli login
```

> **For MCP users**: Claude Code inherits environment variables from the
> shell where you start `claude`. Make sure `HF_TOKEN` is exported in that
> shell (e.g. add the `export` to your `~/.bashrc` or `~/.zshrc`).
> Option B (cached token) also works without any shell export.

5. **Set up the environment and verify**:
```bash
chorus setup --oracle alphagenome
chorus health --oracle alphagenome --timeout 300
```

#### AlphaGenome Usage

```python
import chorus
from chorus.utils import get_genome

genome_path = get_genome('hg38')

# Create and load oracle
oracle = chorus.create_oracle('alphagenome',
                              use_environment=True,
                              reference_fasta=str(genome_path),
                              device='cpu')  # or omit for auto-detect GPU
oracle.load_pretrained_model()

# Discover available tracks
print(oracle.list_assay_types())   # ['ATAC', 'CAGE', 'CHIP', 'DNASE', ...]
print(oracle.get_track_info('ATAC'))  # DataFrame of ATAC tracks

# Predict
tracks = ['ATAC/CL:0000084 ATAC-seq/.']  # T-cell ATAC-seq
predictions = oracle.predict(('chr1', 1_000_000, 2_048_576), tracks)
```

#### AlphaGenome GPU Support

AlphaGenome uses JAX, which supports multiple accelerator backends:

- **NVIDIA GPU (Linux)**: Automatically installs `jax[cuda12]` when NVIDIA GPU is detected during `chorus setup`
- **Apple Silicon (macOS)**: Uses **CPU** by default. `jax-metal` is installed but the Metal backend is experimental and does not yet support all operations AlphaGenome requires (e.g., `default_memory_space`). You can explicitly try `device='metal'` but expect errors.
- **CPU**: Works everywhere as fallback; pass `device='cpu'` to force CPU

```python
# Auto-detect best available device (CUDA GPU > CPU; Metal skipped for AlphaGenome)
oracle = chorus.create_oracle('alphagenome', use_environment=True)

# Force specific device
oracle = chorus.create_oracle('alphagenome', use_environment=True, device='cpu')
oracle = chorus.create_oracle('alphagenome', use_environment=True, device='gpu')    # NVIDIA CUDA
```

## MCP Server (AI Assistant Integration)

Chorus includes an MCP (Model Context Protocol) server that lets AI assistants like Claude
directly load oracles, predict variant effects, and analyze gene expression — all through
natural language conversation.

### Setup for Claude Code

**You do NOT need to run the server manually.** Claude Code manages the MCP server process
automatically. You just need a `.mcp.json` file — and it works from **any project folder**,
not just the Chorus repo.

**Step 1:** One-liner — run this from any project folder:

```bash
curl -sL https://raw.githubusercontent.com/pinellolab/chorus/main/.mcp.json -o .mcp.json
```

Or create `.mcp.json` manually:

```json
{
  "mcpServers": {
    "chorus": {
      "type": "stdio",
      "command": "mamba",
      "args": ["run", "-n", "chorus", "chorus-mcp"],
      "env": {
        "CHORUS_NO_TIMEOUT": "1"
      }
    }
  }
}
```

The `chorus-mcp` command is installed in the `chorus` conda environment, so
`mamba run -n chorus chorus-mcp` works from any directory.

> **Note:** If you use `conda` instead of `mamba`, replace `"command": "mamba"` with `"command": "conda"`.
> The `CHORUS_NO_TIMEOUT` env var disables prediction timeouts, which is recommended for interactive use.

**Step 2:** Start (or restart) Claude Code from your project:

```bash
cd /path/to/my-project    # any folder — does NOT need to be the chorus repo
claude
```

Claude Code reads `.mcp.json` on startup and launches the MCP server in the background.
You should see the Chorus tools available immediately — try asking: *"What oracles are available?"*

**Alternatively**, you can add Chorus to your global Claude Code settings (`~/.claude/settings.json`)
so it's available in every project without needing a per-project `.mcp.json`:

```bash
# Add globally (one-time setup):
claude mcp add chorus -- mamba run -n chorus chorus-mcp
```

### Setup for Claude Desktop

Add this to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "chorus": {
      "command": "mamba",
      "args": ["run", "-n", "chorus", "chorus-mcp"],
      "env": {
        "CHORUS_NO_TIMEOUT": "1"
      }
    }
  }
}
```

Then restart Claude Desktop. Chorus tools will be available in all conversations.

### Manual testing (optional)

You can verify the server starts correctly by running it directly:

```bash
mamba run -n chorus chorus-mcp
# You should see the FastMCP banner. Press Ctrl+C to stop.
```

### Available MCP tools

- **Discovery**: `list_oracles`, `list_tracks`, `list_genomes`, `get_genes_in_region`, `get_gene_tss`
- **Lifecycle**: `load_oracle`, `unload_oracle`, `oracle_status`
- **Low-level prediction**: `predict`, `predict_variant_effect`, `predict_region_replacement`, `predict_region_insertion`
- **Scoring primitives**: `score_prediction_region`, `score_variant_effect_at_region`, `predict_variant_effect_on_gene`
- **Multi-layer analysis (recommended for most users)**:
    - `analyze_variant_multilayer` — score a variant across chromatin, TF, histone, CAGE, RNA, splicing in one call
    - `discover_variant` — find top tracks/cell types for a variant without pre-selecting assays
    - `discover_variant_cell_types` — screen hundreds of cell types to find where a variant matters most
    - `score_variant_batch` — rank many variants (VCF / GWAS set / credible set) by effect magnitude
    - `fine_map_causal_variant` — prioritize the causal SNP in a GWAS locus using multi-layer convergence
    - `analyze_region_swap`, `simulate_integration` — score sequence engineering edits (promoter swaps, construct insertions)

Every analysis tool accepts an optional `user_prompt` parameter and writes it into the top of the report so an HTML/MD opened later still shows the original question. See
[`examples/walkthroughs/`](examples/walkthroughs/) for worked outputs of each tool, or read the
[MCP Walkthrough](docs/MCP_WALKTHROUGH.md) for a step-by-step guide showing what you type in Claude
and what comes back.

Key features:
- **Auto-centering**: `region` is optional in variant tools — auto-sized for each oracle's output window
- **ChromBPNet/BPNet params**: `load_oracle("chrombpnet", assay="CHIP", cell_type="K562", TF="GATA1")`
- **TSS warnings**: `predict_variant_effect_on_gene` warns when the target gene TSS is outside the output window
- **Mixed-resolution**: AlphaGenome's 1bp DNASE + 128bp histone tracks score correctly in a single call

### Variant Analysis with AlphaGenome (Recommended)

AlphaGenome (1Mb window, 5731 tracks) is the recommended primary oracle for variant analysis.
It covers DNASE, ATAC, CAGE, RNA-seq, ChIP-seq histone marks, and TF binding in a single model.

Example conversation with Claude:

> **You:** *Load AlphaGenome and predict the effect of rs12740374 (chr1:109274968 G>T) on hepatocyte CAGE expression*
>
> Claude will call `load_oracle("alphagenome")`, then `predict_variant_effect(...)` with the right tracks,
> and return a summary of chromatin and expression effects.

See `docs/variant_analysis_framework.md` for the full 5-layer analysis guide
with track selection cheat sheets by disease area.

## Troubleshooting

### Device Selection

By default, Chorus auto-detects and uses GPU if available. You can explicitly control device selection:

```python
oracle = chorus.create_oracle('enformer', use_environment=True,
                              reference_fasta=str(genome_path),
                              device='cpu')       # Force CPU
# Or: device='cuda:1' for a specific GPU
# Or: export CHORUS_DEVICE=cpu
```

### Timeout Issues

For slower systems or CPU-only environments, you may need to adjust timeouts:

```python
oracle = chorus.create_oracle('enformer', use_environment=True,
                              reference_fasta=str(genome_path),
                              model_load_timeout=1800,  # 30 min (default 600)
                              predict_timeout=900)      # 15 min (default 300)

# Or disable all timeouts globally
# export CHORUS_NO_TIMEOUT=1
```

Common timeout scenarios:
- **Model loading**: First-time downloads can be slow (~1GB model)
- **CPU predictions**: GPU is 10-100x faster than CPU
- **Network filesystems**: Add 50% to timeouts for NFS/shared storage

### Environment Issues
```bash
# Check if environment exists
chorus health

# Recreate environment
chorus remove --oracle enformer
chorus setup --oracle enformer
```

#### Two mamba installs ⇒ `chorus health` reports phantom failures

If you have both `~/.local/share/mamba/` (typical when `mamba` is installed via
brew/pip) **and** a `~/miniforge3/` (typical when you originally installed via
the Miniforge installer), `mamba env create` may put the new `chorus` env in
one root while the per-oracle `chorus-*` envs live in the other. `chorus health`
then fails with `mamba list -n chorus-<oracle>` returning non-zero because the
running mamba is looking under the wrong root.

**Fix:** point `MAMBA_ROOT_PREFIX` at the root that contains the oracle envs
before invoking `chorus`:

```bash
export MAMBA_ROOT_PREFIX=$HOME/miniforge3   # or wherever your chorus-* envs live
mamba env list                              # confirm chorus-* envs show up
chorus health
```

Add the export to your shell rc file if you want it persistent.

#### Enformer fails with `saved_model.pb` not found after a partial download

TensorFlow Hub caches downloaded models at
`/var/folders/.../T/tfhub_modules/` (macOS) or `/tmp/tfhub_modules/`
(Linux). **This cache is outside `~/.chorus/` and survives chorus
teardowns.** If an earlier Enformer download was interrupted, the
cached directory ends up missing `saved_model.pb` and Enformer fails
to load with:

```
Trying to load a model of incompatible/unknown type. ... contains
neither 'saved_model.pb' nor 'saved_model.pbtxt'.
```

Clear the stale cache and retry:

```bash
# macOS
rm -rf /var/folders/*/*/T/tfhub_modules

# Linux
rm -rf /tmp/tfhub_modules
```

Chorus auto-detects the corrupted-cache case and clears it on the
next `load_pretrained_model()` call, so this is only an issue if the
first attempt after a fresh install fails — the second attempt will
recover automatically.

### Memory Issues
Some oracles require a significant memory (~8-16 GB) for predictions. Solutions:
- Force CPU usage: `device='cpu'`
- Use a different GPU: `device='cuda:1'`
- Reduce batch size if needed

### AlphaGenome Authentication
AlphaGenome weights are hosted on a gated HuggingFace repository. If you see a `GatedRepoError` or 403 error:

```bash
# 1. Accept model terms at https://huggingface.co/google/alphagenome-all-folds
# 2. Authenticate via environment variable (recommended)
export HF_TOKEN="hf_your_token_here"
# Or: mamba run -n chorus-alphagenome huggingface-cli login
```

### LDlink Token (for causal prioritization)
The `fine_map_causal_variant` tool can auto-fetch LD proxies from LDlink.
This requires a free token — register at <https://ldlink.nih.gov/?tab=apiaccess>
and pass it via the `ldlink_token` parameter or set:

```bash
export LDLINK_TOKEN="your_token_here"
```

Without a token, you can still use fine-mapping by providing LD variants
manually via the `ld_variants` parameter (see
[causal_prioritization/](examples/walkthroughs/causal_prioritization/)).

### CUDA/GPU Support
The isolated environments include GPU support. On Linux with NVIDIA GPUs, Chorus auto-detects CUDA and installs GPU-enabled packages during `chorus setup`.

**On macOS with Apple Silicon**, Chorus auto-detects Apple GPU acceleration where supported:

| Oracle | Backend | macOS GPU path |
|---|---|---|
| Borzoi, Sei, LegNet | PyTorch | **MPS** auto-detected via `torch.backends.mps.is_available()` |
| ChromBPNet, Enformer | TensorFlow | **Metal** via `tensorflow-metal` (added automatically by `chorus setup` on macOS arm64) |
| AlphaGenome | JAX | **CPU** — `jax-metal` is installed but Apple's plugin doesn't yet support all ops AlphaGenome needs (e.g. `default_memory_space`); Chorus falls back to CPU |

You can always force a specific device:
```python
oracle = chorus.create_oracle('borzoi', use_environment=True, device='mps')   # Apple GPU
oracle = chorus.create_oracle('borzoi', use_environment=True, device='cpu')   # force CPU
oracle = chorus.create_oracle('borzoi', use_environment=True, device='cuda:0')  # NVIDIA
```

To force CPU usage when GPU causes issues:
```python
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             device='cpu')
```

## Further reading

After the Quick Start, these documents go deeper:

| Doc | When to read it |
|---|---|
| [`docs/MCP_WALKTHROUGH.md`](docs/MCP_WALKTHROUGH.md) | Step-by-step Claude Code conversations with Chorus |
| [`docs/variant_analysis_framework.md`](docs/variant_analysis_framework.md) | 5-layer scoring strategy, track selection by disease area |
| [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) | Full Python API reference (oracles, analysis, utilities, MCP tools) |
| [`docs/METHOD_REFERENCE.md`](docs/METHOD_REFERENCE.md) | Method-level reference for advanced users |
| [`docs/VISUALIZATION_GUIDE.md`](docs/VISUALIZATION_GUIDE.md) | pyGenomeTracks + IGV visualization patterns |
| [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Notes for extending Chorus with new oracles |
| [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) | Upstream oracles, papers, and licenses Chorus builds on |
| [`examples/walkthroughs/`](examples/walkthroughs/) | Worked examples for every MCP tool (variant analysis, batch, causal, discovery, sequence engineering) |

## Contributing

We welcome contributions! Areas needing work:

1. Add more examples and tutorials
2. Implement batch prediction optimizations
3. Add more visualization utilities
4. Add more oracles 

### Adding New Oracles

We've designed Chorus to make it easy to add new genomic prediction models. Each oracle runs in its own isolated conda environment, avoiding dependency conflicts between different frameworks (TensorFlow, PyTorch, JAX, etc.).

**For detailed instructions on implementing a new oracle, see our [Contributing Guide](CONTRIBUTING.md).**

Key steps:
1. Inherit from `OracleBase` and implement required methods
2. Define your conda environment configuration
3. Use the environment isolation system for model loading and predictions
4. Add tests and example notebooks
5. Submit a PR with your implementation

The contributing guide includes complete code examples and templates to get you started.

## Citation

If you use Chorus in your research, please cite:

```bibtex
@software{chorus2026,
  title = {Chorus: A unified interface for genomic sequence oracles},
  author = {Dmitry Penzar , Lorenzo Ruggeri , Rosalba Giugno, Luca Pinello},
  year = {2026},
  url = {https://github.com/pinellolab/chorus}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Chorus integrates several groundbreaking models:
- Enformer (Avsec et al., 2021)
- Borzoi (Linder et al., 2023)
- ChromBPNet / BPNet (Agarwal et al., 2021)
- Sei (Chen et al., 2022)
- LegNet (Penzar et al., 2023)
- AlphaGenome (Google DeepMind, 2026)

For visualization tasks we extensively use [coolbox package](https://github.com/GangCaoLab/CoolBox)

---

## Appendix: Per-track background distributions

This appendix describes how the per-track CDFs that power Chorus's
effect/activity percentiles were computed, what the numbers mean, and how
to use them from both the Python API and an MCP/Claude session.

### What they are and why they exist

A raw `log2FC = +0.45` in a DNase-seq track is hard to interpret. Is it
strong? Is the underlying region even active? The per-track backgrounds
turn that raw number into two complementary genome-aware percentiles:

| Percentile | What it measures | Computed from |
|---|---|---|
| **Effect percentile** | How unusual is *this* variant's effect on this track? | The distribution of variant-effect scores from ~10K random SNPs scored on the same track |
| **Activity percentile** | How active is the reference signal at the variant site, genome-wide? | The distribution of window-summed signal at ~31.5K diverse genomic positions |

Both range `[0, 1]` for unsigned layers (chromatin, ChIP, CAGE,
splicing). For signed layers (gene expression, MPRA, Sei), the effect
percentile ranges `[-1, 1]` (preserving the direction of effect).

The backgrounds are stored as 10,000-point CDFs per track in NPZ files
(one file per oracle, `~/.chorus/backgrounds/{oracle}_pertrack.npz`).
Each oracle's NPZ contains three matrices:

- `effect_cdfs (n_tracks × 10000)` — for the effect percentile
- `summary_cdfs (n_tracks × 10000)` — for the activity percentile
- `perbin_cdfs (n_tracks × 10000)` — for IGV per-bin rescaling
  (omitted for scalar-output oracles like Sei and LegNet)

### How they were calculated

The build scripts live in [`scripts/`](scripts/) — one per oracle. Each
performs three reservoir-sampled passes:

#### 1. Variant effect distribution

10,000 random SNPs sampled uniformly across `chr1`–`chr22`, well away
from chromosome edges. For each SNP:

1. Predict reference and alternate alleles across the full output window.
2. For each track, score the variant effect using the layer-specific
   formula:
   - **log2FC** for unsigned signal layers (chromatin, TF binding, histone
     marks, TSS) — variant effect = `log2((sum_alt + ε) / (sum_ref + ε))`
     in a layer-appropriate window (501 bp for DNase/ChIP-TF/CAGE,
     2001 bp for histone marks, full transcript for RNA).
   - **lnFC** for gene expression — `ln((mean_alt + ε) / (mean_ref + ε))`
     (natural log) averaged over GENCODE protein-coding exons of the
     target gene.
   - **diff** for promoter MPRA — simple `alt - ref` activity difference.
3. Add `|effect|` (unsigned) or raw `effect` (signed) to that track's
   reservoir.

Result: per-track histograms over real human-genome variant effects.

#### 2. Activity (window-sum) distribution

~31,500 positions per track sampled to approximate the genome-wide
distribution of regulatory activity:

| Position type | Count | Purpose |
|---|---|---|
| Random intergenic | 15,000 | Genome-wide null (most genome is silent) |
| ENCODE SCREEN cCREs (per category) | ~11,500 | PLS, dELS, pELS, CA-CTCF, CA-TF, TF, CA-H3K4me3, CA |
| Protein-coding TSSs | 3,000 | Sharp signals: CAGE, H3K4me3, promoter activity |
| Gene-body midpoints (>10 kb genes) | 2,000 | RNA-seq, H3K36me3, broad gene-body marks |

For each position, the layer-appropriate window-sum is added to the
track's reservoir.

**RNA-seq exon-precise sampling** (Borzoi, AlphaGenome): RNA tracks only
collect bins overlapping merged GENCODE v48 protein-coding exons —
intronic bins would distort the activity baseline.

**CAGE summary routing**: CAGE tracks skip cCRE positions for the summary
CDF since CAGE biology lives at TSSs, not enhancers.

#### 3. Per-bin distribution (for IGV visualization)

At each of the same ~31,500 positions, **32 random bins** from the full
output window are added to the perbin reservoir. This captures the
per-bin (not per-window) distribution at the track's native resolution
(1 bp for ATAC/CAGE/RNA/PRO-CAP/splice; 128 bp for ChIP-Histone/TF in
AlphaGenome).

The per-bin CDFs are used by `perbin_floor_rescale_batch` to rescale
raw IGV bin values onto a uniform `[0, 1.5]` display scale where
`1.0` always corresponds to the top-1% genome-wide bin value for that
track. This makes overlaid tracks visually comparable across cell types.

### Sample sizes per oracle

| Oracle | Tracks | Effect samples / track | Activity samples / track | NPZ size |
|---|---|---|---|---|
| AlphaGenome | 5,168 | 10,000 | 31,500 | 260 MB |
| Enformer | 5,313 | 10,000 | 31,500 | 520 MB |
| Borzoi | 7,611 | 10,000 | 31,500 | 770 MB |
| ChromBPNet | per-model | 10,000 | 31,500 | 2.4 MB |
| Sei | 40 classes | 10,000 | 31,500 | 2.8 MB |
| LegNet | 3 cell types | 10,000 | 31,500 | 210 KB |

Effect and activity reservoirs are converted to 10,000-point CDFs (sorted
sample arrays) — so a percentile lookup is a single O(log n) bisect.

### Using backgrounds from the Python API

Auto-load (downloads from HuggingFace if not cached):

```python
from chorus.analysis.normalization import get_pertrack_normalizer

norm = get_pertrack_normalizer("alphagenome")
# → ~/.chorus/backgrounds/alphagenome_pertrack.npz (auto-downloaded if missing)
```

Look up an effect or activity percentile for a single track:

```python
track_id = "DNASE/EFO:0001187 DNase-seq/."  # HepG2 DNase

# How unusual is a +0.45 log2FC effect? (unsigned: pass abs value)
eff_pct = norm.effect_percentile("alphagenome", track_id, abs(0.45),
                                 signed=False)
# → 0.962  (stronger than 96.2% of random SNPs in this track)

# How active is the reference signal at the variant site?
act_pct = norm.activity_percentile("alphagenome", track_id, ref_value=512.0)
# → 0.962  (top 4% of genome-wide regulatory activity for this track)
```

Pass it into the analysis layer to get percentiles attached to every
report:

```python
from chorus.analysis.variant_report import build_variant_report

report = build_variant_report(
    variant_result,
    oracle_name="alphagenome",
    gene_name="SORT1",
    normalizer=norm,    # ← this is what populates the Effect %ile / Activity %ile columns
)
```

Or pre-download all six oracles' backgrounds once:

```python
from chorus.analysis.normalization import download_pertrack_backgrounds
for o in ["alphagenome", "enformer", "borzoi", "chrombpnet", "sei", "legnet"]:
    download_pertrack_backgrounds(o)
```

### Using backgrounds via MCP / Claude

You don't have to do anything. The MCP server auto-attaches the
appropriate normalizer when you call any analysis tool
(`analyze_variant_multilayer`, `score_variant_batch`,
`fine_map_causal_variant`, `analyze_region_swap`, `simulate_integration`,
`discover_variant`, `discover_variant_cell_types`).

The first call for a given oracle triggers a one-time HuggingFace
download (a few hundred MB), cached at `~/.chorus/backgrounds/`.
Subsequent calls reuse the cache.

In the resulting report, every track row gets two extra columns —
`Effect %ile` and `Activity %ile` — and the IGV browser uses the per-bin
CDFs to rescale bin heights for cross-cell-type comparability.

### Documented ranges and how to read the numbers

| Column | Range | Reading |
|---|---|---|
| **Raw effect** (e.g. log2FC) | unbounded; biologically meaningful units | `+1.0` = alt is 2× ref; `-1.0` = alt is 0.5× ref |
| **Effect percentile** (unsigned) | `[0, 1]` | `0.95` = stronger than 95% of ~10K random SNPs in the same track |
| **Effect percentile** (signed) | `[-1, 1]` | `+0.95` = strongly above-baseline gain; `-0.95` = strongly above-baseline loss |
| **Activity percentile** | `[0, 1]` | `0.95` = reference signal at this site is in the top 5% genome-wide for this track |
| **IGV per-bin display value** | `[0, 1.5]` | `1.0` = top-1% bin value genome-wide for this track; `0` = below the noise floor |

**Sanity-check rule of thumb:** a *biologically interesting* variant
typically shows **effect percentile > 0.95** AND **activity percentile >
0.5** in the same track — i.e. an unusually large effect at a site that
already has real regulatory activity.

A high effect percentile with very low activity is usually noise (small
absolute change at a silent site that just happens to be larger than
most random-SNP-at-silent-sites changes).

### Reproducing or extending the backgrounds

Each oracle has a build script at `scripts/build_backgrounds_<oracle>.py`.
Each takes ~4–22 GPU-hours per oracle depending on the track count and
output window. To rebuild a single oracle:

```bash
mamba run -n chorus-alphagenome python scripts/build_backgrounds_alphagenome.py --part variants  --gpu 0
mamba run -n chorus-alphagenome python scripts/build_backgrounds_alphagenome.py --part baselines --gpu 1
mamba run -n chorus              python scripts/build_backgrounds_alphagenome.py --part merge
```

The variants and baselines parts can run in parallel on separate GPUs.
The merge step runs CPU-only and combines the interim NPZ files into the
final `{oracle}_pertrack.npz`. See [`scripts/README.md`](scripts/README.md)
for the full per-oracle commands and runtimes.

If you've rebuilt a background and want to share it: drop the resulting
`{oracle}_pertrack.npz` into `~/.chorus/backgrounds/`. The downloader
will not overwrite local files — it only fetches when the file is
missing.
