# Chorus

A unified interface for genomic sequence oracles - deep learning models that predict genomic regulatory activity from DNA sequences.

## Overview

Chorus provides a consistent, easy-to-use API for working with state-of-the-art genomic deep learning models including:

- **Enformer**: Predicts gene expression and chromatin states from DNA sequences
- **Borzoi**: Enhanced model for regulatory genomics predictions
- **ChromBPNet**: Predicts TF binding and chromatin accessibility at base-pair resolution
- **Sei**: Sequence regulatory effect predictions across 21,907 chromatin profiles
- **LegNet**: Regulatory regions activity prediction using models trained on MPRA data
- **AlphaGenome**: Google DeepMind's model predicting 5,930 genomic tracks at single base-pair resolution from 1MB input

Key features:
- 🧬 Unified API across different models
- 📊 Built-in visualization tools for genomic tracks
- 🔬 Variant effect prediction
- 🎯 In silico mutagenesis and sequence optimization
- 📈 Track normalization and comparison utilities
- 🚀 Enhanced sequence editing logic
- 🔧 Isolated conda environments for each oracle to avoid dependency conflicts
- 🧪 Sub-region scoring, gene expression analysis (CAGE + RNA-seq), and variant-to-gene effect prediction
- 🤖 MCP server for AI assistant integration (Claude, etc.)

## Prerequisites

- **Miniforge** (provides `mamba`): Install from https://github.com/conda-forge/miniforge
- **Git**
- ~20 GB free disk space (for models, genomes, and conda environments)
- Works on **Linux x86_64** and **macOS (Intel/Apple Silicon)**
- GPU support: NVIDIA CUDA (Linux) is auto-detected. Apple Metal is experimental and not fully supported by all oracles (see AlphaGenome section).

## Installation

### Upgrading

The cleanest way to upgrade is to remove existing environments and reinstall:

```bash
cd chorus && git pull
mamba env remove -n chorus -y
# Repeat for each oracle you had installed:
chorus remove --oracle enformer
```

Then follow the installation steps below.

### Fresh Install

```bash
# Clone the repository
git clone https://github.com/pinellolab/chorus.git
cd chorus

# Create the base chorus environment
mamba env create -f environment.yml
mamba activate chorus

# Install chorus package
pip install -e .

# Verify installation
python -c "import chorus; print(f'chorus {chorus.__version__}')"
```

### Setting Up Oracle Environments

Chorus uses isolated conda environments for each oracle to avoid dependency conflicts between TensorFlow, PyTorch, and JAX models.

**Which oracle to start with?** For variant analysis, **AlphaGenome** is the most comprehensive (1 Mb window, 1 bp resolution, 5,930 tracks) but requires ~16 GB RAM and benefits from a GPU. **Enformer** is a good lightweight alternative that runs comfortably on CPU with ~8 GB RAM.

```bash
# Set up all oracle environments
chorus setup --oracle alphagenome   # JAX-based — recommended primary oracle (see AlphaGenome section below for auth)
chorus setup --oracle enformer      # TensorFlow-based
chorus setup --oracle borzoi        # PyTorch-based
chorus setup --oracle chrombpnet    # TensorFlow-based
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

## Quick Start

### Minimal Working Example

```python
import chorus
from chorus.utils import get_genome

# 1. Create oracle with reference genome
genome_path = get_genome('hg38')  # auto-downloads if needed
oracle = chorus.create_oracle('enformer', use_environment=True,
                              reference_fasta=str(genome_path))
oracle.load_pretrained_model()

# 2. Predict DNase accessibility at the beta-globin locus
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
| `examples/single_oracle_quickstart.ipynb` | Enformer | Deep single-oracle tutorial: predictions, region replacement, insertion, variant effects, gene expression, coolbox visualization |
| `examples/comprehensive_oracle_showcase.ipynb` | All 6 | All oracles side by side, cross-oracle comparison, variant analysis with gene expression, sub-region scoring |
| `examples/advanced_multi_oracle_analysis.ipynb` | Enformer + ChromBPNet + LegNet | CHIP-seq TF binding, strand-specific tracks, Interval API, quantile normalization, cell-type switching |

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
- Output: 6,144 bins × 7,610 tracks
- Bin size: 32 bp
- Track types: Gene expression (CAGE, RNA-Seq), chromatin accessibility (DNase/ATAC), histone modifications (ChIP-seq)
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF413AHU for DNase:K562)
  - CAGE IDs (e.g., CNhs11250 for CAGE:K562)
  - Descriptive names (e.g., 'DNase:K562', 'H3K4me3:HepG2')
- Track metadata: Included in the package (file with all 7,610 human track definitions)


### ChromBPNet

Base-pair resolution for chromatin accessibility and TF binding predictions (uses TF-specific tracks)

- Sequence length: 2114 bp input
- Output: 1000 bins 
- Bin size: 1 bp
- Track types: DNase accessibility, TF binding (CHIP-Seq) 
- Track identifiers: 
  - ENCODE IDs (e.g., ENCFF574YLK for DNase:K562)

### Sei

Sequence regulatory effect predictions (uses custom track naming for 21,907 profiles)

- Sequence length: 4096 bp input
- Output: 1 bin 
- Bin size: 4096 bp
- Track types: DNase accessibility, TF binding (CHIP-Seq), histone modifications 
- Track identifiers: 
  - custom Sei track identifiers
- Track metadata: Included in the package (files with all 21907 human track definitions and 41 Sei-defined classes)


### LegNet

LegNet is a fully convolutional neural network designed for efficient modeling of short regulatory DNA sequences.

- Sequence length: 200 bp input
- Output: 1 bin
- Bin size: 200 bp
- Track types: Element activity in MPRA experiment
- Track identifiers:
  - cell line names

### AlphaGenome

AlphaGenome (Google DeepMind, Nature 2026) predicts 5,930 human functional genomic tracks at single base-pair resolution from up to 1 MB of DNA sequence using a JAX-based model.

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
- **Prediction**: `predict`, `predict_variant_effect`, `predict_region_replacement`, `predict_region_insertion`
- **Scoring & Analysis**: `score_prediction_region`, `score_variant_effect_at_region`, `predict_variant_effect_on_gene`

Key features:
- **Auto-centering**: `region` is optional in variant tools — auto-sized for each oracle's output window
- **ChromBPNet params**: `load_oracle("chrombpnet", assay="CHIP", cell_type="K562", TF="GATA1")`
- **TSS warnings**: `predict_variant_effect_on_gene` warns when the target gene TSS is outside the output window
- **Mixed-resolution**: AlphaGenome's 1bp DNASE + 128bp histone tracks score correctly in a single call

### Variant Analysis with AlphaGenome (Recommended)

AlphaGenome (1Mb window, 5930 tracks) is the recommended primary oracle for variant analysis.
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

### CUDA/GPU Support
The isolated environments include GPU support. On Linux with NVIDIA GPUs, Chorus auto-detects CUDA and installs GPU-enabled packages during `chorus setup`. On macOS with Apple Silicon, AlphaGenome defaults to CPU because the JAX Metal backend does not yet support all required operations.

To force CPU usage when GPU causes issues:
```python
oracle = chorus.create_oracle('enformer',
                             use_environment=True,
                             device='cpu')
```

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
- ChromBPNet (Agarwal et al., 2021)
- Sei (Chen et al., 2022)
- LegNet (Penzar et al., 2023)
- AlphaGenome (Google DeepMind, 2026)

For visualization tasks we extensively use [coolbox package](https://github.com/GangCaoLab/CoolBox)
