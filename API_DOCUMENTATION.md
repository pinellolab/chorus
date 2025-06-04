# Chorus API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Prediction Methods](#prediction-methods)
4. [Utility Functions](#utility-functions)
5. [Track Management](#track-management)
6. [Environment Management](#environment-management)
7. [Examples](#examples)

## Overview

Chorus provides a unified interface for genomic sequence prediction models (oracles). Each oracle predicts regulatory activity from DNA sequences, with support for various genomic manipulations and analyses.

## Core Classes

### OracleBase

Base class for all oracle implementations. Provides common functionality and defines the interface that all oracles must implement.

```python
class OracleBase(ABC):
    def __init__(self, use_environment: bool = True)
```

**Attributes:**
- `oracle_name` (str): Name of the oracle (e.g., 'enformer')
- `reference_fasta` (str): Path to reference genome FASTA file
- `loaded` (bool): Whether the model is loaded
- `use_environment` (bool): Whether to use isolated conda environment

### EnformerOracle

Implementation of the Enformer model for predicting gene expression and chromatin states.

```python
class EnformerOracle(OracleBase):
    def __init__(self, use_environment: bool = True, reference_fasta: Optional[str] = None)
```

**Enformer-specific attributes:**
- `sequence_length` (int): 393,216 bp input sequence length
- `target_length` (int): 896 bins in output
- `bin_size` (int): 128 bp per bin
- Output window: 114,688 bp (896 × 128)
- Offset from input edges: 139,264 bp on each side

## Prediction Methods

### 1. predict()

Basic prediction method for DNA sequences or genomic coordinates.

```python
def predict(
    input_data: Union[str, Tuple[str, int, int]],
    assay_ids: List[str],
    create_tracks: bool = False
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `input_data`: Either:
  - DNA sequence string (must be model's required length)
  - Tuple of (chromosome, start, end) for genomic coordinates
- `assay_ids`: List of track identifiers (oracle-specific)
  - Enformer: ENCODE IDs (e.g., 'ENCFF413AHU'), CAGE IDs (e.g., 'CNhs11250'), or descriptions (e.g., 'DNase:K562')
- `create_tracks`: Whether to create track files (not implemented)

**Returns:**
- Dictionary mapping track IDs to prediction arrays
- Each array has shape (n_bins,) where n_bins = output_length / bin_size

**Logic:**
1. If input is coordinates, extracts sequence from reference genome
2. Validates sequence length matches model requirements
3. Runs model prediction
4. Returns predictions for requested tracks

**Example:**
```python
# From sequence
seq = 'ACGT' * 98304  # 393,216 bp for Enformer
predictions = oracle.predict(seq, ['ENCFF413AHU', 'CNhs11250'])

# From coordinates
predictions = oracle.predict(('chrX', 48777634, 48790694), ['ENCFF413AHU'])
```

### 2. predict_region_replacement()

Replace a genomic region with a new sequence and predict the effects.

```python
def predict_region_replacement(
    genomic_region: Union[str, pd.DataFrame],
    seq: str,
    assay_ids: List[str],
    create_tracks: bool = False,
    genome: Optional[str] = None
) -> Dict
```

**Parameters:**
- `genomic_region`: Region to replace
  - String format: "chr1:1000-2000" (1-based, inclusive)
  - DataFrame: First row with columns 'chrom', 'start', 'end'
- `seq`: Replacement DNA sequence (must match region length exactly)
- `assay_ids`: List of track identifiers
- `create_tracks`: Whether to save track files
- `genome`: Reference genome path (uses oracle's reference_fasta if None)

**Returns:**
Dictionary with:
- `raw_predictions`: Dict[track_id, np.ndarray] - Raw model outputs
- `normalized_scores`: Dict[track_id, np.ndarray] - Min-max normalized (0-1)
- `track_objects`: List[Track] - Track objects if create_tracks=True
- `track_files`: List[str] - File paths if create_tracks=True

**Logic:**
1. Validates replacement sequence length matches region length
2. Calculates full context window centered on region
3. Extracts context sequence from reference genome
4. Replaces specified region within context
5. Runs prediction on modified full-length sequence
6. Returns predictions for the output window

**Example:**
```python
# Replace 200bp region with GATA motif repeats
enhancer = 'GATA' * 50  # 200bp
results = oracle.predict_region_replacement(
    'chr11:5247400-5247600',
    enhancer,
    ['ENCFF413AHU']
)
```

### 3. predict_region_insertion_at()

Insert a sequence at a specific genomic position.

```python
def predict_region_insertion_at(
    genomic_position: Union[str, pd.DataFrame],
    seq: str,
    assay_ids: List[str],
    create_tracks: bool = False,
    genome: Optional[str] = None
) -> Dict
```

**Parameters:**
- `genomic_position`: Insertion point
  - String format: "chr1:1000" (1-based)
  - DataFrame: First row with columns 'chrom', 'pos'
- `seq`: DNA sequence to insert (any length that fits in context)
- `assay_ids`: List of track identifiers
- `create_tracks`: Whether to save track files
- `genome`: Reference genome path

**Returns:**
Same format as `predict_region_replacement()`

**Logic:**
1. Calculates required flanking sequence sizes
2. Extracts left flank (before insertion point)
3. Extracts right flank (after insertion point)
4. Constructs: left_flank + inserted_seq + right_flank
5. Ensures total length matches model requirements
6. Runs prediction on modified sequence

**Example:**
```python
# Insert enhancer at specific position
results = oracle.predict_region_insertion_at(
    'chr11:5247500',
    'GATA' * 50,  # Insert 200bp
    ['CNhs11250']
)
```

### 4. predict_variant_effect()

Analyze effects of genetic variants (SNPs, indels).

```python
def predict_variant_effect(
    genomic_region: Union[str, pd.DataFrame],
    variant_position: Union[str, pd.DataFrame],
    alleles: Union[List[str], pd.DataFrame],
    assay_ids: List[str],
    create_tracks: bool = False,
    genome: Optional[str] = None
) -> Dict
```

**Parameters:**
- `genomic_region`: Region containing the variant
  - Should be large enough for model context
- `variant_position`: Position of variant
  - String format: "chr1:1000"
  - Must be within genomic_region
- `alleles`: List of alleles to test
  - First element is reference allele
  - Remaining elements are alternative alleles
  - Can also be DataFrame with 'ref' and 'alt' columns
- `assay_ids`: List of track identifiers
- `create_tracks`: Whether to save track files
- `genome`: Reference genome path

**Returns:**
Dictionary with:
- `predictions`: Dict of allele_name → track predictions
  - 'reference': predictions for reference allele
  - 'alt_1', 'alt_2', etc.: predictions for alternatives
- `effect_sizes`: Dict of alt_allele → track → effect array
  - Effect = alternative - reference
- `track_objects`: Dict if create_tracks=True
- `track_files`: Dict if create_tracks=True
- `variant_info`: Summary of variant tested

**Logic:**
1. Extracts reference sequence for region
2. Validates reference allele matches genome
3. Creates modified sequences for each allele
4. Runs predictions for all alleles
5. Calculates effect sizes (alt - ref)
6. Returns comprehensive results

**Example:**
```python
# Test all possible SNPs at a position
results = oracle.predict_variant_effect(
    'chr11:5247000-5248000',  # 1kb region
    'chr11:5247500',          # Variant position
    ['C', 'A', 'G', 'T'],     # C is reference
    ['ENCFF413AHU']
)

# Access results
ref_pred = results['predictions']['reference']['ENCFF413AHU']
alt1_pred = results['predictions']['alt_1']['ENCFF413AHU']
effect = results['effect_sizes']['alt_1']['ENCFF413AHU']
```

## Utility Functions

### Sequence Utilities (chorus.utils.sequence)

#### extract_sequence()
```python
def extract_sequence(
    genomic_region: str,
    genome: str = "hg38.fa"
) -> str
```

Extracts DNA sequence from reference genome.

**Parameters:**
- `genomic_region`: "chr1:1000-2000" format (1-based, inclusive)
- `genome`: Path to indexed FASTA file

**Returns:**
- DNA sequence string (uppercase)

**Note:** Properly handles coordinate conversion from 1-based genomic to 0-based pysam.

#### apply_variant()
```python
def apply_variant(
    reference_seq: str,
    position: int,
    ref: str,
    alt: str
) -> str
```

Applies a variant to a sequence.

**Parameters:**
- `reference_seq`: Original DNA sequence
- `position`: 0-based position in sequence
- `ref`: Reference allele (must match sequence)
- `alt`: Alternative allele

**Returns:**
- Modified sequence with variant applied

### Genome Management (chorus.utils.genome)

#### get_genome()
```python
def get_genome(genome_name: str = 'hg38') -> Path
```

Downloads and returns path to reference genome.

**Parameters:**
- `genome_name`: One of 'hg38', 'hg19', 'mm10', 'mm9', 'dm6', 'ce11'

**Returns:**
- Path object to genome FASTA file

**Logic:**
1. Checks if genome already downloaded
2. Downloads from UCSC if needed
3. Creates FASTA index
4. Returns path

### Gene Annotations (chorus.utils.annotations)

#### download_gencode()
```python
def download_gencode(
    version: str = 'v48',
    annotation_type: str = 'basic'
) -> Path
```

Downloads GENCODE gene annotations.

**Parameters:**
- `version`: GENCODE version (e.g., 'v48')
- `annotation_type`: 'basic' or 'comprehensive'

**Returns:**
- Path to GTF file

#### get_gene_tss()
```python
def get_gene_tss(gene_name: str) -> pd.DataFrame
```

Gets transcription start sites for a gene.

**Parameters:**
- `gene_name`: Gene symbol (e.g., 'GATA1')

**Returns:**
- DataFrame with columns: transcript_id, chrom, tss, strand, gene_name

### Visualization (chorus.utils.visualization)

#### visualize_chorus_predictions()
```python
def visualize_chorus_predictions(
    predictions: Dict[str, np.ndarray],
    chrom: str,
    start: int,
    track_ids: List[str],
    output_file: Optional[str] = None,
    bin_size: int = 128,
    style: str = 'modern',
    use_pygenometracks: bool = True,
    gtf_file: Optional[str] = None,
    show_gene_names: bool = True
) -> None
```

Creates publication-quality visualizations of predictions.

**Parameters:**
- `predictions`: Dict of track_id → prediction array
- `chrom`: Chromosome name
- `start`: Start coordinate
- `track_ids`: List of tracks to plot
- `output_file`: Save to file if provided
- `bin_size`: Bin size for predictions
- `style`: 'modern', 'classic', or 'minimal'
- `use_pygenometracks`: Use pyGenomeTracks if available
- `gtf_file`: Gene annotation file for gene track
- `show_gene_names`: Whether to label genes

## Track Management

### Track Class
```python
class Track:
    def __init__(
        self,
        name: str,
        assay_type: str,
        cell_type: str,
        data: pd.DataFrame,
        color: Optional[str] = None
    )
```

Represents a genomic signal track.

**Methods:**
- `to_bedgraph(filename)`: Save as BedGraph
- `to_bigwig(filename, chrom_sizes)`: Save as BigWig
- `normalize(method)`: Normalize values
- `smooth(window_size)`: Smooth signal

### save_predictions_as_bedgraph()
```python
def save_predictions_as_bedgraph(
    predictions: Dict[str, np.ndarray],
    chrom: str,
    start: int,
    end: Optional[int] = None,
    output_dir: str = ".",
    prefix: str = "",
    bin_size: Optional[int] = None,
    track_colors: Optional[Dict[str, str]] = None
) -> List[str]
```

Saves predictions as BedGraph files for genome browser visualization.

**Note for Enformer:** Automatically handles coordinate mapping from input window to output window.

## Environment Management

### CLI Commands

```bash
# Set up oracle environment
chorus setup --oracle enformer

# Check environment health
chorus health

# List environments
chorus list

# Remove environment
chorus remove --oracle enformer
```

### Programmatic Access

```python
# Create oracle with environment
oracle = chorus.create_oracle('enformer', use_environment=True)

# Run code in oracle's environment
result = oracle.run_code_in_environment(
    "import tensorflow; print(tensorflow.__version__)"
)
```

## Complete Example

```python
import chorus
from chorus.utils import get_genome, download_gencode

# Setup
genome = get_genome('hg38')
gtf = download_gencode()
oracle = chorus.create_oracle('enformer', reference_fasta=str(genome))
oracle.load_pretrained_model()

# Define tracks (Enformer-specific)
tracks = ['ENCFF413AHU', 'CNhs11250']  # DNase:K562, CAGE:K562

# 1. Wild-type prediction
wt = oracle.predict(('chr11', 5247000, 5248000), tracks)

# 2. Test enhancer insertion
enhancer = 'GATA' * 50
inserted = oracle.predict_region_insertion_at(
    'chr11:5247500',
    enhancer,
    tracks
)

# 3. Test variant
variant = oracle.predict_variant_effect(
    'chr11:5247000-5248000',
    'chr11:5247500',
    ['C', 'A', 'G', 'T'],  # C is reference
    tracks
)

# 4. Analyze gene expression
expr = oracle.analyze_gene_expression(
    predictions=wt,
    gene_name='HBB',  # Beta-globin
    chrom='chr11',
    start=5247000,
    end=5248000,
    gtf_file=str(gtf),
    cage_track_ids=['CNhs11250']
)

# 5. Save for visualization
oracle.save_predictions_as_bedgraph(
    wt,
    chrom='chr11',
    start=5247000,
    end=5248000,
    output_dir='results'
)
```

## Notes on Oracle-Specific Behavior

### Enformer
- Requires exactly 393,216 bp input sequence
- Output covers middle 114,688 bp of input
- Uses ENCODE and CAGE track identifiers
- Supports gene expression analysis via CAGE at TSS

### Future Oracles
- **Borzoi**: Similar to Enformer, enhanced performance
- **ChromBPNet**: Base-resolution, different track naming
- **Sei**: 21,907 profiles, custom track names

## Error Handling

Common exceptions:
- `ModelNotLoadedError`: Call `load_pretrained_model()` first
- `InvalidSequenceError`: Check sequence length and content
- `InvalidAssayError`: Use valid track identifiers for the oracle
- `InvalidRegionError`: Check genomic coordinates
- `FileFormatError`: Ensure genome file is indexed