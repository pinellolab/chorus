# Chorus Examples

This directory contains comprehensive examples demonstrating the capabilities of the Chorus library.

## Main Examples

### 1. GATA1 Comprehensive Analysis
The most complete example showing all features of Chorus using the GATA1 transcription factor region:

- **Script version**: `gata1_comprehensive_analysis.py`
- **Notebook version**: `gata1_comprehensive_analysis.ipynb`

These examples demonstrate:
- Wild-type sequence prediction
- Region replacement
- Sequence insertion
- Variant effect analysis
- Direct sequence prediction
- Saving predictions as BedGraph files

### 2. Quick Start Examples
Simple examples to get started quickly:

- `enformer_quick_start.py` - Minimal example for Enformer
- `simple_predictions.py` - Basic prediction examples

### 3. Feature-Specific Examples

#### Region Manipulation
- `region_manipulation_examples.py` - Comprehensive region replacement and insertion
- `flexible_region_manipulation.py` - Examples with sequences of any length (1bp to full context)

#### Prediction Methods
- `prediction_methods_demo.py` - Demonstrates all three main prediction methods

#### Variant Analysis
- `variant_analysis.ipynb` - Detailed variant effect analysis notebook

### 4. Environment and Setup
- `environment_demo.py` - Demonstrates environment isolation features

## Running the Examples

1. Make sure Chorus is installed and the Enformer environment is set up:
```bash
chorus setup --oracle enformer
```

2. Download the reference genome if needed:
```bash
chorus genome download hg38
```

3. Run any example:
```bash
python examples/gata1_comprehensive_analysis.py
```

Or open the notebooks in Jupyter:
```bash
jupyter notebook examples/gata1_comprehensive_analysis.ipynb
```

## Output Files

Most examples generate BedGraph files that can be loaded into genome browsers (UCSC Genome Browser, IGV, etc.) for visualization. The GATA1 examples use a consistent naming scheme:

- `a_wt_*.bedgraph` - Wild-type predictions
- `b_replacement_*.bedgraph` - Region replacement results
- `c_insertion_*.bedgraph` - Sequence insertion results
- `d_variant_*.bedgraph` - Variant effect predictions
- `e_synthetic_*.bedgraph` - Synthetic sequence predictions