# Chorus Examples

This directory contains examples demonstrating Chorus capabilities.

## Quick Demo

**`quick_demo.py`** - Simple demonstration of all key features:
- Wild-type prediction
- Region replacement
- Sequence insertion
- Variant effect analysis
- Saving predictions as BedGraph

Run it:
```bash
python quick_demo.py
```

## Comprehensive Analysis

**`gata1_comprehensive_analysis.ipynb`** - Complete walkthrough with:
- Detailed explanations of each method
- Gene annotation and visualization
- Performance analysis at TSS positions
- Multiple visualization approaches
- Real biological examples using GATA1 locus

Run it:
```bash
jupyter notebook gata1_comprehensive_analysis.ipynb
```

## Python Script Version

**`gata1_comprehensive_analysis.py`** - Complete Python script version demonstrating:
- All prediction methods (wild-type, replacement, insertion, variant, synthetic)
- Gene expression analysis using CAGE signals at TSS
- Saving outputs as BedGraph files
- Same analysis as the notebook but without visualizations

Run it:
```bash
python gata1_comprehensive_analysis.py
```

## Prerequisites

1. Install Chorus and set up Enformer:
```bash
chorus setup --oracle enformer
```

2. Download reference genome:
```bash
chorus genome download hg38
```

3. Run examples:
```bash
cd examples
python quick_demo.py
```

## Note on Track Identifiers

The examples use ENCODE track identifiers (e.g., ENCFF413AHU) and CAGE identifiers (e.g., CNhs11250) which are specific to Enformer and Borzoi models. Other oracles will have different track naming conventions:

- **Enformer/Borzoi**: ENCODE IDs, CAGE IDs, or descriptive names (e.g., 'DNase:K562')
- **ChromBPNet**: TF-specific tracks (e.g., 'CTCF', 'POLR2A')
- **Sei**: Custom profile names for 21,907 chromatin profiles
- **Other models**: Will vary based on training data