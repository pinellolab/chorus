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

## Gene Expression Analysis

**`gene_expression_analysis.py`** - Focused example on:
- Analyzing gene expression using CAGE signals
- TSS (Transcription Start Site) identification
- Comparing wild-type vs modified sequences
- Future Borzoi implementation notes

**`gata1_comprehensive_analysis.py`** - Python script version of the notebook for non-interactive use.

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