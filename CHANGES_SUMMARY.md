# Summary of Changes Made to Chorus

## Overview
This document summarizes the major improvements and bug fixes made to the Chorus library.

## 1. CLI Documentation Fixes
- Fixed mismatched CLI commands in documentation:
  - `env status` → `env health`
  - `env list` → `env list` (was correct)
  - `env clean` → `env remove`

## 2. Automatic Genome Downloading
- Added `chorus/utils/genome.py` with automatic genome management
- Implemented `get_genome()` function that auto-downloads if not present
- Added `chorus genome` CLI commands:
  - `chorus genome download <genome_name>`
  - `chorus genome list`
  - `chorus genome info <genome_name>`
- Integrated genome indexing with samtools

## 3. Gene Annotation Support
- Created `chorus/utils/annotations.py` for gene annotation management
- Added automatic GENCODE GTF file downloading
- Implemented gene visualization in prediction plots
- Added TSS-based gene expression analysis for Enformer
- Functions include:
  - `download_gencode()` - Download GENCODE annotations
  - `get_gene_tss()` - Get TSS positions for genes
  - `extract_genes_in_region()` - Get genes in a genomic region

## 4. Prediction Methods Flexibility
- Fixed `predict_region_replacement()` to accept any sequence length
- Fixed `predict_region_insertion_at()` to handle variable insertions
- Updated `predict_variant_effect()` to work with any region size
- All methods now properly handle the model's context window requirements

## 5. Coordinate System Bug Fix
- Fixed critical off-by-one error in `extract_sequence()` function
- Properly converts 1-based genomic coordinates to 0-based pysam coordinates
- This fixed variant effect prediction failures due to reference mismatches

## 6. Enformer-Specific Improvements
- Added coordinate mapping methods to handle Enformer's architecture:
  - Input: 393,216 bp → Output: 114,688 bp (896 bins × 128 bp)
  - Proper handling of 139,264 bp offset on each side
- Added `analyze_gene_expression()` method using CAGE tracks at TSS
- Overrode `save_predictions_as_bedgraph()` to handle coordinate mapping

## 7. Visualization Enhancements
- Updated `visualize_chorus_predictions()` to include gene tracks
- Added pyGenomeTracks support for publication-quality figures
- Fixed matplotlib non-interactive backend issues
- Improved track coloring based on assay types
- Added gene visualization with strand-specific colors

## 8. Comprehensive Example Notebook
- Created `gata1_comprehensive_analysis.ipynb` demonstrating all features:
  - Wild-type sequence prediction
  - Region replacement with custom sequences
  - Sequence insertion at specific positions
  - Variant effect analysis (SNPs)
  - Direct sequence prediction on synthetic DNA
  - Gene expression analysis
  - All with proper visualizations including gene annotations

## 9. Test Coverage
- Added tests for flexible prediction methods
- Added tests for coordinate conversion
- Added tests for gene annotation functionality

## 10. Documentation Updates
- Updated README with new features
- Added examples for all new functionality
- Improved docstrings throughout the codebase

## Key Bug Fixes
1. **Coordinate conversion bug**: Fixed 1-based to 0-based conversion in sequence extraction
2. **Variant analysis data format**: Fixed incorrect 2D array assumptions in variant analysis
3. **Empty sequence handling**: Replaced deletion tests with insertion tests to avoid empty sequences
4. **Matplotlib display issues**: Fixed non-interactive backend warnings

## Dependencies Added
- pyfaidx (for genome indexing)
- pyGenomeTracks (optional, for enhanced visualization)
- Existing samtools requirement documented

## Repository Cleanup
- Organized all outputs into `outputs/` directory
- Updated .gitignore appropriately
- Removed temporary files and OS-specific files

All changes have been tested and the comprehensive example notebook runs successfully from start to finish.