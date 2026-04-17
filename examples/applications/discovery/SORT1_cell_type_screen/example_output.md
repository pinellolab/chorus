## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.904 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.625 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.498 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 3 tracks for LNCaP clone FGC
- **Cell types**: LNCaP clone FGC
- **Generated**: 2026-04-17 03:01 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.90); Histone modifications (ChIP-Histone): very strong mark gain (+0.99); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 61.7 | 234 | +1.904 | ≥99th | 0.863 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 406 | 478 | +0.233 | ≥99th | 0.859 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.88e+03 | 3.73e+03 | +0.987 | ≥99th | 0.882 | Very strong mark gain |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### epithelial cell of proximal tubule

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 9 tracks for epithelial cell of proximal tubule
- **Cell types**: epithelial cell of proximal tubule
- **Generated**: 2026-04-17 03:03 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72); Histone modifications (ChIP-Histone): strong mark gain (+0.57); Transcription factor binding (ChIP-TF): moderate binding gain (+0.17); Gene expression (RNA-seq): moderate increase (+0.13).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 78.9 | 246 | +1.625 | ≥99th | 0.881 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 456 | 514 | +0.171 | ≥99th | 0.892 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.51e+03 | 6.67e+03 | +0.567 | ≥99th | 0.883 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 9.47 | 16.2 | +0.721 | ≥99th | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 41.8 | 63 | +0.578 | ≥99th | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.3 | 33.4 | +0.045 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.1 | 4.22 | +0.033 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.03e+03 | 1.06e+03 | +0.033 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.07e+03 | 2.12e+03 | +0.033 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 9.48 | 9.62 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 228 | 225 | -0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.83 | 6.9 | +0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM3 TSS | 4.55e+03 | 4.58e+03 | +0.012 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0246 | 0.0282 | +0.133 | ≥99th | 0.193 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00324 | 0.00367 | +0.098 | ≥99th | 0.138 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.114 | 0.125 | +0.089 | ≥99th | 0.334 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 178 | 186 | +0.042 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0579 | 0.0596 | +0.029 | ≥99th | 0.256 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.0646 | 0.0657 | +0.016 | ≥99th | 0.256 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 388 | 394 | +0.015 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 285 | 287 | +0.007 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — EPS8L3 (exons) | 0.159 | 0.159 | -0.004 | ≤1st | 0.409 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CYB561D1 (exons) | 0.0914 | 0.0911 | -0.004 | ≤1st | 0.303 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00777 | 0.00694 | -0.001 | ≥99th | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.0012 | 0.00118 | -0.000 | — | 0.700 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### renal cortical epithelial cell

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 7 tracks for renal cortical epithelial cell
- **Cell types**: renal cortical epithelial cell
- **Generated**: 2026-04-17 03:05 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.50); TSS activity (CAGE/PRO-CAP): strong increase (+0.68); Gene expression (RNA-seq): moderate increase (+0.16).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 218 | 617 | +1.498 | ≥99th | 0.936 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 8.52 | 14.2 | +0.678 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 40.8 | 59.9 | +0.542 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 26.7 | 27.6 | +0.042 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.031 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.03 | 4.13 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.09e+03 | 1.11e+03 | +0.026 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 10.3 | 10.5 | +0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 210 | 208 | -0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.58 | 6.64 | +0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM3 TSS | 4.13e+03 | 4.16e+03 | +0.010 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0676 | 0.0799 | +0.164 | ≥99th | 0.261 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.136 | 0.156 | +0.134 | ≥99th | 0.371 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0202 | 0.0229 | +0.121 | ≥99th | 0.181 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 60.9 | 64.7 | +0.062 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 189 | 199 | +0.052 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.292 | 0.302 | +0.035 | ≥99th | 0.617 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.138 | 0.141 | +0.026 | ≥99th | 0.389 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 466 | 470 | +0.008 | ≥99th | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — EPS8L3 (exons) | 0.0928 | 0.0922 | -0.006 | ≤1st | 0.308 | Minimal effect |
| RNA:renal cortical epithelial cell — EPS8L3 (exons) | 0.00147 | 0.00146 | -0.005 | ≤1st | 0.106 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00628 | 0.0057 | -0.001 | — | 0.859 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00105 | 0.00106 | +0.000 | — | 0.631 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

