## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.914 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.626 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.485 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Tracks requested**: top tracks for LNCaP clone FGC
- **Cell types**: LNCaP clone FGC
- **Generated**: 2026-04-16 18:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.91); Histone modifications (ChIP-Histone): very strong mark gain (+1.01); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 58.7 | 224 | +1.914 | ≥99th | 0.860 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 399 | 470 | +0.234 | ≥99th | 0.856 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.87e+03 | 3.77e+03 | +1.011 | ≥99th | 0.882 | Very strong mark gain |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### epithelial cell of proximal tubule

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Tracks requested**: top tracks for epithelial cell of proximal tubule
- **Cell types**: epithelial cell of proximal tubule
- **Generated**: 2026-04-16 18:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72); Histone modifications (ChIP-Histone): strong mark gain (+0.58); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18); Gene expression (RNA-seq): moderate increase (+0.14).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.6 | 251 | +1.626 | ≥99th | 0.882 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 510 | +0.179 | ≥99th | 0.890 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.69e+03 | 7.02e+03 | +0.581 | ≥99th | 0.885 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.721 | ≥99th | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.7 | 65.8 | +0.581 | ≥99th | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.8 | 33.8 | +0.042 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.13 | 4.26 | +0.035 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.09e+03 | 2.12e+03 | +0.026 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.4 | +0.024 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.01e+03 | 1.03e+03 | +0.023 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.96 | 7.04 | +0.014 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 227 | 229 | +0.010 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 3.4e+03 | 3.42e+03 | +0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0236 | 0.0275 | +0.144 | ≥99th | 0.192 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00376 | 0.00426 | +0.102 | ≥99th | 0.140 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.108 | 0.119 | +0.098 | ≥99th | 0.324 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.042 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0629 | 0.065 | +0.034 | ≥99th | 0.263 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.0679 | 0.0692 | +0.019 | ≥99th | 0.261 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 393 | 399 | +0.015 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 292 | 294 | +0.006 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMIGO1 (exons) | 0.314 | 0.313 | -0.005 | ≤1st | 0.593 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AKNAD1 (exons) | 0.00871 | 0.00867 | -0.004 | ≤1st | 0.165 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00787 | 0.00699 | -0.001 | ≥99th | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00117 | 0.00114 | -0.000 | — | 0.693 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### renal cortical epithelial cell

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Tracks requested**: top tracks for renal cortical epithelial cell
- **Cell types**: renal cortical epithelial cell
- **Generated**: 2026-04-16 18:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.48); TSS activity (CAGE/PRO-CAP): strong increase (+0.68); Gene expression (RNA-seq): moderate increase (+0.17).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 224 | 628 | +1.485 | ≥99th | 0.938 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.09 | 15.2 | +0.678 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.8 | 62.9 | +0.546 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27 | 27.8 | +0.040 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.07 | 4.18 | +0.031 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.025 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.1e+03 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.4 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.74 | 6.82 | +0.014 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 211 | 212 | +0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM1 TSS | 1.14e+03 | 1.13e+03 | -0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0659 | 0.0786 | +0.174 | ≥99th | 0.257 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.133 | 0.153 | +0.143 | ≥99th | 0.365 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0237 | 0.0269 | +0.125 | ≥99th | 0.189 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.1 | 66 | +0.060 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 193 | 204 | +0.053 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.332 | 0.345 | +0.037 | ≥99th | 0.673 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.137 | 0.142 | +0.035 | ≥99th | 0.389 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 486 | +0.007 | ≥99th | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — EPS8L3 (exons) | 0.102 | 0.101 | -0.006 | ≤1st | 0.326 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 0.474 | 0.476 | +0.005 | ≥99th | 0.804 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00637 | 0.00574 | -0.001 | — | 0.860 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00102 | 0.00102 | +0.000 | — | 0.622 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

