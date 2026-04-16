## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.903 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.628 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.483 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Tracks requested**: top tracks for LNCaP clone FGC
- **Cell types**: LNCaP clone FGC
- **Generated**: 2026-04-16 19:22 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.90); Histone modifications (ChIP-Histone): very strong mark gain (+1.00); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 59.2 | 224 | +1.903 | ≥99th | 0.861 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 400 | 469 | +0.229 | ≥99th | 0.857 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.89e+03 | 3.77e+03 | +0.998 | ≥99th | 0.882 | Very strong mark gain |

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
- **Generated**: 2026-04-16 19:22 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72); Histone modifications (ChIP-Histone): strong mark gain (+0.58); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18); Gene expression (RNA-seq): moderate increase (+0.15).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.5 | 251 | +1.628 | ≥99th | 0.882 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 510 | +0.180 | ≥99th | 0.890 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.69e+03 | 7.01e+03 | +0.580 | ≥99th | 0.885 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.718 | ≥99th | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.7 | 65.7 | +0.578 | ≥99th | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.8 | 33.8 | +0.041 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.08e+03 | 2.13e+03 | +0.033 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.14 | 4.26 | +0.032 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.4 | +0.023 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.02e+03 | 1.03e+03 | +0.022 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 228 | 231 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.94 | 7.04 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CFAP276 TSS | 25.8 | 25.6 | -0.011 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0236 | 0.0276 | +0.148 | ≥99th | 0.192 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00376 | 0.00425 | +0.097 | ≥99th | 0.140 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.109 | 0.12 | +0.096 | ≥99th | 0.325 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.043 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0628 | 0.0649 | +0.032 | ≥99th | 0.263 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.0675 | 0.069 | +0.021 | ≥99th | 0.261 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 393 | 399 | +0.016 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 293 | 294 | +0.005 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMIGO1 (exons) | 130 | 129 | -0.004 | ≤1st | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SARS1 (exons) | 0.21 | 0.211 | +0.004 | ≥99th | 0.476 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00787 | 0.00701 | -0.001 | ≥99th | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00117 | 0.00115 | -0.000 | — | 0.693 | Minimal effect |

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
- **Generated**: 2026-04-16 19:22 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.48); TSS activity (CAGE/PRO-CAP): strong increase (+0.68); Gene expression (RNA-seq): moderate increase (+0.17).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 224 | 628 | +1.483 | ≥99th | 0.938 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.11 | 15.2 | +0.675 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.8 | 62.8 | +0.543 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27.1 | 27.8 | +0.039 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.032 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.09 | 4.18 | +0.026 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.4 | +0.020 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 212 | 215 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.1e+03 | +0.018 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.73 | 6.81 | +0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GPR61 TSS | 9.27 | 9.2 | -0.011 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.066 | 0.0788 | +0.174 | ≥99th | 0.257 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.134 | 0.153 | +0.137 | ≥99th | 0.367 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0237 | 0.0268 | +0.120 | ≥99th | 0.189 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.2 | 66 | +0.060 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 193 | 204 | +0.053 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.332 | 0.344 | +0.037 | ≥99th | 0.673 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.137 | 0.142 | +0.035 | ≥99th | 0.388 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 487 | +0.007 | ≥99th | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 0.474 | 0.477 | +0.006 | ≥99th | 0.804 | Minimal effect |
| RNA:renal cortical epithelial cell — PSMA5 (exons) | 0.226 | 0.227 | +0.005 | ≥99th | 0.515 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00636 | 0.00576 | -0.001 | — | 0.860 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00102 | 0.00102 | +0.000 | — | 0.621 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

