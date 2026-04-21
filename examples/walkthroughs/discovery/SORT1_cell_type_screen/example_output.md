## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.911 | DNASE:LNCaP clone FGC | 3 |
| 2 | epithelial cell of proximal tubule | +1.627 | DNASE:epithelial cell of proximal tubule | 9 |
| 3 | renal cortical epithelial cell | +1.484 | DNASE:renal cortical epithelial cell | 7 |

### LNCaP clone FGC

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 3 tracks for LNCaP clone FGC
- **Cell types**: LNCaP clone FGC
- **Generated**: 2026-04-21 02:06 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.91, DNASE:LNCaP clone FGC); Histone modifications (ChIP-Histone): very strong mark gain (+1.01, CHIP:H3K4me3:LNCaP clone FGC); Transcription factor binding (ChIP-TF): moderate binding gain (+0.24, CHIP:CTCF:LNCaP clone FGC).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 58.9 | 224 | +1.911 | ≥99th | 0.861 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 399 | 470 | +0.237 | ≥99th | 0.856 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.87e+03 | 3.77e+03 | +1.009 | ≥99th | 0.882 | Very strong mark gain |

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
- **Generated**: 2026-04-21 02:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63, DNASE:epithelial cell of proximal tubule); TSS activity (CAGE/PRO-CAP): very strong increase (+0.73, CAGE:epithelial cell of proximal tubule); Histone modifications (ChIP-Histone): strong mark gain (+0.58, CHIP:H3K4me3:epithelial cell of proximal tubule); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18, CHIP:CTCF:epithelial cell of proximal tubule); Gene expression (RNA-seq): moderate increase (+0.15, RNA:epithelial cell of proximal tubule).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.5 | 251 | +1.627 | ≥99th | 0.882 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 510 | +0.180 | ≥99th | 0.890 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.68e+03 | 6.99e+03 | +0.579 | ≥99th | 0.885 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.725 | ≥99th | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.6 | 65.8 | +0.584 | ≥99th | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.9 | 33.8 | +0.039 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.08e+03 | 2.13e+03 | +0.031 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.14 | 4.25 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.4 | +0.024 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.01e+03 | 1.03e+03 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.94 | 7.04 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 3.39e+03 | 3.42e+03 | +0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 228 | 229 | +0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0236 | 0.0275 | +0.146 | ≥99th | 0.192 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00378 | 0.00425 | +0.092 | ≥99th | 0.141 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.109 | 0.119 | +0.090 | ≥99th | 0.325 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.042 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0628 | 0.0649 | +0.032 | ≥99th | 0.263 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.068 | 0.0695 | +0.021 | ≥99th | 0.262 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 393 | 399 | +0.014 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMPD2 (exons) | 0.0617 | 0.0612 | -0.007 | ≤1st | 0.250 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 292 | 294 | +0.006 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GNAT2 (exons) | 0.284 | 0.282 | -0.005 | ≤1st | 0.571 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00786 | 0.00699 | -0.001 | ≥99th | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00117 | 0.00114 | -0.000 | near-zero | 0.693 | Minimal effect |

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
- **Generated**: 2026-04-21 02:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.48, DNASE:renal cortical epithelial cell); TSS activity (CAGE/PRO-CAP): strong increase (+0.68, CAGE:renal cortical epithelial cell); Gene expression (RNA-seq): moderate increase (+0.17, RNA:renal cortical epithelial cell).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 224 | 628 | +1.484 | ≥99th | 0.938 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.08 | 15.2 | +0.682 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.7 | 62.9 | +0.549 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27.1 | 27.8 | +0.036 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.031 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.09 | 4.17 | +0.023 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.4 | +0.020 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.73 | 6.82 | +0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.09e+03 | +0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 3.66e+03 | 3.69e+03 | +0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — ELAPOR1 TSS | 323 | 321 | -0.008 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0659 | 0.0786 | +0.173 | ≥99th | 0.257 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.134 | 0.153 | +0.134 | ≥99th | 0.367 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0238 | 0.0268 | +0.113 | ≥99th | 0.189 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.1 | 65.9 | +0.059 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 193 | 204 | +0.052 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.332 | 0.344 | +0.035 | ≥99th | 0.673 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.138 | 0.143 | +0.033 | ≥99th | 0.390 | Minimal effect |
| RNA:renal cortical epithelial cell — AMPD2 (exons) | 0.11 | 0.109 | -0.008 | ≤1st | 0.340 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 486 | +0.007 | ≥99th | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — PSMA5 (exons) | 0.225 | 0.227 | +0.006 | ≥99th | 0.515 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00636 | 0.00575 | -0.001 | near-zero | 0.860 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00102 | 0.00102 | +0.000 | near-zero | 0.622 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

