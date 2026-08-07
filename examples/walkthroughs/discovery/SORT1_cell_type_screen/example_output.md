## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | HepG2 | +1.334 | DNASE:HepG2 | 562 |
| 2 | MCF 10A | +1.440 | DNASE:MCF 10A | 6 |
| 3 | amniotic epithelial cell | +2.898 | DNASE:amniotic epithelial cell | 3 |

### HepG2

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 30 tracks for HepG2
- **Cell types**: HepG2
- **Generated**: 2026-08-07 12:24 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: TSS activity (CAGE/PRO-CAP): very strong increase (+1.50, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.33, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.25, CHIP:H3K27ac:HepG2); Transcription factor binding (ChIP-TF): very strong binding gain (+1.02, CHIP:ARID3A:HepG2); Gene expression (RNA-seq): strong increase (+0.47, RNA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 660 | 1.67e+03 | +1.334 | ≥99th | 0.973 | Very strong opening |
| ATAC:HepG2 | 452 | 752 | +0.732 | ≥99th | 0.935 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:ARID3A:HepG2 | 2e+03 | 4.07e+03 | +1.023 | ≥99th | 0.997 | Very strong binding gain |
| CHIP:ARID3A:HepG2 | 1.58e+03 | 3.06e+03 | +0.955 | ≥99th | 0.992 | Very strong binding gain |
| CHIP:ARID4B:HepG2 | 3.01e+03 | 5.38e+03 | +0.837 | ≥99th | 0.937 | Very strong binding gain |
| CHIP:AHDC1:HepG2 | 951 | 1.63e+03 | +0.773 | ≥99th | 0.997 | Very strong binding gain |
| CHIP:ARID4A:HepG2 | 1.14e+03 | 1.84e+03 | +0.691 | ≥99th | 0.918 | Strong binding gain |
| CHIP:AFF4:HepG2 | 957 | 1.53e+03 | +0.676 | ≥99th | 0.943 | Strong binding gain |
| CHIP:AHR:HepG2 | 1.33e+03 | 1.93e+03 | +0.542 | ≥99th | 0.914 | Strong binding gain |
| CHIP:ARID2:HepG2 | 583 | 826 | +0.502 | ≥99th | 0.939 | Strong binding gain |
| CHIP:AKAP8:HepG2 | 986 | 1.39e+03 | +0.497 | ≥99th | 0.938 | Strong binding gain |
| CHIP:ARHGAP35:HepG2 | 854 | 1.09e+03 | +0.353 | ≥99th | 0.911 | Strong binding gain |
| _…showing top 10 of 11 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.51e+04 | 3.58e+04 | +1.251 | ≥99th | 0.946 | Very strong mark gain |
| CHIP:H3K9ac:HepG2 | 1.47e+04 | 2.45e+04 | +0.730 | ≥99th | 0.905 | Very strong mark gain |
| CHIP:H3K4me3:HepG2 | 8.14e+03 | 1.13e+04 | +0.475 | ≥99th | 0.881 | Strong mark gain |
| CHIP:H3K27me3:HepG2 | 1.2e+03 | 868 | -0.467 | ≥99th | 0.425 | Strong mark loss |
| CHIP:H4K20me1:HepG2 | 1.1e+03 | 850 | -0.368 | ≥99th | 0.278 | Strong mark loss |
| CHIP:H3K36me3:HepG2 | 1.12e+03 | 882 | -0.347 | ≥99th | 0.730 | Strong mark loss |
| CHIP:H3K4me2:HepG2 | 2.54e+04 | 3.22e+04 | +0.341 | 0.99 | 0.946 | Strong mark gain |
| CHIP:H3K9me3:HepG2 | 513 | 437 | -0.228 | ≥99th | 0.139 | Moderate mark loss |
| CHIP:H3K79me2:HepG2 | 1.09e+03 | 1.23e+03 | +0.174 | ≥99th | 0.705 | Moderate mark gain |
| CHIP:H3K4me1:HepG2 | 1.58e+04 | 1.66e+04 | +0.073 | 0.89 | 0.998 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 25.2 | 73.3 | +1.502 | ≥99th | 0.919 | Very strong increase |
| CAGE:HepG2 — variant site | 75.2 | 174 | +1.203 | ≥99th | 0.942 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.26e+03 | 2.66e+03 | +0.235 | 0.97 | 0.980 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 188 | 218 | +0.213 | 0.97 | 0.954 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.47 | 2.98 | +0.198 | 0.96 | 0.841 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52 | 58.8 | +0.172 | 0.96 | 0.936 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 666 | 751 | +0.172 | 0.96 | 0.965 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 1.74 | 2 | +0.132 | 0.94 | 0.819 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.97 | 8.72 | +0.115 | 0.93 | 0.886 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.53e+03 | 3.67e+03 | +0.058 | 0.85 | 0.986 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HepG2 — PSRC1 (exons) | 0.00283 | 0.00509 | +0.465 | ≥99th | 0.461 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.184 | 0.289 | +0.449 | ≥99th | 0.462 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.739 | 1.15 | +0.441 | ≥99th | 0.689 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.147 | 0.219 | +0.396 | ≥99th | 0.703 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.853 | 1.24 | +0.370 | ≥99th | 0.860 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.00193 | 0.00322 | +0.364 | ≥99th | 0.458 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.103 | 0.148 | +0.362 | ≥99th | 0.697 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 1.21 | 1.58 | +0.268 | ≥99th | 0.885 | Moderate increase |
| RNA:HepG2 — MYBPHL (exons) | 0.0796 | 0.102 | +0.243 | ≥99th | 0.383 | Moderate increase |
| RNA:HepG2 — MYBPHL (exons) | 0.00319 | 0.00433 | +0.239 | ≥99th | 0.470 | Moderate increase |
| _…showing top 10 of 145 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### MCF 10A

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 6 tracks for MCF 10A
- **Cell types**: MCF 10A
- **Generated**: 2026-08-07 12:25 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.44, DNASE:MCF 10A); Gene expression (RNA-seq): moderate increase (+0.15, RNA:MCF 10A).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:MCF 10A | 338 | 919 | +1.440 | ≥99th | 0.931 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:MCF 10A | 294 | 289 | -0.025 | 0.85 | 0.749 | Minimal effect |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:MCF 10A — PSRC1 (exons) | 0.0032 | 0.00387 | +0.148 | ≥99th | 0.476 | Moderate increase |
| RNA:MCF 10A — PSRC1 (exons) | 0.367 | 0.414 | +0.120 | ≥99th | 0.783 | Moderate increase |
| RNA:MCF 10A — CELSR2 (exons) | 0.000445 | 0.000576 | +0.086 | ≥99th | 0.323 | Moderate increase |
| RNA:MCF 10A — MYBPHL (exons) | 0.000636 | 0.000738 | +0.061 | ≥99th | 0.347 | Moderate increase |
| RNA:MCF 10A — CELSR2 (exons) | 0.527 | 0.549 | +0.041 | ≥99th | 0.815 | Minimal effect |
| RNA:MCF 10A — SORT1 (exons) | 0.000442 | 0.000474 | +0.022 | ≥99th | 0.316 | Minimal effect |
| RNA:MCF 10A — SORT1 (exons) | 0.531 | 0.54 | +0.016 | ≥99th | 0.816 | Minimal effect |
| RNA:MCF 10A — MYBPHL (exons) | 0.000431 | 0.000448 | +0.012 | ≥99th | 0.320 | Minimal effect |
| RNA:MCF 10A — SARS1 (exons) | 0.0115 | 0.0116 | +0.006 | 0.98 | 0.575 | Minimal effect |
| RNA:MCF 10A — SYPL2 (exons) | 0.00488 | 0.00484 | -0.006 | ≤1st | 0.506 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:MCF 10A | 0.00736 | 0.00599 | -0.002 | 0.90 | 0.875 | Minimal effect |
| SPLICE_SITES:MCF 10A | 0.00128 | 0.00127 | -0.000 | near-zero | 0.710 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### amniotic epithelial cell

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 3 tracks for amniotic epithelial cell
- **Cell types**: amniotic epithelial cell
- **Generated**: 2026-08-07 12:25 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+2.90, DNASE:amniotic epithelial cell); TSS activity (CAGE/PRO-CAP): strong decrease (-0.61, CAGE:amniotic epithelial cell).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:amniotic epithelial cell | 53 | 401 | +2.898 | ≥99th | 0.840 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:amniotic epithelial cell — variant site | 83.7 | 54.6 | -0.607 | ≥99th | 0.945 | Strong decrease |
| CAGE:amniotic epithelial cell — variant site | 21.9 | 17.5 | -0.306 | 0.98 | 0.919 | Strong decrease |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 1.02e+03 | 989 | -0.039 | 0.81 | 0.970 | Minimal effect |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 4.26 | 4.19 | -0.019 | 0.67 | 0.869 | Minimal effect |
| CAGE:amniotic epithelial cell — GNAI3 TSS | 215 | 218 | +0.018 | 0.66 | 0.956 | Minimal effect |
| CAGE:amniotic epithelial cell — PSRC1 TSS | 1.78e+03 | 1.8e+03 | +0.016 | 0.64 | 0.978 | Minimal effect |
| CAGE:amniotic epithelial cell — SORT1 TSS | 2.86e+03 | 2.89e+03 | +0.013 | 0.58 | 0.985 | Minimal effect |
| CAGE:amniotic epithelial cell — SORT1 TSS | 5.86 | 5.91 | +0.012 | 0.57 | 0.877 | Minimal effect |
| CAGE:amniotic epithelial cell — AMIGO1 TSS | 224 | 222 | -0.011 | 0.55 | 0.957 | Minimal effect |
| CAGE:amniotic epithelial cell — GNAT2 TSS | 27.5 | 27.7 | +0.011 | 0.54 | 0.926 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

