## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | HepG2 | +1.396 | DNASE:HepG2 | 562 |
| 2 | MCF 10A | +1.520 | DNASE:MCF 10A | 6 |
| 3 | left lobe of liver | +1.840 | DNASE:left lobe of liver | 6 |

### HepG2

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 30 tracks for HepG2
- **Cell types**: HepG2
- **Generated**: 2026-04-29 15:43 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: TSS activity (CAGE/PRO-CAP): very strong increase (+1.55, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.40, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.33, CHIP:H3K27ac:HepG2); Transcription factor binding (ChIP-TF): very strong binding gain (+0.99, CHIP:ARID3A:HepG2); Gene expression (RNA-seq): very strong increase (+0.89, RNA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 596 | 1.57e+03 | +1.396 | ≥99th | 0.968 | Very strong opening |
| ATAC:HepG2 | 436 | 736 | +0.753 | ≥99th | 0.933 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:ARID3A:HepG2 | 2.42e+03 | 4.81e+03 | +0.994 | ≥99th | 0.999 | Very strong binding gain |
| CHIP:ARID3A:HepG2 | 1.84e+03 | 3.57e+03 | +0.953 | ≥99th | 0.998 | Very strong binding gain |
| CHIP:ARID4B:HepG2 | 3.87e+03 | 6.67e+03 | +0.785 | ≥99th | 0.951 | Very strong binding gain |
| CHIP:AHDC1:HepG2 | 1.2e+03 | 2e+03 | +0.732 | ≥99th | 0.999 | Very strong binding gain |
| CHIP:AFF4:HepG2 | 1.26e+03 | 1.98e+03 | +0.655 | ≥99th | 0.974 | Strong binding gain |
| CHIP:ARID4A:HepG2 | 1.57e+03 | 2.43e+03 | +0.633 | ≥99th | 0.939 | Strong binding gain |
| CHIP:AHR:HepG2 | 1.93e+03 | 2.68e+03 | +0.476 | ≥99th | 0.951 | Strong binding gain |
| CHIP:ARID2:HepG2 | 790 | 1.09e+03 | +0.460 | ≥99th | 0.986 | Strong binding gain |
| CHIP:AKAP8:HepG2 | 1.35e+03 | 1.85e+03 | +0.457 | ≥99th | 0.972 | Strong binding gain |
| CHIP:ARHGAP35:HepG2 | 1.24e+03 | 1.55e+03 | +0.320 | ≥99th | 0.952 | Strong binding gain |
| _…showing top 10 of 11 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.43e+04 | 3.6e+04 | +1.332 | ≥99th | 0.999 | Very strong mark gain |
| CHIP:H3K9ac:HepG2 | 1.47e+04 | 2.54e+04 | +0.795 | ≥99th | 0.996 | Very strong mark gain |
| CHIP:H3K4me3:HepG2 | 8.15e+03 | 1.19e+04 | +0.550 | ≥99th | 0.966 | Strong mark gain |
| CHIP:H3K27me3:HepG2 | 1.33e+03 | 954 | -0.478 | ≥99th | 0.994 | Strong mark loss |
| CHIP:H3K4me2:HepG2 | 2.57e+04 | 3.38e+04 | +0.392 | ≥99th | 1.000 | Strong mark gain |
| CHIP:H4K20me1:HepG2 | 1.15e+03 | 883 | -0.385 | ≥99th | 0.991 | Strong mark loss |
| CHIP:H3K36me3:HepG2 | 1.25e+03 | 985 | -0.348 | ≥99th | 0.981 | Strong mark loss |
| CHIP:H3K9me3:HepG2 | 549 | 468 | -0.230 | ≥99th | 0.869 | Moderate mark loss |
| CHIP:H3K79me2:HepG2 | 1.15e+03 | 1.35e+03 | +0.227 | ≥99th | 0.948 | Moderate mark gain |
| CHIP:H3K4me1:HepG2 | 1.64e+04 | 1.76e+04 | +0.102 | ≥99th | 1.000 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 22.9 | 69.1 | +1.551 | ≥99th | 1.000 | Very strong increase |
| CAGE:HepG2 — variant site | 70.6 | 167 | +1.235 | ≥99th | 1.000 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.27e+03 | 2.67e+03 | +0.238 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.49 | 3.02 | +0.207 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 627 | 721 | +0.201 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 257 | 294 | +0.195 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52 | 59 | +0.178 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 2.4 | 2.78 | +0.152 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.96 | 8.71 | +0.117 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.61e+03 | 3.76e+03 | +0.056 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HepG2 — CELSR2 (exons) | 0.0269 | 0.0669 | +0.891 | ≥99th | 0.235 | Very strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.0191 | 0.0407 | +0.730 | ≥99th | 0.228 | Very strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.688 | 1.26 | +0.603 | ≥99th | 0.935 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.453 | 0.765 | +0.523 | ≥99th | 0.737 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 62.1 | 98.7 | +0.462 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 212 | 328 | +0.436 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 48.8 | 73.3 | +0.408 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 35.2 | 51 | +0.371 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 244 | 352 | +0.366 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — MYBPHL (exons) | 0.164 | 0.222 | +0.302 | ≥99th | 0.415 | Strong increase |
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
- **Generated**: 2026-04-29 15:44 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.52, DNASE:MCF 10A); Gene expression (RNA-seq): moderate increase (+0.26, RNA:MCF 10A).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:MCF 10A | 296 | 850 | +1.520 | ≥99th | 0.926 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:MCF 10A | 420 | 418 | -0.007 | ≥99th | 0.903 | Minimal effect |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:MCF 10A — CELSR2 (exons) | 0.128 | 0.166 | +0.264 | ≥99th | 0.409 | Moderate increase |
| RNA:MCF 10A — PSRC1 (exons) | 0.671 | 0.823 | +0.204 | ≥99th | 0.944 | Moderate increase |
| RNA:MCF 10A — MYBPHL (exons) | 0.0915 | 0.105 | +0.135 | ≥99th | 0.335 | Moderate increase |
| RNA:MCF 10A — PSRC1 (exons) | 101 | 114 | +0.115 | ≥99th | 1.000 | Moderate increase |
| RNA:MCF 10A — SORT1 (exons) | 0.157 | 0.168 | +0.068 | ≥99th | 0.439 | Moderate increase |
| RNA:MCF 10A — CELSR2 (exons) | 171 | 178 | +0.040 | ≥99th | 1.000 | Minimal effect |
| RNA:MCF 10A — MYBPHL (exons) | 0.0527 | 0.0543 | +0.030 | ≥99th | 0.273 | Minimal effect |
| RNA:MCF 10A — ELAPOR1 (exons) | 0.026 | 0.0264 | +0.016 | ≥99th | 0.221 | Minimal effect |
| RNA:MCF 10A — SORT1 (exons) | 175 | 177 | +0.012 | ≥99th | 1.000 | Minimal effect |
| RNA:MCF 10A — SARS1 (exons) | 2.04 | 2.06 | +0.010 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:MCF 10A | 0.00743 | 0.00609 | -0.002 | ≥99th | 0.878 | Minimal effect |
| SPLICE_SITES:MCF 10A | 0.00121 | 0.00122 | +0.000 | near-zero | 0.696 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### left lobe of liver

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 6 tracks for left lobe of liver
- **Cell types**: left lobe of liver
- **Generated**: 2026-04-29 15:44 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.84, DNASE:left lobe of liver); Gene expression (RNA-seq): very strong increase (+0.86, RNA:left lobe of liver).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:left lobe of liver | 185 | 665 | +1.840 | ≥99th | 0.906 | Very strong opening |
| ATAC:left lobe of liver | 194 | 658 | +1.758 | ≥99th | 0.895 | Very strong opening |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:left lobe of liver — PSRC1 (exons) | 0.27 | 0.641 | +0.863 | ≥99th | 0.783 | Very strong increase |
| RNA:left lobe of liver — PSRC1 (exons) | 26.6 | 57.9 | +0.778 | ≥99th | 1.000 | Very strong increase |
| RNA:left lobe of liver — CELSR2 (exons) | 0.0367 | 0.0789 | +0.750 | ≥99th | 0.267 | Very strong increase |
| RNA:left lobe of liver — CELSR2 (exons) | 30.6 | 47.4 | +0.438 | ≥99th | 1.000 | Strong increase |
| RNA:left lobe of liver — MYBPHL (exons) | 0.0339 | 0.0457 | +0.292 | ≥99th | 0.268 | Moderate increase |
| RNA:left lobe of liver — MYBPHL (exons) | 4.17 | 5.28 | +0.234 | ≥99th | 1.000 | Moderate increase |
| RNA:left lobe of liver — SORT1 (exons) | 0.346 | 0.39 | +0.120 | ≥99th | 0.855 | Moderate increase |
| RNA:left lobe of liver — SORT1 (exons) | 423 | 446 | +0.054 | ≥99th | 1.000 | Moderate increase |
| RNA:left lobe of liver — SARS1 (exons) | 0.463 | 0.481 | +0.037 | ≥99th | 0.929 | Minimal effect |
| RNA:left lobe of liver — PSMA5 (exons) | 0.171 | 0.176 | +0.032 | ≥99th | 0.606 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:left lobe of liver | 0.00367 | 0.00338 | -0.000 | near-zero | 0.837 | Minimal effect |
| SPLICE_SITES:left lobe of liver | 0.000694 | 0.00078 | +0.000 | near-zero | 0.532 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

