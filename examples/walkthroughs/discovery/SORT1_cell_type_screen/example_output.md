## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | HepG2 | +1.331 | DNASE:HepG2 | 562 |
| 2 | MCF 10A | +1.437 | DNASE:MCF 10A | 6 |
| 3 | amniotic epithelial cell | +2.898 | DNASE:amniotic epithelial cell | 3 |

### HepG2

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 30 tracks for HepG2
- **Cell types**: HepG2
- **Generated**: 2026-08-01 03:23 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: TSS activity (CAGE/PRO-CAP): very strong increase (+1.50, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.33, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.26, CHIP:H3K27ac:HepG2); Transcription factor binding (ChIP-TF): very strong binding gain (+0.96, CHIP:ARID3A:HepG2); Gene expression (RNA-seq): very strong increase (+0.85, RNA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 662 | 1.67e+03 | +1.331 | ≥99th | 0.973 | Very strong opening |
| ATAC:HepG2 | 453 | 752 | +0.731 | ≥99th | 0.935 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:ARID3A:HepG2 | 2.56e+03 | 4.97e+03 | +0.957 | ≥99th | 0.999 | Very strong binding gain |
| CHIP:ARID3A:HepG2 | 1.96e+03 | 3.71e+03 | +0.921 | ≥99th | 0.999 | Very strong binding gain |
| CHIP:ARID4B:HepG2 | 4.05e+03 | 6.86e+03 | +0.760 | ≥99th | 0.956 | Very strong binding gain |
| CHIP:AHDC1:HepG2 | 1.25e+03 | 2.04e+03 | +0.712 | ≥99th | 0.999 | Very strong binding gain |
| CHIP:AFF4:HepG2 | 1.31e+03 | 2.03e+03 | +0.634 | ≥99th | 0.979 | Strong binding gain |
| CHIP:ARID4A:HepG2 | 1.6e+03 | 2.46e+03 | +0.615 | ≥99th | 0.942 | Strong binding gain |
| CHIP:AHR:HepG2 | 1.97e+03 | 2.72e+03 | +0.464 | ≥99th | 0.953 | Strong binding gain |
| CHIP:ARID2:HepG2 | 807 | 1.1e+03 | +0.446 | ≥99th | 0.989 | Strong binding gain |
| CHIP:AKAP8:HepG2 | 1.38e+03 | 1.88e+03 | +0.442 | ≥99th | 0.975 | Strong binding gain |
| CHIP:ARHGAP35:HepG2 | 1.26e+03 | 1.56e+03 | +0.315 | ≥99th | 0.954 | Strong binding gain |
| _…showing top 10 of 11 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.57e+04 | 3.76e+04 | +1.257 | ≥99th | 0.999 | Very strong mark gain |
| CHIP:H3K9ac:HepG2 | 1.56e+04 | 2.61e+04 | +0.742 | ≥99th | 0.997 | Very strong mark gain |
| CHIP:H3K4me3:HepG2 | 8.5e+03 | 1.2e+04 | +0.493 | ≥99th | 0.969 | Strong mark gain |
| CHIP:H3K27me3:HepG2 | 1.29e+03 | 930 | -0.468 | ≥99th | 0.993 | Strong mark loss |
| CHIP:H4K20me1:HepG2 | 1.15e+03 | 890 | -0.373 | ≥99th | 0.991 | Strong mark loss |
| CHIP:H3K4me2:HepG2 | 2.67e+04 | 3.41e+04 | +0.353 | 0.98 | 1.000 | Strong mark gain |
| CHIP:H3K36me3:HepG2 | 1.24e+03 | 984 | -0.335 | ≥99th | 0.981 | Strong mark loss |
| CHIP:H3K9me3:HepG2 | 543 | 465 | -0.225 | 0.99 | 0.868 | Moderate mark loss |
| CHIP:H3K79me2:HepG2 | 1.18e+03 | 1.35e+03 | +0.198 | 0.99 | 0.949 | Moderate mark gain |
| CHIP:H3K4me1:HepG2 | 1.67e+04 | 1.78e+04 | +0.089 | 0.88 | 1.000 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 25.3 | 73.6 | +1.502 | ≥99th | 0.916 | Very strong increase |
| CAGE:HepG2 — variant site | 75.4 | 175 | +1.200 | ≥99th | 0.943 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.25e+03 | 2.66e+03 | +0.239 | ≥99th | 0.977 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 189 | 220 | +0.218 | ≥99th | 0.951 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.47 | 3 | +0.202 | ≥99th | 0.838 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 663 | 749 | +0.176 | ≥99th | 0.965 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52.3 | 58.9 | +0.169 | ≥99th | 0.937 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 1.74 | 2.02 | +0.138 | 0.99 | 0.823 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.97 | 8.69 | +0.112 | 0.99 | 0.887 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.54e+03 | 3.66e+03 | +0.048 | 0.96 | 0.984 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HepG2 — CELSR2 (exons) | 0.0331 | 0.0793 | +0.855 | ≥99th | 0.234 | Very strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.0209 | 0.0438 | +0.717 | ≥99th | 0.221 | Very strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.781 | 1.41 | +0.588 | ≥99th | 0.921 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.532 | 0.887 | +0.511 | ≥99th | 0.765 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 59.5 | 93.4 | +0.450 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 204 | 317 | +0.440 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 47.6 | 70.8 | +0.397 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 235 | 341 | +0.370 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 33.4 | 48 | +0.363 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — MYBPHL (exons) | 0.132 | 0.179 | +0.302 | ≥99th | 0.355 | Strong increase |
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
- **Generated**: 2026-08-01 03:23 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.44, DNASE:MCF 10A); Gene expression (RNA-seq): moderate increase (+0.26, RNA:MCF 10A).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:MCF 10A | 338 | 917 | +1.437 | ≥99th | 0.932 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:MCF 10A | 426 | 420 | -0.019 | 0.87 | 0.905 | Minimal effect |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:MCF 10A — CELSR2 (exons) | 0.144 | 0.186 | +0.257 | ≥99th | 0.417 | Moderate increase |
| RNA:MCF 10A — PSRC1 (exons) | 0.883 | 1.07 | +0.194 | ≥99th | 0.950 | Moderate increase |
| RNA:MCF 10A — MYBPHL (exons) | 0.0942 | 0.11 | +0.151 | ≥99th | 0.323 | Moderate increase |
| RNA:MCF 10A — PSRC1 (exons) | 101 | 114 | +0.121 | ≥99th | 1.000 | Moderate increase |
| RNA:MCF 10A — SORT1 (exons) | 0.154 | 0.165 | +0.071 | ≥99th | 0.413 | Moderate increase |
| RNA:MCF 10A — CELSR2 (exons) | 170 | 178 | +0.044 | ≥99th | 1.000 | Minimal effect |
| RNA:MCF 10A — MYBPHL (exons) | 0.064 | 0.0666 | +0.040 | ≥99th | 0.277 | Minimal effect |
| RNA:MCF 10A — SORT1 (exons) | 185 | 188 | +0.018 | ≥99th | 1.000 | Minimal effect |
| RNA:MCF 10A — ELAPOR1 (exons) | 0.022 | 0.0222 | +0.010 | ≥99th | 0.202 | Minimal effect |
| RNA:MCF 10A — PSMA5 (exons) | 0.189 | 0.191 | +0.008 | ≥99th | 0.469 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:MCF 10A | 0.00736 | 0.006 | -0.002 | 0.99 | 0.877 | Minimal effect |
| SPLICE_SITES:MCF 10A | 0.00128 | 0.00126 | -0.000 | near-zero | 0.704 | Minimal effect |

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
- **Generated**: 2026-08-01 03:23 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+2.90, DNASE:amniotic epithelial cell); TSS activity (CAGE/PRO-CAP): strong decrease (-0.61, CAGE:amniotic epithelial cell).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:amniotic epithelial cell | 53 | 402 | +2.898 | ≥99th | 0.844 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:amniotic epithelial cell — variant site | 83.6 | 54.6 | -0.606 | ≥99th | 0.947 | Strong decrease |
| CAGE:amniotic epithelial cell — variant site | 21.9 | 17.5 | -0.302 | ≥99th | 0.915 | Strong decrease |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 1.01e+03 | 982 | -0.038 | 0.96 | 0.970 | Minimal effect |
| CAGE:amniotic epithelial cell — PSRC1 TSS | 1.77e+03 | 1.8e+03 | +0.021 | 0.92 | 0.977 | Minimal effect |
| CAGE:amniotic epithelial cell — SORT1 TSS | 5.85 | 5.92 | +0.015 | 0.88 | 0.879 | Minimal effect |
| CAGE:amniotic epithelial cell — GSTM5 TSS | 251 | 254 | +0.014 | 0.87 | 0.957 | Minimal effect |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 4.26 | 4.21 | -0.014 | 0.86 | 0.866 | Minimal effect |
| CAGE:amniotic epithelial cell — GSTM2 TSS | 3.09e+03 | 3.06e+03 | -0.013 | 0.86 | 0.984 | Minimal effect |
| CAGE:amniotic epithelial cell — WDR47 TSS | 1.98e+03 | 1.96e+03 | -0.010 | 0.80 | 0.978 | Minimal effect |
| CAGE:amniotic epithelial cell — ATXN7L2 TSS | 9.99 | 9.92 | -0.009 | 0.78 | 0.892 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

