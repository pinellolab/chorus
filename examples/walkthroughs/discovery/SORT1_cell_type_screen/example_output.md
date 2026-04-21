## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | fibroblast of mammary gland | +3.297 | DNASE:fibroblast of mammary gland | 5 |
| 2 | amniotic epithelial cell | +2.960 | DNASE:amniotic epithelial cell | 3 |
| 3 | fibroblast of pulmonary artery | +2.905 | DNASE:fibroblast of pulmonary artery | 5 |

### fibroblast of mammary gland

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 5 tracks for fibroblast of mammary gland
- **Cell types**: fibroblast of mammary gland
- **Generated**: 2026-04-21 13:47 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+3.30, DNASE:fibroblast of mammary gland); Histone modifications (ChIP-Histone): very strong mark gain (+0.79, CHIP:H3K4me3:fibroblast of mammary gland); Transcription factor binding (ChIP-TF): strong binding gain (+0.46, CHIP:CTCF:fibroblast of mammary gland); TSS activity (CAGE/PRO-CAP): strong increase (+0.45, CAGE:fibroblast of mammary gland).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of mammary gland | 23.6 | 241 | +3.297 | ≥99th | 0.800 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:fibroblast of mammary gland | 281 | 388 | +0.461 | ≥99th | 0.815 | Strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:fibroblast of mammary gland | 1.32e+03 | 2.27e+03 | +0.786 | ≥99th | 0.866 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:fibroblast of mammary gland — variant site | 1.41 | 2.29 | +0.449 | ≥99th | 1.000 | Strong increase |
| CAGE:fibroblast of mammary gland — variant site | 6.87 | 9.54 | +0.421 | ≥99th | 1.000 | Strong increase |
| CAGE:fibroblast of mammary gland — PSRC1 TSS | 1.72e+03 | 1.8e+03 | +0.066 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — CELSR2 TSS | 373 | 382 | +0.035 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — PSRC1 TSS | 40 | 40.6 | +0.024 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — CELSR2 TSS | 2.15 | 2.19 | +0.018 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — GNAI3 TSS | 198 | 196 | -0.014 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — SORT1 TSS | 4.1 | 4.15 | +0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — MYBPHL TSS | 5.17 | 5.22 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of mammary gland — GSTM5 TSS | 288 | 285 | -0.012 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

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
- **Generated**: 2026-04-21 13:48 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+2.96, DNASE:amniotic epithelial cell); TSS activity (CAGE/PRO-CAP): strong decrease (-0.59, CAGE:amniotic epithelial cell).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:amniotic epithelial cell | 47.6 | 377 | +2.960 | ≥99th | 0.836 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:amniotic epithelial cell — variant site | 78.2 | 51.6 | -0.590 | ≥99th | 1.000 | Strong decrease |
| CAGE:amniotic epithelial cell — variant site | 20 | 16.4 | -0.269 | ≥99th | 1.000 | Moderate decrease |
| CAGE:amniotic epithelial cell — PSRC1 TSS | 1.88e+03 | 1.92e+03 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — GNAI3 TSS | 232 | 230 | -0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 4.02 | 3.97 | -0.015 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — SORT1 TSS | 5.59 | 5.64 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — MYBPHL TSS | 14.1 | 14 | -0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — CELSR2 TSS | 881 | 874 | -0.010 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — AMPD2 TSS | 36 | 36.2 | +0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:amniotic epithelial cell — GSTM5 TSS | 274 | 272 | -0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### fibroblast of pulmonary artery

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 5 tracks for fibroblast of pulmonary artery
- **Cell types**: fibroblast of pulmonary artery
- **Generated**: 2026-04-21 13:48 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+2.90, DNASE:fibroblast of pulmonary artery); Histone modifications (ChIP-Histone): very strong mark gain (+0.78, CHIP:H3K4me3:fibroblast of pulmonary artery); Transcription factor binding (ChIP-TF): strong binding gain (+0.32, CHIP:CTCF:fibroblast of pulmonary artery); TSS activity (CAGE/PRO-CAP): moderate increase (+0.28, CAGE:fibroblast of pulmonary artery).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of pulmonary artery | 20.6 | 161 | +2.905 | ≥99th | 0.776 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:fibroblast of pulmonary artery | 286 | 356 | +0.319 | ≥99th | 0.816 | Strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:fibroblast of pulmonary artery | 1.03e+03 | 1.76e+03 | +0.776 | ≥99th | 0.860 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:fibroblast of pulmonary artery — variant site | 1.66 | 2.24 | +0.282 | ≥99th | 1.000 | Moderate increase |
| CAGE:fibroblast of pulmonary artery — variant site | 4.69 | 5.81 | +0.260 | ≥99th | 1.000 | Moderate increase |
| CAGE:fibroblast of pulmonary artery — PSRC1 TSS | 1.79e+03 | 1.83e+03 | +0.032 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — GNAI3 TSS | 261 | 258 | -0.015 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — GSTM5 TSS | 153 | 151 | -0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — AMPD2 TSS | 29.2 | 29.4 | +0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — PSRC1 TSS | 26.1 | 26.3 | +0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — GSTM1 TSS | 943 | 946 | +0.005 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — CYB561D1 TSS | 717 | 719 | +0.005 | ≥99th | 1.000 | Minimal effect |
| CAGE:fibroblast of pulmonary artery — SARS1 TSS | 44.8 | 45 | +0.005 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

