## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.914 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.625 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.483 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Analysis Request

> Screen all cell types for variant rs12740374 (chr1:109274968 G>T) using AlphaGenome. Find which cell types show the strongest chromatin and regulatory effects. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: top 3 tracks for LNCaP clone FGC
- **Cell types**: LNCaP clone FGC
- **Generated**: 2026-04-18 22:28 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.91, DNASE/EFO:0005726 DNase-seq/. · LNCaP clone FGC); Histone modifications (ChIP-Histone): very strong mark gain (+1.01, CHIP_HISTONE/EFO:0005726 Histone ChIP-seq H3K4me3/. · LNCaP clone FGC); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23, CHIP_TF/EFO:0005726 TF ChIP-seq CTCF/. · LNCaP clone FGC).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 58.8 | 224 | +1.914 | ≥99th | 0.860 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 399 | 470 | +0.234 | ≥99th | 0.856 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.87e+03 | 3.77e+03 | +1.012 | ≥99th | 0.882 | Very strong mark gain |

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
- **Generated**: 2026-04-18 22:29 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.62, DNASE/CL:0002306 DNase-seq/. · epithelial cell of proximal tubule); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72, CAGE/hCAGE CL:0002306/- · epithelial cell of proximal tubule); Histone modifications (ChIP-Histone): strong mark gain (+0.58, CHIP_HISTONE/CL:0002306 Histone ChIP-seq H3K4me3/. · epithelial cell of proximal tubule); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18, CHIP_TF/CL:0002306 TF ChIP-seq CTCF/. · epithelial cell of proximal tubule); Gene expression (RNA-seq): moderate increase (+0.15, RNA_SEQ/CL:0002306 total RNA-seq/- · epithelial cell of proximal tubule).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.6 | 251 | +1.625 | ≥99th | 0.882 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 510 | +0.180 | ≥99th | 0.890 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.68e+03 | 7e+03 | +0.579 | ≥99th | 0.885 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.718 | ≥99th | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.7 | 65.8 | +0.580 | ≥99th | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.14 | 4.25 | +0.031 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.9 | 33.6 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.01e+03 | 1.03e+03 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.09e+03 | 2.13e+03 | +0.027 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.4 | +0.023 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.94 | 7.05 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 3.4e+03 | 3.42e+03 | +0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM2 TSS | 2.07e+03 | 2.06e+03 | -0.009 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0236 | 0.0275 | +0.147 | ≥99th | 0.192 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00378 | 0.00426 | +0.096 | ≥99th | 0.141 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.109 | 0.119 | +0.091 | ≥99th | 0.326 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.043 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0628 | 0.0649 | +0.032 | ≥99th | 0.263 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.068 | 0.0694 | +0.021 | ≥99th | 0.262 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 393 | 399 | +0.014 | ≥99th | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — ELAPOR1 (exons) | 0.00814 | 0.0082 | +0.007 | ≥99th | 0.158 | Minimal effect |
| RNA:epithelial cell of proximal tubule — EPS8L3 (exons) | 0.177 | 0.176 | -0.007 | ≤1st | 0.434 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 292 | 294 | +0.006 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00787 | 0.00701 | -0.001 | ≥99th | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00117 | 0.00114 | -0.000 | near-zero | 0.694 | Minimal effect |

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
- **Generated**: 2026-04-18 22:29 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.48, DNASE/CL:0002584 DNase-seq/. · renal cortical epithelial cell); TSS activity (CAGE/PRO-CAP): strong increase (+0.68, CAGE/hCAGE CL:0002584/- · renal cortical epithelial cell); Gene expression (RNA-seq): moderate increase (+0.17, RNA_SEQ/CL:0002584 total RNA-seq/- · renal cortical epithelial cell).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 224 | 628 | +1.483 | ≥99th | 0.938 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.1 | 15.1 | +0.675 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.8 | 62.9 | +0.544 | ≥99th | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.027 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27.2 | 27.7 | +0.027 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.09 | 4.18 | +0.025 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.1e+03 | +0.025 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.73 | 6.82 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.4 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — EPS8L3 TSS | 90.3 | 89.8 | -0.008 | ≥99th | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM2 TSS | 2.43e+03 | 2.42e+03 | -0.008 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0659 | 0.0785 | +0.173 | ≥99th | 0.257 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.134 | 0.153 | +0.134 | ≥99th | 0.367 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0238 | 0.0269 | +0.118 | ≥99th | 0.189 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.2 | 65.9 | +0.059 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 193 | 204 | +0.053 | ≥99th | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.332 | 0.344 | +0.036 | ≥99th | 0.673 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.138 | 0.143 | +0.035 | ≥99th | 0.389 | Minimal effect |
| RNA:renal cortical epithelial cell — ELAPOR1 (exons) | 0.0293 | 0.0295 | +0.008 | ≥99th | 0.195 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 486 | +0.007 | ≥99th | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 0.473 | 0.476 | +0.006 | ≥99th | 0.803 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00637 | 0.00576 | -0.001 | near-zero | 0.860 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00102 | 0.00102 | +0.000 | near-zero | 0.622 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

