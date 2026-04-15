## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.900 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.626 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.479 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Analysis Request

- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Generated**: 2026-04-15 05:11 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.90); Histone modifications (ChIP-Histone): very strong mark gain (+1.00); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 59 | 223 | +1.900 | 1.000 | 0.861 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 400 | 469 | +0.229 | 1.000 | 0.857 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.88e+03 | 3.75e+03 | +1.000 | 1.000 | 0.882 | Very strong mark gain |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### epithelial cell of proximal tubule

## Analysis Request

- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Generated**: 2026-04-15 05:11 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72); Histone modifications (ChIP-Histone): strong mark gain (+0.58); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18); Gene expression (RNA-seq): moderate increase (+0.14).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.5 | 251 | +1.626 | 1.000 | 0.882 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 511 | +0.181 | 1.000 | 0.890 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.68e+03 | 7.01e+03 | +0.583 | 1.000 | 0.884 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.721 | 1.000 | 1.000 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.7 | 65.7 | +0.576 | 1.000 | 1.000 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.8 | 33.7 | +0.039 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.13 | 4.25 | +0.034 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.01e+03 | 1.03e+03 | +0.031 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.09e+03 | 2.13e+03 | +0.028 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.3 | +0.020 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 227 | 229 | +0.018 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.97 | 7.06 | +0.016 | 1.000 | 1.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CFAP276 TSS | 25.8 | 26 | +0.011 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0237 | 0.0275 | +0.144 | 1.000 | 0.192 | Moderate increase |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00372 | 0.00421 | +0.099 | 1.000 | 0.140 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.108 | 0.119 | +0.091 | 1.000 | 0.325 | Moderate increase |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.041 | 1.000 | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0628 | 0.0648 | +0.031 | 1.000 | 0.263 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.0678 | 0.0691 | +0.018 | 1.000 | 0.261 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 392 | 398 | +0.015 | 1.000 | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 292 | 294 | +0.006 | 1.000 | 1.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — EPS8L3 (exons) | 0.176 | 0.176 | -0.004 | -0.993 | 0.433 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SYPL2 (exons) | 0.0381 | 0.0379 | -0.004 | -0.987 | 0.215 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00787 | 0.00702 | -0.001 | 1.000 | 0.877 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00117 | 0.00115 | -0.000 | 1.000 | 0.694 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.


### renal cortical epithelial cell

## Analysis Request

- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Generated**: 2026-04-15 05:11 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.48); TSS activity (CAGE/PRO-CAP): strong increase (+0.68); Gene expression (RNA-seq): moderate increase (+0.17).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 224 | 627 | +1.479 | 1.000 | 0.938 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.11 | 15.2 | +0.677 | 1.000 | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.8 | 62.8 | +0.542 | 1.000 | 1.000 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27.1 | 27.8 | +0.037 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.08 | 4.18 | +0.028 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.027 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.09e+03 | +0.024 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 210 | 213 | +0.018 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.3 | +0.015 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.76 | 6.84 | +0.014 | 1.000 | 1.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CFAP276 TSS | 27.3 | 27.5 | +0.010 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0661 | 0.0787 | +0.171 | 1.000 | 0.257 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.133 | 0.153 | +0.135 | 1.000 | 0.366 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0235 | 0.0266 | +0.120 | 1.000 | 0.189 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.2 | 66.1 | +0.061 | 1.000 | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 194 | 204 | +0.051 | 1.000 | 1.000 | Moderate increase |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.332 | 0.344 | +0.035 | 1.000 | 0.673 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.137 | 0.142 | +0.031 | 1.000 | 0.388 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 487 | +0.007 | 1.000 | 1.000 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 0.474 | 0.477 | +0.005 | 1.000 | 0.804 | Minimal effect |
| RNA:renal cortical epithelial cell — SYPL2 (exons) | 0.129 | 0.129 | -0.005 | -0.998 | 0.376 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00637 | 0.00577 | -0.001 | 1.000 | 0.860 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00102 | 0.00103 | +0.000 | 1.000 | 0.622 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

