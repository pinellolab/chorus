## Analysis Request

> Screen all cell types to find where rs12740374 (chr1:109274968 G>T) has the strongest regulatory impact. I don't have a specific tissue in mind — let the model tell me where this variant matters most. Gene is SORT1.

- **Tool**: `discover_variant_cell_types`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all DNASE/ATAC tracks (~472 cell types)
- **Generated**: 2026-04-12 02:21 UTC

## Discovery: SORT1 rs12740374 cell-type screen

**Variant**: chr1:109274968 G>T (rs12740374)
**Oracle**: alphagenome
**Gene**: SORT1
**Top cell types**: 3

| Rank | Cell type | Best effect | Best track | N tracks |
|------|-----------|-------------|------------|----------|
| 1 | LNCaP clone FGC | +1.901 | DNASE/EFO:0005726 DNase-seq/. | 3 |
| 2 | epithelial cell of proximal tubule | +1.634 | DNASE/CL:0002306 DNase-seq/. | 9 |
| 3 | renal cortical epithelial cell | +1.490 | DNASE/CL:0002584 DNase-seq/. | 7 |

### LNCaP clone FGC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.90); Histone modifications (ChIP-Histone): very strong mark gain (+1.00); Transcription factor binding (ChIP-TF): moderate binding gain (+0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 59.1 | 223 | +1.901 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:CTCF:LNCaP clone FGC | 400 | 469 | +0.229 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:H3K4me3:LNCaP clone FGC | 1.88e+03 | 3.76e+03 | +0.998 | Very strong mark gain |


### epithelial cell of proximal tubule

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.63); TSS activity (CAGE/PRO-CAP): very strong increase (+0.72); Histone modifications (ChIP-Histone): strong mark gain (+0.58); Transcription factor binding (ChIP-TF): moderate binding gain (+0.18); Gene expression (RNA-seq): moderate increase (+0.15).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| DNASE:epithelial cell of proximal tubule | 80.2 | 251 | +1.634 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:CTCF:epithelial cell of proximal tubule | 450 | 511 | +0.185 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:H3K4me3:epithelial cell of proximal tubule | 4.67e+03 | 7.01e+03 | +0.584 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CAGE:epithelial cell of proximal tubule — variant site | 10.1 | 17.3 | +0.718 | Very strong increase |
| CAGE:epithelial cell of proximal tubule — variant site | 43.6 | 65.8 | +0.582 | Strong increase |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 32.8 | 33.6 | +0.035 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 4.14 | 4.26 | +0.032 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSRC1 TSS | 2.09e+03 | 2.13e+03 | +0.029 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 10.2 | 10.4 | +0.022 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CELSR2 TSS | 1.01e+03 | 1.03e+03 | +0.020 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 6.92 | 7.02 | +0.018 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM1 TSS | 1.36e+03 | 1.35e+03 | -0.011 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 230 | 228 | -0.010 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SARS1 TSS | 41.5 | 41.7 | +0.007 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSMA5 TSS | 57.3 | 57.6 | +0.006 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AKNAD1 TSS | 1.97 | 1.96 | -0.006 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CYB561D1 TSS | 1.06e+03 | 1.06e+03 | -0.005 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — MYBPHL TSS | 0.155 | 0.159 | +0.005 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM3 TSS | 9.61 | 9.64 | +0.005 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CYB561D1 TSS | 10.8 | 10.8 | +0.005 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SYPL2 TSS | 174 | 175 | +0.004 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM5 TSS | 94 | 94.2 | +0.004 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AMPD2 TSS | 34.9 | 35.1 | +0.004 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GPR61 TSS | 5.5 | 5.48 | -0.004 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM4 TSS | 44.5 | 44.6 | +0.004 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM3 TSS | 4.64e+03 | 4.63e+03 | -0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GPSM2 TSS | 19.8 | 19.9 | +0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SORT1 TSS | 3.39e+03 | 3.4e+03 | +0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GPSM2 TSS | 2.06e+03 | 2.07e+03 | +0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM4 TSS | 3.9e+03 | 3.89e+03 | -0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AMPD2 TSS | 2.47e+03 | 2.46e+03 | -0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAI3 TSS | 1.04e+04 | 1.04e+04 | +0.003 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — WDR47 TSS | 3.44 | 3.43 | -0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SARS1 TSS | 1.84e+04 | 1.84e+04 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AKNAD1 TSS | 1.44 | 1.44 | -0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — TAF13 TSS | 28.4 | 28.5 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AMIGO1 TSS | 276 | 276 | -0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAT2 TSS | 31.4 | 31.4 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM1 TSS | 4.73 | 4.73 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — ATXN7L2 TSS | 10.7 | 10.6 | -0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CFAP276 TSS | 25.7 | 25.7 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CLCC1 TSS | 3.3e+03 | 3.3e+03 | -0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — TMEM167B TSS | 201 | 201 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM2 TSS | 2.08e+03 | 2.08e+03 | +0.002 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — EPS8L3 TSS | 0.742 | 0.744 | +0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — WDR47 TSS | 1.98e+03 | 1.98e+03 | +0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — TAF13 TSS | 3.13e+03 | 3.14e+03 | +0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — TMEM167B TSS | 2.93e+03 | 2.93e+03 | -0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM5 TSS | 0.827 | 0.826 | -0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — PSMA5 TSS | 1.15e+04 | 1.15e+04 | -0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CLCC1 TSS | 15.9 | 16 | +0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — ELAPOR1 TSS | 267 | 267 | +0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — SYPL2 TSS | 3.5 | 3.5 | -0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GNAT2 TSS | 1.79e+03 | 1.79e+03 | -0.001 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GSTM2 TSS | 1.86 | 1.86 | -0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — ELAPOR1 TSS | 3.6 | 3.6 | +0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — AMIGO1 TSS | 1.78 | 1.78 | -0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — EPS8L3 TSS | 108 | 108 | -0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — CFAP276 TSS | 77.1 | 77.1 | -0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — GPR61 TSS | 0.0644 | 0.0644 | +0.000 | Minimal effect |
| CAGE:epithelial cell of proximal tubule — ATXN7L2 TSS | 1.04e+03 | 1.04e+03 | +0.000 | Minimal effect |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 0.0236 | 0.0275 | +0.146 | Moderate increase |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 0.108 | 0.119 | +0.096 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.00375 | 0.00423 | +0.096 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CELSR2 (exons) | 180 | 188 | +0.044 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 0.0627 | 0.0648 | +0.033 | Minimal effect |
| RNA:epithelial cell of proximal tubule — MYBPHL (exons) | 0.0675 | 0.0688 | +0.018 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSRC1 (exons) | 392 | 398 | +0.015 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSMA5 (exons) | 0.137 | 0.138 | +0.006 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SARS1 (exons) | 0.21 | 0.211 | +0.006 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SORT1 (exons) | 292 | 294 | +0.006 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CFAP276 (exons) | 0.389 | 0.39 | +0.004 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AKNAD1 (exons) | 0.00869 | 0.00866 | -0.004 | Minimal effect |
| RNA:epithelial cell of proximal tubule — ELAPOR1 (exons) | 5.74 | 5.75 | +0.003 | Minimal effect |
| RNA:epithelial cell of proximal tubule — EPS8L3 (exons) | 0.000765 | 0.00077 | +0.003 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMIGO1 (exons) | 0.316 | 0.315 | -0.003 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CFAP276 (exons) | 0.0983 | 0.0981 | -0.002 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GPR61 (exons) | 4.29 | 4.3 | +0.002 | Minimal effect |
| RNA:epithelial cell of proximal tubule — ELAPOR1 (exons) | 0.0082 | 0.00822 | +0.002 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GNAI3 (exons) | 0.248 | 0.247 | -0.002 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SARS1 (exons) | 1.37e+03 | 1.37e+03 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM4 (exons) | 136 | 135 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM2 (exons) | 13.9 | 13.9 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM2 (exons) | 0.000702 | 0.0007 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AKNAD1 (exons) | 0.209 | 0.209 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SYPL2 (exons) | 0.0382 | 0.0381 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMIGO1 (exons) | 130 | 129 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM1 (exons) | 33.2 | 33.2 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — ATXN7L2 (exons) | 68.3 | 68.2 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — TMEM167B (exons) | 0.579 | 0.579 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM3 (exons) | 0.151 | 0.151 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM4 (exons) | 0.0285 | 0.0285 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CYB561D1 (exons) | 292 | 292 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CYB561D1 (exons) | 0.0911 | 0.091 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM3 (exons) | 275 | 275 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GPSM2 (exons) | 77.1 | 77.1 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CLCC1 (exons) | 291 | 291 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM5 (exons) | 0.000131 | 0.000132 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GPR61 (exons) | 0.252 | 0.252 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — STXBP3 (exons) | 121 | 121 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM5 (exons) | 0.647 | 0.647 | +0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMPD2 (exons) | 318 | 318 | -0.001 | Minimal effect |
| RNA:epithelial cell of proximal tubule — STXBP3 (exons) | 0.042 | 0.0419 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GNAT2 (exons) | 1.95 | 1.95 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GSTM1 (exons) | 0.00204 | 0.00204 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — SYPL2 (exons) | 21.8 | 21.8 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — CLCC1 (exons) | 63.3 | 63.3 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — EPS8L3 (exons) | 0.176 | 0.176 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — WDR47 (exons) | 0.0182 | 0.0182 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — TAF13 (exons) | 0.119 | 0.119 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — WDR47 (exons) | 157 | 157 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — ATXN7L2 (exons) | 0.00908 | 0.00909 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GNAT2 (exons) | 0.282 | 0.282 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GNAI3 (exons) | 1.35e+03 | 1.35e+03 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — AMPD2 (exons) | 0.0614 | 0.0614 | -0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — GPSM2 (exons) | 216 | 216 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — PSMA5 (exons) | 968 | 968 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — TMEM167B (exons) | 875 | 875 | +0.000 | Minimal effect |
| RNA:epithelial cell of proximal tubule — TAF13 (exons) | 713 | 713 | -0.000 | Minimal effect |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00785 | 0.00702 | -0.001 | Minimal effect |
| SPLICE_SITES:epithelial cell of proximal tubule | 0.00116 | 0.00115 | -0.000 | Minimal effect |


### renal cortical epithelial cell

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.49); TSS activity (CAGE/PRO-CAP): strong increase (+0.68); Gene expression (RNA-seq): moderate increase (+0.18).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| DNASE:renal cortical epithelial cell | 223 | 629 | +1.490 | Very strong opening |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CAGE:renal cortical epithelial cell — variant site | 9.1 | 15.1 | +0.676 | Strong increase |
| CAGE:renal cortical epithelial cell — variant site | 42.7 | 62.9 | +0.548 | Strong increase |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 27.1 | 27.7 | +0.033 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSRC1 TSS | 1.85e+03 | 1.89e+03 | +0.027 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 4.09 | 4.18 | +0.027 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 11.2 | 11.3 | +0.018 | Minimal effect |
| CAGE:renal cortical epithelial cell — CELSR2 TSS | 1.08e+03 | 1.09e+03 | +0.017 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 6.71 | 6.8 | +0.017 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM1 TSS | 1.14e+03 | 1.13e+03 | -0.012 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 213 | 212 | -0.010 | Minimal effect |
| CAGE:renal cortical epithelial cell — SARS1 TSS | 33 | 33.1 | +0.006 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSMA5 TSS | 45.3 | 45.5 | +0.006 | Minimal effect |
| CAGE:renal cortical epithelial cell — AKNAD1 TSS | 2.84 | 2.82 | -0.006 | Minimal effect |
| CAGE:renal cortical epithelial cell — CYB561D1 TSS | 1.04e+03 | 1.03e+03 | -0.006 | Minimal effect |
| CAGE:renal cortical epithelial cell — GPR61 TSS | 9.22 | 9.19 | -0.005 | Minimal effect |
| CAGE:renal cortical epithelial cell — MYBPHL TSS | 0.184 | 0.188 | +0.005 | Minimal effect |
| CAGE:renal cortical epithelial cell — SYPL2 TSS | 160 | 161 | +0.005 | Minimal effect |
| CAGE:renal cortical epithelial cell — AMPD2 TSS | 30.4 | 30.5 | +0.005 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM3 TSS | 7.09 | 7.11 | +0.004 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM4 TSS | 28.3 | 28.4 | +0.004 | Minimal effect |
| CAGE:renal cortical epithelial cell — GPSM2 TSS | 2.01e+03 | 2.01e+03 | +0.004 | Minimal effect |
| CAGE:renal cortical epithelial cell — CYB561D1 TSS | 8.46 | 8.48 | +0.004 | Minimal effect |
| CAGE:renal cortical epithelial cell — WDR47 TSS | 1.98e+03 | 1.99e+03 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM5 TSS | 109 | 109 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — SORT1 TSS | 3.65e+03 | 3.66e+03 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — GPSM2 TSS | 14.8 | 14.9 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM4 TSS | 3.32e+03 | 3.32e+03 | -0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — CFAP276 TSS | 27.2 | 27.2 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — AMPD2 TSS | 2.31e+03 | 2.3e+03 | -0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — SARS1 TSS | 1.73e+04 | 1.73e+04 | +0.003 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAT2 TSS | 25.8 | 25.8 | +0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — TAF13 TSS | 24.8 | 24.9 | +0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM3 TSS | 4.22e+03 | 4.21e+03 | -0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — CLCC1 TSS | 3.53e+03 | 3.52e+03 | -0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — WDR47 TSS | 2.97 | 2.97 | -0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — TAF13 TSS | 3.66e+03 | 3.66e+03 | +0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAI3 TSS | 1.04e+04 | 1.04e+04 | +0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM1 TSS | 3.4 | 3.41 | +0.002 | Minimal effect |
| CAGE:renal cortical epithelial cell — EPS8L3 TSS | 0.667 | 0.669 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — ATXN7L2 TSS | 9.56 | 9.55 | -0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — ELAPOR1 TSS | 3.84 | 3.84 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — TMEM167B TSS | 2.91e+03 | 2.91e+03 | -0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — AKNAD1 TSS | 2.37 | 2.37 | -0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — PSMA5 TSS | 9.78e+03 | 9.78e+03 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — AMIGO1 TSS | 1.57 | 1.57 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — TMEM167B TSS | 178 | 178 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — ATXN7L2 TSS | 1e+03 | 1e+03 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM2 TSS | 1.81 | 1.81 | -0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM2 TSS | 2.43e+03 | 2.43e+03 | +0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — GSTM5 TSS | 0.922 | 0.922 | -0.001 | Minimal effect |
| CAGE:renal cortical epithelial cell — ELAPOR1 TSS | 323 | 323 | -0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — SYPL2 TSS | 2.67 | 2.67 | -0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — EPS8L3 TSS | 89.4 | 89.4 | -0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CLCC1 TSS | 12.1 | 12.1 | +0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — CFAP276 TSS | 86.1 | 86.1 | +0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GNAT2 TSS | 1.7e+03 | 1.7e+03 | +0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — AMIGO1 TSS | 303 | 303 | +0.000 | Minimal effect |
| CAGE:renal cortical epithelial cell — GPR61 TSS | 0.119 | 0.119 | +0.000 | Minimal effect |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 0.0659 | 0.0787 | +0.175 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 0.133 | 0.153 | +0.138 | Moderate increase |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.0236 | 0.0267 | +0.120 | Moderate increase |
| RNA:renal cortical epithelial cell — PSRC1 (exons) | 62.1 | 65.9 | +0.059 | Minimal effect |
| RNA:renal cortical epithelial cell — CELSR2 (exons) | 193 | 204 | +0.055 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 0.331 | 0.344 | +0.037 | Minimal effect |
| RNA:renal cortical epithelial cell — MYBPHL (exons) | 0.137 | 0.141 | +0.033 | Minimal effect |
| RNA:renal cortical epithelial cell — PSMA5 (exons) | 0.225 | 0.227 | +0.009 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 0.475 | 0.478 | +0.007 | Minimal effect |
| RNA:renal cortical epithelial cell — SORT1 (exons) | 483 | 486 | +0.007 | Minimal effect |
| RNA:renal cortical epithelial cell — EPS8L3 (exons) | 0.00143 | 0.00144 | +0.005 | Minimal effect |
| RNA:renal cortical epithelial cell — CFAP276 (exons) | 1.03 | 1.04 | +0.004 | Minimal effect |
| RNA:renal cortical epithelial cell — AKNAD1 (exons) | 0.045 | 0.0448 | -0.004 | Minimal effect |
| RNA:renal cortical epithelial cell — AKNAD1 (exons) | 0.173 | 0.172 | -0.003 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM2 (exons) | 11.9 | 11.8 | -0.003 | Minimal effect |
| RNA:renal cortical epithelial cell — SYPL2 (exons) | 0.13 | 0.13 | -0.003 | Minimal effect |
| RNA:renal cortical epithelial cell — AMIGO1 (exons) | 2.25 | 2.25 | -0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM2 (exons) | 0.00232 | 0.00231 | -0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — ELAPOR1 (exons) | 0.0295 | 0.0296 | +0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — SARS1 (exons) | 1.07e+03 | 1.07e+03 | +0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — GPR61 (exons) | 6.56 | 6.58 | +0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — GNAI3 (exons) | 1.45 | 1.44 | -0.002 | Minimal effect |
| RNA:renal cortical epithelial cell — TMEM167B (exons) | 1.15 | 1.15 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM5 (exons) | 0.000544 | 0.000546 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM4 (exons) | 88.8 | 88.7 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — ELAPOR1 (exons) | 11 | 11 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GPSM2 (exons) | 62.3 | 62.4 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM5 (exons) | 0.855 | 0.856 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — WDR47 (exons) | 0.0919 | 0.092 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — CLCC1 (exons) | 301 | 301 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — ATXN7L2 (exons) | 0.012 | 0.012 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM1 (exons) | 22.2 | 22.1 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — CLCC1 (exons) | 50.1 | 50 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — TAF13 (exons) | 0.475 | 0.474 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — SYPL2 (exons) | 28.9 | 28.9 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM3 (exons) | 0.485 | 0.485 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — CYB561D1 (exons) | 0.347 | 0.347 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — GNAT2 (exons) | 7.69 | 7.7 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — AMPD2 (exons) | 0.109 | 0.109 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — STXBP3 (exons) | 129 | 129 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — STXBP3 (exons) | 0.167 | 0.167 | -0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — ATXN7L2 (exons) | 18.8 | 18.8 | +0.001 | Minimal effect |
| RNA:renal cortical epithelial cell — CFAP276 (exons) | 0.109 | 0.109 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — AMPD2 (exons) | 197 | 197 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GPR61 (exons) | 0.81 | 0.809 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — TMEM167B (exons) | 1.17e+03 | 1.17e+03 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — AMIGO1 (exons) | 250 | 250 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM4 (exons) | 0.059 | 0.0591 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — WDR47 (exons) | 213 | 214 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — CYB561D1 (exons) | 307 | 307 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GNAT2 (exons) | 0.544 | 0.544 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — EPS8L3 (exons) | 0.101 | 0.101 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — PSMA5 (exons) | 589 | 589 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — TAF13 (exons) | 814 | 814 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM3 (exons) | 282 | 282 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GSTM1 (exons) | 0.0046 | 0.0046 | -0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GPSM2 (exons) | 153 | 153 | +0.000 | Minimal effect |
| RNA:renal cortical epithelial cell — GNAI3 (exons) | 1.49e+03 | 1.49e+03 | -0.000 | Minimal effect |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| SPLICE_SITES:renal cortical epithelial cell | 0.00634 | 0.00576 | -0.001 | Minimal effect |
| SPLICE_SITES:renal cortical epithelial cell | 0.00101 | 0.00102 | +0.000 | Minimal effect |

---

---

## Interpretation

**What the oracle sees.** Screening all ~472 cell types for rs12740374
returns LNCaP (prostate, +1.9 log2FC), epithelial cell of proximal
tubule (+1.6), and renal cortical epithelial (+1.5) as the top hits for
chromatin accessibility. Notably, liver / HepG2 is NOT in the top 3 even
though the published mechanism is a liver CEBP site.

**How this fits the published biology.** The Musunuru paper localises
the effect to hepatocytes. That LNCaP and kidney cell types out-rank
HepG2 here is a reminder that AlphaGenome's cell-type-level rankings
depend on each cell type's *available* DNASE track quality; a single
strong LNCaP DNASE signal can outrank liver when liver tracks are
sparser or noisier. The variant DOES open chromatin in every top cell
type — the direction is correct, but the absolute ranking should be
taken with a grain of salt.

**Suggested next steps.**
- Use this screen as a *filter*, not an answer. Any variant that reaches
  |log2FC| > 0.3 in a discovery screen is worth a full multi-layer
  analysis in each of the top hit cell types.
- For the specific case of SORT1, the biological answer is known to be
  hepatocyte-driven — treat the LNCaP result as a model quirk and
  prioritise the hepatocyte multi-layer report.
