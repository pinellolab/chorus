## Analysis Request

> Analyze rs12740374 (chr1:109274968 G>T) in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Gene is SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-21 04:48 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): strong opening (+0.45, DNASE:HepG2); Transcription factor binding (ChIP-TF): strong binding gain (+0.37, CHIP:CEBPA:HepG2); TSS activity (CAGE/PRO-CAP): moderate increase (+0.25, CAGE:HepG2); Histone modifications (ChIP-Histone): moderate mark gain (+0.18, CHIP:H3K27ac:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 512 | 699 | +0.448 | ≥99th | 0.962 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 2.1e+03 | 2.73e+03 | +0.375 | ≥99th | 0.989 | Strong binding gain |
| CHIP:CEBPB:HepG2 | 1.22e+03 | 1.46e+03 | +0.267 | ≥99th | 0.971 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.37e+04 | 1.55e+04 | +0.179 | ≥99th | 0.999 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 22 | 26.4 | +0.252 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 45.4 | 46.3 | +0.028 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — MYBPHL TSS | 1.96 | 1.99 | +0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — variant site | 76.4 | 77.2 | +0.015 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — CELSR2 TSS | 2.46 | 2.49 | +0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — WDR47 TSS | 1.63e+03 | 1.62e+03 | -0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — SORT1 TSS | 6.99 | 7.05 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — CFAP276 TSS | 12.8 | 12.9 | +0.011 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — GSTM2 TSS | 931 | 925 | -0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — GSTM1 TSS | 238 | 236 | -0.009 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
