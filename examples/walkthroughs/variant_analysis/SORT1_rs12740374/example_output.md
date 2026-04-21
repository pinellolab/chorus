## Analysis Request

> Analyze rs12740374 (chr1:109274968 G>T) in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Gene is SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-17 19:45 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): strong opening (+0.45); Transcription factor binding (ChIP-TF): strong binding gain (+0.38); TSS activity (CAGE/PRO-CAP): moderate increase (+0.25); Histone modifications (ChIP-Histone): moderate mark gain (+0.18).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 511 | 699 | +0.453 | ≥99th | 0.962 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 2.09e+03 | 2.72e+03 | +0.381 | ≥99th | 0.988 | Strong binding gain |
| CHIP:CEBPB:HepG2 | 1.21e+03 | 1.47e+03 | +0.273 | ≥99th | 0.971 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.37e+04 | 1.55e+04 | +0.180 | ≥99th | 0.999 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 21.9 | 26.3 | +0.254 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 45.5 | 46.4 | +0.025 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — variant site | 76.1 | 77.4 | +0.024 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — PSRC1 TSS | 2.38e+03 | 2.41e+03 | +0.018 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — MYBPHL TSS | 230 | 227 | -0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — SORT1 TSS | 6.95 | 7.02 | +0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — MYBPHL TSS | 1.95 | 1.98 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — EPS8L3 TSS | 38.7 | 39 | +0.010 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — AKNAD1 TSS | 2.06 | 2.08 | +0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — CELSR2 TSS | 2.46 | 2.47 | +0.008 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
