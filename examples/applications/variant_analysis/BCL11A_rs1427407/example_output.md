## Analysis Request

> Analyze rs1427407 (chr2:60490908 G>T) in K562 erythroid cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks. Gene is BCL11A.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 K562 tracks
- **Generated**: 2026-04-17 19:50 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr2:60490908 G>T
**Oracle**: alphagenome
**Gene**: BCL11A
**Other nearby genes**: PAPOLG, REL, PUS10

**Summary**: Transcription factor binding (ChIP-TF): moderate binding loss (-0.12); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.11).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 9.57 | 8.78 | -0.112 | ≥99th | 0.514 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 478 | 440 | -0.118 | ≥99th | 0.960 | Moderate binding loss |
| CHIP:GATA1:K562 | 447 | 434 | -0.042 | ≥99th | 0.882 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 1.26e+03 | 1.26e+03 | +0.002 | 0.78 | 0.910 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 1.85 | 1.7 | -0.078 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 419 | 422 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — variant site | 0.133 | 0.126 | -0.008 | ≥99th | 0.597 | Minimal effect |
| CAGE:K562 — REL TSS | 68.6 | 68.9 | +0.006 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — REL TSS | 1.87e+03 | 1.88e+03 | +0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 2.59e+03 | 2.59e+03 | -0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 2.16 | 2.16 | -0.001 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 19 | 19 | +0.000 | — | 1.000 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
