## Analysis Request

> Analyze rs1427407 (chr2:60490908 G>T) in K562 erythroid cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks. Gene is BCL11A.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 K562 tracks
- **Generated**: 2026-04-21 13:18 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr2:60490908 G>T
**Oracle**: alphagenome
**Gene**: BCL11A
**Other nearby genes**: PAPOLG, REL, PUS10

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate closing (-0.16, DNASE:K562); Transcription factor binding (ChIP-TF): moderate binding loss (-0.13, CHIP:TAL1:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 10.1 | 9 | -0.156 | ≥99th | 0.536 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 496 | 455 | -0.126 | ≥99th | 0.966 | Moderate binding loss |
| CHIP:GATA1:K562 | 472 | 441 | -0.099 | ≥99th | 0.890 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 1.29e+03 | 1.26e+03 | -0.030 | ≥99th | 0.910 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 1.93 | 1.74 | -0.098 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 419 | 414 | -0.020 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — variant site | 0.142 | 0.13 | -0.016 | ≥99th | 0.628 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 2.3 | 2.27 | -0.014 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — REL TSS | 63.6 | 64 | +0.008 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 18.3 | 18.4 | +0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — REL TSS | 1.74e+03 | 1.74e+03 | +0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 2.45e+03 | 2.45e+03 | -0.001 | near-zero | 1.000 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
