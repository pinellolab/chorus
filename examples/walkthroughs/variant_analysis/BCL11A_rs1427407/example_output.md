## Analysis Request

> Analyze rs1427407 (chr2:60490908 T>G) in K562 erythroid cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks. Gene is BCL11A.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 K562 tracks
- **Generated**: 2026-08-09 13:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr2:60490908 T>G
**Oracle**: alphagenome
**Gene**: BCL11A
**Other nearby genes**: PAPOLG, REL, PUS10

**Summary**: Transcription factor binding (ChIP-TF): moderate binding gain (+0.15, CHIP:TAL1:K562); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.14, DNASE:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 8.99 | 10 | +0.145 | 0.90 | 0.480 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 360 | 398 | +0.148 | 0.97 | 0.852 | Moderate binding gain |
| CHIP:GATA1:K562 | 356 | 387 | +0.120 | 0.95 | 0.804 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 1.16e+03 | 1.18e+03 | +0.025 | 0.86 | 0.553 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 1.76 | 1.94 | +0.089 | 0.90 | 0.797 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 383 | 389 | +0.025 | 0.71 | 0.962 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 1.99 | 2.03 | +0.018 | 0.64 | 0.804 | Minimal effect |
| CAGE:K562 — variant site | 0.127 | 0.14 | +0.016 | 0.62 | 0.359 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 18.7 | 18.7 | +0.002 | 0.20 | 0.915 | Minimal effect |
| CAGE:K562 — REL TSS | 66.1 | 66.2 | +0.002 | 0.18 | 0.943 | Minimal effect |
| CAGE:K562 — REL TSS | 1.84e+03 | 1.84e+03 | -0.002 | 0.18 | 0.977 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 2.58e+03 | 2.58e+03 | -0.001 | 0.16 | 0.980 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
