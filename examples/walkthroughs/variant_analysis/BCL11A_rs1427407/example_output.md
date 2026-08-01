## Analysis Request

> Analyze rs1427407 (chr2:60490908 T>G) in K562 erythroid cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks. Gene is BCL11A.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 K562 tracks
- **Generated**: 2026-08-01 04:04 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr2:60490908 T>G
**Oracle**: alphagenome
**Gene**: BCL11A
**Other nearby genes**: PAPOLG, REL, PUS10

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate opening (+0.15, DNASE:K562); Transcription factor binding (ChIP-TF): moderate binding gain (+0.12, CHIP:TAL1:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 8.99 | 10.1 | +0.147 | 0.97 | 0.489 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 466 | 508 | +0.124 | 0.98 | 0.956 | Moderate binding gain |
| CHIP:GATA1:K562 | 450 | 481 | +0.098 | 0.95 | 0.882 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 1.24e+03 | 1.27e+03 | +0.025 | 0.80 | 0.909 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 1.77 | 1.93 | +0.082 | 0.98 | 0.793 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 381 | 389 | +0.030 | 0.92 | 0.959 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 1.99 | 2.02 | +0.018 | 0.87 | 0.805 | Minimal effect |
| CAGE:K562 — variant site | 0.127 | 0.139 | +0.015 | 0.85 | 0.366 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 18.7 | 18.7 | -0.002 | 0.42 | 0.911 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 2.58e+03 | 2.59e+03 | +0.002 | 0.36 | 0.981 | Minimal effect |
| CAGE:K562 — REL TSS | 1.84e+03 | 1.84e+03 | -0.001 | 0.24 | 0.976 | Minimal effect |
| CAGE:K562 — REL TSS | 66.2 | 66.2 | -0.000 | near-zero | 0.939 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
