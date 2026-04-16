## Analysis Request

> Analyze rs1427407 (chr2:60490908 G>T) in K562 erythroid cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks. Gene is BCL11A.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2/K562 tracks
- **Generated**: 2026-04-16 17:53 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr2:60490908 G>T
**Oracle**: alphagenome
**Gene**: BCL11A
**Other nearby genes**: PAPOLG, REL, PUS10

**Summary**: Transcription factor binding (ChIP-TF): moderate binding loss (-0.12); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.11).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 9.56 | 8.77 | -0.113 | ≥99th | 0.514 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 478 | 440 | -0.118 | ≥99th | 0.960 | Moderate binding loss |
| CHIP:GATA1:K562 | 447 | 434 | -0.044 | ≥99th | 0.882 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 1.26e+03 | 1.26e+03 | -0.000 | — | 0.910 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 1.85 | 1.7 | -0.079 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 424 | 421 | -0.007 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — variant site | 0.133 | 0.127 | -0.007 | ≥99th | 0.597 | Minimal effect |
| CAGE:K562 — BCL11A TSS | 2.15 | 2.16 | +0.004 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — REL TSS | 1.88e+03 | 1.87e+03 | -0.003 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — REL TSS | 68.9 | 68.8 | -0.003 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 2.58e+03 | 2.59e+03 | +0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — PAPOLG TSS | 19 | 18.9 | -0.002 | ≥99th | 1.000 | Minimal effect |

## Interpretation

This variant shows **moderate chromatin closing and TF binding loss in K562 erythroid cells**, consistent with its role as a fetal hemoglobin (HbF) regulatory variant. rs1427407 lies in an erythroid enhancer of BCL11A — a repressor of the fetal globin genes. The G>T change reduces TAL1 binding (-0.12) and chromatin accessibility (DNASE -0.11), weakening the enhancer. Reduced BCL11A expression de-represses HBG1/HBG2, increasing fetal hemoglobin. This mechanism is the basis for sickle cell disease therapies targeting the BCL11A enhancer (e.g. Casgevy/exa-cel).

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
