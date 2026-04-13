## Analysis Request

> Analyze the TERT C228T promoter mutation (chr5:1295228 G>A) in K562 cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2/K562 tracks
- **Generated**: 2026-04-12 23:55 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295228 G>A
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: TSS activity (CAGE/PRO-CAP): strong decrease (-0.42); Histone modifications (ChIP-Histone): strong mark loss (-0.32); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.20).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 332 | 289 | -0.201 | 1.000 | 0.914 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 145 | 138 | -0.068 | 1.000 | 0.109 | Minimal effect |
| CHIP:GATA1:K562 | 408 | 406 | -0.004 | 1.000 | 0.861 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 2.65e+03 | 2.11e+03 | -0.324 | 1.000 | 0.952 | Strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 41.1 | 30.5 | -0.419 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — variant site | 600 | 453 | -0.406 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — TERT TSS | 41 | 30.7 | -0.405 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — TERT TSS | 746 | 567 | -0.395 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — LPCAT1 TSS | 3.12e+03 | 3.15e+03 | +0.012 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — ZDHHC11B TSS | 62.2 | 62.6 | +0.010 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — ZDHHC11B TSS | 63.4 | 63.8 | +0.009 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A18 TSS | 0.887 | 0.876 | -0.008 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A3 TSS | 7.61 | 7.66 | +0.007 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — LPCAT1 TSS | 15.5 | 15.6 | +0.006 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 30 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
