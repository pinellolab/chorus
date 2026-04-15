## Analysis Request

> Analyze the TERT C228T promoter mutation (chr5:1295228 G>A) in K562 cells using DNASE, GATA1/TAL1 ChIP, H3K27ac, and CAGE tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2/K562 tracks
- **Generated**: 2026-04-15 05:01 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295228 G>A
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: TSS activity (CAGE/PRO-CAP): strong decrease (-0.43); Histone modifications (ChIP-Histone): strong mark loss (-0.33); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.21).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 332 | 288 | -0.206 | 1.000 | 0.914 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:TAL1:K562 | 145 | 138 | -0.071 | 1.000 | 0.110 | Minimal effect |
| CHIP:GATA1:K562 | 408 | 406 | -0.005 | 1.000 | 0.861 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 2.65e+03 | 2.11e+03 | -0.329 | 1.000 | 0.952 | Strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — variant site | 41.3 | 30.3 | -0.433 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — variant site | 600 | 448 | -0.420 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — TERT TSS | 41.2 | 30.6 | -0.418 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — TERT TSS | 746 | 562 | -0.407 | 1.000 | 1.000 | Strong decrease |
| CAGE:K562 — SLC12A7 TSS | 12.4 | 12.5 | +0.013 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — ZDHHC11B TSS | 62.4 | 62.9 | +0.011 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — NKD2 TSS | 2.68 | 2.7 | +0.011 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC12A7 TSS | 923 | 928 | +0.008 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A18 TSS | 0.885 | 0.877 | -0.006 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — TRIP13 TSS | 4.79e+03 | 4.77e+03 | -0.006 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 30 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
