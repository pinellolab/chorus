## Analysis Request

> Analyze rs1421085 (chr16:53767042 T>C) in HepG2 cells. Gene is FTO. Using HepG2 as the nearest available metabolic cell type.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-21 13:23 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr16:53767042 T>C
**Oracle**: alphagenome
**Gene**: FTO
**Other nearby genes**: RPGRIP1L, AKTIP, RBL2, IRX3

**Summary**: TSS activity (CAGE/PRO-CAP): moderate decrease (-0.12, CAGE:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 102 | 102 | -0.002 | 0.38 | 0.899 | Minimal effect |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 1.03e+03 | 979 | -0.073 | ≥99th | 0.931 | Minimal effect |
| CHIP:CEBPB:HepG2 | 488 | 485 | -0.007 | ≥99th | 0.823 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 7.85e+03 | 7.72e+03 | -0.024 | ≥99th | 0.989 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 4.01 | 3.61 | -0.121 | ≥99th | 1.000 | Moderate decrease |
| CAGE:HepG2 — variant site | 0.749 | 0.719 | -0.025 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 1.21e+03 | 1.21e+03 | +0.008 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — RBL2 TSS | 2.87e+03 | 2.86e+03 | -0.005 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — RBL2 TSS | 6.49 | 6.47 | -0.004 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 80.7 | 80.5 | -0.003 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 1.28e+03 | 1.28e+03 | +0.003 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 1.28e+03 | 1.28e+03 | +0.003 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 2.26 | 2.25 | -0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 1.29e+03 | 1.29e+03 | -0.001 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
