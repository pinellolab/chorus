## Analysis Request

> Analyze rs1421085 (chr16:53767042 T>C) in HepG2 cells. Gene is FTO. Using HepG2 as the nearest available metabolic cell type.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-17 06:41 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr16:53767042 T>C
**Oracle**: alphagenome
**Gene**: FTO
**Other nearby genes**: RPGRIP1L, AKTIP, RBL2, IRX3

**Summary**: No strong regulatory effects detected across any layer.

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 113 | 110 | -0.030 | ≥99th | 0.903 | Minimal effect |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 999 | 984 | -0.022 | ≥99th | 0.926 | Minimal effect |
| CHIP:CEBPB:HepG2 | 488 | 488 | +0.001 | 0.21 | 0.823 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 9.2e+03 | 8.94e+03 | -0.042 | ≥99th | 0.993 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 4 | 3.89 | -0.033 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — variant site | 0.846 | 0.82 | -0.021 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 1.27e+03 | 1.27e+03 | -0.007 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 1.36e+03 | 1.36e+03 | -0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 1.36e+03 | 1.36e+03 | -0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — RBL2 TSS | 2.76e+03 | 2.75e+03 | -0.002 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 78.8 | 78.7 | -0.001 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 4.73e+03 | 4.72e+03 | -0.001 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 1.21e+03 | 1.21e+03 | -0.001 | — | 1.000 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 4.63e+03 | 4.63e+03 | -0.001 | — | 1.000 | Minimal effect |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
