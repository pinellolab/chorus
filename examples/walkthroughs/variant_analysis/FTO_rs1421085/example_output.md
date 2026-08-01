## Analysis Request

> Analyze rs1421085 (chr16:53767042 T>C) in HepG2 cells. Gene is FTO. Using HepG2 as the nearest available metabolic cell type.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 7 HepG2 tracks
- **Generated**: 2026-08-01 04:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr16:53767042 T>C
**Oracle**: alphagenome
**Gene**: FTO
**Other nearby genes**: RPGRIP1L, AKTIP, RBL2, IRX3

**Summary**: TSS activity (CAGE/PRO-CAP): moderate decrease (-0.15, CAGE:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:HepG2 | 82.9 | 80.7 | -0.037 | 0.75 | 0.846 | Minimal effect |
| DNASE:HepG2 | 116 | 114 | -0.017 | 0.62 | 0.904 | Minimal effect |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 1.1e+03 | 1.04e+03 | -0.085 | 0.94 | 0.938 | Minimal effect |
| CHIP:CEBPB:HepG2 | 504 | 502 | -0.006 | 0.20 | 0.833 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 8.97e+03 | 8.8e+03 | -0.027 | 0.81 | 0.993 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 4.25 | 3.73 | -0.150 | ≥99th | 0.864 | Moderate decrease |
| CAGE:HepG2 — variant site | 0.908 | 0.859 | -0.038 | 0.95 | 0.772 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 1.27e+03 | 1.27e+03 | +0.003 | 0.64 | 0.969 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 1.23e+03 | 1.23e+03 | +0.003 | 0.63 | 0.969 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 1.23e+03 | 1.23e+03 | +0.003 | 0.63 | 0.969 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 2.21 | 2.2 | -0.003 | 0.61 | 0.838 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 1.24e+03 | 1.23e+03 | -0.002 | 0.58 | 0.969 | Minimal effect |
| CAGE:HepG2 — RBL2 TSS | 2.77e+03 | 2.77e+03 | -0.002 | 0.48 | 0.982 | Minimal effect |
| CAGE:HepG2 — RBL2 TSS | 6.28 | 6.28 | +0.001 | 0.42 | 0.875 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 75.3 | 75.3 | +0.001 | near-zero | 0.943 | Minimal effect |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
