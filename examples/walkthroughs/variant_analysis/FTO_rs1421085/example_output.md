## Analysis Request

> Analyze rs1421085 (chr16:53767042 T>C) in HepG2 cells. Gene is FTO. Using HepG2 as the nearest available metabolic cell type.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 7 HepG2 tracks
- **Generated**: 2026-08-12 03:58 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr16:53767042 T>C
**Oracle**: alphagenome
**Gene**: FTO
**Other nearby genes**: RPGRIP1L, AKTIP, RBL2, IRX3

**Summary**: Strongest effect per layer anywhere in the prediction window (not necessarily FTO's own track). TSS activity (CAGE/PRO-CAP): moderate decrease (-0.15, CAGE:HepG2 — variant site); Transcription factor binding (ChIP-TF): moderate binding loss (-0.12, CHIP:CEBPA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:HepG2 | 82.7 | 80.8 | -0.035 | 0.62 | 0.844 | Minimal effect |
| DNASE:HepG2 | 116 | 115 | -0.014 | 0.42 | 0.903 | Minimal effect |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 639 | 589 | -0.117 | 0.91 | 0.877 | Moderate binding loss |
| CHIP:CEBPB:HepG2 | 293 | 291 | -0.010 | 0.34 | 0.614 | Minimal effect |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 8.25e+03 | 8.09e+03 | -0.028 | 0.85 | 0.907 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 4.25 | 3.73 | -0.152 | 0.95 | 0.864 | Moderate decrease |
| CAGE:HepG2 — variant site | 0.908 | 0.859 | -0.037 | 0.79 | 0.770 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 1.23e+03 | 1.23e+03 | +0.006 | 0.46 | 0.970 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 75.2 | 75.4 | +0.004 | 0.38 | 0.942 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 1.23e+03 | 1.23e+03 | -0.004 | 0.37 | 0.970 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 1.23e+03 | 1.23e+03 | -0.004 | 0.37 | 0.970 | Minimal effect |
| CAGE:HepG2 — AKTIP TSS | 2.21 | 2.2 | -0.004 | 0.36 | 0.833 | Minimal effect |
| CAGE:HepG2 — IRX3 TSS | 1.27e+03 | 1.28e+03 | +0.003 | 0.34 | 0.971 | Minimal effect |
| CAGE:HepG2 — FTO TSS | 4.58e+03 | 4.58e+03 | +0.003 | 0.31 | 0.991 | Minimal effect |
| CAGE:HepG2 — RPGRIP1L TSS | 4.5e+03 | 4.5e+03 | +0.003 | 0.31 | 0.990 | Minimal effect |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
