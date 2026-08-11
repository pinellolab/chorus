## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-08-11 04:43 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Strongest effect per layer anywhere in the prediction window (not necessarily TERT's own track). Histone modifications (ChIP-Histone): very strong mark gain (+1.48, CHIP:H3K4me3:skeletal muscle cell); TSS activity (CAGE/PRO-CAP): very strong increase (+1.46, CAGE:K562 — TERT TSS); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.15, ATAC:effector memory CD8-positive, alpha-beta T cell); Transcription factor binding (ChIP-TF): very strong binding gain (+1.04, CHIP:POLR2G:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:effector memory CD8-positive, alpha-beta T cell | 278 | 620 | +1.155 | 0.9949 | 0.898 | Very strong opening |
| ATAC:CD8-positive, alpha-beta T cell | 399 | 789 | +0.980 | 0.9936 | 0.906 | Very strong opening |
| DNASE:DND-41 | 1.4e+03 | 1.82e+03 | +0.384 | 0.97 | 0.950 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:POLR2G:K562 | 6.78e+03 | 1.4e+04 | +1.042 | 0.9987 | 0.923 | Very strong binding gain |
| CHIP:POLR2A:GM15510 | 9.97e+03 | 1.65e+04 | +0.729 | 0.9970 | 0.940 | Very strong binding gain |
| CHIP:MAX:K562 | 2.1e+04 | 2.77e+04 | +0.400 | 0.98 | 0.996 | Strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:skeletal muscle cell | 8.43e+03 | 2.35e+04 | +1.480 | 0.9997 | 0.857 | Very strong mark gain |
| CHIP:H3K4me3:HFF-Myc | 1.78e+04 | 3.97e+04 | +1.154 | 0.9996 | 0.878 | Very strong mark gain |
| CHIP:H3K4me3:Jurkat, Clone E6-1 | 4.1e+04 | 6.54e+04 | +0.674 | 0.9973 | 0.913 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — TERT TSS | 768 | 2.12e+03 | +1.462 | 0.9985 | 0.968 | Very strong increase |
| CAGE:K562 — variant site | 767 | 2.11e+03 | +1.462 | 0.9985 | 0.968 | Very strong increase |
| CAGE:HL-60 — variant site | 1.46e+03 | 2.95e+03 | +1.016 | 0.9978 | 0.976 | Very strong increase |
| CAGE:HL-60 — TERT TSS | 1.46e+03 | 2.96e+03 | +1.016 | 0.9978 | 0.976 | Very strong increase |
| CAGE:Jurkat — variant site | 3.41e+03 | 5.15e+03 | +0.597 | 0.9946 | 0.985 | Strong increase |
| CAGE:Jurkat — TERT TSS | 3.41e+03 | 5.16e+03 | +0.597 | 0.9946 | 0.985 | Strong increase |
| CAGE:K562 — SLC6A19 TSS | 2.36 | 2.43 | +0.029 | 0.74 | 0.819 | Minimal effect |
| CAGE:Jurkat — SLC6A18 TSS | 0.332 | 0.352 | +0.021 | 0.64 | 0.358 | Minimal effect |
| CAGE:Jurkat — ZDHHC11B TSS | 155 | 154 | -0.016 | 0.57 | 0.954 | Minimal effect |
| CAGE:HL-60 — ZDHHC11B TSS | 33.9 | 33.6 | -0.012 | 0.52 | 0.930 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0381 | 0.0652 | +0.037 | 0.96 | 0.807 | Minimal effect |
| SPLICE_SITES:HFFc6 | 0.0102 | 0.0197 | +0.013 | 0.97 | 0.859 | Minimal effect |
| SPLICE_SITES:dorsolateral prefrontal cortex | 0.0172 | 0.0246 | +0.010 | 0.95 | 0.857 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
