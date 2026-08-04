## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-08-04 05:11 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Histone modifications (ChIP-Histone): very strong mark gain (+1.48, CHIP:H3K4me3:skeletal muscle cell); TSS activity (CAGE/PRO-CAP): very strong increase (+1.46, CAGE:K562); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.15, ATAC:effector memory CD8-positive, alpha-beta T cell); Transcription factor binding (ChIP-TF): very strong binding gain (+1.04, CHIP:POLR2G:K562); Gene expression (RNA-seq): very strong increase (+0.94, RNA:PFSK-1).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:effector memory CD8-positive, alpha-beta T cell | 278 | 620 | +1.155 | ≥99th | 0.899 | Very strong opening |
| ATAC:CD8-positive, alpha-beta T cell | 399 | 789 | +0.980 | ≥99th | 0.906 | Very strong opening |
| DNASE:DND-41 | 1.4e+03 | 1.82e+03 | +0.384 | ≥99th | 0.949 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:POLR2G:K562 | 6.78e+03 | 1.4e+04 | +1.042 | ≥99th | 0.922 | Very strong binding gain |
| CHIP:RBFOX2:K562 | 5.09e+03 | 1e+04 | +0.979 | ≥99th | 0.916 | Very strong binding gain |
| CHIP:POLR2A:GM15510 | 9.97e+03 | 1.65e+04 | +0.729 | ≥99th | 0.939 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:skeletal muscle cell | 8.43e+03 | 2.35e+04 | +1.480 | ≥99th | 0.856 | Very strong mark gain |
| CHIP:H3K4me3:HFF-Myc | 1.78e+04 | 3.97e+04 | +1.154 | ≥99th | 0.879 | Very strong mark gain |
| CHIP:H3K4me3:Jurkat, Clone E6-1 | 4.1e+04 | 6.54e+04 | +0.674 | ≥99th | 0.912 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — TERT TSS | 768 | 2.12e+03 | +1.462 | ≥99th | 0.966 | Very strong increase |
| CAGE:K562 — variant site | 767 | 2.11e+03 | +1.462 | ≥99th | 0.966 | Very strong increase |
| CAGE:HL-60 — variant site | 1.46e+03 | 2.95e+03 | +1.016 | ≥99th | 0.974 | Very strong increase |
| CAGE:HL-60 — TERT TSS | 1.46e+03 | 2.96e+03 | +1.016 | ≥99th | 0.974 | Very strong increase |
| CAGE:Jurkat — variant site | 3.41e+03 | 5.15e+03 | +0.597 | ≥99th | 0.983 | Strong increase |
| CAGE:Jurkat — TERT TSS | 3.41e+03 | 5.16e+03 | +0.597 | ≥99th | 0.983 | Strong increase |
| CAGE:K562 — SLC6A19 TSS | 2.36 | 2.43 | +0.029 | 0.76 | 0.816 | Minimal effect |
| CAGE:Jurkat — SLC6A18 TSS | 0.332 | 0.352 | +0.021 | 0.67 | 0.368 | Minimal effect |
| CAGE:Jurkat — ZDHHC11B TSS | 155 | 154 | -0.016 | 0.61 | 0.951 | Minimal effect |
| CAGE:HL-60 — ZDHHC11B TSS | 33.9 | 33.6 | -0.012 | 0.57 | 0.926 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:PFSK-1 — TERT (exons) | 0.609 | 1.56 | +0.943 | ≥99th | 0.608 | Very strong increase |
| RNA:HCT116 — TERT (exons) | 0.745 | 1.26 | +0.528 | ≥99th | 0.835 | Strong increase |
| RNA:NCI-H460 — TERT (exons) | 1.73 | 2.75 | +0.462 | ≥99th | 0.917 | Strong increase |
| RNA:PFSK-1 — SLC6A18 (exons) | 0.00207 | 0.00278 | +0.209 | ≥99th | 0.120 | Moderate increase |
| RNA:HCT116 — SLC6A18 (exons) | 7.69e-05 | 0.000308 | +0.194 | ≥99th | 0.144 | Moderate increase |
| RNA:NCI-H460 — SLC6A18 (exons) | 1.55e-05 | 6.03e-05 | +0.043 | ≥99th | 0.087 | Minimal effect |
| RNA:PFSK-1 — SLC6A19 (exons) | 0.00853 | 0.00889 | +0.038 | ≥99th | 0.196 | Minimal effect |
| RNA:HCT116 — NDUFS6 (exons) | 0.00198 | 0.002 | +0.008 | 0.99 | 0.405 | Minimal effect |
| RNA:HCT116 — NKD2 (exons) | 0.00133 | 0.00135 | +0.007 | 0.98 | 0.372 | Minimal effect |
| RNA:HCT116 — ZDHHC11B (exons) | 0.00499 | 0.00502 | +0.006 | 0.98 | 0.485 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0381 | 0.0652 | +0.037 | 0.96 | 0.806 | Minimal effect |
| SPLICE_SITES:HFFc6 | 0.0102 | 0.0197 | +0.013 | 0.97 | 0.852 | Minimal effect |
| SPLICE_SITES:dorsolateral prefrontal cortex | 0.0172 | 0.0246 | +0.010 | 0.95 | 0.850 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
