## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-08-01 03:24 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Histone modifications (ChIP-Histone): very strong mark gain (+1.48, CHIP:H3K4me3:skeletal muscle cell); TSS activity (CAGE/PRO-CAP): very strong increase (+1.46, CAGE:K562); Gene expression (RNA-seq): very strong increase (+1.34, RNA:HCT116); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.16, ATAC:effector memory CD8-positive, alpha-beta T cell); Transcription factor binding (ChIP-TF): very strong binding gain (+1.06, CHIP:POLR2G:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:effector memory CD8-positive, alpha-beta T cell | 276 | 619 | +1.162 | ≥99th | 0.899 | Very strong opening |
| ATAC:CD8-positive, alpha-beta T cell | 398 | 789 | +0.986 | ≥99th | 0.906 | Very strong opening |
| DNASE:DND-41 | 1.4e+03 | 1.82e+03 | +0.382 | ≥99th | 0.949 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:POLR2G:K562 | 9.46e+03 | 1.97e+04 | +1.058 | ≥99th | 0.932 | Very strong binding gain |
| CHIP:RBFOX2:K562 | 7.47e+03 | 1.48e+04 | +0.991 | ≥99th | 0.929 | Very strong binding gain |
| CHIP:POLR2A:GM15510 | 1.21e+04 | 2.05e+04 | +0.760 | ≥99th | 0.945 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:skeletal muscle cell | 8.54e+03 | 2.37e+04 | +1.476 | ≥99th | 0.896 | Very strong mark gain |
| CHIP:H3K4me3:HFF-Myc | 1.86e+04 | 4.15e+04 | +1.162 | ≥99th | 0.949 | Very strong mark gain |
| CHIP:H3K4me3:Jurkat, Clone E6-1 | 4.24e+04 | 6.78e+04 | +0.678 | ≥99th | 0.995 | Strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — TERT TSS | 768 | 2.12e+03 | +1.461 | ≥99th | 0.966 | Very strong increase |
| CAGE:K562 — variant site | 767 | 2.11e+03 | +1.461 | ≥99th | 0.966 | Very strong increase |
| CAGE:HL-60 — variant site | 1.45e+03 | 2.95e+03 | +1.019 | ≥99th | 0.973 | Very strong increase |
| CAGE:HL-60 — TERT TSS | 1.46e+03 | 2.95e+03 | +1.018 | ≥99th | 0.973 | Very strong increase |
| CAGE:Jurkat — variant site | 3.39e+03 | 5.15e+03 | +0.601 | ≥99th | 0.983 | Strong increase |
| CAGE:Jurkat — TERT TSS | 3.4e+03 | 5.15e+03 | +0.601 | ≥99th | 0.983 | Strong increase |
| CAGE:K562 — SLC6A19 TSS | 2.36 | 2.43 | +0.033 | 0.93 | 0.815 | Minimal effect |
| CAGE:Jurkat — SLC6A18 TSS | 0.332 | 0.351 | +0.021 | 0.86 | 0.370 | Minimal effect |
| CAGE:HL-60 — SLC12A7 TSS | 703 | 696 | -0.015 | 0.81 | 0.966 | Minimal effect |
| CAGE:K562 — SLC12A7 TSS | 793 | 785 | -0.014 | 0.82 | 0.966 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HCT116 — SLC6A18 (exons) | 0.0133 | 0.0535 | +1.336 | ≥99th | 0.184 | Very strong increase |
| RNA:NCI-H460 — SLC6A18 (exons) | 0.0027 | 0.0105 | +1.133 | ≥99th | 0.108 | Very strong increase |
| RNA:PFSK-1 — TERT (exons) | 153 | 394 | +0.942 | ≥99th | 1.000 | Very strong increase |
| RNA:HCT116 — TERT (exons) | 188 | 318 | +0.527 | ≥99th | 1.000 | Strong increase |
| RNA:NCI-H460 — TERT (exons) | 435 | 691 | +0.462 | ≥99th | 1.000 | Strong increase |
| RNA:PFSK-1 — SLC6A18 (exons) | 0.358 | 0.482 | +0.298 | ≥99th | 0.403 | Moderate increase |
| RNA:HCT116 — SLC6A19 (exons) | 0.00099 | 0.00147 | +0.214 | ≥99th | 0.109 | Moderate increase |
| RNA:NCI-H460 — SLC6A19 (exons) | 0.000737 | 0.000981 | +0.131 | ≥99th | 0.079 | Moderate increase |
| RNA:PFSK-1 — SLC6A19 (exons) | 3.66 | 3.81 | +0.041 | ≥99th | 0.994 | Minimal effect |
| RNA:HCT116 — NKD2 (exons) | 0.284 | 0.287 | +0.010 | ≥99th | 0.593 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0383 | 0.0656 | +0.037 | 0.99 | 0.805 | Minimal effect |
| SPLICE_SITES:HFFc6 | 0.0103 | 0.0197 | +0.013 | ≥99th | 0.851 | Minimal effect |
| SPLICE_SITES:dorsolateral prefrontal cortex | 0.0173 | 0.0248 | +0.011 | 0.99 | 0.849 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
