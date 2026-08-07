## Analysis Request

> Validate AlphaGenome paper finding: rs12740374 (chr1:109274968 G>T) should show C/EBP-family binding gain in HepG2. Score DNASE, ATAC, CEBPA/CEBPB/CEBPG/CEBPD ChIP, H3K27ac, CAGE and RNA-seq on both strands, using forced HepG2 tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 11 HepG2 tracks
- **Generated**: 2026-08-07 12:05 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+3.32, CHIP:CEBPB:HepG2); TSS activity (CAGE/PRO-CAP): very strong increase (+1.50, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.33, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.25, CHIP:H3K27ac:HepG2); Gene expression (RNA-seq): strong increase (+0.47, RNA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 660 | 1.67e+03 | +1.334 | ≥99th | 0.973 | Very strong opening |
| ATAC:HepG2 | 452 | 752 | +0.732 | ≥99th | 0.935 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPB:HepG2 | 1.08e+03 | 1.07e+04 | +3.316 | ≥99th | 0.964 | Very strong binding gain |
| CHIP:CEBPA:HepG2 | 2.07e+03 | 1.6e+04 | +2.945 | ≥99th | 0.986 | Very strong binding gain |
| CHIP:CEBPG:HepG2 | 1.78e+03 | 9.8e+03 | +2.460 | ≥99th | 0.982 | Very strong binding gain |
| CHIP:CEBPD:HepG2 | 1.46e+03 | 5.99e+03 | +2.033 | ≥99th | 0.957 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.51e+04 | 3.58e+04 | +1.251 | ≥99th | 0.946 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 25.2 | 73.3 | +1.502 | ≥99th | 0.919 | Very strong increase |
| CAGE:HepG2 — variant site | 75.2 | 174 | +1.203 | ≥99th | 0.942 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.26e+03 | 2.66e+03 | +0.235 | 0.97 | 0.980 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 188 | 218 | +0.213 | 0.97 | 0.954 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.47 | 2.98 | +0.198 | 0.96 | 0.841 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52 | 58.8 | +0.172 | 0.96 | 0.936 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 666 | 751 | +0.172 | 0.96 | 0.965 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 1.74 | 2 | +0.132 | 0.94 | 0.819 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.97 | 8.72 | +0.115 | 0.93 | 0.886 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.53e+03 | 3.67e+03 | +0.058 | 0.85 | 0.986 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HepG2 — PSRC1 (exons) | 0.00283 | 0.00509 | +0.465 | ≥99th | 0.461 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 0.147 | 0.219 | +0.396 | ≥99th | 0.703 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.853 | 1.24 | +0.370 | ≥99th | 0.860 | Strong increase |
| RNA:HepG2 — MYBPHL (exons) | 0.00319 | 0.00433 | +0.239 | ≥99th | 0.470 | Moderate increase |
| RNA:HepG2 — MYBPHL (exons) | 0.146 | 0.181 | +0.214 | ≥99th | 0.714 | Moderate increase |
| RNA:HepG2 — SORT1 (exons) | 0.00121 | 0.00138 | +0.077 | ≥99th | 0.389 | Moderate increase |
| RNA:HepG2 — CELSR2 (exons) | 6.46e-05 | 0.000136 | +0.065 | ≥99th | 0.155 | Moderate increase |
| RNA:HepG2 — SORT1 (exons) | 1.36 | 1.44 | +0.060 | ≥99th | 0.904 | Moderate increase |
| RNA:HepG2 — SARS1 (exons) | 0.00396 | 0.00412 | +0.032 | ≥99th | 0.497 | Minimal effect |
| RNA:HepG2 — ELAPOR1 (exons) | 0.0389 | 0.0395 | +0.017 | ≥99th | 0.633 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
