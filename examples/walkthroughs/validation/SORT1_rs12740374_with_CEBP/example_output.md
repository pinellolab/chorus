## Analysis Request

> Validate AlphaGenome paper finding: rs12740374 (chr1:109274968 G>T) should show C/EBP-family binding gain in HepG2. Score DNASE, ATAC, CEBPA/CEBPB/CEBPG/CEBPD ChIP, H3K27ac, CAGE and RNA-seq on both strands, using forced HepG2 tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 11 HepG2 tracks
- **Generated**: 2026-08-01 17:28 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+3.04, CHIP:CEBPB:HepG2); TSS activity (CAGE/PRO-CAP): very strong increase (+1.51, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.33, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.26, CHIP:H3K27ac:HepG2); Gene expression (RNA-seq): very strong increase (+0.72, RNA:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 661 | 1.67e+03 | +1.332 | ≥99th | 0.973 | Very strong opening |
| ATAC:HepG2 | 452 | 752 | +0.732 | ≥99th | 0.935 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPB:HepG2 | 1.38e+03 | 1.14e+04 | +3.044 | ≥99th | 0.977 | Very strong binding gain |
| CHIP:CEBPA:HepG2 | 2.57e+03 | 1.74e+04 | +2.764 | ≥99th | 0.991 | Very strong binding gain |
| CHIP:CEBPG:HepG2 | 2.27e+03 | 1.09e+04 | +2.269 | ≥99th | 0.992 | Very strong binding gain |
| CHIP:CEBPD:HepG2 | 2.01e+03 | 7.08e+03 | +1.818 | ≥99th | 0.990 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.57e+04 | 3.76e+04 | +1.258 | ≥99th | 0.999 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 25.2 | 73.6 | +1.511 | ≥99th | 0.916 | Very strong increase |
| CAGE:HepG2 — variant site | 75.4 | 175 | +1.200 | ≥99th | 0.943 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.26e+03 | 2.66e+03 | +0.237 | ≥99th | 0.977 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 188 | 220 | +0.222 | ≥99th | 0.951 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.47 | 3 | +0.203 | ≥99th | 0.838 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52 | 58.9 | +0.176 | ≥99th | 0.937 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 664 | 749 | +0.172 | ≥99th | 0.965 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 1.74 | 2.02 | +0.141 | 0.99 | 0.823 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.95 | 8.69 | +0.115 | 0.99 | 0.887 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.54e+03 | 3.66e+03 | +0.047 | 0.96 | 0.984 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HepG2 — CELSR2 (exons) | 0.0209 | 0.0438 | +0.718 | ≥99th | 0.221 | Very strong increase |
| RNA:HepG2 — PSRC1 (exons) | 0.781 | 1.41 | +0.588 | ≥99th | 0.921 | Strong increase |
| RNA:HepG2 — CELSR2 (exons) | 47.5 | 70.8 | +0.398 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — PSRC1 (exons) | 236 | 341 | +0.369 | ≥99th | 1.000 | Strong increase |
| RNA:HepG2 — MYBPHL (exons) | 0.474 | 0.638 | +0.298 | ≥99th | 0.787 | Moderate increase |
| RNA:HepG2 — MYBPHL (exons) | 21.6 | 26.7 | +0.212 | ≥99th | 1.000 | Moderate increase |
| RNA:HepG2 — SORT1 (exons) | 0.42 | 0.479 | +0.132 | ≥99th | 0.754 | Moderate increase |
| RNA:HepG2 — SORT1 (exons) | 473 | 501 | +0.057 | ≥99th | 1.000 | Moderate increase |
| RNA:HepG2 — SARS1 (exons) | 0.647 | 0.675 | +0.043 | ≥99th | 0.874 | Minimal effect |
| RNA:HepG2 — ELAPOR1 (exons) | 0.00557 | 0.00579 | +0.033 | ≥99th | 0.178 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
