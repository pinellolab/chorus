## Analysis Request

> Validate AlphaGenome paper finding: rs12740374 (chr1:109274968 G>T) should show CEBPA/CEBPB binding gain in HepG2. Using forced HepG2 tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 7 HepG2 tracks
- **Generated**: 2026-08-01 04:10 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+3.05, CHIP:CEBPB:HepG2); TSS activity (CAGE/PRO-CAP): very strong increase (+1.51, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.33, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.26, CHIP:H3K27ac:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 661 | 1.67e+03 | +1.333 | ≥99th | 0.973 | Very strong opening |
| ATAC:HepG2 | 453 | 752 | +0.730 | ≥99th | 0.935 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPB:HepG2 | 1.38e+03 | 1.14e+04 | +3.046 | ≥99th | 0.977 | Very strong binding gain |
| CHIP:CEBPA:HepG2 | 2.57e+03 | 1.75e+04 | +2.764 | ≥99th | 0.991 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.57e+04 | 3.76e+04 | +1.257 | ≥99th | 0.999 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 25.2 | 73.5 | +1.507 | ≥99th | 0.916 | Very strong increase |
| CAGE:HepG2 — variant site | 75.2 | 174 | +1.202 | ≥99th | 0.943 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.26e+03 | 2.67e+03 | +0.238 | ≥99th | 0.977 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 188 | 219 | +0.218 | ≥99th | 0.951 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.47 | 2.99 | +0.200 | ≥99th | 0.838 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 662 | 753 | +0.187 | ≥99th | 0.965 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52 | 58.9 | +0.177 | ≥99th | 0.937 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 1.74 | 2.01 | +0.135 | 0.99 | 0.823 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.99 | 8.67 | +0.105 | 0.99 | 0.887 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.54e+03 | 3.66e+03 | +0.047 | 0.96 | 0.984 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
