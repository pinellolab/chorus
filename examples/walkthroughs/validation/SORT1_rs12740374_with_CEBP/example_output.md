## Analysis Request

> Validate AlphaGenome paper finding: rs12740374 (chr1:109274968 G>T) should show CEBPA/CEBPB binding gain in HepG2. Using forced HepG2 tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-21 13:27 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+3.02, CHIP:CEBPB:HepG2); TSS activity (CAGE/PRO-CAP): very strong increase (+1.56, CAGE:HepG2); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.40, DNASE:HepG2); Histone modifications (ChIP-Histone): very strong mark gain (+1.33, CHIP:H3K27ac:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 595 | 1.57e+03 | +1.400 | ≥99th | 0.968 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPB:HepG2 | 1.4e+03 | 1.13e+04 | +3.020 | ≥99th | 0.978 | Very strong binding gain |
| CHIP:CEBPA:HepG2 | 2.54e+03 | 1.72e+04 | +2.760 | ≥99th | 0.991 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.43e+04 | 3.6e+04 | +1.335 | ≥99th | 0.999 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 22.9 | 69.3 | +1.558 | ≥99th | 1.000 | Very strong increase |
| CAGE:HepG2 — variant site | 70.7 | 168 | +1.236 | ≥99th | 1.000 | Very strong increase |
| CAGE:HepG2 — PSRC1 TSS | 2.26e+03 | 2.68e+03 | +0.245 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 2.49 | 3.02 | +0.204 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — CELSR2 TSS | 627 | 722 | +0.203 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 256 | 292 | +0.187 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 52.1 | 59.1 | +0.180 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — MYBPHL TSS | 2.41 | 2.78 | +0.147 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 7.98 | 8.7 | +0.111 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — SORT1 TSS | 3.62e+03 | 3.75e+03 | +0.051 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
