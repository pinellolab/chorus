## Analysis Request

> Replace the SORT1 enhancer region chr1:109274500-109275500 with a 630 bp GFP/reporter construct sequence and predict effects on K562 DNASE, H3K27ac, H3K4me3, and CAGE.

- **Tool**: `analyze_region_swap`
- **Oracle**: alphagenome
- **Tracks requested**: 4 K562 tracks
- **Generated**: 2026-04-21 03:23 UTC

## Region Swap Analysis Report

**Variant**: chr1:109275000 wt>replacement
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1
**Modification**: Replaced 1,000 bp region (chr1:109,274,501-109,275,500) with a 630 bp custom sequence
**Modified region**: chr1:109,274,501-109,275,500 (1,000 bp)

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.11, CAGE:K562); Chromatin accessibility (DNASE/ATAC): very strong closing (-3.29, DNASE:K562); Histone modifications (ChIP-Histone): very strong mark loss (-1.35, CHIP:H3K27ac:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 215 | 21.1 | -3.288 | ≥99th | 0.900 | Very strong closing |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 4.98e+03 | 1.95e+03 | -1.350 | ≥99th | 0.987 | Very strong mark loss |
| CHIP:H3K4me3:K562 | 2.16e+03 | 1.06e+03 | -1.033 | ≥99th | 0.895 | Very strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — GSTM2 TSS | 1.14e+03 | 3.13 | -8.114 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — GNAI3 TSS | 1.07e+04 | 63.2 | -7.379 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — GSTM1 TSS | 171 | 0.725 | -6.642 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — CYB561D1 TSS | 877 | 11.4 | -6.144 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ATXN7L2 TSS | 1.26e+03 | 44 | -4.810 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — AMPD2 TSS | 1.3e+03 | 46.8 | -4.767 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — SYPL2 TSS | 31 | 2.86 | -3.053 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — GSTM5 TSS | 10.3 | 0.356 | -3.053 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — GSTM3 TSS | 9.49 | 0.899 | -2.466 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — GSTM4 TSS | 3.14e+03 | 741 | -2.083 | ≥99th | 1.000 | Very strong decrease |
| _…showing top 10 of 29 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
