## Analysis Request

> Replace the SORT1 enhancer region chr1:109274500-109275500 with a 630 bp GFP/reporter construct sequence and predict effects on K562 DNASE, H3K27ac, H3K4me3, and CAGE.

- **Tool**: `analyze_region_swap`
- **Oracle**: alphagenome
- **Tracks requested**: 4 K562 tracks
- **Generated**: 2026-08-12 18:00 UTC

## Region Swap Analysis Report

**Variant**: chr1:109275000 wt>replacement
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1
**Modification**: Replaced 1,000 bp region (chr1:109,274,501-109,275,500) with a 630 bp custom sequence
**Modified region**: chr1:109,274,501-109,275,500 (1,000 bp)

**Summary**: Strongest effect per layer anywhere in the prediction window (not necessarily SORT1's own track). TSS activity (CAGE/PRO-CAP): very strong decrease (-7.96, CAGE:K562 — GSTM2 TSS); Chromatin accessibility (DNASE/ATAC): very strong closing (-3.30, DNASE:K562); Histone modifications (ChIP-Histone): very strong mark loss (-1.48, CHIP:H3K27ac:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 218 | 21.2 | -3.299 | 0.9997 | 0.901 | Very strong closing |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 4.75e+03 | 1.7e+03 | -1.477 | 0.9999 | 0.893 | Very strong mark loss |
| CHIP:H3K4me3:K562 | 2.03e+03 | 955 | -1.091 | 0.9994 | 0.848 | Very strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — GSTM2 TSS | 1.05e+03 | 3.23 | -7.957 | ≥99th (2.72× null max) | 0.971 | Very strong decrease |
| CAGE:K562 — GNAI3 TSS | 1.09e+04 | 63.3 | -7.401 | ≥99th (2.53× null max) | 0.997 | Very strong decrease |
| CAGE:K562 — GSTM1 TSS | 165 | 0.742 | -6.576 | ≥99th (2.25× null max) | 0.953 | Very strong decrease |
| CAGE:K562 — CYB561D1 TSS | 896 | 11.3 | -6.193 | ≥99th (2.11× null max) | 0.969 | Very strong decrease |
| CAGE:K562 — ATXN7L2 TSS | 1.28e+03 | 45.1 | -4.792 | ≥99th (1.64× null max) | 0.973 | Very strong decrease |
| CAGE:K562 — AMPD2 TSS | 1.32e+03 | 47.4 | -4.770 | ≥99th (1.63× null max) | 0.973 | Very strong decrease |
| CAGE:K562 — SYPL2 TSS | 32.3 | 2.98 | -3.063 | ≥99th (1.05× null max) | 0.928 | Very strong decrease |
| CAGE:K562 — GSTM5 TSS | 10.3 | 0.367 | -3.043 | ≥99th (1.04× null max) | 0.894 | Very strong decrease |
| CAGE:K562 — GSTM3 TSS | 9.63 | 0.902 | -2.482 | 0.9998 | 0.892 | Very strong decrease |
| CAGE:K562 — GSTM4 TSS | 3.04e+03 | 697 | -2.121 | 0.9997 | 0.983 | Very strong decrease |
| _…showing top 10 of 29 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **`N× null max`**: the effect exceeded *every* sampled background effect for that track, so the percentile is clamped and cannot rank it further. The multiplier gives the distance to that ceiling — `1.11×` is 11% beyond the most extreme background effect for that track. Common for variants that create or destroy a complete transcription-factor motif, which even a regulatory-region background rarely contains.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
