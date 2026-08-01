## Analysis Request

> Insert a 378 bp CMV promoter construct at chr19:55115000 (PPP1R12C locus / AAVS1 safe harbour) and predict local disruption in K562 using DNASE, H3K27ac, and CAGE tracks.

- **Tool**: `simulate_integration`
- **Oracle**: alphagenome
- **Tracks requested**: 3 K562 tracks
- **Generated**: 2026-08-01 03:23 UTC

## Integration Simulation Report

**Variant**: chr19:55115000 wt>insertion
**Oracle**: alphagenome
**Gene**: PPP1R12C
**Other nearby genes**: TNNT1, EPS8L1, TNNI3, ENSG00000267110
**Modification**: Inserted 378 bp construct at chr19:55,115,001
**Modified region**: chr19:55,115,001-55,115,378 (378 bp)

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.96, CAGE:K562); Chromatin accessibility (DNASE/ATAC): very strong opening (+4.26, DNASE:K562); Histone modifications (ChIP-Histone): very strong mark gain (+1.24, CHIP:H3K27ac:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 24.4 | 485 | +4.261 | ≥99th | 0.764 | Very strong opening |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 3.65e+03 | 8.65e+03 | +1.242 | ≥99th | 0.972 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — RPL28 TSS | 7.08e+04 | 141 | -8.962 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ZNF628 TSS | 2.1e+03 | 6.85 | -8.063 | ≥99th | 0.978 | Very strong decrease |
| CAGE:K562 — KMT5C TSS | 2.27e+03 | 21.9 | -6.631 | ≥99th | 0.979 | Very strong decrease |
| CAGE:K562 — NAT14 TSS | 2.38e+03 | 42.1 | -5.790 | ≥99th | 0.980 | Very strong decrease |
| CAGE:K562 — ZNF581 TSS | 14.7 | 523 | +5.061 | ≥99th | 0.907 | Very strong increase |
| CAGE:K562 — ZNF865 TSS | 1.68e+03 | 126 | -3.730 | ≥99th | 0.975 | Very strong decrease |
| CAGE:K562 — ISOC2 TSS | 5.81 | 72.1 | +3.425 | ≥99th | 0.872 | Very strong increase |
| CAGE:K562 — TMEM238 TSS | 11.7 | 118 | +3.220 | ≥99th | 0.900 | Very strong increase |
| CAGE:K562 — ZNF524 TSS | 797 | 94.8 | -3.058 | ≥99th | 0.968 | Very strong decrease |
| CAGE:K562 — SSC5D TSS | 105 | 14.8 | -2.753 | ≥99th | 0.952 | Very strong decrease |
| _…showing top 10 of 53 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
