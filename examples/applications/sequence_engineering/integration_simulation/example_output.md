## Analysis Request

> Insert a 378 bp CMV promoter construct at chr19:55115000 (PPP1R12C locus / AAVS1 safe harbour) and predict local disruption in K562 using DNASE, H3K27ac, and CAGE tracks.

- **Tool**: `simulate_integration`
- **Oracle**: alphagenome
- **Tracks requested**: 3 K562 tracks
- **Generated**: 2026-04-17 20:08 UTC

## Integration Simulation Report

**Variant**: chr19:55115000 wt>insertion
**Oracle**: alphagenome
**Gene**: PPP1R12C
**Other nearby genes**: TNNT1, EPS8L1, TNNI3, ENSG00000267110
**Modification**: Inserted 378 bp construct at chr19:55,115,001
**Modified region**: chr19:55,115,001-55,115,378 (378 bp)

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.96); Chromatin accessibility (DNASE/ATAC): very strong opening (+4.22); Histone modifications (ChIP-Histone): very strong mark gain (+1.20).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 24.4 | 473 | +4.224 | ≥99th | 0.764 | Very strong opening |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 3.77e+03 | 8.65e+03 | +1.197 | ≥99th | 0.974 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — RPL28 TSS | 7.06e+04 | 140 | -8.963 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ZNF628 TSS | 2.04e+03 | 7.1 | -7.975 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — KMT5C TSS | 2.27e+03 | 22.3 | -6.607 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — NAT14 TSS | 2.47e+03 | 40.8 | -5.889 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ZNF581 TSS | 14.9 | 517 | +5.028 | ≥99th | 1.000 | Very strong increase |
| CAGE:K562 — ZNF865 TSS | 1.66e+03 | 125 | -3.725 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ISOC2 TSS | 5.82 | 70.5 | +3.391 | ≥99th | 1.000 | Very strong increase |
| CAGE:K562 — TMEM238 TSS | 11.3 | 119 | +3.281 | ≥99th | 1.000 | Very strong increase |
| CAGE:K562 — ZNF524 TSS | 823 | 98.1 | -3.055 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — SSC5D TSS | 108 | 16.1 | -2.672 | ≥99th | 1.000 | Very strong decrease |
| _…showing top 10 of 53 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
