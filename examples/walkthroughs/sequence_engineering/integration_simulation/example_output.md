## Analysis Request

> Insert a 378 bp CMV promoter construct at chr19:55115000 (PPP1R12C locus / AAVS1 safe harbour) and predict local disruption in K562 using DNASE, H3K27ac, and CAGE tracks.

- **Tool**: `simulate_integration`
- **Oracle**: alphagenome
- **Tracks requested**: 3 K562 tracks
- **Generated**: 2026-08-05 04:58 UTC

## Integration Simulation Report

**Variant**: chr19:55115000 wt>insertion
**Oracle**: alphagenome
**Gene**: PPP1R12C
**Other nearby genes**: TNNT1, EPS8L1, TNNI3, ENSG00000267110
**Modification**: Inserted 378 bp construct at chr19:55,115,001
**Modified region**: chr19:55,115,001-55,115,378 (378 bp)

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.96, CAGE:K562); Chromatin accessibility (DNASE/ATAC): very strong opening (+4.26, DNASE:K562); Histone modifications (ChIP-Histone): very strong mark gain (+1.30, CHIP:H3K27ac:K562).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:K562 | 24.4 | 484 | +4.256 | ≥99th | 0.764 | Very strong opening |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 3.23e+03 | 7.97e+03 | +1.304 | ≥99th | 0.873 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:K562 — RPL28 TSS | 7.06e+04 | 141 | -8.959 | ≥99th | 1.000 | Very strong decrease |
| CAGE:K562 — ZNF628 TSS | 2.1e+03 | 6.86 | -8.066 | ≥99th | 0.978 | Very strong decrease |
| CAGE:K562 — KMT5C TSS | 2.27e+03 | 21.9 | -6.628 | ≥99th | 0.979 | Very strong decrease |
| CAGE:K562 — NAT14 TSS | 2.38e+03 | 42.1 | -5.791 | ≥99th | 0.979 | Very strong decrease |
| CAGE:K562 — ZNF581 TSS | 14.7 | 521 | +5.053 | ≥99th | 0.907 | Very strong increase |
| CAGE:K562 — ZNF865 TSS | 1.69e+03 | 126 | -3.732 | ≥99th | 0.974 | Very strong decrease |
| CAGE:K562 — ISOC2 TSS | 5.81 | 72.1 | +3.423 | ≥99th | 0.873 | Very strong increase |
| CAGE:K562 — TMEM238 TSS | 11.8 | 118 | +3.220 | ≥99th | 0.900 | Very strong increase |
| CAGE:K562 — ZNF524 TSS | 797 | 94.7 | -3.059 | ≥99th | 0.968 | Very strong decrease |
| CAGE:K562 — SSC5D TSS | 106 | 14.8 | -2.759 | ≥99th | 0.952 | Very strong decrease |
| _…showing top 10 of 53 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
