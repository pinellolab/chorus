## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-08-05 04:59 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.970, max_effect=+3.316, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +1.334 (≥99th) | +2.945 (≥99th) | +3.316 (≥99th) | +1.251 (≥99th) | +1.203 (≥99th) | +1.502 (≥99th) | 0.970 |
| 2 | rs142678968 | 0.95 | +0.015 (0.45) | +0.006 (0.28) | -0.002 (0.07) | +0.003 (0.34) | -0.102 (0.91) | -0.086 (0.90) | 0.408 |
| 3 | rs1624712 | 1.00 | +0.131 (0.91) | +0.036 (0.75) | +0.028 (0.70) | +0.032 (0.88) | +0.034 (0.78) | +0.088 (0.90) | 0.391 |
| 4 | rs7528419 | 1.00 | -0.055 (0.80) | +0.010 (0.38) | -0.004 (0.16) | -0.023 (0.83) | -0.036 (0.79) | -0.062 (0.86) | 0.205 |
| 5 | rs660240 | 0.95 | +0.002 (0.08) | +0.023 (0.64) | +0.019 (0.58) | -0.004 (0.40) | -0.003 (0.36) | +0.018 (0.68) | 0.179 |
| 6 | rs1626484 | 1.00 | -0.059 (0.81) | -0.007 (0.30) | +0.004 (0.14) | -0.024 (0.84) | +0.026 (0.74) | -0.033 (0.77) | 0.125 |
| 7 | rs602633 | 0.86 | +0.077 (0.85) | +0.032 (0.72) | +0.026 (0.68) | +0.006 (0.55) | +0.002 (0.31) | +0.006 (0.48) | 0.023 |
| 8 | rs56960352 | 0.91 | -0.039 (0.72) | -0.014 (0.48) | -0.021 (0.61) | -0.004 (0.43) | +0.005 (0.45) | +0.004 (0.42) | 0.017 |
| 9 | rs4970836 | 0.91 | -0.024 (0.59) | -0.014 (0.49) | -0.003 (0.10) | +0.009 (0.64) | +0.004 (0.38) | +0.058 (0.85) | 0.011 |
| 10 | rs1277930 | 0.91 | -0.007 (0.23) | -0.029 (0.70) | -0.030 (0.71) | +0.000 | +0.027 (0.74) | +0.094 (0.91) | 0.010 |
| 11 | rs599839 | 0.91 | -0.008 (0.27) | -0.002 (0.06) | -0.008 (0.31) | +0.000 | -0.013 (0.62) | +0.003 (0.34) | 0.000 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
