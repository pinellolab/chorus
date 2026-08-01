## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-08-01 03:22 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.977, max_effect=+3.050, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +1.331 (≥99th) | +2.769 (≥99th) | +3.050 (≥99th) | +1.258 (≥99th) | +1.201 (≥99th) | +1.503 (≥99th) | 0.977 |
| 2 | rs1624712 | 1.00 | +0.129 (0.97) | +0.036 (0.84) | +0.029 (0.73) | +0.030 (0.82) | +0.035 (0.95) | +0.089 (0.98) | 0.399 |
| 3 | rs7528419 | 1.00 | -0.057 (0.91) | +0.009 (0.39) | -0.002 (0.06) | -0.017 (0.71) | -0.039 (0.95) | -0.068 (0.97) | 0.206 |
| 4 | rs660240 | 0.95 | +0.006 (0.28) | +0.011 (0.49) | +0.013 (0.45) | -0.005 (0.35) | +0.001 (0.42) | +0.025 (0.93) | 0.190 |
| 5 | rs142678968 | 0.95 | +0.017 (0.62) | -0.007 (0.33) | -0.004 (0.15) | +0.001 | -0.097 (0.98) | -0.086 (0.98) | 0.167 |
| 6 | rs1626484 | 1.00 | -0.059 (0.91) | -0.010 (0.46) | +0.000 | -0.022 (0.77) | +0.031 (0.94) | -0.029 (0.94) | 0.145 |
| 7 | rs602633 | 0.86 | +0.079 (0.94) | +0.023 (0.73) | +0.021 (0.63) | +0.006 (0.39) | +0.003 (0.67) | +0.008 (0.81) | 0.027 |
| 8 | rs56960352 | 0.91 | -0.042 (0.86) | -0.010 (0.43) | -0.015 (0.51) | -0.005 (0.31) | +0.006 (0.77) | +0.007 (0.79) | 0.018 |
| 9 | rs4970836 | 0.91 | -0.023 (0.73) | -0.007 (0.35) | -0.005 (0.16) | +0.009 (0.50) | +0.004 (0.69) | +0.052 (0.97) | 0.011 |
| 10 | rs1277930 | 0.91 | -0.008 (0.35) | -0.024 (0.74) | -0.029 (0.73) | +0.001 (0.08) | +0.025 (0.93) | +0.096 (0.98) | 0.011 |
| 11 | rs599839 | 0.91 | -0.007 (0.31) | -0.006 (0.27) | -0.007 (0.27) | -0.005 (0.33) | -0.013 (0.87) | +0.003 (0.65) | 0.000 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
