## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-17 20:08 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.964, max_effect=+0.448, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +0.448 (≥99th) | +0.377 (≥99th) | +0.272 (≥99th) | +0.177 (≥99th) | -0.010 (≥99th) | +0.004 (≥99th) | 0.964 |
| 2 | rs4970836 | 0.91 | -0.041 (≥99th) | -0.030 (≥99th) | -0.027 (≥99th) | -0.014 (≥99th) | -0.000 | -0.000 | 0.586 |
| 3 | rs1624712 | 1.00 | +0.124 (≥99th) | +0.048 (≥99th) | +0.024 (≥99th) | +0.026 (≥99th) | -0.004 (≥99th) | -0.001 (≥99th) | 0.492 |
| 4 | rs660240 | 0.95 | +0.073 (≥99th) | +0.022 (≥99th) | +0.019 (≥99th) | +0.029 (≥99th) | -0.008 (≥99th) | +0.000 | 0.254 |
| 5 | rs142678968 | 0.95 | -0.024 (≥99th) | +0.003 (0.95) | -0.011 (≥99th) | -0.004 (≥99th) | -0.004 (≥99th) | +0.001 (≥99th) | 0.201 |
| 6 | rs1626484 | 1.00 | +0.005 (≥99th) | +0.005 (≥99th) | +0.008 (≥99th) | -0.001 | -0.001 | -0.009 (≥99th) | 0.192 |
| 7 | rs7528419 | 1.00 | +0.017 (≥99th) | +0.014 (≥99th) | +0.010 (≥99th) | +0.020 (≥99th) | -0.001 (≥99th) | -0.002 (≥99th) | 0.176 |
| 8 | rs602633 | 0.86 | +0.038 (≥99th) | +0.033 (≥99th) | +0.032 (≥99th) | +0.003 (≥99th) | +0.001 (≥99th) | -0.008 (≥99th) | 0.037 |
| 9 | rs56960352 | 0.91 | -0.014 (≥99th) | -0.004 (≥99th) | -0.008 (≥99th) | -0.005 (≥99th) | -0.005 (≥99th) | -0.003 (≥99th) | 0.020 |
| 10 | rs1277930 | 0.91 | -0.004 (≥99th) | -0.014 (≥99th) | -0.016 (≥99th) | -0.000 | +0.004 (≥99th) | +0.000 | 0.006 |
| 11 | rs599839 | 0.91 | +0.002 (0.62) | +0.021 (≥99th) | +0.015 (≥99th) | +0.001 | -0.007 (≥99th) | -0.002 (≥99th) | 0.005 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
