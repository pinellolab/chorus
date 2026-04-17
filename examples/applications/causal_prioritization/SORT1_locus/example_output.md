## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-17 06:51 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.964, max_effect=+0.449, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +0.449 (≥99th) | +0.377 (≥99th) | +0.267 (≥99th) | +0.180 (≥99th) | -0.001 (≥99th) | -0.003 (≥99th) | 0.964 |
| 2 | rs4970836 | 0.91 | -0.039 (≥99th) | -0.027 (≥99th) | -0.027 (≥99th) | -0.015 (≥99th) | -0.003 (≥99th) | -0.003 (≥99th) | 0.583 |
| 3 | rs1624712 | 1.00 | +0.123 (≥99th) | +0.048 (≥99th) | +0.021 (≥99th) | +0.026 (≥99th) | +0.012 (≥99th) | +0.000 | 0.490 |
| 4 | rs660240 | 0.95 | +0.071 (≥99th) | +0.028 (≥99th) | +0.013 (≥99th) | +0.031 (≥99th) | +0.003 (≥99th) | +0.006 (≥99th) | 0.249 |
| 5 | rs142678968 | 0.95 | -0.020 (≥99th) | -0.003 (0.95) | -0.013 (≥99th) | -0.001 (0.44) | +0.007 (≥99th) | +0.000 | 0.196 |
| 6 | rs1626484 | 1.00 | +0.000 | +0.003 (0.71) | +0.000 | -0.003 (≥99th) | -0.006 (≥99th) | +0.003 (≥99th) | 0.195 |
| 7 | rs7528419 | 1.00 | +0.008 (≥99th) | +0.008 (≥99th) | +0.005 (0.86) | +0.020 (≥99th) | -0.001 | -0.002 (≥99th) | 0.174 |
| 8 | rs602633 | 0.86 | +0.037 (≥99th) | +0.031 (≥99th) | +0.032 (≥99th) | +0.003 (≥99th) | +0.001 | +0.001 (≥99th) | 0.035 |
| 9 | rs56960352 | 0.91 | -0.015 (≥99th) | +0.000 | -0.005 (0.92) | -0.003 (≥99th) | +0.009 (≥99th) | +0.002 (≥99th) | 0.020 |
| 10 | rs599839 | 0.91 | +0.003 (0.77) | +0.024 (≥99th) | +0.010 (≥99th) | +0.003 (≥99th) | -0.001 (≥99th) | -0.007 (≥99th) | 0.007 |
| 11 | rs1277930 | 0.91 | -0.005 (≥99th) | -0.016 (≥99th) | -0.016 (≥99th) | -0.002 (0.63) | -0.007 (≥99th) | -0.003 (≥99th) | 0.006 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
