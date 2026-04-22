## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-22 17:15 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.983, max_effect=+3.029, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +1.395 (≥99th) | +2.765 (≥99th) | +3.029 (≥99th) | +1.334 (≥99th) | -0.010 (≥99th) | -0.009 (≥99th) | 0.983 |
| 2 | rs142678968 | 0.95 | +0.009 (≥99th) | +0.008 (≥99th) | +0.004 (0.80) | +0.000 | +0.001 (≥99th) | -0.002 (≥99th) | 0.441 |
| 3 | rs1624712 | 1.00 | +0.142 (≥99th) | +0.049 (≥99th) | +0.039 (≥99th) | +0.035 (≥99th) | -0.008 (≥99th) | -0.004 (≥99th) | 0.401 |
| 4 | rs660240 | 0.95 | +0.011 (≥99th) | +0.009 (≥99th) | +0.004 (0.71) | -0.000 | -0.005 (≥99th) | -0.002 (≥99th) | 0.201 |
| 5 | rs7528419 | 1.00 | -0.065 (≥99th) | -0.004 (≥99th) | -0.017 (≥99th) | -0.025 (≥99th) | -0.003 (≥99th) | -0.002 (≥99th) | 0.195 |
| 6 | rs1626484 | 1.00 | -0.061 (≥99th) | -0.023 (≥99th) | -0.007 (≥99th) | -0.020 (≥99th) | -0.006 (≥99th) | +0.005 (≥99th) | 0.145 |
| 7 | rs602633 | 0.86 | +0.064 (≥99th) | +0.017 (≥99th) | +0.020 (≥99th) | -0.000 | -0.012 (≥99th) | -0.001 (≥99th) | 0.025 |
| 8 | rs56960352 | 0.91 | -0.043 (≥99th) | -0.012 (≥99th) | -0.015 (≥99th) | -0.008 (≥99th) | -0.007 (≥99th) | +0.003 (≥99th) | 0.019 |
| 9 | rs4970836 | 0.91 | -0.025 (≥99th) | -0.006 (≥99th) | -0.003 (0.50) | +0.008 (≥99th) | -0.004 (≥99th) | -0.005 (≥99th) | 0.014 |
| 10 | rs1277930 | 0.91 | -0.011 (≥99th) | -0.027 (≥99th) | -0.029 (≥99th) | -0.000 | -0.001 (≥99th) | -0.000 | 0.009 |
| 11 | rs599839 | 0.91 | -0.006 (≥99th) | +0.004 (≥99th) | -0.006 (≥99th) | -0.005 (≥99th) | -0.001 | +0.002 (≥99th) | 0.000 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
