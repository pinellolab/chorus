## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-21 01:19 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.963, max_effect=+0.431, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +0.431 (≥99th) | +0.371 (≥99th) | +0.282 (≥99th) | +0.166 (≥99th) | +0.008 (≥99th) | +0.004 (≥99th) | 0.963 |
| 2 | rs4970836 | 0.91 | -0.040 (≥99th) | -0.031 (≥99th) | -0.023 (≥99th) | -0.013 (≥99th) | +0.012 (≥99th) | +0.005 (≥99th) | 0.586 |
| 3 | rs1624712 | 1.00 | +0.109 (≥99th) | +0.054 (≥99th) | +0.019 (≥99th) | +0.023 (≥99th) | +0.010 (≥99th) | +0.005 (≥99th) | 0.471 |
| 4 | rs660240 | 0.95 | +0.073 (≥99th) | +0.028 (≥99th) | +0.018 (≥99th) | +0.031 (≥99th) | +0.003 (≥99th) | +0.008 (≥99th) | 0.247 |
| 5 | rs142678968 | 0.95 | -0.030 (≥99th) | -0.002 (0.61) | -0.016 (≥99th) | -0.004 (≥99th) | -0.010 (≥99th) | +0.005 (≥99th) | 0.185 |
| 6 | rs1626484 | 1.00 | +0.006 (≥99th) | +0.005 (≥99th) | +0.004 (0.78) | -0.006 (≥99th) | +0.005 (≥99th) | -0.004 (≥99th) | 0.185 |
| 7 | rs7528419 | 1.00 | +0.012 (≥99th) | +0.009 (≥99th) | +0.013 (≥99th) | +0.020 (≥99th) | +0.002 (≥99th) | -0.002 (≥99th) | 0.166 |
| 8 | rs602633 | 0.86 | +0.041 (≥99th) | +0.034 (≥99th) | +0.035 (≥99th) | +0.002 (0.95) | -0.005 (≥99th) | +0.003 (≥99th) | 0.034 |
| 9 | rs56960352 | 0.91 | -0.018 (≥99th) | -0.004 (≥99th) | -0.005 (0.93) | -0.006 (≥99th) | -0.004 (≥99th) | -0.000 | 0.018 |
| 10 | rs1277930 | 0.91 | -0.003 (0.88) | -0.014 (≥99th) | -0.018 (≥99th) | -0.001 | -0.003 (≥99th) | -0.004 (≥99th) | 0.003 |
| 11 | rs599839 | 0.91 | +0.010 (≥99th) | +0.023 (≥99th) | +0.010 (≥99th) | +0.002 (0.48) | -0.004 (≥99th) | +0.005 (≥99th) | 0.003 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
