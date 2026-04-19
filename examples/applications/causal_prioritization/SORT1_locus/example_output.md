## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-18 22:19 UTC

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
| 1 | rs12740374 ★ | 1.00 | +0.448 (≥99th) | +0.376 (≥99th) | +0.266 (≥99th) | +0.181 (≥99th) | +0.006 (≥99th) | +0.005 (≥99th) | 0.964 |
| 2 | rs4970836 | 0.91 | -0.040 (≥99th) | -0.032 (≥99th) | -0.028 (≥99th) | -0.013 (≥99th) | -0.003 (≥99th) | -0.002 (≥99th) | 0.598 |
| 3 | rs1624712 | 1.00 | +0.122 (≥99th) | +0.051 (≥99th) | +0.021 (≥99th) | +0.025 (≥99th) | +0.005 (≥99th) | +0.000 | 0.491 |
| 4 | rs660240 | 0.95 | +0.068 (≥99th) | +0.028 (≥99th) | +0.015 (≥99th) | +0.031 (≥99th) | -0.004 (≥99th) | -0.000 | 0.250 |
| 5 | rs142678968 | 0.95 | -0.026 (≥99th) | -0.003 (0.95) | -0.014 (≥99th) | -0.003 (≥99th) | +0.007 (≥99th) | -0.007 (≥99th) | 0.194 |
| 6 | rs1626484 | 1.00 | +0.007 (≥99th) | +0.005 (≥99th) | +0.003 (0.52) | +0.002 (0.47) | -0.001 (≥99th) | +0.005 (≥99th) | 0.194 |
| 7 | rs7528419 | 1.00 | +0.014 (≥99th) | +0.007 (≥99th) | +0.013 (≥99th) | +0.020 (≥99th) | +0.001 (≥99th) | -0.003 (≥99th) | 0.176 |
| 8 | rs602633 | 0.86 | +0.039 (≥99th) | +0.034 (≥99th) | +0.032 (≥99th) | +0.004 (≥99th) | -0.001 | -0.008 (≥99th) | 0.038 |
| 9 | rs56960352 | 0.91 | -0.014 (≥99th) | +0.002 (0.46) | -0.005 (0.92) | -0.003 (≥99th) | +0.003 (≥99th) | +0.005 (≥99th) | 0.020 |
| 10 | rs599839 | 0.91 | +0.004 (0.91) | +0.024 (≥99th) | +0.010 (≥99th) | +0.002 (0.77) | +0.006 (≥99th) | +0.001 | 0.009 |
| 11 | rs1277930 | 0.91 | -0.006 (≥99th) | -0.016 (≥99th) | -0.018 (≥99th) | -0.002 (0.46) | -0.005 (≥99th) | -0.006 (≥99th) | 0.005 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
