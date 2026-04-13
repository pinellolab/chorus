## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-13 03:07 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: HepG2
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.964, max_effect=+0.449, 4 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 | CAGE:HepG2 | Composite |
|------|---------|-----|---|---|---|---|---|---|-----------|
| 1 | rs12740374 ★ | 1.00 | +0.449 (100%) | +0.375 (100%) | +0.262 (100%) | +0.181 (100%) | -0.005 (100%) | -0.006 (100%) | 0.964 |
| 2 | rs4970836 | 0.91 | -0.040 (100%) | -0.031 (100%) | -0.027 (100%) | -0.015 (100%) | +0.002 (100%) | -0.001 (100%) | 0.584 |
| 3 | rs1624712 | 1.00 | +0.122 (100%) | +0.051 (100%) | +0.019 (100%) | +0.028 (100%) | +0.002 (100%) | +0.004 (100%) | 0.490 |
| 4 | rs660240 | 0.95 | +0.069 (100%) | +0.023 (100%) | +0.013 (100%) | +0.031 (100%) | -0.003 (100%) | +0.003 (100%) | 0.252 |
| 5 | rs1626484 | 1.00 | +0.005 (100%) | +0.011 (100%) | +0.001 (21%) | +0.000 (43%) | -0.004 (100%) | +0.002 (100%) | 0.193 |
| 6 | rs142678968 | 0.95 | -0.025 (100%) | +0.000 (29%) | -0.016 (100%) | -0.002 (46%) | +0.001 (100%) | +0.005 (100%) | 0.192 |
| 7 | rs7528419 | 1.00 | +0.011 (100%) | +0.005 (100%) | +0.013 (100%) | +0.022 (100%) | -0.001 (100%) | +0.002 (100%) | 0.176 |
| 8 | rs602633 | 0.86 | +0.037 (100%) | +0.033 (100%) | +0.032 (100%) | +0.001 (44%) | +0.005 (100%) | +0.005 (100%) | 0.036 |
| 9 | rs56960352 | 0.91 | -0.015 (100%) | -0.002 (46%) | -0.007 (100%) | -0.007 (100%) | -0.014 (100%) | -0.004 (100%) | 0.020 |
| 10 | rs599839 | 0.91 | +0.004 (100%) | +0.024 (100%) | +0.010 (100%) | +0.004 (100%) | -0.010 (100%) | -0.007 (100%) | 0.007 |
| 11 | rs1277930 | 0.91 | -0.005 (100%) | -0.016 (100%) | -0.018 (100%) | +0.001 (44%) | -0.007 (100%) | +0.001 (100%) | 0.003 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
