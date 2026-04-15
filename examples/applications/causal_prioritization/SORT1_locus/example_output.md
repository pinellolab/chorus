## Analysis Request

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374 with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Cell types**: HepG2
- **Generated**: 2026-04-15 05:12 UTC

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
| 1 | rs12740374 ★ | 1.00 | +0.448 (100%) | +0.376 (100%) | +0.269 (100%) | +0.180 (100%) | +0.001 (100%) | -0.003 (100%) | 0.964 |
| 2 | rs4970836 | 0.91 | -0.041 (100%) | -0.032 (100%) | -0.027 (100%) | -0.014 (100%) | -0.004 (100%) | +0.001 (100%) | 0.583 |
| 3 | rs1624712 | 1.00 | +0.120 (100%) | +0.051 (100%) | +0.024 (100%) | +0.025 (100%) | -0.007 (100%) | -0.004 (100%) | 0.488 |
| 4 | rs660240 | 0.95 | +0.068 (100%) | +0.026 (100%) | +0.019 (100%) | +0.027 (100%) | -0.011 (100%) | +0.001 (100%) | 0.246 |
| 5 | rs1626484 | 1.00 | +0.005 (100%) | +0.004 (100%) | +0.004 (78%) | -0.001 (44%) | -0.001 (100%) | +0.003 (100%) | 0.193 |
| 6 | rs142678968 | 0.95 | -0.029 (100%) | +0.001 (30%) | -0.018 (100%) | -0.005 (100%) | -0.000 (82%) | +0.000 (14%) | 0.191 |
| 7 | rs7528419 | 1.00 | +0.012 (100%) | +0.003 (79%) | +0.006 (100%) | +0.019 (100%) | +0.006 (100%) | -0.001 (100%) | 0.174 |
| 8 | rs602633 | 0.86 | +0.039 (100%) | +0.033 (100%) | +0.033 (100%) | +0.003 (100%) | -0.001 (100%) | +0.005 (100%) | 0.036 |
| 9 | rs56960352 | 0.91 | -0.016 (100%) | +0.000 (29%) | -0.003 (65%) | -0.006 (100%) | -0.002 (100%) | -0.004 (100%) | 0.029 |
| 10 | rs599839 | 0.91 | +0.007 (100%) | +0.021 (100%) | +0.012 (100%) | +0.003 (100%) | -0.000 (21%) | +0.004 (100%) | 0.004 |
| 11 | rs1277930 | 0.91 | -0.005 (100%) | -0.016 (100%) | -0.016 (100%) | -0.001 (44%) | -0.003 (100%) | +0.002 (100%) | 0.001 |

Each cell: **raw effect** (effect percentile). Composite score combines effect magnitude, layer convergence, and baseline activity.
