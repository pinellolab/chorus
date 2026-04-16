# SORT1 Locus — Causal Variant Fine-Mapping

## Locus: 1p13.3 (rs12740374 sentinel, 11 LD variants)

Fine-mapping the SORT1/CELSR2 GWAS locus for LDL cholesterol. Scores
each of 11 LD-correlated variants (r²≥0.85) across 6 HepG2 regulatory
tracks and ranks by composite causal evidence combining effect magnitude,
layer convergence, directional agreement, and baseline activity.

## Example prompt

> Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is rs12740374
> with 11 LD variants (r²≥0.85). Score each variant across HepG2 DNASE,
> CEBPA/CEBPB ChIP, H3K27ac, and CAGE. Rank by composite causal evidence.

## What Claude does

1. `load_oracle('alphagenome')`
2. `fine_map_causal_variant('alphagenome', 'rs12740374', ld_variants=[...], assay_ids=[...HepG2 tracks...])`
3. Generates a ranked table with per-track scores + composite causal score

## Key results

rs12740374 ranks #1 with composite score 0.964, driven by strong
convergent effects across all regulatory layers (DNASE +0.45, CEBPA +0.38,
CEBPB +0.27, H3K27ac +0.18). This matches the published causal variant
from Musunuru et al. (2010).
