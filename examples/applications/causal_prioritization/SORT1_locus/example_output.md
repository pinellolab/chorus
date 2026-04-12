## Analysis Request

> I'm fine-mapping the SORT1 LDL cholesterol GWAS locus. The sentinel is rs12740374 and I have 11 LD variants (r²≥0.85). Score each one across all regulatory layers in HepG2 and rank by causal evidence. Gene is SORT1.

- **Tool**: `fine_map_causal_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all oracle tracks (discovery mode)
- **Generated**: 2026-04-12 14:14 UTC

## Causal Variant Prioritization Report

**Sentinel**: rs12740374
**Oracle**: alphagenome
**Cell type(s)**: multiple tissues (all-tracks mode)
**Gene**: SORT1
**Variants scored**: 11

**Top candidate**: rs12740374 (composite=0.954, max_effect=+1.909, 5 layers affected, convergence=1.00)
The sentinel SNP itself is the top candidate.

| Rank | Variant | r² | Max Effect | Layers | Convergence | Top Layer | Composite |
|------|---------|-----|-----------|--------|-------------|-----------|-----------|
| 1 | rs12740374 ★ | 1.00 | +1.909 | 5 | 1.00 | chromatin_accessibility | 0.954 |
| 2 | rs660240 | 0.95 | +0.205 | 3 | 1.00 | tss_activity | 0.545 |
| 3 | rs7528419 | 1.00 | +0.212 | 2 | 1.00 | tss_activity | 0.464 |
| 4 | rs56960352 | 0.91 | -0.583 | 3 | 1.00 | chromatin_accessibility | 0.424 |
| 5 | rs142678968 | 0.95 | +0.122 | 1 | 1.00 | gene_expression | 0.370 |
| 6 | rs4970836 | 0.91 | -0.552 | 2 | 1.00 | tss_activity | 0.350 |
| 7 | rs1624712 | 1.00 | +0.297 | 3 | 0.33 | chromatin_accessibility | 0.348 |
| 8 | rs602633 | 0.86 | -0.314 | 3 | 0.33 | chromatin_accessibility | 0.233 |
| 9 | rs1626484 | 1.00 | -0.193 | 2 | 0.00 | gene_expression | 0.230 |
| 10 | rs599839 | 0.91 | -0.152 | 1 | 1.00 | gene_expression | 0.220 |
| 11 | rs1277930 | 0.91 | -0.101 | 1 | 1.00 | chromatin_accessibility | 0.210 |

---

---

## Interpretation

**What the oracle sees.** Across 11 LD variants (r²≥0.85) the sentinel
rs12740374 wins decisively: composite score 0.954, maximum effect +1.9
log2FC, effects in **5 of 5** scored regulatory layers, directional
convergence 1.00 (every layer agrees on direction). The next-best
variant, rs660240, scores 0.545 with a +0.21 log2FC effect in only 3
layers. All other variants are well below 0.5.

**How this fits the published biology.** This is exactly the pattern
published fine-mapping studies found for the SORT1 locus: rs12740374 is
the single causal variant despite being in perfect LD with several
other SNPs. Chorus's composite causal score recovers it by combining
effect magnitude with multi-layer convergence, which is the same
intuition wet-lab functional assays encode.

**Suggested next steps.**
- This is the template for any GWAS fine-mapping task. Replace the
  lead variant with your own and rerun — if one variant in your
  credible set has composite >> 0.7 and multi-layer convergence = 1.0,
  that's your prime functional candidate.
- Inspect `result.scores[0].per_layer_scores` in the JSON for the
  mechanism: here the top layer is `chromatin_accessibility`, which
  matches the published C/EBP-mediated enhancer opening.
- For truly ambiguous loci, run the same fine-mapping with two oracles
  (AlphaGenome + Borzoi) and intersect the top candidate — agreement
  between independent architectures strengthens the conclusion.
