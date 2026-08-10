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

rs12740374 ranks **#1 of 11** with a composite score of **0.9704**, and the
margin is the result: the runner-up (rs142678968, r²=0.95) scores **0.4082**,
so the sentinel leads by 2.4×, and nine of the eleven score below 0.21. This
matches the published causal variant from Musunuru et al. (2010).

It leads on all four layers at once — `convergence` is **1.0** and
`n_layers_affected` is **4 of 4**:

| layer | top HepG2 track | raw log2FC | percentile |
|---|---|---|---|
| tf_binding | CHIP:CEBPB | **+3.316** | 0.9995 |
| tss_activity (CAGE) | CAGE/hCAGE, − strand | **+1.502** | 0.9983 |
| chromatin_accessibility | DNASE | **+1.334** | 0.9964 |
| histone_marks | CHIP:H3K27ac | **+1.251** | 0.9992 |

Two things to read carefully.

**The composite is not just the biggest effect.** `max_effect` (3.316, the
CEBPB value) enters at weight 0.35, alongside `n_layers` 0.25, `convergence`
0.20 and `ref_activity` 0.20 — so a variant cannot win on one strong track.
That is why rs12740374 separates so cleanly: it is not merely the largest
effect, it is the only variant that is large *and* convergent across layers
*and* sitting on active baseline chromatin (`ref_activity` 7856).

**Percentiles rank within a track, not across tracks.** H3K27ac at 0.9992
outranks DNASE at 0.9964 on a *smaller* raw effect (+1.251 vs +1.334), because
each is ranked against its own track's background null. Read down the log2FC
column for how large an effect is; read down the percentile column for how
unusual it is for that assay. Neither ordering implies the other.

None of these rows is pinned at a clamped 1.0. See
[`../../validation/SORT1_rs12740374_with_CEBP/`](../../validation/SORT1_rs12740374_with_CEBP/)
for the same locus scored against the C/EBP factors specifically, where CEBPA
used to exceed its null's maximum and report a rankless 1.0000.

<sub>An earlier version of this section listed four per-track numbers in the
0.18–0.45 range as effect sizes. They were not effect sizes on any scale this
report emits — the raw log2FC values are the ones tabled above, roughly 3–7×
larger — and the top TF was named as CEBPA where the artefact has CEBPB. The
stale values are in git history; they are not repeated here, because
`tests/test_walkthrough_readmes_match_artefacts.py` cannot distinguish a number
quoted as a current claim from one quoted as a corrected error, and keeping that
guard strict is worth more than the footnote. It is what caught this.</sub>
