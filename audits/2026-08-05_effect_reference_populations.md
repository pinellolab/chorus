# Anchoring every effect null on the regions its assay measures

*2026-08-05. Every number below was measured on this box; the scripts are in
`scripts/`. Three of my own measurements were wrong before they were right, and those
are recorded too, because the mistakes are more instructive than the result.*

## The question

A variant-effect percentile answers "how unusual is this effect, compared to what?".
The "what" is the effect null's reference population. For a localised assay, uniformly
random genomic positions are the wrong answer: almost none carry the signal the model
reads, so the pseudocount damps their log-ratios toward zero, the null's body collapses,
and real effects pile up at the top of the scale where the column no longer
discriminates.

## What the fleet actually looked like

Measured, not assumed — by reading which pass consumes each position source:

| oracle | peak-anchored **effect** positions before this work |
|---|---|
| ChromBPNet | ✅ 10,000 DHS summits ±150 bp **+** 10,000 uniform |
| Cherimoya | ✅ `snps = random + dhs`, explicitly unioned |
| AlphaGenome, Enformer, Borzoi | ❌ gene-anchored, no peak component |
| Sei, LegNet | ❌ cCREs used for **baselines only** |
| EPInformer-seq | ❌ its `sample_positions()` (random + cCRE + DHS) fed `build_baseline_backgrounds()` only |

So five of eight had *some* peak anchoring and three had none — and two of those three
already sampled cCREs for their baseline pass. **An asymmetry inside a single oracle**
(effect uniform, baseline peak-anchored) is harder to defend than any difference
between oracles.

Layers are also shared far more widely than per-layer treatment assumes:

| layer | AG | enformer | borzoi | chrombpnet | cherimoya | epinf |
|---|---|---|---|---|---|---|
| chromatin_accessibility | 472 | 684 | 906 | 9 | 1,518 | 11 |
| histone_marks | 1,116 | 1,890 | ⊂3,886 | ⊂744 | — | 22 |
| tf_binding | 1,617 | 2,101 | ⊂3,886 | ⊂744 | — | — |
| tss_activity | 558 | 638 | 1,276 | — | — | — |
| gene_expression | 667 | — | 1,543 | — | — | — |

Accessibility exists in **six** oracles. Sei is a separate taxonomy entirely — 40
sequence classes, not per-assay tracks — so it maps onto no layer at all.

## The design, and the mistake that produced it

**Attempt 1 — per-layer reference sets.** Rejected for two reasons. It requires keying
on a per-row layer field, and the builders disagreed on vocabulary: Enformer wrote its
internal `spec_key` (`DNASE`, `CHIP_TF`, …) while AlphaGenome wrote canonical names, so
a composition keyed on `chromatin_accessibility` matched **472 of AlphaGenome's rows
and 0 of Enformer's 5,313** — silently, for the one oracle where the fix had been
measured to help. Same defect class as #122 and #144. And with accessibility spread
across six oracles, a per-layer class applied to only some of them makes `0.98
accessibility percentile` mean three different things.

**Attempt 2 — one mixture at fixed N.** Gave cCRE 25% of the existing 6,000 positions.
**It made things worse.** The statistic that decides whether a percentile still
discriminates is the null *maximum*, and a maximum grows with the number of draws —
so splitting a fixed budget shortens every component's tail:

| reference set | accessibility max | tf_binding max |
|---|---|---|
| gene-anchored, 5,949 positions | 1.653 | 3.539 |
| cCRE-only, 5,986 positions | **2.754** | 3.301 |
| 25/75 mixture, 5,962 total | 1.697 | **2.937** ← below both |

Enformer TF saturation went from 25% of rows to **92%**.

**Attempt 3 — a union at doubled N.** Keep each component at full size: 6,000
gene-anchored *plus* 6,000 cCRE. Then the union's maximum is exactly
`max(max_gene, max_cCRE)`, so it is **provably never worse than the better component
for any layer** — a guarantee, not a measurement. The gene-anchored half reproduces
the previously shipped counts exactly (1,200 / 1,200 / 1,980 / 720), so the cCRE half
is purely additive.

## Result

| oracle | tracks | p99 | p99.9 | tracks wider |
|---|---|---|---|---|
| Sei | 40 | **2.05×** | 1.80× | 100% |
| EPInformer-seq | 33 | 1.38× | 1.28× | 76% |
| LegNet | 3 | 1.30× | 1.17× | 67% |
| Enformer | 5,313 | 1.26× | 1.33× | 84% |
| Borzoi | 7,611 | 1.19× | 1.19× | 82% |

At SORT1, half of Enformer's accessibility rows had a percentile pinned at exactly
1.0000. Now zero.

**LegNet got a different set on purpose.** It is a 200 bp promoter MPRA model with
`window_bp=None`, so the sampled position *is* the whole thing being modelled. The
cCRE catalogue is 62% dELS (1,469,205 of 2,348,854 distal enhancer-like) against 2%
PLS (47,532 promoter-like), and DHS summits track accessibility rather than promoter
identity — either would give a promoter model a null made mostly of enhancers. So:
TSS ±250 bp at 40%, PLS at 30%, pELS at 15%, uniform at 15%.

**Calibration held.** Strong TSS-proximal liver eQTLs (GTEx v8, tissue-matched):
AlphaGenome RNA 0.781 → **0.778**, CAGE 0.659 → **0.625**. Both in band, both 0%
saturated.

## Three of my measurements were wrong first

1. **A 400-position cCRE probe looked *narrower* than a 5,949-position gene-anchored
   build.** It was not — the maximum grows with sample count, and I was comparing
   across a 15× difference in draws.
2. **My compose guard compared global maxima across layers** (63,390 vs 87,781 = 1.39×)
   and would have refused a valid operation. The gap lived entirely in RNA rows, which
   fan out one sample per gene; the accessibility rows being swapped were 5,986 vs
   5,949. It now compares per row, over only the rows being swapped.
3. **I reported EPInformer-seq as regressing at 0.89×.** That came from comparing
   `median(new)` against `median(old)` — the median *track* of each set, which need not
   be the same track. The median of per-track ratios is **1.38×**, with 25 of 33 tracks
   wider.

All three are the same error: reaching for a summary statistic that does not mean what
I wanted it to mean. The acceptance metric is now per-track p99/p99.9, not the maximum.
The maximum is still the right thing to *report* — it is literally "percentile pinned
at 1.0" — but the wrong thing to tune on.

## Not fixed, and not claimed to be

* AlphaGenome `histone_marks` (20% of rows) and Enformer `tf_binding` (25%) keep
  whatever their better component gives. Both need a *per-track* reference population —
  that mark's own broad domains, that factor's own ChIP peaks — which is a different
  design, not a different fraction.
* Sei, LegNet and EPInformer-seq have **no positive set** — no eQTL equivalent exists
  for MPRA activity or Sei sequence classes — so they are justified by tail width and
  assay biology, not by a calibration check. That is weaker ground than the
  accessibility fix stood on.

## Verification

* 1,107 fast tests pass, zero failures.
* All 8 shipped backgrounds: monotone, zero NaN/inf, provenance readable, no `*_counts`
  forming a consecutive-integer run (#123's fingerprint).
* 13 walkthroughs + the multi-oracle path regenerated; 34 stale README numbers
  corrected; all 14 (JSON, TSV) pairs agree on counts and row identities.
* eQTL calibration gate passes for RNA and CAGE.

## Scripts

* `scripts/merge_effect_shards.py` — exact position-shard union; the CDF is built once
  from raw samples rather than by pooling shard grids.
* `scripts/apply_effect_rebuild.py` — swaps effect rows into a shipped background and
  nothing else; refuses on track-id reordering or a failed grid guard.
* `scripts/refresh_readme_background_table.py` — generates the README table from
  artefacts.
* `scripts/probe_cage_tss_null.py`, `scripts/probe_eqtl_effect_scale.py` — the
  reference-class probes.
