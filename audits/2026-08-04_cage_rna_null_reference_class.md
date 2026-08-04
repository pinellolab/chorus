# Closing out the RNA and CAGE null models — what four reference classes measure

*2026-08-04. Measured, not argued: every number below comes from a run on this box,
and the scripts are in `scripts/`.*

## The question

CAGE is a TSS-localised peak, so "random genomic variants" is the wrong reference
class for it. The direction was to use annotated TSSs instead. Taken literally that
is a clean principle — no invented mixture fractions — so it was worth measuring
before spending ~11 h rebuilding on it.

The positive set throughout is **strong TSS-proximal liver eQTLs** from GTEx v8
(`|slope| >= 0.5`, `maf >= 0.05`, `p <= 1e-10`), scored in tissue-matched liver CAGE
tracks. If a null cannot rank *these* as notable, it cannot rank anything.

## Four reference classes

| CAGE effect null | eQTL percentile p10 / **p50** / p90 | saturated |
|---|---|---|
| uniform-random positions (shipped before) | 0.24 / **0.857** / 0.98 | 0 % |
| gene-anchored mixture (shipped now) | 0.15 / **0.659** / 0.92 | 0 % |
| **all annotated TSSs, variant AT the TSS** | 0.05 / **0.323** / 0.69 | 0 % |
| **all annotated TSSs, offsets drawn from real eQTL `tss_distance`** | 0.18 / **0.713** / 0.96 | 3.0 % |

`scripts/probe_cage_tss_null.py`, 300 TSSs × 2 alleles each, 4 liver CAGE tracks.

### The literal reading over-corrects, and it is worth being precise about why

Placing the variant exactly at the annotated TSS puts it at the **peak maximum**.
With a `+1` pseudocount, a high-signal window is barely damped, so `log2((a+1)/(r+1))`
approaches the true fractional change; a low-signal window is damped heavily toward
zero. High-signal positions therefore produce systematically larger effects. Measured:
the TSS-only null's median |effect| is **0.036–0.039**, while a strong eQTL's median
is **0.020** — the null's typical variant perturbs CAGE nearly twice as much as a
validated eQTL does.

The consequence is that a strong, experimentally-confirmed eQTL reports at the **32nd
percentile** — below median. That is the mirror image of the failure this whole
exercise set out to fix, and it would be worse than what shipped, because a
below-median number reads as "nothing here."

### The same principle, applied at realistic distances, is the best of the four

Real variants are not at the peak apex. Anchoring on all annotated protein-coding
TSSs but drawing the variant's offset from the empirical `tss_distance` distribution
of significant liver eQTLs (within ±10 kb) gives **p50 0.713** — above the currently
shipped gene-anchored mixture (0.659), and with no invented 40/33/12/15 fractions.

So the direction was right in kind. Only the "exactly at the TSS" implementation of
it is wrong, and the fix is one line in the sampler: keep the TSS anchors, jitter the
offsets.

Cost against the honest caveat: 3.0 % of eQTL rows saturate at 1.0 under this null,
against 0 % under the mixture. That is a small price and, unlike the 0.323, it errs
in the direction that does not hide signal.

## RNA needed no new reference class at all

| RNA effect null | eQTL percentile p10 / **p50** / p90 | saturated |
|---|---|---|
| uniform-random positions (shipped before) | 0.58 / **0.899** / 0.995 | 0 % |
| gene-anchored mixture (shipped now) | 0.59 / **0.781** / 0.969 | 0 % |

An eQTL-anchored RNA null was considered and is **not needed**. The saturation that
motivated it was computed from effect magnitudes (0.05 / 0.5 / 5.0) that were
invented rather than measured; the real distribution of strong-eQTL RNA effects has
median 0.0008 and max 0.031 against a null max of 0.0225. The committed examples'
alarming `+0.718` RNA figures were artefacts of the #149 denominator bug, which
overstated by 251–1736×. The mass was already in the right place; the numerator was
wrong.

Related correction: `frac < 0.1` is not a usable degeneracy metric, because 99.9 % of
*real eQTL effects* also fall below 0.1.

## What the flagship example now says, and why it is coherent

`variant_analysis/SORT1_rs12740374`, before and after the rebuild:

| layer | |raw| p50 | q p50 before → after | saturated before → after |
|---|---|---|---|
| chromatin_accessibility | 1.033 | 0.999 → 0.998 | 0 → 0 |
| tf_binding | 3.131 | 1.000 → 1.000 | **2 → 0** |
| histone_marks | 1.251 | 0.999 → 1.000 | 0 → 0 |
| tss_activity (CAGE) | 0.0048 | 0.749 → **0.460** | 1 → 0 |

rs12740374 creates a CEBPB site. The report now reads: strong accessibility and TF
binding effects near the top of their nulls, a modest gene-expression effect
(`gene_expression` q 0.977 → 0.869 in the CEBP walkthrough), and **little change in
TSS activity itself** (CAGE at the 46th percentile off a raw effect of 0.0048). That
is the biologically right shape for a variant that creates a TF footprint rather than
relocating a promoter — and it is only legible now that CAGE is no longer measured
against a null of mostly-zero-effect random positions.

Across the three regenerated AlphaGenome walkthroughs, saturated rows fell from
**19 to 3** and distinct percentile values rose in every layer.

## One thing that did not improve, stated plainly

Enformer `chromatin_accessibility` at SORT1 went from 4/12 saturated to **6/12**, and
its median percentile from 0.960 to 1.000. This is not a new regression: 0.960 was
the *artefact ceiling* (0.9605) from the padded grid (#143), so those rows were
already pinned — the repair only made the pinning visible instead of disguising it as
a plausible-looking 0.96. The underlying fact is that Enformer's accessibility effect
null is genuinely too narrow for a variant this strong. It clears the release gate
(7 distinct values across 12 rows, so the column is not constant) but it is the next
thing to look at, and it should not be described as fixed.

## Recommendation

1. Rebuild the CAGE/PROCAP effect rows against **TSS-anchored positions with
   eQTL-matched offsets** (p50 0.713). One sampler change; rides on the same forward
   passes as every other layer, so the marginal cost is the passes themselves.
2. Leave RNA on the gene-anchored mixture (p50 0.781). No eQTL null.
3. Keep the region mixture for every other layer.
4. Do not ship the "variant exactly at the TSS" null.

## Scripts

* `scripts/probe_cage_tss_null.py` — the four-way comparison above; `--offset-like-eqtl LIVER`
  selects the jittered variant.
* `scripts/probe_eqtl_effect_scale.py` — `--layer RNA_SEQ|CAGE`, `--max-tss-distance`.
* `scripts/gate_end_to_end_determinism.py` — two-process bitwise report diff.
* `scripts/stamp_background_provenance.py` — as-built `build_config`, establishable facts only.
