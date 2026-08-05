# SORT1 rs12740374 — ChromBPNet Base-Resolution Analysis

## Variant: rs12740374 (chr1:109274968 G>T)

Same variant as the AlphaGenome SORT1 example, but analyzed with
ChromBPNet at **1bp resolution**. ChromBPNet provides only chromatin
accessibility (ATAC/DNASE) — no histone marks, CAGE, or RNA — but at
base-pair resolution, revealing the exact position of the effect.

## Example prompt

> I want to zoom in on rs12740374 (chr1:109274968 G>T) at base-pair
> resolution. Load ChromBPNet for DNase accessibility in HepG2 and
> analyze this variant. Does the variant create a new accessibility
> peak right at the SNP position?

## What Claude does

1. `load_oracle('chrombpnet', assay='DNASE', cell_type='HepG2')`
2. `analyze_variant_multilayer('chrombpnet', 'chr1:109274968', 'G', ['T'], ['auto'])`
3. Report shows a single chromatin accessibility layer at 1bp resolution

## Results

**Summary**: Chromatin accessibility (DNASE): very strong opening (+1.376,
effect percentile 0.9995).

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| DNASE:HepG2 | 287 | 747 | +1.376 | Very strong opening |

ChromBPNet reports a **2.6-fold increase** in local DNase accessibility at
1 bp resolution, stronger than 99.95 % of variants in its reference
population. The AlphaGenome DNASE analysis of the same variant agrees on
direction and lands within a few percent on magnitude (see
[../SORT1_rs12740374/](../SORT1_rs12740374/) for its exact figure), despite a
1 Mb receptive field against ~2 kb and a 128 bp binned sum against a
base-resolution peak.

That agreement is the point of running both: rs12740374 creates a C/EBP
binding site, and an effect that survives two independent models with
different training data, receptive fields and aggregation is far more
credible than either alone.

The report has only one layer (chromatin) because ChromBPNet is
a single-assay oracle. Compare with the
[AlphaGenome analysis](../SORT1_rs12740374/) which shows 5+ layers
for the same variant.

## When to use ChromBPNet

- **Motif disruption**: See exactly which bases are affected
- **Fast screening**: ~1s per variant (vs ~30s for AlphaGenome)
- **TF binding**: Load with `assay='CHIP', TF='CEBPA'` to check
  specific TF motif disruption at base resolution
- **Complement AlphaGenome**: Use ChromBPNet for the detailed local
  view, AlphaGenome for the broad multi-layer context

## When AlphaGenome and ChromBPNet *can* disagree

They agree at this locus. But they can diverge,
sometimes in sign, and it is worth knowing why before trusting either in
isolation. Three reasons:

1. **Different training data.** AlphaGenome's DNASE:HepG2 track
   summarises ENCODE DNase-seq with a smoothing kernel over ~128 bp
   bins. ChromBPNet's ATAC:HepG2 is a bias-corrected profile fit to a
   single ENCODE ATAC-seq experiment at 1 bp resolution. DNase and Tn5
   have different cut-site biases, each handled differently by the two
   models.

2. **Different receptive fields.** AlphaGenome uses a 1 Mb window;
   ChromBPNet uses ~2 kb. For a variant whose effect depends on
   long-range enhancer–promoter contact (rs12740374 sits in a SORT1
   liver enhancer ~30 kb from the TSS), broader context can change the
   predicted direction.

3. **Effect aggregation.** AlphaGenome's reported effect is a binned
   sum over ~128 bp; ChromBPNet's is the peak height at the variant
   itself. A motif-shift of 1–2 bp can raise the local peak
   (ChromBPNet opening) while redistributing signal across the wider
   window (AlphaGenome neutral or opposite).

An earlier version of this page reported an ATAC run at −0.111 and built
this section around explaining a contradiction with AlphaGenome. That
contradiction was an artefact of stale prose: the committed run is DNase,
not ATAC, and both oracles open strongly. The regeneration scripts rewrite
`example_output.{json,md,tsv}` and the HTML but have never touched these
READMEs, so every correctness fix left the narrative behind.
`tests/test_walkthrough_readmes_match_artefacts.py` now fails on any number
here that its own artefact does not contain.

**Practical rule.** Agreement across both oracles ≈ a strong, robust
signal. Disagreement is informative — usually the effect is either
base-resolution-local (trust ChromBPNet) or long-range-context
(trust AlphaGenome), not that one is wrong.
