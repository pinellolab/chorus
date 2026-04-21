# SORT1 rs12740374 — ChromBPNet Base-Resolution Analysis

## Variant: rs12740374 (chr1:109274968 G>T)

Same variant as the AlphaGenome SORT1 example, but analyzed with
ChromBPNet at **1bp resolution**. ChromBPNet provides only chromatin
accessibility (ATAC/DNASE) — no histone marks, CAGE, or RNA — but at
base-pair resolution, revealing the exact position of the effect.

## Example prompt

> I want to zoom in on rs12740374 (chr1:109274968 G>T) at base-pair
> resolution. Load ChromBPNet for ATAC accessibility in HepG2 and
> analyze this variant. Does the variant create a new accessibility
> peak right at the SNP position?

## What Claude does

1. `load_oracle('chrombpnet', assay='ATAC', cell_type='HepG2')`
2. `analyze_variant_multilayer('chrombpnet', 'chr1:109274968', 'G', ['T'], ['auto'])`
3. Report shows a single chromatin accessibility layer at 1bp resolution

## Results

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate closing (-0.11).

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| ATAC:HepG2 | 687 | 636 | -0.111 | Moderate closing |

ChromBPNet reports moderate closing (-0.11 log2FC) at 1bp resolution
in HepG2 ATAC. **The AlphaGenome DNASE analysis of the same variant
shows strong opening (+0.45)** — opposite direction. See the
"Why AlphaGenome DNASE and ChromBPNet ATAC can disagree" section
below for the biological explanation. This is a legitimate divergence,
not a bug: DNase-seq and ATAC-seq measure chromatin accessibility
differently, and the two models use different window sizes and
aggregation strategies.

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

## Why AlphaGenome DNASE and ChromBPNet ATAC can disagree

Both oracles report chromatin accessibility at the same locus in HepG2,
but raw effects can diverge (sometimes in sign). Three reasons:

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

**Practical rule.** Agreement across both oracles ≈ a strong, robust
signal. Disagreement is informative — usually the effect is either
base-resolution-local (trust ChromBPNet) or long-range-context
(trust AlphaGenome), not that one is wrong.
