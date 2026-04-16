# SORT1 rs12740374 — Validation with CEBP Binding

## Variant: rs12740374 (chr1:109274968 G>T)

Validation example reproducing the key finding from Musunuru et al. (2010,
*Nature*): rs12740374 creates a C/EBP transcription factor binding site
in a liver enhancer. This example uses AlphaGenome's full track catalog
for HepG2 to verify the CEBPA/CEBPB binding gain and correlated chromatin/
expression changes.

## Example prompt

> Validate the Musunuru 2010 finding for rs12740374. Score this variant
> across ALL HepG2 tracks in AlphaGenome to verify CEBP binding gain,
> chromatin opening, and downstream expression changes. Gene is SORT1.

## Key results

The AlphaGenome prediction reproduces the published mechanism:
- CEBPA binding gain: +0.37 (strong)
- CEBPB binding gain: +0.22 (moderate)
- DNASE opening: +0.43 (strong)
- H3K27ac activation: +0.18 (moderate)

Multi-layer convergence in the same direction provides strong evidence
that this is indeed the causal regulatory variant.

## See also

- [SORT1 variant analysis](../../variant_analysis/SORT1_rs12740374/) — focused 6-track analysis
- [SORT1 causal prioritization](../../causal_prioritization/SORT1_locus/) — fine-mapping with 11 LD variants
