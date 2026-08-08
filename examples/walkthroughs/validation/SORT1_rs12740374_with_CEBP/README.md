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
Values are raw log2 fold-changes, with the percentile against each track's own
background in brackets:

- CEBPA binding gain: **+2.945** (0.9998)
- CEBPB binding gain: **+3.316** (0.9995)
- DNASE opening: **+1.334** (0.9964)
- H3K27ac activation: **+1.251** (0.9992)

Two things worth reading carefully. **CEBPA ranks higher than CEBPB (0.9998 against
0.9995) despite a smaller raw effect** — each is ranked against its own track's
background, and those differ; percentiles are comparable within a track, not across
tracks. And **nothing here is pinned**: 0 of 246 scored rows sit at a clamped 1.0. Before
the 2026-08 rebuild, CEBPA at this locus exceeded its null's maximum and reported 1.0
with no ranking information — see
[`docs/BACKGROUND_NULL_PROTOCOL.md`](../../../../docs/BACKGROUND_NULL_PROTOCOL.md) §10.

Multi-layer convergence in the same direction provides strong evidence
that this is indeed the causal regulatory variant.

## See also

- [SORT1 variant analysis](../../variant_analysis/SORT1_rs12740374/) — focused 6-track analysis
- [SORT1 causal prioritization](../../causal_prioritization/SORT1_locus/) — fine-mapping with 11 LD variants
- [SORT1 multi-oracle validation](../SORT1_rs12740374_multioracle/) — cross-validate the same variant with ChromBPNet + LegNet + AlphaGenome
