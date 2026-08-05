# SORT1 rs12740374 — Multi-Layer Variant Analysis

## Variant: rs12740374 (chr1:109274968 G>T)

The top GWAS variant for LDL cholesterol at the 1p13.3 locus.
Musunuru et al. (2010, Nature) showed this SNP creates a C/EBP
transcription-factor binding site in a liver enhancer, upregulating
SORT1 expression and lowering LDL.

This is the recommended starting example — it demonstrates the
full multi-layer analysis across chromatin, TF binding, histone marks,
CAGE, and RNA using AlphaGenome.

## Example prompt

> Analyze variant rs12740374 (chr1:109274968 G>T) using AlphaGenome.
> Focus on HepG2 liver cells with DNASE, CEBPA, CEBPB, H3K27ac,
> and CAGE tracks. Gene is SORT1. Does this variant create a new
> C/EBP binding site as published?

## What Claude does

1. `load_oracle('alphagenome')`
2. `analyze_variant_multilayer('alphagenome', 'chr1:109274968', 'G', ['T'], assay_ids=[...HepG2 tracks...])`
3. Generates a multi-layer report with per-track raw scores, percentiles, and IGV browser

## Key results

The AlphaGenome analysis reproduces the published biology:

| Track | Effect (log2FC) | Effect %ile | Activity %ile | Interpretation |
|-------|----------------|------------|---------------|----------------|
| CHIP:CEBPB:HepG2 | +2.981 | ≥99th | 0.977 | Very strong TF binding gain |
| CHIP:CEBPA:HepG2 | +2.626 | ≥99th | 0.991 | Very strong TF binding gain |
| CAGE:HepG2 (variant site) | +1.502 | ≥99th | 0.916 | Very strong transcription increase |
| DNASE:HepG2 | +1.332 | ≥99th | 0.973 | Very strong chromatin opening |
| CHIP:H3K27ac:HepG2 | +1.251 | ≥99th | 0.999 | Very strong enhancer activation |
| ATAC:HepG2 | +0.730 | ≥99th | 0.935 | Very strong chromatin opening |

> **Why all `≥99th`?** Two separate reasons, and only the first is a
> display choice.
>
> Chorus collapses the top bucket to `≥99th` rather than rendering a
> `99.3rd` / `99.7th` / `99.9th` gradient it cannot support. **But for
> this variant the percentile is not doing the work — the effects
> themselves are large.** A `+2.98` log2FC on CEBPB is an 8-fold
> binding gain; the `≥99th` is a consequence, not the evidence.
>
> Read the **Effect** column first and the percentile second. The
> percentile is a weak discriminator on this oracle: AlphaGenome's
> per-track effect background is built from **1,697–1,909** random
> genomic positions with a random alternate allele (**not** ~10,000
> common SNPs, and not gnomAD — no code samples gnomAD). For
> `DNASE:HepG2`, **95.1%** of that background falls below
> `|log2FC| = 0.1`, with a median of `0.0126`. So almost *any*
> non-trivial effect clears the 99th percentile, and a high percentile
> beside a small effect means very little. Tracking as
> [#83](https://github.com/pinellolab/chorus/issues/83).
>
> Earlier revisions of this file reported `+0.449` for `DNASE:HepG2`
> and explained the uniform `≥99th` as purely a display choice against
> "~10,000 random SNPs". Both were wrong: the effects were suppressed
> by a variant-window bug (fixed in
> [#92](https://github.com/pinellolab/chorus/pull/92)) and the
> percentiles were inflated by a denominator bug (fixed in
> [#119](https://github.com/pinellolab/chorus/pull/119)).

## Output files

| File | Description |
|------|-------------|
| `example_output.md` | Markdown report with all scored tracks |
| `example_output.json` | Machine-readable per-track scores |
| `example_output.tsv` | Tab-separated summary |
| `rs12740374_SORT1_alphagenome_report.html` | Interactive IGV browser report |

## See also

- [SORT1 ChromBPNet](../SORT1_chrombpnet/) — same variant at 1bp resolution
- [SORT1 Enformer](../SORT1_enformer/) — same variant with Enformer oracle
- [SORT1 Validation](../../validation/SORT1_rs12740374_with_CEBP/) — extended validation with CEBP tracks
