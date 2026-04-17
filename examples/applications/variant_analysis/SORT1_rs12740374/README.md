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

| Track | Effect (log2FC) | Effect %ile | Interpretation |
|-------|----------------|------------|----------------|
| DNASE:HepG2 | +0.449 | ≥99th | Strong chromatin opening |
| CHIP:CEBPA:HepG2 | +0.376 | ≥99th | Strong TF binding gain |
| CHIP:CEBPB:HepG2 | +0.274 | ≥99th | Moderate TF binding gain |
| CHIP:H3K27ac:HepG2 | +0.178 | ≥99th | Moderate enhancer activation |
| CAGE:HepG2 (variant site) | +0.253 | ≥99th | Moderate transcription increase |

> **Why all `≥99th`?** Each effect percentile is computed against
> ~10,000 random SNPs; Chorus collapses the top bucket to `≥99th`
> rather than rendering a spurious `99.3rd` / `99.7th` / `99.9th`
> gradient in a CDF tail that doesn't have enough samples to
> meaningfully distinguish them.

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
