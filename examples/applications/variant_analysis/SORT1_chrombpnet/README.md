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

**Summary**: Chromatin accessibility (DNASE/ATAC): strong opening (+0.44).

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| ATAC:HepG2 | 676 | 918 | +0.441 | Strong opening |

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
