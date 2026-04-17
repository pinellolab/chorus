# SORT1 rs12740374 — Enformer Multi-Layer Analysis

## Variant: rs12740374 (chr1:109274968 G>T)

Same variant as the AlphaGenome SORT1 example, but analyzed with
**Enformer**. Enformer has a 114kb output window and supports chromatin,
TF binding, histone marks, and CAGE — but **not RNA-seq**. This example
shows how reports automatically adapt when a layer is unavailable.

## Example prompt

> Load Enformer and analyze rs12740374 (chr1:109274968 G>T) in HepG2
> cells. Use DNASE, CTCF ChIP, H3K27ac, and CAGE tracks. Compare the
> results with what AlphaGenome showed for the same variant.

## What Claude does

1. `load_oracle('enformer')`
2. `list_tracks('enformer', query='HepG2')` — finds DNASE, CTCF, H3K27ac, CAGE tracks
3. `analyze_variant_multilayer('enformer', 'chr1:109274968', 'G', ['T'], assay_ids, gene_name='SORT1')`
4. Report shows 4 layers (no RNA section — Enformer doesn't have RNA tracks)

## Results

**Summary**: TF binding: very strong binding loss (-0.89); Chromatin:
very strong opening (+0.86); TSS activity: strong increase (+0.66);
Histone marks: moderate mark gain (+0.20).

| Layer | Top Effect | Interpretation |
|-------|-----------|----------------|
| TF binding (CTCF) | -0.892 | Very strong binding loss |
| Chromatin (DNASE) | +0.864 | Very strong opening |
| TSS (CAGE) | +0.661 | Strong increase |
| Histone (H3K27ac) | +0.205 | Moderate mark gain |

**Key observation**: The report automatically omits the Gene expression
(RNA-seq) section because Enformer doesn't support RNA tracks. The
remaining 4 layers tell a coherent story — the variant opens chromatin,
disrupts CTCF binding, increases TSS activity, and modestly gains
H3K27ac marks.

## Cross-oracle comparison

| Layer | AlphaGenome | Enformer | ChromBPNet |
|-------|-------------|----------|------------|
| Chromatin | +0.447 | +0.864 | +0.441 |
| TF binding | — | -0.892 (CTCF) | — |
| Histone | +0.178 | +0.205 | — |
| TSS (CAGE) | +0.425 | +0.661 | — |
| RNA | +0.07 | *not available* | — |

Each oracle captures a different aspect. Combining all three gives the
most complete picture.
