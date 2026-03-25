# Region Swap Example: Promoter Replacement in K562

## Scenario

Replace a weak promoter region (chr1:1000500-1001500) with a stronger
synthetic promoter sequence to boost gene expression. This simulates
a common promoter engineering experiment.

## Example prompt

> I want to replace the promoter region at chr1:1000500-1001500 with a
> stronger promoter sequence. Predict how this swap changes chromatin
> accessibility, histone marks, and TSS activity in K562 cells using
> AlphaGenome.

## What Claude does

1. Calls `load_oracle('alphagenome')`
2. Calls `list_tracks('alphagenome', query='K562')` to find DNASE, CAGE, H3K27ac, H3K4me3
3. Calls `analyze_region_swap(oracle_name='alphagenome', region='chr1:1000500-1001500', replacement_sequence='...', assay_ids=[...])`
4. Returns a multi-layer report comparing wild-type vs replacement

## Results summary

| Layer | Effect | Interpretation |
|-------|--------|----------------|
| Chromatin (DNASE) | +1.804 log2FC | Very strong opening |
| Histone H3K27ac | +0.454 log2FC | Strong mark gain |
| Histone H3K4me3 | +0.822 log2FC | Very strong mark gain |
| TSS (CAGE) variant site | +2.318 log2FC | Very strong increase |
| TSS (CAGE) ISG15 TSS | +1.977 log2FC | Very strong increase |

The replacement promoter dramatically increases chromatin accessibility,
active histone marks, and transcription initiation — consistent with
inserting a strong promoter in place of a weaker one.

## Output files

- `promoter_swap_K562_alphagenome_report.html` — interactive HTML report with color-coded tables
- `example_output.md` — markdown table for quick reading
- `example_output.json` — structured JSON for programmatic analysis
