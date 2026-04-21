# Region Swap Example: SORT1 Enhancer Replacement in K562

## Scenario

Replace the SORT1 liver enhancer (chr1:109274500-109275500) with a 630 bp
GFP/reporter construct sequence. This simulates what happens when a
regulatory region is disrupted by a large transgene insertion — a common
CRISPR knock-in or reporter-assay scenario.

## Example prompt

> Replace the SORT1 enhancer region chr1:109274500-109275500 with a
> 630 bp GFP/reporter construct sequence and predict effects on K562
> DNASE, H3K27ac, H3K4me3, and CAGE.

## What Claude does

1. Calls `load_oracle('alphagenome')`
2. Calls `analyze_region_swap(oracle_name='alphagenome', region='chr1:109274500-109275500', replacement_sequence='...', assay_ids=[...K562 tracks...])`
3. Returns a multi-layer report comparing wild-type vs replacement

## Results summary

| Layer | Effect | Interpretation |
|-------|--------|----------------|
| Chromatin (DNASE:K562) | −3.289 log2FC | Very strong closing |
| Histone H3K27ac:K562 | −1.350 log2FC | Very strong mark loss |
| Histone H3K4me3:K562 | −1.032 log2FC | Very strong mark loss |
| CAGE — GSTM2 TSS | −8.110 log2FC | Very strong decrease |
| CAGE — GNAI3 TSS | −7.380 log2FC | Very strong decrease |

Replacing a functional enhancer with arbitrary reporter sequence
produces coherent loss across all regulatory layers: chromatin closes,
active histone marks disappear, and nearby TSS activity collapses.
This is the expected behavior — the enhancer was doing real work, and
the replacement sequence doesn't carry the same regulatory elements.

## Interpretation

This example demonstrates how Chorus can predict the functional
consequences of large sequence edits. The dramatic signal loss in
cis-regulatory tracks (DNASE, H3K27ac) and neighbouring-TSS CAGE
confirms that this region is a bona-fide active enhancer in K562 and
that disrupting it would silence nearby gene expression — useful as
an in-silico negative control before committing to an expensive
wet-lab knock-in.

## Output files

- `region_swap_SORT1_K562_report.html` — interactive HTML report with IGV browser
- `example_output.md` — markdown table for quick reading
- `example_output.json` — structured per-track scores
- `example_output.tsv` — tab-separated summary
