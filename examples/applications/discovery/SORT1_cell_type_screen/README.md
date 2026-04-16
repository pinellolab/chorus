# Cell-Type Discovery Mode

## Concept

When you have a variant of unknown function, you don't know which cell type
to analyze it in. Discovery mode solves this by **screening all available
cell types** in a single prediction, ranking them by effect magnitude, then
running full multi-layer analysis on the top hits.

## How it works

1. **Scout phase**: Predict the variant effect on all DNASE/ATAC tracks
   across ~472 cell types in a single AlphaGenome forward pass
2. **Rank**: Sort cell types by |log2FC| in chromatin accessibility
3. **Deep dive**: For the top N cell types, pull in all their tracks
   (CAGE, RNA-seq, histone marks, TF ChIP) and build full reports

## Example: rs12740374 at the SORT1 locus

This variant is a known liver eQTL for SORT1 (cardiovascular risk).
Discovery mode reveals which cell types show the strongest chromatin
response — potentially uncovering effects in unexpected cell types.

## MCP Tool

```
discover_variant_cell_types(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G",
    alt_alleles=["T"],
    gene_name="SORT1",
    top_n=5,
)
```

## Python API

```python
from chorus.analysis import discover_cell_types, discover_and_report

# Step 1: Screen all cell types
hits = discover_cell_types(
    oracle,
    variant_position="chr1:109274968",
    alleles=["G", "T"],
    top_n=10,
    min_effect=0.2,
)

# Step 2: Full reports for top hits
results = discover_and_report(
    oracle,
    variant_position="chr1:109274968",
    alleles=["G", "T"],
    gene_name="SORT1",
    top_n=3,
    output_path="discovery_reports/",
)
```
