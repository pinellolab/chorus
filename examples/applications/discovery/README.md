# Discovery Mode Examples

Find which cell types are most affected by a variant — no prior knowledge needed.

> Ask Claude in plain language about any variant where you *don't* know
> the relevant tissue. The prompts below are concrete demonstrations so
> you can see the output format (ranked cell types + a full multi-layer
> report for each top hit); adapt them freely to your own variants.

## Example Prompts

### For a biologist

> I found a GWAS variant rs12740374 at chr1:109274968 (G>T) associated with
> LDL cholesterol, but I'm not sure which tissue is most relevant. Can you
> screen all available cell types and tell me where this variant has the
> strongest regulatory effect?

Claude will: load AlphaGenome, call `discover_variant_cell_types` to screen
~472 cell types by DNASE/ATAC effect, then run full multi-layer analysis on
the top hits. You'll get a ranked list plus detailed reports for each top
cell type.

### For a geneticist

> I have a variant of uncertain significance at chr3:46373453 (A>G). It's in
> a non-coding region and I don't know the relevant tissue. Use discovery mode
> to find the top 5 cell types where this variant changes chromatin accessibility,
> then give me the full multi-layer analysis for each. Include TF binding
> tracks if available for the top cell types.

### For a clinical researcher

> We identified a de novo non-coding variant in a patient with an undiagnosed
> condition. Position: chr7:158945632 (C>T). I need to know:
> 1. Which tissues/cell types show the strongest regulatory effect?
> 2. For the top 3, what regulatory layers are disrupted?
> 3. Are any nearby genes affected?

### For a bioinformatician

> Load alphagenome. Call discover_variant_cell_types with:
> - position: chr1:109274968
> - ref_allele: G, alt_alleles: [T]
> - top_n: 10, min_effect: 0.1
>
> Return the cell_type_ranking as JSON. For the top 3, save HTML reports.

## How it works

```
Step 1: Scout — predict variant on ALL DNASE/ATAC tracks (~472 cell types)
         ↓
Step 2: Rank — sort cell types by |log2FC| in chromatin accessibility
         ↓
Step 3: Deep dive — for top N cell types, pull all their tracks
         (CAGE, RNA, histone, TF ChIP) and build full multi-layer reports
```

### Understanding the output

The scout phase scores each cell type using **log2FC of chromatin accessibility**
in a 501bp window around the variant. This is a fast proxy for regulatory
impact — cell types with the largest chromatin changes are most likely to be
functionally affected.

The deep-dive reports for top cell types include all available layers
(chromatin, TF binding, histone marks, TSS activity). See the
[variant analysis README](../variant_analysis/) for score interpretation.

### Oracle notes

Discovery mode works best with **AlphaGenome** because it has the broadest
cell-type coverage (~472 cell types with DNASE/ATAC tracks). Other oracles
can be used but will screen fewer cell types:
- **Enformer**: ~200 cell types (ENCODE tracks)
- **Borzoi**: ~100 cell types
- **ChromBPNet**: **not compatible** with discovery mode — ChromBPNet loads
  one cell type at a time, so it cannot screen across hundreds of cell types.
  Use AlphaGenome or Enformer for discovery, then optionally follow up
  with ChromBPNet for base-resolution analysis of the top cell types.

### MCP tool call

```python
discover_variant_cell_types(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G",
    alt_alleles=["T"],
    gene_name="SORT1",    # optional
    top_n=5,               # number of cell types to analyze
    min_effect=0.15,       # minimum |log2FC| threshold
)
```

### Python API

```python
from chorus.analysis import discover_cell_types, discover_and_report

# Screen all cell types
hits = discover_cell_types(oracle, "chr1:109274968", ["G", "T"], top_n=10)

# Full reports for top hits
results = discover_and_report(
    oracle, "chr1:109274968", ["G", "T"],
    gene_name="SORT1", top_n=3,
    output_path="discovery_reports/",
)
```

## Example

### [SORT1_cell_type_screen/](SORT1_cell_type_screen/)

Screen of rs12740374 across all available cell types.

**Top 3 cell types by chromatin effect:**

| Rank | Cell Type | Effect (log2FC) | Tracks |
|------|-----------|----------------|--------|
| 1 | LNCaP clone FGC | +1.905 | 25 |
| 2 | Epithelial cell of proximal tubule | +1.623 | 22 |
| 3 | Renal cortical epithelial cell | +1.480 | 22 |

Outputs include:
- `cell_type_ranking.json` — ranked list of 100+ cell types with effect scores
- `discovery_summary.json` — structured results for top 3
- `rs12740374_SORT1_*_alphagenome_report.html` — full multi-layer HTML reports for each top cell type

**Interpreting these results**: The top hits are prostate (LNCaP) and kidney
epithelial cell types. This variant (rs12740374) is a known liver eQTL — the
fact that non-liver cell types also show strong chromatin effects demonstrates
that regulatory variants can have cross-tissue impact. The deep-dive reports
for each cell type reveal which additional layers (TF binding, histone marks,
TSS activity) are affected beyond chromatin accessibility.
