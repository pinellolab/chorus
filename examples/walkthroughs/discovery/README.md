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
| 1 | HepG2 | +1.334 | 562 |
| 2 | MCF 10A | +1.440 | 6 |
| 3 | amniotic epithelial cell | +2.898 | 3 |

Ranked by `alt × |effect|`, **not** by log2FC — which is why the order does not
follow the effect column. Weighting by the alternate-allele signal favours a site
that is both responsive and actually active, so HepG2 leads despite the smallest
fold change of the three, and it carries **562** tracks against 3–6 for the
others.

Outputs include:
- `discovery_summary.json` — structured results for the top 3, including the
  ranking metric and the best track per cell type
- `example_output.{json,md,tsv}` — all scored rows across the screened cell types
- `chr1_109274968_G_T_SORT1_alphagenome_<cell_type>_report.html` — full
  multi-layer HTML report for each of the top 3

**Interpreting these results**: the strongest chromatin response is in **HepG2**,
a liver line — which is the expected answer, since rs12740374 is a known liver
eQTL acting through a C/EBP site. That the screen recovers it without being told
to look at liver is the point of running a discovery sweep at all. The two
non-liver hits (amniotic epithelial cell, MCF 10A) show larger raw log2FC on
3 and 6 tracks respectively; treat those as leads rather than conclusions, both
because the evidence is thinner and because a large fold change on a low-signal
track is easy to come by. The deep-dive reports show which additional layers
(TF binding, histone marks, TSS activity) move beyond chromatin accessibility.
