# Chorus Application Examples

Each subfolder contains worked examples with outputs in all three formats
(markdown, JSON, HTML). Copy any example prompt below into Claude to reproduce
the analysis or adapt it for your own variants.

## Which tool do I use?

| I want to... | Tool | Category |
|--------------|------|----------|
| Analyze a known variant in a known cell type | `analyze_variant_multilayer` | [variant_analysis/](variant_analysis/) |
| Find which cell types a variant affects | `discover_variant_cell_types` | [discovery/](discovery/) |
| Swap a promoter/enhancer and predict effects | `analyze_region_swap` | [sequence_engineering/](sequence_engineering/) |
| Predict disruption from a construct insertion | `simulate_integration` | [sequence_engineering/](sequence_engineering/) |
| Score many variants and rank by effect | `score_variant_batch` | [batch_scoring/](batch_scoring/) |
| Fine-map a GWAS locus to find the causal variant | `fine_map_causal_variant` | [causal_prioritization/](causal_prioritization/) |

## Quick start by role

**Biologist / Geneticist** — Start with [variant_analysis/](variant_analysis/).
Paste a prompt like: *"Load AlphaGenome and analyze rs12740374 (chr1:109274968 G>T)
in HepG2 cells. Use DNASE, CEBPA ChIP, H3K27ac, CAGE, and RNA tracks. Gene is SORT1."*

**Bioinformatician** — Start with [batch_scoring/](batch_scoring/) if you have
a VCF, or [variant_analysis/](variant_analysis/) for single variants. All outputs
are available as JSON, TSV, and pandas DataFrames for downstream pipelines.

**Computational biologist** — See the [scoring strategies](variant_analysis/SORT1_rs12740374/multilayer_variant_analysis.md)
for details on per-layer window sizes, aggregation, and effect formulas.
The [variant analysis README](variant_analysis/) has a full oracle compatibility table.

**MD / Clinical researcher** — Start with [discovery/](discovery/) if you don't
know the relevant tissue, or [variant_analysis/](variant_analysis/) if you do.
The HTML reports provide visual summaries with color-coded effect badges.

## Understanding the scores

All tools produce **effect scores** measuring how much a variant/modification
changes a regulatory signal. The formula depends on the layer:

| Score type | Layers | How to read it |
|-----------|--------|---------------|
| **log2FC** | Chromatin, TF binding, Histone, TSS | +1.0 means alt is 2x ref; -1.0 means alt is 0.5x ref |
| **logFC** | Gene expression (RNA) | +0.7 means alt is ~2x ref expression |
| **diff** | Promoter activity (MPRA) | Simple difference in activity units |

**Quick guide to magnitudes (log2FC):**
- < 0.1: Minimal — unlikely to be functional
- 0.1–0.3: Moderate — worth investigating
- 0.3–0.7: Strong — likely functional
- \> 0.7: Very strong — high-confidence regulatory effect

**Quantile scores** (when shown): compare a variant's effect against thousands
of background variants. A quantile of 0.86 means the effect is larger than 86%
of random variants — strong evidence for functionality.

## Which oracle should I use?

| Oracle | Best for | Layers available | Window |
|--------|----------|-----------------|--------|
| **AlphaGenome** | Comprehensive analysis | All 7 layers (DNASE, ChIP-TF, ChIP-Histone, CAGE, RNA, PRO-CAP, splicing) | 1 Mb |
| **Enformer** | General-purpose | Chromatin, TF, Histone, CAGE (no RNA) | 114 kb |
| **Borzoi** | Distal gene expression | Chromatin, TF, Histone, CAGE, RNA | 196 kb |
| **ChromBPNet** | Motif-resolution TF binding | Chromatin, TF only (DNASE/ATAC, ChIP-TF) | 1 kb |
| **Sei** | Regulatory element classification | Sequence class only | 4 kb |
| **LegNet** | Promoter activity (MPRA) | MPRA only | 230 bp |

All analysis tools work with any oracle — they automatically adapt to the
available track types. If an oracle doesn't support a layer (e.g., Enformer
has no RNA), that layer is simply omitted from the report.

### Loading ChromBPNet

ChromBPNet requires specifying the assay and cell type at load time:

```
load_oracle('chrombpnet', assay='ATAC', cell_type='K562')           # chromatin accessibility
load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='GATA1')  # TF binding
```

ChromBPNet generates its own track IDs (e.g. `ATAC:K562`, `CHIP:K562:GATA1:+`),
so you do not need to specify `assay_ids` — just pass any placeholder list.

### Combining oracles

You can load multiple oracles simultaneously and run separate analyses with
each. This is the recommended approach for deep variant characterization:

```
1. Load AlphaGenome → analyze_variant_multilayer (broad multi-layer view)
2. Load ChromBPNet (ATAC, K562) → analyze_variant_multilayer (base-resolution chromatin)
3. Load ChromBPNet (CHIP, K562, TF=GATA1) → analyze_variant_multilayer (motif disruption)
4. Compare results across oracles
```

## Categories

### [variant_analysis/](variant_analysis/)
Hypothesis-driven analysis of known variants in specific cell types.
Four worked examples: SORT1, BCL11A, FTO, TERT.

### [discovery/](discovery/)
Discovery mode — screen hundreds of cell types to find where a variant
has the strongest impact, then run full multi-layer analysis.

### [sequence_engineering/](sequence_engineering/)
Region swap and integration simulation — predict effects of sequence
modifications on local regulatory activity.

### [batch_scoring/](batch_scoring/)
Score multiple variants from a VCF or variant list and rank by effect magnitude.

### [causal_prioritization/](causal_prioritization/)
Fine-map a GWAS locus — score all LD variants and identify the most likely
causal variant using multi-layer convergence evidence. Includes locus plot.

### [validation/](validation/)
Replication of key examples from the AlphaGenome Nature paper (Avsec et al.
2026) to verify that Chorus produces consistent findings. Includes SORT1
with C/EBP tracks, TERT in melanocytes, and HBG2 HPFH variant.

## Output Formats

| Format | Method | Best for |
|--------|--------|----------|
| Markdown | `report.to_markdown()` | Claude reasoning, quick review in terminal |
| JSON | `report.to_dict()` | Programmatic analysis, pipelines, notebooks |
| HTML | `report.to_html(path)` | Visual review, sharing with collaborators |
| DataFrame | `report.to_dataframe()` | pandas/R analysis, filtering, plotting |
| TSV | `result.to_tsv(path)` | Batch scoring: Excel, R, command-line tools |
