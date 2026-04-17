# Chorus Application Examples

> **These are demonstrations, not rigid templates.** Chorus is designed to
> be driven through Claude in natural language — ask in your own words
> about your own variants, cell types, or constructs, and Claude will pick
> the right tool and arguments. The concrete examples below exist so you
> can see what the outputs look like (markdown, JSON, TSV, HTML) before
> trying your own questions. Every generated report carries the original
> prompt at the top so you (or a collaborator) can tell a month later
> exactly what was asked.

## Which tool do I use?

| I want to... | Tool | Example |
|---|---|---|
| Analyze a known variant in a known cell type | `analyze_variant_multilayer` | [variant_analysis/](variant_analysis/) |
| Find which cell types a variant affects | `discover_variant_cell_types` | [discovery/](discovery/) |
| Score many variants and rank by effect | `score_variant_batch` | [batch_scoring/](batch_scoring/) |
| Fine-map a GWAS locus to find the causal variant | `fine_map_causal_variant` | [causal_prioritization/](causal_prioritization/) |
| Swap a promoter/enhancer and predict effects | `analyze_region_swap` | [sequence_engineering/](sequence_engineering/) |
| Predict disruption from a construct insertion | `simulate_integration` | [sequence_engineering/](sequence_engineering/) |

## Quick start by role

**Biologist / Geneticist** — Start with [variant_analysis/](variant_analysis/).
Example prompt: *"Analyze rs12740374 (chr1:109274968 G>T) in HepG2 cells
using DNASE, CEBPA ChIP, H3K27ac, and CAGE tracks. Gene is SORT1."*

**Bioinformatician** — Start with [batch_scoring/](batch_scoring/) if you have
a VCF or variant list. All outputs are available as JSON, TSV, and pandas
DataFrames for downstream pipelines.

**Computational biologist** — See the [scoring strategies](variant_analysis/SORT1_rs12740374/multilayer_variant_analysis.md)
for details on per-layer window sizes, aggregation, and effect formulas.

**MD / Clinical researcher** — Start with [discovery/](discovery/) if you don't
know the relevant tissue, or [variant_analysis/](variant_analysis/) if you do.
The HTML reports provide visual summaries with an embedded genome browser.

## Which oracle should I use?

| Oracle | Best for | Output window | Resolution | Key layers |
|--------|----------|--------------|------------|------------|
| **AlphaGenome** | Comprehensive multi-layer analysis | 1 Mb | 1 bp | DNASE, ChIP-TF, ChIP-Histone, CAGE, RNA, PRO-CAP, splicing |
| **Enformer** | General-purpose, lightweight | 114 kb | 128 bp | DNASE, ChIP-TF, ChIP-Histone, CAGE |
| **Borzoi** | Distal gene expression effects | 197 kb | 32 bp | DNASE, ChIP-TF, ChIP-Histone, CAGE, RNA |
| **ChromBPNet** | Base-resolution motif disruption | 1 kb | 1 bp | DNASE/ATAC or ChIP-TF (one assay per model) |
| **Sei** | Regulatory element classification | 4 kb | — | 40 sequence classes |
| **LegNet** | Promoter activity (MPRA) | 230 bp | — | MPRA activity score |

**Recommendation**: Start with **AlphaGenome** for the broadest coverage.
Use **ChromBPNet** as a second opinion for base-resolution motif effects.
All analysis tools work with any oracle — they automatically adapt to
the available layers.

## Understanding the scores

All tools produce **effect scores** measuring how much a variant or
modification changes a regulatory signal:

| Score type | Layers | How to read it |
|-----------|--------|---------------|
| **log2FC** | Chromatin, TF binding, Histone, TSS | +1.0 = alt is 2x ref; -1.0 = alt is 0.5x ref |
| **logFC** | Gene expression (RNA) | +0.7 = alt is ~2x ref expression |
| **diff** | Promoter activity (MPRA) | Simple difference in activity units |

**Quick guide to magnitudes (log2FC):**
- < 0.1: Minimal — unlikely to be functional
- 0.1–0.3: Moderate — worth investigating
- 0.3–0.7: Strong — likely functional
- \> 0.7: Very strong — high-confidence regulatory effect

**Effect percentile** (when shown): compares a variant's effect against
~10,000 random SNPs scored on the same oracle. A percentile of 0.95 means
the effect is larger than 95% of random variants.

**Activity percentile** (when shown): ranks the reference signal at the
variant site against ~30,000 genome-wide positions including ENCODE cCREs.
A value of 0.95 means the site is already in the top 5% of regulatory
activity — a strong regulatory element.

## Categories

### [variant_analysis/](variant_analysis/)
Hypothesis-driven analysis of known variants in specific cell types.
Four worked examples: SORT1 (HepG2 liver), BCL11A (K562 erythroid),
FTO (metabolic), TERT promoter (K562).

### [discovery/](discovery/)
Discovery mode — screen hundreds of cell types to find where a variant
has the strongest impact, then run full multi-layer analysis.

### [batch_scoring/](batch_scoring/)
Score multiple variants from a VCF or variant list and rank by effect.
Output: per-track columns showing raw + percentile for each assay.

### [causal_prioritization/](causal_prioritization/)
Fine-map a GWAS locus — score all LD variants across specific tracks
and identify the most likely causal variant using multi-layer convergence.

### [sequence_engineering/](sequence_engineering/)
Region swap and integration simulation — predict effects of sequence
modifications on local regulatory activity.

### [validation/](validation/)
Replication of key examples from the AlphaGenome Nature paper (Avsec et al.
2026) to verify that Chorus produces consistent findings.

## Output Formats

Every analysis tool produces outputs in four formats:

| Format | Method | Best for |
|--------|--------|----------|
| Markdown | `report.to_markdown()` | Claude reasoning, quick review in terminal |
| JSON | `report.to_dict()` | Programmatic analysis, pipelines, notebooks |
| TSV | `report.to_tsv(path)` or `report.to_dataframe()` | Excel, R, pandas |
| HTML | `report.to_html(path)` | Visual review with embedded IGV genome browser |
