# Chorus Walkthroughs — pre-run, MCP-driven worked examples

> **"Pre-run" means the reports, not the notebooks.** Each walkthrough directory ships its
> `*_report.html`, JSON and TSV already generated — open the HTML and the answer is there, no install
> required. The `notebook.ipynb` beside it is **code-generated and ships with no outputs**: it is the
> same analysis as runnable code, for when you want to change a parameter, not a transcript of a run.
> Executing one needs the matching oracle env and, for most of them, a GPU. If you want a notebook
> that comes with its results already in it, use [`examples/notebooks/`](../notebooks/) instead.

> **These are demonstrations, not rigid templates.** Chorus is designed to
> be driven through Claude in natural language — ask in your own words
> about your own variants, cell types, or constructs, and Claude will pick
> the right tool and arguments. The concrete walkthroughs below exist so
> you can see what the outputs look like (markdown, JSON, TSV, HTML) before
> trying your own questions. Every generated report carries the original
> prompt at the top so you (or a collaborator) can tell a month later
> exactly what was asked.
>
> **Looking for the Python tutorial?** See [`../notebooks/`](../notebooks/)
> for three end-to-end Jupyter notebooks that exercise the chorus library
> directly (wild-type prediction, variant effects, region swap,
> multi-oracle comparison).
>
> **Want to reproduce a specific walkthrough as code?** Each walkthrough
> directory below also ships a `notebook.ipynb` that runs the same Python
> API the MCP tool wraps. Execute with
> `jupyter nbconvert --to notebook --execute --inplace <dir>/notebook.ipynb`
> from the repo root after `chorus setup`. Notebooks delegate to
> per-oracle conda envs via `use_environment=True`, so the base `chorus`
> env is the only kernel you need — but you do have to **register** it,
> because `chorus setup` does not:
>
> ```bash
> mamba activate chorus
> python -m ipykernel install --user --name chorus \
>     --display-name "Python 3 (chorus)"
> ```
>
> Every `notebook.ipynb` here declares kernel name `chorus`, so without
> that step `nbconvert` fails with
> `jupyter_client.kernelspec.NoSuchKernel: No such kernel named chorus`.

## Which tool do I use?

| I want to... | Tool | Example |
|---|---|---|
| Analyze a known variant in a known cell type | `analyze_variant_multilayer` | [variant_analysis/](variant_analysis/) |
| Find which cell types a variant affects | `discover_variant_cell_types` | [discovery/](discovery/) |
| Score many variants and rank by effect | `score_variant_batch` | [batch_scoring/](batch_scoring/) |
| Fine-map a GWAS locus to find the causal variant | `fine_map_causal_variant` | [causal_prioritization/](causal_prioritization/) |
| Swap a promoter/enhancer and predict effects | `analyze_region_swap` | [sequence_engineering/](sequence_engineering/) |
| Predict disruption from a construct insertion | `simulate_integration` | [sequence_engineering/](sequence_engineering/) |
| Cross-validate a variant with multiple oracles | `MultiOracleReport` | [validation/SORT1_rs12740374_multioracle/](validation/SORT1_rs12740374_multioracle/) |

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
| **Sei** | Regulatory element classification | 4 kb | — | 21,907 chromatin profiles + 40 sequence classes |
| **Cherimoya / CATv1** | Accessibility across the widest biosample set | 1 kb | 1 bp | DNASE / ATAC, 1,518 ENCODE experiments |
| **LegNet** | Promoter activity (MPRA) | 200 bp | — | MPRA activity score |
| **EPInformer-seq** | Per-cell enhancer activity | 2,114 bp | — | DNase cut-sites + H3K27ac, 11 Roadmap cells |

**Recommendation**: Start with **AlphaGenome** for the broadest coverage.
Use **ChromBPNet** as a second opinion for base-resolution motif effects.
All analysis tools work with any oracle — they automatically adapt to
the available layers.

## Understanding the scores

All tools produce **effect scores** measuring how much a variant or
modification changes a regulatory signal. Every rendered HTML report ships
with a **"How to read this report"** box at the top that defines the score
formula for every layer present in that report — so you don't need to
memorise the table below before opening an example.

| Score type | Layers | How to read it |
|-----------|--------|---------------|
| **log2FC** (`log2((alt+ε)/(ref+ε))`) | Chromatin, TF binding, Histone, TSS, Splicing | +1.0 ≈ alt signal is 2× ref; −1.0 ≈ alt is 0.5× ref |
| **lnFC** (`ln((alt+ε)/(ref+ε))`) | Gene expression (RNA) | Natural log fold change — RNA-seq convention |
| **Δ (alt − ref)** | Promoter activity (MPRA), Sei regulatory classes | Raw difference in activity (not a ratio) |

**Quick guide to log2FC magnitudes** (other layers scale similarly):
- < 0.1: Minimal — unlikely to be functional
- 0.1–0.3: Moderate — worth investigating
- 0.3–0.7: Strong — likely functional
- \> 0.7: Very strong — high-confidence regulatory effect

**Effect percentile** (when shown): compares a variant's effect against a
per-track background of ~18,000 variants sampled from the regulatory regions
that assay measures — ENCODE cCREs, DHS summits, promoters and gene features,
*not* uniformly random positions. A percentile of 0.95 means the effect is
larger than 95% of that background. The strata, and the measurements behind
them, are in [`docs/BACKGROUND_NULL_PROTOCOL.md`](../../docs/BACKGROUND_NULL_PROTOCOL.md) §3.

**Activity percentile** (when shown): ranks the reference signal at the
variant site against ~30,000 genome-wide positions including ENCODE cCREs.
A value of 0.95 means the site is already in the top 5% of regulatory
activity — a strong regulatory element.

**Provenance in summaries.** Every headline number in a report carries the
**specific track + cell type** that produced it (e.g. `+0.45, DNASE:HepG2 ·
HepG2`). Per-layer table headers also show the formula used (e.g.
`Effect log2FC`) so the unit is never ambiguous.

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
2026) to verify that Chorus produces consistent findings. Also contains a
[multi-oracle cross-validation example](validation/SORT1_rs12740374_multioracle/)
that scores the classic SORT1 variant with **three independent oracles**
(ChromBPNet for chromatin, LegNet for MPRA, AlphaGenome as generalist) and
surfaces a consensus matrix flagging where they agree — and where they
don't — on direction.

## Output Formats

Every analysis tool produces outputs in four formats:

| Format | Method | Best for |
|--------|--------|----------|
| Markdown | `report.to_markdown()` | Claude reasoning, quick review in terminal |
| JSON | `report.to_dict()` | Programmatic analysis, pipelines, notebooks |
| TSV | `report.to_tsv(path)` or `report.to_dataframe()` | Excel, R, pandas |
| HTML | `report.to_html(path)` | Visual review with embedded IGV genome browser |

> **An HTML report needs network the first time you open it.** The IGV panel resolves its reference
> sequence from `hgdownload.soe.ucsc.edu`, because igv.js requires a sequence source and hg38 is ~3 GB
> — too large to bundle. Everything else the panel needs (chromosome sizes, the ideogram, all your
> data) already travels inside the file. With no network the panel does not degrade, it never appears:
> measured `canvases 0/0 painted`, because igv.js allocates no canvases if it cannot build a genome.
>
> Two ways to close that, both opt-in:
>
> ```bash
> # (a) the real sequence, served locally -- reports look exactly as they do online
> python -m http.server -d "$(chorus config data-dir --show)/genomes" 8000
> export CHORUS_IGV_SEQUENCE_URL=http://localhost:8000/hg38.fa
>
> # (b) no server, no network: bundle a placeholder so the panel always paints
> export CHORUS_IGV_BUNDLE_SEQUENCE=1
> ```
>
> With (b) a report renders offline in ~1.6 s instead of timing out — measured 100/100 canvases
> painted with every external request blocked. The trade-off is that the sequence track shows `N`
> rather than real bases if you zoom in far enough to read them, so it is not the default. (An
> inlined FASTA is positioned from the start of the contig, and these reports sit tens of megabases
> in; showing real bases at the wrong coordinates would be worse than showing none.) A `file://` path
> does not work for (a) — a page opened from disk may not read sibling files, which is a browser rule.


## Want to add one?

Worked examples are the smallest useful contribution to chorus — one entry in a declarative list
plus one script run. The canonical step-by-step, including which regeneration script owns which
kind of example and which conda env it needs, is
[CONTRIBUTING.md § Contributing an example or walkthrough](../../CONTRIBUTING.md#contributing-an-example-or-walkthrough).
