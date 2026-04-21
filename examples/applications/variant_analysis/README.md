# Variant Analysis Examples

Hypothesis-driven multi-layer variant analysis using `analyze_variant_multilayer`.
Each example shows a real GWAS/clinical variant scored across regulatory layers.

> The prompts below are **demonstrations, not templates**. Ask Claude
> about *your own* variant in plain language (rsID, coordinates, cell
> type, target gene), and the right tool call happens automatically.
> These concrete examples exist so you can see what the outputs look
> like (MD / JSON / TSV / HTML) before running your own questions.

## Example Prompts

### For a biologist

> I'm studying the SORT1 locus and cardiovascular risk. Can you analyze
> the effect of rs12740374 (chr1:109274968 G>T) on gene regulation in
> liver cells? I want to understand if this variant changes chromatin
> accessibility, TF binding, histone marks, or SORT1 expression.

Claude will: load AlphaGenome, search for HepG2 tracks (DNASE, CEBPA ChIP,
H3K27ac, CAGE, RNA), call `analyze_variant_multilayer`, and explain the
results in biological terms.

### For a geneticist

> I have a GWAS hit at chr2:60490908 (G>T) associated with fetal
> hemoglobin levels. The nearest gene is BCL11A. Analyze this variant
> in erythroid cells — I need to know which regulatory layer is disrupted
> and whether it's consistent with the known enhancer mechanism.
> Include GATA1 and TAL1 ChIP tracks if available.

### For a bioinformatician

> Load alphagenome. Run analyze_variant_multilayer with:
> - position: chr1:109274968
> - ref_allele: G, alt_alleles: [T]
> - assay_ids: [DNASE/EFO:0001187 DNase-seq/.,
>   CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA/.,
>   CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.,
>   CAGE/EFO:0001187 CAGE plus strand,
>   CAGE/EFO:0001187 CAGE minus strand]
> - gene_name: SORT1
>
> Return the raw JSON and save the HTML report.

### For an MD / clinical researcher

> A patient has a VUS at chr2:60490908 G>T (rs1427407) in the BCL11A
> erythroid enhancer. Can you show which regulatory layer is affected
> in K562 cells — chromatin accessibility, GATA1/TAL1 binding, histone
> marks, or BCL11A expression? I need to understand the mechanism to
> interpret the patient's elevated fetal hemoglobin.

## Track selection tips

For comprehensive multi-layer analysis, include tracks from as many layers
as possible:

| Layer | Track types to request | What it tells you |
|-------|----------------------|-------------------|
| Chromatin | DNASE, ATAC | Is the variant in open/closed chromatin? |
| TF binding | ChIP-TF (e.g. CEBPA, GATA1, CTCF) | Does it disrupt a TF motif? |
| Histone marks | H3K27ac, H3K4me3, H3K4me1 | Active enhancer/promoter? |
| TSS activity | CAGE | Does it change transcription initiation? |
| Gene expression | RNA-seq | Does it change mRNA levels? |

**Don't forget TF ChIP tracks** — these are often the most informative for
understanding mechanism. Search with `list_tracks('alphagenome', query='CEBPA')`
or ask Claude to find relevant TFs for your cell type.

## Understanding the scores

### Effect scores

Each layer uses a different scoring formula:

| Layer | Formula | Example | Meaning |
|-------|---------|---------|---------|
| Chromatin, TF, Histone, TSS | log2FC | +0.5 | Alt signal is 2^0.5 = 1.41x the ref signal |
| Gene expression (RNA) | lnFC (natural log) | +0.7 | Alt expression is e^0.7 ≈ 2.0× the ref |
| Promoter activity (MPRA) | difference | +2.0 | Alt is 2.0 units higher than ref |

**Rule of thumb for log2FC scores:**
- |effect| < 0.1 → Minimal (< 7% change)
- |effect| 0.1–0.3 → Moderate (7–23% change)
- |effect| 0.3–0.7 → Strong (23–62% change)
- |effect| > 0.7 → Very strong (> 62% change)

### Quantile scores (when available)

When quantile normalization is applied, scores are compared against a
background distribution of effects seen across many variants:
- **Quantile 0.86** means this effect is larger than 86% of background variants
- Range is [0, 1] for unsigned layers (chromatin, TF, histone, TSS, splicing)
- Range is [-1, 1] for signed layers (gene expression, MPRA)

### Reference signal percentile (Ref %ile)

When baseline backgrounds are available, each track also shows the **reference
signal activity percentile** — how active this region is genome-wide:
- **Ref %ile 0.91** means the reference signal at this locus is higher than
  91% of genomic positions for this track
- A variant with a high effect quantile (0.95) on a track with a high ref
  percentile (0.91) is disrupting a highly active regulatory element — strong
  evidence for functional impact
- A high effect on a low-activity region (ref %ile < 0.2) may be less
  biologically meaningful

### Background availability

Quantile and percentile scores appear automatically when pre-computed
backgrounds exist in `~/.chorus/backgrounds/`. These are built once per
oracle by the `scripts/build_backgrounds_*.py` scripts and auto-loaded
when an oracle is loaded via MCP or the Python API.

Backgrounds are available for all 6 oracles: AlphaGenome, Enformer, Borzoi,
ChromBPNet, Sei, and LegNet. They are hosted on HuggingFace at
[lucapinello/chorus-backgrounds](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
and **downloaded automatically** when an oracle is loaded. No manual setup needed.

To rebuild after adding new tracks or models, run `scripts/build_backgrounds_<oracle>.py`.

## Oracle compatibility

Different oracles support different track types:

| Oracle | Chromatin | TF binding | Histone | TSS (CAGE) | RNA-seq | MPRA | Window |
|--------|-----------|-----------|---------|------------|---------|------|--------|
| **AlphaGenome** | DNASE, ATAC | ChIP-TF | ChIP-Histone | CAGE, PRO-CAP | RNA | — | 1 Mb |
| **Enformer** | DNASE, ATAC | ChIP-TF | ChIP-Histone | CAGE | — | — | 114 kb |
| **Borzoi** | DNASE, ATAC | ChIP-TF | ChIP-Histone | CAGE | RNA | — | 196 kb |
| **ChromBPNet** | DNASE, ATAC | ChIP-TF | — | — | — | — | 1 kb |
| **Sei** | — | — | — | — | — | — | 4 kb |
| **LegNet** | — | — | — | — | — | MPRA | 200 bp |

**Choosing an oracle:**
- **AlphaGenome** (recommended): broadest coverage, 1bp resolution, all layers
- **Enformer**: good general-purpose, no RNA but has CAGE for TSS activity
- **Borzoi**: best for distal gene expression (196kb window, has RNA)
- **ChromBPNet/BPNet**: best for base-resolution motif disruption (1bp, DNASE/ATAC or ChIP-TF)
- Use multiple oracles for cross-validation (see below)

### Using ChromBPNet

ChromBPNet loads one model per assay/cell-type pair (unlike AlphaGenome which
loads all tracks at once):

```
# Chromatin accessibility at base resolution
load_oracle('chrombpnet', assay='ATAC', cell_type='K562')

# TF binding — specify which TF
load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='GATA1')
```

ChromBPNet generates its own track IDs automatically, so you can pass any
`assay_ids` list (e.g. `["auto"]`) — the model will return tracks named
`ATAC:K562` or `CHIP:K562:GATA1:+/CHIP:K562:GATA1:-`.

The report will show only chromatin accessibility (for ATAC/DNASE) or
TF binding (for CHIP). This is ideal for zooming into the variant site
at single-base resolution.

### Multi-oracle workflow

For deep characterization, combine multiple oracles. Each call to
`analyze_variant_multilayer` uses one oracle — run separate calls and compare:

**Example: Full characterization of rs12740374**

> 1. Load AlphaGenome and run analyze_variant_multilayer with HepG2 DNASE,
>    CEBPA ChIP, H3K27ac, CAGE, and RNA tracks. Gene is SORT1.
>    This gives the broad multi-layer view.
>
> 2. Now load ChromBPNet (assay=ATAC, cell_type=HepG2) and run
>    analyze_variant_multilayer on the same variant.
>    This gives base-resolution chromatin accessibility.
>
> 3. Load ChromBPNet (assay=CHIP, cell_type=K562, TF=CEBPA) and run
>    analyze_variant_multilayer on the same variant.
>    This shows whether the variant creates or destroys a C/EBP motif.
>
> 4. Compare the results across all three. Do they agree?

Claude can keep all three oracles loaded simultaneously and run each analysis
in sequence.

## What happens under the hood

1. **Load oracle**: `load_oracle('alphagenome')` — loads the model (~2 min)
2. **Find tracks**: `list_tracks('alphagenome', query='HepG2')` — finds DNASE, CAGE, ChIP, RNA tracks
3. **Analyze**: `analyze_variant_multilayer(oracle_name, position, ref, alts, assay_ids, gene_name)`
4. **Score**: each track scored with modality-specific strategy (window, aggregation, formula)
5. **Report**: structured output with scores organized by regulatory layer

## Examples

### [SORT1_rs12740374/](SORT1_rs12740374/)
**rs12740374** (chr1:109274968 G>T) — GWAS variant for LDL cholesterol.
Creates a C/EBP binding site in a liver-specific enhancer, increasing
SORT1 expression. Analyzed in HepG2 with quantile normalization.

Key findings: Strong chromatin opening (+0.447 log2FC, quantile 0.86),
strong TSS activity increase, moderate gene expression change.

### [BCL11A_rs1427407/](BCL11A_rs1427407/)
**rs1427407** (chr2:60490908 G>T) — GWAS variant for fetal hemoglobin.
Disrupts a GATA1/TAL1 motif in the erythroid enhancer. Therapeutic target
of Casgevy (first FDA-approved CRISPR therapy for sickle cell disease).

Key finding: Modest effect in K562 — demonstrates why cell-type selection
matters (primary erythroid progenitors would show stronger effects).

### [FTO_rs1421085/](FTO_rs1421085/)
**rs1421085** (chr16:53767042 T>C) — Obesity-associated variant in FTO locus.
Alters ARID5B binding, affecting IRX3/IRX5 expression ~500kb away.

Key finding: Subtle local effects — this variant acts over very long range,
illustrating the limits and strengths of different oracle window sizes.

### [SORT1_chrombpnet/](SORT1_chrombpnet/)
**rs12740374 via ChromBPNet** — Same SORT1 variant analyzed at **1bp resolution**
with ChromBPNet (ATAC accessibility only). Shows how ChromBPNet reports
contain a single layer. Useful for base-resolution motif analysis.

Key finding: Strong opening (+0.44 log2FC) at base resolution, consistent
with AlphaGenome results.

### [SORT1_enformer/](SORT1_enformer/)
**rs12740374 via Enformer** — Same SORT1 variant analyzed with Enformer
(chromatin, TF, histone, CAGE — **no RNA**). Demonstrates how reports
automatically adapt when a layer is unavailable.

Key finding: Very strong CTCF binding loss (-0.89), very strong chromatin
opening (+0.86), strong TSS increase (+0.66). Cross-oracle comparison table
included.
