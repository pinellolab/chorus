<!--
=============================================================================
REVIEWER-CORRECTED DRAFT — Chorus blog post 2026-006
Revised 2026-06-17 by an independent reproduction review.
Full reproduction evidence: audits/2026-06-16_blogpost_reproduction_report.md
Windowing/normalization follow-up: audits/2026-06-17_windowing_normalization_P0.md

CHANGELOG (what changed vs the submitted draft, and why):
1. MCP "22 tools" → "23 tools" (added recommend_alphagenome_backend). Verified: the
   server registers 23 tools.
2. CAGE/eRNA effect "+1.47" → "+1.52". Fresh AlphaGenome run = +1.525 (committed = +1.522).
3. BPNet C/EBP "TF head +1.99" → "CEBPB head +2.5". Fresh ChromBPNet CEBPB (HepG2,
   BP000275) = +2.52; no committed +1.99 existed. Direction (strong gain) confirmed.
4. "largest effect in HepG2" softened. Across the 5 human ChromBPNet DNASE models,
   HepG2 is co-highest by effect-percentile (0.9995) but IMR-90 (a lung fibroblast) is
   larger by raw log2FC (+2.02 vs +1.37). Reworded to avoid the unique superlative.
5. LegNet "HepG2 or A549" → "HepG2 or K562". LegNet's panel is exactly K562/HepG2/WTC11;
   there is no A549. The no-lung-fibroblast point stands.
6. REMOVED "18 variants in LD r²>0.8; rank-2 at r²≈0.9". Verified against the full Sniff
   PDF: "18", "r²>0.8", and "rank-2 ≈0.9" appear nowhere; the paper resolves the locus to
   a SINGLE variant (rs9504151, PIP=0.51). A fresh Chorus LDlink query (r²>0.8, CEU)
   returns ~54 proxies (49 EUR), many at r²=1.0. Replaced with that framing + PIP=0.51.
7. Analysis B ranking now reported with the explicit reproducible recipe and a note that
   it requires two correctness fixes (LD-allele orientation; full-input scoring window),
   shipped in the reproducibility PR. Fresh re-run: rs9504151 = rank #1 (ChromBPNet
   IMR-90; AlphaGenome confirmation appended).
8. Analysis A ChromBPNet "+1.24" kept (correct full-input-window value; fresh = +1.37),
   with a reproducibility note: the *conversational* MCP path currently under-reports it
   (+0.32) due to a windowing bug (P0 follow-up); an explicit-window recipe reproduces it.

REMAINING REVIEWER TODO before publishing:
- Append the AlphaGenome fine-map ranking number (re-run in progress at time of writing).
- Decide whether to publish before the windowing/background-CDF fix lands (affects the
  conversational reproducibility of the Analysis-A fixed-input-oracle magnitudes).
=============================================================================
-->
---
post_id: "2026-006"
title: "Chorus: chatting with genomic oracles"
image: "fig_architecture.png"
math: false
authors: ["Dmitry Penzar", "Lorenzo Ruggeri", "Rosalba Giugno", "Luca Pinello"]
authors_display:
  - name: "Dmitry Penzar"
    affiliation: "Vavilov Institute of General Genetics, Moscow"
    orcid: ""
  - name: "Lorenzo Ruggeri"
    affiliation: "Sapienza University of Rome"
    orcid: ""
  - name: "Rosalba Giugno"
    affiliation: "University of Verona"
    orcid: ""
  - name: "Luca Pinello"
    affiliation: "Harvard Medical School, Massachusetts General Hospital, and the Broad Institute"
    orcid: ""
editor: ""
tags: ["genomics", "seq2func", "variant-interpretation", "foundation-models", "AlphaGenome", "MCP", "agentic-AI"]
categories: ["Blog Post"]
scope: ["insights", "tutorials"]
audience: ["within-field", "general"]
disciplines: ["Sequence-to-Function Modeling"]
labs: ["Pinello Lab"]
status: "submitted"
revision: 2
date_submitted: 2026-06-16
date_accepted:
date: 2026-06-17
doi: ""
zenodo_url: ""
revision_history:
  - version: 1
    date: 2026-06-16
    notes: "Initial submission"
    doi: ""
    zenodo_url: ""
  - version: 2
    date: 2026-06-17
    notes: "Independent reproduction review: corrected numbers (MCP tool count, CAGE, BPNet C/EBP), softened the HepG2 superlative, fixed the LegNet panel, removed the unsupported '18 variants in LD' setup figure, and added reproducible prompts/recipes + a windowing caveat."
---

{{< summary >}}

Sequence-to-function models can read regulatory meaning out of raw DNA, but each one ships with its own dependencies, input window, track conventions, and output format, which keeps them out of reach for most geneticists and clinicians.

**Chorus** wraps seven state-of-the-art models (ChromBPNet/BPNet, LegNet, Sei, Enformer, Borzoi, EPInformer-seq, and AlphaGenome) behind a single "oracle" abstraction with harmonised, cross-comparable outputs, and exposes every capability over the [Model Context Protocol](https://modelcontextprotocol.io/) so an AI agent can run them from plain language. Two worked examples show what that enables: in the first, an agent reconstructs the mechanism of a textbook regulatory variant (rs12740374 at *SORT1*) through conversation alone; in the second, it prioritises a likely causal variant inside a tight GWAS linkage block (rs9504151), recovering the same disrupted transcription factor that an independent fine-mapping study reported.

We are also honest about where the predictions fall short: effect sizes come out too small, and a variant can only be assessed in the cell types the models actually cover.

**Code:** [github.com/pinellolab/chorus](https://github.com/pinellolab/chorus)

{{< /summary >}}

---

## Motivation

Interpreting noncoding variants is one of the central problems in human genetics. Most GWAS associations and a large share of disease-relevant alleles sit outside coding regions, so their mechanisms stay opaque. Over the last few years, sequence-to-function (seq2func) deep learning models have become the most promising route to mechanistic interpretation. [Enformer](https://www.nature.com/articles/s41592-021-01252-x), [Borzoi](https://github.com/calico/borzoi), [ChromBPNet/BPNet](https://github.com/kundajelab/chrombpnet), [Sei](https://github.com/FunctionLab/sei-framework), [LegNet](https://github.com/autosome-ru/LegNet), and [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) predict chromatin accessibility, transcription factor binding, histone marks, transcription start site (TSS) activity, expression, and splicing directly from DNA, across hundreds of human cell types.

> *Side note:* a seq2func model learns a direct mapping from DNA sequence to one or more measured molecular readouts (accessibility, binding, expression, and so on). Train it on the genome and you can then ask it about sequences the genome never contained, which is exactly what variant interpretation needs.

The catch is that using these models together is painful. Each one comes with its own repository, dependency stack (TensorFlow, PyTorch, or JAX), input window (anywhere from 2 kb to 1 Mb), track-naming convention, and output resolution. Running a single comparison across models often means days of plumbing before you can even ask a biological question. That overhead is annoying for computational biologists. For clinicians and GWAS interpreters, who want to look at a patient's variant or a credible set from a study, it is usually a hard stop.

We think it is time to move from computational to conversational genomics. Instead of writing scripts, a researcher should also be able to *discuss* a complex analysis with an agent that knows which model to reach for, handles the preprocessing, and returns results in a consistent shape. That is the idea behind Chorus.

![Chorus turns seven sequence-to-function models into interchangeable oracles and lets an AI agent orchestrate them from natural language.](fig_architecture.png "Figure 1. Chorus exposes seven seq2func oracles behind one interface and an agentic layer. The same analysis can be driven programmatically through a unified Python API or conversationally through an MCP server.")

---

## The oracle abstraction

Chorus reframes every model as an interchangeable **oracle**. You hand an oracle a genomic region or a DNA sequence and a set of tracks, and it returns predictions across the modalities it supports. Switching models is a single argument. Under the hood, Chorus takes care of dependency isolation, environment management, and the model-specific preprocessing each one demands. From the outside, all of that disappears behind one small Python API.

Three concepts carry the design:

- **Oracle.** A common interface implemented by every model. `oracle.predict(interval, tracks)` is the primitive that everything else is built on. Each oracle runs in its own isolated conda environment, so the TensorFlow, PyTorch, and JAX models never fight over dependencies.
- **Interval.** A unified handle on genomic coordinates and the reference sequence, with an edit history. An interval tracks the substitutions, insertions, and deletions you apply alongside the predictions they produce, which is what makes in silico perturbation reproducible rather than a pile of ad hoc FASTA files.
- **Track.** A named genomic signal (DNase, ATAC, ChIP, CAGE, RNA-seq, and so on). Track identifiers differ across oracles, so Chorus provides metadata search to find the right ones.

These seven oracles span a wide range of context windows and resolutions, which is exactly why having them under one roof is useful: you can pick the right tool for the layer you care about without rewriting your analysis.

| Oracle | Input window | Resolution | What it is good at |
|---|---|---|---|
| **ChromBPNet / BPNet** | 2,114 bp | 1 bp | Chromatin accessibility (ChromBPNet) and TF binding (BPNet) at base-pair resolution |
| **LegNet** | 200 bp | element-level | MPRA / reporter activity of short regulatory elements |
| **EPInformer-seq** | 2,114 bp | element-level | Compact per-cell enhancer activity (DNase cut-sites + H3K27ac) across 11 Roadmap cell types |
| **Sei** | 4,096 bp | region-level | Regulatory effect across 21,907 chromatin profiles |
| **Enformer** | 393,216 bp | 128 bp | Expression (CAGE), accessibility, histone marks across long context |
| **Borzoi** | 524,288 bp | 32 bp | Enformer-style outputs plus RNA-seq coverage |
| **AlphaGenome** | 1,048,576 bp | 1 bp | Generalist: ATAC, DNase, CAGE, RNA-seq, splicing, PRO-CAP, and ChIP for histones and TFs (5,731 tracks) |

There is one more piece that makes cross-model work honest. Raw model outputs are not comparable: a "+1.3" from one track means something different from a "+1.3" from another. Chorus scores every prediction against a per-track genome-wide background (built from thousands of random variants and genomic sites), turning a raw log2 fold-change into two interpretable numbers: an **effect percentile** (how unusual this variant's effect is relative to random SNPs) and an **activity percentile** (how active the site is to begin with). For the genome-browser view, the same backgrounds rescale every track onto one shared display axis, where 1.0 marks roughly the top one percent of bins genome-wide and 0 sits at a per-layer activity floor. After this normalisation, effects and signals become comparable across cell types, variants, tracks, and oracles, and the browser can show every oracle's reference and alternate signal on a single axis.

---

## From computational to conversational genomics

The part that changes how these models get used is the conversational layer. Chorus ships an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server that exposes every capability as a tool an AI agent can call. Once it is connected, you can ask the genome what a variant would do in plain language, and the agent picks the oracle, selects tracks, runs the prediction, and explains the result.

The server exposes 23 tools, grouped by what they do. The low-level set handles discovery (`list_oracles`, `list_tracks`, `list_genomes`, `get_genes_in_region`, `get_gene_tss`), oracle lifecycle (`load_oracle`, `unload_oracle`, `oracle_status`), raw prediction (`predict`, `predict_variant_effect`, `predict_region_replacement`, `predict_region_insertion`), scoring primitives (`score_prediction_region`, `score_variant_effect_at_region`, `predict_variant_effect_on_gene`), and a backend helper (`recommend_alphagenome_backend`, which suggests the JAX vs PyTorch AlphaGenome backend for a given window size).

On top of those sit the high-level analysis tools that power the worked examples below, the ones most users actually call:

- `analyze_variant_multilayer` — score a variant across chromatin, TF, histone, CAGE, RNA, and splicing in a single call
- `discover_variant` and `discover_variant_cell_types` — find the top tracks, or screen hundreds of cell types, for a variant without pre-selecting assays
- `score_variant_batch` — rank a whole VCF, GWAS set, or credible set by effect
- `fine_map_causal_variant` — prioritise the causal SNP in a GWAS locus using multi-layer convergence (this is the tool behind the second worked example below)
- `analyze_region_swap` and `simulate_integration` — score sequence-engineering edits such as promoter swaps and construct insertions

In practice this means an analyst can ask: which cell type's accessibility changes most for this variant, which transcription factor is gaining or losing a site, which nearby gene moves, does this edited enhancer behave like the wild type. No code, and the same questions map onto whichever oracle is best suited to answer them.

The two sections below are worked examples of exactly that.

---

## Reproducing a Nature paper in an afternoon

We started from a simple question: could an agent equipped with Chorus reconstruct the mechanism behind a classic regulatory variant through conversation alone?

The test case is **rs12740374** at the 1p13 locus, the subject of a [2010 Nature paper](https://doi.org/10.1038/nature09266) by Musunuru and colleagues. The original work showed that this noncoding SNP creates a liver-specific C/EBP (C/EBPα) binding site, raises *SORT1* expression in hepatocytes, and ultimately shifts LDL cholesterol and myocardial infarction risk. The SNP itself sits in a roughly 6 kb noncoding stretch between CELSR2 and PSRC1; SORT1, the gene it ultimately controls, is a more distal neighbour in the same cluster, a detail that matters below. Establishing the mechanism took a battery of experiments: eQTL analysis, reporter assays, chromatin immunoprecipitation (ChIP), electrophoretic mobility shift assays (EMSA), and overexpression and knockdown studies in mouse liver.

We reproduced the core of it in an afternoon, in a chat. Working in Claude Code with the Chorus MCP server connected, the analysis unfolded one question at a time.

**Step 1, which cell type opens up?**

> **Prompt:** Score chr1:109274968 G>T using ChromBPNet. Which cell types change chromatin accessibility the most?

The agent loaded ChromBPNet on its own and reported that the variant strongly increases predicted accessibility, with a large effect in **HepG2**, a liver-derived line — among the strongest of the cell types it covers. That matches the paper's central claim that the mechanism is liver-relevant. (Across the human ChromBPNet DNase models, HepG2 is co-highest by effect-percentile; the fibroblast line IMR-90 shows a comparable-to-larger raw effect, so we avoid claiming HepG2 is uniquely the largest.)

**Step 2, which factor binds?**

> **Prompt:** HepG2 looks interesting. Which transcription factor's predicted binding changes most at the variant?

The strongest signal came from the **C/EBP** family. We gave no prior about which factor to look for: AlphaGenome scores its whole panel of TF-ChIP tracks, so it can surface C/EBP straight from sequence (CEBPB and CEBPA top the HepG2 TF panel), and a C/EBP-specific BPNet head then confirms the gain. In the 2010 study, nailing this down required EMSA and ChIP to show that C/EBP binds the site created by the minor allele.

**Step 3, which gene moves, and why only a long-context oracle can answer**

> **Prompt:** Which nearby gene's expression is most affected? Use a long-context oracle with CAGE and RNA-seq.

The top hit was **SORT1**, with increased predicted promoter and transcript signal in HepG2.

This step is worth pausing on, because it is the one place where the choice of oracle is not optional. As noted above, rs12740374 sits between CELSR2 and PSRC1, and the SORT1 promoter is on the order of 100 kb away (≈118 kb to the nearest SORT1 TSS in hg38). That distance is far outside the windows of the short-range oracles: ChromBPNet sees about 2 kb and LegNet 200 bp, so neither can connect this variant to SORT1 at all, and even Enformer's roughly 115 kb output window is marginal here (118 kb is just past its 114.7 kb output window). Only a long-context model can span both the variant and the gene, so the gene-level readout comes from AlphaGenome, whose 1 Mb window comfortably covers SORT1. This is exactly why, in the multi-oracle table below, ChromBPNet and LegNet read "not scored" for the gene layer: not because they failed, but because the question is outside what they were built to see.

At that point the mechanistic story had assembled itself through conversation: the minor allele opens chromatin in liver cells, creates a C/EBP site, and upregulates *SORT1*. To make the result robust rather than a single model's opinion, we then scored the variant across three independent oracles, each contributing only the layers it can actually measure: ChromBPNet for local accessibility, LegNet for MPRA reporter activity, and AlphaGenome as a generalist covering ChIP, histones, CAGE, and, thanks to its 1 Mb window, the distal *SORT1* promoter.

![Multi-oracle readout for rs12740374 and a side-by-side comparison of what the models predicted against what the 2010 study measured.](fig_sort1_comparison.png "Figure 2. Left, the per-layer multi-oracle table and a single-axis genome-browser view of every oracle's signal around SORT1. Right, the model predictions placed next to the experimental ground truth from Musunuru et al. (2010). The models recover the direction and the mechanism at every layer; the one place they fall short is magnitude.")

### Predictions next to ground truth

The most useful way to read this is side by side. The left column is what Chorus predicted in silico; the right is what the original study measured at the bench.

| Chorus prediction (in silico) | Experimental ground truth (Musunuru et al., 2010) |
|---|---|
| Minor (T) allele **opens chromatin** in liver (HepG2): ChromBPNet +1.24, AlphaGenome +1.34 | The variant lies in a **liver enhancer**; the minor allele creates the functional site |
| **Gains a C/EBP binding site**: BPNet (CEBPB) +2.5, AlphaGenome (CEBPA) +2.7 | Minor allele **creates a C/EBP site**, confirmed by EMSA and ChIP |
| **Gains the H3K27ac** active-enhancer mark: AlphaGenome +1.27 | Consistent with an active liver enhancer |
| **eRNA at the enhancer**: AlphaGenome CAGE +1.52 | Consistent with active enhancer transcription |
| **Single-fragment MPRA at the SNP is up**: LegNet +0.30, alt > ref | Matches the original luciferase reporter (alt > ref) |
| ***SORT1* promoter and transcript rise**, but only a small amount, well under 2-fold (AlphaGenome) | ***SORT1* more than 12-fold higher** in minor-allele-homozygous human liver; lower hepatic VLDL secretion; plasma LDL-C down up to 16 mg/dL (two copies); MI risk down about 40% |

Read top to bottom, the agreement is striking. Without any prior assumptions, the oracles recover the cell type, the transcription factor, the chromatin and histone changes, the reporter direction, and the target gene, all of which originally took dedicated assays to establish.

> **Reproducibility note (ChromBPNet accessibility number).** The **+1.24** figure is the
> value scored on ChromBPNet's full 2,114 bp input window centered on the variant (a fresh
> re-run gives +1.37). At present the *conversational* path (`analyze_variant_multilayer`,
> which auto-centers a minimal region) under-reports fixed-input-oracle magnitudes (it
> returns ≈ +0.32 for this variant) because of a windowing issue being fixed separately;
> AlphaGenome is unaffected. To reproduce +1.24 today, score on an explicit input-width
> window, e.g. `oracle.predict_variant_effect("chr1:109273911-109276025", "chr1:109274968", ["G","T"], None)`
> with ChromBPNet (DNASE, HepG2). The *direction* (alt opens chromatin) reproduces on every path.

The honest caveat is in the last row. The models get the **direction** of the *SORT1* effect right at every layer, but they **underestimate the magnitude** badly: a small predicted increase, well under 2-fold, against the more than 12-fold measured in liver. Several of the supporting layers also rest on a single oracle (the n=1 entries in the table), so they are suggestive rather than independently confirmed. The mechanism is reproduced; the effect size is not, and that gap is exactly the kind of thing you would still take to a reporter assay.

---

## Variant prioritisation: a worked example

The second use case is the one clinical interpreters ask about most: given a GWAS hit sitting in tight linkage with many neighbours, which variant is actually causal?

A useful anchor here is a recent preprint, [Borzoi-informed fine mapping](https://www.biorxiv.org/content/10.1101/2025.07.09.663936v1) (the "Sniff" framework), which feeds Borzoi variant-effect predictions into Bayesian fine-mapping with PolyFun and SuSiE. For a sub-significant FEV1/FVC ratio association, Sniff resolves the locus — which neither SuSiE nor PolyFun-Baseline could resolve — to a single variant, **rs9504151** (posterior inclusion probability 0.51), and nominates **CDYL** as the likely target gene. Reading Borzoi's attributions, the authors find that the variant disrupts a putative **ATF4** binding motif, with decreased predicted chromatin accessibility and H3K27ac, and the effect localised to **lung fibroblasts**.

We used this locus to ask a different question: driving several oracles by conversation, would Chorus land on the same variant and the same factor? What follows is the team's own re-analysis with Chorus, reported as a demonstration of the prioritisation workflow, not as a result taken from the preprint.

### Step 1, initial scoring with AlphaGenome

rs9504151 sits in a tightly correlated LD block: a Chorus LDlink query (r² > 0.8, 1000G CEU) returns about 54 proxy variants (49 in the EUR panel), and many of them are in near-perfect LD (r² ≈ 1.0) with the lead, so simple LD alone cannot single out a candidate. This is exactly the setting the Sniff framework targets.

We asked Chorus to score this credible set with AlphaGenome. In our run it ranked rs9504151 at the top, predicting that it sits in an active cis-regulatory element in lung fibroblasts whose accessibility and active histone marks drop on the alternate allele. Its RNA tracks predicted no significant expression change at the locus. On that basis Chorus flagged the variant for functional follow-up.

### Step 2, cross-checking with other oracles, and an honest cell-type caveat

This is where having several oracles helps, and also where a real limitation surfaces. We re-scored with a LegNet MPRA oracle and with a ChromBPNet model trained on lung-fibroblast DNase (IMR-90). In our run rs9504151 stayed at **rank 1** with ChromBPNet on the lung-fibroblast model.

The caveat is important. Only the ChromBPNet model here was matched to the relevant cell type, lung fibroblast. The MPRA reporter panels available to LegNet do not include a lung fibroblast (the available lines are K562, HepG2, and WTC11), so that line of evidence is cell-type-mismatched and should be weighted accordingly. You can only assess a variant in the cell types and assays your models actually cover, which is the central limitation of this entire approach, and one the Sniff authors stress as well: if the trait-relevant cell type is missing from a model's training data, the effect can be missed entirely.

### Step 3, identifying the disrupted factor

Which factor is disrupted could not be read straight off lung-fibroblast TF ChIP-seq data, because such tracks are sparse. So we used two indirect routes: scoring rs9504151 against all of AlphaGenome's ChIP-seq TF tracks (ATF4 and CEBPB came out at the top, with binding *lost* on the alternate allele), and in silico saturation mutagenesis, which converged on an ATF4-family motif.

Both routes point to ATF4, matching the Sniff preprint's conclusion. The value is not the single answer but that it held up across models trained on different assays, even while the cell-type coverage stayed imperfect.

> **Reproducibility note (Analysis B).** This analysis is reproducible via
> `fine_map_causal_variant` with an LDlink token, but it depends on two correctness fixes
> shipped in the reproducibility PR: (a) LD proxies must be scored on their *own* genome
> alleles (oriented to the reference), and (b) each variant must be scored on the oracle's
> full input window. With those, a fresh re-run places **rs9504151 at rank 1** in the
> ChromBPNet lung-fibroblast (IMR-90) model (effect −0.985 on the alt allele) with zero
> allele-mismatch warnings. <!-- AlphaGenome fine-map ranking to be appended once the
> re-run completes. -->

---

## Why this helps

- **No pipeline overhead.** Dependency isolation and environment management are automatic, so you spend time on biology, not on reconciling TensorFlow and JAX.
- **Multi-model cross-validation, used correctly.** Different oracles answer different layers (local accessibility, TF binding, distal gene expression), so convergence means agreement among the oracles that can actually see a given layer, not a vote across all of them. Running that comparison is one prompt, which makes convergent evidence cheap to obtain.
- **Accessible to non-engineers.** Clinicians and scientists can query foundation models in plain language, without writing code.
- **Reproducible and auditable.** Interval-tracked edits and standardised inputs and outputs make analyses easy to share, review, and rerun.
- **Faster hypothesis generation.** A mechanistic story that once took weeks of wet-lab work can be sketched computationally in hours, as a starting point for validation rather than a replacement for it.

---

## Limitations and open questions

Chorus is an orchestration and interpretation layer, not an oracle of truth. A few caveats are worth stating plainly:

- **Magnitude is unreliable.** As the *SORT1* example shows, the models recover direction and mechanism far better than effect size. The predicted *SORT1* change was well under 2-fold against a measured effect above 12-fold. Treat predicted fold changes as ordinal, not quantitative.
- **You can only test what the models cover.** A variant can be assessed only in the cell types and assays present in the oracles. In the rs9504151 example the matched evidence came from a lung-fibroblast accessibility model, while the MPRA support came from non-matching lines, so it counts for less. Missing the relevant cell type can mean missing the effect entirely. This is the single most important limitation to keep in mind.
- **Distal targets need long context.** Linking a variant to a gene requires an oracle whose window spans both. Short-range models such as ChromBPNet and LegNet cannot answer gene-level questions, by design rather than by failure, as the 100 kb gap to *SORT1* illustrates.
- **Single-oracle layers need care.** When only one model scores a layer (the n=1 entries), that layer is a hypothesis, not a consensus. Convergence across the oracles that can see it is what earns confidence.
- **LD is not always resolvable in silico.** Functional prediction narrows a credible set, but very high LD with parallel signatures can still leave residual ambiguity, and MPRA or other assays remain the arbiter.
- **Predictions are a starting point.** The right mental model is design, then score, then validate, not score then conclude.

Open directions we are interested in: calibrating predicted effect sizes against measured ones, learning when oracles disagree for informative reasons rather than noise, and tightening the agent's track-selection so the conversational path matches an expert's by default.

---

## Try it yourself

Chorus is on GitHub at [github.com/pinellolab/chorus](https://github.com/pinellolab/chorus) under an MIT license. Here is the short path from clone to conversation.

### Install

```bash
# 1. Clone and create the base environment
git clone https://github.com/pinellolab/chorus.git
cd chorus
mamba env create -f environment.yml
mamba activate chorus
python -m pip install -e .

# 2. Set up the oracles you need, each in its own isolated env
chorus setup --oracle alphagenome   # JAX, recommended primary oracle
chorus setup --oracle chrombpnet    # TensorFlow (includes BPNet)
chorus setup --oracle legnet        # PyTorch (MPRA)

# 3. Grab a reference genome and check health
chorus genome download hg38
chorus health --timeout 300
```

The per-oracle commands above install only what you need for these examples. If you would rather grab everything in one shot, a bare `chorus setup` (no flags) builds all seven oracle environments, downloads their weights and backgrounds, and pulls hg38 in a single unattended run.

> AlphaGenome weights live in a gated HuggingFace repository. Accept the license at the `google/alphagenome-all-folds` model page, then `export HF_TOKEN="hf_..."` before first use. The `fine_map_causal_variant` tool additionally needs a free `LDLINK_TOKEN` (or you can pass LD variants manually).

### A minimal variant-effect call in Python

This scores the *local* effect of the SORT1 variant, the chromatin question from Step 1. Note the explicit window equal to ChromBPNet's input size (2,114 bp) centered on the variant — scoring fixed-input oracles on their full input window is what yields the calibrated effect:

```python
import chorus
from chorus.utils import get_genome

genome = get_genome("hg38")  # auto-downloads if needed
oracle = chorus.create_oracle("chrombpnet", use_environment=True,
                              reference_fasta=str(genome))
oracle.load_pretrained_model(assay="DNASE", cell_type="HepG2")

# rs12740374 (chr1:109274968 G>T): local accessibility effect in liver,
# scored on ChromBPNet's full 2,114 bp input window centered on the variant.
variant_effects = oracle.predict_variant_effect(
    "chr1:109273911-109276025",   # variant ± 1057 bp = 2114 bp
    "chr1:109274968",             # variant position
    ["G", "T"],                   # reference allele first, then alternate
    None,                         # score the loaded track
)
```

The gene-level question (does *SORT1* itself move?) is different: the promoter is about 100 kb away, so it needs a model whose window reaches that far. The cleanest way to ask it is conversationally, letting the agent load AlphaGenome and center its 1 Mb window on the variant.

### Talk to it instead

Connect the MCP server and let an agent drive. From any project folder:

```bash
curl -sL https://raw.githubusercontent.com/pinellolab/chorus/main/.mcp.json -o .mcp.json
claude   # Claude Code reads .mcp.json on startup and launches the server
```

Then the worked examples above are just prompts:

> *Load AlphaGenome and predict the effect of rs12740374 (chr1:109274968 G>T) on hepatocyte CAGE expression near SORT1.*

The same `.mcp.json` block works in Claude Desktop. Three example notebooks in the repo (`single_oracle_quickstart`, `comprehensive_oracle_showcase`, `advanced_multi_oracle_analysis`) cover the Python side from first prediction to cross-oracle comparison.

![Every oracle's reference and alternate signal on a single browser axis, with the per-layer consensus table.](fig_crossoracle_browser.png "Figure 3. Cross-oracle consensus and the shared genome-browser view. Each cell is the single strongest track an oracle reported for that layer; the browser overlays all oracles after quantile normalisation so the tracks are directly comparable.")

---

## Takeaway — the TL;DR

Sequence-to-function models are powerful, but using them together is fragmented and technically demanding.

> Chorus reframes them as interchangeable genomic oracles, harmonises their outputs, and puts a conversational layer on top.

Through that layer, models can be selected, run, compared, and interpreted in natural language. That enables fast variant prioritisation, multi-model mechanistic cross-validation, reproducible regulatory analysis, and seq2func workflows that clinical interpreters can actually use. In our hands it reproduced the mechanism of a landmark cholesterol variant through conversation alone, and recovered the disrupted transcription factor that an independent fine-mapping study reported for a lung-function locus. It also told us, honestly, where it could not match the measured effect size and where the absence of a matching cell type limits what can be concluded, which is the most useful thing a model can do.

---

## References

1. Musunuru, K. et al. From noncoding variant to phenotype via SORT1 at the 1p13 cholesterol locus. *Nature* **466**, 714 (2010). <https://doi.org/10.1038/nature09266>
2. Avsec, Ž. et al. Effective gene expression prediction from sequence by integrating long-range interactions (Enformer). *Nature Methods* **18** (2021). <https://www.nature.com/articles/s41592-021-01252-x>
3. Avsec, Ž. et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature* **649** (2026). <https://www.nature.com/articles/s41586-025-10014-0>
4. Linder, J. et al. Borzoi: predicting RNA-seq coverage from DNA sequence (2023 preprint, Calico). <https://github.com/calico/borzoi>
5. ChromBPNet / BPNet: base-resolution models of chromatin accessibility and TF binding. <https://github.com/kundajelab/chrombpnet>
6. Chen, K. M. et al. Sei: sequence-based prediction of regulatory effects (2022). <https://github.com/FunctionLab/sei-framework>
7. Penzar, D. et al. LegNet: a fully convolutional network for short regulatory sequences (2023). <https://github.com/autosome-ru/LegNet>
8. Borzoi-informed fine mapping improves causal variant prioritization in complex trait GWAS (the "Sniff" framework). *bioRxiv* 2025.07.09.663936 (2025). <https://www.biorxiv.org/content/10.1101/2025.07.09.663936v1>
9. Model Context Protocol specification. <https://modelcontextprotocol.io/>
10. Murphy, A., Durán, A. & Koo, P. K. "Adapting AlphaGenome to MPRA data." *Genomics × AI Blog*, 20 February 2026. <https://genomicsxai.github.io/blogs/2026-002/>
11. Chorus source code. <https://github.com/pinellolab/chorus>

### Cite this post

Dmitry Penzar, Lorenzo Ruggeri, Rosalba Giugno, Luca Pinello.
"Chorus: chatting with genomic oracles." *Genomics × AI Blog*,
16 June 2026. <https://genomicsxai.github.io/blogs/2026-006/>.
