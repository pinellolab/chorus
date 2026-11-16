# SORT1 rs12740374 — which tissue does the variant hit, and through which layers?

## Variant: rs12740374 (chr1:109274968 G>T, hg38)

The LDL-cholesterol GWAS variant at the 1p13.3 / SORT1 locus. T is the ALT (minor) allele.
This walkthrough asks the question the other way round from
[`variant_analysis/SORT1_rs12740374`](../../variant_analysis/SORT1_rs12740374/): it does **not** start from
liver. It scores every track of three oracles, ranks tissues layer by layer, and only then looks at the winner.

* **Oracles, one per layer** (the defaults of `rank_variant_tissues`): Cherimoya for DNase / ATAC, BPNet (the
  `chrombpnet` oracle, CHIP models) for ChIP-TF, AlphaGenome for ChIP-histone, CAGE and RNA-seq.
* **One prompt, MCP calls only.** [`prompt.md`](prompt.md) is the whole request. It contains no shell or Python
  step; every choice that needed one when this was first run (which cell type, which track ids) is written
  into it.

> **What is an MCP call and what is a choice.** Every analysis step - loading a model, discovery, ranking, the
> gene-response plot, the multilayer report, each prediction, the locus report - is a chorus MCP call, and the
> prompt names its parameters. The only things that are not calls are *choices* between their results: the target
> tissue, the target cell type, and which few factors to draw in the locus report. In an interactive session those
> are the user's decisions, taken from the rankings; the prompt writes them as rules so that an unattended replay
> makes the same ones. Applying the rules means reading tool-written CSVs (no code); no MCP reply yet carries the
> per-cell-type coverage and effect, so that part is not a call.

> **How the tissue and the cell type are chosen.** Nothing is fixed in advance. The tissue is the Global rank 1
> of the tissue-level ACAT ranking (step 2). The cell type (step 4) is a cell type of that tissue that has data in
> all layers (chromatin, TSS, histone, TF, RNA-seq); if several do, the one with the largest absolute effect in
> layers A-D. On this run's data three Liver cell types cover all layers - `liver` (tissue sample), HepG2 and
> hepatocyte - and the strongest |log2FC| is 3.3705 for HepG2, 2.3404 for `liver` and 1.2984 for hepatocyte, so
> **HepG2** is selected. Ranking the cell types by expression instead (mean |median log2FC| of PSRC1, CELSR2,
> MYBPHL, SORT1: left lobe of liver 0.5182, `liver` 0.4854, right lobe 0.4764, HepG2 0.3751) would pick a cell
> type with no histone, CAGE or TF track, which is why the rule asks for full layer coverage first.

> **Status of the prompt.** Every output in this folder comes from one replay of this prompt, pasted as a single
> message into an interactive Claude Code session (2026-11-16), against the refreshed background nulls
> (`backgrounds-2026-08-16-sei-full`). The shipped [`prompt.md`](prompt.md) differs from the text that was replayed
> in one sentence: the replayed prompt also said "use GPU 0 for every model (pass device="cuda:0" to every
> load_oracle)", which is specific to the machine it ran on, and it was removed. The `user_prompt` quoted at the top of
> the reports (for example `04_multilayer_HepG2/multilayer_report.md`) is the replayed text, so it still has that
> sentence. The run was **not** unattended: the model applied the choice rules itself, and a few points the prompt
> leaves open were settled by judgement (see "Things to know"). Treat an unattended replay as still untested.

## Prerequisites (read before replaying)

1. **BPNet ChIP-TF discovery has to be wired into `discover_variant`.** Out of the box the `chrombpnet`
   branch of `chorus/analysis/discovery.py` iterated only `iter_unique_models()`, i.e. ATAC / DNase: 9 tracks,
   no ChIP-TF. The version that produced this folder also iterates `iter_unique_bpnet_models()` (744 unique
   TF x cell-line models, each with a per-track CDF). That change is **not in v0.8.0**; it ships with the
   commit that adds this walkthrough, and without it the ChIP-TF column of the ranking cannot be reproduced.
2. **No MCP idle timeout.** Cherimoya and BPNet discovery run silently for a long time (the client reported each
   as still running after 120 s and moved it to the background). A client that aborts silent calls kills them.
   `.mcp.json` carries `"timeout": 86400000` for the `chorus` server (a local, uncommitted edit).
3. **A free GPU.** The replay pinned `device="cuda:0"` for every model; the shipped prompt does not, so pick the
   card with `CUDA_VISIBLE_DEVICES` (or say so in your message). BPNet (TensorFlow) reserves most of the card; in
   an earlier run each worker took about 22.8 GiB of a 24 GiB GPU.
4. **Playwright + Chromium** for the locus PNG (`build_locus_report(screenshot=true)`).

## What Claude does

| Step | MCP calls | Result |
|---|---|---|
| 1 Discovery | `load_oracle` + `discover_variant` x 3 | AlphaGenome 4,501 tracks; Cherimoya 407 tracks; ChromBPNet 753 models |
| 2 Ranking | `rank_variant_tissues` (ACAT) at tissue and system level | **Liver is Global rank 1** at both |
| 3 Expression | `plot_variant_gene_response(tissue="Liver")` | PSRC1, CELSR2, MYBPHL, SORT1 named |
| 4 Layers | `analyze_variant_multilayer` (AlphaGenome, HepG2 track list) | all four layers move |
| 5 Locus | 6 x `predict_variant_effect` (AlphaGenome, Cherimoya, 4 BPNet factors), then `build_locus_report` | 17-track HTML + PNG over the 1 Mb window |

Not part of this walkthrough: comparing the result with the original paper. The step was tried
(`fetch_paper` on the PubMed page returned `almost no text could be extracted`) and then dropped.

## Key results

### Discovery

| Oracle | What it scored | Notes |
|---|---|---|
| AlphaGenome | 4,501 tracks; 29 genes in the RNA table | one forward pass covers every track |
| Cherimoya | 407 tracks (205 ATAC, 202 DNase) | one experiment per biosample, out of 1,518 |
| ChromBPNet | 753 models: 744 ChIP-TF + 5 DNase + 4 ATAC (1,497 tracks counting both strands) | one model per cell type x factor |

Run times and peak RAM were not measured in this replay.

### Tissue ranking (ACAT; rank within each layer, 1 = strongest; – = no track)

| Global | Tissue | DNase | ATAC | ChIP-TF | Histone | RNA-seq | CAGE |
|---|---|---|---|---|---|---|---|
| 1 | **Liver** | 11 | 9 | 1 | 1 | 1 | 1 |
| 2 | Myeloid cells | 1 | 25 | 7 | 8 | 11 | 3 |
| 3 | Uterus / endometrium | – | 6 | 2 | 3 | 47 | 50 |
| 4 | Adrenal gland | – | 21 | 38 | 2 | 9 | 2 |
| 5 | Lung | 6 | 11 | 4 | 16 | 4 | 12 |

System level: **Liver** (1), Adipose (2), Respiratory (3), Breast / mammary (4), Endocrine (5).
Liver's per-layer ranks there are DNase 6, ATAC 4, ChIP-TF 1, histone 1, RNA-seq 1, CAGE 1.
Figures: [`02_ranking_tissue/layer_rank_table.png`](02_ranking_tissue/layer_rank_table.png) (top 15),
[`02_ranking_system/layer_rank_table.png`](02_ranking_system/layer_rank_table.png). Both replies had empty `notes`.

Read it with care: Liver wins on ChIP-TF, histone, RNA-seq and CAGE, but its chromatin ranks are only 11 of 37
(DNase) and 9 of 33 (ATAC). It is not the top tissue for accessibility.

### Which genes move in Liver ([`03_gene_response_liver/`](03_gene_response_liver/))

12 liver RNA-seq tracks, 29 genes, AlphaGenome noise floor 0.06016 log2FC.

| Gene | Median log2FC | Tracks as outlier | Tracks above noise floor |
|---|---|---|---|
| PSRC1 | +0.5732 | 12 / 12 | 12 / 12 |
| CELSR2 | +0.549 | 12 / 12 | 12 / 12 |
| MYBPHL | +0.3023 | 12 / 12 | 11 / 12 |
| SORT1 | +0.0732 | 11 / 12 | 8 / 12 |

SORT1 itself is the weakest of the four, and inside the noise floor on 4 of 12 tracks.

### The four layers in HepG2 ([`04_multilayer_HepG2/`](04_multilayer_HepG2/))

HepG2 was selected by the rule above: it ties with `liver` and hepatocyte on layer coverage and has the largest
absolute effect (3.3705). It also has many more tracks than the others (677 vs 44 for `liver`, across the three
tables, not counting splice-site tracks). The multilayer report was run on 25 HepG2 tracks: DNase, ATAC, both CAGE
strands, all 10 histone marks, the 6 ChIP-TF tracks with the largest effect (CEBPB, CEBPA, CEBPG, HLF, CEBPD,
ATF4) and all 5 RNA-seq tracks. HepG2 has no PRO-CAP track (not covered).

| Layer | Track | Ref -> Alt | Effect (log2FC) | Effect %ile |
|---|---|---|---|---|
| A chromatin | DNASE:HepG2 | 631 -> 1.62e3 | +1.355 | 0.9965 |
| A chromatin | ATAC:HepG2 | 439 -> 737 | +0.747 | 0.9922 |
| B TSS | CAGE:HepG2 (variant site, - strand) | 24.3 -> 73.2 | +1.549 | 0.9983 |
| C histone | H3K27ac | 1.43e4 -> 3.49e4 | +1.287 | 0.9993 |
| C histone | H3K27me3 | 1.23e3 -> 887 | -0.475 | 0.9998 |
| D TF | CEBPB | 1.06e3 -> 1.1e4 | +3.370 | 0.9996 |
| D TF | CEBPA | 2.01e3 -> 1.6e4 | +2.997 | 0.9999 |
| D TF | CEBPG / HLF | 1.73e3 -> 9.85e3 / 451 -> 2.44e3 | +2.508 / +2.435 | 0.9997 / 0.9996 |

All values are AlphaGenome's. The same cell type in the other oracles (from their discovery tables): Cherimoya
HepG2 ATAC +1.120; ChromBPNet BPNet CEBPB +2.715 (the stronger strand; the table keeps one row per model),
CEBPA +1.865, HLF +1.558, CEBPD +1.201.
RNA-seq in HepG2 (top rows only; the report lists 145): CELSR2 +0.453, PSRC1 +0.452, MYBPHL +0.239. SORT1's own
exon rows are smaller: +0.057 (polyA, - strand), +0.054 (total, - strand), +0.073 (polyA, unstranded).

### Locus view ([`05_locus_report/`](05_locus_report/))

[`chr1_109274968_G_T_locus_report.html`](05_locus_report/chr1_109274968_G_T_locus_report.html) (and `.png`):
17 tracks over `chr1:108750680-109799255` at 128 bp bins, reference blue and ALT orange, with GENCODE genes:
Cherimoya ATAC (`ENCSR042AWH`), the 10 AlphaGenome histone marks (one track each), AlphaGenome CAGE and polyA+
RNA-seq (both strands summed), and BPNet CEBPB, CEBPA, HLF and CEBPD (both strands summed).
In the PNG the orange (T) excess at the variant is clearly visible in the Cherimoya ATAC track, H3K27ac and the
BPNet CEBPB, CEBPA and CEBPD tracks, and slight in H3K4me1, H3K9ac and RNA-seq. The CAGE and BPNet HLF effects
are not visible at 1 Mb scale, where other peaks dominate the axis.

## Things to know

* **File overwrite.** `analyze_variant_multilayer` writes `chr1_109274968_G_T_SORT1_alphagenome_report.html`,
  the same name `discover_variant` used for AlphaGenome, so the multilayer report **replaced the discovery HTML**.
  Only the multilayer one is kept here. The discovery CSV tables are intact.
* **Cherimoya's one-experiment-per-biosample rule** gave HepG2 only an ATAC model (no DNase), `ENCSR042AWH`.
  `list_tracks` shows two more HepG2 experiments (ATAC `ENCSR291GJU`, DNase `ENCSR149XIL`) that discovery did not
  score and the locus report does not draw. Its table has three liver-mapped ATAC biosamples (ENCSR624ODL left
  lobe of liver, ENCSR124NNL liver, ENCSR373TDL right lobe of liver) and one hepatocyte DNase (ENCSR364MFN).
  ChromBPNet's own DNASE:HepG2 is in the ChromBPNet table.
* **Points the prompt leaves open, settled by judgement in this replay.** (1) Which HepG2 Cherimoya experiment to
  load for the locus report: the one discovery scored, `ENCSR042AWH`. (2) The "AlphaGenome histone marks" track
  was drawn as one track per mark, because averaging 10 different marks is not meaningful. (3) RNA-seq in the
  locus report uses the polyA+ `+` and `-` tracks only; adding the total and unstranded tracks would double
  count. (4) The cell-type coverage count and the top-6 ChIP-TF / top-4 BPNet picks were made by filtering the
  tool-written CSVs with `grep` / `awk`; no MCP reply carries them. A prompt that fixes these would make an
  unattended replay deterministic.
* **Near tie, AlphaGenome chromatin.** Ranks 3 and 4 (DNase in amniotic epithelial cell vs ATAC in right lobe of
  liver) differ by 0.211 %, inside the cross-process drift band, so which one is reported can change between runs.
* **BPNet CEBPA and CEBPD** over the 1 Mb window are mostly at or below zero (95.6 % and 91.9 % of 128 bp bins
  drawn at 0) and their largest per-position effects are tiny (0.022 and 0.013, against 0.196 for CEBPB), so those
  tracks show only a few spikes. Their discovery log2FC (+1.865, +1.201) comes from very small baselines; read them
  as weak evidence. CEBPB (10.7 % of bins at 0) does not have this issue.
* **Raw scores moved slightly against the previous version of this folder** (largest absolute change 0.0102 for
  Cherimoya, 0.0811 for AlphaGenome, 0.0005 for ChromBPNet). The cause was not isolated (cross-process / GPU
  drift, or code differences between the two runs). A percentile near a threshold can therefore move, and the
  near tie above can flip.
* **Raw bedgraphs are not shipped.** The per-track reference / alt predictions are ~37 MB each at 1 bp; the
  locus HTML embeds its binned data and `05_locus_report/predictions_*.json` list what was saved.

## Folder

```
prompt.md                      the single prompt (MCP calls only)
01_discovery/                  track_scores / prediction_matrix CSVs per oracle + AlphaGenome RNA tables
02_ranking_tissue/             rank_variant_tissues, group_level="tissue" (figure PNG/PDF + tables)
02_ranking_system/             same at group_level="system"
03_gene_response_liver/        plot_variant_gene_response (figure + per-gene / per-track CSVs)
04_multilayer_HepG2/           analyze_variant_multilayer report (HTML) + its markdown
05_locus_report/               build_locus_report HTML + PNG, and the prediction manifests
```
