# Multi-oracle validation — rs12740374 at the SORT1 locus

**What this example demonstrates.** A single variant is scored by **four
independent deep-learning oracles** and the answers are fused into one
consensus view so a new user can tell at a glance whether the oracles agree
on direction, and which assay / cell type drove each call.

The classic SORT1 LDL-cholesterol variant
[`rs12740374`](https://www.ncbi.nlm.nih.gov/snp/rs12740374) is ideal for this
kind of validation because its mechanism is well characterised: the minor
(T) allele creates a C/EBPα binding site in a HepG2 enhancer that drives
*SORT1* expression. Any honest oracle should flag (a) increased chromatin
accessibility, (b) increased C/EBP binding and (c) increased downstream CAGE
activity at this variant — all on the HepG2 cell type.

## Oracles used

| Oracle | Role | Regulatory layer |
| --- | --- | --- |
| **ChromBPNet** | chromatin accessibility specialist | DNase/ATAC |
| **Cherimoya** (CATv1) | chromatin accessibility specialist | DNase/ATAC |
| **LegNet** | MPRA / promoter activity specialist | LentiMPRA (promoter) |
| **AlphaGenome** | generalist multi-track model | ChIP, histones, CAGE |

**ChromBPNet and Cherimoya are deliberately pointed at the same question** —
HepG2 DNase accessibility — rather than being given different assays to cover.
Two independently trained models agreeing on one variant is a stronger claim
than one model asserting it, and because they share a 2,114 bp input window and
base-pair-resolution output, their rows and IGV tracks are directly comparable
instead of merely adjacent. They agree here on direction and differ on
magnitude — and because AlphaGenome also carries a HepG2 DNase track, the same
question is answered three independent ways:

| Oracle | HepG2 DNase track | raw log2FC | percentile |
| --- | --- | --- | --- |
| Cherimoya | `DNASE:ENCSR149XIL` | **+1.793** | 0.9999 |
| ChromBPNet | `DNASE:HepG2` | **+1.376** | 0.9995 |
| AlphaGenome | `DNASE:HepG2` | **+1.334** | 0.9964 |

Read that as concordance on the finding and honest uncertainty on the size: a
2.5–3.5× predicted increase in accessibility, from three models that agree the
variant is above the 99.6th percentile of their own nulls.

**Three things not to conclude from this table**, each measured rather than
argued (full workup in `docs/BACKGROUND_NULL_PROTOCOL.md` §12):

- **Do not read `ref`/`alt` across rows.** They are model-specific
  depth-normalised scales. Cherimoya's reference is 2.1× ChromBPNet's for the
  *identical* sequence and neither is wrong. Only the log2FC and percentile
  columns are cross-comparable.
- **Do not conclude Cherimoya is the outlier.** ChromBPNet and Cherimoya are in
  fact the *same* ENCODE experiment (`ENCSR149XIL`), with chr1 held out of
  training for both. The apparent AlphaGenome/ChromBPNet agreement is an artefact
  of the 501 bp scoring window: at 51 bp the two accessibility specialists agree
  to 1.6% (3.62 vs 3.57) and *both* disagree with AlphaGenome at 2.51. The curves
  cross at 47 bp.
- **Do not read the 1.37× spread as large.** Across all 18,672 reference variants
  these two models correlate at r = 0.888 with a mean difference of −0.001 log2,
  and among strong effects 18–22% of loci disagree by more than this one does.

What the spread mostly reflects is that a single cross-validation fold is a
sample, not the model: Cherimoya's five folds give 3.47 (fold 0, shipped), 2.39,
2.72, 2.77, 2.77 — and ChromBPNet's 2.60 sits inside that range.
Cherimoya track ids are `ASSAY:ENCSR` rather than `ASSAY:biosample`, because
(assay, biosample) is ambiguous for 1,188 of its 1,518 experiments — the
accession is what identifies a model.

## What to look at first

1. **Consensus matrix** — each row is a regulatory layer, each oracle column
   shows its strongest track for that layer (with assay and cell type). The
   "Agreement (direction)" column flags whether the oracles push the same way:
   `✅ all ↑`, `✅ all ↓`, `⚠ disagree`, or `↑ only (n=1)` when only a single
   oracle is competent on that layer.

   **It compares direction only** — literally the sign of each oracle's effect —
   so it is shown alongside the magnitude spread rather than on its own. The
   accessibility row here reads `✅ all ↑ · 3 oracles, +1.33…+1.79`: unanimous on
   direction, and differing by 1.37× in linear fold change between the extremes.
   Without the spread, that renders identically to three oracles agreeing exactly,
   and a reader has no way to tell which they are looking at.
2. **Cross-oracle genome browser** — one unified IGV instance stacks every
   oracle's ref (grey) / alt (coloured) signal tracks on a single x-axis.
   The default locus is AlphaGenome's 1 Mb window so you can see long-range
   context; the specialists (ChromBPNet and Cherimoya ~2 kb, LegNet ~200 bp) render blank
   outside their own windows, which is the *intended* visual cue that
   they can only reach local positions. Signals are floor-rescaled so 1.0
   on every track means "genome-wide p99 peak for this assay".
3. **Per-oracle evidence** — collapsible blocks drilling into each oracle's
   winning track per layer, including reference / alternate predicted values,
   effect percentiles, and a link to the oracle's standalone report.
4. **Glossary** — at the top of the page, defines every number's **units**
   (log2FC vs lnFC vs Δ) so you never have to guess what `+0.3` means.

## How this was produced

Each oracle runs in its own conda env (their dependencies don't coexist), so
the regeneration is split into four per-oracle runs plus one consolidator
step:

```bash
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-cherimoya   python scripts/regenerate_multioracle.py --oracle cherimoya
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus             python scripts/regenerate_multioracle.py --consolidate
```

The `chorus-cherimoya` env is the only one carrying the `cherimoya` package.
Run that step anywhere else and it does **not** fail fast — it logs
`Failed to load <track>` per track and carries on, producing a report with no
scores in it.

Each per-oracle run saves three artefacts:

- `<oracle>_variant_report.json` — inspectable summary (no prediction
  arrays; round-trips through `VariantReport.from_dict`).
- `<oracle>_variant_report.pkl` — full `VariantReport` **with prediction
  arrays**, used by the consolidator to draw IGV signal tracks. These
  are `.gitignore`d because they are large (AlphaGenome ~610 MB, Cherimoya
  ~194 MB, ChromBPNet ~106 MB).
- `rs12740374_SORT1_<oracle>_report.html` — standalone per-oracle
  HTML report, linked from the unified page.

The `--consolidate` step prefers pickles when present (→ IGV with live
signal) and falls back to JSON-only (→ IGV panel with modification
marker but empty signal) for any oracle whose pickle is missing.

## Files in this directory

| File | Contents |
| --- | --- |
| `rs12740374_SORT1_multioracle_report.html` | **Main report** — read this first |
| `example_output.md` | Markdown summary (consensus table) |
| `example_output.json` | Machine-readable consensus matrix |
| `<oracle>_variant_report.json` | Raw per-oracle VariantReport (inspectable, no predictions) |
| `<oracle>_variant_report.pkl` *(gitignored)* | Per-oracle predictions — regenerate by running the per-oracle command above |
| `rs12740374_SORT1_<oracle>_report.html` | Standalone per-oracle report (linked from the main page) |
