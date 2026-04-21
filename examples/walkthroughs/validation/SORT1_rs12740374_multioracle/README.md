# Multi-oracle validation — rs12740374 at the SORT1 locus

**What this example demonstrates.** A single variant is scored by **three
independent deep-learning oracles** and the three answers are fused into one
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
| **LegNet** | MPRA / promoter activity specialist | LentiMPRA (promoter) |
| **AlphaGenome** | generalist multi-track model | ChIP, histones, CAGE |

## What to look at first

1. **Consensus matrix** — each row is a regulatory layer, each oracle column
   shows its strongest track for that layer (with assay and cell type). The
   "Agreement" column flags whether the oracles push in the same direction:
   `✅ all ↑`, `✅ all ↓`, `⚠ disagree`, or `↑ only (n=1)` when only a single
   oracle is competent on that layer.
2. **Cross-oracle genome browser** — one unified IGV instance stacks every
   oracle's ref (grey) / alt (coloured) signal tracks on a single x-axis.
   The default locus is AlphaGenome's 1 Mb window so you can see long-range
   context; the specialists (ChromBPNet ~1 kb, LegNet ~200 bp) render blank
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
the regeneration is split into three per-oracle runs plus one consolidator
step:

```bash
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus             python scripts/regenerate_multioracle.py --consolidate
```

Each per-oracle run saves three artefacts:

- `<oracle>_variant_report.json` — inspectable summary (no prediction
  arrays; round-trips through `VariantReport.from_dict`).
- `<oracle>_variant_report.pkl` — full `VariantReport` **with prediction
  arrays**, used by the consolidator to draw IGV signal tracks. These
  are `.gitignore`d because AlphaGenome's pickle is ~600 MB.
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
