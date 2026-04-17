# Integration Simulation Example: CMV Cassette at AAVS1

## Scenario

Simulate inserting a 378 bp CMV-promoter construct at the AAVS1 safe
harbor locus (chr19:55115000, within the PPP1R12C gene) in K562 cells.
This is a standard gene therapy integration site — we want to predict
both disruption of local chromatin and any new transcriptional activity
from the cassette.

## Example prompt

> Insert a 378 bp CMV promoter construct at chr19:55115000
> (PPP1R12C locus / AAVS1 safe harbour) and predict local disruption
> in K562 using DNASE, H3K27ac, and CAGE tracks.

## What Claude does

1. Calls `load_oracle('alphagenome')`
2. Calls `simulate_integration(oracle_name='alphagenome', position='chr19:55115000', construct_sequence='...', assay_ids=[...K562 tracks...], gene_name='PPP1R12C')`
3. Returns a report comparing wild-type vs post-insertion

## Results summary

| Layer | Effect | Interpretation |
|-------|--------|----------------|
| Chromatin (DNASE:K562) | +4.224 log2FC | Very strong opening |
| Histone H3K27ac:K562 | +1.199 log2FC | Very strong mark gain |
| CAGE — RPL28 TSS | −8.957 log2FC | Very strong decrease |
| CAGE — ZNF628 TSS | −7.988 log2FC | Very strong decrease |
| CAGE — KMT5C TSS | −6.598 log2FC | Very strong decrease |

**Key findings:**
- **Strong new chromatin opening** (DNASE +4.22): the inserted CMV
  promoter creates a large nuclease-sensitive region at the site
- **H3K27ac gain** (+1.20): active chromatin marks accumulate at the
  construct, consistent with an active CMV promoter
- **Nearby TSS silencing** (CAGE −8.96 at RPL28, −7.99 at ZNF628):
  the construct appears to hijack local transcriptional resources
  and disrupt neighbouring gene expression

This pattern — strong new activity at the insertion site combined with
disruption of nearby promoters — illustrates why "safe harbour" loci
still require empirical validation. The prediction agrees with
published observations that AAVS1 insertions can affect neighbouring
PPP1R12C/TNNT3/TNNT1 expression in some cell types.

## Output files

- `integration_CMV_PPP1R12C_report.html` — interactive HTML report with IGV browser
- `example_output.md` — markdown table
- `example_output.json` — structured per-track scores
- `example_output.tsv` — tab-separated summary
