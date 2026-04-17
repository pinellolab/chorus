# Integration Simulation Example: GFP Cassette at AAVS1

## Scenario

Simulate inserting a GFP expression cassette at the AAVS1 safe harbor
locus (chr19:55115000, within the PPP1R12C gene). This is a standard
gene therapy integration site — we want to predict both disruption of
local chromatin and any new transcriptional activity from the cassette.

## Example prompt

> I'm planning to integrate a GFP cassette at the AAVS1 safe harbor
> site (chr19:55115000). Predict how this insertion would affect the
> local chromatin environment and nearby gene PPP1R12C. Use AlphaGenome
> with K562 DNASE, H3K27ac, and CAGE tracks.

## What Claude does

1. Calls `load_oracle('alphagenome')`
2. Calls `list_tracks('alphagenome', query='K562')` to find relevant tracks
3. Calls `simulate_integration(oracle_name='alphagenome', position='chr19:55115000', construct_sequence='...', assay_ids=[...], gene_name='PPP1R12C')`
4. Returns a report comparing wild-type vs post-insertion

## Results summary

| Layer | Effect | Interpretation |
|-------|--------|----------------|
| Chromatin (DNASE) | -0.900 log2FC | Very strong closing |
| Histone H3K27ac | -0.149 log2FC | Moderate mark loss |
| TSS (CAGE) variant site | +1.324 log2FC | Very strong increase |

**Key findings:**
- **Chromatin disruption**: Strong closing at the insertion site — the
  construct displaces the native open chromatin structure
- **Histone mark reduction**: Moderate loss of H3K27ac, suggesting the
  insertion partially disrupts the active chromatin state
- **New TSS activity**: Strong CAGE increase at the insertion site,
  consistent with the cassette's CMV promoter driving transcription

This pattern (local disruption + new activity) is typical of transgene
insertions and demonstrates the value of multi-layer analysis for
evaluating safe harbor sites.

## Output files

- `AAVS1_GFP_K562_alphagenome_report.html` — interactive HTML report
- `example_output.md` — markdown table
- `example_output.json` — structured JSON
