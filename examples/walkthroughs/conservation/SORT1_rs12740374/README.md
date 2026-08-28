# SORT1 rs12740374 — Effect Size vs. Cross-Species Conservation

## Variant: rs12740374 (chr1:109274968 G>T)

Same variant, oracle, cell type and tracks as the plain
[ChromBPNet analysis](../../variant_analysis/SORT1_chrombpnet/) — this walkthrough adds
`show_conservation=True` and reads the three conservation scores at the variant's own
base directly, to answer a question a variant report alone cannot: **is this effect
sitting on top of a sequence evolution has kept, or one it hasn't?**

rs12740374 is one of the best-validated regulatory variants known: it lies in a
liver-specific enhancer at the *SORT1*/*CELSR2* locus and disrupts a C/EBP
transcription-factor binding site, altering *SORT1* expression and LDL cholesterol in
humans (Musunuru et al., *Nature* 2010) — confirmed by luciferase assay, EMSA and a
mouse knock-in, not just a GWAS hit.

## Example prompt

> Score rs12740374 (chr1:109274968 G>T) using ChromBPNet DNase in HepG2, and overlay
> conservation tracks (phyloP, phastCons, GPN-Star) on the report. Gene is SORT1.

## What Claude does

1. `load_oracle('chrombpnet', assay='DNASE', cell_type='HepG2')`
2. `analyze_variant_multilayer('chrombpnet', 'chr1:109274968', 'G', ['T'], ['auto'], show_conservation=True)`
3. Report shows the chromatin-accessibility layer plus GPN-Star (coverage + sequence
   logo), phyloP 100-way and phastCons 100-way tracks in the IGV browser
4. Separately, `chorus.analysis.conservation.read_phylop_values` /
   `read_phastcons_values` / `read_entropy_values` are called directly on the variant's
   own base — the report's IGV overlay is a picture, not a number, so the three scores
   below are measured independently and saved into `example_output.json`.

## Results

**Effect** (identical to the [plain ChromBPNet run](../../variant_analysis/SORT1_chrombpnet/),
since conservation is a display-only overlay and does not change any score):

| Track | Ref | Alt | Effect | Effect %ile | Interpretation |
|-------|-----|-----|--------|-------------|----------------|
| DNASE:HepG2 | 287 | 747 | +1.376 | ≥99th (1.18× null max) | Very strong opening |

The effect **exceeds every one of the ~18,000 background variants** ChromBPNet's DNASE
null was built from — about as strong as this scoring system can register.

**Conservation, measured at chr1:109274968 itself**:

| Source | Score | Reading |
|---|---|---|
| phyloP 100-way | −0.046 | Indistinguishable from neutral (0 = clock-like drift; this base shows no excess constraint and no excess acceleration) |
| phastCons 100-way | 0.001 | Not part of any conserved element in the 100-way vertebrate alignment |
| GPN-Star entropy | 0.772 bits (of 2 max) | The model is not confident in the reference base — the IGV coverage score `1 − entropy` (clipped) is 0.228, on the low half of its 0–1 range |

All three sources — one classical (phyloP), one element-based (phastCons), one from a
sequence language model (GPN-Star) — agree: **this base carries no detectable
cross-species conservation signal.**

## The point

A saturating regulatory effect and an unconserved base are not in tension. Enhancers
turn over fast in evolution even when the gene they control, and the phenotype it
affects, are deeply conserved — a TF motif can be gained or lost at a given base pair
many times across the tree without ever being "the same" nucleotide under selection at
that exact position. rs12740374 is a clean illustration precisely because it is not a
borderline case: independent experimental validation says this effect is real, and three
independent conservation measures agree the base is not constrained.

**Practical rule.** Don't use conservation as a filter to discount a strong oracle
prediction, and don't use it to rank candidate variants when the causal mechanism is
regulatory rather than protein-coding — cross-species constraint answers "has selection
acted on this exact base," which is a different question from "does this base matter for
gene regulation in one species, one cell type, right now." A conserved position and a
strong oracle effect would corroborate each other; an unconserved position with a strong,
experimentally validated effect — this case — shows the two lines of evidence measure
different things, not that one of them is wrong.

## Reproduce the conservation scores directly

```python
from chorus.analysis.conservation import (
    read_phylop_values, read_phastcons_values, read_entropy_values,
)

read_phylop_values("chr1", 109274967, 109274968)      # -> array([-0.046])
read_phastcons_values("chr1", 109274967, 109274968)    # -> array([0.001])
read_entropy_values("chr1", 109274967, 109274968)      # -> array([0.772])
```

or from the CLI, without loading an oracle at all:

```bash
chorus conservation status                # confirms phyloP/phastCons/GPN-Star are cached
chorus annotation describe gpn_star       # verifies the bigwig's genome build
```

## Caveats

- **hg38 only.** `show_conservation=True` raises on a non-hg38 report rather than
  plotting human conservation against other coordinates.
- **Budget ~70 GB** the first time `show_conservation=True` runs with the GPN-Star
  sequence-logo track enabled (phyloP + phastCons + GPN-Star coverage are ~25 GB; the
  four per-allele LLR bigwigs behind the logo track are ~45 GB more). All of it downloads
  once and is cached like oracle weights — see
  [`docs/BACKGROUND_NULL_PROTOCOL.md`](../../../../docs/BACKGROUND_NULL_PROTOCOL.md) for
  how the effect null itself is built, which is a separate thing from these conservation
  sources.
- **The GPN-Star coverage score is a fixed transform, not a fitted probability.**
  `1 − entropy` (clipped to [0, 1]) is a convenient 0–1 scale for the IGV track, not a
  calibrated "probability of functional constraint" — treat 0.228 as "low relative to
  this locus," not as a percentile against any background.
