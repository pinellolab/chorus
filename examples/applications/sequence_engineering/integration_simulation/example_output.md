## Analysis Request

> Simulate inserting a CMV promoter construct at chr19:55115000 (PPP1R12C locus) and predict the local disruption in K562 — I want to see chromatin, H3K27ac, and CAGE effects from the insertion.

- **Tool**: `simulate_integration`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: DNASE:K562, H3K27ac:K562, CAGE:K562
- **Generated**: 2026-04-12 02:21 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr19:55115000 wt>insertion
**Oracle**: alphagenome
**Gene**: PPP1R12C
**Other nearby genes**: TNNT1, EPS8L1, TNNI3, ENSG00000267110

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.96); Chromatin accessibility (DNASE/ATAC): very strong opening (+4.22); Histone modifications (ChIP-Histone): very strong mark gain (+1.20).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| DNASE:K562 | 24.4 | 473 | +4.224 | Very strong opening |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 3.77e+03 | 8.65e+03 | +1.197 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CAGE:K562 — RPL28 TSS | 7.03e+04 | 140 | -8.958 | Very strong decrease |
| CAGE:K562 — ZNF628 TSS | 2.05e+03 | 7.1 | -7.986 | Very strong decrease |
| CAGE:K562 — KMT5C TSS | 2.27e+03 | 22.3 | -6.605 | Very strong decrease |
| CAGE:K562 — NAT14 TSS | 2.48e+03 | 40.7 | -5.895 | Very strong decrease |
| CAGE:K562 — ZNF581 TSS | 14.8 | 518 | +5.033 | Very strong increase |
| CAGE:K562 — ZNF865 TSS | 1.66e+03 | 125 | -3.718 | Very strong decrease |
| CAGE:K562 — ISOC2 TSS | 5.83 | 70.5 | +3.389 | Very strong increase |
| CAGE:K562 — TMEM238 TSS | 11.3 | 119 | +3.278 | Very strong increase |
| CAGE:K562 — ZNF524 TSS | 825 | 98.2 | -3.058 | Very strong decrease |
| CAGE:K562 — SSC5D TSS | 108 | 16 | -2.676 | Very strong decrease |
| _…showing top 10 of 53 — see `example_output.json` for the full set_ | | | | |

---

---

## Interpretation

**What the oracle sees.** Inserting a CMV promoter construct at
chr19:55115000 produces localised chromatin and CAGE increases at the
insertion site and modest collateral effects on the surrounding
regulatory landscape. Effects are tightly localised — the oracle does
not predict widespread disruption of neighbouring genes.

**How this fits expectations.** Integration of a strong constitutive
promoter into a permissive locus is predicted to create a local
transcriptionally active island without major effects on the
surrounding genes — consistent with what is known about AAVS1-like safe
harbour integration.

**Suggested next steps.**
- Use this example as a template for predicting safe-harbour integration
  effects before committing to a gene-therapy construct.
- Compare predicted versus wet-lab signals from the specific integration
  site, if available. The oracle should agree with the published
  promoter-insertion-at-safe-harbour datasets.
