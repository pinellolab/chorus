## Analysis Request

> Can you simulate replacing the chr1:109274500-109275500 region with a strong K562 promoter sequence? I want to see how the swap would change DNASE, H3K27ac, H3K4me3, and CAGE around the SORT1 enhancer.

- **Tool**: `analyze_region_swap`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: DNASE:K562, H3K27ac:K562, H3K4me3:K562, CAGE:K562
- **Generated**: 2026-04-12 02:21 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109275000 wt>replacement
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-8.12); Chromatin accessibility (DNASE/ATAC): very strong closing (-3.29); Histone modifications (ChIP-Histone): very strong mark loss (-1.35).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| DNASE:K562 | 216 | 21.1 | -3.293 | Very strong closing |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CHIP:H3K27ac:K562 | 4.98e+03 | 1.95e+03 | -1.353 | Very strong mark loss |
| CHIP:H3K4me3:K562 | 2.16e+03 | 1.06e+03 | -1.033 | Very strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Interpretation |
|---|---|---|---|---|
| CAGE:K562 — GSTM2 TSS | 1.14e+03 | 3.12 | -8.116 | Very strong decrease |
| CAGE:K562 — GNAI3 TSS | 1.07e+04 | 63.3 | -7.378 | Very strong decrease |
| CAGE:K562 — GSTM1 TSS | 174 | 0.726 | -6.661 | Very strong decrease |
| CAGE:K562 — CYB561D1 TSS | 876 | 11.4 | -6.140 | Very strong decrease |
| CAGE:K562 — ATXN7L2 TSS | 1.26e+03 | 44.2 | -4.805 | Very strong decrease |
| CAGE:K562 — AMPD2 TSS | 1.29e+03 | 46.8 | -4.761 | Very strong decrease |
| CAGE:K562 — SYPL2 TSS | 31.1 | 2.87 | -3.053 | Very strong decrease |
| CAGE:K562 — GSTM5 TSS | 10.2 | 0.357 | -3.044 | Very strong decrease |
| CAGE:K562 — GSTM3 TSS | 9.51 | 0.9 | -2.467 | Very strong decrease |
| CAGE:K562 — GSTM4 TSS | 3.14e+03 | 739 | -2.087 | Very strong decrease |
| _…showing top 10 of 29 — see `example_output.json` for the full set_ | | | | |

---

---

## Interpretation

**What the oracle sees.** Replacing the SORT1 enhancer region with the
prepared K562 promoter sequence produces strong chromatin closing and
H3K27ac loss at the swap site and a very strong CAGE decrease at the
SORT1 TSS. The predicted effect is a *loss* of regulatory activity.

**How this fits expectations.** The swap removes a well-characterised
liver enhancer and replaces it with an out-of-context promoter
sequence. In the K562 (erythroid leukaemia) context used by these
assays, the inserted sequence is not a functional K562 promoter,
so the loss-of-function prediction is biologically consistent with
"removed the real signal and replaced it with a weaker one".

**Suggested next steps.**
- This is a *loss-of-function* demonstration, not a gain-of-function
  one. For a gain demo, swap in a known strong K562 promoter (e.g. the
  MYB intron-1 enhancer) into a normally silent locus.
- The same workflow (`analyze_region_swap`) can be used for enhancer
  grafting experiments, reporter construct design, and enhancer
  deletion simulation.
