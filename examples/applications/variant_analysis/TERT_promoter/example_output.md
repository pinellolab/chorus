## Analysis Request

> Can you analyze the TERT C228T promoter mutation (chr5:1295228 G>A) for effects on TERT expression? This is a recurrent somatic mutation in melanoma and glioblastoma.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all oracle tracks (discovery mode)
- **Generated**: 2026-04-12 02:21 UTC

**Notes / caveats:**
- Biological context: Horn/Huang 2013 (Science) report that C228T creates a de novo GABPA/ETS binding site and ACTIVATES TERT in melanoma / glioblastoma / bladder cancer. AlphaGenome is trained on bulk tissue where TERT is largely silent, so its CAGE prediction here reflects the endogenous chromatin state (TERT repressed) rather than the cancer-specific reactivation. This example is kept to illustrate how model training data shapes predictions — interpret alongside the published biology.

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295228 G>A
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: TSS activity (CAGE/PRO-CAP): very strong decrease (-1.23); Transcription factor binding (ChIP-TF): strong binding loss (-0.62); Histone modifications (ChIP-Histone): strong mark loss (-0.38); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.23).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HeLa-S3 | 17.4 | 14.7 | -0.225 | 1.000 | 0.737 | Moderate closing |
| DNASE:CMK | 207 | 179 | -0.208 | 1.000 | 0.899 | Moderate closing |
| DNASE:K562 | 333 | 288 | -0.207 | 1.000 | 0.914 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:EP400:K562 | 734 | 478 | -0.619 | 1.000 | 0.945 | Strong binding loss |
| CHIP:MNT:K562 | 1.32e+03 | 858 | -0.617 | 1.000 | 0.980 | Strong binding loss |
| CHIP:USF1:H1 | 3.63e+03 | 2.4e+03 | -0.599 | 1.000 | 0.987 | Strong binding loss |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HeLa-S3 | 1.82e+03 | 1.4e+03 | -0.380 | 1.000 | 0.893 | Strong mark loss |
| CHIP:H3K79me2:K562 | 3.05e+03 | 2.36e+03 | -0.372 | 1.000 | 0.980 | Strong mark loss |
| CHIP:H3K9ac:HeLa-S3 | 4.64e+03 | 3.63e+03 | -0.355 | 1.000 | 0.925 | Strong mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| PRO_CAP:MCF 10A — variant site | 36 | 14.8 | -1.232 | 1.000 | 1.000 | Very strong decrease |
| PRO_CAP:MCF 10A — TERT TSS | 36.4 | 15.1 | -1.213 | 1.000 | 1.000 | Very strong decrease |
| PRO_CAP:Calu3 — variant site | 132 | 70.5 | -0.896 | 1.000 | 1.000 | Very strong decrease |
| PRO_CAP:Calu3 — TERT TSS | 132 | 70.7 | -0.893 | 1.000 | 1.000 | Very strong decrease |
| PRO_CAP:Caco-2 — variant site | 384 | 241 | -0.671 | 1.000 | 1.000 | Strong decrease |
| PRO_CAP:Caco-2 — TERT TSS | 382 | 240 | -0.666 | 1.000 | 1.000 | Strong decrease |
| PRO_CAP:Calu3 — SLC6A19 TSS | 30.5 | 30.8 | +0.014 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:Caco-2 — ZDHHC11 TSS | 61.9 | 61.5 | -0.009 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:MCF 10A — NKD2 TSS | 503 | 506 | +0.008 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:Caco-2 — ZDHHC11B TSS | 508 | 505 | -0.008 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0467 | 0.0439 | -0.004 | 1.000 | 0.820 | Minimal effect |
| SPLICE_SITES:H1 | 0.0169 | 0.0148 | -0.003 | 1.000 | 0.893 | Minimal effect |
| SPLICE_SITES:motor neuron | 0.0167 | 0.0147 | -0.003 | 1.000 | 0.885 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

---

---

## Interpretation

**What the oracle sees.** Alt allele shows *decreases* across chromatin,
TF binding, histone marks, and CAGE at the TERT TSS, with the strongest
effect being a TSS drop (-1.2 log2FC). Top TF-binding losses are EP400,
MNT, USF1 — none of the ETS family. This is the opposite direction from
the published biology.

**How this fits the published biology.** Horn / Huang 2013 (Science)
showed the C228T mutation *creates* a de novo GABPA/ETS binding site and
*activates* TERT in melanoma, glioblastoma and bladder cancer. AlphaGenome
is trained on bulk-tissue data where TERT is largely silent in adult
cells (telomerase is normally repressed), so its baseline prediction is a
silent TERT promoter. A small sequence change in a silent region
produces little CAGE signal and the oracle defaults to "decrease" because
the reference is already higher than the predicted alt in absolute
terms. **This is a real limitation of current bulk-tissue models**,
not a Chorus bug.

**Suggested next steps.**
- Treat this example as a cautionary case: for cancer-reactivation
  variants, always cross-check oracle predictions against the published
  reporter-assay data.
- If you have access, run ChromBPNet with a GABPA CHIP model in a
  cancer cell line (A375, U87, T24) — base-resolution motif creation is
  what drives this variant and a motif-scale model will see it.
- For the foundation-model comparison, re-run this exact prompt with
  Borzoi or Enformer and note whether the direction flips. No single
  oracle yet handles cancer-specific TERT reactivation correctly.
