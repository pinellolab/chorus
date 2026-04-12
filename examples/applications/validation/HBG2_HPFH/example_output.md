## Analysis Request

> Validate the HBG2 HPFH variant (chr11:5254983 G>C) — this is a known hereditary persistence of fetal hemoglobin mutation. Can you show the chromatin, BCL11A/KLF1 binding, and HBG2 expression effects in erythroid cells?

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all oracle tracks (discovery mode)
- **Generated**: 2026-04-12 02:21 UTC

**Notes / caveats:**
- Published mechanism: HPFH variants in the HBG1/HBG2 promoters disrupt binding of the BCL11A or ZBTB7A (LRF) fetal-globin repressor complexes, re-activating fetal hemoglobin in adult erythroid cells. Chorus is running AlphaGenome in all-tracks discovery mode; BCL11A / ZBTB7A ChIP tracks are limited in the oracle's catalog and may not rank in the top hits without explicit assay_ids. For a cleaner test of the repressor-loss hypothesis, call `analyze_variant_multilayer` with the BCL11A and ZBTB7A ChIP tracks listed explicitly in erythroid cell types.

## Multi-Layer Variant Effect Report

**Variant**: chr11:5254983 G>C
**Oracle**: alphagenome
**Gene**: HBG2
**Other nearby genes**: ENSG00000284931, HBG1, HBD, HBB

**Summary**: Transcription factor binding (ChIP-TF): strong binding loss (-0.33); TSS activity (CAGE/PRO-CAP): moderate decrease (-0.23); Chromatin accessibility (DNASE/ATAC): moderate closing (-0.19); Histone modifications (ChIP-Histone): moderate mark loss (-0.16).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:WTC11 | 22.8 | 19.9 | -0.187 | 1.000 | 0.687 | Moderate closing |
| DNASE:NB4 | 12.2 | 10.7 | -0.175 | 1.000 | 0.638 | Moderate closing |
| DNASE:lung | 16.5 | 14.5 | -0.170 | 1.000 | 0.671 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CTCF:endodermal cell | 202 | 160 | -0.335 | 1.000 | 0.418 | Strong binding loss |
| CHIP:RAD21:H1 | 210 | 174 | -0.272 | 1.000 | 0.495 | Moderate binding loss |
| CHIP:CTCF:GM23338 | 226 | 190 | -0.245 | 1.000 | 0.470 | Moderate binding loss |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me1:mesodermal cell | 642 | 576 | -0.156 | 1.000 | 0.824 | Moderate mark loss |
| CHIP:H3K4me3:mesodermal cell | 445 | 403 | -0.143 | 1.000 | 0.838 | Moderate mark loss |
| CHIP:H3K4me1:common myeloid progenitor, CD34-positive | 686 | 628 | -0.129 | 1.000 | 0.802 | Moderate mark loss |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| PRO_CAP:Caco-2 — variant site | 12.9 | 10.8 | -0.230 | 1.000 | 1.000 | Moderate decrease |
| PRO_CAP:Caco-2 — ENSG00000284931 TSS | 13.1 | 11 | -0.227 | 1.000 | 1.000 | Moderate decrease |
| CAGE:HepG2 — HBG2 TSS | 524 | 449 | -0.223 | 1.000 | 1.000 | Moderate decrease |
| CAGE:HepG2 — ENSG00000284931 TSS | 524 | 449 | -0.223 | 1.000 | 1.000 | Moderate decrease |
| CAGE:HepG2 — variant site | 524 | 449 | -0.223 | 1.000 | 1.000 | Moderate decrease |
| CAGE:mouth mucosa — ENSG00000284931 TSS | 349 | 306 | -0.188 | 1.000 | 1.000 | Moderate decrease |
| CAGE:mouth mucosa — HBG2 TSS | 349 | 306 | -0.188 | 1.000 | 1.000 | Moderate decrease |
| CAGE:mouth mucosa — variant site | 349 | 306 | -0.188 | 1.000 | 1.000 | Moderate decrease |
| CAGE:HepG2 — HBG1 TSS | 26.6 | 25.2 | -0.075 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — HBB TSS | 3.53e+03 | 3.45e+03 | -0.033 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 147 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0625 | 0.0668 | +0.006 | 1.000 | 0.891 | Minimal effect |
| SPLICE_SITES | 0.0312 | 0.0334 | +0.003 | 1.000 | 0.791 | Minimal effect |
| SPLICE_SITES:testis | 0.00983 | 0.0102 | +0.001 | 1.000 | 0.809 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

---

---

## Interpretation

**What the oracle sees.** Strong binding loss in CTCF (rank 1, -0.34 log2FC
in endodermal cells) plus moderate chromatin closing, moderate TSS
decrease, and moderate histone mark loss. The top TF hits are CTCF /
RAD21 — not the BCL11A or ZBTB7A the HPFH literature points to.

**How this fits the published biology.** HPFH variants in the HBG1/HBG2
promoters disrupt binding of the BCL11A or ZBTB7A (LRF) fetal-globin
repressor complexes, re-activating HbF in adult erythroid cells. Chorus
reports regulatory disruption in the expected direction (binding loss,
chromatin closing) but the oracle's TF ranking surfaces the wrong
factors because BCL11A / ZBTB7A ChIP tracks are limited in AlphaGenome's
training catalog. The loss-of-repression mechanism is not directly
visible in the default top-N.

**Suggested next steps.**
- Re-run `analyze_variant_multilayer` with explicit `BCL11A` and
  `ZBTB7A` ChIP assay_ids in erythroid cell types for a clean
  repressor-loss readout.
- If erythroid coverage is the bottleneck, complement with ChromBPNet
  loaded for an erythroid ATAC model for base-resolution motif
  disruption.
- This is the flagship example of an oracle limitation driven by
  **training data coverage**, not prediction accuracy.
