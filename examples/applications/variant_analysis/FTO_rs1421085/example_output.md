## Analysis Request

> Analyze the obesity-associated FTO variant rs1421085 (chr16:53767042 T>C) in adipocyte tracks. IRX3/IRX5 are the proposed target genes — show me the chromatin and TF binding effects.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all oracle tracks (discovery mode)
- **Generated**: 2026-04-12 02:21 UTC

**Notes / caveats:**
- Published mechanism: Claussnitzer et al. NEJM 2015 showed that the risk allele (C) disrupts an ARID5B repressor motif, de-repressing IRX3 and IRX5 in adipocyte progenitors. ARID5B ChIP data is not well represented in AlphaGenome's training set, so the direct ARID5B binding-loss signal may be absent from the top tracks. IRX3/IRX5 expression effects are in the gene expression layer — check the RNA rows for those genes directly.

## Multi-Layer Variant Effect Report

**Variant**: chr16:53767042 T>C
**Oracle**: alphagenome
**Gene**: FTO
**Other nearby genes**: RPGRIP1L, AKTIP, RBL2, IRX3

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate closing (-0.23); Transcription factor binding (ChIP-TF): moderate binding loss (-0.18); TSS activity (CAGE/PRO-CAP): moderate decrease (-0.11).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of dermis | 60.3 | 51.1 | -0.234 | 1.000 | 0.854 | Moderate closing |
| DNASE:fibroblast of skin of left biceps | 36.7 | 31.3 | -0.225 | 1.000 | 0.832 | Moderate closing |
| DNASE:fibroblast of skin of left quadriceps | 39 | 33.3 | -0.221 | 1.000 | 0.831 | Moderate closing |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:RCOR1:SK-N-SH | 1.21e+03 | 1.06e+03 | -0.179 | 1.000 | 0.901 | Moderate binding loss |
| CHIP:TCF12:SK-N-SH | 1.36e+03 | 1.2e+03 | -0.175 | 1.000 | 0.993 | Moderate binding loss |
| CHIP:CHD2:SK-N-SH | 1.03e+03 | 932 | -0.144 | 1.000 | 0.884 | Moderate binding loss |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:muscle of leg | 1.36e+03 | 1.27e+03 | -0.098 | 1.000 | 0.875 | Minimal effect |
| CHIP:H3K27ac:GM23248 | 3.9e+03 | 3.67e+03 | -0.089 | 1.000 | 0.971 | Minimal effect |
| CHIP:H3K27ac:endothelial cell of umbilical vein | 4.73e+03 | 4.45e+03 | -0.087 | 1.000 | 0.982 | Minimal effect |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| PRO_CAP:endothelial cell of umbilical vein — variant site | 32.1 | 29.7 | -0.111 | 1.000 | 1.000 | Moderate decrease |
| PRO_CAP:MCF 10A — variant site | 5.75 | 5.29 | -0.101 | 1.000 | 1.000 | Moderate decrease |
| PRO_CAP:Caco-2 — variant site | 4.25 | 4.6 | +0.092 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:endothelial cell of umbilical vein — FTO TSS | 489 | 492 | +0.008 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:endothelial cell of umbilical vein — RPGRIP1L TSS | 494 | 497 | +0.008 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:endothelial cell of umbilical vein — AKTIP TSS | 1.4e+03 | 1.4e+03 | -0.007 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:Caco-2 — IRX3 TSS | 384 | 385 | +0.006 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:MCF 10A — AKTIP TSS | 1.48e+03 | 1.48e+03 | -0.005 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:MCF 10A — RPGRIP1L TSS | 689 | 691 | +0.003 | 1.000 | 1.000 | Minimal effect |
| PRO_CAP:MCF 10A — FTO TSS | 677 | 679 | +0.003 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 18 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.28 | 0.272 | -0.009 | 1.000 | 0.906 | Minimal effect |
| SPLICE_SITES | 0.0462 | 0.0443 | -0.003 | 1.000 | 0.882 | Minimal effect |
| SPLICE_SITES:dorsolateral prefrontal cortex | 0.01 | 0.00973 | -0.000 | 1.000 | 0.815 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

---

---

## Interpretation

**What the oracle sees.** Effects are modest (top raw scores in the 0.1
to 0.3 log2FC range) and span multiple TF, chromatin, and histone tracks
in neuroblastoma and several other lineages. IRX3 / IRX5 expression
changes appear in the RNA layer but in the smallest-effect category.
None of the layers reach "very strong" by our magnitude-gated labels.

**How this fits the published biology.** Claussnitzer et al.
(NEJM 2015) showed that the risk allele (C) disrupts an ARID5B repressor
motif in adipocyte progenitors, de-repressing IRX3 and IRX5 and driving
a thermogenesis-to-lipid-storage switch. Chorus agrees with the modesty
of the effect but does not recover the ARID5B mechanism — ARID5B ChIP is
not well represented in AlphaGenome's training corpus, and adipocyte
progenitor tracks are sparse. This is a case where the oracle's cell
type coverage lags what the paper used.

**Suggested next steps.**
- Use `discover_variant_cell_types` to confirm whether any of the
  available adipose / preadipocyte tracks in AlphaGenome show a stronger
  signal than the current default top hits.
- Score the variant with a LegNet MPRA model or a ChromBPNet adipocyte
  ATAC model for a second opinion.
- If IRX3 / IRX5 expression is the readout you care about, query
  `analyze_variant_multilayer` with `gene_name="IRX3"` (and separately
  "IRX5") so the RNA layer is scored at those specific TSSs.
