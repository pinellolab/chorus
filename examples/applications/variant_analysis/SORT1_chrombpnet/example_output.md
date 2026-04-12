## Analysis Request

> Load ChromBPNet with the HepG2 ATAC model and score rs12740374 (chr1:109274968 G>T) — I want base-resolution chromatin effects around the CEBP motif.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: chrombpnet
- **Normalizer**: per-track background CDFs
- **Tracks requested**: ATAC:HepG2
- **Generated**: 2026-04-12 02:21 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: chrombpnet
**Gene**: SORT1
**Other nearby genes**: CELSR2

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate closing (-0.11).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:HepG2 | 687 | 636 | -0.111 | 0.967 | 0.820 | Moderate closing |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.

---

---

## Interpretation

**What the oracle sees.** ChromBPNet scores the variant on one specific
assay/cell-type combination (HepG2 ATAC). The effect is modest and
layer-limited to chromatin accessibility because ChromBPNet is a
single-assay base-resolution model.

**How this fits the published biology.** A modest HepG2 chromatin
effect is consistent with the classic C/EBP site creation story:
ChromBPNet captures the motif disruption at base resolution but only
reports one layer. Treat this as a complement — not a replacement — to
the multi-layer AlphaGenome result on the same variant.

**Suggested next steps.**
- Load `chrombpnet` with `assay="CHIP"`, `cell_type="HepG2"`,
  `TF="CEBPA"` for a direct motif-creation readout.
- Use this model for high-throughput VCF triage, then escalate the top
  hits to AlphaGenome for multi-layer analysis.
