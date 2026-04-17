## Analysis Request

> Score chr1:109274968 G>T using ChromBPNet ATAC model in HepG2. Gene: SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: chrombpnet
- **Normalizer**: per-track background CDFs
- **Tracks requested**: ATAC:HepG2
- **Generated**: 2026-04-17 07:18 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: chrombpnet
**Gene**: SORT1
**Other nearby genes**: CELSR2

**Summary**: Chromatin accessibility (DNASE/ATAC): moderate closing (-0.11).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| ATAC:HepG2 | 687 | 636 | -0.111 | 0.97 | 0.820 | Moderate closing |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
