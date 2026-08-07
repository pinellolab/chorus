## Analysis Request

> Score chr1:109274968 G>T using ChromBPNet DNASE model in HepG2. Gene: SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: chrombpnet
- **Normalizer**: per-track background CDFs
- **Tracks requested**: DNASE:HepG2
- **Generated**: 2026-08-07 11:51 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: chrombpnet
**Gene**: SORT1
**Other nearby genes**: CELSR2

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.38, DNASE:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 287 | 747 | +1.376 | ≥99th | 0.906 | Very strong opening |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
