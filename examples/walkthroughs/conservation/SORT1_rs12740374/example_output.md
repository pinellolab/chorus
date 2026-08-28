## Analysis Request

> Score chr1:109274968 G>T using ChromBPNet DNASE model in HepG2. Gene: SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: chrombpnet
- **Normalizer**: per-track background CDFs
- **Tracks requested**: DNASE:HepG2
- **Generated**: 2026-08-28 13:03 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: chrombpnet
**Gene**: SORT1
**Other nearby genes**: CELSR2

**Summary**: Strongest effect per layer anywhere in the prediction window (not necessarily SORT1's own track). Chromatin accessibility (DNASE/ATAC): very strong opening (+1.38, DNASE:HepG2).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 287 | 747 | +1.376 | ≥99th (1.18× null max) | 0.885 | Very strong opening |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **`N× null max`**: the effect exceeded *every* sampled background effect for that track, so the percentile is clamped and cannot rank it further. The multiplier gives the distance to that ceiling — `1.11×` is 11% beyond the most extreme background effect for that track. Common for variants that create or destroy a complete transcription-factor motif, which even a regulatory-region background rarely contains.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
