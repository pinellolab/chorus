## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-13 11:48 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.47); TSS activity (CAGE/PRO-CAP): strong increase (+0.34); Histone modifications (ChIP-Histone): moderate mark gain (+0.30); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.21).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 347 | 401 | +0.209 | 1.000 | 0.918 | Moderate opening |
| DNASE:MM.1S | 327 | 374 | +0.196 | 1.000 | 0.923 | Moderate opening |
| ATAC:GM19025 | 167 | 188 | +0.177 | 1.000 | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 160 | +0.468 | 1.000 | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 465 | 569 | +0.291 | 1.000 | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 760 | 923 | +0.279 | 1.000 | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.29e+03 | 4.04e+03 | +0.298 | 1.000 | 0.929 | Moderate mark gain |
| CHIP:H3K27ac:OCI-LY3 | 782 | 959 | +0.294 | 1.000 | 0.892 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.46e+03 | 2.99e+03 | +0.283 | 1.000 | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — variant site | 82.2 | 104 | +0.337 | 1.000 | 1.000 | Strong increase |
| CAGE:GM12878 — TERT TSS | 82.2 | 104 | +0.336 | 1.000 | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 901 | 1.1e+03 | +0.294 | 1.000 | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 902 | 1.11e+03 | +0.294 | 1.000 | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.1 | 40.4 | +0.278 | 1.000 | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.1 | 40.3 | +0.278 | 1.000 | 1.000 | Moderate increase |
| CAGE:K562 — SLC6A19 TSS | 365 | 368 | +0.013 | 1.000 | 1.000 | Minimal effect |
| CAGE:GM12878 — SLC6A19 TSS | 57 | 57.5 | +0.011 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — NKD2 TSS | 608 | 611 | +0.008 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A18 TSS | 0.819 | 0.829 | +0.008 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0409 | 0.0433 | +0.056 | 1.000 | 0.302 | Moderate increase |
| RNA:OCI-LY7 — TERT (exons) | 2 | 2.06 | +0.028 | 1.000 | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0124 | 0.0127 | +0.026 | 1.000 | 0.230 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.00231 | 0.00228 | -0.010 | -0.985 | 0.168 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 594 | 600 | +0.010 | 1.000 | 1.000 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.414 | 0.41 | -0.009 | -1.000 | 0.865 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.00202 | 0.00204 | +0.007 | 1.000 | 0.162 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.126 | 0.127 | +0.005 | 1.000 | 0.463 | Minimal effect |
| RNA:OCI-LY7 — SLC6A3 (exons) | 0.0146 | 0.0145 | -0.004 | -0.960 | 0.236 | Minimal effect |
| RNA:OCI-LY7 — MRPL36 (exons) | 1.11 | 1.1 | -0.003 | -0.930 | 1.000 | Minimal effect |
| _…showing top 10 of 28 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES:H1 | 0.0172 | 0.0191 | +0.003 | 1.000 | 0.888 | Minimal effect |
| SPLICE_SITES | 0.0359 | 0.0378 | +0.003 | 1.000 | 0.800 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0159 | 0.0177 | +0.003 | 1.000 | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
