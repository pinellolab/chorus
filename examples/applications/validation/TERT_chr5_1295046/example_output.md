## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-16 19:36 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.46); TSS activity (CAGE/PRO-CAP): strong increase (+0.34); Histone modifications (ChIP-Histone): strong mark gain (+0.30); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.22).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 345 | 401 | +0.215 | ≥99th | 0.918 | Moderate opening |
| DNASE:MM.1S | 325 | 374 | +0.202 | ≥99th | 0.923 | Moderate opening |
| ATAC:GM19025 | 166 | 188 | +0.182 | ≥99th | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 159 | +0.463 | ≥99th | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 465 | 570 | +0.293 | ≥99th | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 756 | 923 | +0.287 | ≥99th | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.27e+03 | 4.04e+03 | +0.303 | ≥99th | 0.929 | Strong mark gain |
| CHIP:H3K27ac:OCI-LY3 | 780 | 957 | +0.295 | ≥99th | 0.892 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.45e+03 | 2.99e+03 | +0.288 | ≥99th | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — TERT TSS | 81.6 | 104 | +0.345 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 81.6 | 104 | +0.344 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 897 | 1.1e+03 | +0.296 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 898 | 1.1e+03 | +0.296 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.1 | 40.8 | +0.294 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.1 | 40.8 | +0.294 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — SLC6A19 TSS | 56.6 | 59.1 | +0.063 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A19 TSS | 366 | 377 | +0.040 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A18 TSS | 0.816 | 0.831 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:GM12878 — NKD2 TSS | 384 | 381 | -0.011 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0401 | 0.0429 | +0.065 | ≥99th | 0.301 | Moderate increase |
| RNA:OCI-LY7 — TERT (exons) | 1.99 | 2.06 | +0.034 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0123 | 0.0127 | +0.032 | ≥99th | 0.230 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.002 | 0.00204 | +0.013 | ≥99th | 0.162 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 593 | 600 | +0.011 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.126 | 0.127 | +0.007 | ≥99th | 0.461 | Minimal effect |
| RNA:OCI-LY7 — CLPTM1L (exons) | 0.658 | 0.662 | +0.006 | ≥99th | 0.983 | Minimal effect |
| RNA:OCI-LY7 — NKD2 (exons) | 5.88 | 5.85 | -0.005 | ≤1st | 1.000 | Minimal effect |
| RNA:OCI-LY7 — NDUFS6 (exons) | 0.657 | 0.659 | +0.003 | ≥99th | 0.975 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.41 | 0.409 | -0.003 | ≤1st | 0.862 | Minimal effect |
| _…showing top 10 of 28 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0358 | 0.0379 | +0.003 | ≥99th | 0.800 | Minimal effect |
| SPLICE_SITES:H1 | 0.0172 | 0.0191 | +0.003 | ≥99th | 0.888 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0159 | 0.0177 | +0.003 | ≥99th | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
