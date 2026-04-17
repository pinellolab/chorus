## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-17 20:09 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.47); TSS activity (CAGE/PRO-CAP): strong increase (+0.34); Histone modifications (ChIP-Histone): strong mark gain (+0.30); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.21).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 346 | 401 | +0.212 | ≥99th | 0.918 | Moderate opening |
| DNASE:MM.1S | 325 | 374 | +0.200 | ≥99th | 0.923 | Moderate opening |
| ATAC:GM19025 | 166 | 188 | +0.182 | ≥99th | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 160 | +0.465 | ≥99th | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 464 | 571 | +0.297 | ≥99th | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 756 | 923 | +0.289 | ≥99th | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.27e+03 | 4.03e+03 | +0.303 | ≥99th | 0.929 | Strong mark gain |
| CHIP:H3K27ac:OCI-LY3 | 778 | 955 | +0.296 | ≥99th | 0.891 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.46e+03 | 2.99e+03 | +0.283 | ≥99th | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — variant site | 82 | 104 | +0.340 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — TERT TSS | 82 | 104 | +0.340 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 899 | 1.1e+03 | +0.293 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 900 | 1.1e+03 | +0.293 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.3 | 40.5 | +0.277 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.2 | 40.5 | +0.277 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — ZDHHC11B TSS | 19 | 18.8 | -0.008 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A18 TSS | 0.821 | 0.83 | +0.008 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A19 TSS | 369 | 370 | +0.006 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC12A7 TSS | 9.24 | 9.2 | -0.006 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0407 | 0.0428 | +0.049 | ≥99th | 0.302 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 1.99 | 2.06 | +0.033 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0124 | 0.0128 | +0.027 | ≥99th | 0.230 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 594 | 600 | +0.010 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.00201 | 0.00203 | +0.008 | ≥99th | 0.162 | Minimal effect |
| RNA:OCI-LY7 — SLC6A3 (exons) | 0.0144 | 0.0145 | +0.005 | ≥99th | 0.236 | Minimal effect |
| RNA:OCI-LY7 — MRPL36 (exons) | 1.1 | 1.1 | +0.005 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.411 | 0.409 | -0.005 | ≤1st | 0.862 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.00229 | 0.00227 | -0.005 | ≤1st | 0.168 | Minimal effect |
| RNA:OCI-LY7 — CLPTM1L (exons) | 0.659 | 0.662 | +0.004 | ≥99th | 0.983 | Minimal effect |
| _…showing top 10 of 28 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0358 | 0.0378 | +0.003 | ≥99th | 0.800 | Minimal effect |
| SPLICE_SITES:H1 | 0.0171 | 0.0191 | +0.003 | ≥99th | 0.888 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0158 | 0.0176 | +0.003 | ≥99th | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
