## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-21 05:02 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.46, CHIP:E2F1:K562); TSS activity (CAGE/PRO-CAP): strong increase (+0.35, CAGE:GM12878); Histone modifications (ChIP-Histone): strong mark gain (+0.30, CHIP:H3K27ac:GM12878); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.22, DNASE:GM12865).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 345 | 401 | +0.217 | ≥99th | 0.918 | Moderate opening |
| DNASE:MM.1S | 325 | 374 | +0.202 | ≥99th | 0.923 | Moderate opening |
| ATAC:GM19025 | 166 | 188 | +0.182 | ≥99th | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 159 | +0.459 | ≥99th | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 464 | 568 | +0.292 | ≥99th | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 757 | 922 | +0.284 | ≥99th | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.26e+03 | 4.03e+03 | +0.304 | ≥99th | 0.929 | Strong mark gain |
| CHIP:H3K27ac:OCI-LY3 | 778 | 954 | +0.294 | ≥99th | 0.891 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.45e+03 | 2.99e+03 | +0.286 | ≥99th | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — variant site | 81.6 | 104 | +0.346 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — TERT TSS | 81.6 | 104 | +0.346 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 896 | 1.1e+03 | +0.295 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 897 | 1.1e+03 | +0.295 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.1 | 40.7 | +0.289 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.1 | 40.6 | +0.289 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — SLC6A19 TSS | 58.1 | 59.2 | +0.028 | ≥99th | 1.000 | Minimal effect |
| CAGE:GM12878 — SLC12A7 TSS | 728 | 718 | -0.020 | ≥99th | 1.000 | Minimal effect |
| CAGE:GM12878 — SLC12A7 TSS | 10.9 | 10.9 | -0.009 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A19 TSS | 373 | 375 | +0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0405 | 0.0429 | +0.058 | ≥99th | 0.301 | Moderate increase |
| RNA:NCI-H460 — SLC6A18 (exons) | 0.00259 | 0.00272 | +0.033 | ≥99th | 0.113 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 1.99 | 2.06 | +0.033 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0124 | 0.0128 | +0.030 | ≥99th | 0.230 | Minimal effect |
| RNA:NCI-H460 — TERT (exons) | 414 | 426 | +0.029 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 593 | 600 | +0.012 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.126 | 0.127 | +0.011 | ≥99th | 0.461 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.00201 | 0.00204 | +0.010 | ≥99th | 0.162 | Minimal effect |
| RNA:OCI-LY7 — CLPTM1L (exons) | 0.659 | 0.663 | +0.006 | ≥99th | 0.983 | Minimal effect |
| RNA:NCI-H460 — ZDHHC11B (exons) | 3.02 | 3 | -0.006 | ≤1st | 1.000 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0356 | 0.0379 | +0.003 | ≥99th | 0.800 | Minimal effect |
| SPLICE_SITES:H1 | 0.0171 | 0.0191 | +0.003 | ≥99th | 0.888 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0158 | 0.0177 | +0.003 | ≥99th | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
