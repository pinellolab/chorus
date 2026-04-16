## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-16 18:22 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.47); TSS activity (CAGE/PRO-CAP): strong increase (+0.33); Histone modifications (ChIP-Histone): strong mark gain (+0.30); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.21).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 345 | 400 | +0.211 | ≥99th | 0.918 | Moderate opening |
| DNASE:MM.1S | 326 | 374 | +0.199 | ≥99th | 0.923 | Moderate opening |
| ATAC:GM19025 | 166 | 188 | +0.179 | ≥99th | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 160 | +0.465 | ≥99th | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 464 | 570 | +0.297 | ≥99th | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 756 | 923 | +0.287 | ≥99th | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.27e+03 | 4.02e+03 | +0.300 | ≥99th | 0.929 | Strong mark gain |
| CHIP:H3K27ac:OCI-LY3 | 777 | 956 | +0.298 | ≥99th | 0.891 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.46e+03 | 2.99e+03 | +0.284 | ≥99th | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — variant site | 81.8 | 103 | +0.331 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — TERT TSS | 81.8 | 103 | +0.331 | ≥99th | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 900 | 1.1e+03 | +0.286 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 901 | 1.1e+03 | +0.286 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.2 | 40.3 | +0.271 | ≥99th | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.2 | 40.3 | +0.271 | ≥99th | 1.000 | Moderate increase |
| CAGE:GM12878 — SLC6A19 TSS | 58.8 | 57.3 | -0.037 | ≥99th | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A19 TSS | 376 | 371 | -0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:GM12878 — NKD2 TSS | 384 | 380 | -0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:GM12878 — SLC12A7 TSS | 11 | 10.9 | -0.007 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0404 | 0.043 | +0.060 | ≥99th | 0.301 | Moderate increase |
| RNA:NCI-H460 — SLC6A18 (exons) | 0.00259 | 0.00273 | +0.040 | ≥99th | 0.113 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 2 | 2.06 | +0.032 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0124 | 0.0128 | +0.029 | ≥99th | 0.230 | Minimal effect |
| RNA:NCI-H460 — TERT (exons) | 414 | 426 | +0.029 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.00201 | 0.00204 | +0.010 | ≥99th | 0.162 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 593 | 600 | +0.010 | ≥99th | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.125 | 0.126 | +0.007 | ≥99th | 0.460 | Minimal effect |
| RNA:OCI-LY7 — SLC6A3 (exons) | 0.115 | 0.116 | +0.006 | ≥99th | 0.474 | Minimal effect |
| RNA:NCI-H460 — ZDHHC11B (exons) | 3 | 2.99 | -0.005 | ≤1st | 1.000 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0357 | 0.0377 | +0.003 | ≥99th | 0.800 | Minimal effect |
| SPLICE_SITES:H1 | 0.0172 | 0.0191 | +0.003 | ≥99th | 0.888 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0159 | 0.0177 | +0.003 | ≥99th | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
