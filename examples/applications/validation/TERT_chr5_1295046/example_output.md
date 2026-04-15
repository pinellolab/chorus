## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-15 05:24 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: Transcription factor binding (ChIP-TF): strong binding gain (+0.46); TSS activity (CAGE/PRO-CAP): strong increase (+0.33); Histone modifications (ChIP-Histone): strong mark gain (+0.30); Chromatin accessibility (DNASE/ATAC): moderate opening (+0.22).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:GM12865 | 346 | 402 | +0.215 | 1.000 | 0.918 | Moderate opening |
| DNASE:MM.1S | 325 | 375 | +0.204 | 1.000 | 0.923 | Moderate opening |
| ATAC:GM19025 | 166 | 189 | +0.180 | 1.000 | 0.902 | Moderate opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:E2F1:K562 | 115 | 158 | +0.460 | 1.000 | 0.065 | Strong binding gain |
| CHIP:LIN54:HepG2 | 464 | 566 | +0.289 | 1.000 | 0.869 | Moderate binding gain |
| CHIP:E2F1:MCF-7 | 755 | 922 | +0.288 | 1.000 | 0.884 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:GM12878 | 3.28e+03 | 4.04e+03 | +0.304 | 1.000 | 0.929 | Strong mark gain |
| CHIP:H3K27ac:OCI-LY3 | 782 | 960 | +0.297 | 1.000 | 0.892 | Moderate mark gain |
| CHIP:H3K27ac:MM.1S | 2.46e+03 | 2.99e+03 | +0.283 | 1.000 | 0.955 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:GM12878 — variant site | 82.1 | 104 | +0.334 | 1.000 | 1.000 | Strong increase |
| CAGE:GM12878 — TERT TSS | 82.1 | 104 | +0.334 | 1.000 | 1.000 | Strong increase |
| CAGE:GM12878 — variant site | 898 | 1.1e+03 | +0.297 | 1.000 | 1.000 | Moderate increase |
| CAGE:GM12878 — TERT TSS | 899 | 1.1e+03 | +0.297 | 1.000 | 1.000 | Moderate increase |
| CAGE:K562 — variant site | 33.2 | 40.3 | +0.273 | 1.000 | 1.000 | Moderate increase |
| CAGE:K562 — TERT TSS | 33.2 | 40.3 | +0.273 | 1.000 | 1.000 | Moderate increase |
| CAGE:GM12878 — SLC6A19 TSS | 57 | 58.3 | +0.032 | 1.000 | 1.000 | Minimal effect |
| CAGE:K562 — SLC6A19 TSS | 368 | 373 | +0.018 | 1.000 | 1.000 | Minimal effect |
| CAGE:GM12878 — NKD2 TSS | 2.12 | 2.11 | -0.008 | 1.000 | 1.000 | Minimal effect |
| CAGE:GM12878 — ZDHHC11B TSS | 26.1 | 26 | -0.007 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:colonic mucosa — TERT (exons) | 0.00915 | 0.00832 | -0.085 | -1.000 | 0.154 | Moderate decrease |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0407 | 0.0433 | +0.062 | 1.000 | 0.302 | Moderate increase |
| RNA:OCI-LY7 — TERT (exons) | 1.99 | 2.06 | +0.033 | 1.000 | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A18 (exons) | 0.0124 | 0.0129 | +0.030 | 1.000 | 0.231 | Minimal effect |
| RNA:OCI-LY7 — TERT (exons) | 592 | 599 | +0.012 | 1.000 | 1.000 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.00201 | 0.00204 | +0.010 | 1.000 | 0.162 | Minimal effect |
| RNA:OCI-LY7 — SLC6A19 (exons) | 0.126 | 0.127 | +0.009 | 1.000 | 0.461 | Minimal effect |
| RNA:colonic mucosa — SLC6A18 (exons) | 0.422 | 0.425 | +0.008 | 1.000 | 0.722 | Minimal effect |
| RNA:colonic mucosa — SLC6A19 (exons) | 728 | 734 | +0.008 | 1.000 | 1.000 | Minimal effect |
| RNA:OCI-LY7 — ZDHHC11B (exons) | 0.412 | 0.409 | -0.006 | -0.987 | 0.862 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0357 | 0.0382 | +0.003 | 1.000 | 0.800 | Minimal effect |
| SPLICE_SITES:H1 | 0.0172 | 0.0191 | +0.003 | 1.000 | 0.888 | Minimal effect |
| SPLICE_SITES:mesendoderm | 0.0159 | 0.0177 | +0.003 | 1.000 | 0.889 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
