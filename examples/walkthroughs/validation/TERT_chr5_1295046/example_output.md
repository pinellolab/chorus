## Analysis Request

> Validate the TERT chr5:1295046 T>G variant from the AlphaGenome paper. Score across all tracks in discovery mode. Gene is TERT.

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all tracks (discovery mode)
- **Generated**: 2026-04-21 13:31 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr5:1295046 T>G
**Oracle**: alphagenome
**Gene**: TERT
**Other nearby genes**: CLPTM1L, SLC6A18, SLC6A19, SLC6A3

**Summary**: TSS activity (CAGE/PRO-CAP): very strong increase (+2.62, PRO_CAP:MCF 10A); Transcription factor binding (ChIP-TF): very strong binding gain (+2.12, CHIP:GABPB1:K562); Histone modifications (ChIP-Histone): very strong mark gain (+2.02, CHIP:H3K4me3:AG09319); Chromatin accessibility (DNASE/ATAC): very strong opening (+1.24, DNASE:EH); Gene expression (RNA-seq): very strong increase (+1.17, RNA:HCT116).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:EH | 24.9 | 60 | +1.237 | ≥99th | 0.811 | Very strong opening |
| DNASE:immature natural killer cell | 105 | 249 | +1.234 | ≥99th | 0.886 | Very strong opening |
| DNASE:HCEC 1CT | 24.3 | 58.2 | +1.226 | ≥99th | 0.787 | Very strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:GABPB1:K562 | 366 | 1.6e+03 | +2.124 | ≥99th | 0.857 | Very strong binding gain |
| CHIP:ELF1:K562 | 66.7 | 239 | +1.825 | ≥99th | 0.040 | Very strong binding gain |
| CHIP:ELF4:K562 | 56.8 | 202 | +1.817 | ≥99th | 0.041 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:AG09319 | 1.65e+03 | 6.7e+03 | +2.025 | ≥99th | 0.864 | Very strong mark gain |
| CHIP:H3K4me3:cardiac muscle cell | 3.72e+03 | 1.37e+04 | +1.881 | ≥99th | 0.876 | Very strong mark gain |
| CHIP:H3K4me3:fibroblast of pulmonary artery | 832 | 2.92e+03 | +1.808 | ≥99th | 0.855 | Very strong mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| PRO_CAP:MCF 10A — variant site | 47.8 | 298 | +2.616 | ≥99th | 1.000 | Very strong increase |
| PRO_CAP:MCF 10A — TERT TSS | 47.9 | 299 | +2.614 | ≥99th | 1.000 | Very strong increase |
| CAGE:mesenchymal stem cell of Wharton's jelly — variant site | 26.9 | 118 | +2.094 | ≥99th | 1.000 | Very strong increase |
| CAGE:mesenchymal stem cell of Wharton's jelly — TERT TSS | 27 | 119 | +2.093 | ≥99th | 1.000 | Very strong increase |
| CAGE:natural killer cell — TERT TSS | 20.3 | 88.2 | +2.066 | ≥99th | 1.000 | Very strong increase |
| CAGE:natural killer cell — variant site | 20.3 | 88.1 | +2.065 | ≥99th | 1.000 | Very strong increase |
| PRO_CAP:MCF 10A — SLC6A18 TSS | 0.12 | 0.139 | +0.024 | ≥99th | 0.890 | Minimal effect |
| CAGE:natural killer cell — SLC6A3 TSS | 5.11 | 5.04 | -0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:mesenchymal stem cell of Wharton's jelly — SLC12A7 TSS | 573 | 568 | -0.013 | ≥99th | 1.000 | Minimal effect |
| CAGE:natural killer cell — NKD2 TSS | 4.27 | 4.23 | -0.012 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 45 — see `example_output.json` for the full set_ | | | | | | |

#### Gene expression (RNA-seq)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| RNA:HCT116 — SLC6A18 (exons) | 0.00935 | 0.0323 | +1.170 | ≥99th | 0.181 | Very strong increase |
| RNA:NCI-H460 — SLC6A18 (exons) | 0.00189 | 0.00603 | +0.888 | ≥99th | 0.104 | Very strong increase |
| RNA:HT1080 — SLC6A18 (exons) | 0.00298 | 0.00854 | +0.873 | ≥99th | 0.135 | Very strong increase |
| RNA:HT1080 — TERT (exons) | 133 | 256 | +0.654 | ≥99th | 1.000 | Strong increase |
| RNA:HCT116 — TERT (exons) | 189 | 316 | +0.515 | ≥99th | 1.000 | Strong increase |
| RNA:NCI-H460 — TERT (exons) | 434 | 682 | +0.452 | ≥99th | 1.000 | Strong increase |
| RNA:HCT116 — SLC6A19 (exons) | 0.00108 | 0.00149 | +0.181 | ≥99th | 0.117 | Moderate increase |
| RNA:NCI-H460 — SLC6A19 (exons) | 0.000837 | 0.00106 | +0.113 | ≥99th | 0.086 | Moderate increase |
| RNA:HT1080 — SLC6A19 (exons) | 0.00129 | 0.00154 | +0.107 | ≥99th | 0.104 | Moderate increase |
| RNA:HT1080 — SLC6A3 (exons) | 0.183 | 0.185 | +0.008 | ≥99th | 0.454 | Minimal effect |
| _…showing top 10 of 42 — see `example_output.json` for the full set_ | | | | | | |

#### Splicing (splice sites)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| SPLICE_SITES | 0.0399 | 0.0664 | +0.036 | ≥99th | 0.809 | Minimal effect |
| SPLICE_SITES:HFFc6 | 0.00997 | 0.0189 | +0.013 | ≥99th | 0.850 | Minimal effect |
| SPLICE_SITES:HT1080 | 0.00914 | 0.0169 | +0.011 | ≥99th | 0.882 | Minimal effect |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
