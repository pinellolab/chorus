## Analysis Request

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

- **Tool**: `discover_variant`
- **Oracle**: enformer
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all Enformer tracks (discovery mode)
- **Generated**: 2026-04-17 07:07 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: enformer
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening (+1.24); Transcription factor binding (ChIP-TF): very strong binding gain (+1.13); Histone modifications (ChIP-Histone): very strong mark gain (+0.73); TSS activity (CAGE/PRO-CAP): strong increase (+0.51).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:LNCaP clone FGC | 0.66 | 2.91 | +1.236 | ≥99th | 0.863 | Very strong opening |
| DNASE:HeLa-S3 G1b phase | 3.75 | 9.54 | +1.149 | ≥99th | 0.940 | Very strong opening |
| DNASE:epithelial cell of proximal tubule | 1.78 | 4.98 | +1.108 | ≥99th | 0.916 | Very strong opening |
| DNASE:MCF 10A treated with 1 uM tamoxifen for 24 hours | 3.21 | 7.8 | +1.064 | ≥99th | 0.909 | Very strong opening |
| DNASE:placenta female embryo (105 days) | 4.19 | 9.8 | +1.058 | ≥99th | 0.916 | Very strong opening |
| DNASE:epithelial cell of prostate | 2.65 | 6.57 | +1.053 | ≥99th | 0.933 | Very strong opening |
| DNASE:MCF-7 | 8.93 | 19.5 | +1.047 | ≥99th | 0.963 | Very strong opening |
| DNASE:T47D | 6.94 | 15.2 | +1.029 | ≥99th | 0.934 | Very strong opening |
| DNASE:epithelial cell of esophagus | 3.53 | 8.17 | +1.016 | ≥99th | 0.921 | Very strong opening |
| DNASE:MCF-7 treated with 100 nM estradiol for 1 hour | 8.73 | 18.2 | +0.980 | ≥99th | 0.953 | Very strong opening |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:HNF4A:liver male adult (32 years) | 13.4 | 30.5 | +1.125 | ≥99th | 0.938 | Very strong binding gain |
| CHIP:RXRA:liver male adult (32 years) | 15.2 | 33.8 | +1.100 | ≥99th | 0.940 | Very strong binding gain |
| CHIP:HNF4A:liver female child (4 years) | 13.9 | 30.3 | +1.071 | ≥99th | 0.925 | Very strong binding gain |
| CHIP:FOS:MCF-7 | 21.9 | 43.6 | +0.961 | ≥99th | 0.989 | Very strong binding gain |
| CHIP:SP1:liver male adult (32 years) | 19.1 | 37.6 | +0.941 | ≥99th | 0.911 | Very strong binding gain |
| CHIP:STAT3:MCF 10A originated from MCF 10A treated with 1 uM afimoxifene for 36 hours | 28.7 | 54.5 | +0.902 | ≥99th | 0.955 | Very strong binding gain |
| CHIP:YY1:liver male adult (32 years) | 8.42 | 16.5 | +0.895 | ≥99th | 0.861 | Very strong binding gain |
| CHIP:STAT3:MCF 10A genetically modified using stable transfection treated with 1 uM afimoxifene for 12 hours | 28 | 51.7 | +0.859 | ≥99th | 0.970 | Very strong binding gain |
| CHIP:GATA3:MCF-7 | 37.8 | 67.6 | +0.825 | ≥99th | 0.998 | Very strong binding gain |
| CHIP:RXRA:liver female child (4 years) | 15.1 | 27.4 | +0.816 | ≥99th | 0.929 | Very strong binding gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:22Rv1 treated with 10 nM 17B-hydroxy-5a-androstan-3-one for 4 hours | 79.9 | 134 | +0.734 | ≥99th | 0.873 | Very strong mark gain |
| CHIP:H3K4me1:common myeloid progenitor, CD34-positive male adult (42 years) | 102 | 61.1 | -0.728 | ≥99th | 0.818 | Very strong mark loss |
| CHIP:H3K27ac:22Rv1 | 79.2 | 130 | +0.712 | ≥99th | 0.875 | Very strong mark gain |
| CHIP:H3K4me1:common myeloid progenitor, CD34-positive female adult (33 years) | 97 | 61 | -0.660 | ≥99th | 0.831 | Strong mark loss |
| CHIP:H3K27ac:liver male adult (31 year) | 58.3 | 91.1 | +0.635 | ≥99th | 0.825 | Strong mark gain |
| CHIP:H3K27ac:psoas muscle female adult (30 years) | 26.4 | 41.6 | +0.635 | ≥99th | 0.813 | Strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis female adult (53 years) | 92.6 | 142 | +0.614 | ≥99th | 0.853 | Strong mark gain |
| CHIP:H3K4me3:liver male adult (78 years) | 30.6 | 47.1 | +0.607 | ≥99th | 0.761 | Strong mark gain |
| CHIP:H3K4me1:common myeloid progenitor, CD34-positive female adult (27 years) | 97.8 | 64.2 | -0.599 | ≥99th | 0.805 | Strong mark loss |
| CHIP:H3K27ac:gastrocnemius medialis male adult (37 years) | 106 | 161 | +0.599 | ≥99th | 0.852 | Strong mark gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:breast carcinoma cell line:MDA-MB-453 — variant site | 4.01 | 6.12 | +0.505 | ≥99th | 0.906 | Strong increase |
| CAGE:placenta, adult, pool1 — variant site | 1.73 | 2.73 | +0.448 | ≥99th | 0.876 | Strong increase |
| CAGE:breast carcinoma cell line:MCF7 — variant site | 6.46 | 9.03 | +0.428 | ≥99th | 0.926 | Strong increase |
| CAGE:kidney, adult, pool1 — variant site | 3.16 | 4.59 | +0.426 | ≥99th | 0.883 | Strong increase |
| CAGE:liver, adult, pool1 — variant site | 0.283 | 0.693 | +0.400 | ≥99th | 0.816 | Strong increase |
| CAGE:Skeletal muscle cells differentiated into Myotubes - multinucleated, — variant site | 2.05 | 1.37 | -0.367 | ≥99th | 0.879 | Strong decrease |
| CAGE:Astrocyte - cerebellum, — variant site | 3.35 | 2.43 | -0.343 | ≥99th | 0.895 | Strong decrease |
| CAGE:endometrioid adenocarcinoma cell line:JHUEM-1 — variant site | 3.26 | 4.33 | +0.323 | ≥99th | 0.908 | Strong increase |
| CAGE:colon carcinoma cell line:CACO-2 — variant site | 2.49 | 3.36 | +0.322 | ≥99th | 0.904 | Strong increase |
| CAGE:kidney, fetal, pool1 — variant site | 4.28 | 5.59 | +0.321 | ≥99th | 0.883 | Strong increase |
| _…showing top 10 of 48 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
