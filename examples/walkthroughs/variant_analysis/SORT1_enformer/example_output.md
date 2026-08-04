## Analysis Request

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

- **Tool**: `discover_variant`
- **Oracle**: enformer
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all Enformer tracks (discovery mode)
- **Generated**: 2026-08-04 04:35 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: enformer
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+4.37, CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human…); Chromatin accessibility (DNASE/ATAC): very strong opening (+2.25, DNASE:fibroblast of lung); Histone modifications (ChIP-Histone): very strong mark gain (+1.89, CHIP:H3K4me1:neutrophil male); TSS activity (CAGE/PRO-CAP): very strong increase (+1.40, CAGE:liver, adult, pool1).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of lung | 0.838 | 7.73 | +2.247 | ≥99th | 0.857 | Very strong opening |
| DNASE:amniotic epithelial cell | 2.35 | 14.3 | +2.193 | ≥99th | 0.898 | Very strong opening |
| DNASE:fibroblast of villous mesenchyme | 2.01 | 11 | +1.997 | ≥99th | 0.883 | Very strong opening |
| DNASE:HL-60 | 1.69 | 9.52 | +1.969 | ≥99th | 0.868 | Very strong opening |
| DNASE:NB4 | 1.8 | 9.09 | +1.851 | ≥99th | 0.876 | Very strong opening |
| DNASE:WI38 genetically modified using stable transfection originated from WI38 | 2.57 | 11.8 | +1.843 | ≥99th | 0.904 | Very strong opening |
| ATAC:BM1137-GMP2-mid-ATAC-1 / Bone Marrow CD34+ / GMP-B | 11.7 | 35.2 | +1.512 | ≥99th | 0.855 | Very strong opening |
| ATAC:BM1137-GMP1-low-ATAC-2 / Bone Marrow CD34+ / GMP-A | 6.29 | 18.9 | +1.446 | ≥99th | 0.848 | Very strong opening |
| ATAC:BM1137-GMP3-high-ATAC-2 / Bone Marrow CD34+ / GMP-C | 13.1 | 37.3 | +1.442 | ≥99th | 0.874 | Very strong opening |
| ATAC:BM0106-UNK-ATAC-2 / Bone Marrow CD34+ / UNK | 7.12 | 19.1 | +1.310 | ≥99th | 0.848 | Very strong opening |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 2.56 | 72.7 | +4.372 | ≥99th | 0.771 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_LowDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 3.99 | 97.9 | +4.310 | ≥99th | 0.793 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_noDMI / hMSC / Human Mesenchymal Stem Cells | 7.06 | 106 | +3.729 | ≥99th | 0.744 | Very strong binding gain |
| CHIP:CEBPB:IMR-90 | 10.4 | 140 | +3.632 | ≥99th | 0.908 | Very strong binding gain |
| CHIP:CEBPB:K562 | 12.1 | 148 | +3.507 | ≥99th | 0.919 | Very strong binding gain |
| CHIP:eGFP-CEBPB:K562 genetically modified using stable transfection | 7.61 | 85.2 | +3.323 | ≥99th | 0.898 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 15 | 146 | +3.198 | ≥99th | 0.955 | Very strong binding gain |
| CHIP:eGFP-CEBPG:K562 genetically modified using stable transfection | 6.86 | 54.8 | +2.827 | ≥99th | 0.869 | Very strong binding gain |
| CHIP:CEBPB:A549 | 12.3 | 90.4 | +2.786 | ≥99th | 0.955 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 15.5 | 108 | +2.722 | ≥99th | 0.932 | Very strong binding gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me1:neutrophil male | 24.9 | 94.5 | +1.886 | ≥99th | 0.725 | Very strong mark gain |
| CHIP:H3K27ac:liver female adult (25 years) | 77.6 | 225 | +1.524 | ≥99th | 0.853 | Very strong mark gain |
| CHIP:H3K27ac:liver male adult (31 year) | 92.5 | 259 | +1.477 | ≥99th | 0.860 | Very strong mark gain |
| CHIP:H3K27ac:heart left ventricle male adult (32 years) | 67.4 | 182 | +1.417 | ≥99th | 0.873 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte male adult (21 year) | 120 | 274 | +1.189 | ≥99th | 0.891 | Very strong mark gain |
| CHIP:H3K27ac:right lobe of liver female adult (53 years) | 158 | 344 | +1.118 | ≥99th | 0.874 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte female | 155 | 331 | +1.091 | ≥99th | 0.897 | Very strong mark gain |
| CHIP:H3K27ac:skeletal muscle tissue female adult (72 years) | 147 | 263 | +0.834 | ≥99th | 0.881 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis female adult (53 years) | 175 | 297 | +0.763 | ≥99th | 0.894 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis male adult (54 years) | 179 | 298 | +0.731 | ≥99th | 0.890 | Very strong mark gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:liver, adult, pool1 — variant site | 0.473 | 2.88 | +1.396 | ≥99th | 0.846 | Very strong increase |
| CAGE:Hepatocyte, — variant site | 1.11 | 4.24 | +1.310 | ≥99th | 0.858 | Very strong increase |
| CAGE:liver, adult, pool1 — PSRC1 TSS | 8.43 | 12 | +0.460 | ≥99th | 0.921 | Strong increase |
| CAGE:Hepatocyte, — PSRC1 TSS | 6.83 | 9.73 | +0.454 | ≥99th | 0.921 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — variant site | 18.5 | 25.6 | +0.452 | ≥99th | 0.916 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — MYBPHL TSS | 40.1 | 51 | +0.340 | 0.99 | 0.926 | Strong increase |
| CAGE:epitheloid carcinoma cell line: HelaS3 ENCODE, biol_ — variant site | 11.7 | 14.6 | +0.298 | 0.98 | 0.908 | Moderate increase |
| CAGE:thalamus, adult, — MYBPHL TSS | 7.02 | 8.67 | +0.270 | 0.99 | 0.902 | Moderate increase |
| CAGE:locus coeruleus, adult, — MYBPHL TSS | 5.89 | 7.19 | +0.248 | 0.98 | 0.886 | Moderate increase |
| CAGE:spinal cord, adult, — variant site | 15.2 | 18.2 | +0.244 | 0.99 | 0.926 | Moderate increase |
| _…showing top 10 of 48 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
