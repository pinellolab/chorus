## Analysis Request

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

- **Tool**: `discover_variant`
- **Oracle**: enformer
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all Enformer tracks (discovery mode)
- **Generated**: 2026-08-01 03:21 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: enformer
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+4.23, CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human…); Chromatin accessibility (DNASE/ATAC): very strong opening (+2.26, DNASE:fibroblast of lung); Histone modifications (ChIP-Histone): very strong mark gain (+1.83, CHIP:H3K4me1:neutrophil male); TSS activity (CAGE/PRO-CAP): very strong increase (+1.71, CAGE:liver, adult, pool1).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of lung | 0.917 | 8.16 | +2.257 | ≥99th | 0.861 | Very strong opening |
| DNASE:amniotic epithelial cell | 2.41 | 14.9 | +2.218 | ≥99th | 0.900 | Very strong opening |
| DNASE:fibroblast of villous mesenchyme | 2.16 | 12 | +2.041 | ≥99th | 0.886 | Very strong opening |
| DNASE:HL-60 | 1.87 | 10.5 | +1.998 | 0.96 | 0.872 | Very strong opening |
| DNASE:WI38 genetically modified using stable transfection originated from WI38 | 2.67 | 12.6 | +1.889 | ≥99th | 0.905 | Very strong opening |
| DNASE:NB4 | 1.96 | 9.75 | +1.863 | 0.96 | 0.879 | Very strong opening |
| ATAC:BM1137-GMP2-mid-ATAC-1 / Bone Marrow CD34+ / GMP-B | 12.8 | 37.3 | +1.476 | 0.96 | 0.861 | Very strong opening |
| ATAC:BM1137-GMP3-high-ATAC-2 / Bone Marrow CD34+ / GMP-C | 13.9 | 38.9 | +1.427 | 0.96 | 0.877 | Very strong opening |
| ATAC:BM1137-GMP1-low-ATAC-2 / Bone Marrow CD34+ / GMP-A | 6.86 | 19.8 | +1.407 | 0.96 | 0.854 | Very strong opening |
| ATAC:BM0106-UNK-ATAC-2 / Bone Marrow CD34+ / UNK | 7.92 | 20.2 | +1.250 | 0.96 | 0.859 | Very strong opening |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 3.07 | 75.3 | +4.230 | ≥99th | 0.814 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_LowDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 4.76 | 102 | +4.161 | ≥99th | 0.831 | Very strong binding gain |
| CHIP:CEBPB:IMR-90 | 12.1 | 147 | +3.500 | 0.96 | 0.925 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_noDMI / hMSC / Human Mesenchymal Stem Cells | 9.38 | 111 | +3.435 | ≥99th | 0.863 | Very strong binding gain |
| CHIP:CEBPB:K562 | 15.2 | 162 | +3.337 | 0.96 | 0.945 | Very strong binding gain |
| CHIP:eGFP-CEBPB:K562 genetically modified using stable transfection | 10 | 96.9 | +3.153 | ≥99th | 0.942 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 20.4 | 160 | +2.912 | 0.96 | 0.972 | Very strong binding gain |
| CHIP:eGFP-CEBPG:K562 genetically modified using stable transfection | 8.9 | 62.1 | +2.673 | ≥99th | 0.936 | Very strong binding gain |
| CHIP:CEBPB:A549 | 15.2 | 98.9 | +2.620 | 0.96 | 0.968 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 21.8 | 122 | +2.436 | ≥99th | 0.974 | Very strong binding gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me1:neutrophil male | 26.8 | 97.6 | +1.827 | 0.96 | 0.733 | Very strong mark gain |
| CHIP:H3K27ac:liver female adult (25 years) | 79.3 | 231 | +1.530 | ≥99th | 0.854 | Very strong mark gain |
| CHIP:H3K27ac:liver male adult (31 year) | 94.3 | 267 | +1.490 | ≥99th | 0.861 | Very strong mark gain |
| CHIP:H3K27ac:heart left ventricle male adult (32 years) | 68.7 | 187 | +1.433 | ≥99th | 0.874 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte male adult (21 year) | 125 | 287 | +1.193 | 0.96 | 0.898 | Very strong mark gain |
| CHIP:H3K27ac:right lobe of liver female adult (53 years) | 161 | 354 | +1.132 | ≥99th | 0.875 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte female | 162 | 347 | +1.095 | 0.96 | 0.904 | Very strong mark gain |
| CHIP:H3K27ac:skeletal muscle tissue female adult (72 years) | 150 | 268 | +0.833 | 0.96 | 0.883 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis female adult (53 years) | 179 | 306 | +0.769 | 0.96 | 0.896 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis male adult (54 years) | 184 | 307 | +0.737 | 0.96 | 0.891 | Very strong mark gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:liver, adult, pool1 — variant site | 0.822 | 4.94 | +1.706 | ≥99th | 0.868 | Very strong increase |
| CAGE:Hepatocyte, — variant site | 1.39 | 5.49 | +1.439 | 0.96 | 0.871 | Very strong increase |
| CAGE:liver, adult, pool1 — PSRC1 TSS | 8.43 | 12 | +0.460 | 0.96 | 0.921 | Strong increase |
| CAGE:Hepatocyte, — PSRC1 TSS | 6.83 | 9.73 | +0.454 | 0.96 | 0.921 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — variant site | 26.9 | 37.1 | +0.446 | 0.96 | 0.921 | Strong increase |
| CAGE:epitheloid carcinoma cell line: HelaS3 ENCODE, biol_ — variant site | 16.9 | 22.1 | +0.370 | 0.96 | 0.915 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — MYBPHL TSS | 40.1 | 51 | +0.340 | 0.96 | 0.926 | Strong increase |
| CAGE:spinal cord, adult, — variant site | 17.8 | 21.7 | +0.274 | 0.96 | 0.928 | Moderate increase |
| CAGE:thalamus, adult, — MYBPHL TSS | 7.02 | 8.67 | +0.269 | 0.96 | 0.902 | Moderate increase |
| CAGE:globus pallidus, adult, — variant site | 31.6 | 38.1 | +0.262 | 0.96 | 0.930 | Moderate increase |
| _…showing top 10 of 48 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
