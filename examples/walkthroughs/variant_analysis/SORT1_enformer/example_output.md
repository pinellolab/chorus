## Analysis Request

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

- **Tool**: `discover_variant`
- **Oracle**: enformer
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all Enformer tracks (discovery mode)
- **Generated**: 2026-08-09 17:18 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: enformer
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Strongest effect per layer anywhere in the prediction window (not necessarily SORT1's own track). Transcription factor binding (ChIP-TF): very strong binding gain (+4.37, CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human…); Chromatin accessibility (DNASE/ATAC): very strong opening (+2.25, DNASE:fibroblast of lung); Histone modifications (ChIP-Histone): very strong mark gain (+1.89, CHIP:H3K4me1:neutrophil male); TSS activity (CAGE/PRO-CAP): very strong increase (+1.31, CAGE:Hepatocyte, — variant site).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:fibroblast of lung | 0.838 | 7.73 | +2.247 | 0.9998 | 0.857 | Very strong opening |
| DNASE:CD14-positive monocyte male adult (21 year) | 0.902 | 7.75 | +2.202 | 0.9997 | 0.860 | Very strong opening |
| DNASE:amniotic epithelial cell | 2.35 | 14.3 | +2.193 | 0.9997 | 0.898 | Very strong opening |
| DNASE:fibroblast of villous mesenchyme | 2.01 | 11 | +1.997 | 0.9997 | 0.883 | Very strong opening |
| DNASE:HL-60 | 1.69 | 9.52 | +1.969 | 0.9998 | 0.868 | Very strong opening |
| DNASE:foreskin fibroblast male newborn | 1.84 | 9.28 | +1.855 | 0.9989 | 0.870 | Very strong opening |
| DNASE:WI38 genetically modified using stable transfection originated from WI38 | 2.57 | 11.8 | +1.843 | 0.9991 | 0.904 | Very strong opening |
| ATAC:BM1137-GMP2-mid-ATAC-1 / Bone Marrow CD34+ / GMP-B | 11.7 | 35.2 | +1.512 | 0.9963 | 0.855 | Very strong opening |
| ATAC:BM1137-GMP1-low-ATAC-2 / Bone Marrow CD34+ / GMP-A | 6.29 | 18.9 | +1.446 | 0.9969 | 0.848 | Very strong opening |
| ATAC:BM1137-GMP3-high-ATAC-2 / Bone Marrow CD34+ / GMP-C | 13.1 | 37.3 | +1.442 | 0.9946 | 0.874 | Very strong opening |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 2.56 | 72.7 | +4.372 | ≥99th (1.08× null max) | 0.771 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_LowDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 3.99 | 97.9 | +4.310 | ≥99th (1.03× null max) | 0.793 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_noDMI / hMSC / Human Mesenchymal Stem Cells | 7.06 | 106 | +3.729 | ≥99th (1.18× null max) | 0.744 | Very strong binding gain |
| CHIP:CEBPB:IMR-90 | 10.4 | 140 | +3.632 | 0.9997 | 0.908 | Very strong binding gain |
| CHIP:CEBPB:K562 | 12.1 | 148 | +3.506 | 0.9998 | 0.919 | Very strong binding gain |
| CHIP:eGFP-CEBPB:K562 genetically modified using stable transfection | 7.61 | 85.2 | +3.323 | 0.9999 | 0.898 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 15 | 146 | +3.197 | 0.9996 | 0.955 | Very strong binding gain |
| CHIP:eGFP-CEBPG:K562 genetically modified using stable transfection | 6.86 | 54.8 | +2.827 | 0.9999 | 0.869 | Very strong binding gain |
| CHIP:CEBPB:A549 | 12.3 | 90.4 | +2.786 | 0.9997 | 0.955 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 15.5 | 108 | +2.722 | 0.9998 | 0.932 | Very strong binding gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me1:neutrophil male | 24.9 | 94.5 | +1.886 | 0.9996 | 0.725 | Very strong mark gain |
| CHIP:H3K27ac:liver female adult (25 years) | 77.6 | 225 | +1.524 | 0.9998 | 0.853 | Very strong mark gain |
| CHIP:H3K27ac:liver male adult (31 year) | 92.5 | 259 | +1.477 | 0.9994 | 0.860 | Very strong mark gain |
| CHIP:H3K27ac:heart left ventricle male adult (32 years) | 67.4 | 182 | +1.417 | 0.9997 | 0.873 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte male adult (21 year) | 120 | 275 | +1.190 | 0.9966 | 0.891 | Very strong mark gain |
| CHIP:H3K27ac:right lobe of liver female adult (53 years) | 158 | 344 | +1.118 | 0.9992 | 0.874 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte female | 155 | 331 | +1.091 | 0.9960 | 0.897 | Very strong mark gain |
| CHIP:H3K27ac:skeletal muscle tissue female adult (72 years) | 147 | 263 | +0.834 | 0.9990 | 0.881 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis female adult (53 years) | 175 | 297 | +0.763 | 0.9991 | 0.894 | Very strong mark gain |
| CHIP:H3K27ac:gastrocnemius medialis male adult (54 years) | 179 | 298 | +0.731 | 0.9987 | 0.890 | Very strong mark gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:Hepatocyte, — variant site | 1.11 | 4.24 | +1.310 | 0.9991 | 0.858 | Very strong increase |
| CAGE:Hepatocyte, — PSRC1 TSS | 6.83 | 9.73 | +0.454 | 0.9933 | 0.921 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — variant site | 18.5 | 25.6 | +0.452 | 0.98 | 0.916 | Strong increase |
| CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol_ — MYBPHL TSS | 40.1 | 51 | +0.340 | 0.98 | 0.926 | Strong increase |
| CAGE:thalamus, adult, — MYBPHL TSS | 7.02 | 8.67 | +0.269 | 0.98 | 0.902 | Moderate increase |
| CAGE:locus coeruleus, adult, — MYBPHL TSS | 5.89 | 7.18 | +0.247 | 0.97 | 0.886 | Moderate increase |
| CAGE:spinal cord, adult, — variant site | 15.2 | 18.2 | +0.244 | 0.98 | 0.926 | Moderate increase |
| CAGE:globus pallidus, adult, — variant site | 27.2 | 32.3 | +0.243 | 0.98 | 0.927 | Moderate increase |
| CAGE:substantia nigra, adult, — MYBPHL TSS | 7.13 | 8.6 | +0.240 | 0.98 | 0.903 | Moderate increase |
| CAGE:substantia nigra, adult, — variant site | 30.2 | 35.7 | +0.237 | 0.98 | 0.928 | Moderate increase |
| _…showing top 10 of 48 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against a per-track background of ~18,000 variants sampled from the regulatory regions this assay measures (cCREs, DHS summits, promoters, gene features) — not uniformly random positions. 0.95 = stronger than 95% of that background.
- **`N× null max`**: the effect exceeded *every* sampled background effect for that track, so the percentile is clamped and cannot rank it further. The multiplier gives the distance to that ceiling — `1.11×` is 11% beyond the most extreme background effect for that track. Common for variants that create or destroy a complete transcription-factor motif, which even a regulatory-region background rarely contains.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
