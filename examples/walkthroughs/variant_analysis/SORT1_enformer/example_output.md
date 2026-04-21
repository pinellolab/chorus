## Analysis Request

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

- **Tool**: `discover_variant`
- **Oracle**: enformer
- **Normalizer**: per-track background CDFs
- **Tracks requested**: all Enformer tracks (discovery mode)
- **Generated**: 2026-04-21 13:09 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: enformer
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Transcription factor binding (ChIP-TF): very strong binding gain (+4.36, CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human…); Chromatin accessibility (DNASE/ATAC): very strong opening (+2.70, DNASE:CD14-positive monocyte male adult (21 year)); Histone modifications (ChIP-Histone): very strong mark gain (+2.57, CHIP:H3K4me3:CD14-positive monocyte female); TSS activity (CAGE/PRO-CAP): very strong increase (+1.54, CAGE:liver, adult, pool1).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:CD14-positive monocyte male adult (21 year) | 1.03 | 12.2 | +2.703 | ≥99th | 0.866 | Very strong opening |
| DNASE:skeletal muscle cell | 0.583 | 7.82 | +2.478 | ≥99th | 0.829 | Very strong opening |
| DNASE:fibroblast of pulmonary artery | 0.691 | 8.39 | +2.473 | ≥99th | 0.841 | Very strong opening |
| DNASE:fibroblast of lung | 1 | 9.65 | +2.413 | ≥99th | 0.866 | Very strong opening |
| DNASE:fibroblast of mammary gland female | 0.324 | 6.04 | +2.410 | ≥99th | 0.803 | Very strong opening |
| DNASE:CD14-positive monocyte female | 1.34 | 10.9 | +2.351 | ≥99th | 0.881 | Very strong opening |
| DNASE:HL-60 | 1.88 | 13.5 | +2.335 | ≥99th | 0.872 | Very strong opening |
| DNASE:CD14-positive monocyte male adult (37 years) | 0.748 | 7.81 | +2.333 | ≥99th | 0.864 | Very strong opening |
| DNASE:amniotic epithelial cell | 2.57 | 16.8 | +2.319 | ≥99th | 0.902 | Very strong opening |
| DNASE:NB4 | 1.65 | 11.3 | +2.217 | ≥99th | 0.873 | Very strong opening |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 2.81 | 77.3 | +4.363 | ≥99th | 0.794 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_LowDensity_DMI / hMSC / Human Mesenchymal Stem Cells | 4.64 | 105 | +4.231 | ≥99th | 0.826 | Very strong binding gain |
| CHIP:CEBPB:IMR-90 | 11.5 | 153 | +3.622 | ≥99th | 0.919 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, CEBPb_HighDensity_noDMI / hMSC / Human Mesenchymal Stem Cells | 8.87 | 114 | +3.547 | ≥99th | 0.849 | Very strong binding gain |
| CHIP:CEBPB:K562 | 16.4 | 185 | +3.417 | ≥99th | 0.952 | Very strong binding gain |
| CHIP:eGFP-CEBPB:K562 genetically modified using stable transfection | 10.6 | 109 | +3.245 | ≥99th | 0.948 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, InVitroCistromics_CEBPb_10uL / hMSC / Human Mesenchymal Stem Cells | 0.463 | 10.3 | +2.945 | ≥99th | 0.200 | Very strong binding gain |
| CHIP:CEBPB:HepG2 | 21.3 | 166 | +2.907 | ≥99th | 0.974 | Very strong binding gain |
| CHIP:CEBPb:ChIP-seq, InVitroCistromics_CEBPb_1.0uL / hMSC / Human Mesenchymal Stem Cells | 0.698 | 11.5 | +2.878 | ≥99th | 0.258 | Very strong binding gain |
| CHIP:eGFP-CEBPG:K562 genetically modified using stable transfection | 9.36 | 71.3 | +2.803 | ≥99th | 0.944 | Very strong binding gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K4me3:CD14-positive monocyte female | 1.86 | 16 | +2.573 | ≥99th | 0.084 | Very strong mark gain |
| CHIP:H3K4me1:neutrophil male | 26.5 | 160 | +2.549 | ≥99th | 0.732 | Very strong mark gain |
| CHIP:H3K4me3:CD14-positive monocyte female | 25.1 | 103 | +1.996 | ≥99th | 0.785 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte male adult (21 year) | 95.9 | 321 | +1.731 | ≥99th | 0.853 | Very strong mark gain |
| CHIP:H3K27ac:CD14-positive monocyte female | 32.4 | 104 | +1.656 | ≥99th | 0.748 | Very strong mark gain |
| CHIP:H3K27ac:neutrophil | 22.5 | 72.2 | +1.638 | ≥99th | 0.709 | Very strong mark gain |
| CHIP:H3K4me1:neutrophil | 29.4 | 90.8 | +1.592 | ≥99th | 0.711 | Very strong mark gain |
| CHIP:H3K4me1:CD14-positive monocyte female | 128 | 385 | +1.578 | ≥99th | 0.863 | Very strong mark gain |
| CHIP:H3K4me1:mononuclear cell male | 52.7 | 154 | +1.532 | ≥99th | 0.776 | Very strong mark gain |
| CHIP:H3K27ac:fibroblast of lung female child (11 year) and male adult (45 years) | 34.2 | 98.6 | +1.502 | ≥99th | 0.805 | Very strong mark gain |
| _…showing top 10 of 12 — see `example_output.json` for the full set_ | | | | | | |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:liver, adult, pool1 — variant site | 0.902 | 4.52 | +1.538 | ≥99th | 0.871 | Very strong increase |
| CAGE:CD14+ monocytes - mock treated, — variant site | 0.618 | 3.36 | +1.431 | ≥99th | 0.796 | Very strong increase |
| CAGE:Hepatocyte, — variant site | 1.43 | 5.05 | +1.315 | ≥99th | 0.873 | Very strong increase |
| CAGE:CD14+ monocytes - treated with Cryptococcus, — variant site | 0.555 | 2.67 | +1.238 | ≥99th | 0.767 | Very strong increase |
| CAGE:CD14+ monocytes - treated with Trehalose dimycolate (TDM), — variant site | 0.195 | 1.74 | +1.199 | ≥99th | 0.492 | Very strong increase |
| CAGE:CD14+ monocyte derived endothelial progenitor cells, — variant site | 1.23 | 3.84 | +1.119 | ≥99th | 0.877 | Very strong increase |
| CAGE:CD14+ monocytes - treated with IFN + N-hexane, — PSRC1 TSS | 12.2 | 26.7 | +1.071 | ≥99th | 0.928 | Very strong increase |
| CAGE:CD14+ monocytes - treated with BCG, — variant site | 0.22 | 1.5 | +1.034 | ≥99th | 0.524 | Very strong increase |
| CAGE:CD14+CD16- Monocytes, — variant site | 0.509 | 2.01 | +0.995 | ≥99th | 0.383 | Very strong increase |
| CAGE:CD14+ monocytes - treated with B-glucan, — variant site | 0.465 | 1.85 | +0.960 | ≥99th | 0.760 | Very strong increase |
| _…showing top 10 of 48 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
