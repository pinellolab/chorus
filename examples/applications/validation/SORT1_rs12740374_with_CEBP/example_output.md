## Analysis Request

> Validate AlphaGenome paper finding: rs12740374 (chr1:109274968 G>T) should show CEBPA/CEBPB binding gain in HepG2. Using forced HepG2 tracks.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks (DNASE, CEBPA, CEBPB, H3K27ac, CAGE+/-)
- **Generated**: 2026-04-13 13:06 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): strong opening (+0.45); Transcription factor binding (ChIP-TF): strong binding gain (+0.37); TSS activity (CAGE/PRO-CAP): moderate increase (+0.25); Histone modifications (ChIP-Histone): moderate mark gain (+0.18).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 512 | 698 | +0.447 | 1.000 | 0.962 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 2.1e+03 | 2.71e+03 | +0.369 | 1.000 | 0.989 | Strong binding gain |
| CHIP:CEBPB:HepG2 | 1.22e+03 | 1.47e+03 | +0.271 | 1.000 | 0.971 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.37e+04 | 1.55e+04 | +0.178 | 1.000 | 0.999 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 21.9 | 26.3 | +0.252 | 1.000 | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 45.5 | 46.5 | +0.032 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — variant site | 76.2 | 77.2 | +0.019 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — PSRC1 TSS | 2.4e+03 | 2.43e+03 | +0.018 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — MYBPHL TSS | 1.97 | 2 | +0.018 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — GSTM1 TSS | 240 | 238 | -0.015 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — SORT1 TSS | 6.98 | 7.05 | +0.012 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — CELSR2 TSS | 2.45 | 2.48 | +0.011 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — GPR61 TSS | 5.98 | 6.03 | +0.011 | 1.000 | 1.000 | Minimal effect |
| CAGE:HepG2 — CFAP276 TSS | 20.9 | 20.8 | -0.008 | 1.000 | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
