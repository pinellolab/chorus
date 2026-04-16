## Analysis Request

> Analyze rs12740374 (chr1:109274968 G>T) in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Gene is SORT1.

- **Tool**: `analyze_variant_multilayer`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2/K562 tracks
- **Generated**: 2026-04-16 17:48 UTC

## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1
**Other nearby genes**: PSRC1, CELSR2, MYBPHL, SARS1

**Summary**: Chromatin accessibility (DNASE/ATAC): strong opening (+0.45); Transcription factor binding (ChIP-TF): strong binding gain (+0.38); TSS activity (CAGE/PRO-CAP): moderate increase (+0.25); Histone modifications (ChIP-Histone): moderate mark gain (+0.18).

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| DNASE:HepG2 | 512 | 699 | +0.449 | ≥99th | 0.962 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:CEBPA:HepG2 | 2.1e+03 | 2.72e+03 | +0.376 | ≥99th | 0.988 | Strong binding gain |
| CHIP:CEBPB:HepG2 | 1.22e+03 | 1.47e+03 | +0.274 | ≥99th | 0.971 | Moderate binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CHIP:H3K27ac:HepG2 | 1.37e+04 | 1.55e+04 | +0.178 | ≥99th | 0.999 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Effect %ile | Activity %ile | Interpretation |
|---|---|---|---|---|---|---|
| CAGE:HepG2 — variant site | 22 | 26.4 | +0.253 | ≥99th | 1.000 | Moderate increase |
| CAGE:HepG2 — PSRC1 TSS | 45.4 | 46.4 | +0.029 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — MYBPHL TSS | 1.95 | 1.99 | +0.019 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — variant site | 76.2 | 77.2 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — PSRC1 TSS | 2.4e+03 | 2.42e+03 | +0.017 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — ELAPOR1 TSS | 74.6 | 73.8 | -0.016 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — CELSR2 TSS | 2.45 | 2.49 | +0.014 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — EPS8L3 TSS | 39.1 | 38.8 | -0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — SORT1 TSS | 6.98 | 7.04 | +0.012 | ≥99th | 1.000 | Minimal effect |
| CAGE:HepG2 — GPR61 TSS | 5.99 | 6.03 | +0.009 | ≥99th | 1.000 | Minimal effect |
| _…showing top 10 of 58 — see `example_output.json` for the full set_ | | | | | | |

## Interpretation

This variant shows **strong, convergent effects across multiple regulatory layers in HepG2 liver cells**, consistent with the published mechanism. The T allele at rs12740374 creates a C/EBP transcription factor binding site in a liver enhancer (Musunuru et al., 2010, *Nature*), leading to chromatin opening, CEBPA/CEBPB binding gain, H3K27ac activation, and increased transcription. This multi-layer agreement provides strong evidence that rs12740374 is the functional variant at the 1p13 GWAS locus. The downstream effect is increased SORT1 expression in liver, enhancing LDL cholesterol clearance.

---
**Score guide:**
- **Effect %ile**: Variant effect ranked against ~10K random SNPs. 0.95 = stronger than 95% of random variants.
- **Activity %ile**: Reference signal ranked genome-wide against ENCODE SCREEN cCREs + random regions. 0.95 = more active than 95% of genomic positions.
