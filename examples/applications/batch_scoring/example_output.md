## Analysis Request

> Score 5 SORT1-locus GWAS variants in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by regulatory effect. Gene is SORT1.

- **Tool**: `score_variant_batch`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-18 22:19 UTC

## Batch Variant Scoring Results

**5 variants scored**

| Variant | ID | DNASE:HepG2 Ref | DNASE:HepG2 Alt | DNASE:HepG2 log2FC | DNASE:HepG2 Effect %ile | CHIP:CEBPA:HepG2 Ref | CHIP:CEBPA:HepG2 Alt | CHIP:CEBPA:HepG2 log2FC | CHIP:CEBPA:HepG2 Effect %ile | CHIP:CEBPB:HepG2 Ref | CHIP:CEBPB:HepG2 Alt | CHIP:CEBPB:HepG2 log2FC | CHIP:CEBPB:HepG2 Effect %ile | CHIP:H3K27ac:HepG2 Ref | CHIP:H3K27ac:HepG2 Alt | CHIP:H3K27ac:HepG2 log2FC | CHIP:H3K27ac:HepG2 Effect %ile | CAGE:HepG2 (+) Ref | CAGE:HepG2 (+) Alt | CAGE:HepG2 (+) log2FC | CAGE:HepG2 (+) Effect %ile | CAGE:HepG2 (-) Ref | CAGE:HepG2 (-) Alt | CAGE:HepG2 (-) log2FC | CAGE:HepG2 (-) Effect %ile |
|---------|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chr1:109274968 G>T | rs12740374 | 512 | 699 | +0.448 | ≥99th | 2.1e+03 | 2.72e+03 | +0.371 | ≥99th | 1.22e+03 | 1.46e+03 | +0.268 | ≥99th | 1.37e+04 | 1.55e+04 | +0.180 | ≥99th | 39 | 38.8 | -0.004 | ≥99th | 3.85e+03 | 3.84e+03 | -0.004 | ≥99th |
| chr1:109279175 G>A | rs4970836 | 8.22 | 7.96 | -0.041 | ≥99th | 201 | 197 | -0.031 | ≥99th | 202 | 198 | -0.027 | ≥99th | 3.32e+03 | 3.28e+03 | -0.015 | ≥99th | 43.2 | 43.3 | +0.004 | ≥99th | 4.19e+03 | 4.18e+03 | -0.001 | ≥99th |
| chr1:109275216 T>C | rs660240 | 398 | 418 | +0.071 | ≥99th | 1.28e+03 | 1.3e+03 | +0.021 | ≥99th | 770 | 779 | +0.017 | ≥99th | 1.63e+04 | 1.67e+04 | +0.032 | ≥99th | 45 | 45.1 | +0.003 | ≥99th | 4.31e+03 | 4.31e+03 | -0.000 | near-zero |
| chr1:109275684 G>T | rs1626484 | 69.4 | 69.6 | +0.003 | 0.67 | 539 | 541 | +0.005 | ≥99th | 512 | 512 | +0.001 | 0.21 | 1.34e+04 | 1.34e+04 | -0.000 | near-zero | 37.3 | 37.3 | +0.000 | near-zero | 3.94e+03 | 3.95e+03 | +0.001 | ≥99th |
| chr1:109274570 A>G | rs7528419 | 118 | 119 | +0.008 | ≥99th | 958 | 960 | +0.002 | 0.62 | 776 | 778 | +0.002 | 0.31 | 1.46e+04 | 1.48e+04 | +0.018 | ≥99th | 40.7 | 40.7 | -0.003 | ≥99th | 4.3e+03 | 4.29e+03 | -0.002 | ≥99th |

Each track shows: **Ref** (reference allele prediction), **Alt** (alternate allele prediction), **log2FC** (log2 fold-change alt/ref), **Effect %ile** (ranked against ~10K random SNPs).

**Track identifiers** (for tracing back to oracle data):

- DNASE:HepG2: `DNASE/EFO:0001187 DNase-seq/.`
- CHIP:CEBPA:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.`
- CHIP:CEBPB:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.`
- CHIP:H3K27ac:HepG2: `CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.`
- CAGE:HepG2 (+): `CAGE/hCAGE EFO:0001187/+`
- CAGE:HepG2 (-): `CAGE/hCAGE EFO:0001187/-`
