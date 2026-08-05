## Analysis Request

> Score 5 SORT1-locus GWAS variants in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by regulatory effect. Gene is SORT1.

- **Tool**: `score_variant_batch`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-08-05 04:58 UTC

## Batch Variant Scoring Results

**5 variants scored**

| Variant | ID | DNASE:HepG2 Ref | DNASE:HepG2 Alt | DNASE:HepG2 log2FC | DNASE:HepG2 Effect %ile | CHIP:CEBPA:HepG2 Ref | CHIP:CEBPA:HepG2 Alt | CHIP:CEBPA:HepG2 log2FC | CHIP:CEBPA:HepG2 Effect %ile | CHIP:CEBPB:HepG2 Ref | CHIP:CEBPB:HepG2 Alt | CHIP:CEBPB:HepG2 log2FC | CHIP:CEBPB:HepG2 Effect %ile | CHIP:H3K27ac:HepG2 Ref | CHIP:H3K27ac:HepG2 Alt | CHIP:H3K27ac:HepG2 log2FC | CHIP:H3K27ac:HepG2 Effect %ile | CAGE:HepG2 (+) Ref | CAGE:HepG2 (+) Alt | CAGE:HepG2 (+) log2FC | CAGE:HepG2 (+) Effect %ile | CAGE:HepG2 (-) Ref | CAGE:HepG2 (-) Alt | CAGE:HepG2 (-) log2FC | CAGE:HepG2 (-) Effect %ile |
|---------|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chr1:109274968 G>T | rs12740374 | 660 | 1.67e+03 | +1.334 | ≥99th | 2.07e+03 | 1.6e+04 | +2.945 | ≥99th | 1.08e+03 | 1.07e+04 | +3.316 | ≥99th | 1.51e+04 | 3.58e+04 | +1.251 | ≥99th | 34.7 | 34.7 | -0.000 | near-zero | 3.82e+03 | 3.82e+03 | -0.002 | 0.30 |
| chr1:109274570 A>G | rs7528419 | 137 | 132 | -0.055 | 0.80 | 904 | 910 | +0.010 | 0.38 | 670 | 668 | -0.004 | 0.16 | 1.79e+04 | 1.76e+04 | -0.023 | 0.83 | 41.1 | 41.2 | +0.003 | 0.33 | 4.52e+03 | 4.52e+03 | +0.002 | 0.25 |
| chr1:109275684 G>T | rs1626484 | 74.1 | 71.1 | -0.059 | 0.81 | 410 | 408 | -0.007 | 0.30 | 398 | 398 | +0.004 | 0.14 | 1.17e+04 | 1.16e+04 | -0.024 | 0.84 | 35.8 | 36 | +0.007 | 0.50 | 3.83e+03 | 3.83e+03 | -0.001 | near-zero |
| chr1:109279175 G>A | rs4970836 | 7.99 | 7.84 | -0.024 | 0.59 | 130 | 129 | -0.014 | 0.49 | 138 | 137 | -0.003 | 0.10 | 2.98e+03 | 2.99e+03 | +0.009 | 0.64 | 39.7 | 39.6 | -0.003 | 0.33 | 4.14e+03 | 4.13e+03 | -0.003 | 0.38 |
| chr1:109275216 T>C | rs660240 | 408 | 408 | +0.002 | 0.08 | 631 | 641 | +0.023 | 0.64 | 443 | 449 | +0.019 | 0.58 | 1.59e+04 | 1.58e+04 | -0.004 | 0.40 | 38.3 | 38.4 | +0.004 | 0.39 | 4.01e+03 | 4.02e+03 | +0.003 | 0.34 |

Each track shows: **Ref** (reference allele prediction), **Alt** (alternate allele prediction), **log2FC** (log2 fold-change alt/ref), **Effect %ile** (ranked against ~10K random SNPs).

**Track identifiers** (for tracing back to oracle data):

- DNASE:HepG2: `DNASE/EFO:0001187 DNase-seq/.`
- CHIP:CEBPA:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.`
- CHIP:CEBPB:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.`
- CHIP:H3K27ac:HepG2: `CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.`
- CAGE:HepG2 (+): `CAGE/hCAGE EFO:0001187/+`
- CAGE:HepG2 (-): `CAGE/hCAGE EFO:0001187/-`
