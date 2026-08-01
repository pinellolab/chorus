## Analysis Request

> Score 5 SORT1-locus GWAS variants in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by regulatory effect. Gene is SORT1.

- **Tool**: `score_variant_batch`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-08-01 03:22 UTC

## Batch Variant Scoring Results

**5 variants scored**

| Variant | ID | DNASE:HepG2 Ref | DNASE:HepG2 Alt | DNASE:HepG2 log2FC | DNASE:HepG2 Effect %ile | CHIP:CEBPA:HepG2 Ref | CHIP:CEBPA:HepG2 Alt | CHIP:CEBPA:HepG2 log2FC | CHIP:CEBPA:HepG2 Effect %ile | CHIP:CEBPB:HepG2 Ref | CHIP:CEBPB:HepG2 Alt | CHIP:CEBPB:HepG2 log2FC | CHIP:CEBPB:HepG2 Effect %ile | CHIP:H3K27ac:HepG2 Ref | CHIP:H3K27ac:HepG2 Alt | CHIP:H3K27ac:HepG2 log2FC | CHIP:H3K27ac:HepG2 Effect %ile | CAGE:HepG2 (+) Ref | CAGE:HepG2 (+) Alt | CAGE:HepG2 (+) log2FC | CAGE:HepG2 (+) Effect %ile | CAGE:HepG2 (-) Ref | CAGE:HepG2 (-) Alt | CAGE:HepG2 (-) log2FC | CAGE:HepG2 (-) Effect %ile |
|---------|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chr1:109274968 G>T | rs12740374 | 662 | 1.67e+03 | +1.330 | ≥99th | 2.57e+03 | 1.75e+04 | +2.764 | ≥99th | 1.38e+03 | 1.14e+04 | +3.046 | ≥99th | 1.57e+04 | 3.76e+04 | +1.258 | ≥99th | 34.7 | 34.6 | -0.005 | 0.74 | 3.82e+03 | 3.81e+03 | -0.001 | 0.46 |
| chr1:109274570 A>G | rs7528419 | 137 | 132 | -0.053 | 0.90 | 1.01e+03 | 1.02e+03 | +0.009 | 0.39 | 776 | 774 | -0.003 | 0.10 | 1.8e+04 | 1.78e+04 | -0.018 | 0.71 | 41 | 41.3 | +0.010 | 0.84 | 4.52e+03 | 4.51e+03 | -0.004 | 0.70 |
| chr1:109275684 G>T | rs1626484 | 73.8 | 71 | -0.056 | 0.91 | 560 | 556 | -0.010 | 0.46 | 522 | 521 | -0.004 | 0.15 | 1.35e+04 | 1.33e+04 | -0.020 | 0.74 | 35.8 | 35.7 | -0.003 | 0.67 | 3.82e+03 | 3.82e+03 | -0.000 | near-zero |
| chr1:109279175 G>A | rs4970836 | 7.99 | 7.84 | -0.024 | 0.73 | 194 | 193 | -0.007 | 0.35 | 194 | 193 | -0.005 | 0.16 | 3.36e+03 | 3.38e+03 | +0.010 | 0.53 | 39.4 | 39.6 | +0.010 | 0.84 | 4.12e+03 | 4.13e+03 | +0.003 | 0.66 |
| chr1:109275216 T>C | rs660240 | 407 | 408 | +0.004 | 0.18 | 1.31e+03 | 1.31e+03 | +0.007 | 0.32 | 778 | 785 | +0.013 | 0.45 | 1.68e+04 | 1.68e+04 | -0.001 | near-zero | 38.4 | 38.4 | -0.001 | 0.43 | 4.02e+03 | 4.01e+03 | -0.003 | 0.64 |

Each track shows: **Ref** (reference allele prediction), **Alt** (alternate allele prediction), **log2FC** (log2 fold-change alt/ref), **Effect %ile** (ranked against ~10K random SNPs).

**Track identifiers** (for tracing back to oracle data):

- DNASE:HepG2: `DNASE/EFO:0001187 DNase-seq/.`
- CHIP:CEBPA:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.`
- CHIP:CEBPB:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.`
- CHIP:H3K27ac:HepG2: `CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.`
- CAGE:HepG2 (+): `CAGE/hCAGE EFO:0001187/+`
- CAGE:HepG2 (-): `CAGE/hCAGE EFO:0001187/-`
