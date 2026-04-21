## Analysis Request

> Score 5 SORT1-locus GWAS variants in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by regulatory effect. Gene is SORT1.

- **Tool**: `score_variant_batch`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-21 13:30 UTC

## Batch Variant Scoring Results

**5 variants scored**

| Variant | ID | DNASE:HepG2 Ref | DNASE:HepG2 Alt | DNASE:HepG2 log2FC | DNASE:HepG2 Effect %ile | CHIP:CEBPA:HepG2 Ref | CHIP:CEBPA:HepG2 Alt | CHIP:CEBPA:HepG2 log2FC | CHIP:CEBPA:HepG2 Effect %ile | CHIP:CEBPB:HepG2 Ref | CHIP:CEBPB:HepG2 Alt | CHIP:CEBPB:HepG2 log2FC | CHIP:CEBPB:HepG2 Effect %ile | CHIP:H3K27ac:HepG2 Ref | CHIP:H3K27ac:HepG2 Alt | CHIP:H3K27ac:HepG2 log2FC | CHIP:H3K27ac:HepG2 Effect %ile | CAGE:HepG2 (+) Ref | CAGE:HepG2 (+) Alt | CAGE:HepG2 (+) log2FC | CAGE:HepG2 (+) Effect %ile | CAGE:HepG2 (-) Ref | CAGE:HepG2 (-) Alt | CAGE:HepG2 (-) log2FC | CAGE:HepG2 (-) Effect %ile |
|---------|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chr1:109274968 G>T | rs12740374 | 595 | 1.57e+03 | +1.400 | ≥99th | 2.54e+03 | 1.72e+04 | +2.760 | ≥99th | 1.4e+03 | 1.13e+04 | +3.020 | ≥99th | 1.43e+04 | 3.6e+04 | +1.336 | ≥99th | 39 | 38.8 | -0.008 | ≥99th | 3.86e+03 | 3.85e+03 | -0.004 | ≥99th |
| chr1:109274570 A>G | rs7528419 | 119 | 113 | -0.067 | ≥99th | 967 | 968 | +0.001 | near-zero | 771 | 766 | -0.010 | ≥99th | 1.53e+04 | 1.5e+04 | -0.025 | ≥99th | 42 | 42 | -0.002 | ≥99th | 4.38e+03 | 4.37e+03 | -0.003 | ≥99th |
| chr1:109279175 G>A | rs4970836 | 7.99 | 7.84 | -0.025 | ≥99th | 197 | 196 | -0.009 | ≥99th | 197 | 196 | -0.007 | ≥99th | 3.29e+03 | 3.31e+03 | +0.007 | ≥99th | 35.4 | 35.4 | +0.003 | ≥99th | 3.94e+03 | 3.93e+03 | -0.001 | near-zero |
| chr1:109275684 G>T | rs1626484 | 63.4 | 60.9 | -0.057 | ≥99th | 527 | 522 | -0.015 | ≥99th | 512 | 510 | -0.004 | 0.78 | 1.2e+04 | 1.19e+04 | -0.016 | ≥99th | 41.8 | 42 | +0.006 | ≥99th | 4.14e+03 | 4.14e+03 | -0.001 | near-zero |
| chr1:109275216 T>C | rs660240 | 368 | 371 | +0.014 | ≥99th | 1.27e+03 | 1.28e+03 | +0.009 | ≥99th | 768 | 773 | +0.009 | ≥99th | 1.56e+04 | 1.56e+04 | -0.002 | 0.69 | 42.1 | 41.7 | -0.014 | ≥99th | 4.1e+03 | 4.09e+03 | -0.006 | ≥99th |

Each track shows: **Ref** (reference allele prediction), **Alt** (alternate allele prediction), **log2FC** (log2 fold-change alt/ref), **Effect %ile** (ranked against ~10K random SNPs).

**Track identifiers** (for tracing back to oracle data):

- DNASE:HepG2: `DNASE/EFO:0001187 DNase-seq/.`
- CHIP:CEBPA:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.`
- CHIP:CEBPB:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.`
- CHIP:H3K27ac:HepG2: `CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.`
- CAGE:HepG2 (+): `CAGE/hCAGE EFO:0001187/+`
- CAGE:HepG2 (-): `CAGE/hCAGE EFO:0001187/-`
