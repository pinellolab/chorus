## Analysis Request

> Score 5 SORT1-locus GWAS variants in HepG2 liver cells using DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by regulatory effect. Gene is SORT1.

- **Tool**: `score_variant_batch`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Tracks requested**: 6 HepG2 tracks
- **Generated**: 2026-04-14 01:52 UTC

## Batch Variant Scoring Results

**5 variants scored**

| Variant | ID | DNASE:HepG2 | CHIP:CEBPA:HepG2 | CHIP:CEBPB:HepG2 | CHIP:H3K27ac:HepG2 | CAGE:HepG2 (+) | CAGE:HepG2 (-) |
|---------|-----|---|---|---|---|---|---|
| chr1:109274968 G>T | rs12740374 | +0.448 (100%) | +0.372 (100%) | +0.271 (100%) | +0.182 (100%) | +0.005 (100%) | +0.002 (100%) |
| chr1:109279175 G>A | rs4970836 | -0.041 (100%) | -0.032 (100%) | -0.029 (100%) | -0.014 (100%) | +0.008 (100%) | +0.001 (100%) |
| chr1:109275216 T>C | rs660240 | +0.071 (100%) | +0.027 (100%) | +0.013 (100%) | +0.031 (100%) | -0.007 (100%) | -0.003 (100%) |
| chr1:109275684 G>T | rs1626484 | +0.006 (100%) | +0.003 (71%) | +0.001 (21%) | -0.001 (44%) | +0.001 (100%) | +0.004 (100%) |
| chr1:109274570 A>G | rs7528419 | +0.015 (100%) | +0.007 (100%) | +0.005 (86%) | +0.019 (100%) | -0.006 (100%) | -0.001 (100%) |

Each cell shows: **raw effect** (effect percentile).
Effect percentile ranks this variant's effect against ~10K random SNPs.

**Track identifiers** (for tracing back to oracle data):

- DNASE:HepG2: `DNASE/EFO:0001187 DNase-seq/.`
- CHIP:CEBPA:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.`
- CHIP:CEBPB:HepG2: `CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.`
- CHIP:H3K27ac:HepG2: `CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.`
- CAGE:HepG2 (+): `CAGE/hCAGE EFO:0001187/+`
- CAGE:HepG2 (-): `CAGE/hCAGE EFO:0001187/-`
