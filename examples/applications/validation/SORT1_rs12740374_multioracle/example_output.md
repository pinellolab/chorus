# Multi-oracle validation — rs12740374

- **Variant:** chr1:109,274,968 G>T
- **Gene:** SORT1
- **Oracles:** alphagenome, chrombpnet, legnet

## Cross-oracle consensus

| Layer | alphagenome | chrombpnet | legnet | Agreement |
|---|---|---|---|---|
| Chromatin accessibility (DNASE/ATAC) (log2fc) | +0.453 · DNASE/EFO:0001187 DNase-seq/. · HepG2 | -0.111 · ATAC:HepG2 · HepG2 | — | disagree |
| Transcription factor binding (ChIP-TF) (log2fc) | +0.381 · CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/. · HepG2 | — | — | all ↑ |
| Histone modifications (ChIP-Histone) (log2fc) | +0.180 · CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/. · HepG2 | — | — | all ↑ |
| TSS activity (CAGE/PRO-CAP) (log2fc) | +0.254 · CAGE/hCAGE EFO:0001187/- · HepG2 | — | — | all ↑ |
| Promoter activity (MPRA) (diff) | — | — | -0.028 · LentiMPRA:HepG2 · HepG2 | all ↓ |