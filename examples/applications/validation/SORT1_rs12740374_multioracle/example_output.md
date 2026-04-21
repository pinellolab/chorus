# Multi-oracle validation — rs12740374

- **Variant:** chr1:109,274,968 G>T
- **Gene:** SORT1
- **Oracles:** chrombpnet, legnet, alphagenome

## Cross-oracle consensus

| Layer | chrombpnet | legnet | alphagenome | Agreement |
|---|---|---|---|---|
| Chromatin accessibility (DNASE/ATAC) (log2fc) | -0.111 · ATAC:HepG2 · HepG2 | — | +0.453 · DNASE:HepG2 · HepG2 | disagree |
| Promoter activity (MPRA) (diff) | — | -0.028 · LentiMPRA:HepG2 · HepG2 | — | only ↓ (n=1) |
| Transcription factor binding (ChIP-TF) (log2fc) | — | — | +0.381 · CHIP:CEBPA:HepG2 · HepG2 | only ↑ (n=1) |
| Histone modifications (ChIP-Histone) (log2fc) | — | — | +0.180 · CHIP:H3K27ac:HepG2 · HepG2 | only ↑ (n=1) |
| TSS activity (CAGE/PRO-CAP) (log2fc) | — | — | +0.254 · CAGE:HepG2 · HepG2 | only ↑ (n=1) |