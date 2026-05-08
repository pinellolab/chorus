# Multi-oracle validation — rs12740374

- **Variant:** chr1:109,274,968 G>T
- **Gene:** SORT1
- **Oracles:** chrombpnet, legnet, alphagenome
- **Generated:** 2026-05-07 16:11 UTC

## Cross-oracle consensus

| Layer | chrombpnet | legnet | alphagenome | Agreement |
|---|---|---|---|---|
| Chromatin accessibility (DNASE/ATAC) (log2FC) | +1.173 · DNASE:HepG2 | — | +1.358 · DNASE:HepG2 | all ↑ |
| Promoter activity (MPRA) (Δ (alt−ref)) | — | +0.000 · LentiMPRA:HepG2 | — | only ↑ (n=1) |
| Transcription factor binding (ChIP-TF) (log2FC) | — | — | +2.809 · CHIP:CEBPA:HepG2 | only ↑ (n=1) |
| Histone modifications (ChIP-Histone) (log2FC) | — | — | +1.290 · CHIP:H3K27ac:HepG2 | only ↑ (n=1) |
| TSS activity (CAGE/PRO-CAP) (log2FC) | — | — | +1.545 · CAGE:HepG2 | only ↑ (n=1) |