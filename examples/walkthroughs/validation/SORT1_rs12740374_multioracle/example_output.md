# Multi-oracle validation — rs12740374

- **Variant:** chr1:109,274,968 G>T
- **Gene:** SORT1
- **Oracles:** chrombpnet, legnet, alphagenome
- **Generated:** 2026-06-17 19:29 UTC

## Cross-oracle consensus

| Layer | chrombpnet | legnet | alphagenome | Agreement |
|---|---|---|---|---|
| Chromatin accessibility (DNASE/ATAC) (log2FC) | +1.374 · DNASE:HepG2 | — | +1.333 · DNASE:HepG2 | all ↑ |
| Promoter activity (MPRA) (Δ (alt−ref)) | — | +0.347 · LentiMPRA:HepG2 | — | only ↑ (n=1) |
| Transcription factor binding (ChIP-TF) (log2FC) | — | — | +2.767 · CHIP:CEBPA:HepG2 | only ↑ (n=1) |
| Histone modifications (ChIP-Histone) (log2FC) | — | — | +1.264 · CHIP:H3K27ac:HepG2 | only ↑ (n=1) |
| TSS activity (CAGE/PRO-CAP) (log2FC) | — | — | +1.525 · CAGE:HepG2 | only ↑ (n=1) |