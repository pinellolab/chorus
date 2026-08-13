# Multi-oracle validation — rs12740374

- **Variant:** chr1:109,274,968 G>T
- **Gene:** SORT1
- **Oracles:** chrombpnet, cherimoya, legnet, alphagenome
- **Generated:** 2026-08-13 01:47 UTC

## Cross-oracle consensus

| Layer | chrombpnet | cherimoya | legnet | alphagenome | Agreement (direction) |
|---|---|---|---|---|---|
| Chromatin accessibility (DNASE/ATAC) (log2FC) | +1.376 · DNASE:HepG2 | +1.793 · DNASE:HepG2 | — | +1.334 · DNASE:HepG2 | all ↑ · +1.33…+1.79 |
| Promoter activity (MPRA) (Δ (alt−ref)) | — | — | +0.347 · LentiMPRA:HepG2 | — | only ↑ (n=1) |
| Transcription factor binding (ChIP-TF) (log2FC) | — | — | — | +2.945 · CHIP:CEBPA:HepG2 | only ↑ (n=1) |
| Histone modifications (ChIP-Histone) (log2FC) | — | — | — | +1.251 · CHIP:H3K27ac:HepG2 | only ↑ (n=1) |
| TSS activity (CAGE/PRO-CAP) (log2FC) | — | — | — | +1.502 · CAGE:HepG2 | only ↑ (n=1) |