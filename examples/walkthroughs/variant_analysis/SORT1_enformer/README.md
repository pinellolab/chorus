# SORT1 rs12740374 — Enformer Discovery Mode

## Variant: rs12740374 (chr1:109274968 G>T)

Same variant as the AlphaGenome SORT1 example, but scored with
**Enformer** across all 5,313 ENCODE tracks in discovery mode. Enformer
has a 114 kb output window and supports chromatin, TF binding, histone
marks, and CAGE — but **not RNA-seq**. Discovery mode exposes the
cross-tissue signature of this variant without pre-specifying cell type.

## Example prompt

> Analyze chr1:109274968 G>T using Enformer discovery mode. Gene: SORT1.

## What Claude does

1. `load_oracle('enformer')`
2. `discover_variant('enformer', 'chr1:109274968', 'G', ['T'], gene_name='SORT1')` — scores all tracks, ranks by effect magnitude
3. Report shows the top tracks across 4 regulatory layers (no RNA section — Enformer doesn't have RNA tracks)

## Results

**Summary**: Chromatin accessibility (DNASE/ATAC): very strong opening
(+1.24); Transcription factor binding (ChIP-TF): very strong binding
gain (+1.13); Histone modifications (ChIP-Histone): very strong mark
gain (+0.73); TSS activity (CAGE/PRO-CAP): strong increase (+0.51).

Top hits by layer:

| Layer | Top Track | Effect | Interpretation |
|-------|-----------|--------|----------------|
| Chromatin | DNASE:LNCaP clone FGC | +1.236 | Very strong opening |
| Chromatin | DNASE:HeLa-S3 G1b phase | +1.149 | Very strong opening |
| TF binding | CHIP:HNF4A:liver (adult) | +1.125 | Very strong binding gain |
| TF binding | CHIP:RXRA:liver (adult) | +1.100 | Very strong binding gain |
| Histone | CHIP:H3K27ac:22Rv1 | +0.734 | Very strong mark gain |
| CAGE | CAGE:breast MDA-MB-453 | +0.505 | Strong increase |

**Key observations**:
- The strongest hits span many cell types (LNCaP, HeLa, MCF-7, placenta,
  kidney, esophagus) — consistent with a broadly active chromatin element
- **Liver TF signature is clear**: HNF4A and RXRA (both liver-specific
  transcription factors) show very strong binding gain in adult-liver
  tracks, directly matching the Musunuru 2010 mechanism (C/EBP family +
  liver nuclear factors upregulating SORT1)
- The Gene expression (RNA-seq) section is automatically omitted because
  Enformer doesn't have RNA tracks

## Cross-oracle comparison

Compare with the [AlphaGenome focused HepG2 analysis](../SORT1_rs12740374/)
(+0.449 DNASE:HepG2, +0.378 CEBPA:HepG2) and the [ChromBPNet 1bp analysis](../SORT1_chrombpnet/)
(−0.111 ATAC:HepG2 — opposite direction, see the ChromBPNet README for
why). Enformer's discovery-mode panorama complements the other two.

## Output files

- `rs12740374_SORT1_enformer_report.html` — interactive IGV report
- `example_output.md` — markdown with all scored tracks
- `example_output.json` — structured per-track scores
- `example_output.tsv` — tab-separated summary
