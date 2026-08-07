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

**Summary**: rs12740374 creates a C/EBP motif, and Enformer's strongest response
is exactly that: CEBPB binding (+4.37 log2FC), beyond every one of the 17,909
sampled background effects for that track. Chromatin opens (+2.25), H3K4me1
increases (+1.89), and hepatocyte CAGE rises (+1.31).

Top hit per layer — **Effect is the raw log2 fold-change**; the percentile ranks it
against that track's background:

| Layer | Top Track | Effect (log2FC) | Effect %ile | Interpretation |
|-------|-----------|-----------------|-------------|----------------|
| TF binding | CHIP:CEBPb (CEBPb_HighDensity) | +4.372 | ≥99th (1.08× null max) | Motif created |
| Chromatin | DNASE:fibroblast of lung | +2.247 | 0.9998 | Very strong opening |
| Histone | CHIP:H3K4me1:neutrophil male | +1.886 | 0.9996 | Very strong mark gain |
| CAGE | CAGE:Hepatocyte | +1.310 | 0.9991 | Strong increase |

The CEBPB row is the one to read carefully. Its effect exceeds the largest of the
17,909 background effects sampled for that track, so the percentile is **clamped at
1.0 and cannot rank it further** — `1.08× null max` is the distance past that
ceiling. That is expected here rather than a defect: a null drawn from random
regulatory positions contains few single-base changes that build a complete
transcription-factor motif, which is precisely what this variant does. See
[`docs/BACKGROUND_NULL_PROTOCOL.md`](../../../../docs/BACKGROUND_NULL_PROTOCOL.md).

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
(+0.449 DNASE:HepG2, +0.387 CEBPA:HepG2) and the [ChromBPNet 1bp analysis](../SORT1_chrombpnet/)
(−0.111 ATAC:HepG2 — opposite direction, see the ChromBPNet README for
why). Enformer's discovery-mode panorama complements the other two.

## Output files

- `rs12740374_SORT1_enformer_report.html` — interactive IGV report
- `example_output.md` — markdown with all scored tracks
- `example_output.json` — structured per-track scores
- `example_output.tsv` — tab-separated summary
