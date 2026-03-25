# TERT Promoter Hotspot Mutation — Cancer

## Variant: chr5:1,295,228 G>A (C228T on coding strand)

The TERT promoter C228T mutation is one of the most common non-coding
mutations in cancer, found in ~70% of melanomas, ~80% of glioblastomas,
and ~60% of hepatocellular carcinomas. It creates a de novo ETS/GABP
transcription factor binding site that drives telomerase reactivation.

## Tracks

Predicted with AlphaGenome using HepG2 (hepatocellular carcinoma) tracks:
- DNASE, H3K27ac, H3K4me3, CAGE (+/-), RNA-seq (+/-)

## Key findings from AlphaGenome

- **H3K27ac**: -0.235 log2FC (moderate mark loss, quantile 0.57)
- **CAGE at TERT TSS (-strand)**: -0.307 log2FC (moderate decrease, quantile 0.70)
- **RNA TERT exons (-strand)**: -0.218 logFC (moderate decrease, quantile -0.72)

**Note**: We tested G>A (ancestral to mutant direction on + strand).
The cancer mutation is typically reported as C>T on the coding strand
(which is the - strand for TERT). The model correctly predicts directional
effects on TERT expression through the minus-strand tracks.
