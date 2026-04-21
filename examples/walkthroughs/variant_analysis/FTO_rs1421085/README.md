# FTO/IRX3 Obesity Locus

## Variant: rs1421085 (chr16:53,767,042 T>C)

This variant is in an intronic enhancer of FTO but exerts its effect on
obesity by **disrupting ARID5B repressor binding**, leading to increased
expression of **IRX3** and **IRX5** ~500kb away in adipocyte progenitors.
The risk allele (C) shifts adipocyte fate from energy-dissipating beige
to energy-storing white fat (Claussnitzer et al., NEJM 2015).

## Tracks

The committed `example_output.md` is run with **HepG2 liver tracks**
(DNASE, CEBPA/CEBPB ChIP, H3K27ac, CAGE) as a "nearest available
metabolic cell type" — matching the example prompt. HepG2 is not the
causal tissue: rs1421085 acts in adipocyte progenitors and IRX3 sits
~500 kb away, so expect **minimal effects** in the HepG2 run. The
example is included to show what a "no-signal" call looks like.

**For a scientifically ideal run**, switch to AlphaGenome's adipose
tracks by changing the `assay_ids` in the prompt:

```python
# Subcutaneous adipose tissue + adipose-derived mesenchymal stem cells
assay_ids = [
    "ATAC/UBERON:0002190 ATAC-seq/.",          # subcutaneous adipose
    "ATAC/CL:0002540 ATAC-seq/.",              # ADMSC
    "CAGE/hCAGE UBERON:0002190/+", "CAGE/hCAGE UBERON:0002190/-",
    "RNA_SEQ/UBERON:0002190 polyA plus RNA-seq/+",
    "RNA_SEQ/UBERON:0002190 polyA plus RNA-seq/-",
]
```

## Biology

The effect is highly cell-type-specific to adipocyte progenitors. IRX3 is
~500kb from the variant — only visible with AlphaGenome's 1Mb window.
In the HepG2 run you'll see "No strong regulatory effects detected" —
this is the **correct** result: the variant doesn't act in liver.
