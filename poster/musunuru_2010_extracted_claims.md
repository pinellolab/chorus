# Musunuru 2010 — verified claims used in the poster panel

Source: Musunuru et al., "From noncoding to a phenotype association
of SNPs at the 1p13 cholesterol locus", *Nature* 466, 714-719 (2010).
PMC: [PMC3062476](https://pmc.ncbi.nlm.nih.gov/articles/PMC3062476/)
DOI: 10.1038/nature09266

Extracted 2026-05-19 via WebFetch against the PMC HTML. The agent
verifying this panel can re-fetch to confirm; quotes are verbatim
where in quotation marks.

## Allele and TF

- **Minor allele creates the C/EBP site**: *"the minor allele creating
  the site and the major allele disrupting it"*.
- **Specifically T**: *"rs12740374 minor allele (T) sequence, and the
  rs12740374 major allele (G) sequence"*.
- **Specifically C/EBPα** (not CEBPB): *"C/EBPα is a liver-enriched
  transcriptional factor"*; *"C/EBPα antibodies impaired the binding"*;
  *"addition of C/EBPα to the 3T3 cells restored the haplotype-specific
  effect"*.

## Gene-level expression in human liver

- *"Minor allele homozygotes displayed more than 12-fold higher SORT1
  and PSRC1 expression than major allele homozygotes"*.
- *"with no significant change for CELSR2"*.

This is the key fact-check for our row 6 / context claim:
- SORT1: ~12× UP in minor-allele liver
- PSRC1: ~12× UP
- CELSR2: no significant change

## Functional consequence in mice

- *"Sort1-overexpressing mice showed a marked decrease in total plasma
  cholesterol (70% reduction at two weeks, 73% reduction in very small
  LDL particles ..."*
- *"57% decrease in the rate of VLDL secretion"*
- *"Sort1 knockdown resulted in a 46% increase in total cholesterol
  ... more than two-fold increase in LDL-C"*

## Clinical / population

- *"Individuals of European descent who are homozygous for the major
  alleles ... have up to 16 mg/dl higher LDL-C as well as ~40% increased
  risk of MI when compared with minor allele homozygotes"*.
- *"effect comparable to those of common variants of LDLR and PCSK9"*.

So **minor allele = protective**: lower LDL-C, lower MI risk.

## In-vitro reporter assay (hepatoma)

- *"the minor haplotype produced significantly greater luciferase
  expression than the major haplotype"*.

This is what LegNet at the SNP site (narrow ~230 bp window) should
reproduce — alt allele scores higher in a single-fragment MPRA setup.
Our verification: LegNet ref=0.372, alt=0.669, raw=+0.297 at the SNP
site → alt > ref, direction-correct match to Musunuru's luciferase.

## Mismatches with AlphaGenome's predicted SORT1 / CELSR2 expression

What AG predicts (this verification, chorus v0.5.6):

| Gene   | AG RNA-seq max log₂FC | AG fold | Musunuru-measured fold |
|--------|----------------------:|--------:|-----------------------:|
| SORT1  | +0.14                 | ~1.15×  | **~12×**               |
| PSRC1  | +0.60                 | ~1.52×  | ~12×                   |
| CELSR2 | +0.86                 | ~1.82×  | **no significant change** |

Two honest caveats the panel surfaces:

1. AG's SORT1 RNA-seq prediction is **directionally correct but
   ~10× magnitude underestimate** vs the experimental measurement.
2. AG's CELSR2 RNA-seq prediction (+0.86, ~1.82×) is a **false positive**
   relative to Musunuru's qPCR result of no significant change.

Both flagged in row 6 verdict tag ("direction ✓, magnitude underestimated")
and the mechanism / ground-truth columns.
