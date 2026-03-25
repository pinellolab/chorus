# AlphaGenome Paper Validation

Replication of key variant analyses from the AlphaGenome paper to verify
that Chorus produces consistent findings with the published results.

**Paper**: Avsec et al., "Advancing regulatory variant effect prediction
with AlphaGenome", Nature 649:1206-1218 (January 2026).
[DOI: 10.1038/s41586-025-10014-0](https://www.nature.com/articles/s41586-025-10014-0)

## Scoring formula verification

Our scoring implementations (in `chorus/analysis/scorers.py`) match the
AlphaGenome paper's recommended variant scoring formulas exactly:

| Modality | Window | Formula | Match |
|----------|--------|---------|-------|
| ATAC/DNase | 501bp | log2[(sum_alt+1)/(sum_ref+1)] | Exact |
| ChIP-TF | 501bp | log2[(sum_alt+1)/(sum_ref+1)] | Exact |
| ChIP-Histone | 2001bp | log2[(sum_alt+1)/(sum_ref+1)] | Exact |
| CAGE/PRO-CAP | 501bp | log2[(sum_alt+1)/(sum_ref+1)] | Exact |
| RNA-seq | Gene exons | log(mean_alt+0.001) - log(mean_ref+0.001) | Exact |

## Validation summary

| Variant | Locus | Paper Claim | Chorus Result | Status |
|---------|-------|-------------|---------------|--------|
| rs12740374 (G>T) | SORT1 / Fig.3 | C/EBP binding gain + CELSR2/PSRC1 upregulation in HepG2 | CEBPA +0.379, CEBPB +0.269, DNASE +0.450, CAGE +0.250 | Confirmed |
| chr5:1295046 (T>G) | TERT / Fig.4 | ETS/ELF1 binding gain + TERT expression increase in melanocytes | CAGE +0.120 at TERT TSS (correct direction). ELF1 binding cannot be validated — no melanocyte TF ChIP tracks available. | Partially confirmed |
| chr11:5254983 (G>C) | HBG2 HPFH | CTCF binding gain + HBG2 expression increase | K562: CTCF **-0.172** (loss), CAGE -0.098. Monocyte: CTCF -0.019, CAGE -0.072. Neither cell type shows the expected gain. | Not reproduced |

## Validation examples

### [SORT1_rs12740374_with_CEBP/](SORT1_rs12740374_with_CEBP/)
**Paper Fig.3**: rs12740374 (chr1:109274968 G>T) in HepG2 liver cells.

Expected: C/EBP TF binding gain at variant site, increased CELSR2 and
PSRC1 expression. The variant creates a C/EBP binding motif in a
liver-specific enhancer.

**Result**: All layers show concordant activation — this is the strongest
validation case.

Tracks used: DNASE, CEBPA ChIP, CEBPB ChIP, H3K27ac, CAGE+/-, RNA+/-
(all HepG2, matching the paper's liver analysis).

### [TERT_chr5_1295046/](TERT_chr5_1295046/)
**Paper Fig.4**: chr5:1295046 T>G in melanocytes.

Expected: ETS/ELF1 TF binding gain, increased TERT expression. The
variant creates an ETS factor binding motif driving telomerase
reactivation in melanoma.

**Result**: TSS activity (CAGE) shows the expected increase at the TERT
promoter (+0.120), confirming the direction of the paper's finding.
However, the ELF1 binding gain cannot be directly validated because
AlphaGenome does not provide melanocyte ELF1 TF ChIP-seq tracks. The
paper used ISM (in-silico mutagenesis) motif analysis to identify the ETS
motif, not direct ChIP scoring.

**Limitation**: The only missing piece — ELF1 binding — could potentially
be validated in a cross-cell-type context (K562 has ELF1 ChIP tracks),
though this would not match the paper's melanocyte context.

Tracks used: DNASE, H3K27ac, H3K4me1, CAGE+/-, RNA+/- (all melanocyte
ontology CL:2000045 / CL:0002566).

### [HBG2_HPFH/](HBG2_HPFH/)
**Paper**: chr11:5254983 G>C, hereditary persistence of fetal hemoglobin.

Expected: CTCF binding gain, increased HBG2 expression. The variant
creates a CTCF binding motif that reactivates fetal globin genes.

**Result — tested in two cell types:**

| Layer | K562 (erythroid) | Monocyte (CD14+) | Paper expectation |
|-------|-----------------|------------------|-------------------|
| CTCF ChIP | **-0.172** (moderate loss) | -0.019 (minimal) | Gain |
| DNase | +0.008 (minimal) | -0.014 (minimal) | — |
| H3K27ac | -0.044 (minimal) | +0.006 (minimal) | — |
| HBG2 CAGE | -0.098 (decrease) | -0.072 (decrease) | Increase |
| HBG2 RNA | -0.044 (decrease) | — | Increase |

**Key finding**: Neither cell type reproduces the paper's CTCF binding
gain. K562 actually shows a *moderate CTCF loss* (-0.172), which is the
opposite direction. Both cell types show HBG2 CAGE decrease rather than
the expected increase.

**Assessment**: Since both an erythroid and monocyte cell type show
consistent opposite-direction effects, the discrepancy is likely
**methodological** rather than cell-type-specific:

1. **ISM vs log2FC** — The paper used ISM (in-silico mutagenesis) and
   motif analysis to characterise this variant, not the log2FC scoring
   approach we use. ISM identifies which bases affect a prediction, while
   log2FC measures net signal change. A variant could create a CTCF motif
   that ISM detects while the net binding sum changes minimally or even
   decreases at the 501bp window level.
2. **Window-level vs motif-level** — Our 501bp window CTCF scoring sums
   all signal in the window. The CTCF binding gain may be localised to a
   narrow motif window that is diluted in the wider summation.
3. **This is educational** — it demonstrates that different scoring
   approaches (ISM vs log2FC) can give different interpretive conclusions
   for the same variant.

Tracks used:
- K562: DNASE, CTCF ChIP, H3K27ac, CAGE+/-, RNA+/- (EFO:0002067)
- Monocyte: DNASE, CTCF ChIP, H3K27ac, H3K4me1, H3K4me3, CAGE+/-,
  RNA+/- (CL:0001054 / CL:0002057)

## How to reproduce

```
# Load AlphaGenome
load_oracle('alphagenome')

# SORT1 — strongest validation
analyze_variant_multilayer(
    oracle_name='alphagenome',
    position='chr1:109274968',
    ref_allele='G', alt_alleles=['T'],
    assay_ids=[
        'DNASE/EFO:0001187 DNase-seq/.',
        'CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA/.',
        'CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.',
        'CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.',
        'CAGE/hCAGE EFO:0001187/+',
        'CAGE/hCAGE EFO:0001187/-',
        'RNA_SEQ/EFO:0001187 polyA plus RNA-seq/+',
        'RNA_SEQ/EFO:0001187 polyA plus RNA-seq/-',
    ],
    gene_name='CELSR2',
)

# TERT melanocyte validation
analyze_variant_multilayer(
    oracle_name='alphagenome',
    position='chr5:1295046',
    ref_allele='T', alt_alleles=['G'],
    assay_ids=[
        'DNASE/CL:2000045 DNase-seq/.',
        'CHIP_HISTONE/CL:2000045 Histone ChIP-seq H3K27ac/.',
        'CAGE/hCAGE CL:0002566/+',
        'CAGE/hCAGE CL:0002566/-',
        'RNA_SEQ/CL:2000045 polyA plus RNA-seq/+',
        'RNA_SEQ/CL:2000045 polyA plus RNA-seq/-',
    ],
    gene_name='TERT',
)

# HBG2 HPFH — K562 (erythroid)
analyze_variant_multilayer(
    oracle_name='alphagenome',
    position='chr11:5254983',
    ref_allele='G', alt_alleles=['C'],
    assay_ids=[
        'DNASE/EFO:0002067 DNase-seq/.',
        'CHIP_TF/EFO:0002067 TF ChIP-seq CTCF/.',
        'CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.',
        'CAGE/hCAGE EFO:0002067/+',
        'CAGE/hCAGE EFO:0002067/-',
        'RNA_SEQ/EFO:0002067 polyA plus RNA-seq/+',
        'RNA_SEQ/EFO:0002067 polyA plus RNA-seq/-',
    ],
    gene_name='HBG2',
)

# HBG2 HPFH — Monocyte (for comparison)
analyze_variant_multilayer(
    oracle_name='alphagenome',
    position='chr11:5254983',
    ref_allele='G', alt_alleles=['C'],
    assay_ids=[
        'DNASE/CL:0001054 DNase-seq/.',
        'CHIP_TF/CL:0001054 TF ChIP-seq CTCF/.',
        'CHIP_HISTONE/CL:0001054 Histone ChIP-seq H3K27ac/.',
        'CHIP_HISTONE/CL:0001054 Histone ChIP-seq H3K4me1/.',
        'CHIP_HISTONE/CL:0001054 Histone ChIP-seq H3K4me3/.',
        'CAGE/hCAGE CL:0002057/+',
        'CAGE/hCAGE CL:0002057/-',
        'RNA_SEQ/CL:0001054 polyA plus RNA-seq/+',
        'RNA_SEQ/CL:0001054 polyA plus RNA-seq/-',
    ],
    gene_name='HBG2',
)
```

## Oracle compatibility

These validation examples use AlphaGenome exclusively since the paper
specifically tests AlphaGenome's predictions. The same variants could
be analyzed with Enformer or Borzoi for cross-oracle comparison.
