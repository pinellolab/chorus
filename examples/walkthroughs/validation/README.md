# AlphaGenome Paper Validation

Replication of key variant analyses from the AlphaGenome paper to verify
that Chorus produces consistent findings with the published results.

**Paper**: Avsec et al., "Advancing regulatory variant effect prediction
with AlphaGenome", Nature 649:1206-1218 (January 2026).
[DOI: 10.1038/s41586-025-10014-0](https://www.nature.com/articles/s41586-025-10014-0)

> These are replication runs — same variants, same expected biology as
> the paper. They also serve as reference outputs: open any
> `example_output.md` or HTML report to see what a "good" Chorus run
> looks like on a well-characterised regulatory variant.

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
| rs12740374 (G>T) | SORT1 / Fig.3 | C/EBP binding gain + CELSR2/PSRC1 upregulation in HepG2 | CEBPA +0.379, CEBPB +0.269, DNASE +0.450, CAGE +0.239 | Confirmed |
| chr5:1295046 (T>G) | TERT / Fig.4 | ETS/ELF1 binding gain + TERT expression increase in melanocytes | CAGE +0.34 at TERT TSS (correct direction). ELF1 binding cannot be validated — no melanocyte TF ChIP tracks available. | Partially confirmed |

## Validation examples

### [SORT1_rs12740374_with_CEBP/](SORT1_rs12740374_with_CEBP/)
**Paper Fig.3**: rs12740374 (chr1:109274968 G>T) in HepG2 liver cells.

Expected: C/EBP TF binding gain at variant site, increased CELSR2 and
PSRC1 expression. The variant creates a C/EBP binding motif in a
liver-specific enhancer.

**Result**: All layers show concordant activation — this is the strongest
validation case.

Tracks used: DNASE, ATAC, CEBPA/CEBPB/CEBPG/CEBPD ChIP, H3K27ac, CAGE+/-,
RNA+/- (all HepG2, matching the paper's liver analysis) — 11 tracks, 123
scored rows across the prediction window.

This list was previously aspirational: it already claimed `RNA+/-`, but the
example scored neither RNA strand, and it covered only two of the four C/EBP
family members. The track set was widened on 2026-08-01 so the claim is true,
and so this example is no longer a byte-for-byte duplicate of
[`variant_analysis/SORT1_rs12740374`](../variant_analysis/SORT1_rs12740374/),
which scores the same variant with the narrower set.

The whole C/EBP family responds, which is the point of the validation:

| Track | Effect (log2FC) | Effect %ile |
|---|---|---|
| CHIP:CEBPB:HepG2 | +2.981 | ≥99th |
| CHIP:CEBPA:HepG2 | +2.709 | ≥99th |
| CHIP:CEBPG:HepG2 | +2.269 | ≥99th |
| CHIP:CEBPD:HepG2 | +1.739 | ≥99th |
| DNASE:HepG2 | +1.332 | ≥99th |
| CHIP:H3K27ac:HepG2 | +1.251 | ≥99th |
| ATAC:HepG2 | +0.732 | ≥99th |
| RNA:HepG2 (best of 58 rows) | +0.718 | ≥99th |

`+0.718` on RNA is a **1.65-fold** predicted increase — real, and well short
of the >12-fold difference in measured SORT1 mRNA between genotypes. A
single-variant model predicting the local regulatory change is not predicting
the full downstream expression difference, and the gap is expected rather than
a failure. Note also that AlphaGenome's RNA-seq effect background sits almost
entirely below `|log2FC| = 0.1`, so the RNA **percentile** carries very little
information here even though the effect itself is meaningful — read the log2FC
column, not the percentile. See
[#83](https://github.com/pinellolab/chorus/issues/83).

### [TERT_chr5_1295046/](TERT_chr5_1295046/)
**Paper Fig.4**: chr5:1295046 T>G in melanocytes.

Expected: ETS/ELF1 TF binding gain, increased TERT expression. The
variant creates an ETS factor binding motif driving telomerase
reactivation in melanoma.

**Result**: TSS activity (CAGE) shows the expected increase at the TERT
promoter (+0.34), confirming the direction of the paper's finding.
However, the ELF1 binding gain cannot be directly validated because
AlphaGenome does not provide melanocyte ELF1 TF ChIP-seq tracks. The
paper used ISM (in-silico mutagenesis) motif analysis to identify the ETS
motif, not direct ChIP scoring.

**Limitation**: The only missing piece — ELF1 binding — could potentially
be validated in a cross-cell-type context (K562 has ELF1 ChIP tracks),
though this would not match the paper's melanocyte context.

Tracks used: DNASE, H3K27ac, H3K4me1, CAGE+/-, RNA+/- (all melanocyte
ontology CL:2000045 / CL:0002566).

### [SORT1_rs12740374_multioracle/](SORT1_rs12740374_multioracle/)
**Multi-oracle cross-validation** of the same SORT1 variant with three
independent models: **ChromBPNet** (chromatin accessibility specialist),
**LegNet** (MPRA / promoter activity specialist), and **AlphaGenome**
(generalist). The consolidated report renders a cross-oracle consensus
matrix flagging where the models agree on direction (✅ all ↑) and where
they disagree (⚠ disagree) per regulatory layer — complementary to the
single-oracle validation above.

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

```

## Oracle compatibility

These validation examples use AlphaGenome exclusively since the paper
specifically tests AlphaGenome's predictions. The same variants could
be analyzed with Enformer or Borzoi for cross-oracle comparison.
