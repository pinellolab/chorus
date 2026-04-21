# Multi-Layer Variant Interpretation

## Overview

The `analyze_variant_multilayer` tool scores a variant's effect across all
regulatory layers using modality-specific strategies derived from the
AlphaGenome framework. It automatically classifies tracks into layers,
applies the right window/aggregation/formula for each, and produces a
structured report.

For non-coding variants, nearby genes are auto-detected within the prediction
window so that RNA expression effects can be scored without explicit gene
specification.

## Scoring Strategies

| Layer | Window | Aggregation | Formula | Effect percentile range |
|-------|--------|-------------|---------|----------------|
| Chromatin (DNASE/ATAC) | 501bp | sum | log2FC | [0, 1] |
| TF binding (ChIP-TF) | 501bp | sum | log2FC | [0, 1] |
| Histone marks (ChIP-Histone) | 2001bp | sum | log2FC | [0, 1] |
| TSS activity (CAGE) | 501bp | sum | log2FC | [0, 1] |
| Gene expression (RNA) | gene exons | mean | lnFC | [-1, 1] |
| Promoter activity (MPRA) | full output | mean | diff | [-1, 1] |
| Splicing | 501bp | sum | log2FC | [0, 1] |

## Normalization

Reports include two types of genome-wide context when background distributions
are available (built by `scripts/build_backgrounds_*.py`):

- **Effect quantile** — How unusual is this variant's effect compared to ~10K
  random SNPs? A quantile of 0.95 means the effect is larger than 95% of random
  variant effects on this layer.

- **Ref signal percentile (Ref %ile)** — How active is this region in the
  reference genome? A percentile of 0.90 means the predicted signal at this
  locus is higher than 90% of genomic positions. This contextualises the
  variant's effect: disrupting a highly active element (high ref percentile)
  is more likely to be functionally significant.

The combination is powerful: a variant with a 95th percentile effect on a track
that is already in the 91st percentile of genome-wide activity is disrupting a
highly active regulatory element — strong evidence for functional impact.

---

## Example MCP Prompts

### 1. Basic non-coding variant analysis

```
Analyze the regulatory effect of rs12740374 (chr1:109274968 G>T) using
AlphaGenome.

1. Load AlphaGenome
2. Search for HepG2 tracks: DNASE, H3K27ac, CEBPA ChIP, and CAGE
3. Use analyze_variant_multilayer with the selected tracks and gene_name="SORT1"
4. Interpret which regulatory layers are most affected
```

### 2. Multi-layer GWAS variant follow-up

```
I have a GWAS hit at chr6:32631828 (A>G) associated with autoimmune disease.
Help me understand its regulatory mechanism.

1. Load AlphaGenome
2. Find relevant immune cell tracks (T-cell or lymphocyte DNASE, ATAC,
   H3K27ac, CAGE, and any available TF ChIP tracks)
3. Run analyze_variant_multilayer — let it auto-detect nearby genes for
   expression scoring
4. Based on the results, tell me:
   - Is this variant in open chromatin?
   - Does it affect any TF binding sites?
   - Does it change expression of nearby genes?
   - What is the likely regulatory mechanism?
```

### 3. Comparing variant effects across cell types

```
Compare the regulatory effect of rs1421085 (chr16:53767042 T>C, FTO obesity
locus) across adipocyte and brain cell types.

1. Load AlphaGenome
2. For adipocytes: select DNASE, H3K27ac, CAGE tracks
3. Run analyze_variant_multilayer with gene_name="IRX3"
4. Repeat with brain/neural cell type tracks
5. Compare which layers show cell-type-specific effects
```

### 4. Enhancer variant with distal target gene

```
Analyze variant chr2:60718043 (G>A) which is in an enhancer ~400kb from
its target gene BCL11A.

1. Load AlphaGenome (needs 1Mb window to reach BCL11A)
2. Select erythroid/K562 tracks: DNASE, H3K27ac, GATA1 ChIP, CAGE, RNA
3. Run analyze_variant_multilayer with gene_name="BCL11A"
4. The tool will score chromatin/TF effects at the variant AND expression
   effects at BCL11A exons — showing the full enhancer-to-gene chain
```

### 5. Cross-oracle validation

```
Validate the effect of rs12740374 using multiple oracles:

1. Load AlphaGenome and run analyze_variant_multilayer with DNASE:HepG2,
   CEBPA ChIP:HepG2, H3K27ac:HepG2, and CAGE:HepG2 tracks
2. Load Enformer and run the same analysis with equivalent Enformer tracks
3. Compare scores across oracles — do they agree on the direction and
   magnitude of effect?
```

### 6. ChromBPNet base-resolution motif analysis

```
Zoom in on rs12740374 at base-pair resolution to see if it disrupts
or creates specific TF binding motifs:

1. Load ChromBPNet for ATAC accessibility in HepG2:
   load_oracle('chrombpnet', assay='ATAC', cell_type='HepG2')
2. Run analyze_variant_multilayer — this shows 1bp-resolution chromatin
   accessibility change right at the variant
3. Load ChromBPNet for CEBPA binding in HepG2:
   load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='CEBPA')
4. Run analyze_variant_multilayer — this shows whether the variant
   creates or destroys a C/EBP binding site at base resolution
5. Compare with the AlphaGenome results from step 1
```

### 7. Combining oracles (recommended deep analysis)

```
Do a complete multi-oracle analysis of rs12740374:

1. AlphaGenome: analyze_variant_multilayer with HepG2 DNASE, CEBPA ChIP,
   H3K27ac, CAGE, and RNA tracks. Gene is SORT1. This gives the broad
   multi-layer picture.

2. ChromBPNet (ATAC, HepG2): analyze_variant_multilayer on the same
   variant. This gives base-resolution chromatin accessibility.

3. ChromBPNet (CHIP, K562, TF=CEBPA): analyze_variant_multilayer on the
   same variant. This shows whether the C/EBP motif is created.

4. Compare across oracles and summarize: Do all three agree the variant
   opens chromatin and creates CEBP binding?
```

---

## Python API Usage

```python
from chorus.analysis import build_variant_report, get_normalizer

# Assume oracle is loaded and variant_result obtained:
# variant_result = oracle.predict_variant_effect(
#     genomic_region="chr1:109274968-109274969",
#     variant_position="chr1:109274968",
#     alleles=["G", "T"],
#     assay_ids=["DNASE:HepG2", "CAGE:HepG2", "H3K27ac:HepG2"],
# )

# Build report — auto-detects nearby genes for RNA scoring
report = build_variant_report(
    variant_result,
    oracle_name="alphagenome",
    gene_name="SORT1",  # optional — auto-detected if omitted
)

# Structured output
print(report.to_markdown())   # Formatted tables for reading
df = report.to_dataframe()    # For programmatic analysis
d = report.to_dict()          # JSON-serializable

# With quantile normalization (when backgrounds are pre-computed)
normalizer = get_normalizer("alphagenome")  # loads from ~/.chorus/backgrounds/
report = build_variant_report(
    variant_result,
    oracle_name="alphagenome",
    normalizer=normalizer,
)
```

---

## Expected Output Format

```
## Multi-Layer Variant Effect Report

**Variant**: chr1:109274968 G>T
**Oracle**: alphagenome
**Gene**: SORT1

#### Chromatin accessibility (DNASE/ATAC)

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| DNASE:HepG2 | 504 | 756 | +0.584 | Strong opening |

#### Transcription factor binding (ChIP-TF)

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| CEBPA:HepG2 | 12.3 | 45.7 | +1.874 | Very strong binding gain |

#### Histone modifications (ChIP-Histone)

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| H3K27ac:HepG2 | 2950 | 3240 | +0.137 | Moderate mark gain |

#### TSS activity (CAGE/PRO-CAP)

| Track | Ref | Alt | Effect | Interpretation |
|-------|-----|-----|--------|----------------|
| CAGE:HepG2 | 239 | 479 | +0.997 | Very strong increase |
```
