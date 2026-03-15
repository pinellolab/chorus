# Chorus: Non-Coding Variant Analysis Framework

A systematic approach to interpreting GWAS variants using genomic deep learning oracles.

## The 5-Layer Analysis

For any non-coding variant, ask these questions in order:

### Layer 1: Chromatin Context
**Question:** Is the variant in an open/accessible regulatory region?
**Tracks:** DNASE or ATAC in the disease-relevant cell type
**What to look for:** Variant overlaps an accessibility peak → it's in a regulatory element

### Layer 2: Regulatory Element Type
**Question:** What kind of element is it — active enhancer, poised enhancer, or promoter?
**Tracks:** H3K27ac (active enhancer/promoter), H3K4me1 (poised enhancer), H3K4me3 (promoter)
**What to look for:**
- H3K27ac+ H3K4me1+ → active enhancer
- H3K27ac- H3K4me1+ → poised enhancer
- H3K4me3+ → promoter

### Layer 3: TF Binding
**Question:** Which transcription factors bind at this element, and does the variant alter binding?
**Tracks:** ChIP-seq for candidate TFs (based on disease biology)
**What to look for:** Ref→Alt effect on TF binding signal. Positive = creates binding site. Negative = disrupts.

### Layer 4: Gene Expression
**Question:** Does the variant change expression of the target gene?
**Tracks:** CAGE (TSS-level) or RNA-seq (gene body) in the relevant tissue
**What to look for:** Fold change at the target gene TSS

### Layer 5: Cell-Type Specificity
**Question:** Is the effect specific to the disease-relevant tissue?
**Approach:** Compare effects across cell types (e.g., HepG2 vs K562 for a liver variant)

---

## Oracle Selection Guide

| Oracle | Resolution | Window | Best for |
|--------|-----------|--------|----------|
| **Enformer** | 128bp | 114kb output | Chromatin marks, TF ChIP, CAGE at nearby genes |
| **Borzoi** | 32bp | 196kb output | Higher-resolution, genes up to ~100kb away |
| **ChromBPNet** | 1bp | 1kb output | Base-resolution accessibility/TF binding at the variant |
| **AlphaGenome** | 1bp | 1Mb output | Distal target genes (up to 500kb), RNA-seq, largest track catalog |
| **Sei** | N/A | 4kb | Regulatory element classification |
| **LegNet** | N/A | 230bp | MPRA activity prediction |

**Rule of thumb:**
- Variant-to-gene < 50kb → Enformer is sufficient
- Variant-to-gene 50-100kb → Use Borzoi or smart region placement in Enformer
- Variant-to-gene > 100kb → AlphaGenome (1Mb window) is required
- Base-resolution TF motif disruption → ChromBPNet

---

## Worked Example: rs12740374 (SORT1 / LDL Cholesterol)

### Background
- **Variant:** chr1:109274968 G>T
- **Trait:** LDL cholesterol, coronary artery disease
- **Known mechanism:** Creates a C/EBP binding site in a liver enhancer, increasing SORT1 expression

### Step 1: Discover tracks
```
list_tracks("enformer", query="HepG2")
# Key tracks found:
# ENCFF136DBS  — DNASE HepG2 (accessibility)
# ENCFF058GCZ  — H3K4me1 HepG2 (poised enhancer)
# ENCFF003HJB  — CEBPB ChIP HepG2 (C/EBP-beta)
# ENCFF559CVP  — CEBPA ChIP HepG2 (C/EBP-alpha)
# ENCFF080FZD  — HNF4A ChIP HepG2 (liver master TF)

list_tracks("enformer", query="liver")
# CNhs10624 — CAGE liver (expression)
```

### Step 2: Load oracle
```
load_oracle("enformer")
```

### Step 3: Score at variant site
```
score_variant_effect_at_region(
    oracle_name="enformer",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    assay_ids=[
        "ENCFF136DBS",   # DNASE HepG2
        "ENCFF058GCZ",   # H3K4me1 HepG2
        "ENCFF003HJB",   # CEBPB ChIP
        "ENCFF559CVP",   # CEBPA ChIP
        "ENCFF080FZD",   # HNF4A ChIP
        "CNhs10624",     # CAGE Liver
    ],
    at_variant=True,
    window_bins=5,
)
```

### Results

| Layer | Track | Ref | Alt | Effect | Interpretation |
|-------|-------|-----|-----|--------|----------------|
| 1. Accessibility | DNASE HepG2 | 1.02 | 1.17 | +0.14 | Opens chromatin |
| 2. Element type | H3K4me1 HepG2 | 16.63 | 16.37 | -0.25 | Poised→active transition |
| 3. TF binding | CEBPA ChIP | 3.86 | 4.68 | **+0.82** | Creates C/EBP-alpha site |
| 3. TF binding | CEBPB ChIP | 3.50 | 3.72 | +0.22 | Creates C/EBP-beta site |
| 3. TF binding | HNF4A ChIP | 4.79 | 5.18 | +0.38 | Recruits liver master TF |
| 4. Expression | CAGE Liver | 0.059 | 0.104 | +0.045 | +76% local expression |

### Step 4: Gene expression
```
# Nearby gene (in window):
predict_variant_effect_on_gene("enformer", ..., gene_name="CELSR2", ...)
# → fold_change: 1.036, log2FC: +0.052

# Target gene (too far for Enformer):
predict_variant_effect_on_gene("enformer", ..., gene_name="SORT1", ...)
# → WARNING: SORT1 TSS is 118kb away, outside 115kb output window
# → Use borzoi or alphagenome instead
```

### Biological conclusion
Enformer correctly predicts that rs12740374:
1. **Creates a C/EBP binding site** (CEBPA effect +0.82, strongest signal)
2. Opens chromatin and recruits the liver master regulator HNF4A
3. Increases local liver expression
4. The poised→active enhancer transition (H3K4me1 decrease) is consistent with H3K27ac gain

This matches the published mechanism (Musunuru et al., Nature 2010).

---

## Track Selection Cheat Sheet

### For erythroid variants (sickle cell, thalassemia, HbF)
```
list_tracks("enformer", query="K562")       # DNASE, CHIP tracks
list_tracks("enformer", query="GATA1")      # Key erythroid TF
```
Key tracks: DNASE K562, GATA1 ChIP, TAL1 ChIP, CAGE K562

### For liver/lipid variants (LDL, cholesterol, NAFLD)
```
list_tracks("enformer", query="HepG2")      # Hepatocyte cell line
list_tracks("enformer", query="liver")       # Primary tissue
```
Key tracks: DNASE HepG2, CEBPB/CEBPA ChIP, HNF4A ChIP, FOXA1/2 ChIP, CAGE liver

### For immune variants (autoimmune, inflammatory)
```
list_tracks("enformer", query="GM12878")     # B-cell lymphoblastoid
list_tracks("enformer", query="Jurkat")      # T-cell
```

### For adipose/metabolic variants (BMI, T2D)
```
list_tracks("alphagenome", query="fat")      # Adipocyte CAGE
list_tracks("alphagenome", query="adipose")  # Adipose tissue
```
Note: Use AlphaGenome for these — IRX3/FTO effects span 500kb+

---

## Known Limitations & Workarounds

### 1. Gene TSS outside output window
**Problem:** Distal enhancer variants (most GWAS variants) often regulate genes >100kb away.
Enformer's 114kb window cannot reach them.

**Workaround:**
- Use `region` parameter to span variant-to-TSS when distance < output window
- Use Borzoi (196kb) or AlphaGenome (1Mb) for distal targets
- The MCP server now warns you with the exact distance

### 2. Multi-word track search returns 0 results
**Problem:** `list_tracks("enformer", query="CAGE K562")` returns nothing.

**Workaround:** Search one term at a time:
```
list_tracks("enformer", query="CAGE")    # then scan for K562 in results
list_tracks("enformer", query="K562")    # then scan for CAGE in results
```

### 3. Cell-type specificity not guided
**Problem:** No tool suggests which cell type is relevant for a given variant/trait.

**Workaround:** Use biological knowledge — match tissue to trait:
- Blood trait → K562, GM12878
- Liver trait → HepG2, liver CAGE
- Brain trait → astrocyte, neuron tracks
- Metabolic → adipose, pancreas

### 4. No enhancer/promoter annotation
**Problem:** No tool tells you if the variant is IN a peak vs. in background.

**Workaround:** Run wild-type prediction first, check if the variant position
has high DNASE/ATAC signal (>1.0 suggests an accessible peak).

---

## Notebook Quick Start

```python
from chorus import create_oracle
from chorus.utils.genome import GenomeManager
from chorus.core.result import score_variant_effect

# Setup
gm = GenomeManager()
fasta = str(gm.get_genome_path("hg38"))
oracle = create_oracle("enformer", use_environment=True, reference_fasta=fasta)
oracle.load_pretrained_model()

# Predict variant effect
result = oracle.predict_variant_effect(
    genomic_region="chr1:109274968-109274969",  # auto-sized by oracle
    variant_position="chr1:109274968",
    alleles=["G", "T"],
    assay_ids=["ENCFF136DBS", "ENCFF003HJB", "CNhs10624"],
)

# Score at variant site
scores = score_variant_effect(
    result, at_variant=True, window_bins=5, scoring_strategy="mean"
)

# Gene expression
gene_result = oracle.analyze_variant_effect_on_gene(result, "CELSR2")
print(f"Fold change: {gene_result['per_allele']['alt_1']['vs_reference']['CNhs10624']['fold_change']:.3f}")
```
