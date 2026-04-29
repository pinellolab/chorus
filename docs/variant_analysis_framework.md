# Chorus: Non-Coding Variant Analysis Framework

A systematic approach to interpreting GWAS variants using genomic deep learning oracles.

## Recommended Oracle Strategy

**AlphaGenome is the primary oracle.** With a 1Mb output window at 1bp resolution and 5731 tracks
(DNASE, ATAC, CAGE, RNA-seq, ChIP-seq, splice sites, PRO-CAP), it is the most comprehensive
model available. It can capture variant-to-gene effects up to 500kb+ — essential for distal
enhancer variants, which are the majority of GWAS signals.

> Chorus ships **two interchangeable AlphaGenome backends** — `alphagenome` (JAX, default)
> and `alphagenome_pt` (PyTorch, same model + same weights, MPS-capable on Apple Silicon).
> Both install by default and produce equivalent outputs (1–2 % per-track fp32 noise).
> Use `chorus.recommend_alphagenome_backend(window_size_bp)` (or the matching MCP tool) to
> pick the right backend for your platform + window size. The framework below applies
> identically to both.

| Oracle | Resolution | Output window | Role in analysis |
|--------|-----------|---------------|------------------|
| **AlphaGenome** | 1bp | **1 Mb** | **Primary.** Full 5-layer analysis: accessibility, histone marks, TF ChIP, CAGE, RNA-seq. Can reach distal target genes. |
| **ChromBPNet** | 1bp | 1 kb | **Local TF analysis.** Base-resolution ATAC/DNASE accessibility and motif-level TF disruption at the variant. |
| **BPNet (via ChromBPNet)** | 1bp | 1 kb | **TF-specific binding.** Load with `assay="CHIP", TF="GATA1"` for base-resolution binding prediction of specific TFs. |
| **Enformer** | 128bp | 114 kb | **Complementary.** Large ENCODE ChIP-seq catalog (5313 tracks). Useful when AlphaGenome lacks a specific track. |
| **Borzoi** | 32bp | 196 kb | **Complementary.** Higher resolution than Enformer, larger window. |
| **Sei** | N/A | 4 kb | **Classification.** Predicts regulatory element classes, not per-track signals. |
| **LegNet** | N/A | 200 bp | **MPRA.** Predicts lentiMPRA reporter activity. |

**Important:** AlphaGenome contains CAGE, RNA-seq, DNASE, ATAC, ChIP-seq, and more — it
subsumes much of what Enformer and Borzoi offer, with a far larger output window. Always
start with AlphaGenome. Use Enformer/Borzoi only when you need a specific ENCODE track
not available in AlphaGenome's catalog.

---

## The 5-Layer Analysis

For any non-coding variant, ask these questions in order:

### Layer 1: Chromatin Accessibility
**Question:** Is the variant in an open/accessible regulatory region?
**Tracks:** DNASE or ATAC in the disease-relevant cell type
**What to look for:** Variant overlaps an accessibility peak → it's in a regulatory element.

### Layer 2: Regulatory Element Type
**Question:** What kind of element — active enhancer, poised enhancer, or promoter?
**Tracks:** H3K27ac (active enhancer/promoter), H3K4me1 (poised enhancer), H3K4me3 (promoter)
**What to look for:**
- H3K27ac+ H3K4me1+ → active enhancer
- H3K27ac- H3K4me1+ → poised enhancer
- H3K4me3+ → active promoter

### Layer 3: TF Binding
**Question:** Which transcription factors bind there, and does the variant alter binding?
**Tracks:** ChIP-seq for candidate TFs (based on disease biology)
**What to look for:** Ref→Alt effect on TF signal. Positive = creates site. Negative = disrupts.

**For base-resolution TF analysis:** Use ChromBPNet/BPNet models:
```
load_oracle("chrombpnet", assay="CHIP", cell_type="K562", TF="GATA1")
```

### Layer 4: Gene Expression
**Question:** Does the variant change expression of the target gene?
**Tracks:** CAGE (TSS-level quantification) AND RNA-seq (gene body coverage)
**What to look for:** Fold change at target gene TSS (CAGE) or across exons (RNA-seq)

**Critical:** Most GWAS variants are distal enhancers regulating genes 50-500kb away.
AlphaGenome's 1Mb window is essential here — Enformer (114kb) and Borzoi (196kb) often
cannot reach the target gene. Even when a nearby gene shows an effect, the true causal
gene may be further away. Always check both nearby genes AND the known/suspected target.

### Layer 5: Cell-Type Specificity
**Question:** Is the effect specific to the disease-relevant tissue?
**Approach:** Compare effects across cell types. AlphaGenome has CAGE and RNA-seq for
many primary cell types and tissues, making cross-tissue comparison straightforward.

---

## Worked Example: rs12740374 (SORT1 / LDL Cholesterol)

### Background
- **Variant:** chr1:109274968 G>T
- **Trait:** LDL cholesterol, coronary artery disease
- **Known mechanism:** Creates a C/EBP binding site in a liver enhancer, increasing SORT1 expression
- **Key challenge:** SORT1 TSS is 118kb from variant — needs AlphaGenome's 1Mb window

### Recommended approach: AlphaGenome for full analysis

```
# Step 1: Find hepatocyte tracks in AlphaGenome
list_tracks("alphagenome", query="hepatocyte")
# → CAGE/hCAGE CL:0000182/+  (hepatocyte CAGE, + strand)
# → CAGE/hCAGE CL:0000182/-  (hepatocyte CAGE, - strand)

list_tracks("alphagenome", query="HepG2")
# → CAGE/hCAGE EFO:0001187/+  (HepG2 CAGE)
# → DNASE tracks, CHIP tracks...

list_tracks("alphagenome", query="RNA")
# → RNA_SEQ tracks for many cell types including hepatocytes

# Step 2: Load AlphaGenome
load_oracle("alphagenome")  # ~80s first time, cached after

# Step 3: Predict variant effect with hepatocyte expression tracks
predict_variant_effect(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    assay_ids=[
        "CAGE/hCAGE CL:0000182/+",   # hepatocyte CAGE + strand
        "CAGE/hCAGE CL:0000182/-",   # hepatocyte CAGE - strand
    ],
    # region auto-centered — 1Mb window covers SORT1 TSS at 118kb!
)

# Step 4: Score at SORT1 TSS (118kb from variant — only AlphaGenome can do this)
score_variant_effect_at_region(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    assay_ids=["CAGE/hCAGE CL:0000182/-"],  # SORT1 is on - strand
    score_region="chr1:109390000-109400000",  # SORT1 TSS region
)

# Step 5: Gene expression analysis — works with AlphaGenome!
predict_variant_effect_on_gene(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    gene_name="SORT1",
    assay_ids=["CAGE/hCAGE CL:0000182/+", "CAGE/hCAGE CL:0000182/-"],
    # SORT1 TSS at 118kb — fits easily in AlphaGenome's 1Mb window!
)
```

### Complementary: Enformer for deep TF ChIP catalog

AlphaGenome covers the main tracks, but Enformer has a larger ENCODE ChIP-seq catalog.
Use Enformer when you need specific TF binding predictions not available in AlphaGenome:

```
# Enformer has extensive HepG2 ChIP-seq:
load_oracle("enformer")
score_variant_effect_at_region(
    oracle_name="enformer",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    assay_ids=[
        "ENCFF136DBS",   # DNASE HepG2
        "ENCFF003HJB",   # CEBPB ChIP HepG2
        "ENCFF559CVP",   # CEBPA ChIP HepG2
        "ENCFF080FZD",   # HNF4A ChIP HepG2
    ],
    at_variant=True, window_bins=5,
)
```

### Enformer results (validated in testing)

| Layer | Track | Ref | Alt | Effect | Interpretation |
|-------|-------|-----|-----|--------|----------------|
| 1. Accessibility | DNASE HepG2 | 1.02 | 1.17 | +0.14 | Opens chromatin |
| 2. Element type | H3K4me1 HepG2 | 16.63 | 16.37 | -0.25 | Poised→active transition |
| 3. TF binding | CEBPA ChIP | 3.86 | 4.68 | **+0.82** | Creates C/EBP-alpha site |
| 3. TF binding | CEBPB ChIP | 3.50 | 3.72 | +0.22 | Creates C/EBP-beta site |
| 3. TF binding | HNF4A ChIP | 4.79 | 5.18 | +0.38 | Recruits liver master TF |
| 4. Expression | CAGE Liver | 0.059 | 0.104 | +0.045 | +76% local expression |

### AlphaGenome results (validated in testing)

| Location scored | Track | Effect | Interpretation |
|----------------|-------|--------|----------------|
| At variant | CAGE hepatocyte (+) | +0.00065 | Local expression increase |
| IRX3 TSS (520kb) | CAGE hepatocyte (-) | +0.00073 | Distal gene upregulated |
| FTO TSS (66kb) | CAGE hepatocyte (+) | -0.00046 | Nearby gene slightly decreased |

Note: AlphaGenome results shown for the FTO/IRX3 locus (rs1421085) where the 1Mb
window is essential. For rs12740374/SORT1, AlphaGenome can reach the SORT1 TSS at
118kb — something Enformer cannot do.

### Biological conclusion
The multi-oracle approach correctly predicts:
1. **Creates a C/EBP binding site** (Enformer CEBPA effect +0.82)
2. Opens chromatin and recruits HNF4A
3. Increases hepatic expression
4. AlphaGenome can additionally measure the effect at the true target gene (SORT1)
   118kb away — confirming the distal regulatory mechanism

---

## Worked Example: rs1421085 (FTO/IRX3 / Obesity)

This variant demonstrates why AlphaGenome is essential.

- **Variant:** chr16:53767042 T>C in FTO intron 1
- **True target gene:** IRX3 — **520kb away** (not FTO!)
- **Mechanism:** Disrupts ARID5B repressor → increases IRX3/IRX5 in preadipocytes

**Only AlphaGenome can see this:**
```
load_oracle("alphagenome")
# Score at IRX3 TSS — 520kb from the variant
score_variant_effect_at_region(
    oracle_name="alphagenome",
    position="chr16:53767042",
    ref_allele="T", alt_alleles=["C"],
    assay_ids=["CAGE/hCAGE CL:0000182/+", "CAGE/hCAGE CL:0000182/-"],
    score_region="chr16:54283000-54290000",  # IRX3 gene
)
```

**Result:** IRX3 CAGE effect = **+0.000726** (increased expression), matching published biology.
Enformer (114kb window) cannot reach IRX3 at all.

---

## Track Selection by Oracle

### AlphaGenome (5731 tracks — start here)
```
list_tracks("alphagenome", query="hepatocyte")   # CAGE/RNA for liver
list_tracks("alphagenome", query="K562")          # CAGE for erythroid
list_tracks("alphagenome", query="fat")           # CAGE for adipocyte
list_tracks("alphagenome", query="GATA1")         # TF ChIP
list_tracks("alphagenome", query="RNA")           # RNA-seq (1504 tracks!)
list_tracks("alphagenome", query="DNASE")         # Accessibility
```
Track ID format: `CAGE/hCAGE CL:0000182/+`, `CHIP_TF/EFO:0002067 TF ChIP-seq GATA1/.`

### ChromBPNet/BPNet (base-resolution local analysis)
```
list_tracks("chrombpnet", query="K562")
# ATAC K562, DNASE K562, plus 100+ TF-cell_type CHIP combinations

# Load ATAC accessibility model:
load_oracle("chrombpnet", assay="ATAC", cell_type="K562")

# Load specific TF binding model:
load_oracle("chrombpnet", assay="CHIP", cell_type="K562", TF="GATA1")
```

### Enformer (5313 tracks — complementary for deep ENCODE ChIP catalog)
```
list_tracks("enformer", query="HepG2")    # 266 tracks including many TF ChIPs
list_tracks("enformer", query="GATA1")    # TF ChIP tracks
list_tracks("enformer", query="liver")    # Primary tissue tracks
```

### Disease-specific track recommendations

| Disease area | Cell type | Key TFs | Oracle for expression |
|-------------|-----------|---------|----------------------|
| **Sickle cell / HbF** | K562 (erythroid) | GATA1, TAL1, KLF1 | AlphaGenome (BCL11A enhancer is ~60kb distal) |
| **LDL / CAD** | HepG2, hepatocyte | CEBPA/B, HNF4A, FOXA1/2 | AlphaGenome (SORT1 is 118kb from variant) |
| **Obesity / BMI** | Adipocyte, preadipocyte | ARID5B, PPARG | AlphaGenome (IRX3 is 500kb from FTO variant!) |
| **Autoimmune** | GM12878, T-cell | NFKB, STAT, IRF | AlphaGenome for distal targets |
| **Neuropsychiatric** | Astrocyte, neuron | SOX, POU3F | AlphaGenome for long-range enhancers |

---

## Important Caveats

### 1. "Nearest gene" is often not the causal gene
The variant rs1421085 sits in FTO but the true target is IRX3, 500kb away.
**Always check multiple candidate genes**, not just the nearest one. GWAS loci
often skip over several genes to regulate a distal target via chromatin looping.
AlphaGenome's 1Mb window lets you systematically score all genes in the locus.

### 2. Cell-type matters more than model choice
A perfect model with the wrong cell type gives misleading results. rs12740374
shows strong effects in HepG2/liver tracks but minimal signal in K562 (wrong tissue).
**Always use disease-relevant cell types** — this is more important than which oracle you use.

### 3. Sequence models cannot capture 3D chromatin looping
These models predict from DNA sequence alone. They learn correlations between
sequence features and regulatory activity, but they don't explicitly model
chromatin loops, TADs, or CTCF-mediated insulation. AlphaGenome's large window
partially compensates by capturing long-range sequence dependencies, but true
3D genome effects (e.g., loop extrusion, compartment switching) are not modeled.

### 4. Validate predictions with experimental data
Model predictions are hypotheses, not proof. Use them to:
- **Prioritize** which variants to test experimentally
- **Select** which TFs and cell types to focus on
- **Design** reporter assays, CRISPRi experiments, or allele-specific ChIP

---

## MCP Workflow Quick Reference

```
# 1. Discover what's available
list_oracles()
list_tracks("alphagenome", query="hepatocyte")

# 2. Load oracles (cached after first load)
load_oracle("alphagenome")
load_oracle("chrombpnet", assay="ATAC", cell_type="K562")

# 3. Full variant prediction (region auto-centered)
predict_variant_effect(
    oracle_name="alphagenome",
    position="chr1:109274968",
    ref_allele="G", alt_alleles=["T"],
    assay_ids=["CAGE/hCAGE CL:0000182/+", "CAGE/hCAGE CL:0000182/-"],
)

# 4. Score at variant site
score_variant_effect_at_region(
    ..., at_variant=True, window_bins=5
)

# 5. Score at target gene TSS
score_variant_effect_at_region(
    ..., score_region="chr1:109390000-109400000"
)

# 6. Gene expression fold change
predict_variant_effect_on_gene(
    ..., gene_name="SORT1"
)
```

## Notebook Quick Start

```python
from chorus import create_oracle
from chorus.utils.genome import GenomeManager
from chorus.core.result import score_variant_effect

# Setup — AlphaGenome as primary oracle
gm = GenomeManager()
fasta = str(gm.get_genome_path("hg38"))
oracle = create_oracle("alphagenome", use_environment=True, reference_fasta=fasta)
oracle.load_pretrained_model()

# Predict variant effect — 1Mb window captures distal genes
result = oracle.predict_variant_effect(
    genomic_region="chr1:109274968-109274969",
    variant_position="chr1:109274968",
    alleles=["G", "T"],
    assay_ids=[
        "CAGE/hCAGE CL:0000182/+",  # hepatocyte CAGE + strand
        "CAGE/hCAGE CL:0000182/-",  # hepatocyte CAGE - strand
    ],
)

# Score at variant site
scores = score_variant_effect(result, at_variant=True, window_bins=50, scoring_strategy="mean")

# Score at SORT1 TSS — 118kb away, easily within 1Mb window
sort1_scores = score_variant_effect(
    result, chrom="chr1", start=109390000, end=109400000, scoring_strategy="mean"
)

# Gene expression fold change
gene_result = oracle.analyze_variant_effect_on_gene(result, "SORT1")
```
