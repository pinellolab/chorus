# Sequence Engineering Examples

Predict effects of sequence modifications: region replacement and construct
insertion. Both tools reuse the multi-layer scoring infrastructure.

> These examples show what the output of a swap/insertion looks like.
> Describe your own edit in plain language (the region, the construct
> sequence, the tissue you care about) and Claude will call the right
> tool. No special prompt format is required.

## Region Swap — `analyze_region_swap`

Replace a genomic region with a custom sequence and score effects across
all regulatory layers.

### Example Prompts

**Biologist (promoter engineering):**

> I want to replace a weak promoter upstream of my gene of interest with a
> stronger CMV-like promoter. The current promoter is at chr1:1000500-1001500.
> Can you predict how this swap would change chromatin accessibility, histone
> marks, and TSS activity in K562 cells?

**Computational biologist (enhancer design):**

> I'm testing synthetic enhancer sequences. Replace the region chr11:5248000-5249000
> (HBB locus enhancer) with my designed sequence: ACGT[...]. Compare the wild-type
> vs replacement effects on DNASE, H3K27ac, and CAGE tracks in K562 erythroid cells.

**Bioinformatician:**

> Load alphagenome. Call analyze_region_swap with:
> - region: chr1:1000500-1001500
> - replacement_sequence: [your DNA sequence]
> - assay_ids: [DNASE:K562, CAGE:K562, H3K27ac:K562, H3K4me3:K562]
> - description: "CMV promoter swap"
>
> Save the HTML report and return the JSON.

### MCP tool call

```python
analyze_region_swap(
    oracle_name="alphagenome",
    region="chr1:1000500-1001500",          # region to replace
    replacement_sequence="ACGTACGT...",      # new DNA sequence
    assay_ids=["DNASE:K562", "CAGE:K562", "H3K27ac:K562"],
    gene_name="SORT1",                       # optional
    description="Replace weak promoter with CMV",  # optional
)
```

### Example output: [region_swap/](region_swap/)
Promoter replacement in K562 — strong chromatin opening (+1.8 log2FC),
histone mark gain, and TSS activity increase at the swapped region.

---

## Integration Simulation — `simulate_integration`

Insert a construct (viral vector, transgene cassette) at a position and
predict regulatory disruption at the target locus.

### Example Prompts

**Gene therapy researcher:**

> I'm designing an AAV integration at the AAVS1 safe harbor locus
> (chr19:55115000). My construct is 4.7kb including a CMV promoter,
> GFP, and polyA signal. Can you predict how inserting this cassette
> would disrupt the local chromatin environment and nearby gene
> expression (PPP1R12C)?

**MD planning gene therapy:**

> We're evaluating candidate integration sites for a gene therapy vector.
> Compare the predicted chromatin disruption at three candidate positions:
> chr19:55115000 (AAVS1), chr3:46373000 (CCR5), chr1:206800000 (CR1).
> Which site causes the least disruption to surrounding regulatory elements?

**Computational biologist:**

> Simulate inserting a 2kb construct at chr19:55115000. Use K562 tracks
> (DNASE, H3K27ac, CAGE). I want to see the difference in chromatin
> accessibility and whether the insertion creates new TSS activity from
> the cassette promoter.

### MCP tool call

```python
simulate_integration(
    oracle_name="alphagenome",
    position="chr19:55115000",              # insertion point
    construct_sequence="ACGTACGT...",        # construct DNA
    assay_ids=["DNASE:K562", "CAGE:K562", "H3K27ac:K562"],
    gene_name="PPP1R12C",                   # optional
    description="AAV-GFP at AAVS1",         # optional
)
```

### Example output: [integration_simulation/](integration_simulation/)
GFP cassette insertion at AAVS1 — local chromatin closing (-0.9 log2FC
at DNASE), moderate histone mark loss, but new TSS activity (+1.3 log2FC)
from the cassette promoter. Nearby gene PPP1R12C auto-detected.

---

## Understanding the scores

Both tools produce the same multi-layer scores as variant analysis.
Effects are measured as **log2 fold-change** between wild-type and
modified sequences. See the [variant analysis README](../variant_analysis/)
for the full score interpretation guide.

Key difference from variant analysis: the "reference" is the wild-type
sequence and the "alternate" is your modified sequence (replacement or
insertion). Positive scores mean the modification increases activity;
negative scores mean it decreases activity.

## Oracle compatibility

Both tools work with any oracle. The available layers depend on which
tracks you select:

- **AlphaGenome**: all layers, including TF binding and RNA expression
- **Enformer**: chromatin, TF, histone, CAGE (no RNA)
- **Borzoi**: chromatin, TF, histone, CAGE, RNA
- **ChromBPNet**: chromatin and TF only — base-pair resolution for
  whether your modified sequence creates or destroys TF binding sites

### Using ChromBPNet for sequence engineering

ChromBPNet's 1bp resolution is particularly useful for promoter/enhancer
engineering to see exactly how each base in your replacement sequence
contributes to activity:

```
# Check chromatin accessibility of your designed sequence
load_oracle('chrombpnet', assay='ATAC', cell_type='K562')
analyze_region_swap('chrombpnet', region, new_sequence, assay_ids=['auto'])

# Check if your replacement creates desired TF binding
load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='CTCF')
analyze_region_swap('chrombpnet', region, new_sequence, assay_ids=['auto'])
```

### Multi-oracle comparison for integration sites

For gene therapy integration site evaluation, combine broad and narrow views:

```
1. AlphaGenome: simulate_integration → multi-layer disruption across 1Mb
2. ChromBPNet (ATAC): simulate_integration → base-resolution local disruption
3. Compare: does the insertion disrupt any critical regulatory elements?
```
