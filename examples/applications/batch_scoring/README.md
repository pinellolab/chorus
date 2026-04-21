# Batch Variant Scoring

Score multiple variants and rank by regulatory effect magnitude using
`score_variant_batch`. Useful for prioritizing variants from VCFs,
GWAS fine-mapping, or ClinVar triage.

> Paste a variant list in any format — rsIDs, VCF lines, space-separated
> coordinates — and ask Claude to score them. It'll parse the input and
> build the `variants` argument for you. No need to hand-craft JSON.

## Example Prompts

### For a geneticist (VCF triage)

> I have 12 non-coding variants of uncertain significance from a clinical
> exome. Here are the positions:
>
> chr1:109274968 G>T (rs12740374)
> chr1:109275684 C>T (rs629301)
> chr1:109272715 A>G (rs12037222)
> chr1:109278590 G>A (rs2228603)
> chr1:109271200 T>C (rs7528419)
>
> Score all of them in HepG2 liver cells and rank by effect size.
> Which ones are most likely to be functional?

Claude will: parse the variants, call `score_variant_batch` with HepG2
DNASE/CAGE/H3K27ac tracks, and return a ranked table showing which variants
have the strongest regulatory effects and in which layers.

### For a bioinformatician (VCF pipeline)

> I have a VCF file at /path/to/variants.vcf with ~50 variants near the
> SORT1 locus. Load alphagenome, parse the VCF, and call score_variant_batch
> with assay_ids for HepG2 DNASE and CAGE tracks. Return the top 20 by
> max_effect as JSON.

### For a computational biologist (fine-mapping)

> I'm fine-mapping a GWAS locus for LDL cholesterol. I have a 95% credible
> set of 8 variants. Score each one across chromatin, histone, and TSS layers
> in hepatocytes. I want to see which variant is the most likely causal SNP
> based on the largest regulatory disruption.

### For an MD (clinical prioritization)

> A patient has several non-coding variants flagged by whole genome sequencing.
> Can you score these variants and tell me which ones are most likely to have
> regulatory consequences? I need a ranked list to discuss with the genetics
> team.

## MCP tool call

```python
score_variant_batch(
    oracle_name="alphagenome",
    variants=[
        {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
        {"chrom": "chr1", "pos": 109275684, "ref": "C", "alt": "T", "id": "rs629301"},
        {"chrom": "chr1", "pos": 109272715, "ref": "A", "alt": "G", "id": "rs12037222"},
        {"chrom": "chr1", "pos": 109278590, "ref": "G", "alt": "A", "id": "rs2228603"},
        {"chrom": "chr1", "pos": 109271200, "ref": "T", "alt": "C", "id": "rs7528419"},
    ],
    assay_ids=["DNASE:HepG2", "CAGE:HepG2", "H3K27ac:HepG2"],
    gene_name="SORT1",
    top_n=20,
)
```

**Note**: Claude can parse VCF content directly. You can paste VCF lines
and ask Claude to construct the variants list.

## Python API

```python
from chorus.analysis import score_variant_batch

# From a VCF file:
import pandas as pd
vcf = pd.read_csv("variants.vcf", sep="\t", comment="#", header=None,
                   names=["CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO"])
variants = [{"chrom": r.CHROM, "pos": r.POS, "ref": r.REF, "alt": r.ALT, "id": r.ID}
            for _, r in vcf.iterrows()]

# Or manually:
variants = [
    {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
    {"chrom": "chr1", "pos": 109275684, "ref": "C", "alt": "T", "id": "rs629301"},
    # ...
]

# For AlphaGenome, use full track IDs (find them with oracle.search_tracks('HepG2 DNase'))
result = score_variant_batch(oracle, variants, assay_ids=["DNASE:HepG2"],
                             oracle_name="alphagenome")  # oracle_name enables percentile normalization

# Output formats
print(result.to_markdown())                 # Ranked table
result.to_tsv("batch_scores.tsv")           # TSV file for Excel/R/downstream
df = result.to_dataframe()                  # pandas DataFrame
d = result.to_dict()                        # JSON for pipelines
```

## Example output

### [example_output.md](example_output.md)

```
| Variant | ID | DNASE:HepG2 Ref | DNASE:HepG2 Alt | DNASE:HepG2 log2FC | DNASE:HepG2 Effect %ile | CHIP:CEBPA:HepG2 Ref | CHIP:CEBPA:HepG2 Alt | CHIP:CEBPA:HepG2 log2FC | CHIP:CEBPA:HepG2 Effect %ile | …
|---------|-----|------------------|------------------|---------------------|--------------------------|-----------------------|-----------------------|--------------------------|-------------------------------|---
| chr1:109274968 G>T | rs12740374 | 512 | 699 | +0.449 | ≥99th | 2.1e+03 | 2.73e+03 | +0.377 | ≥99th | … |
| chr1:109279175 G>A | rs4970836 | 8.21 | 7.96 | -0.039 | ≥99th | 201 | 198 | -0.027 | ≥99th | … |
| chr1:109275216 T>C | rs660240 | 397 | 417 | +0.071 | ≥99th | 1.28e+03 | 1.3e+03 | +0.028 | ≥99th | … |
| chr1:109275684 G>T | rs1626484 | 69.5 | 69.5 | +0.000 | — | 540 | 541 | +0.003 | 0.71 | … |
| chr1:109274570 A>G | rs7528419 | 118 | 119 | +0.008 | ≥99th | 954 | 960 | +0.008 | ≥99th | … |
```

The full table shows **4 columns per track** (Ref, Alt, log2FC, Effect %ile)
so you can see both absolute allele predictions and the derived effect at a glance.

**Columns per track**:
- **Ref**: Reference-allele prediction (raw signal)
- **Alt**: Alternate-allele prediction (raw signal)
- **log2FC**: Log2 fold-change alt/ref (the effect size)
- **Effect %ile**: log2FC magnitude ranked against ~10K random SNPs (`≥99th` for the saturated top bucket)
  — high values mean the variant is in an already-active regulatory region

**Interpretation**: rs12740374 is the clear top hit — 97th percentile
effect on a region that is already in the 91st percentile of genome-wide
chromatin accessibility, consistent with disrupting an active enhancer.
rs629301 shows a moderate TSS effect. The remaining variants have smaller
effects consistent with being in linkage disequilibrium but not causal.

### Output files

- `example_output.md` — ranked markdown table
- `example_output.tsv` — tab-separated values (for Excel, R, command-line)
- `example_output.json` — structured JSON with per-layer scores

## Understanding the scores

**Max Effect** is the largest |log2FC| (or |lnFC|, |Δ|) across all tracks
and layers for each variant. It represents the single strongest regulatory
effect that variant produces.

**Top Layer** tells you *what kind* of regulation is affected (chromatin
opening, TF binding disruption, histone mark change, TSS activity, etc.).

**Per-layer scores** (in the TSV and JSON) show the maximum effect within
each regulatory layer — useful for understanding the mechanism.

See the [variant analysis README](../variant_analysis/) for the full score
interpretation guide including magnitude thresholds.

## Oracle compatibility

Batch scoring works with any oracle. The layers scored depend on the tracks
you provide:

- **AlphaGenome** + DNASE + ChIP-TF + H3K27ac + CAGE + RNA: all 5+ layers scored
- **Enformer** + DNASE + CAGE: chromatin + TSS layers only (no RNA)
- **Borzoi** + DNASE + CAGE + RNA: chromatin + TSS + expression
- **ChromBPNet** (ATAC or DNASE): chromatin accessibility only — fast, single-layer

For clinical triage, using AlphaGenome with at least DNASE + CAGE + H3K27ac
gives a good balance of coverage and speed.

### Using ChromBPNet for fast batch screening

ChromBPNet is fast (~1s per variant vs ~30s for AlphaGenome) and works well
for initial screening of large variant sets. Load once with
`load_oracle('chrombpnet', assay='ATAC', cell_type='K562')`, then batch
score to get chromatin accessibility effects. Follow up on top hits with
AlphaGenome for full multi-layer analysis.

### Two-stage workflow

> Stage 1: Load ChromBPNet (ATAC, K562). Score all 50 variants from my VCF
> with score_variant_batch. Return the top 10 by effect.
>
> Stage 2: Load AlphaGenome. Re-score just those top 10 variants with
> DNASE, CEBPA ChIP, H3K27ac, CAGE, and RNA tracks for full multi-layer
> characterization.
