# Causal Variant Prioritization

Given a GWAS sentinel variant and its LD proxies, score all variants across
regulatory layers and identify the most likely causal variant using
`fine_map_causal_variant`.

> Natural language works — describe your locus, sentinel, and gene in
> plain English (or paste an LD-proxy list). The prompts below are
> concrete demonstrations; the SORT1 locus example shows how the
> ranked output looks and which per-layer scores to inspect.

## Example Prompts

### For a geneticist (GWAS fine-mapping)

> I have a GWAS hit at rs12740374 (chr1:109274968 G>T) for LDL cholesterol.
> There are several variants in LD at this locus. Can you score all of them
> and tell me which one is most likely to be the causal variant? Use
> AlphaGenome with HepG2 DNASE, H3K27ac, and CAGE tracks. Gene is SORT1.
>
> Here are the LD variants:
> - chr1:109275684 C>T rs629301 (r²=0.95)
> - chr1:109272715 A>G rs12037222 (r²=0.88)
> - chr1:109278590 G>A rs2228603 (r²=0.72)
> - chr1:109271200 T>C rs7528419 (r²=0.65)

### For a bioinformatician (auto LD lookup)

> Load alphagenome. Call fine_map_causal_variant with:
> - lead_variant: "rs12740374"
> - assay_ids: [DNASE/EFO:0001187 DNase-seq/., CAGE/hCAGE EFO:0001187/+,
>   CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.]
> - gene_name: SORT1
> - population: CEU
> - r2_threshold: 0.8
> - ldlink_token: [your token]
>
> This will auto-fetch LD variants from LDlink and score all of them.

### For an MD (clinical variant prioritization)

> A patient has a GWAS-associated locus with 5 variants in the same LD block.
> Which one is most likely to be the functional variant that affects gene
> regulation? Score them all and rank by regulatory evidence.

### For a computational biologist

> I'm fine-mapping a GWAS locus. Score my 95% credible set of variants
> and rank by a composite score that considers: (1) effect magnitude,
> (2) number of regulatory layers affected, (3) convergence of direction
> across layers, and (4) whether the variant sits in active chromatin.

## How it works

```
1. Provide sentinel variant + LD variants (or auto-fetch from LDlink)
         ↓
2. Score EACH variant through multi-layer analysis (chromatin, TF, histone, TSS, RNA)
         ↓
3. Compute 4 component scores per variant:
   - max_effect: strongest single-layer disruption
   - n_layers_affected: how many layers show an effect
   - convergence: do all layers agree in direction?
   - ref_activity: is the variant in active chromatin?
         ↓
4. Combine into composite causal score (weighted, min-max normalized)
         ↓
5. Rank and produce: locus plot + ranked table + per-variant detail cards
```

### Composite score columns

| Column | Weight | What it measures |
|--------|--------|-----------------|
| **Max Effect** | 35% | Largest |effect| across all tracks and layers |
| **Layers Affected** | 25% | Number of regulatory layers with |effect| > 0.1 |
| **Convergence** | 20% | Do affected layers agree in direction? (1.0 = all same) |
| **Ref Activity** | 20% | Mean reference signal at chromatin/histone tracks |

All columns are sortable in the HTML report. The default ranking uses the
composite score, but you can click any column header to re-sort.

## LD variant source

**Automatic** (recommended): Provide an rsID and LDlink token. The tool
queries the [LDlink LDproxy API](https://ldlink.nih.gov/LDlinkRest/) to
fetch all variants in LD. Register for a free token at
https://ldlink.nih.gov/?tab=apiaccess

**Manual**: Provide a list of variant dicts with `chrom`, `pos`, `ref`,
`alt`, and optionally `id` and `r2`.

## MCP tool call

```python
fine_map_causal_variant(
    oracle_name="alphagenome",
    lead_variant="rs12740374",         # or "chr1:109274968 G>T"
    ld_variants=[                       # optional: auto-fetch if omitted
        {"chrom": "chr1", "pos": 109275684, "ref": "C", "alt": "T",
         "id": "rs629301", "r2": 0.95},
        # ...more variants...
    ],
    assay_ids=["DNASE:HepG2", "CAGE:HepG2", "H3K27ac:HepG2"],
    gene_name="SORT1",
    population="CEU",
    r2_threshold=0.8,
    ldlink_token="your_token_here",     # needed for auto-fetch
)
```

## Example

### [SORT1_locus/](SORT1_locus/)
rs12740374 (sentinel) + 10 LD variants at the SORT1/CELSR2 locus.
The sentinel is correctly identified as the top candidate (composite=0.964)
with the strongest chromatin opening and convergent multi-layer effects.

Output includes:
- `rs12740374_SORT1_alphagenome_causal_report.html` — interactive report with locus plot
- `example_output.md` — ranked table
- `example_output.json` — structured results

## Oracle compatibility

Works with any oracle. For best results, use AlphaGenome or Borzoi with
multiple track types (DNASE + ChIP-TF + H3K27ac + CAGE + RNA) to maximize
the number of layers available for convergence scoring.

ChromBPNet can be used but only scores 1 layer (chromatin or TF), so
the convergence and n_layers components will be minimal.
