# Tissue Ranking Examples

Start from a variant and *no* assumption about the tissue: score every track of several oracles, rank
tissues and body systems layer by layer, then follow the winner into its genes, cell type and locus.

> Ask Claude in plain language about a variant whose relevant tissue you do not know. The walkthrough below
> is one concrete, MCP-only demonstration; its [`prompt.md`](SORT1_rs12740374/prompt.md) is written so the whole
> analysis can be replayed from a single message.

## Walkthroughs

| Directory | Variant | What it shows |
|---|---|---|
| [`SORT1_rs12740374/`](SORT1_rs12740374/) | rs12740374 (chr1:109274968 G>T) | AlphaGenome + Cherimoya + BPNet discovery, ACAT tissue / system ranking (Liver is rank 1), liver gene response, HepG2 layers, 17-track locus report |

## Tools involved

`discover_variant` (per oracle) -> `rank_variant_tissues` -> `plot_variant_gene_response` ->
`analyze_variant_multilayer` -> `predict_variant_effect` + `build_locus_report`.
