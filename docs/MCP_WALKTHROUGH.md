# Using Chorus with Claude: MCP Walkthrough

This guide walks through a real Chorus + Claude Code session, showing
what a user types in natural language and what Claude returns.

## Prerequisites

1. Chorus installed + at least one oracle set up:
   ```bash
   mamba activate chorus
   chorus setup --oracle alphagenome   # or enformer, etc.
   ```

2. MCP configured (one-time — see [main README](../README.md#mcp-server-ai-assistant-integration)):
   ```bash
   claude mcp add chorus -- mamba run -n chorus chorus-mcp
   ```

3. Claude Code launched:
   ```bash
   claude     # from any project folder
   ```

---

## Example 1: Analyse a known variant

**You say:**

> Load AlphaGenome and analyze rs12740374 at chr1:109274968 G>T. Gene is
> SORT1. I want to know whether this variant changes chromatin, TF binding,
> and SORT1 expression in HepG2 liver cells.

**What Claude does:**

1. Calls `load_oracle("alphagenome")` — loads model weights (~30 s on GPU).
2. Calls `discover_variant(oracle_name="alphagenome", position="chr1:109274968",
   ref_allele="G", alt_alleles=["T"], gene_name="SORT1", user_prompt="Load AlphaGenome...")`
   — scores all 5,731 tracks, ranks by effect, builds a multi-layer report.
3. Returns a markdown summary (inline) + saves an HTML report with an
   embedded IGV genome browser to the working directory.

**What you get back (summary):**

- **Chromatin**: very strong DNASE opening (+1.9 log2FC) in liver-adjacent
  cell types
- **TF binding**: RXRA, SP1, HNF4A binding gain in liver tracks
- **Histone marks**: H3K27ac and H3K4me3 gain
- **CAGE / TSS**: SORT1 TSS activity increases
- All four layers converge on the same direction — classic enhancer
  activation.

The HTML report opens in your browser and shows each track with the
reference (grey) and alternate (blue) signal overlaid on an IGV browser,
so you can visually confirm the prediction.

---

## Example 2: Discover cell types (no prior hypothesis)

**You say:**

> I have a GWAS hit at chr16:53767042 T>C near FTO. I don't know which
> tissue is relevant. Screen all cell types and show me where the variant
> has the biggest regulatory effect.

**What Claude does:**

1. Calls `discover_variant_cell_types(oracle_name="alphagenome",
   position="chr16:53767042", ref_allele="T", alt_alleles=["C"],
   gene_name="FTO")` — screens ~472 cell types for DNASE/ATAC effects,
   then runs full multi-layer analysis on the top 5 cell types.
2. Returns a ranked cell-type list + one report per top cell type.

---

## Example 3: Fine-map a GWAS locus

**You say:**

> I'm fine-mapping the SORT1 LDL locus. Lead SNP is rs12740374. Can you
> score all the LD variants and tell me which one is most likely causal?

**What Claude does:**

1. Parses the rsID and fetches LD proxies (or uses your list).
2. Calls `fine_map_causal_variant(oracle_name="alphagenome",
   lead_variant="rs12740374", gene_name="SORT1")` — scores each LD
   variant across all regulatory layers, computes a composite causal
   score weighting effect size, number of affected layers, directional
   convergence, and baseline activity.
3. Returns a ranked table. Variant with composite >> 0.7 and
   convergence = 1.0 is your prime functional candidate.

---

## Example 4: Score a VCF batch

**You say:**

> Here are 5 SNPs from my credible set. Score them and rank by effect:
>
> chr1:109274968 G>T (rs12740374)
> chr1:109275684 G>T (rs1626484)
> chr1:109275216 T>C (rs660240)
> chr1:109279175 G>A (rs4970836)
> chr1:109274570 A>G (rs7528419)

Claude parses the free-text list, calls `score_variant_batch`, and
returns a ranked table.

---

## Example 5: Predict a sequence edit

**You say:**

> Simulate inserting a CMV promoter at chr19:55115000 in K562. What
> happens to local chromatin and gene expression?

Claude calls `simulate_integration` and returns effects across the
regulatory layers around the insertion site.

---

## Report format

Every analysis tool produces outputs in **four formats**:

| Format | How to get it | Best for |
|--------|--------------|----------|
| Markdown | Inline in Claude's response | Quick read |
| JSON | `example_output.json` | Programmatic analysis |
| TSV | `example_output.tsv` | Excel / R / command-line |
| HTML | Saved to working directory | Sharing with collaborators |

## The Analysis Request header

Every report carries an **Analysis Request** block at the top:

```
## Analysis Request

> Load AlphaGenome and analyze rs12740374 ...

- **Tool**: `discover_variant`
- **Oracle**: alphagenome
- **Normalizer**: per-track background CDFs
- **Generated**: 2026-04-12 01:30 UTC
```

This is generated automatically by Claude forwarding your original prompt
into the tool call. When you (or a colleague) open the HTML a month
later, you can immediately see what was asked, which oracle and normalizer
produced the numbers, and when it was generated.

## Tips

- **Natural language works.** You don't need to memorise tool names or
  parameters. Describe what you want in plain English and Claude picks the
  right tool.
- **AlphaGenome is recommended** for most users: 1 Mb window, 5,731
  tracks, single base-pair resolution, covers all regulatory layers.
- **ChromBPNet** is useful as a second opinion at base resolution for
  specific TF binding questions.
- **Start broad, then narrow.** Use `discover_variant` or
  `discover_variant_cell_types` first, then follow up with
  `analyze_variant_multilayer` on the top cell types for a focused report.
- **Batch first, then detail.** For a list of >5 variants, use
  `score_variant_batch` to triage, then run full multi-layer analysis on
  only the top 1–3 hits.
- **Manage loaded oracles.** Use `oracle_status` to see which oracles are
  currently in memory, and `unload_oracle('<name>')` to free GPU/RAM when
  you're done with one. Oracles are cached across tool calls, so you only
  pay the load cost once per session.
