# Chorus Release Checklist — New User Readiness

A step-by-step plan to validate Chorus from a fresh user's perspective,
ensure solid foundations, and showcase the MCP server's biological analysis capabilities.

---

## Phase 1: Installation & Foundation (Python Library)

### 1.1 Fresh install validation
- [ ] Clone repo, create conda environments following README instructions
- [ ] Run `chorus setup --oracle enformer` (and each oracle) — verify environment creation
- [ ] Run `chorus download-genome hg38` — verify reference genome download
- [ ] Run full test suite: `pytest tests/ -x -q` — **must see 119 passed, 0 failures**
- [ ] Verify console entry points: `chorus --help`, `chorus-mcp --help`

### 1.2 Notebook walkthrough — comprehensive_oracle_showcase.ipynb
This is the primary learning path for new users. Validate every cell runs:

- [ ] **Cell group 1: Setup** — imports, oracle creation, genome loading
- [ ] **Cell group 2: Enformer** — load model, predict at GATA1 locus, visualize tracks
- [ ] **Cell group 3: Borzoi** — load, predict same region at 32bp resolution
- [ ] **Cell group 4: ChromBPNet** — load ATAC K562, base-resolution prediction
- [ ] **Cell group 5: Sei** — sequence classification
- [ ] **Cell group 6: LegNet** — MPRA activity prediction
- [ ] **Cell group 7: AlphaGenome** — load, predict with 1Mb window, 5930 tracks
- [ ] **Cell group 8: Variant effect** — predict_variant_effect, effect sizes
- [ ] **Cell group 9: Sub-region scoring** — score_prediction_region, score_variant_effect_at_region
- [ ] **Cell group 10: Gene expression** — analyze_gene_expression, CAGE at TSS, RNA exon sum

**Documentation quality checks for the notebook:**
- [ ] Every cell has a markdown header explaining what it does and why
- [ ] Key biological concepts are explained (what is CAGE? what is H3K27ac?)
- [ ] Output is interpreted — not just "here's a number" but "this means..."
- [ ] New features (variant effect, gene expression, sub-region scoring) have clear examples
- [ ] AlphaGenome is positioned as the primary oracle (1Mb window, all track types)
- [ ] Mixed-resolution tracks are demonstrated (DNASE 1bp + histone 128bp in AlphaGenome)

### 1.3 API documentation review
- [ ] README.md covers: installation, setup, all 6 oracles, quick start, MCP server
- [ ] API_DOCUMENTATION.md covers: all public methods with signatures and examples
- [ ] METHOD_REFERENCE.md covers: predict, variant_effect, gene_expression, scoring
- [ ] All new methods documented: score_variant_effect(), analyze_variant_effect_on_gene()
- [ ] Return types documented — users know what dict keys to expect

### 1.4 Test coverage gaps to fill
- [ ] Add test for mixed-resolution at_variant scoring (AlphaGenome 1bp + 128bp)
- [ ] Add test for auto-region centering (_auto_region returns 1bp region)
- [ ] Add test for bedgraph filename sanitization (track IDs with `/` characters)
- [ ] Add test for TSS out-of-window warning in predict_variant_effect_on_gene
- [ ] Add test for ChromBPNet track key mismatch (assay_ids vs prediction keys)
- [ ] Verify: `pytest tests/ -x -q` still shows 119+ passed after new tests

---

## Phase 2: MCP Server Showcase (Biological Analysis)

### 2.1 MCP server basic validation
- [ ] Verify .mcp.json is correct and server starts: `mamba run -n chorus chorus-mcp`
- [ ] From Claude Code, verify all 9 MCP tools are available:
  - list_oracles, list_tracks, list_genomes
  - load_oracle, unload_oracle, oracle_status
  - predict, predict_variant_effect, predict_variant_effect_on_gene
  - score_prediction_region, score_variant_effect_at_region
  - get_genes_in_region, get_gene_tss

### 2.2 New user MCP walkthrough
Simulate a biologist who just installed Chorus and wants to analyze a GWAS variant:

**Step 1: Discovery**
- [ ] `list_oracles()` — see all 6 oracles with specs, install status, loaded status
- [ ] `list_tracks("alphagenome", query="hepatocyte")` — find liver tracks
- [ ] `list_tracks("enformer", query="HepG2")` — find HepG2 CHIP tracks
- [ ] `list_tracks("chrombpnet", query="K562")` — see ATAC/DNASE/CHIP combos
- [ ] Verify search returns usable track identifiers (not just type/cell lists)

**Step 2: Loading**
- [ ] `load_oracle("enformer")` — ~10s, verify status
- [ ] `load_oracle("alphagenome")` — ~80s, verify status
- [ ] `load_oracle("chrombpnet", assay="ATAC", cell_type="K562")` — verify TF/fold/model_type params
- [ ] `oracle_status()` — see all loaded oracles

**Step 3: Gene context**
- [ ] `get_genes_in_region("chr1", 109240000, 109400000)` — genes near SORT1 variant
- [ ] `get_gene_tss("SORT1")` — TSS positions
- [ ] `get_gene_tss("BCL11A")` — TSS positions

### 2.3 The 5-Layer Variant Analysis (rs12740374 / SORT1)

This is the showcase. Run through all 5 layers using the right cell type and oracle:

**Layer 1: Chromatin accessibility**
- [ ] AlphaGenome: `score_variant_effect_at_region` with `DNASE/CL:0000182 DNase-seq/.`
- [ ] Verify: positive effect (variant opens chromatin in hepatocytes)

**Layer 2: Regulatory element type**
- [ ] AlphaGenome: H3K27ac (`CHIP_HISTONE/CL:0000182 Histone ChIP-seq H3K27ac/.`)
- [ ] AlphaGenome: H3K4me1 (`CHIP_HISTONE/CL:0000182 Histone ChIP-seq H3K4me1/.`)
- [ ] AlphaGenome: H3K4me3 (`CHIP_HISTONE/CL:0000182 Histone ChIP-seq H3K4me3/.`)
- [ ] Verify: H3K27ac increases (active enhancer gain), H3K4me1 decreases (poised→active)
- [ ] Verify: H3K4me3 unchanged (this is an enhancer, not a promoter)
- [ ] Verify: mixed-resolution tracks (1bp + 128bp) all return non-null scores

**Layer 3: TF binding**
- [ ] Enformer: CEBPB ChIP HepG2 (`ENCFF003HJB`)
- [ ] Enformer: CEBPA ChIP HepG2 (`ENCFF559CVP`)
- [ ] Enformer: HNF4A ChIP HepG2 (`ENCFF080FZD`)
- [ ] Verify: C/EBP tracks show strongest increase (variant creates C/EBP binding site)
- [ ] Verify: HNF4A co-recruited (liver master TF)

**Layer 4: Gene expression**
- [ ] AlphaGenome: CAGE hepatocyte at variant site
- [ ] AlphaGenome: CAGE hepatocyte at SORT1 TSS (118kb away) — **only possible with 1Mb window**
- [ ] AlphaGenome: RNA-seq hepatocyte at SORT1 TSS
- [ ] `predict_variant_effect_on_gene("enformer", ..., gene_name="SORT1")` — verify TSS warning
- [ ] `predict_variant_effect_on_gene("enformer", ..., gene_name="CELSR2")` — verify fold change

**Layer 5: Cell-type specificity**
- [ ] Compare DNASE effect in hepatocyte (AlphaGenome) vs K562 (Enformer)
- [ ] Verify: strong effect in liver, weak/absent in K562

### 2.4 The Distal Enhancer Test (rs1421085 / FTO → IRX3)

This showcases why AlphaGenome's 1Mb window is essential:

- [ ] `score_variant_effect_at_region("alphagenome", ...)` at variant site — local effect
- [ ] Score at FTO TSS (66kb) — slight decrease
- [ ] Score at IRX3 TSS (520kb!) — **positive increase in IRX3 expression**
- [ ] Verify: this effect is invisible to Enformer (114kb window) and Borzoi (196kb)

### 2.5 ChromBPNet base-resolution test (rs1427407 / BCL11A)

- [ ] `load_oracle("chrombpnet", assay="ATAC", cell_type="K562")`
- [ ] `predict_variant_effect("chrombpnet", position="chr2:60490908", ...)`
- [ ] Verify: base-resolution accessibility change at the variant
- [ ] Verify: bedgraph files saved with clean filenames

### 2.6 Report generation
- [ ] Run `generate_biology_report.py` — produces per-variant PNG plots + markdown
- [ ] Verify: plots show gene annotations, ref/alt overlay, effect tracks
- [ ] Review `variant_analysis_framework.md` — oracle guide, 5-layer approach, worked examples

---

## Phase 3: Polish & Final Checks

### 3.1 Documentation updates
- [ ] README: verify MCP server section is current (TF/fold/model_type params, auto-centering)
- [ ] README: add AlphaGenome as "recommended primary oracle" in variant analysis section
- [ ] comprehensive_oracle_showcase.ipynb: add AlphaGenome variant analysis cells
- [ ] comprehensive_oracle_showcase.ipynb: add mixed-resolution example
- [ ] variant_analysis_framework.md: final review after all testing

### 3.2 Edge cases and robustness
- [ ] Multi-word search: `list_tracks("enformer", query="CAGE K562")` returns 0 — document workaround
- [ ] Very large variant effects: test with a known causal coding variant
- [ ] Multiple alternate alleles: test with tri-allelic variant
- [ ] No expression tracks warning: verify warning appears with DNASE-only assay_ids

### 3.3 Final test run
- [ ] `pytest tests/ -x -q` — 119+ passed, 0 failures
- [ ] Full notebook re-execution: comprehensive_oracle_showcase.ipynb runs clean
- [ ] MCP server restart + full walkthrough from fresh state
- [ ] `git status` — clean working tree, all changes committed

---

## Validation Criteria

The release is ready when:
1. **All checkboxes above are checked**
2. **Tests: 119+ passed, 0 failures**
3. **Notebook runs end-to-end without errors**
4. **MCP server handles all 3 variants across 3 oracles**
5. **The 5-layer analysis recapitulates known biology for rs12740374/SORT1**
6. **AlphaGenome's 1Mb window captures IRX3 at 520kb from rs1421085**
7. **Mixed-resolution tracks (1bp + 128bp) all score correctly**
8. **No null/zero scores where real signal is expected**
9. **TSS out-of-window warning fires correctly**
10. **A biologist reading the framework doc can run their own variant analysis**
