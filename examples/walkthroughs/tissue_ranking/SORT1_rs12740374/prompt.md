Investigate the variant rs12740374, hg38 chr1:109274968, G>T, where T is the ALT (minor) allele and the
gene of interest is SORT1. You do not know in advance which tissue or cell type the variant acts in: both
must come out of the analysis below. Use the chorus MCP tools for every analysis step (no custom code).
Rules: every number you report must come from a tool reply or from a table a tool wrote; if a tool fails, stop, show the exact error and say what
you would change instead of skipping the step; forward this whole message as user_prompt wherever a tool
accepts it. Some calls run for more than an hour and are silent while they run - wait for them, do not retry
them.

1. DISCOVERY - one discover_variant call per oracle (one pass covers all tracks; never loop per track):
   position="chr1:109274968", ref_allele="G", alt_alleles=["T"], gene_name="SORT1",
   variant_label="SORT1 rs12740374". Keep every reply's track_scores_csv and, from alphagenome, also
   rna_gene_percentiles_csv.
   a. load_oracle alphagenome, then discover_variant (histone marks, CAGE, RNA-seq).
   b. load_oracle cherimoya with assay="ATAC", cell_type="HepG2", encode_id="ENCSR042AWH" (this model is only
      the entry point - discovery then iterates one experiment per biosample), then discover_variant
      (DNase / ATAC).
   c. load_oracle chrombpnet with assay="CHIP", TF="CEBPB", cell_type="HepG2" (again only the entry point -
      discovery then iterates all ATAC / DNase models and all BPNet ChIP-TF models), then discover_variant
      (ChIP-TF).

2. RANKING - rank_variant_tissues with track_scores_csv = [the alphagenome, cherimoya and chrombpnet
   track_scores_csv, and the alphagenome rna_gene_percentiles_csv], method="acat", rna_gene="SORT1",
   top_n=15, variant_label="SORT1 rs12740374", default layer_oracles. Run it with group_level="tissue" and
   again with group_level="system". Show the top 15 tissues (and the top 5 systems) with their rank in every
   layer, the oracle that supplied each layer, the reply's notes and the figure paths.
   CHOICE RULE: the target tissue is the one with Global rank 1 in the TISSUE-level ACAT ranking. State it,
   spelled exactly as in the ranking table, and use it in step 3. Do not assume a tissue.

3. EXPRESSION - plot_variant_gene_response(rna_gene_percentiles_csv=<the alphagenome one>,
   tissue=<the target tissue from step 2>, variant_label="SORT1 rs12740374"). Report each named gene: median
   log2FC, tracks where it is an outlier, tracks above the noise floor, and the reply's notes. The reply also
   names the by-track table (files.by_track_csv) that lists the cell types of the target tissue that have
   RNA-seq tracks.

4. MOLECULAR LAYERS in the target cell type - A) chromatin accessibility B) TSS activity C) histone marks
   D) TF binding, plus RNA-seq.
   CHOICE RULE (target cell type). In an interactive session this is the user's call, made from the rankings
   above; for an unattended replay, apply the rule below. The candidates are the cell types listed in the
   step-3 by-track table (read
   the file, no code). Prefer a cell type for which ALL layers are covered: data in A (DNase or ATAC), B (CAGE
   or PRO-CAP), C (any histone mark), D (any ChIP-TF) and RNA-seq, counted over the alphagenome, cherimoya and
   chrombpnet track_scores_csv tables and the alphagenome RNA table. If exactly one candidate covers all layers,
   take it. If several do (a tie), take the one with the largest absolute effect: the largest |raw_score|
   (log2FC) over its layer A-D tracks in those tables. If none covers all layers, take the one covering the
   most layers and say so. Show a table of the candidates (layers covered, strongest |log2FC|) and state the
   winner. Do not assume a cell type, and do not assume its ontology id: AlphaGenome track ids differ per cell
   type (EFO:..., CL:..., UBERON:...).
   Then find the cell type's AlphaGenome tracks with
   list_tracks(oracle_name="alphagenome", query=<the cell type name>) and keep the identifiers whose cell type
   is exactly the target (every DNase, ATAC, CAGE / PRO-CAP, ChIP-histone, ChIP-TF and RNA-seq track). If a
   layer has no track for this cell type, say "not covered" for that layer; never substitute another cell
   type. If there are more than 10 ChIP-TF tracks, keep the 6 with the largest absolute effect in the
   alphagenome track_scores_csv. Then analyze_variant_multilayer on alphagenome with position="chr1:109274968",
   ref_allele="G", alt_alleles=["T"], gene_name="SORT1" and those assay_ids. Summarise each layer with ref, alt,
   effect and percentile from the reply. NOTE: the report is written to
   <output dir>/chr1_109274968_G_T_SORT1_alphagenome_report.html, the same name discovery used for AlphaGenome,
   so it replaces that file.

5. LOCUS REPORT for the AlphaGenome window chr1:108750680-109799255, for the same target cell type. All
   predictions must be new, with region="chr1:108750680-109799255", position="chr1:109274968",
   ref_allele="G", alt_alleles=["T"]. A loaded oracle keeps its model: load_oracle returns "already_loaded"
   without switching, so call unload_oracle before loading the next model. Use the oracle that step 2 assigned
   to each layer, and for the target cell type only (omit and say so when a layer has no track):
   - alphagenome (histone, CAGE, RNA-seq): one predict_variant_effect with the ids from step 4 for those
     layers (both strands of CAGE and RNA-seq).
   - cherimoya (DNase / ATAC): list_tracks(oracle_name="cherimoya", query=<cell type name>) gives the ENCODE
     experiment accession; load_oracle cherimoya with that encode_id (plus its assay and cell_type), then
     predict_variant_effect with assay_ids=["<assay>:<accession>"].
   - chrombpnet (BPNet ChIP-TF): list_tracks(oracle_name="chrombpnet", query=<cell type name>) lists the BPNet
     models for the cell type; take up to 4 factors, those with the largest absolute effect in the chrombpnet
     track_scores_csv. For each: unload_oracle chrombpnet, load_oracle chrombpnet assay="CHIP", TF=<factor>,
     cell_type=<cell type>, then predict_variant_effect with assay_ids=["CHIP:<cell type>:<factor>"] (it
     returns the "+" and "-" strand tracks).
   Then one build_locus_report(position="chr1:109274968", ref_allele="G", alt_allele="T",
   region="chr1:108750680-109799255", screenshot=true, title="SORT1 rs12740374 - <cell type> locus view") with
   the tracks in this order, each {"oracle", "assay_id" | "assay_ids", "name"}: Cherimoya chromatin
   (aggregate "max"); AlphaGenome histone marks; AlphaGenome CAGE (both strands, combine "sum", aggregate
   "sum"); AlphaGenome RNA-seq (both strands, combine "sum"); BPNet factors (both strands, combine "sum",
   aggregate "max").

FINISH with one summary: the file path of every output (ranking figures, gene-response figure, multilayer
report, locus HTML and PNG), the top-ranking tables, the cell-type ranking from step 3, the per-layer effects
in the target cell type, and what was not covered or failed.
