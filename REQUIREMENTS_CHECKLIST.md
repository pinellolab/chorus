# Application Output Requirements Checklist

Every single application example MUST satisfy ALL of these requirements.
Check each file individually — don't batch or assume.

## Universal requirements (ALL reports)

- [ ] **Original prompt**: HTML and MD reports MUST have the user's natural-language prompt at the top (Analysis Request block with quoted prompt text)
- [ ] **Track IDs traceable**: Every track shown in tables MUST include the exact oracle assay_id somewhere (in the column header, a footnote, or the JSON) so a user can trace back to the original prediction data
- [ ] **Raw + percentile scores**: Every scored track MUST show both the raw effect score AND the effect percentile (e.g., "+0.449 (100%)")
- [ ] **No empty columns**: TSV files must not have empty/None columns for scored tracks
- [ ] **Honest labels**: Interpretation labels must match raw magnitude (no "Very strong" on ±0.01 effects)
- [ ] **Report title**: Title must be specific to the analysis type (not generic "Multi-Layer Variant Effect Report" for swaps/insertions)

## IGV Browser requirements

- [ ] **IGV loads**: The igv.min.js library must be properly inlined (no CDN dependency that fails on HPC clusters)
- [ ] **Variant marker**: For SNPs, a red marker at the exact variant position
- [ ] **Region marker**: For region swaps, the red highlight spans the FULL replaced region
- [ ] **Insertion marker**: For insertions, the red highlight spans the full inserted construct length
- [ ] **Tracks visible**: IGV shows ref (grey) and alt (coloured) signal overlaid for every scored track

## Per-application requirements

### Batch scoring
- [ ] Per-track columns (one per assay:cell_type), not aggregate "max effect"
- [ ] Each cell: raw score + percentile
- [ ] Column headers include human-readable track description
- [ ] Track IDs (exact oracle identifiers) in the JSON `track_scores` section

### Causal prioritization
- [ ] Per-track columns for each scored track
- [ ] Each cell: raw score + percentile
- [ ] Composite score column
- [ ] The ranked TABLE must be present (not just track columns without the ranking)

### Discovery
- [ ] Master HTML that links to per-cell-type HTML reports
- [ ] Cell-type ranking table at the top
- [ ] TSV with no empty columns
- [ ] Per-cell-type IGV HTML reports linked from the master

### Variant analysis
- [ ] Specific biologically motivated tracks (not "all tracks")
- [ ] Clear cell type context (HepG2 for SORT1, K562 for BCL11A, etc.)
- [ ] Track descriptions include TF name / histone mark / cell type

### Sequence engineering (region swap + integration)
- [ ] Report title says "Region Swap Analysis Report" or "Integration Simulation Report"
- [ ] The INSERTED/REPLACED SEQUENCE is documented in the report (what was put in, how long, what it represents)
- [ ] IGV marker highlights the FULL modified region (not 2-3 bp)
- [ ] The report explains what region was modified and what the expected biological effect is

### Validation
- [ ] Uses the exact tracks the published paper used
- [ ] CEBPA/CEBPB visible for SORT1 validation
- [ ] Gene name correct (SORT1, not CELSR2)
