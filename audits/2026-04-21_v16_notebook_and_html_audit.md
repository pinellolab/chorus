# v16 notebook + HTML report walkthrough — 2026-04-21

Walked the three `examples/notebooks/*.ipynb` cell-by-cell (every cell's
source **and** execution output) and screenshotted eight of the shipped
`examples/walkthroughs/**/*.html` reports via headless Chrome. Goal: catch
anything a new user would see that is inconsistent, wrong, or confusing.

## Fixed in this PR

1. **Broken link in both multi-oracle notebooks.** Cell 0 of
   `comprehensive_oracle_showcase.ipynb` and
   `advanced_multi_oracle_analysis.ipynb` said
   `[`examples/walkthroughs/`](applications/)`. The label was correct
   but the URL still pointed at the old `applications/` directory (renamed
   to `walkthroughs/` in `340f30e`). Fixed to `../walkthroughs/`.
2. **AlphaGenome track count drift.** Markdown in
   `comprehensive_oracle_showcase.ipynb` claimed "5,930 tracks across 11
   modalities" in three places (overview table, section 6 preamble, and a
   code comment in Operation 8). The running notebook prints
   `Loaded 5731 AlphaGenome tracks` and `Assay types: 7`. Fixed to
   "5,731 tracks" / "7 assay types" everywhere, and added PRO-CAP to the
   assay list in the overview table so it matches what `list_assay_types()`
   actually returns (`ATAC, CAGE, CHIP, DNASE, PRO_CAP, RNA, SPLICE_SITES`).
3. **Enformer context-window typo.** `advanced_multi_oracle_analysis.ipynb`
   cell 9 called Enformer "a state-of-the-art sequence-to-activity model
   that can predict **5.313 human genomic tracks** with a context window
   of 196 kbp". The library reports `oracle.sequence_length == 393216` —
   matches the comprehensive notebook's summary table (393 kb input). Also
   fixed the decimal separator (`5.313` → `5,313`). Changed to "393 kbp".

## Not fixed here — flagged for a follow-up

**Off-by-one in `predict_variant_effect`'s ref-allele check.** Both
`single_oracle_quickstart.ipynb` (cell 39) and
`comprehensive_oracle_showcase.ipynb` (cell 35) ship an execution output
that contains

    WARNING - Provided reference allele 'C' does not match the genome at
    this position ('T'). Chorus will use the provided allele.

for the test variant at `chrX:48786129`. The notebook code pulls the ref
base via `extract_sequence('chrX:48786129-48786129')` which returns `'C'`
(matches every external reference — UCSC, dbSNP, hg38 FASTA). But inside
`chorus/core/base.py:322-328`:

    real_pos = region_interval.ref2query(var_pos, ref_global=True)
    if region_interval[real_pos].upper() != ref_allele.upper():
        logger.warning(...)

the internal check reads the **next** base (`'T'`) instead of `'C'`. Same
shift reproduces for the rs12740374 example: `chr1:109274968` is `G` by
every external reference (and by `extract_sequence`), but
`region_interval[ref2query(109274968, ref_global=True)]` returns `'T'`
(the base at 109274969).

Consequence of the bug: every variant analysis in the shipped walkthroughs
logs this warning even when the user provided the correct allele. More
seriously, the reference allele substitution at line 330 happens at the
shifted position, so the ref/alt predictions are computed at one base off
the user's intended coordinate. Predicted effect *directions* still come
out right (the walkthroughs all agree with Musunuru et al. 2010 for
rs12740374), which is probably why this has been shipping unnoticed — the
regulatory signal is coherent across ±1 bp in every case the library
tests on.

Not fixing in this PR because it's a code-behaviour change (not a doc
drift), and because any fix needs to be audited against what happens at
indels and reverse-strand variants, not just SNVs. Best handled as its own
focused PR.

## HTML reports screenshotted

Rendered at 1400×3000 via Chrome --headless and eye-reviewed. Nothing
report-level was wrong:

- `walkthroughs/validation/SORT1_rs12740374_multioracle/rs12740374_SORT1_multioracle_report.html`
- `walkthroughs/variant_analysis/SORT1_rs12740374/rs12740374_SORT1_alphagenome_report.html`
- `walkthroughs/causal_prioritization/SORT1_locus/rs12740374_SORT1_locus_causal_report.html`
- `walkthroughs/batch_scoring/batch_sort1_locus_scoring.html`
- `walkthroughs/discovery/SORT1_cell_type_screen/chr1_109274968_G_T_SORT1_alphagenome_LNCaP_clone_FGC_report.html`
- `walkthroughs/sequence_engineering/region_swap/region_swap_SORT1_K562_report.html`
- `walkthroughs/validation/SORT1_rs12740374_with_CEBP/rs12740374_SORT1_CEBP_validation_report.html`

All display the glossary, formula chips, cross-layer tables, and
(client-side JS) IGV browser placeholder correctly. The embedded IGV
itself doesn't render in headless file:// mode — not a bug, it's a
loader limitation for local previews.
