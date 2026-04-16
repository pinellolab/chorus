# Chorus Full Audit v5

**Date**: 2026-04-16
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `chorus-applications` @ `df7d613`
**Auditor**: Claude Opus 4.7 (1M context)
**Audit branch**: `audit/2026-04-16-v5`

## Scope

Post-merge audit of the chorus-applications branch after the 4th-pass
persona audit landed. Changes since v4 audit (2026-04-16 `fc242bc`):

- `01d8446` Expand batch scoring to show ref/alt/log2FC/percentile per track
- `27b430b` Regenerate all example outputs with latest fixes
- `45d2303` Add missing READMEs + interpretation sections
- `f935bf5` Fourth-pass persona audit: glossary, interpretations, VCF example
- `df7d613` Fix user prompt in all HTML reports; remove stale/orphaned files

Scope: test suite, example inventory, HTML/Selenium audit, MD/JSON/TSV
consistency, normalization CDFs, notebook execution, biology spot-check.
No oracle env or cache was recreated (verified fresh in v4 yesterday).

## Executive summary

1. **BUG (MEDIUM)**: 2 pytest tests fail on `main` — `tests/test_analysis.py`
   references old batch_scoring column names (`_raw`, `_pctile`) but commit
   `01d8446` renamed the columns to `_ref`, `_alt`, `_log2fc`,
   `_effect_pctile`, `_activity_pctile`. 279 pass, 2 fail.
2. **BUG (MEDIUM)**: Hardcoded `"HepG2/K562 tracks"` string in the
   `Tracks requested` field of every non-discovery AlphaGenome example —
   surfaces in MD, JSON, and HTML reports. The actual tracks are often
   K562-only or HepG2-only, never the mix the label implies. Source:
   `scripts/regenerate_examples.py:182`.
3. **BUG (MINOR)**: Stale docstring in `chorus/analysis/batch_scoring.py:82`
   still describes the old `_raw` / `_pctile` column names.
4. All 12 application examples: READMEs present, JSON valid, TSV
   well-formed, MD contains Analysis Request + enriched CHIP display +
   `≥99th` / `≤1st` percentile format.
5. All HTML reports pass Selenium audit: analysis request visible, IGV
   loads, 0 SEVERE console errors, no CDN igv.js references, enriched
   CHIP track names and biological interpretation column present.
6. Normalization CDFs sane across 6 oracles (18,159 total tracks):
   monotone, p50 ≤ p95 ≤ p99, all `effect_counts > 0`, signed_flags
   correct per layer-type.
7. Biology spot-check on SORT1, TERT, BCL11A matches published
   literature direction and magnitude.

## Phase 1: Test suite

| Command | Result |
|---------|--------|
| `pytest tests/ --ignore=tests/test_smoke_predict.py` | **279 passed, 2 failed** |

### Failing tests (batch_scoring column rename drift)

Both failures are column-name drift. Commit `01d8446` renamed
`_raw` → `_log2fc`, added `_ref` and `_alt`, and renamed `_pctile` to
two columns `_effect_pctile` / `_activity_pctile`. The tests in
`tests/test_analysis.py` at lines 1432–1434 and 2354–2355 were not
updated.

**`TestEnrichedBatchFields::test_batch_to_dataframe_per_track`** (line 1432):
```python
assert "DNASE:K562_raw" in df.columns      # column is now DNASE:K562_log2fc
assert "DNASE:K562_pctile" in df.columns   # column is now DNASE:K562_effect_pctile
```

**`TestBatchDisplayModes::test_cage_strand_disambiguation_in_dataframe`**
(line 2354):
```python
assert "CAGE:HepG2 (+)_raw" in df.columns  # column is now CAGE:HepG2 (+)_log2fc
```

Proposed patch (follow-up PR):
- Update test assertions to match current column scheme
- Verify no other call sites rely on the legacy column names

## Phase 2: Example inventory

All 12 apps present with complete files:

| App | README | HTML | JSON valid | TSV rows |
|-----|--------|------|------------|----------|
| batch_scoring | ✓ | 1 | ✓ | 5 |
| causal_prioritization/SORT1_locus | ✓ | 1 | ✓ | 11 |
| discovery/SORT1_cell_type_screen | ✓ | 0 (by design) | ✓ | 19 |
| sequence_engineering/integration_simulation | ✓ | 1 | ✓ | 3 |
| sequence_engineering/region_swap | ✓ | 1 | ✓ | 4 |
| validation/SORT1_rs12740374_with_CEBP | ✓ | 2 | ✓ | 62 |
| validation/TERT_chr5_1295046 | ✓ | 1 | ✓ | 17 |
| variant_analysis/BCL11A_rs1427407 | ✓ | 1 | ✓ | 12 |
| variant_analysis/FTO_rs1421085 | ✓ | 1 | ✓ | 16 |
| variant_analysis/SORT1_chrombpnet | ✓ | 1 | ✓ | 1 |
| variant_analysis/SORT1_enformer | ✓ | 1 | ✓ | 20 |
| variant_analysis/SORT1_rs12740374 | ✓ | 1 | ✓ | 62 |

Total: **12 READMEs, 12 HTML reports across 11 dirs, 12 JSON + 12 MD + 12 TSV.**

## Phase 3: Selenium HTML audit

All 12 HTML reports opened in headless Chrome (1400×4000). Screenshots
captured at `audits/2026-04-16_v5_screenshots/`.

| HTML | Analysis request | IGV | CDN | SEVERE console | Screenshot |
|------|-----------------|-----|-----|----------------|------------|
| batch_scoring/batch_sort1_locus_scoring | ✓ | n/a (table) | none | 0 | ✓ |
| causal_prioritization/SORT1_locus/...causal_report | ✓ | loaded | none | 0 | ✓ |
| integration_simulation/integration_CMV_PPP1R12C_report | ✓ | loaded | none | 0 | ✓ |
| region_swap/region_swap_SORT1_K562_report | ✓ | loaded | none | 0 | ✓ |
| SORT1_CEBP/chr1_109274968_G_T_SORT1_enformer_RAW_autoscale | ✓ | loaded | none | 0 | ✓ |
| SORT1_CEBP/rs12740374_SORT1_CEBP_validation_report | ✓ | loaded | none | 0 | ✓ |
| TERT_chr5_1295046/chr5_1295046_T_G_TERT_alphagenome_report | ✓ | loaded | none | 0 | ✓ |
| BCL11A/rs1427407_BCL11A_alphagenome_report | ✓ | loaded | none | 0 | ✓ |
| FTO/rs1421085_FTO_alphagenome_report | ✓ | loaded | none | 0 | ✓ |
| SORT1_chrombpnet/rs12740374_SORT1_chrombpnet_report | ✓ | loaded | none | 0 | ✓ |
| SORT1_enformer/rs12740374_SORT1_enformer_report | ✓ | loaded | none | 0 | ✓ |
| SORT1_rs12740374/rs12740374_SORT1_alphagenome_report | ✓ | loaded | none | 0 | ✓ |

Additional spot-check on SORT1_rs12740374 with longer (15 s) render wait
confirms the following strings are present in the rendered DOM text:

- `CHIP:CEBPA:HepG2` (enriched CHIP display from `_track_description`)
- `DNASE:HepG2`, `CAGE:HepG2`, `H3K27ac`
- `≥99th` (the `_fmt_percentile` display for `q ≥ 0.99`)

**Result**: all 12 HTMLs clean. The v4 finding about ChromBPNet CDN
igv.js is **RESOLVED** — no CDN references detected anywhere.

## Phase 4: MD / JSON / TSV consistency

All 12 apps pass:

- TSV has matching column counts across all data rows (trailing empty
  line ignored)
- MD contains Analysis Request with user prompt
- 10/12 MDs use `≥99th` / ordinal percentile formatting (the 2 that
  don't — causal_prioritization and SORT1_chrombpnet — use a different
  percentile column layout by design)
- JSON parses, contains `variant`, `oracle`, `gene_name`, `alleles`
  top-level keys with per-layer track scores

## Phase 5: Normalization CDFs

All 6 oracles pass:

| Oracle | Tracks | Monotone | p50≤p95≤p99 | min count | signed / unsigned |
|--------|--------|----------|-------------|-----------|-------------------|
| alphagenome | 5168 | ✓ | ✓ | 1697 | 667 / 4501 |
| borzoi | 7611 | ✓ | ✓ | 6563 | 1543 / 6068 |
| chrombpnet | 24 | ✓ | ✓ | 9609 | 0 / 24 |
| enformer | 5313 | ✓ | ✓ | 9600 | 0 / 5313 |
| sei | 40 | ✓ | ✓ | 9609 | 40 / 0 |
| legnet | 3 | ✓ | ✓ | 9609 | 3 / 0 |

Signed-flag semantics match `LAYER_CONFIGS`:
- `CHIP`, `ATAC`, `DNASE`, `CAGE`, `CHIP_HISTONE` → unsigned
- `RNA_SEQ`, `PROCAP` (alphagenome signed), SEI classification, LegNet
  regression → signed

All `effect_counts > 0` (min 1697 on AlphaGenome, 6563 on Borzoi, 9609
elsewhere — well above the 100-sample threshold).

## Phase 6: Biology spot-check

### SORT1 rs12740374 (HepG2, AlphaGenome)

Committed `example_output.json` values (G→T allele change):

| Track | Ref | Alt | log2FC | Expected direction (Musunuru 2010) |
|-------|-----|-----|--------|-----------------------------------|
| DNASE:HepG2 | 512 | 699 | **+0.449** | ↑ (chromatin opens) ✓ |
| CEBPA:HepG2 | 2100 | 2725 | **+0.376** | ↑ (T creates C/EBP site) ✓ |
| CEBPB:HepG2 | 1216 | 1471 | **+0.274** | ↑ (paralog co-binding) ✓ |
| H3K27ac:HepG2 | 13700 | 15500 | **+0.178** | ↑ (active enhancer mark) ✓ |

All 4 primary tracks positive as expected. Reproduces the published
mechanism.

### TERT chr5:1295046 T>G (discovery mode, AlphaGenome)

All ChIP/CAGE signals show **positive** gain at TERT TSS:

| Track | log2FC | Interpretation |
|-------|--------|----------------|
| DNASE:GM12865 | +0.215 | Moderate opening |
| CHIP:E2F1:K562 | +0.463 | Strong binding gain |
| CHIP:H3K27ac:GM12878 | +0.303 | Strong mark gain |
| CAGE:GM12878 — TERT TSS | +0.345 | Strong increase |

Consistent with TERT promoter gain-of-function biology (Horn 2013 /
Huang 2013). The v3 audit plan's concern that the TERT example might
show a decrease is **not substantiated** — the direction is correct.

### BCL11A rs1427407 (K562 erythroid, AlphaGenome)

K562 shows modest effects (as the README explicitly documents —
primary erythroid progenitors would show stronger effects). All tracks
in `example_output.tsv` use K562 (EFO:0002067) consistently. Header
claim "HepG2/K562" is a stale label bug (see Finding #2), but the
underlying data is K562-only.

## Phase 7: Notebook execution

| Notebook | Cells | Errors | Stale "no baselines" msgs |
|----------|-------|--------|---------------------------|
| single_oracle_quickstart | 49 | 0 | 0 |
| comprehensive_oracle_showcase | 59 | 0 | 0 |
| advanced_multi_oracle_analysis | 127 | 0 | 0 |

All three notebooks executed end-to-end in the `chorus` kernel with no
errors and no stale "no baselines available" messages — consistent
with the `get_normalizer()` → `PerTrackNormalizer` path that was fixed
in v4.

## Proposed follow-up (separate PR)

### MEDIUM — fix batch_scoring test drift

`tests/test_analysis.py`:
- Line 1432–1434: replace `_raw`, `_pctile` with current columns.
- Line 2354–2355: same.

```python
# Update to match current batch_scoring.py output columns:
assert "DNASE:K562_log2fc" in df.columns
assert "DNASE:K562_effect_pctile" in df.columns
assert df["DNASE:K562_log2fc"].iloc[0] == 0.5
# ...
assert "CAGE:HepG2 (+)_log2fc" in df.columns
assert "CAGE:HepG2 (-)_log2fc" in df.columns
```

### MEDIUM — fix stale "HepG2/K562 tracks" label

`scripts/regenerate_examples.py:182`:

```python
# Before:
tracks_requested=f"{len(assay_ids)} HepG2/K562 tracks" if assay_ids else "all tracks",

# After — use the actual cell type from the example dict:
tracks_requested=(
    f"{len(assay_ids)} {example.get('cell_type', 'tracks')}"
    if assay_ids else "all tracks"
),
```

Each example dict already has enough info (e.g. `HEPG2_TRACKS` / `K562_TRACKS`
sets) to infer the cell type — or add a `"cell_type"` key to each example.

After patching, regenerate the 5 affected example outputs (SORT1,
SORT1_CEBP, FTO, BCL11A, and the HTML reports).

### MINOR — fix stale docstring

`chorus/analysis/batch_scoring.py:82`:

```python
# Before:
"""One row per variant. Per-track columns with ``_raw`` and ``_pctile`` suffixes."""

# After:
"""One row per variant. Per-track columns with ``_ref``, ``_alt``, ``_log2fc``,
``_effect_pctile``, and ``_activity_pctile`` suffixes."""
```

## Verdict

**PASS with 3 documentation/test drifts.** No regressions in production
behavior since v4. The 2 pytest failures and the hardcoded
"HepG2/K562 tracks" label are both consequences of the 4th-pass persona
audit landing without updating all dependent strings. None affect
end-user-facing predictions or normalization. All biology checks,
all HTML reports, all 3 notebooks, and all 6 CDF stacks remain
correct.
