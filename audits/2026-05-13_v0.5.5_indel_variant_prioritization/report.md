# v0.5.5 — Indel + multi-allelic variant prioritization

**Date**: 2026-05-13
**Branch**: `fix/v0.5.5-indel-variant-prioritization`
**Trigger**: collaborator report on chr6 GWAS workflow where indel and
multi-allelic LD proxies were silently dropped or rejected.

## Context

`predict_variant_effect` hard-rejected any allele with `len != 1` at
`chorus/core/base.py:415-424` (`InvalidRegionError("…single-nucleotide
variant…")`). The downstream machinery — `apply_variant`,
`Interval.replace`, and `parse_ld_response` — already supported indels
in principle: they just never saw them because the gate fired first.

The collaborator's chr6 examples that v0.5.4 silently dropped:

```
rs869216555   chr6:4570564   (GAAAAAATAAAAA/-)   13-bp deletion
rs1418831098  chr6:4573808   (A/-)               1-bp deletion
rs150831380   chr6:4594841   (-/CT)              2-bp insertion
.             chr6:4581552   (CT/-)              2-bp deletion
```

Plus multi-allelic rows with `Correlated_Alleles="T=CT,A=-"` (two
sentinel↔proxy mappings per row, of which only the first was parsed).

## What v0.5.5 ships

| Surface | Change |
|---|---|
| `chorus/utils/sequence.py` (new) | `normalize_allele(a)` accepts LDlink `"-"` and `None`, uppercases, validates ACGTN. `classify_variant(ref, alt)` → `snv` / `insertion` / `deletion` / `mnv` / `complex` |
| `chorus/core/base.py` `predict_variant_effect` | SNV-only validator replaced. Now accepts any `(ref, alt)` over normalized ACGTN bases, including empty strings. Auto-widens `region_interval` if user's region is narrower than `len(ref)`. Multi-base ref comparison + multi-base replace |
| `chorus/utils/sequence.py` `get_centered_window` | Equal-length restriction lifted. For deletions, pulls extra flanking on the right to keep `len(alt_seq) == length`. For insertions, trims the right flank. Variant anchored at `length // 2` in both returned sequences |
| `chorus/utils/ld.py` `parse_ld_response` | New `_extract_allele_pairs` helper. Priority: (1) `Correlated_Alleles` `SENT=PROXY,SENT=PROXY` → one LDVariant per pair with `(ref=SENT, alt=PROXY)`; (2) comma-split `Alleles` `(REF/ALT1,ALT2)` → one LDVariant per alt; (3) single `(REF/ALT)`. All alleles routed through `normalize_allele`. New `LDVariant.kind` field |
| `chorus/utils/ld.py` `fetch_ld_variants` | New `snvs_only: bool = False` filter |
| `chorus/analysis/causal.py` `prioritize_causal_variants` | New `snvs_only: bool = False` filter. `CausalVariantScore.kind` field surfaced in `to_dict` + markdown report `[(deletion)]` style suffix |
| `chorus/mcp/server.py` `fine_map_causal_variant` | New `snvs_only: bool = False` parameter, threaded through to both `fetch_ld_variants` and `prioritize_causal_variants`. Docstring notes that indels are now scored by default |
| Tests | `test_indel_rejected_before_model_run` → `test_indel_accepted_and_scored` (4 indel cases). New `test_predict_variant_effect_rejects_garbage_allele` keeps the validator honest. New: `test_normalize_allele_table`, `test_classify_variant`, `test_get_centered_window_indels`, `test_parse_ld_response_dash_indel`, `test_parse_ld_response_multi_allelic_alt`, `test_parse_ld_response_correlated_alleles_fanout`, `test_fetch_ld_variants_snvs_only_filter`, `test_prioritize_causal_variants_indels`, `test_prioritize_causal_variants_snvs_only` |

## Scope decisions (per user clarification on 2026-05-13)

- **Correlated_Alleles fanout = two variants per row.** Each
  `SENT=PROXY` pair is treated as an independent mutation to score
  (so a row with `T=CT,A=-` emits both `(T → CT)` and `(A → "")`).
  Fallback to `Alleles` column when `Correlated_Alleles` is `NA` /
  missing.
- **`-` accepted everywhere.** `normalize_allele` runs at every
  entry point. Users can pass `"-"` as ref or alt to any public API.
- **`snvs_only=False` is the default.** The new behaviour ships
  enabled. Callers reproducing the pre-v0.5.5 SNV-only flow should
  pass `snvs_only=True`.

## Backward-compat note

This release relaxes a long-standing validation contract. Callers
that depended on `InvalidRegionError("…single-nucleotide variant…")`
to filter out indels will now have those variants pass through and be
scored. Migration: pass `snvs_only=True` to `fine_map_causal_variant`
/ `prioritize_causal_variants` / `fetch_ld_variants` to restore the
old behaviour.

The `test_indel_rejected_before_model_run` test is flipped to
`test_indel_accepted_and_scored`. Any downstream code that imports
chorus internals and asserts on rejection will need updating.

## Verification

- ✅ Targeted unit tests: 9 new + 5 existing all pass.
- ✅ Smoke checks:
  - `normalize_allele('-') == ''`, rejects `'X'`, `'AT?'`
  - `classify_variant('A','')` → `'deletion'`; `('A','AT')` → `'complex'`
  - `get_centered_window` returns `length`-bp ref + alt for SNV, insertion (`-/AT`), deletion (`G/-`), MNV, anchored insertion (`G/GAT`)
  - `parse_ld_response` fans out `Correlated_Alleles="T=CT,A=-"` into two records
  - `prioritize_causal_variants(snvs_only=True)` drops indels before scoring
- Full base regression: see `pytest_log.txt` next to this report.

## Out of scope

- **Large structural variants (>50 bp).** AG's 1 MB input window
  fits them but scoring quality is uncalibrated; the per-track CDFs
  were built from SNVs.
- **VCF normalisation / allele left-shifting.** Chorus accepts
  whatever the caller passes verbatim. Users wanting `bcftools norm`
  semantics should run that first.
- **`score_variant_batch` indel test.** Its variant-dict contract
  already accepts arbitrary `(ref, alt)`; once `predict_variant_effect`
  accepts indels (this release), it flows through. Add a regression
  test in a follow-up release.
- **Track-output array alignment around the indel.** A 13-bp
  deletion shifts genomic positions downstream by 13 bp in the alt
  sequence; the model's bin-i output for the alt corresponds to a
  shifted genome position. Windowed-sum scoring is robust to small
  shifts, but exact-position track-rendering inside the indel is
  off by `delta` bp. Acknowledged limitation; documenting only.
