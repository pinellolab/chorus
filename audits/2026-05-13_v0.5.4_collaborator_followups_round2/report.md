# v0.5.4 audit — collaborator-variants round 2

**Date**: 2026-05-13 (verified against chorus v0.5.3 main + survey on 2026-05-12)
**Branch**: `fix/v0.5.4-collaborator-followups-round2`
**Tracking issue for item 5 (deferred)**: #83

## Context

A second collaborator (Dmitry Penzar) reproduced an earlier walkthrough (CDYL) and got matching results, then surfaced five issues. Every claim was verified directly against the code before edit. Item 5 (AG normalization over-optimism) is real and substantial — it has been deferred to issue #83 with the full diagnostic + three implementation options for whoever picks it up.

## Verified findings → fix

| # | Claim | Verified | Fix shipped in v0.5.4 |
|---|---|---|---|
| **1** | rsID not accepted by `analyze_variant_multilayer` / `discover_variant` / `discover_variant_cell_types` | Partial — only `fine_map_causal_variant` supported it | Added `ldlink_token` + `genome_build` params and an rsID-prefix branch that calls `fetch_ld_variants(r2_threshold=1.1)` (sentinel-only) to resolve coords. `score_variant_batch` intentionally skipped — its variant-dict contract already needs explicit coords (VCF case) |
| **2** | AG `assay_ids` in 4 walkthrough READMEs use display-name format ("DNASE:HepG2") which raises `ValueError("Assay ID not found in metadata")` at runtime | Verified | Replaced with real identifiers from `metadata.search_tracks()`. Added inline comment showing the lookup. Affected files: `batch_scoring/README.md`, `causal_prioritization/README.md`, `sequence_engineering/README.md`, `variant_analysis/SORT1_rs12740374/multilayer_variant_analysis.md` |
| **3** | LDlink request timeout (30 s default) hidden from MCP/CLI | Verified | Added `ldlink_timeout: float = 30.0` to `fine_map_causal_variant` MCP signature; threads to `fetch_ld_variants` |
| **4** | hg38 hardcoded in LDlink calls (`ld.py:101`); also defaulted-but-configurable in `_igv_report.py` and `causal.py` | Partial | Added `genome_build: str = "grch38"` to `fetch_ld_variants` with hg19/grch37/GRCh37/hg38/grch38/GRCh38 aliases; exposed through `fine_map_causal_variant`, `analyze_variant_multilayer`, `discover_variant`, `discover_variant_cell_types` |
| **5** | AG normalization gives over-optimistic percentiles (even tiny effects → >99th) | **Verified — diagnostic in issue #83** | **Deferred** to issue #83 with the full per-layer percentile-vs-effect-magnitude survey + three implementation options |

## Item 5 diagnosis (full data lives in issue #83)

Direct CDF inspection on the on-disk `~/.chorus/backgrounds/alphagenome_pertrack.npz`:

```
layer          mean pct at:  0.01    0.05    0.1     0.5     1.0
ATAC                         0.351   0.896   0.972   0.997   0.999
DNASE                        0.377   0.884   0.963   0.999   1.000
CHIP_HISTONE                 0.211   0.720   0.906   0.995   0.998
CHIP_TF                      0.610   0.894   0.961   0.994   0.998
RNA_SEQ                      0.999   1.000   1.000   1.000   1.000
```

Root cause: `scripts/build_backgrounds_alphagenome.py:395-410` samples random genomic positions + random alt alleles. Most random positions hit inactive chromatin → log₂FC clusters near zero → the percentile reflects rarity-vs-noise, not biological importance.

There IS an existing magnitude-based interpretation cap in `variant_report.py::_interpret_score`, but the percentile column and the interpretation column show side-by-side in the report and disagree. The cleanest fix is to suppress percentile below the per-layer magnitude floor (option B in the issue), with option C (rebuild CDF from functional positions) as a follow-up project.

## Files changed

```
chorus/__init__.py                                                  version bump
chorus/mcp/server.py                                                #1, #3, #4
chorus/utils/ld.py                                                  #3, #4
examples/walkthroughs/batch_scoring/README.md                       #2
examples/walkthroughs/causal_prioritization/README.md               #2
examples/walkthroughs/sequence_engineering/README.md                #2
examples/walkthroughs/variant_analysis/SORT1_rs12740374/multilayer_variant_analysis.md  #2
setup.py                                                            version bump
```

## Verification

- ✅ Smoke checks: `ldlink_token` + `genome_build` present on all 4 MCP signatures; `_resolve_rsid_to_position` raises a clear `ValueError` when no LDlink token is available.
- ✅ Bogus `genome_build` raises `ValueError` with the list of valid aliases.
- Pytest log: `pytest_log.txt` next to this report.

## Out of scope

- `score_variant_batch` rsID resolution — its variant-dict contract takes `chrom`/`pos`/`ref`/`alt` per row (VCF use case). Adding rsID resolution there would slow down batch jobs and require N LDlink round-trips. Use `fine_map_causal_variant` (which accepts a single rsID) or pre-resolve variants in Python before calling `score_variant_batch`.
- IGV/causal report `genome` parameter — already configurable via the existing `genome: str = "hg38"` kwarg at `_igv_report.py:327`, just undocumented. README note recommended in a future docs pass.
- Item 5 — issue #83.
