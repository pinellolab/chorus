# v0.5.3 audit — collaborator-variants follow-ups

**Date**: 2026-05-12
**Branch**: `fix/v0.5.3-collaborator-followups`
**Upstream report**: external worktree at `~/Projects/chorus.bak-2026-04-28/audits/2026-05-11_collaborator_variants/` (not in this tree)

## Context

A wet-lab collaborator ran an end-to-end multi-oracle variant audit on
chorus v0.5.2 and surfaced eight findings (3 P1, 5 P2, 1 P3) plus a
structural ask about server-side ranking defaults. Every claim was
verified directly against the code before any edit. Two findings were
re-shaped on verification:

1. **The MCP `analyze_variant_multilayer` StopIteration crash (P1.2)
   and the `predict_variant_effect(assay_ids=[])` returns-0-tracks bug
   (P2.2) are the same root cause.** The MCP docstring at
   `server.py:963` explicitly invites `assay_ids=[]` to mean "all
   tracks", but `_predict()` checked `if assay_ids is None`. Empty list
   → empty predictions dict → `next(iter(ref_pred.values()))` at
   `variant_report.py:903` raises `StopIteration`. Fixing the empty-list
   handling removes the crash.

2. **The CHIP-Histone window-widening the collaborator proposed is
   already in place.** `scorers.py:103` has `histone_marks: window_bp=2001`.
   Only `tf_binding` is still 501 bp (~4 bins at 128 bp/bin). Rather
   than widen `tf_binding` (which would invalidate the per-track CDFs
   built at 501 bp on HuggingFace AND dilute genuine narrow TF
   footprints), the v0.5.3 fix surfaces a `low_effective_bins`
   diagnostic flag so quantization-vulnerable rows are visible without
   silently changing the data shape.

## What v0.5.3 ships

| ID | Surface | Change |
|---|---|---|
| **P1.1** | `chorus/analysis/variant_report.py` | `build_variant_report` type-hint widened to `PerTrackNormalizer \| QuantileNormalizer \| None`; explicit `TypeError` on anything else; clear log warning when a `QuantileNormalizer` is passed (table scoring works but IGV rescale won't); HTML metadata callout in the IGV section when in this state |
| **P1.2 / P2.2** | `chorus/oracles/{alphagenome,enformer,borzoi}.py` | `_predict(assay_ids=[])` now treated like `None` (= all tracks). New `EmptyPredictionsError` in `chorus/core/exceptions.py`. Explicit guard at `variant_report.py` instead of `StopIteration` |
| **P1.3** | `chorus/analysis/variant_report.py` | IGV browser auto-truncates to top-50 tracks by `abs(raw_score)` with `logger.warning` and yellow HTML callout (was: silent strip with no IGV) |
| **P2.1** | `chorus/oracles/alphagenome_source/alphagenome_metadata.py` | New `iter_tracks()` public iterator that excludes padding rows; `_tracks` docstring updated to mark it as internal and explain why padding rows are kept in the raw list (output-array index alignment) |
| **P2.3** | `chorus/analysis/variant_report.py` | New `TrackScore.low_effective_bins` boolean field set when scoring-window / native-bin-resolution < 8 (fires on AG CHIP-TF at 501 bp / 128 bp); surfaced in `to_dict()` and downstream consumers. **No window change** (preserves CDFs) |
| **P2.4** | new `chorus/oracles/bpnet.py` | Public `load_bpnet_model`, `encode_sequence`, `predict_bpnet` helpers exposed from `chorus.oracles`; encapsulates the `sys.path` + `BPNet.arch` + `tasks.json` + bias-tensor recipe that previously lived only inside `chrombpnet.py:_load_direct` |
| **P2.5** | `chorus/analysis/normalization.py` | New `is_ready_for_oracle(name)` module function unifies `PerTrackNormalizer.has_oracle` (NPZ) + `QuantileNormalizer.has_oracle_quantiles` (NPYs), honoring the `_CDF_ALIASES` map (so `alphagenome_pt` resolves via `alphagenome`) |
| **P3** | `chorus/utils/sequence.py`, `README.md` | New `get_centered_window(fasta, chrom, pos_1based, length, ref, alt)` — 1-based-position-safe centered ref/alt window with strict ref-base validation; README note under the quick-start variant example |
| **Ranking** | `chorus/analysis/discovery.py`, `chorus/mcp/server.py` | `discover_variant` / `discover_variant_effects` / `_score_all_tracks` / `_rank_and_select` / `_rank_cell_types` all default to `ranking_metric="alt_x_abs_effect"`. Cell-type and layer rankings now carry `ref_value`, `alt_value`, `ranking_score`, `ranking_metric`, and `low_baseline_warning` (alt < 5 AND \|effect\| > 1.5). Top-level result includes `_ranking_metric` marker |

## Out of scope (deliberate)

| Item | Reason |
|---|---|
| Widen `tf_binding` window 501 → 2001 | Would invalidate all per-track CDFs on HuggingFace AND average in unchanged flanking bins. `low_effective_bins` flag delivers the diagnostic without breaking backward compat |
| `histone_marks` window change | Already at 2001 bp / 128 bp = 15 bins, safe |
| Ranking change in `analyze_variant_multilayer` | Operates on user-curated `assay_ids`; the closed-baseline bias is less of a concern after manual curation. Future work |
| Ranking change in `score_variant_batch` | Sorts variants (not tracks) by per-variant max effect; different semantics from track-level ranking |
| `analyze_variant_multilayer` HTML metadata mismatch (QuantileNormalizer flow) | Resolved by P1.1: the HTML callout now appears whenever the IGV rescale will fall back to autoscale |

## Files changed

```
chorus/__init__.py
chorus/analysis/discovery.py
chorus/analysis/normalization.py
chorus/analysis/variant_report.py
chorus/core/exceptions.py
chorus/mcp/server.py
chorus/oracles/__init__.py
chorus/oracles/alphagenome.py
chorus/oracles/alphagenome_source/alphagenome_metadata.py
chorus/oracles/borzoi.py
chorus/oracles/enformer.py
chorus/oracles/bpnet.py            (new)
chorus/utils/__init__.py
chorus/utils/sequence.py
README.md
setup.py
```

## Verification (see plan: /Users/lp698/.claude/plans/flickering-seeking-valley.md)

- Targeted import + behavior smoke (this session): all 5 checks pass
  (chorus 0.5.3, AG padding hidden via `iter_tracks`, `TrackScore.low_effective_bins` surfaces, `is_ready_for_oracle('alphagenome_pt')` True via alias, `EmptyPredictionsError` raised on empty ref_pred).
- Full base pytest: see `pytest_log.txt` next to this file.
