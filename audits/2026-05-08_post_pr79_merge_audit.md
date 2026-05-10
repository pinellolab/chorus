# Post-PR-#79 merge audit — handoff for Lorenzo

**Date:** 2026-05-08  
**Branch:** `fix/post-v040-followups` (commits `0c2b8e6` + `b2cb4e8` on top of merge `63df601`)  
**Auditor:** Luca + Claude (Opus 4.7, 1M context)  
**Scope:** verify the merged code (Lorenzo's PR #79 + our local "unify track-rescale" follow-ups) is ready to merge into `main`.

---

## TL;DR

✅ **376 tests pass** (warm-state pytest after fresh CDF swap).  
✅ **All 18 walkthrough HTMLs render** with peaks visible — programmatic IGV inspection found 0 issues.  
✅ **Default-call behaviour is unified**: IGV, matplotlib, CoolBox, and notebooks all produce CDF-rescaled output with no extra params (same `1.0 = p99` semantics, same `3.0` cap).  
✅ **README + walkthroughs README + NORMALIZATION_GUIDE + VISUALIZATION_GUIDE re-read**; 2 P1 stale-claim fixes applied (display range `1.5 → 3.0`).  
✅ **README links audited** (51 targets) — all resolve. (Two Zenodo URLs return HTTP 403 to scripted HEAD but are valid via API/browser.)  
✅ **Branch pushed** to `origin/fix/post-v040-followups`.

~~🟡 **One deferred item**: the local DHS-augmented ChromBPNet CDF
was missing the 744 BPNet/CHIP tracks; needed a rebuild + HF upload.~~
✅ **Closed 2026-05-09**: the GPU-machine agent rebuilt the 744 CHIP
rows with DHS augmentation on ml007 (~6 h, 2× A100); the local 42
ATAC/DNASE rows were spliced in on this Mac; resulting **uniform
786-track DHS-augmented NPZ** (sha
`526beb2ce8310f6fdb331f766eac55ce3262b67f1a43416532d8bad8f83183eb`)
uploaded to `huggingface.co/datasets/lucapinello/chorus-backgrounds`.
See `audits/2026-05-09_dhs_chrombpnet_full_rebuild.md` for the
sub-audit.  After the upload, all 786 tracks have uniform sampling
(`effect_counts=18672`, `summary_counts=34004`,
`perbin_counts=1088128`).

---

## What changed in this branch (vs `origin/main`)

```
b2cb4e8 chore(examples): regenerate SORT1 chrombpnet + multi-oracle artefacts
0c2b8e6 feat(viz): unify track-rescale across IGV / matplotlib / CoolBox / notebooks
63df601 Merge Lorenzo's PR #79 into fix/post-v040-followups
965d0dd Added updated multioracle examples           (Lorenzo)
fc38632 fix: support mixed-resolution tracks…       (Lorenzo)
9151338 fix: genome concurrent decompression race + stale ChromBPNet health probe
fecf407 feat: add chorus cleanup command
0e4fb6a feat: add --setup-timeout to chorus setup
85b12ca fix: CHIP strand suffix mismatch in normalization + alphagenome_pt CDF alias
```

### Lorenzo's PR #79 (kept as-is)

| Change | File | Notes |
|---|---|---|
| `_match_track_id` / `_find_matching_cdf` (perbin → summary → effect fallback) | `normalization.py` | LegNet uses `summary_cdfs` for IGV rescale via the fallback |
| `_calculate_track_bin_size` per-oracle dispatch | `_igv_report.py` | chrombpnet `bin=20`, legnet `bin=resolution`, others `window/3000` |
| `aggregation_method` param on `_downsample_to_features` | `_igv_report.py` | mean / max |
| `windowFunction: "max"` IGV WIG hint for high-res oracles | `_igv_report.py`, `multi_oracle_report.py` | Browser-side aggregation |
| `(per-track norm)` LegNet panel-label suffix | `multi_oracle_report.py` | Tells users the values aren't directly comparable |
| `get_max_output_size()` widens multi-oracle region to ~1 Mb | `regenerate_multioracle.py` | |
| New `t_start = variant_pos - (actual_bp_in_array // 2)` IGV formula | `_igv_report.py` | Identical to our parallel local fix |

### Our local follow-ups (one squashed feat commit)

| Change | File | Why |
|---|---|---|
| Single unified helper `rescale_for_display(values, layer, normalizer, oracle_name, assay_id) → (out, cfg)` | `_igv_report.py` | Single source of truth — IGV, matplotlib, CoolBox, notebooks all use it |
| `apply_floor_rescale` returns 4-tuple `(rescaled, ref, alt, signed)` | `_igv_report.py` | So callers can pick symmetric vs unsigned scale_cfg |
| `signed_floor_rescale_batch` — symmetric signed rescale to `[-3, +3]` using `p99(|cdf|)` | `normalization.py` | Borzoi RNA / Sei / LentiMPRA repressive effects now visible (were clipped to 0) |
| `is_signed()` fuzzy track-id matching incl. CHIP `:+`/`:-` strand stripping | `normalization.py` | LegNet `LentiMPRA:HepG2` correctly resolves to `HepG2` row → signed guard fires |
| `OraclePrediction.add()` backfills `track.assay_id` from dict key | `core/result.py` | ChromBPNet tracks with `assay_id=None` now usable by CoolBox/matplotlib auto-load |
| CoolBox `get_coolbox_representation(normalize=True)` auto-loads normalizer | `core/result.py` | `pred[i].get_coolbox_representation()` with no args → CDF-rescaled output |
| matplotlib `render_track_figures(normalize=True)` auto-loads | `_track_figure.py` | Same default behaviour |
| `_has_samples` guard moved inside `_find_matching_cdf` | `normalization.py` | Failed-build perbin rows fall through to summary instead of saturating to `max_value` |
| `max_value` default `1.5 → 3.0` in `perbin_floor_rescale_batch` | `normalization.py` | Matches `_DISPLAY_MAX = 3.0`; was silent clip-bug for any caller without explicit `max_value=` |
| `ChromBPNetOracle.predict_sliding(seq)` | `oracles/chrombpnet.py` | Slides 2114-bp model across arbitrary intervals with cigar substitutions preserved → ChromBPNet visible across the full multi-oracle 1 Mb locus |
| `_predict()` auto-routes wide queries to `predict_sliding` | `oracles/chrombpnet.py` | PR #79's wider `genomic_region` was triggering a pre-existing IndexError in `_predict_direct`'s sliding formula |
| `_calculate_track_bin_size` chrombpnet uses `agg="max"` (was `"mean"`) | `_igv_report.py` | PR #79's docstring said "max pooling preserves peaks for ChromBPNet" but code returned `"mean"` — corrected |
| Lower per-layer floors: `chromatin_accessibility 0.95→0.90`, `promoter_activity 0.95→0.85` | `_igv_report.py` | Peak base/shoulder visible alongside peak top |
| Causal-report IGV (`causal._build_causal_igv`) goes through the unified helper | `analysis/causal.py` | Same path as variant + multi-oracle reports |
| matplotlib symmetric y-axis fallback for signed layers (no normalizer) | `_track_figure.py` | Repressive RNA/Sei/MPRA signal stays visible in zoom-in/out PNGs |
| DHS-vocabulary utilities `load_dhs_vocabulary()` / `sample_dhs_positions()` | `utils/annotations.py` | Used by the (deferred) DHS-augmented CDF rebuild |
| Multi-oracle wide-locus wiring uses `predict_sliding` | `scripts/regenerate_multioracle.py` | |
| Test updates: `test_apply_floor_rescale_passthrough` (4-tuple); `test_perbin_none_for_scalar_oracles` (perbin → summary fallback); new `test_rescale_for_display_unified_helper` | `tests/test_analysis.py` | |
| README display-range fixes | `README.md` | `1.5 → 3.0` at lines 1191, 1269 |

---

## Verification matrix

| Check | Result | Notes |
|---|---|---|
| `pytest -m "not integration"` | ✅ 376 passed, 1 skipped, 5 deselected | After swapping in HF-shipped CDF |
| ChromBPNet single-oracle SORT1 regen | ✅ `+0.318 log2FC, ≥99th, Activity 0.603` | Matches expected biology |
| Multi-oracle SORT1 (chrombpnet/legnet/alphagenome + consolidate) | ✅ all 4 artefacts regenerated | LegNet panel: scale `[-3, +3]`, 20,174 negative + 797 positive features |
| All 18 walkthrough HTMLs IGV-parsed | ✅ 0 issues — every panel has data | One file (batch_scoring) is table-only, no IGV expected |
| README link audit (51 links) | ✅ all resolve | 2 Zenodo URLs return 403 to scripted HEAD; valid via API |
| Doc consistency (README, walkthroughs README, NORMALIZATION_GUIDE, VISUALIZATION_GUIDE) | ✅ 2 P1 stale claims fixed | `[0, 1.5]` display-range references → `[0, 3.0]` |

### Default-call behaviour (no extra params required)

| Path | Auto-load mechanism | Verified |
|---|---|---|
| IGV variant report | reads `report._normalizer` | ✅ |
| IGV multi-oracle | reads each `rep._normalizer` | ✅ |
| IGV causal | reads `top_s._variant_report._normalizer` | ✅ |
| matplotlib `render_track_figures(...)` | `normalize=True` → `get_normalizer(first.source_model)` | ✅ |
| CoolBox `track.get_coolbox_representation()` | `normalize=True` → `get_normalizer(self.source_model)` | ✅ |

Opt-out: `normalize=False` for matplotlib + CoolBox; `igv_raw=True` on the variant report.

### CDF flow per oracle (Lorenzo's principled-not-hack concern)

| Oracle | Variant `effect_pctile` | Variant `activity_pctile` | IGV per-bin rescale | Notes |
|---|---|---|---|---|
| ChromBPNet | `effect_cdfs` | `summary_cdfs` | `perbin_cdfs` | All three CDFs read directly via the unified helper |
| LegNet | `effect_cdfs` | `summary_cdfs` | `summary_cdfs` (signed) → symmetric `[-3, +3]` rescale via `signed_floor_rescale_batch` | Was per-track autoscale before; now consistent semantics across oracles |
| AlphaGenome / Enformer / Borzoi | `effect_cdfs` | `summary_cdfs` | `perbin_cdfs` | Lorenzo's fallback never triggers (all three present) |
| Sei | `effect_cdfs` (signed) | `summary_cdfs` | n/a — heatmap, not signal track | |

No CDF is bypassed; no hardcoded thresholds; the DHS-augmented samples (when present) are still doing work at every CDF read.

---

## Deferred work (post-merge follow-ups)

1. ~~**DHS-augmented ChromBPNet CDF — rebuild ALL 786 tracks, then
   upload to HF.**~~ ✅ **Closed 2026-05-09.**  Hybrid build (744 CHIP
   rows on ml007 + 42 ATAC/DNASE rows from the local Mac May-7
   rebuild, spliced together on this Mac), uniform 786-track NPZ
   live on `lucapinello/chorus-backgrounds`.  Sub-audit:
   `audits/2026-05-09_dhs_chrombpnet_full_rebuild.md`.

2. **Doc P2 polish** — surface the unified `rescale_for_display()` helper in `VISUALIZATION_GUIDE.md` (currently only mentioned in `README.md`); add a paragraph explaining that signed layers now use symmetric `[-3, +3]` rescale.

3. **`scripts/regenerate_examples.py` BCL11A/FTO/SORT1_with_CEBP examples** — Lorenzo's PR commented these out. The HTMLs are still in the repo but the regen script won't refresh them. Decide: re-enable, or remove the stale HTMLs.

---

## How to validate the merge before merging into main

```bash
# 1. Pull the branch
git fetch origin fix/post-v040-followups
git checkout fix/post-v040-followups

# 2. Tests (warm — uses your existing envs + cached CDFs)
mamba run -n chorus pytest tests/ -q -m "not integration"
# Expected: 376 passed

# 3. Regenerate the SORT1 multi-oracle (the canonical demo)
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --consolidate

# 4. Open the report — visually verify the IGV panel
open examples/walkthroughs/validation/SORT1_rs12740374_multioracle/rs12740374_SORT1_multioracle_report.html
# Look for: chrombpnet panel covers full 1 Mb width (predict_sliding); legnet panel
# shows BOTH positive and negative tails (symmetric signed scale, was clipped to 0).
```

— end of audit —
