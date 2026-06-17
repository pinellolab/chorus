# P0 follow-up — variant-scoring window collapse for fixed-input oracles (+ normalization entanglement)

**Date:** 2026-06-17
**Status:** **query-side fix IMPLEMENTED** (central region-widening in `predict_variant_effect`;
ChromBPNet 1 bp region now reproduces +1.374, full suite 407 passed/0 failed). **Background-CDF
rebuild is the required companion** and is in progress on GPU (ml008) — until the rebuilt CDFs
are uploaded, the percentile/normalized scores for *affected* oracles are inconsistent with the
new (correct) query window, so this fix must not be merged before the CDF rebuild lands.
**Severity:** P0 for any *conversational / MCP* variant-effect magnitude on a fixed-input oracle

## Root cause (confirmed)

Not N-padding of the input (the input is real genome). For a region smaller than the oracle's
input window, the returned ``prediction_interval`` spans a different number of bp than the
values array implies (``prediction_interval_bp / len(values) != bin_size``), so the scorer maps
the 501 bp window to the wrong array indices and the effect collapses ~4×. ChromBPNet (1 bp
region): values=2114 but ``prediction_interval``=1000 bp → mismatch → +0.318; full 2114 bp
region: values=2115, ``prediction_interval``=2115 bp → match → +1.374. Fix = widen the region
to the oracle's input window centered on the variant in ``predict_variant_effect`` so the
interval and values always agree. Enformer/AlphaGenome were already consistent (Enformer slices
its prediction to the output window; AlphaGenome strips N-padding), so the fix is a no-op for them.

## Summary

When chorus scores a variant, it builds a genomic `region` and the oracle predicts on it.
If that region is **smaller than the oracle's input window**, the variant effect
**collapses ~4×** for **fixed-input oracles**. Example (ChromBPNet, rs12740374 / HepG2 DNASE):

| region passed | log2FC | ref signal |
|---|---|---|
| 1 bp (`pos:pos+1`) | **+0.318** | ~48 |
| full input window (2114 bp, variant-centered) | **+1.374** | ~288 |

The **input sequence is fine** in both cases (real genome, 0 % N — verified). The collapse
is in how the **output / prediction interval** is mapped to genomic coordinates for a
sub-input region: the variant ends up mapped into an N-padded output interval, so the
501 bp scoring window lands off the real peak. (My initial "N-padding of the *input*"
hypothesis was wrong and is retracted; the padding is in the output-interval mapping.)

## Scope — what's affected

**Oracles** (by `_predict` window handling):
- **Affected (fixed-input):** ChromBPNet (confirmed), and by the same `extend()`/output-mapping
  pattern: Borzoi, Enformer, Sei, LegNet, EPInformer-seq (Enformer additionally has an
  input≫output centering subtlety — see §3).
- **NOT affected:** AlphaGenome — it strips N-padding (`_strip_n_padding`) so a 1 bp query
  still scores on real, variant-centered context. (This is why the article's AlphaGenome
  numbers reproduced and ChromBPNet's did not.)

**Entry points that pass a sub-input (1 bp) region:**
- `chorus/mcp/server.py` `_auto_region()` → used by **4 MCP tools**: `predict_variant_effect`,
  `score_variant_effect_at_region`, `predict_variant_effect_on_gene`, `analyze_variant_multilayer`.
- `chorus/analysis/discovery.py` (≈ lines 197, 378, 701, 747) — `discover_variant`, `discover_variant_cell_types`.
- `chorus/analysis/batch_scoring.py` (≈ line 465) — `score_variant_batch`.
- `chorus/analysis/build_backgrounds.py` (≈ line 266) — **per-track background CDF generation.**
- `chorus/analysis/causal.py` `prioritize_causal_variants` — **already fixed** (full-input window) in the reproducibility PR.

## Why it can't be a one-line fix — the normalization entanglement

`build_backgrounds.py` builds the per-track CDFs (chorus's headline effect/activity
percentiles, ~2 GB of NPZ on HuggingFace) **with the same 1 bp region**. So the
normalization reference is *self-consistent with the collapsed path*: a collapsed query
is compared against a collapsed background, so the **percentiles** are internally valid
even though the **raw log2FC magnitudes** are wrong.

⇒ **Fixing the query windowing (to the full input window) without rebuilding the CDFs
would desync queries from the background distributions and make the percentile scores
wrong.** The fix therefore requires:

## Recommended fix (dedicated PR)

1. **Central widening** in `chorus/core/base.py` `predict_variant_effect` (~line 389, before
   `region_interval` is built): when the passed region is narrower than `self.sequence_length`,
   widen it to the input window centered on the variant (pull real genome). Guard for
   integer `sequence_length`. This fixes all 4 MCP tools + discovery + batch at once.
2. **Cover the `predict()` tuple path too** (`integration.py` and any callers that hit
   `oracle.predict((chrom, pos, pos+1), ...)` directly) — the central fix above only covers
   `predict_variant_effect`.
3. **Handle Enformer's input(393 kb)≫output(114 kb) centering** (the case the original
   `_auto_region` docstring deliberately avoided) so the output window still maps to the
   variant after widening.
4. **Rebuild all per-track background CDFs** with the fixed windowing (re-run
   `build_backgrounds.py` per oracle, re-upload to `lucapinello/chorus-backgrounds`), then
   re-validate effect/activity percentiles.
5. **Regenerate the committed example outputs** (the walkthroughs were generated via
   `scripts/regenerate_multioracle.py`'s explicit-window path, so their raw numbers are
   already full-window-correct; the percentiles will shift once CDFs are rebuilt).
6. Empirically **confirm the collapse on each fixed-input oracle** (Borzoi/Enformer/Sei/
   LegNet/EPInformer-seq) before/after, the way ChromBPNet was confirmed.

## Interim guidance (until the above lands)
- **AlphaGenome** variant magnitudes via the conversational path are correct.
- **Fixed-input oracle** variant magnitudes via the conversational path are **under-reported**;
  use an explicit variant-centered window equal to the oracle input size (e.g. ChromBPNet
  2114 bp) in the Python API to get correct magnitudes — see the article's reproducibility note.
- **Fine-mapping** (`fine_map_causal_variant`) is fixed (full-input window); its ranking is
  validated correct (rs9504151 #1), though the activity-percentile component of the composite
  still references the collapsed-window CDFs until they are rebuilt (ranking is raw-effect-dominated).
