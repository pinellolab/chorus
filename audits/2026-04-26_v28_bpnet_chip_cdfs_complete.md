# BPNet/CHIP CDFs: 6-GPU sharded rebuild, 786-track NPZ on HF

**Date**: 2026-04-26
**Platform**: 3 Linux machines (ml003: 2x V100-16GB, ml007: 2x A100-40GB, ml008: 2x A100-80GB)
**Branch**: `audit/2026-04-26-bpnet-cdfs-complete`
**HF artifact**: `huggingface.co/datasets/lucapinello/chorus-backgrounds`
@ commit [`c1e5fc1`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/commit/c1e5fc161ad4f5acedd2607dd6217ef5996096a1)

## Why

PR #53 added `--shard` support to `build_backgrounds_chrombpnet.py` for
distributing the 744 BPNet/CHIP model CDF build across multiple GPUs.
This audit covers the first full 6-GPU sharded build, merge, verification,
and HuggingFace push.

## What shipped

### Bug fix — BPNet strand-dimension mismatch

BPNet CHIP models output profile predictions with shape `(B, L, 2)`
(two strands) and counts with shape `(B, 2)`, but `predict_profiles_batch()`
assumed 2D outputs `(B, L)` and `(B, 1)`. The multiplication
`softmax_probs * np.exp(counts[:, 0:1])` failed with a broadcast error
`(64, 1000, 2) vs (64, 1)`, producing 0 effect samples for every model.

**Fix**: added strand-sum before scoring in `predict_profiles_batch()`:
```python
if probabilities.ndim == 3:
    probabilities = probabilities.sum(axis=-1)   # (B, L, 2) -> (B, L)
if counts.ndim == 2 and counts.shape[1] > 1:
    counts = counts.sum(axis=-1, keepdims=True)  # (B, 2) -> (B, 1)
```

### 6-shard distributed build

Orchestrated via `scripts/run_bpnet_cdf_build.sh`:

| Shard | Machine | GPU | Models | Wall time |
|-------|---------|-----|--------|-----------|
| 0 | ml003 | V100 #0 | 124 | ~50 min |
| 1 | ml003 | V100 #1 | 124 | ~50 min |
| 2 | ml007 | A100-40GB #0 | 124 | ~40 min |
| 3 | ml007 | A100-40GB #1 | 124 | ~37 min |
| 4 | ml008 | A100-80GB #0 | 124 | ~32 min |
| 5 | ml008 | A100-80GB #1 | 124 | ~28 min |

Total wall time: ~50 min (limited by slowest V100 shards).

### Per-model output (all 744 models)

- **9,609 effect samples** per model (variant scoring)
- **29,004 summary + 928,128 perbin samples** per model (baseline scoring)
- **0 failed models** — all 744 BPNet/CHIP models loaded and scored successfully
- No JASPAR mirror flakiness encountered

### Merge (CHIP shards)

```bash
mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge-shards
```

Output: `768 tracks (24 existing + 744 new from 6 shards)`

### Gap-fill: 18 missing ChromBPNet ATAC/DNASE tracks

The pre-existing NPZ had only 24 of 42 ChromBPNet tracks. The 18 missing
tracks from the v28 catalog expansion (PR #50) were backfilled on ml007:

```bash
mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
    --part both --only-missing --gpu 0
```

18 models (10 ATAC + 8 DNASE) downloaded from ENCODE and scored in ~2.5 h
(download-dominated, ~10 min/model). All produced 9,609 effect + 29,004
summary + 928,128 perbin samples each, matching the existing tracks.

### Verification (final 786-track NPZ)

- Total tracks: **786** (42 ATAC/DNASE + 744 CHIP)
- `effect_counts > 0`: **True** (all tracks)
- `summary_counts > 0`: **True** (all tracks)
- `perbin_counts > 0`: **True** (all tracks)
- Monotone `effect_cdfs`: **True** (first 50 checked)
- Monotone `summary_cdfs`: **True** (first 50 checked)
- File size: **82.4 MB**

### HuggingFace push

1. Initial push (768 tracks, 80.5 MB) at commit `8d78353`.
2. Final push (786 tracks, 82.4 MB) at commit
   [`c1e5fc1`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/commit/c1e5fc161ad4f5acedd2607dd6217ef5996096a1).

### Round-trip verification

```
download_pertrack_backgrounds('chrombpnet') -> downloaded: 1
Remote tracks: 786
```

## Preflight issues resolved

1. **SSH host key changes** on ml003 and ml007 — stale keys removed from
   `~/.ssh/known_hosts`, new keys accepted. ml008 was unaffected.
2. **Mamba lock contention** — 6 concurrent `mamba run` processes on shared
   NFS caused lock waits on ml003 shards (0, 1). Resolved automatically
   after ~4 min; no intervention needed.
3. **Failed to launch ptxas** (TF warning on ml007/ml008) — harmless,
   no impact on results.

## Tests pass

All 744 models produced consistent, non-zero outputs across all 6 shards.
Zero errors, zero failed model loads, zero OOM events.
