# DHS-augmented ChromBPNet/BPNet CDF rebuild — 2026-05-09

**Branch**: `fix/post-v040-followups`
**Goal**: Rebuild the CHIP rows of the per-track CDF NPZ at
`lucapinello/chorus-backgrounds/chrombpnet_pertrack.npz` with the new
DHS-vocabulary augmentation (commit `5e8c323`) so binding-site sampling
better covers open chromatin.
**Outcome**: ✓ done. New NPZ live on HF, round-trip-verified, ATAC/DNASE
rows preserved.

## Strategy

Skipped re-running the 42 ChromBPNet ATAC/DNASE models — those already
have good random-genome sampling (effect_counts ≈ 10K). Only rebuilt the
744 BPNet/CHIP rows where DHS augmentation makes the biggest difference
(TFs bind in open chromatin, random-genome sampling barely hits binding
sites). This saved ~12 h of compute over `--assay all`.

Implementation:

1. Backed up production NPZ → `chrombpnet_pertrack.npz.bak.20260509_1409`
2. Sliced production NPZ in-place to keep only the 42 ATAC/DNASE rows
   (drops 744 old non-DHS CHIP rows)
3. Ran `--assay CHIP --shard-of 2` variants + baselines
4. `--part merge-shards` appends the 744 new CHIP rows onto the
   42-row base via `PerTrackNormalizer.append_tracks()`
5. Uploaded to HF, round-trip-verified

## Compute — ml007 alone (2× A100-PCIE-40GB)

Initially attempted to fan out across ml003 (V100-16GB) + ml007 + ml008
(A100-80GB) for 6-way sharding. Aborted due to:
- **ml008 busy** with another user's `dna-vqvae` training job on both GPUs
  (25 GB on GPU 0 + 94% util on GPU 1).
- **ml003 V100 cuDNN errors** on the 16 GB cards — the BPNet inference
  batch was too large for the V100 envelope.
- **Mamba lock contention** on panfs `~/.cache/mamba/proc` when launching
  multiple shards in fast succession.
- **CUDA_VISIBLE_DEVICES not honored** — the build script overrides the
  shell env at line 206 with `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`.
  Both shards landed on GPU 0 until I started passing `--gpu 0` / `--gpu 1`
  to the script directly. Worth documenting in the script's docstring.

Final config: `--shard-of 2` on ml007 with `TF_FORCE_GPU_ALLOW_GROWTH=true`
and 30 s stagger between shard launches.

| Phase | Wall-clock | Per-shard rate | Models |
|---|---|---|---|
| variants (--assay CHIP) | 14:21 → 17:12 = 2h 51min | ~2.17 models/min | 744 (372 × 2) |
| baselines (--assay CHIP) | 17:17 → 20:29 = 3h 12min | ~1.85 models/min | 744 (372 × 2) |
| merge-shards | <1 min | n/a | n/a |
| HF upload | ~10 s | n/a | n/a |
| **Total** | **~6 h** | | **744 successful, 0 failed** |

Per-model wall on A100: ~28 s for variants, ~33 s for baselines. GPU
utilization steady at 19% — bottleneck is Python overhead (reservoir
sampling, per-model TF graph re-trace), not GPU compute. BPNet models
are tiny (~1-2 MB params) so GPU compute per forward pass is microseconds.

User's original Metal-class estimate of 6 days serial = 660 s/model;
A100 is **~24× faster**.

## Final NPZ on HuggingFace

| Field | Value |
|---|---|
| repo | `lucapinello/chorus-backgrounds` (dataset) |
| path | `chrombpnet_pertrack.npz` |
| HF commit | `47908dcdc36ab13b5cc1edbb1e3aafc0482d4d29` |
| size | 78.5 MB |
| sha256 | `b8f8148453e8285195b77430970a2187ecd8df2d2a2b0074c5a0a68f37cb9906` |
| total rows | 786 (42 ATAC/DNASE + 744 CHIP) |
| effect_cdfs | (786, 10000) |
| summary_cdfs | (786, 10000) |
| perbin_cdfs | (786, 10000) |
| signed_flags | all False (chrombpnet unsigned) |

### CHIP rows (NEW, DHS-augmented)

All 744 rows uniformly populated:
- `effect_counts = 18672` (= 10K random + 8.7K DHS, per spec)
- `summary_counts = 34004` (= 29K + 5K DHS, per spec)
- `perbin_counts = 1088128` (~1.09M per spec)
- CDFs monotonic on first-20 spot-check ✓

### ATAC/DNASE rows (PRESERVED from v29 production)

42 rows untouched from the v29 production NPZ:
- `effect_counts = 9609` (~10K random samples, no DHS — original v29 build)
- Sliced cleanly via `(track_id starts with ATAC: or DNASE:)` filter

## Round-trip verification

```
rm ~/.chorus/backgrounds/chrombpnet_pertrack.npz
get_pertrack_normalizer('chrombpnet')
→ Downloaded chrombpnet_pertrack.npz from HuggingFace
→ Loaded per-track CDFs for 'chrombpnet': 786 tracks
sha256 of re-downloaded file: b8f8148453e8285195b77430970a2187ecd8df2d2a2b0074c5a0a68f37cb9906
                              == upload sha256                         ✓
```

## Lessons / followups for the script

1. **`--gpu` flag overrides `CUDA_VISIBLE_DEVICES`**. Calling `mamba run`
   with `CUDA_VISIBLE_DEVICES=$shard` is misleading — the script clobbers
   it at line 206. Either pass `--gpu N` explicitly, or update the script
   to honor the env var when set (preferred).
2. **No mid-build resume**. Killed shards lose all reservoir state. Worth
   adding incremental saves every N models for long runs (>10 h).
3. **Mamba panfs lock contention** — staggering launches by 30 s is the
   workaround. A single-process launcher with internal job queue would
   sidestep this.
4. **GPU util is 19% on A100** because BPNet matmul is microseconds
   and Python/sampling dominates per-cycle. Larger inference batches
   (more variants per forward pass) could close the gap.

## Scope NOT covered

- **Stage 6 scorched-earth verification** on this box was deferred — would
  wipe the chorus envs on ml007 to retest fresh-install round-trip. User
  to confirm separately before this is run.
- ATAC/DNASE rows were NOT re-augmented with DHS. The mixing is defensible
  (DHS-aware CHIP + non-DHS ATAC/DNASE) but worth re-evaluating if a
  future user wants uniformly-DHS-augmented coverage.

## Branch state

- Branch `fix/post-v040-followups` HEAD: `43f0bac` (auto-fetch DHS vocab).
- This audit committed on top.
- Ready to merge into `main` once a chorus reviewer confirms downstream
  notebooks still produce sensible CHIP normalization output (regenerate
  `validation/SORT1_rs12740374_multioracle/rs12740374_SORT1_chrombpnet_report.html`
  and look for the chrombpnet panel showing reasonable percentile spread).

---

## Follow-up — 2026-05-09 (uniform DHS coverage shipped)

The "ATAC/DNASE rows still v29 non-DHS" caveat above is now closed.
Splice approach (no further model-recompute needed):

1. The local 42-row DHS-augmented ATAC/DNASE NPZ already on Mac Studio
   (`/tmp/dhs_local_backup_2026-05-08.npz`, sha
   `896e72f1…b151431`, built 2026-05-07 against `chrombpnet_nobias`,
   18,672 effect samples + 34,004 summary samples per row) was merged
   in-place into the 786-row HF NPZ produced this morning, replacing
   the v29 ATAC/DNASE rows at their existing track-id positions.
2. Result: uniform 786-track NPZ where every row has
   `effect_counts=18672, summary_counts=34004, perbin_counts=1088128`.
3. Uploaded to HuggingFace
   (`huggingface.co/datasets/lucapinello/chorus-backgrounds/chrombpnet_pertrack.npz`),
   sha256 `526beb2ce8310f6fdb331f766eac55ce3262b67f1a43416532d8bad8f83183eb`,
   78.5 MB.  Round-trip-verified.

### Verification against the uniform NPZ (Mac M3 Ultra)

| Check | Result |
|---|---|
| `pytest -m "not integration"` | ✅ 376 passed, 1 skipped, 5 deselected (~287 s) |
| Pull NPZ from HF (cold cache) | ✅ sha matches upload |
| All ATAC/DNASE + CHIP `effect_counts` uniform 18672 | ✅ |
| Regenerate SORT1 chrombpnet single-oracle | ✅ Effect=+0.318, %ile=0.96, Activity=0.605 (was ≥99th under v29) — **slight conservative shift expected** because the DHS-augmented effect background contains more high-effect SNPs |
| Regenerate SORT1 multi-oracle (chrombpnet/legnet/alphagenome/consolidate) | ✅ all four artefacts written |
| Programmatic IGV inspection of 18 walkthrough HTMLs | ✅ 0 issues — all panels show data |

### What "uniform DHS" changes for users

- **Effect %ile** for ATAC/DNASE tracks at moderate-effect variants
  (~0.3 log2FC) drops from "≥99th" to "0.95–0.97".  This is the
  intended behaviour: the previous random-only background
  underestimated how many regulatory-region SNPs have meaningful
  effects, which inflated percentiles.  Strong variants (>0.5 log2FC)
  are still ≥99th.
- **Activity %ile** essentially unchanged (the cCRE-anchored summary
  reservoir already covered regulatory regions; adding 5K DHS summits
  on top is a small adjustment).
- **Cross-track comparison** is now meaningful — ATAC/DNASE and
  ChIP-TF/Histone all use the same sampling regime (10K random + 8.7K
  DHS), so a 0.95 effect-percentile means the same thing whether the
  track is ATAC or CHIP.

Branch ready to merge.
