# ChromBPNet CDF rebuild against `chrombpnet_nobias` — agent handoff

You're picking up the ChromBPNet per-track CDF rebuild that the 0.3.0
release (`9e42bfd`) called out as a deferred follow-up. Goal: produce
a fresh `chrombpnet_pertrack.npz` whose percentile lookups match what
`oracle.predict()` returns for the post-0.3 default
(`model_type='chrombpnet_nobias'`), and upload it to
`huggingface.co/datasets/lucapinello/chorus-backgrounds`.

## Why

The CDFs on HF today were built in 0.2.x against the **bias-aware**
`chrombpnet` variant. Chorus 0.3.0 flipped the default to
`chrombpnet_nobias` (bias-corrected). User-facing percentile
assignments now look up `chrombpnet_nobias` predictions against
`chrombpnet` empirical distributions — the bias systematically
shifts the mapping. Audit at
`audits/2026-04-28_chrombpnet_slim_mirror/report.md` flagged this as
"a future release should rebuild the CDFs against `chrombpnet_nobias`
to make the percentiles point at the matching distribution."

## Hardware needed

- **Linux + CUDA GPU** (A100 or similar; chorus's lab box is fine)
- ~50 GB free disk for the slim-mirror models + interim CDF shards
- The `chorus-chrombpnet` conda env (TensorFlow). Build with
  `chorus setup --oracle chrombpnet` if you don't have it. Includes
  CUDA-enabled TF.

Estimated wall-clock with the now-tracked numbers in
`scripts/build_backgrounds_chrombpnet.py`'s `--assay` help text:

- ATAC/DNase (42 models): ~22 min/model × 42 = **~15 hours on Metal**.
  Should be ~3-5× faster on A100 → **~3-5 hours**.
- CHIP/BPNet (1259 models): ~3 min/model × 1259 = **~63 hours on Metal**.
  ~10-20 hours on A100.

Total CUDA budget: **~13-25 hours**, well-suited for an overnight run
with `--shard` parallelism if you want to split across GPUs.

## What's already done

- `scripts/build_backgrounds_chrombpnet.py` was updated in commit
  `<this-commit>` to take `--model-type` (default `chrombpnet_nobias`,
  matches the 0.3+ chorus default). Re-running today produces the
  right CDFs without a flag.
- The slim HF mirror at `lucapinello/chorus-chrombpnet-slim` already
  ships fold-0 `chrombpnet_nobias` for all 42 ATAC/DNase models +
  all 744 BPNet/CHIP models (1.49 GB total). The build script will
  auto-fetch from there — no ENCODE tarballs needed for the rebuild.

## Run the build

From the repo root, on the CUDA box:

```bash
cd chorus
git pull origin main  # make sure you have the --model-type flag

# Confirm env exists
mamba env list | grep chrombpnet

# IMPORTANT — verify huggingface_hub is in the env. The
# chorus-chrombpnet env yml (post-PR-#60) lists huggingface_hub>=0.20.0,
# but older installs predate that and need a manual install. Without
# huggingface_hub the build script falls back to the ~700 MB-per-model
# ENCODE tarball flow, which turns a 3-5 h ATAC/DNase run into a 30+ h
# run because it re-downloads tarballs we don't need.
mamba run -n chorus-chrombpnet python -c "import huggingface_hub" \
  || mamba run -n chorus-chrombpnet pip install "huggingface_hub>=0.20.0"

# Confirm slim mirror is reachable + has all 786 models
mamba run -n chorus-chrombpnet python -c "
from chorus.oracles.chrombpnet_source.chrombpnet_globals import (
    iter_unique_models, iter_unique_bpnet_models,
)
print('ATAC/DNASE models:', len(list(iter_unique_models())))
print('CHIP/BPNet models:', len(list(iter_unique_bpnet_models())))
"
# Expect: 42 / 1259 (note: the 744 figure in the slim-mirror manifest is
# de-duped; the 1259 includes per-cell-type duplicates that point at
# the same h5 file in the mirror).

# === Phase 1: ATAC/DNase models (42 models, ~3-5 h on A100) ===
mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
  --part variants --assay ATAC_DNASE --gpu 0 \
  --model-type chrombpnet_nobias \
  2>&1 | tee logs/bg_chrombpnet_variants_atac.log

mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
  --part baselines --assay ATAC_DNASE --gpu 0 \
  --model-type chrombpnet_nobias \
  2>&1 | tee logs/bg_chrombpnet_baselines_atac.log

# Phase 1 finished: run `--part merge` (or `--part merge-incremental`) NOW
# to consume the ATAC/DNASE interim NPZ before Phase 2 writes a new one.
# Otherwise Phase 2 will refuse to overwrite (since v0.4.x's --force-gated
# safety check, fixing #71/#73). Pass `--force` if you intentionally want
# to discard the Phase 1 interim and rebuild Phase 2 in isolation.

# === Phase 2: CHIP/BPNet models (1259 models, ~10-20 h on A100) ===
mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
  --part variants --assay CHIP --gpu 0 \
  --model-type chrombpnet_nobias \
  2>&1 | tee logs/bg_chrombpnet_variants_chip.log

mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
  --part baselines --assay CHIP --gpu 0 \
  --model-type chrombpnet_nobias \
  2>&1 | tee logs/bg_chrombpnet_baselines_chip.log

# === Phase 3: merge interim shards into one NPZ ===
mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge \
  2>&1 | tee logs/bg_chrombpnet_merge.log

# Output: ~/.chorus/backgrounds/chrombpnet_pertrack.npz
```

If you have 2 GPUs available, parallelize the CHIP phase across them:

```bash
# Terminal 1 (GPU 0):
CUDA_VISIBLE_DEVICES=0 ... --part variants --assay CHIP --gpu 0 --shard 0 --shard-of 2 ...
CUDA_VISIBLE_DEVICES=0 ... --part baselines --assay CHIP --gpu 0 --shard 0 --shard-of 2 ...

# Terminal 2 (GPU 1):
CUDA_VISIBLE_DEVICES=1 ... --part variants --assay CHIP --gpu 0 --shard 1 --shard-of 2 ...
CUDA_VISIBLE_DEVICES=1 ... --part baselines --assay CHIP --gpu 0 --shard 1 --shard-of 2 ...

# After both finish:
mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge-shards
```

The outer `CUDA_VISIBLE_DEVICES` is what pins each terminal to its physical GPU. The inner `--gpu 0` is now a no-op when the env var is set (v0.4.x fix for #72/#74; previously the `--gpu` arg silently clobbered the outer var and both terminals fought over GPU 0).

## Spot-check before upload

```bash
mamba run -n chorus python -c "
import numpy as np
npz = np.load('~/.chorus/backgrounds/chrombpnet_pertrack.npz', allow_pickle=True)
print('Keys:', list(npz.keys()))
print('Track count:', len(npz['track_ids']))
print('Effect counts dist:', np.percentile(npz['effect_counts'], [0, 25, 50, 75, 100]))
print('Per-track signedness:', npz['signed_flags'][:5], '... (n_signed=', npz['signed_flags'].sum(), ')')
# Sanity: percentiles are monotone
for r in npz['summary_cdfs'][:5]:
    n = r.shape[0]
    p50, p95, p99 = int(.50*n), int(.95*n), int(.99*n)
    assert r[p50] <= r[p95] <= r[p99], 'CDF not monotone'
print('All sanity checks passed')
"
```

Pass criteria:
- 786 tracks (or whatever the current model catalogue is — same count as old NPZ minus any since-removed models)
- All `effect_counts > 0`
- Summary CDFs monotone p50 ≤ p95 ≤ p99
- No NaN / Inf in any CDF

## Upload to HF

```bash
mamba run -n chorus python -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj=os.path.expanduser('~/.chorus/backgrounds/chrombpnet_pertrack.npz'),
    path_in_repo='chrombpnet_pertrack.npz',
    repo_id='lucapinello/chorus-backgrounds',
    repo_type='dataset',
    commit_message='Rebuild ChromBPNet CDFs against chrombpnet_nobias (0.3.0+ default)',
)
"
```

(You'll need an HF token with write access to `lucapinello/chorus-backgrounds`
in your env — this is the same token used for other chorus uploads.)

## Write up the audit

In `audits/2026-04-29_chrombpnet_cdf_rebuild/report.md`:

- Hardware (GPU model, distro, RAM)
- Wall-clock per phase (variants + baselines for each assay family)
- Track count + sample percentiles (one or two cell-types) showing the
  shift from the old `chrombpnet` CDFs
- Any models that failed to load (the build script logs warnings;
  collect the count and make sure it's small / reproducible)
- Hash of the uploaded NPZ for reproducibility

Then commit the audit + log files on a new branch
`audit/2026-04-29-chrombpnet-cdf-rebuild` and open a PR with title
"audit: rebuild ChromBPNet CDFs against `chrombpnet_nobias` (0.3.0+ default)".

## What to do if something breaks

- **Slim-mirror download fails for a specific model**: log it, skip,
  continue. The `--only-missing` flag lets you fill gaps later. We
  expect 0-1 failures across 786 models; > 5% is a problem worth
  surfacing.
- **TensorFlow OOM mid-run**: drop `--batch-size` from 64 to 32 or 16
  and resume with `--only-missing`. The reservoir state is checkpointed
  per-track to interim NPZ files.
- **Want to compare old vs new percentiles for sanity**: pull the
  current `chrombpnet_pertrack.npz` from HF before starting the rebuild
  (`huggingface-cli download lucapinello/chorus-backgrounds chrombpnet_pertrack.npz --repo-type dataset`),
  diff a few tracks side-by-side. Expected: rankings preserved on
  large-effect SNPs, magnitudes shift by 5-30 % on ambiguous cases
  (the biology's the same; the mapping just lines up properly now).

## Related

- chorus PR #59 (0.3.0 release) — the default flip that necessitates this
- `audits/2026-04-28_chrombpnet_slim_mirror/report.md` — flagged the deferred CDF rebuild
- `scripts/build_backgrounds_chrombpnet.py` — the build script (now `--model-type` aware)
- `chorus/analysis/normalization.py:PerTrackNormalizer` — how chorus consumes the CDFs at predict-time
