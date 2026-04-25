# ChromBPNet full ENCODE catalog + CDF rebuild + HF push

**Date**: 2026-04-25
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB / Metal GPU
**Branches**: `feat/2026-04-25-chrombpnet-full-encode-catalog` (#50, merged),
`feat/2026-04-25-chrombpnet-incremental-cdf-build` (#51, merged),
`fix/2026-04-25-chrombpnet-cdf-merge-log` (this audit)
**HF artifact**: `huggingface.co/datasets/lucapinello/chorus-backgrounds`
@ commit [`5f64545`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/commit/5f645455cd5db14447cc522b12ea58d818a53e4b)

## Why

User: "I want to have all the chrombpnet models and bpnet models and
also make sure we add the missing CDF for these models so everything
is better included!"

Pre-v28 chorus exposed 24 of 42 ENCODE-published ChromBPNet model
annotations. Per-track CDF backgrounds (used for percentile-normalised
variant scoring) covered only those 24 tracks. Adding new models to
the registry without rebuilding CDFs would leave them un-normalised.
This audit closes both gaps end-to-end.

## What shipped

### PR #50 — registry expansion (merged earlier today)

`chrombpnet_globals.py` now lists 42 distinct ENCFFs across all
ENCODE biosamples + developmental stages, plus an `iter_unique_models()`
helper that dedupes the registry's aliases (e.g. `"limb"` and
`"limb_E12.5"` both resolve to the same ENCFF, build script iterates
the canonical set once).

### PR #51 — incremental build flags (merged earlier today)

`scripts/build_backgrounds_chrombpnet.py` gained `--only-missing`
(skip models already in the existing NPZ) and `--part merge-incremental`
(stitch new rows onto the existing NPZ). Cuts rebuild time from
~19 h to ~7 h on the v28 dataset (24 → 42).

### This audit — the actual rebuild + HF push

1. **Pre-download** all 17 missing ENCODE tarballs in parallel
   (4 workers, ~20 min for 12 GB total).
2. **Build CDFs** for the 18 new tracks on the M3 Ultra Metal GPU,
   using `chorus-chrombpnet`'s `tensorflow-metal` 1.2.0:
   ```bash
   mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \\
       --part both --only-missing --gpu 0
   ```
3. **Verify** the resulting 42-track NPZ (monotone CDFs, p50≤p95≤p99,
   all counts > 0).
4. **Upload** to `lucapinello/chorus-backgrounds` via HF write token.
5. **Round-trip verify** via `download_pertrack_backgrounds("chrombpnet")` —
   42 tracks downloaded back cleanly.

### Trailing fix (this PR)

The `--part merge-incremental` step writes the merged NPZ in place,
which invalidates the open numpy `existing` handle. The trailing
`logger.info(...)` then accessed `existing["track_ids"]` and crashed
with `BadZipFile`. The merge had already succeeded; only the log
message broke. Fixed by capturing `existing_count` before the write.

## Numbers

| Item | v28 before | v28 after |
| --- | --- | --- |
| ChromBPNet ATAC/DNase ENCFFs in registry | 24 | 42 |
| Distinct biosample × stage entries | 24 | 42 |
| Per-track CDF rows in NPZ | 24 | 42 |
| NPZ size on disk | 2.5 MB | 4.4 MB |
| Effect samples per track | 9,609 | 9,609 (uniform) |
| Summary samples per track | 29,004 | 29,004 (uniform) |
| Perbin samples per track | 928,128 | 928,128 (uniform) |

All quality checks pass for all 42 tracks:
- effect/summary/perbin CDF rows monotone non-decreasing
- p50 ≤ p95 ≤ p99 on summary CDFs
- all counts strictly > 0 (no failed-to-build tracks)

## Time budget

```
Stage                                Wall-clock
-----------------------------------  -----------
Pre-download 17 tarballs (parallel)  ~20 min
Incremental build (18 new models)    6 h 15 min  (12:06 → 18:21)
Bad CRC retry for 1 model            22 min      (18:28 → 18:50)
HF upload                            <1 min
Round-trip verify                    <1 min
-----------------------------------  -----------
Total                                ~6 h 35 min
```

Per-model compute on Metal: 8.7 min variants + 13.3 min baselines = 22 min.
Network bandwidth dominated the pre-download (3 MB/s per ENCODE connection,
4 parallel = ~12 MB/s aggregate).

## Hiccups encountered

1. **Truncated download** — `ENCFF941SIQ.tar.gz`
   (ATAC:neural_tube_unknown_stage) finished at 354 MB / 720 MB, leaving
   a corrupt gzip. Build script logged warning + continued, leaving the
   row with `counts=0` in the merged NPZ. Fix: re-download the tarball,
   delete the zero-count row from NPZ, re-run `--only-missing` (now
   filters 41/42 tracks present, builds the 1 missing one), 22 min.

2. **Trailing logger crash post-merge** — see "Trailing fix" above.

## Out of scope here

- **BPNet / CHIP CDFs** — 1,259 TF binding models in
  `chrombpnet_JASPAR_metadata.tsv`. None have CDFs today; building all
  would take ~10 days of GPU compute. Left for a future audit. Users
  loading `cell_type="K562", assay="CHIP", TF="GATA1"` get raw scores
  but no percentile-normalised effect.

- **Tarball cleanup** — `downloads/chrombpnet/` is now ~30 GB. The
  setup-marker writes happen per-cell-type so a future
  `chorus setup --oracle chrombpnet --force` would re-download
  everything from scratch. Could be optimised by sharing extracted
  weights across cell-type variants, but that's an upstream chrombpnet
  change.

## Verified

```bash
# Local rebuild (auto-merge done by --part both --only-missing)
ls -la ~/.chorus/backgrounds/chrombpnet_pertrack.npz
# 4.4 MB, 42 tracks

# HF upload check
mv ~/.chorus/backgrounds/chrombpnet_pertrack.npz ~/.chorus/backgrounds/chrombpnet_pertrack_local.npz
mamba run -n chorus python -c "
from chorus.analysis.normalization import download_pertrack_backgrounds
print(download_pertrack_backgrounds('chrombpnet'))
"
# → 1 (downloaded)
# → 42 tracks confirmed via np.load
```
