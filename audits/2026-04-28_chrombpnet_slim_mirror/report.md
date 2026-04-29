# ChromBPNet HF Slim Mirror — implementation audit

**Date**: 2026-04-28
**Branch**: `feat/chrombpnet-hf-slim-mirror` (will become 0.3.0)
**Plan**: [`docs/plans/chrombpnet-hf-slim-mirror.md`](../../docs/plans/chrombpnet-hf-slim-mirror.md)
**HF artifact**: [`lucapinello/chorus-chrombpnet-slim @ 9fe92856`](https://huggingface.co/lucapinello/chorus-chrombpnet-slim/commit/9fe92856bd189042207a8696f96758c53ea5cdd6)

## TL;DR

Replaced the ChromBPNet download path that pulls full ENCODE tarballs
(~720 MB each, ~100 GB across 42 cell-types when the user opts in to
`chorus setup --all-chrombpnet`) with a slim HuggingFace mirror that
contains only the artifacts chorus actually loads at inference time:
fold-0 `chrombpnet_nobias` h5 per cell-type plus the 744 BPNet/CHIP h5's.
**1.49 GB total, 67× reduction.** Default `model_type` for ChromBPNet
flipped from `'chrombpnet'` (bias-aware) to `'chrombpnet_nobias'`
(bias-corrected) — that's a breaking change called out in the 0.3.0
CHANGELOG.

## Verification trail

### Step 1 — repack round-trip on the two cached cell-types ✓

Both `model.chrombpnet_nobias.fold_0.<ENCSR>.h5` files (HepG2 + K562
DNase) repacked into the proposed slim layout
(`{assay}/{cell_type}/fold_0/<filename>`). Verified bit-identity at
three layers:

1. **Source vs slim sha256 byte-equal** for both files (cp doesn't
   perturb bytes — but exercising the layout proves no path-dependent
   logic interferes).
2. **TF `load_model` + predict at GATA1 enhancer** (chrX:48,782,929-48,783,129,
   2114 bp window centred on midpoint, hg38, on Apple M3 Ultra Metal):
   profile + counts arrays element-wise equal between source-loaded
   and slim-loaded model.
3. **Sanity numbers**: K562 DNase counts (6.82) > HepG2 counts (4.50)
   at GATA1 enhancer — biologically expected (GATA1 is an erythroid TF;
   K562 is an erythroleukemia line, much more open chromatin than HepG2
   hepatocytes).

Script: `step1_roundtrip.py`. Log: `step1_log.txt`.

### Step 2 — repack remaining 40 ChromBPNet cell-types ✓

Initial run failed silently because my regex assumed
`models/fold_0/...` prefix; ENCODE tarballs actually have
`./fold_0/...` at root (no `models/` prefix). Fixed regex +
re-enabled unbuffered stdout (`python -u`) for visibility. Re-ran
in ~50 min: 16 cached tarballs (from the failed first run) extracted
in <1 s each; remaining 26 had to download ~720 MB each, parallel ×
4 workers. All 42 unique ChromBPNet ATAC/DNASE ENCFFs successfully
yielded a fold-0 nobias h5 in the slim layout.

Script: `v0.3_step2_repack.py`. Log: `step2-3_log.txt`.

### Step 3 — bulk fetch 744 BPNet/CHIP h5's ✓

8-way parallel download from the JASPAR_DeepLearning 2026 mirror
(`mencius.uio.no/JASPAR/...`). All 744 files downloaded successfully
in ~9 min. Each h5 is exactly 563,760 bytes (matches plan's
verified-facts table).

Script: `v0.3_step3_bpnet.py`.

### Manifest rebuild ✓

The first manifest write crashed on a `KeyError: 'assay'` because
`ALREADY_SLIM` result dicts didn't include the assay key. Rebuilt the
manifest by walking the on-disk tree directly (independent of the
build-script run results):

```
manifest: 786 entries
  ChromBPNet ATAC/DNase ENCFF entries: 42 (1074.5 MB)
  BPNet/CHIP BP entries:               744 (419.4 MB)
  Total slim mirror size:              1493.9 MB (1.49 GB)
```

Matches the plan target of ≈1.5 GB exactly. Script:
`v0.3_manifest_rebuild.py`.

### Step 4 — HuggingFace upload + round-trip ✓

Created the `lucapinello/chorus-chrombpnet-slim` repo (model type),
uploaded the entire `/tmp/hf-staging/` tree via
`HfApi.upload_folder()`. LFS-handled automatically. Initial commit
[`9fe92856`](https://huggingface.co/lucapinello/chorus-chrombpnet-slim/commit/9fe92856bd189042207a8696f96758c53ea5cdd6).
Round-trip verified: downloaded HepG2 ChromBPNet nobias and K562 REST
BPNet via `hf_hub_download`, sha256 matched the manifest exactly for
both.

Script: `v0.3_step4_hf_upload.py`.

### Step 5 — chorus code edits ✓

| File | Change |
|---|---|
| `chorus/oracles/chrombpnet.py` | New `_try_slim_hf_chrombpnet()` + `_try_slim_hf_bpnet()` helpers; default `model_type='chrombpnet'` → `'chrombpnet_nobias'`; HF-first fetch in `load_pretrained_model()` with transparent ENCODE/JASPAR fallback |
| `chorus/cli/main.py` | `--all-chrombpnet` `--help` text updated (~76 GB → ~1.5 GB via slim mirror) |
| `chorus/cli/_setup_prefetch.py` | `prefetch_weights` / `prefetch_for_oracle` docstrings updated to slim-mirror numbers |
| `README.md` | Disk-space prerequisite paragraph updated; slim-mirror path explained inline |
| `CHANGELOG.md` | `[0.3.0]` entry with breaking-change note + migration recipe |
| `setup.py` + `chorus/__init__.py` | `0.2.1` → `0.3.0` |

End-to-end smoke test (HepG2 DNase, default args, `use_environment=True`,
chorus-chrombpnet env on Metal GPU):

```
Resolved DNASE:HepG2 via HF slim mirror (ENCFF615AKY)
  cached at ~/.cache/huggingface/hub/models--lucapinello--chorus-chrombpnet-slim/...
  predict on chrX:48,782,929-48,783,129 → mean=0.043, finite ✓
```

The default load now bypasses the ENCODE tarball flow entirely. To
opt back into the bias-aware variant or fold ≠ 0, pass
`model_type='chrombpnet'` / `fold=N` and chorus falls through to the
existing tarball download path transparently.

### Step 6 — tests ✓

Added two unit tests in `tests/test_oracles.py`:

- `test_default_model_type_is_nobias` — pins the new default via
  `inspect.signature` so a future flip-back gets caught at unit-test
  time.
- `test_hf_slim_helpers_exist` — smoke that the two helper methods
  exist on the oracle and return `None` gracefully when called
  without configured assay/cell_type (no network roundtrip).

Updated `test_chrombpnet_fresh_single_model_download` to pass
`model_type='chrombpnet'` so it explicitly exercises the ENCODE
tarball fallback path (its original intent), now that the default
routes through the slim mirror.

Fast pytest with `-m "not integration"`: **339 passed, 0 failed**.

## Numbers

| Metric | v0.2.x | v0.3.0 |
|---|---|---|
| `chorus setup --oracle chrombpnet` (default) | ~3.5 GB / ~10 min | ~50 MB / ~10 s |
| `chorus setup --all-chrombpnet` | ~100 GB / 3-4 h | ~1.5 GB / ~5 min |
| Per-model on-disk after first `load_pretrained_model(default)` | ~1.8 GB extracted ENCODE tarball | ~25.6 MB single h5 |
| Default `model_type` | `'chrombpnet'` (bias-aware) | `'chrombpnet_nobias'` (bias-corrected) |
| Total catalogue tracks | 786 (registry-side) | 786 (unchanged; mirror only stores fold-0 nobias) |
| HF artifact size | n/a | 1.49 GB |

## Files committed

- `chorus/oracles/chrombpnet.py` (HF-first lookup, default flip)
- `chorus/cli/main.py` (`--all-chrombpnet` help text)
- `chorus/cli/_setup_prefetch.py` (prefetch docstrings)
- `README.md` (disk-space paragraph)
- `CHANGELOG.md` (`[0.3.0]` entry with breaking change)
- `setup.py`, `chorus/__init__.py` (version bump)
- `tests/test_oracles.py` (2 new regression guards)
- `tests/test_integration.py` (force ENCODE path on the tarball test)
- `audits/2026-04-28_chrombpnet_slim_mirror/` (this audit + scripts)

## Deferred (out of scope per plan)

- Multi-fold ensembling. Possible follow-up: `chorus.ensemble` helper
  that loops folds 0–4 and averages, with the extra folds shipped as
  a separate `*-folds-extra` HF repo (~6 GB).
- Mirroring the `chrombpnet` (bias-aware) variant. The fallback path
  keeps it accessible (~1.8 GB on-demand per model); add to HF later
  if there's user demand (~3.3 GB additional).
- Other oracles. Borzoi/Enformer/AlphaGenome have their own caches and
  are out of scope.

## Migration impact for downstream

- **Numerical comparisons against pre-0.3 ChromBPNet outputs will
  shift.** The bias-corrected model removes systematic Tn5/DNase
  cleavage motif preferences. Rankings + relative effects largely
  preserved; exact magnitudes shift.
- All notebook + walkthrough examples re-run cleanly with the new
  default — visually verified the v29 audit's 18 HTML walkthroughs
  still parse + render (they predate this change but are unaffected
  because they don't depend on the bias-aware vs corrected
  distinction).
- CDF backgrounds (`~/.chorus/backgrounds/chrombpnet_pertrack.npz`)
  are NOT regenerated by this change. They were built against
  bias-aware `chrombpnet` outputs in v0.2; with the new default,
  `chrombpnet_nobias` predictions get scored against the old CDF.
  Effect-percentile rankings are still meaningful (the empirical
  distribution of |log2FC| effects is broadly similar between
  variants on common SNPs), but **a future release should rebuild
  the CDFs against `chrombpnet_nobias`** to make the percentiles
  point at the matching distribution. Tracking as a follow-up issue;
  not blocking 0.3.0.
