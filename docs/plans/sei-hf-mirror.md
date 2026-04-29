# Sei HuggingFace mirror — agent handoff (target: 0.4.0)

**Status**: planning, not yet started.
**Branch suggestion**: `feat/sei-hf-mirror` (off `main` after `feat/chrombpnet-hf-slim-mirror` lands).
**Author of this brief**: v30 scorched-earth audit session, 2026-04-29.
**References**:
- This plan mirrors the design of `docs/plans/chrombpnet-hf-slim-mirror.md` (already implemented as the v0.3.0 release).
- v30 audit finding F1 (`audits/2026-04-29_v30_slim_mirror_scorched_earth/findings/F1_huggingface_hub_missing_in_chrombpnet_env.md`) — read this first to avoid the same pitfall.

## TL;DR

Mirror the Sei weights bundle (`https://zenodo.org/record/4906997/files/sei_model.tar.gz`, ~2.5 GB unpacked) to a new HuggingFace model repo `lucapinello/chorus-sei`. Switch `chorus/oracles/sei.py:_download_sei_model` to fetch from HF first, with the existing Zenodo URL as fallback. Empirically Zenodo/CERN drops to ~50 MB/min during off-hours; HF CloudFront sustains ~50–100 MB/sec — that's a **~50–80× speedup** for the sei phase of `chorus setup --oracle sei`.

## Why this is needed

Audited during the v30 scorched-earth audit (2026-04-29):

- Sei prefetch ran 09:17 → finished after >70 min (extrapolated; user interrupted at the ~3 GB mark) for what v29 had clocked at ~35 min — Zenodo throughput is volatile and CERN occasionally throttles.
- Sei is the longest single phase of `chorus setup --oracle all`. Every other oracle finishes in under ~3 min; sei dominates total wall time.
- We already pay to host `lucapinello/chorus-chrombpnet-slim` (1.49 GB) and `lucapinello/chorus-backgrounds` (~1.6 GB) on HF. Adding sei (~2.5 GB) is a small marginal cost for a large user-facing speed win.

## Three design decisions (already discussed and agreed)

1. **Layout**: mirror **unpacked** (per-file), not the tarball. Matches `chorus-chrombpnet-slim` pattern. Means no tar extraction at install time, sha256 verifiable per file, cleaner cache pruning.
2. **Files to mirror**: just what `sei.py` actually loads at inference time. Per `chorus/oracles/sei.py:90–107`:
   - `model/sei.pth` — main weights (~2 GB)
   - `model/projvec_targets.npy`
   - `model/histone_inds.npy`
   - `model/target.names`
   - `model/seqclass_info.txt`

   All five live under `model/` inside the Zenodo tarball. Mirror them at `model/<filename>` in the HF repo (preserve the directory structure so chorus's existing path constants don't need to change).
3. **Source-of-truth precedence**: HF primary, Zenodo fallback. `_download_sei_model()` should try HF; on miss / network failure / repo unavailable, fall back to the existing Zenodo tarball download path. Same defensive pattern as `_try_slim_hf_chrombpnet`.

## F1 lesson — do not skip this step

The v0.3.0 ChromBPNet slim mirror was technically wired correctly but the **per-oracle env yaml didn't list `huggingface_hub`**, so the prefetch script (which runs inside `chorus-chrombpnet`) silently fell back to the original ENCODE tarball flow. The headline "1.5 GB slim mirror" claim was true at runtime in the chorus base env, but **false during `chorus setup`** — which is the path most users hit.

Bake this in for sei from day one:

- Add `huggingface_hub>=0.20.0` to `environments/chorus-sei.yml` in the same commit that adds `_try_hf_sei()`. Don't ship the helper without the dep.
- Don't use silent `except ImportError: return None` — `logger.warning(...)` so future "why didn't HF fire?" debugging is one log line away.
- Add a regression test (mirror `tests/test_oracles.py::TestChromBPNetOracle::test_chrombpnet_env_yaml_has_huggingface_hub` for sei).

## Concrete implementation steps

### Step 0 — coordinate
Branch off `main` after `feat/chrombpnet-hf-slim-mirror` is merged. Don't rebase published audit branches per CLAUDE.md.

### Step 1 — local repack + round-trip check
- Use the cached tarball at `downloads/sei/sei_model.tar.gz` (extract if not already done) as the source.
- Build the HF staging layout in a scratch dir:
  ```
  <hf-staging>/model/sei.pth
  <hf-staging>/model/projvec_targets.npy
  <hf-staging>/model/histone_inds.npy
  <hf-staging>/model/target.names
  <hf-staging>/model/seqclass_info.txt
  <hf-staging>/manifest.json
  <hf-staging>/README.md
  ```
- Compute sha256 for each file → stash in `manifest.json`.
- Smoke-test: `SeiOracle()` pointed at the staging dir loads + predicts a single 4096 bp window successfully. Output should be **bit-identical** to a load from the unmodified `downloads/sei/model/`.
- Don't proceed to upload until the round-trip is bit-equal for at least one chrX region.

### Step 2 — HF upload
- Repo name: `lucapinello/chorus-sei` (model repo, public, LFS for the .pth and .npy files).
- Use `huggingface_hub.HfApi.upload_folder` with `repo_type='model'`. LFS handles binaries automatically.
- Repo README: explain what's in the mirror, that it's a drop-in for the Zenodo tarball at `zenodo.org/record/4906997`, and how to use chorus to fetch it.

### Step 3 — chorus integration
Files to edit:
- `chorus/oracles/sei.py`
  - Add `HF_SEI_REPO = "lucapinello/chorus-sei"` constant.
  - Add `_try_hf_sei(self)` helper that returns the local cached `download_dir` path on success, `None` on miss. Implementation cribbed from `_try_slim_hf_chrombpnet`. Crucially: log `warning` (not `info`) when `huggingface_hub` is missing, citing F1 in the message.
  - In `_download_sei_model()`: call `_try_hf_sei()` first; only fall through to the existing Zenodo tarball download if it returns `None`.
- `environments/chorus-sei.yml`
  - Add `huggingface_hub>=0.20.0` to the pip block. (Yes, even though the chorus base env has it via `setup.py install_requires` — the prefetch script runs inside the per-oracle env, see F1.)
- `chorus/cli/_setup_prefetch.py`
  - No code change needed: `_DEFAULT_LOAD_KWARGS` doesn't have a sei entry, the bare `oracle.load_pretrained_model()` call in the prefetch script will trigger `_download_sei_model()` which now prefers HF.
  - Update the docstring near line 175–195 if it mentions "sei from Zenodo (~35 min)" — change to "sei from HF (~1–2 min, with Zenodo fallback)".
- `CHANGELOG.md`
  - 0.4.0 entry under `### Added`: "Sei weights now mirrored at `lucapinello/chorus-sei`. `chorus setup --oracle sei` drops from ~35 min (Zenodo, sometimes >1 h on bad nights) to ~1–2 min (HF CloudFront). Zenodo URL kept as a fallback for users who hit HF rate limits."

### Step 4 — tests
- New unit test: `test_oracles.py::TestSeiOracle::test_sei_env_yaml_has_huggingface_hub` — exact mirror of the chrombpnet F1 regression test, parses `environments/chorus-sei.yml` and asserts `huggingface_hub` is listed in the pip block.
- Smoke test for `_try_hf_sei`: assert it returns `None` when the oracle is uninitialised (no network roundtrip), same shape as the existing `test_hf_slim_helpers_exist` for chrombpnet.
- Integration test (gated `-m integration`): with a clean `~/.cache/huggingface/hub/`, `SeiOracle().load_pretrained_model()` should succeed and the cache should populate at `models--lucapinello--chorus-sei`. Skip if no network.

### Step 5 — audit report
Per CLAUDE.md, drop `audits/YYYY-MM-DD_sei_hf_mirror/report.md` summarizing what changed, what you tested, what you deferred. Include actual wall-time measurements before/after on a fresh `chorus setup --oracle sei` run.

## Numbers to use in PR description / CHANGELOG

- Zenodo throughput observed in v30 audit: ~50 MB/min (today, 2026-04-29) — variable; v29 saw 35 min for the same download
- HF CloudFront throughput typical: ~50–100 MB/sec (1000–2000× faster than Zenodo's bad nights)
- Sei tarball: ~2.5 GB unpacked (sei.pth ~2 GB + 4 small files)
- Expected download time after mirror: **~1–2 min** vs **35–100 min** today

## Verified facts (don't re-derive)

- Source URL: `https://zenodo.org/record/4906997/files/sei_model.tar.gz` (cited at `chorus/oracles/sei.py:570`).
- File layout inside the tarball: `model/sei.pth`, `model/projvec_targets.npy`, `model/histone_inds.npy`, `model/target.names`, `model/seqclass_info.txt` (cited at `chorus/oracles/sei.py:90–107`).
- `huggingface_hub` is in the chorus base env via `setup.py install_requires` but **not** in any per-oracle env yaml except `chorus-alphagenome.yml`. The prefetch path runs inside the per-oracle env and silently falls through if the dep is missing — this caused v30 audit F1.

## Out-of-scope for this PR

- Mirroring **other** Zenodo-hosted weights. Scan if needed: only sei currently uses Zenodo as primary. Borzoi pulls from HF (`johahi/borzoi-replicate-0`), enformer from TFHub, alphagenome from Google's HF, legnet from a small Zenodo bundle that finishes in <30 sec.
- Multi-model sei ensembling. Sei is a single model; no fold structure to deal with.
- Updating sei version. Use the same model files as the current Zenodo bundle — bit-identical mirror.
