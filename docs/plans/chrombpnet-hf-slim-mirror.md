# ChromBPNet HF slim mirror — agent handoff

**Status**: planning, not yet started. Branch: TBD (suggest `feat/chrombpnet-hf-slim-mirror`).
**Owner**: handed off to next agent.
**Author of this brief**: prior session (Opus 4.7, 2026-04-28) after Fig 3f DNA-Diffusion verification work.

## TL;DR

Replace the current ChromBPNet download path (ENCODE-portal `.tar.gz` bundles, ~100 GB total when fully prefetched) with a slim HuggingFace mirror containing only the artifacts chorus actually loads at inference time. Target size **~1.5 GB**. Default model variant changes from `chrombpnet` (bias-aware) to `chrombpnet_nobias` (bias-corrected) at the same time — that's the scientifically correct default and matches Kundaje paper recommendations.

## Why this is needed

Audited current state (see prior session transcript, 2026-04-28):

- ENCODE bundles each cell-type's ChromBPNet models as one `ENCFF*.tar.gz` containing **5 folds × 3 model variants × extras**. After extraction, one cell-type = ~1.7 GB on disk; the full registry (42 unique tarballs + 744 BPNet/CHIP h5's) = ~100 GB.
- Chorus inference loads exactly **one h5** per `load_oracle` call: `models/fold_<N>/model.<variant>.fold_<N>.<ENCID>.h5`. Default kwargs = `fold=0`, `model_type='chrombpnet'`. No fold averaging anywhere in chorus or in the upstream Kundaje repo (verified against `kundajelab/chrombpnet/chrombpnet/evaluation/make_bigwigs/predict_to_bigwig.py` — single h5 → `model.predict` → bigwig, no ensemble).
- The remaining ~98 GB is all unused: folds 1–4, the alternate `chrombpnet`/`bias_scaled` variants, duplicated TF SavedModel format alongside each h5, training stdout logs (~16 MB per fold), preserved tarballs.

The bias-corrected `chrombpnet_nobias` h5 is **25.6 MB**. Hosting just that for fold 0 across all 42 unique cell-types is **1.07 GB**. Plus 744 BPNet/CHIP h5's at 563 KB each = 0.4 GB. **Slim mirror total ≈ 1.5 GB** (67× reduction).

## Three design decisions agreed in the planning chat

1. **Variant**: ship `chrombpnet_nobias` only as the slim mirror's default. It's the bias-corrected accessibility model that's recommended for variant analysis, motif discovery, region-swap predictions — which is what chorus is used for. The full `chrombpnet` (bias-aware) and `bias_scaled` variants remain available via the existing ENCODE-tarball fallback for users who explicitly request `model_type='chrombpnet'` or `'bias_scaled'`.
2. **Folds**: fold 0 only in the slim mirror. Keep ENCODE-tarball fallback for fold ≠ 0 users (advanced ensembling). Optionally add a separate `*-folds-extra` HF repo later (~6 GB for folds 1–4 of `chrombpnet_nobias`) if anyone asks.
3. **Source-of-truth precedence**: HF primary, ENCODE fallback. The `_download_chrombpnet_model()` path tries HF for `(cell_type, fold=0, variant=chrombpnet_nobias)` first; falls back to the existing ENCODE tarball flow for everything else.

## Default-variant change is a user-visible behavior change

Today: `load_oracle('chrombpnet', assay='DNASE', cell_type='K562')` loads the bias-aware `chrombpnet` h5.
After this change: same call loads `chrombpnet_nobias` instead. **Predictions will shift.**

Treat this as a bug fix, not a regression — the new default is what the Kundaje paper recommends for biology — but call it out clearly:

- CHANGELOG entry under the next minor version (0.3.0): "BREAKING: ChromBPNet default `model_type` is now `chrombpnet_nobias` (bias-corrected). Old behavior: pass `model_type='chrombpnet'` explicitly."
- Mention in the README ChromBPNet section.
- Update any audit notebooks / examples that rely on numerical comparison to old chrombpnet outputs (search `examples/` and `tests/`).

## Concrete implementation steps

### Step 0 — coordinate
Another agent recently shipped v0.2.1 (`7a5ed0a`). Branch off `main` after pulling latest. Don't rebase published audit branches per CLAUDE.md.

### Step 1 — repack locally, prove round-trip
- For the two cell-types already on disk (HepG2, K562 DNase), build the slim layout:
  ```
  <hf-staging>/DNASE/{cell_type}/fold_0/model.chrombpnet_nobias.fold_0.<ENCID>.h5
  ```
- Load the slim h5 directly via `tf.keras.models.load_model(...)` and run a prediction at `chrX:48,782,929-48,783,129` (the GATA1 enhancer used in the Fig 3f verification). Compare against the same prediction loaded from the existing `~/Projects/chorus.bak-2026-04-28/downloads/chrombpnet/DNASE_HepG2/models/fold_0/model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5`. They must be **bit-identical** (same h5 file content, just relocated).
- Don't proceed to upload until that check passes for both K562 and HepG2.

### Step 2 — fetch + repack the remaining 40 cell-types
- Loop the 42-entry registry (`CHROMBPNET_MODELS_DICT['DNASE']` ∪ `['ATAC']`).
- For each ENCFF tar that isn't already cached locally: download into a scratch dir, extract, copy out `models/fold_0/model.chrombpnet_nobias.fold_0.*.h5`, discard the rest.
- Alias-aware: 42 unique ENCFF tarballs map to 57 (assay, cell_type) registry entries because of stage aliases (e.g. `limb_E12.5` ≡ `limb`). The slim mirror should be keyed by ENCFF accession (deduplicated), not by registry alias. Lookup table at chorus side maps `(assay, cell_type)` → ENCFF → HF path.
- Generate a `manifest.json` per HF repo: `{ENCFF_accession: {sha256, size, source_encsr, cell_type, assay}}`. Future audits use this to verify integrity.

### Step 3 — BPNet/CHIP repack (744 models)
- Already individual h5's at `https://mencius.uio.no/JASPAR/JASPAR_DeepLearning/2026/models/BP*/BP*_model.h5` — no extraction needed.
- Each is **fixed size 563,760 bytes** (verified against BP000001/100/500/700 — JASPAR exports them at a uniform layout).
- Mirror them to HF under `BPNet/{BASE_ID}.h5` with the same manifest format. No transformation.

### Step 4 — HF upload
- Repo name suggestion: `lucapinello/chorus-chrombpnet-slim` (single repo for both ATAC/DNase + BPNet, since combined size is well under the 50 GB HF limit).
- Use `huggingface_hub.HfApi.upload_folder` with `repo_type='model'`. LFS handles the binaries automatically.
- Add a clear README on the HF repo explaining: what's in the mirror, what's deliberately missing (folds 1–4, variants other than nobias, raw tarballs), and how to fetch the full bundle from ENCODE if needed.

### Step 5 — switch chorus to use the mirror
Files to edit:
- `chorus/oracles/chrombpnet.py`
  - Change `load_pretrained_model` default `model_type='chrombpnet'` → `'chrombpnet_nobias'`.
  - In `_download_chrombpnet_model`: try HF first via `huggingface_hub.hf_hub_download` for `(fold=0, model_type='chrombpnet_nobias')`. On HF 404 or for other (fold, variant) combos, fall back to existing ENCODE tarball flow.
- `chorus/cli/_setup_prefetch.py`
  - Update the `_chrombpnet_prefetch_script` to use the slim path. The "fast-path default" already only does K562 + HepG2 DNase; that becomes ~50 MB instead of 3.5 GB.
  - Update the comment at line 183 (`tarballs + 5-fold extracted weights, plus 744 tiny BPNet h5 files`) to reflect the new layout.
- `chorus/mcp/server.py:445`: docstring for `load_oracle`'s `fold` param still says "default 0" — fine. But check whether `model_type` is exposed; if so, update its docstring to say new default is `chrombpnet_nobias`.
- `README.md`: update disk-space claim (was just fixed to "~100 GB" in 0.2.1 — will become "~1.5 GB for slim mirror, ~100 GB for full ENCODE registry via fallback").
- `CHANGELOG.md`: 0.3.0 entry per the breaking-change note above.

### Step 6 — tests
- Add a unit test that `load_pretrained_model()` with no `model_type` kwarg ends up at a `chrombpnet_nobias` h5 (regression guard).
- Add an integration test (gated behind a `--slow` marker) that downloads K562 DNase from HF, runs `predict()` at `chrX:48,782,929-48,783,129`, and checks the result against a stored reference array.
- Run the full audit checklist (`audits/AUDIT_CHECKLIST.md`) on a fresh clone before merging — section 4 (CDFs) and section 7 (HTML reports) are the most likely to surface fallout from a default change.

### Step 7 — write a dated audit report
Per `CLAUDE.md`: drop `audits/2026-MM-DD_chrombpnet_slim_mirror.md` summarizing what changed, what you tested, what you deferred. Reference this plan doc.

## Numbers to use in PR description / CHANGELOG

- 100 GB → 1.5 GB (slim mirror)
- Default variant: `chrombpnet` → `chrombpnet_nobias` (bias-corrected)
- 42 ChromBPNet ATAC/DNase × 25.6 MB = 1.07 GB
- 744 BPNet/CHIP × 563 KB = 0.42 GB
- Cache hit avoids ~700 MB tarball download per cell-type
- Full ENCODE bundles still reachable via fallback for users who need folds 1–4 or alternate variants

## Verified facts (don't re-derive)

- ChromBPNet inference loads a single Keras h5; no fold averaging in upstream Kundaje code (`predict_to_bigwig.py`, `training/predict.py`).
- Chorus default is currently `fold=0`, `model_type='chrombpnet'` (chrombpnet.py:273).
- Sizes per fold for one cell-type's DNase model:
  - `model.chrombpnet.fold_N.*.h5` = 77.5 MB
  - `model.chrombpnet_nobias.fold_N.*.h5` = 25.6 MB
  - `model.bias_scaled.fold_N.*.h5` = 2.7 MB
- BPNet h5 size: 563,760 bytes uniformly (verified for BP000001, BP000100, BP000500, BP000700).
- CHROMBPNET_MODELS_DICT: 30 ATAC + 27 DNase entries = 57 registry rows but only **42 unique ENCFF tarballs** (stage aliases dedupe). CHIP entries: 0 (handled separately via JASPAR_metadata.tsv → 744 BPNet models).
- One of the registry entries (`ENCSR498NUZ` retina DNase-seq) has no model file yet per the comment in `chrombpnet_globals.py:6`. Slim mirror should skip it gracefully and fall back to ENCODE.

## Out-of-scope for this PR

- Multi-fold ensembling. Possible follow-up: add a `chorus.ensemble` helper that loops fold 0–4 and averages, with the extra folds shipped as a separate `*-folds-extra` HF repo (~6 GB).
- Migrating the bias-aware (`chrombpnet`) variant to HF. The fallback path keeps it accessible, but if there's user demand we could mirror it later (~3.3 GB more).
- Other oracles. Borzoi/Enformer/AlphaGenome have their own caches and are not in scope here.
