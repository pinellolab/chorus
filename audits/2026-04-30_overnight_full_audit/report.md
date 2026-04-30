# Overnight scorched-earth audit — final report

**Started**: 2026-04-29 22:57 EDT
**Finished**: 2026-04-30 ~04:00 EDT (final report)
**Operator**: Claude (autonomous overnight)
**Mode**: literal scorched-earth — deleted all 7 chorus oracle envs + base env, rebuilt from `environment.yml` per README, then exercised everything.
**Branch state at start**: `main` @ `ac17f0c` (HF mirror consolidation merge)
**Branch state at finish**: `main` + open PR [#69](https://github.com/pinellolab/chorus/pull/69) (one P0 fix found)

## TL;DR

**Ready to add the CDF tomorrow.** Found 1 P0 install bug → fixed + PR'd as #69. Found 1 P1 UX issue (pre-existing). Everything else passes:
- All 7 oracle envs rebuild from a fresh state via `chorus setup`
- All 7 oracles report `chorus health` Healthy
- All 7 oracles load + predict + return finite values via the Python API
- All 3 shipped notebooks (`single_oracle_quickstart`, `advanced_multi_oracle_analysis`, `comprehensive_oracle_showcase`) execute end-to-end with 0 cell errors
- The MCP `analyze_variant_multilayer` tool runs end-to-end on SORT1 rs12740374
- The new `recommend_alphagenome_backend` MCP tool returns sensible recommendations across 4 platform configs
- The fast test suite passes 368/368 on the freshly-built env
- The README's canonical Enformer example works (`oracle.predict(('chr11', 5247000, 5248000), ['ENCFF413AHU'])` → finite shape (896,))

## Findings (severity-ordered)

### P0 — `chorus setup --oracle enformer` crashes on a fresh install (FIXED)

**Symptom** (caught in Phase E): `chorus setup` succeeded for 6/7 oracles but `enformer` crashed with `ModelNotLoadedError: Failed to load Enformer model: lucapinello/chorus-enformer does not exist..`.

**Root cause**: in PR #68 the Enformer mirror routing was wired into the subprocess templates (`enformer_source/templates/{load,predict}_template.py`) but NOT into `chorus/oracles/enformer.py:_load_direct`. The `default_model_path` was correctly flipped to `lucapinello/chorus-enformer`, but `_load_direct` passed it directly to `tensorflow_hub.hub.load(weights)` which only accepts URLs. Same gap existed in `borzoi.py:_load_direct` (would have hit anyone using `use_environment=False` with the chorus mirror — `chorus setup`'s prefetch path).

**Fix**: PR [#69](https://github.com/pinellolab/chorus/pull/69), commit `4125256`. Adds the same HF-detection logic to `_load_enformer_with_tfhub_recovery` and `borzoi.py:_load_direct` that was already in their subprocess templates.

**Verification**: `chorus setup --oracle enformer` now succeeds in 5s (warm HF cache); `chorus health --oracle enformer` reports Healthy; the README's canonical example returns finite predictions.

**Severity**: P0 because a fresh install is broken without this; the 6/7 partial state would block the user's morning CDF work because `chorus setup` returns non-zero.

### P1 — `Sei.predict(assay_ids=None)` crashes (pre-existing)

**Symptom** (caught in Phase G): `oracle.predict(...)` without explicit `assay_ids` raises `TypeError: 'NoneType' object is not iterable` at `chorus/oracles/sei.py:361` because `Sei._validate_assay_ids` doesn't handle the `None` sentinel that other oracles treat as "all assays".

**Severity**: P1, pre-existing (not a regression from this audit's scope of changes). Workaround documented in the audit log: pass explicit IDs from `oracle._get_all_assay_ids()`. Worth a small follow-up: either default to "all classes" in `Sei._validate_assay_ids` (to match the base-class contract) or raise a clear `ValueError` that points at `list_class_types`.

### Doc references (P2 — informational)

Phase A's stale-reference grep found these non-stale references — they're documentation of mirror provenance, not bugs:

- `tfhub` mentioned in `README.md` (cache-troubleshooting section), `tests/test_error_recovery.py`, and `scripts/build_backgrounds_enformer.py` — all legitimate (former is documenting the TFHub cache directory layout for users who hit it; the latter two predate the HF mirror and reference the original URL by name).
- `johahi` in `CHANGELOG.md`, `README.md`, `docs/NORMALIZATION_GUIDE.md`, `docs/plans/sei-hf-mirror.md`, `scripts/build_backgrounds_borzoi.py`, and the chorus loader fall-back paths — all documenting that the chorus-borzoi mirror is sourced from johahi's repos. Right.
- `zenodo.org/record/4906997` in README, sei-hf-mirror plan, and `chorus/oracles/sei.py:get_zenodo_link()` — all documenting the canonical Zenodo source. Right.

## Phase-by-phase

| Phase | What | Result | Wall time |
|---|---|---|---|
| A | Static + repo consistency | ✅ 368 fast tests, MCP imports clean | 5 min |
| B | Preflight snapshot | ✅ 8 envs sized, HF cache 7.8 GB warm, hg38 3 GB, all 6 CDFs cached | <1 min |
| C | Tear down all 8 envs | ✅ removed cleanly | <2 min |
| D | Rebuild base env | ✅ chorus 0.3.0 installed editable | ~5 min |
| E | `chorus setup` (all 7 oracles) | ✅ after P0 fix; 6/7 first pass + 1/1 enformer re-prefetch | ~14 min + 1 min |
| F | `chorus health` for each | ✅ all 7 Healthy | ~2 min |
| G | Per-oracle smoke (load + 1 predict + finite check) | ✅ 7/7 PASS with valid IDs (Sei needs explicit; documented) | ~6 min |
| H | Notebook execution end-to-end | ✅ 3/3 notebooks (49 + 127 + 59 cells, 0 errors) | ~30 min total |
| I | `analyze_variant_multilayer` smoke | ✅ SORT1 rs12740374 PASS, 29s, HTML report written | <1 min |
| J | MCP server smoke | ✅ list_oracles=7, recommend works for 4 platform configs, list_tracks works | <1 min |
| K | README walkthrough | ✅ canonical Enformer example + `chorus genome list` + backgrounds list pass | <1 min |
| Final fast suite (post-rebuild) | | ✅ **368 passed**, 1 skipped, 0 failed | 5 min |

Total wall clock: ~80 min compute (heavy on Metal for the notebooks) plus my orchestration.

## What's left for tomorrow

1. **Merge PR #69** — the Enformer/Borzoi `_load_direct` fix. Prerequisites the CDF rebuild (the rebuild script uses `use_environment=False` which goes through `_load_direct`).
2. **Receive CDF rebuild output** from the CUDA-box agent → upload to `huggingface.co/datasets/lucapinello/chorus-backgrounds`.
3. *Optional*: file a tiny issue for the Sei `assay_ids=None` UX paper-cut (P1; not blocking).

## Caveats

- **HF cache was warm at start.** Install times reflect a warm-cache user; a literal first-time download (no `~/.cache/huggingface/`) would add ~10 min for chorus mirrors total. The cache wasn't blown away because that risk wasn't worth the marginal "even more new user" signal.
- **Notebook execution used Metal**, not CUDA. The advanced + comprehensive notebooks may behave slightly differently on Linux/CUDA (different op precision, JAX backend); not validated overnight. The MCP/walkthrough audit on Linux/CUDA is the user's other-machine task.
- **One smoke test (Phase I) had a false-positive ERROR** when I called `analyze_variant_multilayer` without the now-required `assay_ids` positional argument. Re-ran with the right signature → PASS. Worth a doc note that the signature changed at some point.

## Sign-off for tomorrow's CDF upload

Path is clear:
- ChromBPNet env was rebuilt from scratch in this audit → loads + predicts correctly via slim mirror
- The CDF upload target (`lucapinello/chorus-backgrounds` dataset repo) is unchanged; existing CDFs untouched on disk + on HF
- The `--model-type chrombpnet_nobias` flag in `scripts/build_backgrounds_chrombpnet.py` (PR #67, merged) is what the CUDA agent is using
- After PR #69 merges, `chorus setup --oracle enformer` will work for any user pulling latest main, so a future fresh install on top of the new CDFs won't break

**Recommended morning sequence**:
1. Merge PR #69 (1 click; CI green expected since changes are surgical)
2. Pull CDF NPZ from CUDA agent → spot-check (HANDOFF.md has the criteria) → upload via `HfApi.upload_file` to `lucapinello/chorus-backgrounds`
3. Smoke a single ChromBPNet variant analysis with the new CDF to confirm percentiles look sensible
4. Tag the release; CHANGELOG already has the Unreleased section ready for version bump
