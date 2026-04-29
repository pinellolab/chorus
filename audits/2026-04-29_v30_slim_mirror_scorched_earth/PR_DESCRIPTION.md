# Release 0.3.0 — Slim ChromBPNet HF mirror + `chrombpnet_nobias` default

Closes the v0.3.0 release line: replaces the ~100 GB ENCODE-tarball-based ChromBPNet weight prefetch with a 1.49 GB HuggingFace mirror, and flips the default `model_type` from the bias-aware `chrombpnet` to the bias-corrected `chrombpnet_nobias` that the upstream Kundaje pipeline routes to all biology stages.

## Headline numbers (measured on the v30 scorched-earth audit, M4 Max / Metal)

| Metric | 0.2.1 (ENCODE tarballs) | 0.3.0 (HF slim) | Δ |
|---|---:|---:|---|
| `chorus setup --oracle chrombpnet` wall | ~16 min | **1 m 24 s** | **11× faster** |
| `downloads/chrombpnet/` after default setup | ~3.5 GB (HepG2 + K562 ENCODE bundles) | **0 B** (sentinel only) | 71× smaller |
| HF cache delta | 0 | 49 MB (manifest + 2 nobias h5 symlinks) | exactly the slim payload |
| `chorus setup --all-chrombpnet` (full registry) | ~3–4 h, ~100 GB | ~5 min, **1.49 GB** | 67× smaller, 36–48× faster |
| Default ChromBPNet variant | `chrombpnet` (bias-aware) | `chrombpnet_nobias` (bias-corrected) | semantic change — see below |

## Motivation

ChromBPNet ships each ENCODE-portal cell-type bundle as one `ENCFF*.tar.gz` containing 5 folds × 3 model variants × duplicated TF SavedModel format × training stdout logs × preserved tarballs — **~1.7 GB extracted per cell-type, ~100 GB across the 786 published tracks**. Chorus inference loads exactly **one h5** per `load_oracle` call (`fold_0/model.<variant>.fold_0.<ENCID>.h5`), and the upstream Kundaje `predict_to_bigwig.py` / `training/predict.py` do not ensemble across folds. So the other 99% of the bundle is dead weight on disk.

This PR mirrors only the artifacts chorus actually loads (fold-0 `chrombpnet_nobias` h5 per ATAC/DNase cell-type, plus the 744 BPNet/CHIP h5's) to a new HF model repo `lucapinello/chorus-chrombpnet-slim` and switches `chorus/oracles/chrombpnet.py` to fetch from there first, falling back to ENCODE/JASPAR for users who request alternate folds or `model_type='chrombpnet'`/`'bias_scaled'`.

The default `model_type` flip is grounded in upstream documentation:
- `kundajelab/chrombpnet` README line 104 + wiki Output-format lines 13–14: *"chrombpnet_nobias.h5 — TF-Model i.e model to predict bias corrected accessibility profile."*
- `chrombpnet/pipelines.py` routes `chrombpnet_nobias.h5` to **every biology stage** (marginal footprinting line 111, DeepLIFT contribution scores line 136, TFModisco motifs line 147). The full `chrombpnet.h5` only appears in QC scatter plots vs observed bigwigs (line 93).

The chorus-typical workflow (variant analysis, region-swap, motif discovery, MCP region predictions) is exactly the "biology stage" set, so `chrombpnet_nobias` is the right default.

## ⚠️ Breaking change

`load_pretrained_model(assay='DNASE', cell_type='K562')` (and equivalent `chorus.create_oracle('chrombpnet').load_pretrained_model(...)` calls without an explicit `model_type` kwarg) **now load the bias-corrected variant** instead of the bias-aware one. **Numerical predictions will shift in magnitude and shape** because Tn5/DNase enzyme cleavage motifs are removed from the output.

To preserve old behaviour exactly:
```python
oracle.load_pretrained_model(
    assay="DNASE", cell_type="K562", model_type="chrombpnet",
)
```

The bias-aware `chrombpnet` variant is no longer in the default cache; chorus falls back to the full ENCODE tarball flow for it (~700 MB compressed, ~1.7 GB extracted per cell-type). All other API shapes are unchanged.

CHANGELOG, README, and the per-oracle CDF normalization data are aligned to the new default. The shipped notebooks already use the variant chorus picks by default and have been re-executed against `chrombpnet_nobias`.

## Changes by area

### Slim mirror infrastructure (`a13282c`, `802c7b1`, `678dbd5`)

- **New HF model repo** [`lucapinello/chorus-chrombpnet-slim`](https://huggingface.co/lucapinello/chorus-chrombpnet-slim): 786 model files (42 fold-0 ChromBPNet `chrombpnet_nobias` h5's for ATAC + DNase × cell-type, 744 BPNet/CHIP h5's, sha256-pinned in `manifest.json` keyed by ENCFF/JASPAR accession). 1.49 GB total. README on the repo explains the layout and how to fetch the full ENCODE bundle if you need alternate variants.
- **`chorus/oracles/chrombpnet.py`**:
  - New `_try_slim_hf_chrombpnet()` and `_try_slim_hf_bpnet()` helpers fetch via `huggingface_hub.hf_hub_download`. Manifest-driven for ATAC/DNase (sha256 verified by HF), filename-keyed for BPNet (`BPNet/<BP_BASE_ID>.<ver>.h5`).
  - `load_pretrained_model` calls the appropriate slim helper first; on miss / network error / repo unavailable, falls back to the existing `_download_chrombpnet_model()` (ENCODE) or `_download_model_from_JASPAR()` (mencius.uio.no) flows.
  - Default `model_type='chrombpnet'` → `'chrombpnet_nobias'`.

### F1 fix — env yaml dependency + non-silent ImportError (`a13282c`)

The v30 scorched-earth audit caught a P0 bug in the initial slim-mirror landing: `environments/chorus-chrombpnet.yml` did not list `huggingface_hub`, so when the prefetch script ran inside `chorus-chrombpnet` env (via `mamba run -n chorus-chrombpnet`), `_try_slim_hf_chrombpnet` hit `ImportError`, silently returned `None`, and fell back to the 700 MB ENCODE tarball flow. The "1.5 GB / ~10 s" claim was true at runtime in the chorus base env but **false during `chorus setup`** — the path most users hit on first install.

Fix:
- Add `huggingface_hub>=0.20.0` to `environments/chorus-chrombpnet.yml` pip block.
- Replace silent `except ImportError: return None` in both `_try_slim_hf_chrombpnet` and `_try_slim_hf_bpnet` with `logger.warning(...)` that names the missing dep + the env yaml to fix. Future "why didn't HF fire?" debugging is one log line away.

### Tests (`a13282c`)

- `tests/test_oracles.py::TestChromBPNetOracle::test_default_model_type_is_nobias` — pins the new default. (was added in `802c7b1`)
- `tests/test_oracles.py::TestChromBPNetOracle::test_hf_slim_helpers_exist` — wiring smoke for the slim helpers. (`802c7b1`)
- `tests/test_oracles.py::TestChromBPNetOracle::test_chrombpnet_env_yaml_has_huggingface_hub` — F1 regression guard. Parses the env yaml directly and asserts `huggingface_hub` appears in the pip block. **Verified to fail without the fix and pass with it.** (`a13282c`)
- 3 new integration tests covering the slim mirror end-to-end (`802c7b1`).

Full pytest matrix on the post-fix branch: **346 passed, 4 deselected, 14 warnings** (fast) + **3 passed, 1 skipped** (integration, `-m integration`).

### Audit + docs (`fbe47a0`, `441adce`, `62fd9b0`)

- `audits/2026-04-28_chrombpnet_slim_mirror/` — prior agent's local round-trip verification (Step 1 bit-equality check on cached files).
- `audits/2026-04-29_v30_slim_mirror_scorched_earth/` — full scorched-earth audit on a clean install, including:
  - `report.md` — design verification, three-oracle Fig 3f triangulation table, recommendations.
  - `findings/F1_huggingface_hub_missing_in_chrombpnet_env.md` — root cause, fix details, end-to-end re-test results (16× faster, 71× smaller on disk after fix).
  - `setup.log`, `verify.log`, `reverify*.log`, `f1_retest.log` — full trace artifacts.
- `docs/plans/chrombpnet-hf-slim-mirror.md` — original design doc that the implementation followed.
- `docs/plans/sei-hf-mirror.md` — queued v0.4.0 follow-up to mirror the Sei weights to HF as well (Zenodo took 70 min on this audit run vs 35 min on v29 — Zenodo throughput is volatile and the same slim-mirror pattern would help).

## Three-oracle Fig 3f triangulation (independent validation)

To validate that the new default predicts correctly, the audit reproduced the DNA-Diffusion paper's Fig 3f swap experiment with three independent oracles. Locus: `chrX:48,782,929-48,783,129` (hg38, K562 GATA1 enhancer). Replacement: 200 bp HepG2-optimized DNA-Diffusion sequence (`99598_GENERATED_HEPG2`, read from Fig 3g and matched to the deposited STARR-seq library).

| Cell-type | Enformer | AlphaGenome | ChromBPNet (post-PR) | Paper claim |
|---|---:|---:|---:|---|
| HepG2 | +4.22 ↑ | +8.22 ↑ | **+7.09 ↑** | ↑ ✓ |
| K562 | −4.04 ↓ | −5.03 ↓ | **−5.81 ↓** | ↓ ✓ |
| GM12878 | +0.04 (~0) | +0.32 (~0) | **−1.52** | ~unchanged ✓ (mild closing — consistent with ChromBPNet's tighter 2114 bp window) |

All three oracles independently reproduce the paper's qualitative claim. ChromBPNet's GM12878 divergence (−1.52 vs ~0 from the longer-context oracles) is biologically plausible — a HepG2-tuned 200 bp insertion creates local features that AlphaGenome's 1 Mb context smooths out across surrounding regulatory architecture.

## Test plan checklist

- [x] `pytest tests/ -m "not integration and not slow"` — **346 passed**, 4 deselected, 14 warnings (425 s)
- [x] `pytest tests/ -m integration` — **3 passed**, 1 skipped, 346 deselected (681 s)
- [x] `chorus list` — 6/6 oracles installed
- [x] `chorus health` — 6/6 healthy
- [x] `chorus --version` → `0.3.0`
- [x] `chorus backgrounds status` — 6 oracles, chrombpnet 786 tracks
- [x] `chorus setup --help` shows `--all-chrombpnet`
- [x] Default `model_type` is `'chrombpnet_nobias'` (regression test)
- [x] Slim mirror fires during `chorus setup` (F1 re-test): 1 m 24 s wall, 49 MB HF cache, 0 B local downloads — confirmed end-to-end
- [x] Three-oracle Fig 3f triangulation: directional agreement across Enformer / AlphaGenome / ChromBPNet
- [ ] Reviewer to spot-check on Linux/CUDA (the audit was macOS/Metal only — see `audits/2026-04-27_v29_linux_cuda/` for the prior Linux baseline that this branch hasn't been re-tested on)

## Reviewer guidance

Highest-impact files to look at:
- `chorus/oracles/chrombpnet.py` — slim helpers (~70 LOC), default flip (1 LOC), HF-first dispatch (~20 LOC)
- `environments/chorus-chrombpnet.yml` — F1 fix (1 line + comment)
- `tests/test_oracles.py` — new regression tests (~80 LOC)
- `CHANGELOG.md` — breaking-change note for the variant flip

Lower-priority but worth a skim:
- `audits/2026-04-29_v30_slim_mirror_scorched_earth/report.md` for the audit narrative
- `docs/plans/sei-hf-mirror.md` for the queued 0.4.0 work

The HF repo (`lucapinello/chorus-chrombpnet-slim`) and the manifest layout are stable — sha256-pinned per accession, schema-versioned in `manifest.json`. Future agents can extend by appending entries; chorus's resolver is keyed by ENCFF/BP_BASE_ID, not by file order.

## Out-of-scope (explicit)

- **Sei HF mirror** — queued as 0.4.0 in `docs/plans/sei-hf-mirror.md`. Same pattern as this PR; should be a small follow-up that includes the F1 lesson (add `huggingface_hub` to `chorus-sei.yml` in the same commit as `_try_hf_sei`).
- **Mirroring `chrombpnet` (bias-aware) and `bias_scaled` variants** — fallback path keeps them accessible from ENCODE; no demand to mirror until a user asks.
- **Multi-fold ensembling** — possible follow-up. Current chorus + upstream Kundaje don't ensemble across folds; folds 1–4 stay as a `*-folds-extra` HF repo if anyone needs them later.

## Commit walk

| Commit | What |
|---|---|
| `62fd9b0` | docs: original plan doc for the slim mirror |
| `678dbd5` | slim-mirror Step 1 round-trip check (HepG2 + K562 DNase, bit-equal verified) |
| `802c7b1` | Slim mirror code + default flip + initial tests + 0.3.0 version bump |
| `a13282c` | F1 fix: huggingface_hub in env yaml + warn-on-import-fail + regression test + sei 0.4.0 plan |
| `fbe47a0` | v30 scorched-earth audit report + artifacts |
| `441adce` | F1 closure: end-to-end re-test (1m 24s vs 22m 44s) |

🤖 Generated with [Claude Code](https://claude.com/claude-code)
