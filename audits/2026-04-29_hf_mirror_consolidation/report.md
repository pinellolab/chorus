# HF mirror consolidation — chorus-controlled mirrors for Enformer, Borzoi, Sei, LegNet

**Date**: 2026-04-29
**Branch**: `feat/hf-mirror-consolidation` → PR [#68](https://github.com/pinellolab/chorus/pull/68)
**Trigger**: chorus's install path was depending on third-party hosts that have shown lifecycle volatility — most pressingly TFHub (deprecated by Google in 2024, redirecting to Kaggle Models). Plus `johahi/borzoi-replicate-*` and Zenodo single-link weights. User asked to consolidate weight provenance under the chorus org for resilience.

## Inventory after consolidation

| Oracle | Mirror (chorus) | Original source | Size | Verified |
|---|---|---|---|---|
| **Enformer** | [`lucapinello/chorus-enformer`](https://huggingface.co/lucapinello/chorus-enformer) | TFHub `deepmind/enformer/1` (now redirects to Kaggle) | 961 MB | ✅ load round-trip on macOS Metal |
| **Borzoi** | [`lucapinello/chorus-borzoi`](https://huggingface.co/lucapinello/chorus-borzoi) (4 folds) | [`johahi/borzoi-replicate-{0..3}`](https://huggingface.co/johahi/borzoi-replicate-0) | ~6 GB | ✅ upload round-trip |
| **Sei** | [`lucapinello/chorus-sei`](https://huggingface.co/lucapinello/chorus-sei) | Zenodo [4906997](https://zenodo.org/record/4906997) | 3,281,639,255 bytes | ✅ md5 `4297aafb711aec4ecccb645b8928ea26` matches Zenodo metadata |
| **LegNet** | [`lucapinello/chorus-legnet`](https://huggingface.co/lucapinello/chorus-legnet) | Zenodo [17863550](https://zenodo.org/records/17863550) (chorus-pinned) | ~38 MB | ✅ size match |
| **ChromBPNet** (default fold-0 nobias) | [`lucapinello/chorus-chrombpnet-slim`](https://huggingface.co/lucapinello/chorus-chrombpnet-slim) | ENCODE per-experiment tarballs | 1.49 GB (786 h5's) | already shipped in 0.3.0 |
| **AlphaGenome JAX** | `google/alphagenome-all-folds` (gated, official) | Google DeepMind | ~700 MB sharded | not mirrored — official source |
| **AlphaGenome PT** | `gtca/alphagenome_pytorch` (third-party, public) | Conversion of JAX checkpoint | 878 MB | not mirrored — upstream port maintainer; mirroring would risk weight drift if upstream updates |
| **Per-track CDFs** | [`lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds) | chorus-built | ~2 GB across 6 oracles | already shipped |

## What's mirrored vs not

**Mirrored** (chorus-controlled durability):
- Enformer / Borzoi / Sei / LegNet weights — under `lucapinello/chorus-*`
- ChromBPNet slim mirror — under `lucapinello/chorus-chrombpnet-slim` (predates this consolidation)
- Per-track CDFs — under `lucapinello/chorus-backgrounds` (predates this consolidation)

**Deliberately NOT mirrored**:
- **AlphaGenome JAX** (`google/alphagenome-all-folds`): Google DeepMind is the authoritative source; the model is gated under non-commercial terms and chorus shouldn't host a copy that bypasses Google's gating mechanism.
- **AlphaGenome PT** (`gtca/alphagenome_pytorch`): the upstream port author (gtca) maintains the conversion and may update it. Mirroring under chorus would risk users on stale weights when the port upgrades. The non-commercial terms still apply regardless of mirror — see PR #62 README.

## Loader pattern (every chorus mirror)

Each oracle's loader prefers the chorus mirror and falls back to the original source on any failure:

```python
def _try_hf_mirror(...):
    try:
        from huggingface_hub import hf_hub_download   # or snapshot_download
        local = hf_hub_download(repo_id="lucapinello/chorus-<oracle>", filename=...)
        # use local
        return True
    except ImportError:
        # huggingface_hub not in env — log + fall back
        return False
    except Exception as exc:
        # network / repo missing / HF unreachable — log + fall back
        return False

def _download_<oracle>_model(self):
    if not self._try_hf_mirror(...):
        # Fall back to existing TFHub / johahi / Zenodo / etc. path
        ...
```

This means:
- Happy path: chorus mirror is fast + chorus-stable.
- HF-down: fall back to original source. No breakage.
- `huggingface_hub` not installed: fall back. Older installs predating this PR continue to work via the original source.

## Attribution + licensing

Each mirror README explicitly identifies four things separately:

1. **Where the weights came from** (original source URL + paper citation)
2. **Who owns the weights** (upstream lab / company)
3. **Which model terms apply to the weights regardless of mirror** (the mirror does NOT override license terms — e.g. Sei stays CC-BY-NC even on HF)
4. **Which code license applies to the chorus loader** (chorus's Apache 2.0)

This format follows the [`gtca/alphagenome_pytorch`](https://huggingface.co/gtca/alphagenome_pytorch) attribution template the user pointed to. Concrete language used per mirror:

> The weights were copied from [original source]. Those weights were created by [upstream] and are the property of [upstream owner]. The model parameters, outputs, and any derivatives thereof remain subject to [upstream license/terms]. The chorus loader code is released under [chorus license]. These terms are consistent with the terms for the reference code and the model weights.

## Issues encountered + resolutions

### 1. Sei truncated download (1× retry needed)

First curl of the Sei tarball completed with exit 0 but only 363 MB landed (vs 3.28 GB expected). Curl didn't detect Zenodo's mid-stream connection close as a failure. Caught when the user noted the size felt small.

**Fix**: re-download with strict size + md5 verification against Zenodo's API metadata (`https://zenodo.org/api/records/4906997` exposes `size` and `checksum: md5:...`). The second download (using the API content URL `/files/.../content` instead of the public file URL) landed clean.

**Note**: a `curl -C -` resume attempt in between produced a corrupt 3.29 GB file (8.5 MB longer than expected, totally different md5) — likely byte-overlap from the resume offset. **Don't trust `curl -C -` resumes for long-running mid-stream-truncated downloads** without re-verifying. Better path: delete and re-download fresh, then verify.

### 2. `huggingface_hub` not in older chrombpnet env (CDF build path)

The CDF rebuild script falls back to ENCODE tarballs (~700 MB per model × 786 models = 100s of GB) instead of the slim mirror (25 MB per model) when `huggingface_hub` isn't importable. PR #60 added it to `chorus-chrombpnet.yml`, but envs created before that PR don't have it.

**Fix in HANDOFF.md** (PR #67): added an explicit `pip install` step at the top of the CUDA-box runbook so older envs work. Caught this on the M3 Ultra during a sanity run before handing off.

### 3. Duplicate file detection on HF re-upload

When re-uploading the corrected Sei tarball, the `HfApi.upload_file` reported "Skipping to prevent empty commit" — initially confusing because the file content had changed. Verified via `repo_info(files_metadata=True)` that the actual file size on HF matched the new 3.28 GB (HF's CAS dedup pool just happened to have the new content cached).

## Time + bandwidth

Total upload to HF over residential connection:
- Enformer: 16 s for 1 GB
- LegNet: 6 s for 38 MB
- Borzoi: 114 s for 6 GB across 4 folds
- Sei: ~6 s for 3.28 GB (HF's CAS dedup made this fast)

Total mirror size on HF: ~11.3 GB across 4 new repos.

## Follow-ups

- **CDF rebuild on CUDA box** — separate workflow, in flight on the user's lab GPU, gated by the CUDA agent. Output is a fresh `chrombpnet_pertrack.npz` rebuild against `chrombpnet_nobias` (the 0.3.0 default), uploaded to the existing `lucapinello/chorus-backgrounds` dataset.
- **Smoke-test Borzoi / LegNet HF-first paths** locally — Enformer was verified end-to-end via `tf.saved_model.load`; the other two have only the upload round-trip and the fall-back path verified. Adding integration smoke tests is a small follow-up if the equivalence-test infra is reused.
- **Document mirror pattern in CONTRIBUTING / docs** — the HF-first / original-fallback pattern should be the standard for any future weight-fetching code.

## Critical files modified in PR #68

- `chorus/oracles/enformer.py` — `default_model_path` flipped to HF repo id
- `chorus/oracles/enformer_source/templates/{load,predict}_template.py` — HF path detection + fallback
- `chorus/oracles/borzoi_source/templates/{load,predict}_template.py` — `_load_borzoi(fold)` helper
- `chorus/oracles/sei.py` — `_try_hf_mirror` helper
- `chorus/oracles/legnet.py` — `_try_hf_mirror` helper
- `environments/chorus-{enformer,borzoi,sei,legnet}.yml` — `huggingface_hub>=0.20`
