# F1 (P0) — `huggingface_hub` missing in `chorus-chrombpnet` env, slim mirror silently disabled during prefetch

**Severity**: P0 — the headline v0.3.0 feature ("1.5 GB slim mirror replaces ~100 GB ENCODE tarballs") doesn't actually fire during the most common code path (`chorus setup`).
**Status**: open.
**Branch / commit**: `feat/chrombpnet-hf-slim-mirror` @ `802c7b1`.
**Discovered**: v30 scorched-earth audit, 2026-04-29.

## Symptom

After a clean `chorus setup --oracle all` against v0.3.0:

- `downloads/chrombpnet/` is **3.5 GB** (HepG2 + K562 ENCODE tarballs unpacked), unchanged from v0.2.x baseline.
- `~/.cache/huggingface/hub/` contains alphagenome and borzoi caches but **zero `chorus-chrombpnet-slim` entries** for the prefetched cell-types.
- ChromBPNet phase took ~22 min wall (env build 40s + tarball download/extract ~22 min). The slim path would have downloaded only the 25.6 MB `chrombpnet_nobias` h5 in seconds.
- Loaded model path resolves to `…/downloads/chrombpnet/DNASE_K562/models/fold_0/chrombpnet_nobias/chrombpnet_wo_bias` (ENCODE-tarball-extracted SavedModel), not an HF cache path.

## Root cause

`environments/chorus-chrombpnet.yml` does not list `huggingface_hub` as a dependency. The prefetch script runs inside the per-oracle env (`mamba run -n chorus-chrombpnet …`), so the import here:

```python
# chorus/oracles/chrombpnet.py:280
def _try_slim_hf_chrombpnet(self) -> Optional[Path]:
    if self.assay == "CHIP" or self.fold != 0:
        return None
    encff = CHROMBPNET_MODELS_DICT.get(self.assay, {}).get(self.cell_type)
    if not encff:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None    # ← silently swallows the import error
    …
```

…raises `ModuleNotFoundError`, which is caught and returns `None`. The caller at line 432–440 then falls back to the existing ENCODE tarball flow:

```python
slim_path = None
if not is_custom:
    if assay != "CHIP" and fold == 0 and model_type == "chrombpnet_nobias":
        slim_path = self._try_slim_hf_chrombpnet()    # → None
    elif assay == "CHIP":
        slim_path = self._try_slim_hf_bpnet()          # would also be None for CHIP
if slim_path is not None:
    self.model_path = str(slim_path)
else:
    # Existing ENCODE/JASPAR tarball flow
    self._download_chrombpnet_model()                  # ← what actually fires
    …
```

The flag `_try_slim_hf_bpnet` has the same gating, so once the `--all-chrombpnet` opt-in path fires for the 744 BPNet models, **none** of those would hit HF either — they'd all fall back to JASPAR's `mencius.uio.no` URLs.

## Direct verification

```text
$ mamba run -n chorus-chrombpnet python -c "import huggingface_hub; print(huggingface_hub.__version__)"
ModuleNotFoundError: No module named 'huggingface_hub'

$ mamba run -n chorus python -c "import huggingface_hub; print(huggingface_hub.__version__)"
1.12.2
```

Cross-env audit:

| env | `huggingface_hub` listed in yaml |
|---|---|
| chorus-alphagenome | ✓ |
| chorus-base | ✗ |
| chorus-borzoi | ✗ |
| chorus-chrombpnet | ✗ ← **breaks slim mirror** |
| chorus-enformer | ✗ |
| chorus-legnet | ✗ |
| chorus-sei | ✗ |

The base env (`chorus`) has it via `chorus`'s own `setup.py` install_requires, but the per-oracle envs only get whatever is listed in their yaml.

## Fix

Add `huggingface_hub` to `environments/chorus-chrombpnet.yml`:

```yaml
- pip:
    …
    - huggingface_hub>=0.20.0
```

The base `chorus` install brings `huggingface_hub` ≥ 0.20.0 transitively, so version pin should match.

For consistency and future-proofing, consider adding it to **all** per-oracle envs — any oracle code that wants to fetch from HF (slim mirror, future track CDFs, custom weight URLs) needs `huggingface_hub` in the same env where the oracle runs.

Optional secondary improvement (defence in depth): change the silent `except ImportError: return None` to log a warning so future "why isn't HF being used?" symptoms surface in the log:

```python
except ImportError:
    logger.warning(
        "huggingface_hub not available in this env; "
        "falling back to ENCODE tarball for ChromBPNet weights."
    )
    return None
```

## Re-test plan

After fixing the env yaml:

1. `mamba env remove -n chorus-chrombpnet -y && mamba run -n chorus chorus setup --oracle chrombpnet`
2. Expect `downloads/chrombpnet/` to **stay empty** (slim path used).
3. Expect `~/.cache/huggingface/hub/models--lucapinello--chorus-chrombpnet-slim/` to populate with `manifest.json` + the two fast-path h5's (~52 MB total).
4. Expect the chrombpnet phase wall-time to drop from ~22 min to **<1 min** (just env build + ~25 MB downloads).

## Impact on the v0.3.0 release claim

The CHANGELOG entry says:

> **HuggingFace slim mirror** at `lucapinello/chorus-chrombpnet-slim` containing only the artifacts chorus actually loads at inference time: 42 fold-0 ChromBPNet `chrombpnet_nobias` h5's (1,074 MB) + 744 BPNet/CHIP h5's (419 MB) = **1.49 GB total**. Replaces the previous ~100 GB ENCODE-tarball-based prefetch path.

This is true *for runtime* (a user calling `chorus.create_oracle('chrombpnet', use_environment=True).load_pretrained_model(...)` from the chorus base env) but **false for `chorus setup`** which is the path most users hit on first install. We should either:
- Fix the env (recommended — restores the headline behaviour), OR
- Soften the CHANGELOG to say "1.49 GB at runtime via the chorus base env; setup prefetch still pulls ENCODE tarballs by design" (worse — defeats the purpose).
