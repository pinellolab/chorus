# Fresh-Install Linux CUDA Audit — 2026-04-16

**Machine**: Linux x86_64, 2x NVIDIA A100 80 GB PCIe, CUDA available  
**Methodology**: scorched-earth — every chorus* env deleted, `~/.chorus` + `~/.cache/huggingface` wiped, fresh clone to SSD (`/srv/local/lp698/chorus-fresh-audit`)  
**Branch**: `chorus-applications` at commit 1f8de0c (includes all merged PRs through #10)  
**Duration**: ~8 hours (mostly env creation + chrombpnet background rebuild)

---

## 1. Fresh Install (following README verbatim)

| Step | Command | Result | Time |
|------|---------|--------|------|
| Clone | `git clone ... && git checkout chorus-applications` | OK | <1 min |
| Create env | `mamba env create -f environment.yml` | OK | ~25 min |
| pip install | `pip install -e .` | chorus 0.1.0 installed | <1 min |
| Kernel | `python -m ipykernel install --user --name chorus` | OK | <1 sec |
| HF auth | `hf auth login --token ...` | OK | <1 sec |
| Genome | `chorus genome download hg38` | OK → `genomes/hg38.fa` | ~5 min |
| Verify | `python -c "import chorus; print(chorus.__version__)"` | `0.1.0` | instant |

### Oracle Environment Setup (6 oracles)

| Oracle | Framework | Command | Result | Time |
|--------|-----------|---------|--------|------|
| enformer | TF 2.13 | `chorus setup --oracle enformer` | OK | ~20 min |
| alphagenome | JAX | `chorus setup --oracle alphagenome` | OK | ~25 min |
| chrombpnet | TF 2.8 | `chorus setup --oracle chrombpnet` | OK | ~15 min |
| borzoi | PyTorch | `chorus setup --oracle borzoi` | **FAIL x2** then OK | ~20 min |
| sei | PyTorch | `chorus setup --oracle sei` | OK | ~15 min |
| legnet | PyTorch | `chorus setup --oracle legnet` | OK | ~15 min |

**Borzoi/sei/legnet issue**: `conda.anaconda.org/pytorch` and `/nvidia` channels unreachable from Partners network (300s timeout). Fixed by adding `pytorch` and `nvidia` to `~/.condarc` `custom_multichannels` routing through `prefix.dev`. This is a network-specific issue; users on open networks won't hit it.

### GPU Verification

| Oracle | Framework | GPU Detected | Method |
|--------|-----------|-------------|--------|
| enformer | TF 2.13 | Yes (via env runner LD_LIBRARY_PATH) | `tf.config.list_physical_devices('GPU')` |
| alphagenome | JAX | Yes | `jax.devices() → CudaDevice(id=0)` |
| chrombpnet | TF 2.8 | Yes (via env runner LD_LIBRARY_PATH) | Auto-detected 1 GPU(s) in logs |
| borzoi | PyTorch | Yes | `torch.cuda.is_available() → True, A100 80GB` |
| sei | PyTorch | Yes | `torch.cuda.is_available() → True, A100 80GB` |
| legnet | PyTorch | Yes | `torch.cuda.is_available() → True, A100 80GB` |

---

## 2. Test Suite

**287/287 tests pass** (run twice — once before fixes, once after normalization fix cherry-pick)

```
================= 287 passed, 14 warnings in 684.15s (0:11:24) =================
```

Includes all 6 real-oracle smoke tests (each loads model, runs prediction, verifies output shape).

---

## 3. Notebooks

| Notebook | Cells | With Outputs | Errors | Size |
|----------|-------|-------------|--------|------|
| single_oracle_quickstart | 49 | 32 | **0** | 641 KB |
| comprehensive_oracle_showcase | 59 | 37 | **0** | 748 KB |
| advanced_multi_oracle_analysis | 127 | 44 | **0** | 2128 KB |

All executed end-to-end on the fresh clone with fresh envs.

---

## 4. MCP Tool Smoke Test

| Tool | Input | Result |
|------|-------|--------|
| `list_oracles` | — | 6 oracles, all envs installed |
| `load_oracle` | enformer, device=auto | Loaded in 20.5s, 5313 background tracks |
| `predict_variant_effect` | rs12740374 (chr1:109274968 G>T), ENCFF413AHU | abs_max_effect=0.393, activity_percentile=0.93 |
| `unload_oracle` | enformer | Unloaded OK |

---

## 5. Selenium HTML Screenshot Sweep

**14/16 render cleanly**, 2 expected:

| Status | Count | Detail |
|--------|-------|--------|
| OK | 14 | IGV browser loads, tracks visible, no JS errors |
| NO-IGV | 1 | `batch_scoring` — scoring table by design, no genome browser |
| JS-ERR | 1 | `causal_prioritization` — jsdelivr CDN blocked by Partners SSL inspection (ERR_CERT_AUTHORITY_INVALID). Pre-generated HTML still uses CDN `<script>` tag; new user-generated HTMLs will use self-contained igv.js per the `_igv_report.py` fix |

---

## 6. ChromBPNet Background Rebuild

Rebuilt all 24 tracks (ATAC + DNASE) from scratch on GPU 0:

| Phase | Models | Time | Output |
|-------|--------|------|--------|
| Variants (effect CDFs) | 24/24 | ~3.5 hours | `chrombpnet_effect_cdfs_interim.npz` |
| Baselines (activity CDFs) | 24/24 | ~2.5 hours | `chrombpnet_baseline_cdfs_interim.npz` |
| Merge | — | <1 min | `chrombpnet_pertrack.npz` (2.5 MB, 24 tracks) |

**Verification**: `DNASE:hindbrain` now returns `effect_percentile(0.0) = 0.0` (was falsely `1.0` due to zero-count CDF row).

**Uploaded** to `lucapinello/chorus-backgrounds` on HuggingFace.

---

## 7. Issues Found + Fixed During Audit

| # | Severity | Issue | Fix | Commit |
|---|----------|-------|-----|--------|
| 1 | **HIGH** | TF 2.14.1 hits XLA JIT crash on Enformer's `cross_replica_batch_norm/Rsqrt` | Pin `tensorflow==2.13.*` in `chorus-enformer.yml` | 5e19ae3 |
| 2 | **HIGH** | ChromBPNet `tarfile.extractall` races between concurrent callers (`FileExistsError`) | fcntl lock + fold_0 existence check in `chrombpnet.py` | 5e19ae3 |
| 3 | **HIGH** | ChromBPNet `DNASE:hindbrain` zero-count CDF row → false 100th percentile | Guard in `normalization.py` (returns None) + full NPZ rebuild | 5ebb328 + HF upload |
| 4 | **MEDIUM** | Pre-generated causal HTML uses jsdelivr CDN (blocked on corporate networks) | `_igv_report.py` auto-downloads + inlines igv.min.js (new reports only; existing examples need regeneration) | 5ebb328 |
| 5 | **MEDIUM** | `get_normalizer()` preferred legacy `.npy` over new per-track NPZ | Fixed in `normalization.py` (cherry-pick 6ceb817) | 1f8de0c |
| 6 | **LOW** | Jupyter kernel "chorus" missing after env wipe (README step 3 easy to skip) | Documented — `ipykernel install --user` required | — |
| 7 | **LOW** | `conda.anaconda.org/pytorch` unreachable on Partners network | Added `pytorch`/`nvidia` to `custom_multichannels` in `~/.condarc` routing through `prefix.dev` | User config (not code) |

---

## 8. Remaining Items (not blocking)

1. **Regenerate causal_prioritization HTML** with self-contained igv.js (requires GPU + `regenerate_remaining_examples.py`)
2. **Report resolved device in `load_oracle` response** — currently shows `null` for auto-detect; should show `cuda:0` or `cpu`
3. **Enformer GPU on TF 2.13**: works through env runner (LD_LIBRARY_PATH), but `device` field in `load_oracle` response doesn't reflect this

---

## Verdict

**Production-ready for beta release.** The fresh-install path works end-to-end on Linux CUDA:
- All 6 oracles install, load, and predict correctly
- 287/287 tests pass
- 3 notebooks execute with 0 errors
- MCP tools functional
- 14/16 HTML reports render (2 expected non-issues)
- Per-track normalization verified correct after ChromBPNet NPZ rebuild

The three HIGH issues found were all fixed and committed during this audit session.
