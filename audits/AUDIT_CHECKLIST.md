# Chorus Audit Checklist

A comprehensive, reusable checklist for future "ship-ready" audits of the Chorus library. Covers installation, docs, notebooks, shipped walkthrough reports (incl. IGV rendering), CDF/normalization correctness, GPU/device detection, HuggingFace auth, MCP server, and error-path quality.

Runbook convention: items that can be mechanised are called out with the exact command. Items that need human judgement are marked with **👁**. Severity tiers: **P0** blocks ship; **P1** fix before release; **P2** polish.

---

## 1. Installation & environment

- [ ] `environment.yml` parses and its channels resolve (`mamba env create -f environment.yml --dry-run` or a real create). **P0**
- [ ] `pip install -e .` completes from a fresh clone with no stderr errors. **P0**
- [ ] `chorus --help` lists every advertised subcommand; each subcommand's `--help` is non-empty. **P1**
- [ ] `chorus setup --oracle <name>` completes for at least one oracle on the target machine. Run this for **every** oracle on a Linux/CUDA host and on a macOS-arm64 host before a release. **P0**
- [ ] `EnvironmentManager.environment_exists('<name>')` returns `True` after setup for every oracle that got set up. **P1**
- [ ] Re-running `chorus setup --oracle X` on an existing env is idempotent (no double-install, no permission errors). **P2**
- [ ] `chorus genome download hg38` downloads to the expected path and the resulting FASTA is indexed (`.fai` present). **P0**
- [ ] `download_gencode(version='v48', annotation_type='basic')` pulls and caches the GTF. **P1**
- [ ] The cache paths are user-overridable via env vars where documented (`CHORUS_DOWNLOAD_DIR`, `CHORUS_NO_TIMEOUT`, `CHORUS_DEVICE`). **P2**
- [ ] `~/.chorus/backgrounds/` auto-downloads per-track NPZs on first use from `huggingface.co/datasets/lucapinello/chorus-backgrounds` (Sei + LegNet especially, which are not pre-built locally). **P1**

## 2. HuggingFace authentication (AlphaGenome gate)

- [ ] `HF_TOKEN` env var path works — AlphaGenome loads without raising. **P0**
- [ ] `huggingface-cli login` path works — AlphaGenome loads. **P1**
- [ ] No-token, no-login path raises a **single clear error** that names `HF_TOKEN`, the exact gated repo URL (`huggingface.co/google/alphagenome-all-folds`), and the `huggingface-cli login` alternative. **P0**
  - Covered by `tests/test_error_recovery.py::TestAuthFailurePaths::test_alphagenome_missing_hf_token_error`.
- [ ] The repo URL in **all** three code paths matches what the README tells users to accept:
  - `chorus/oracles/alphagenome.py` (direct load)
  - `chorus/oracles/alphagenome_source/templates/load_template.py` (env-runner load)
  - `README.md` / `environments/README.md` (doc)
- [ ] `list_tracks('alphagenome')` works **without** an HF token (metadata is cached / bundled; only weights are gated).
- [ ] User whose `HF_TOKEN` is only in `~/.zshrc` gets a clear hint that they may need to `export` it in the shell that starts `claude`. **P2**

## 3. GPU / device detection

**Base env (`chorus`):**
```
mamba run -n chorus python -c 'from chorus.core.platform import detect_platform; p=detect_platform(); print(p.key, p.has_cuda)'
```
Expect `macos_arm64 False`, `linux_x86_64_cuda True`, or `linux_x86_64 False` per host.

**Per-oracle probe** (run on the release host):
```
for env in chorus-enformer chorus-borzoi chorus-chrombpnet chorus-sei chorus-legnet chorus-alphagenome; do
  mamba run -n "$env" python -c '
try: import torch; print("torch cuda:", torch.cuda.is_available(), "mps:", torch.backends.mps.is_available())
except ImportError: pass
try: import tensorflow as tf; print("tf devs:", [d.name for d in tf.config.list_physical_devices()])
except ImportError: pass
try: import jax; print("jax devs:", [str(d) for d in jax.devices()])
except ImportError: pass
'
done
```

- [ ] Enformer & ChromBPNet envs detect a GPU device on Linux/CUDA (and Metal on macOS). **P0**
- [ ] Borzoi, Sei, LegNet (PyTorch) return `cuda: True` on Linux, `mps: True` on macOS. **P0**
- [ ] AlphaGenome (JAX) prints a non-empty device list. **P0**
- [ ] No oracle pins to `cuda:0` in code — should default to `'cuda'` so `CUDA_VISIBLE_DEVICES` is respected. Confirm via
  `grep -rn "cuda:0'" chorus/oracles/ chorus/oracles/*/templates/` returns only docstring examples, not live defaults. **P1**
- [ ] `CHORUS_DEVICE=cpu` forces CPU even if a GPU is visible. **P2**

## 4. Per-track CDF / normalization

```python
import numpy as np
from chorus.analysis.normalization import get_normalizer
for name in ['enformer', 'borzoi', 'chrombpnet', 'sei', 'legnet', 'alphagenome']:
    nz = get_normalizer(name)
    entry = nz._loaded[name]
    ecdf, scdf = entry.get('effect_cdfs'), entry.get('summary_cdfs')
    assert all(np.all(np.diff(ecdf[i]) >= -1e-9) for i in range(min(10, ecdf.shape[0]))), name
    n_pts = scdf.shape[1]
    for i in range(min(10, scdf.shape[0])):
        assert scdf[i, int(.5*n_pts)] <= scdf[i, int(.95*n_pts)] + 1e-9 <= scdf[i, int(.99*n_pts)] + 2e-9
```

- [ ] All 6 oracles load via `get_normalizer(oracle_name)` without `None`. **P0**
- [ ] Every `effect_cdfs` row is **monotonically non-decreasing** (sorted). **P0**
- [ ] Every `summary_cdfs` row satisfies `p50 ≤ p95 ≤ p99`. **P0**
- [ ] `signed_flags` matches the oracle's nature:
  - enformer/chrombpnet: 0% signed
  - borzoi: ~20% signed (RNA strands)
  - sei: 100% signed
  - legnet: 100% signed (MPRA = Δ)
  - alphagenome: ~13% signed
- [ ] Track counts match published specs: 5,313 / 7,611 / 24 / 40 / 3 / 5,168. **P1**
- [ ] `perbin_cdfs` present for Enformer/Borzoi/ChromBPNet/AlphaGenome (scalar oracles Sei + LegNet omit it by design). **P1**
- [ ] Cache dir `~/.chorus/backgrounds/` is the canonical location (no per-project duplication). **P2**

## 5. Python API sanity

- [ ] `chorus.create_oracle('<name>', use_environment=False)` succeeds for all 6 names; invalid name gives `ValueError` that names the valid options. **P0**
- [ ] `create_oracle(...).sequence_length` matches the README hardware matrix: Enformer 393,216, Borzoi 524,288, ChromBPNet 2,114, Sei 4,096, LegNet 200, AlphaGenome 1,048,576. **P0**
- [ ] `oracle.predict(...)` without a model raises `ModelNotLoadedError` with a helpful message. **P1**
- [ ] `oracle.predict(('chrZZ', 1, 100000), [...])` on a bad chromosome raises a clear error (not a low-level KeyError). **P1**
- [ ] `predict_variant_effect` does **not** warn `Provided reference allele … does not match the genome at this position` for correctly-provided dbSNP/UCSC 1-based alleles. Regression test: `tests/test_prediction_methods.py::test_variant_position_is_1_based`. **P0**
- [ ] `predict_variant_effect` **does** still warn when the user's ref allele genuinely differs from the genome base. **P0**
- [ ] `extract_sequence('chr1:109274968-109274968')` returns `'G'` (rs12740374 SORT1), `'T'` for rs1421085 (FTO chr16:53767042), etc. Tie notebook examples to real dbSNP coordinates. **P1**
- [ ] `oracle.fine_tune(...)` on Sei/LegNet raises `NotImplementedError` with a message pointing at AlphaGenome/Borzoi for on-the-fly track adaptation. **P2**

## 6. Notebooks — cell-by-cell fresh execution

```
mamba run -n chorus jupyter nbconvert --to notebook --execute \
  examples/notebooks/single_oracle_quickstart.ipynb \
  --output /tmp/fresh.ipynb --ExecutePreprocessor.timeout=600
```

For each of `single_oracle_quickstart.ipynb`, `comprehensive_oracle_showcase.ipynb`, `advanced_multi_oracle_analysis.ipynb`:

- [ ] Fresh execution exit code 0 — **every cell completes**. **P0**
- [ ] **Zero errors** and **zero WARNING** lines in any cell output. **P1**
- [ ] Track counts printed by each oracle's `list_assay_types()` / `get_track_info()` match the README hardware matrix. **P1**
- [ ] Numbers in narrative cells (markdown) either exactly match the execution output or sit within the ±0.006 CPU non-determinism band. **P1**
- [ ] Every markdown link resolves in the committed repo (`[text](path)` targets exist; cross-notebook links correct; `applications/` never appears in live docs). **P1**
- [ ] No `/srv/local/<user>/...` or other machine-specific absolute paths appear in documented example commands (shipped **output** may contain them — cosmetic). **P2**
- [ ] Notebooks are committed with cleared metadata that doesn't leak the author's kernel path. **P2**

## 7. Shipped HTML reports — visual rendering + content

```python
# Render with full JS (selenium) so IGV actually loads — headless Chrome alone gives a placeholder.
from selenium import webdriver
opts = webdriver.ChromeOptions()
opts.add_argument('--headless=new')
opts.add_argument('--disable-gpu')
opts.add_argument('--allow-file-access-from-files')
opts.add_argument('--window-size=1600,4500')
opts.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
driver = webdriver.Chrome(options=opts)
driver.get(f'file://{html}')
time.sleep(12)  # let CDN JS load
driver.save_screenshot(f'/tmp/{name}.png')
errs = [l for l in driver.get_log('browser') if l['level'] in ('SEVERE','ERROR')]
```

For each `examples/walkthroughs/**/*.html`:

- [ ] Renders at 1600×4500 without JS errors in the browser console. **P0**
- [ ] IGV browser block shows real signal tracks (not just the placeholder text). **P0**
- [ ] Glossary block present with log2FC/lnFC/Δ formula legend. **P1**
- [ ] Every per-layer table has: Track · Cell Type · Ref · Alt · Effect [formula badge] · Ref%ile · Activity%ile · Interpretation. **P1**
- [ ] Formula badges match layer: log2FC on chromatin/TF/histone/TSS, lnFC on RNA-seq/CAGE gene expression, Δ (alt−ref) on MPRA. **P0**
- [ ] Cell-type column doesn't duplicate text already in the track label (e.g. `CHIP:CEBPA:HepG2 · HepG2` is a known regression). **P1**
- [ ] Consensus matrix (multi-oracle reports only) uses single-voter `n=1` labels correctly. **P1**
- [ ] "How to read this report" collapsible defines every numeric column. **P2**
- [ ] 👁 The Interpretation badge ("Strong opening", "Moderate binding gain", etc.) is consistent with the sign and magnitude of the effect and the assay convention.
- [ ] Every `README.md` number in the same walkthrough dir is within ±0.006 of the `example_output.md` it's derived from. **P1**

## 8. MCP server

- [ ] `chorus-mcp` subprocess starts cleanly on stdio.
- [ ] `list_oracles` returns exactly 6 oracles with spec fields matching the Python API (`sequence_length`, assay types, resolution). **P0**
- [ ] Exactly 22 tools registered via FastMCP (`mcp._list_tools()`). **P1**
- [ ] MCP tool count matches what walkthrough READMEs & `docs/MCP_WALKTHROUGH.md` advertise.
- [ ] System-prompt instructions in `chorus/mcp/server.py` are in sync with real specs (track counts, assay names, recommended oracle per task). **P1**
- [ ] `analyze_variant_multilayer` end-to-end: spawn `chorus-mcp`, connect with `fastmcp.Client`, run rs12740374 against AlphaGenome HepG2 tracks, assert the returned dict shape matches what walkthroughs document. (Integration-marked; run on release host.) **P1**
- [ ] Error paths in MCP tools surface `{"error": ..., "error_type": ..., "tool": ...}` — not raw tracebacks. **P1**

## 9. Error messages — first-user friendliness

Trigger and inspect each:

- [ ] `create_oracle('fakeOracle')` → names the valid options.
- [ ] `predict(...)` pre-load → `ModelNotLoadedError` with the fix hint.
- [ ] Missing reference_fasta → names the kwarg and `chorus genome download hg38`.
- [ ] Missing oracle env → logs `Run chorus setup --oracle <name>` hint and downgrades to `use_environment=False` (graceful degradation — **not** a crash). Regression: `tests/test_error_recovery.py::test_missing_oracle_env_falls_back_gracefully`.
- [ ] HF token missing (AlphaGenome) → names `HF_TOKEN`, the exact gated repo URL, and the `huggingface-cli login` alternative.
- [ ] Network drop during `download_pertrack_backgrounds` → returns 0 and logs a warning, does not raise. Regression: `tests/test_error_recovery.py::test_hf_hub_download_failure_returns_zero`.

## 10. Consistency of claims across the repo

Repo-wide drift grep — any match should be investigated:

```
grep -rn '5,930\|5930\|196 kbp\|examples/applications/' --include='*.md' --include='*.py' --include='*.ipynb' .
grep -rn '7,612' scripts/ examples/ --include='*.md'
grep -rn 'LegNet.*230 bp\|input_size_bp.*230' chorus/ scripts/ --include='*.py' --include='*.md'
```

- [ ] Canonical numbers: **AlphaGenome 5,731** / **Enformer 5,313** / **Borzoi 7,611** / **Sei 21,907** (total) but 40 CDF-backed classes / **LegNet 200 bp input, 3 CDFs** / **ChromBPNet 24 CDFs per-model**. No doc may disagree. **P1**
- [ ] Formula conventions documented **once** and cited by every report/notebook: `log2FC` (default), `lnFC` (gene expression), `Δ (alt−ref)` (MPRA). **P1**
- [ ] Directory naming: live docs only reference `examples/walkthroughs/` and `examples/notebooks/`. The old `examples/applications/` path must only appear in `audits/` historical snapshots. **P0**
- [ ] README "Hardware matrix per oracle" section is in sync with `chorus/mcp/server.py::ORACLE_SPECS`. **P1**
- [ ] 👁 No "TODO", "coming soon", "WIP" markers in live docs (`audits/` and git history excluded).

## 11. Test suite

```
mamba run -n chorus python -m pytest tests/ --ignore=tests/test_smoke_predict.py -q
```

- [ ] Fast suite ≥ 334 pass, ≤ 1 skip, 0 error (smoke test skipped by design).
- [ ] `pytest -m integration` on a release host: SEI/LegNet CDF download, ChromBPNet fresh model download, MCP E2E all pass.
- [ ] CI workflow at `.github/workflows/tests.yml` runs green on the PR.
- [ ] Coverage of new code paths: any new oracle / normalizer / tool needs its own test.

## 12. Reproducibility

- [ ] Regen scripts in `scripts/` produce outputs byte-identical (or within ±0.006) to committed walkthroughs when given the same inputs. **P1**
- [ ] `scripts/regenerate_multioracle.py --consolidate` is idempotent and picks up fresh per-oracle JSONs. **P1**
- [ ] Reference-genome + annotation files can be reproduced by re-running the documented `chorus genome download` / `download_gencode` calls. **P2**

---

## Appendix — artefacts to produce per audit

A full audit should leave behind, in `audits/YYYY-MM-DD_vNN_<label>/`:

- `report.md` — findings summary (one bullet per finding: file:line, problem, fix).
- `screenshots/*.png` — one per shipped HTML (selenium-rendered, 1600×4500).
- `nb_fresh_output/*.ipynb` — fresh re-execution of every notebook.
- `cdf_check.txt` — output of the CDF-sanity script from §4.
- `device_probe.txt` — output of the per-env GPU probe from §3.

These let the next auditor diff your findings against theirs mechanically.
