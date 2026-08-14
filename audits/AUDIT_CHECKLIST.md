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
- [ ] The cache paths are user-overridable via env vars where documented (`CHORUS_NO_TIMEOUT`, `CHORUS_DEVICE`, `CHORUS_BACKGROUNDS_REPO`). Note there is **no** download-dir override: `CHORUS_ROOT` is derived from the package location and `annotations/`, `downloads/`, `genomes/` are created under it unconditionally (`chorus/core/globals.py`). **P2**
- [ ] `<data-dir>/backgrounds/` auto-downloads per-track NPZs on first use from `huggingface.co/datasets/lucapinello/chorus-backgrounds` 8 NPZs ship: alphagenome, borzoi, cherimoya, chrombpnet, enformer, epinformerseq, legnet, sei. **P1**

## 2. HuggingFace authentication (AlphaGenome gate)

- [ ] `HF_TOKEN` env var path works — AlphaGenome loads without raising. **P0**
- [ ] `huggingface-cli login` path works — AlphaGenome loads. **P1**
- [ ] No-token, no-login path raises a **single clear error** that names `HF_TOKEN`, the exact gated repo URL (`huggingface.co/google/alphagenome-all-folds`), and the `huggingface-cli login` alternative. **P0**
  - Covered by `tests/test_error_recovery.py::TestAuthFailurePaths::test_alphagenome_missing_hf_token_error_is_actionable`.
- [ ] The repo URL in **all** three code paths matches what the README tells users to accept:
  - `chorus/oracles/alphagenome.py` (direct load)
  - `chorus/oracles/alphagenome_source/templates/load_template.py` (env-runner load)
  - `README.md` / `environments/README.md` (doc)
- [ ] `list_tracks('alphagenome')` works **without** an HF token (metadata is cached / bundled; only weights are gated). Note this is the **MCP tool**, not a Python API — `chorus.list_tracks` does not exist, so call it through `chorus.mcp.server` or the MCP client. Verified 2026-08-12: 331 K562 hits with no token in the environment.
- [ ] User whose `HF_TOKEN` is only in `~/.zshrc` gets a clear hint that they may need to `export` it in the shell that starts `claude`. **P2**

## 3. GPU / device detection

**Base env (`chorus`):**
```
mamba run -n chorus python -c 'from chorus.core.platform import detect_platform; p=detect_platform(); print(p.key, p.has_cuda)'
```
Expect `macos_arm64 False`, `linux_x86_64_cuda True`, or `linux_x86_64 False` per host.

**Per-oracle probe** (run on the release host):
```
for env in chorus-enformer chorus-borzoi chorus-chrombpnet chorus-cherimoya \
           chorus-epinformerseq chorus-sei chorus-legnet \
           chorus-alphagenome chorus-alphagenome_pt; do
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
  - ⚠ The bare probe above reports **CPU-only** for both TF envs even on a healthy CUDA host: their CUDA libs live in `nvidia-*-cu11` pip wheels that only `EnvironmentRunner._prepare_env` puts on `LD_LIBRARY_PATH`. Route the probe through chorus (or export `LD_LIBRARY_PATH` yourself) or you cannot distinguish "no GPU support" from "GPU only on the runner path".
- [ ] Borzoi, Sei, LegNet (PyTorch) return `cuda: True` on Linux, `mps: True` on macOS. **P0**
- [ ] AlphaGenome (JAX) prints a non-empty device list. **P0**
- [ ] No oracle pins to `cuda:0` in code — should default to `'cuda'` so `CUDA_VISIBLE_DEVICES` is respected. Confirm via
  `grep -rn "cuda:0'" chorus/oracles/ chorus/oracles/*/templates/` returns only docstring examples, not live defaults. **P1**
- [ ] Passing `device='cuda:N'` must not *replace* an outer `CUDA_VISIBLE_DEVICES` mask. The TF and Cherimoya templates assign `os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]`, so a bare ordinal overrides the scheduler's mask and lands on a GPU the caller was not granted. **P1**
- [ ] `CHORUS_DEVICE=cpu` forces CPU even if a GPU is visible. **P2**

## 4. Per-track CDF / normalization

```python
import numpy as np
from chorus.analysis.normalization import get_normalizer
for name in ['alphagenome', 'borzoi', 'cherimoya', 'chrombpnet',
             'enformer', 'epinformerseq', 'legnet', 'sei']:
    nz = get_normalizer(name)
    entry = nz._loaded[name]
    ecdf, scdf = entry.get('effect_cdfs'), entry.get('summary_cdfs')
    assert all(np.all(np.diff(ecdf[i]) >= -1e-9) for i in range(min(10, ecdf.shape[0]))), name
    n_pts = scdf.shape[1]
    for i in range(min(10, scdf.shape[0])):
        assert scdf[i, int(.5*n_pts)] <= scdf[i, int(.95*n_pts)] + 1e-9 <= scdf[i, int(.99*n_pts)] + 2e-9
```

- [ ] All 8 oracles with shipped backgrounds load via `get_normalizer(oracle_name)` without `None`. **P0**
- [ ] Every `effect_cdfs` row is **monotonically non-decreasing** (sorted). **P0**
- [ ] Every `summary_cdfs` row satisfies `p50 ≤ p95 ≤ p99`. **P0**
- [ ] `signed_flags` matches the oracle's nature:
  - enformer / chrombpnet / cherimoya / epinformerseq: 0% signed
  - borzoi: ~20% signed (RNA strands) — measured 20.3%
  - sei: 100% signed
  - legnet: 100% signed (MPRA = Δ)
  - alphagenome: ~13% signed — measured 12.9%
- [ ] Track counts match published specs, measured from the NPZs: enformer 5,313 / borzoi 7,611 / **chrombpnet 753** (9 human ATAC-DNASE + 744 CHIP) / sei 40 / legnet 3 / alphagenome 5,168 / cherimoya 1,518 / epinformerseq 33. **P1**
- [ ] `perbin_cdfs` present for Enformer / Borzoi / ChromBPNet / AlphaGenome / **Cherimoya**; the scalar-output oracles Sei, LegNet and **EPInformer-seq** omit it by design. **P1**
- [ ] Cache dir `<data-dir>/backgrounds/` is the canonical location (no per-project duplication). **P2**
- [ ] **ChromBPNet count inversion is `expm1`, not `exp`, on both sides.** The count head predicts `log(1 + count)`; the oracle and the CDF builder must agree or every ChromBPNet percentile is silently wrong. Regression: `tests/test_chrombpnet_counts.py` (incl. the builder↔oracle consistency assertion). **P0**
- [ ] The `summary`/`perbin` CDFs of ChromBPNet and Cherimoya legitimately contain a few small **negative** entries (a near-dead window gives `log(count+1) < 0`, so `expm1 < 0`); they are left unclamped so builder and `predict()` agree. Do not "fix" them. The `effect` CDFs have none. **P1**

## 5. Python API sanity

- [ ] `chorus.create_oracle('<name>', use_environment=False)` succeeds for all **9** registered names (alphagenome, alphagenome_pt, borzoi, cherimoya, chrombpnet, enformer, epinformerseq, legnet, sei); invalid name gives `ValueError` that names the valid options. **P0**
- [ ] `create_oracle(...).sequence_length` matches the README hardware matrix: Enformer 393,216, Borzoi 524,288, ChromBPNet 2,114, Cherimoya 2,114, EPInformer-seq 2,114, Sei 4,096, LegNet 200, AlphaGenome 1,048,576, AlphaGenome-PT 1,048,576. **P0**
- [ ] `oracle.predict(...)` without a model raises `ModelNotLoadedError` with a helpful message. **P1**
- [ ] `oracle.predict(('chrZZ', 1, 100000), [...])` on a bad chromosome raises a clear error (not a low-level KeyError). **P1**
- [ ] `predict_variant_effect` does **not** warn `Provided reference allele … does not match the genome at this position` for correctly-provided dbSNP/UCSC 1-based alleles. Regression test: `tests/test_prediction_methods.py::TestPredictionMethods::test_variant_position_is_1_based`. **P0**
- [ ] `predict_variant_effect` **does** still warn when the user's ref allele genuinely differs from the genome base. **P0**
- [ ] **The direct path is exercised, not just the env path.** `use_environment=False` is the `create_oracle` default, yet it had no fast-suite coverage — which is why the Enformer track-routing and Borzoi interval/`.numpy()` P0s (#115, #116) survived green CI. Every oracle needs at least one direct-path predict test. **P0**
- [ ] `extract_sequence('chr1:109274968-109274968')` returns `'G'` (rs12740374 SORT1), `'T'` for rs1421085 (FTO chr16:53767042), etc. Tie notebook examples to real dbSNP coordinates. **P1**
- [ ] `oracle.fine_tune(...)` raises `NotImplementedError`. Note **no** oracle implements it — `borzoi.py` and `alphagenome.py` both say "Fine-tuning is not yet implemented" — so a message pointing the user at AlphaGenome/Borzoi is itself misleading and should be reworded. **P2**

## 6. Notebooks — cell-by-cell fresh execution

The notebooks declare kernelspec `chorus`, which **`chorus setup` does not
register** — without this first step every `nbconvert` invocation below dies
with `NoSuchKernel` (see `examples/notebooks/README.md:45`):

```
mamba run -n chorus python -m ipykernel install --user --name chorus \
  --display-name "Python 3 (chorus)"

mamba run -n chorus jupyter nbconvert --to notebook --execute \
  examples/notebooks/single_oracle_quickstart.ipynb \
  --output /tmp/fresh.ipynb --ExecutePreprocessor.timeout=600
```

`examples/notebooks/` now ships **6** notebooks (the three below plus `cherimoya_quickstart`, `epinformerseq_testing`, `klf1_validated_enhancer_profiles`), and `examples/walkthroughs/` ships 13 more `notebook.ipynb`. At minimum, for each of `single_oracle_quickstart.ipynb`, `comprehensive_oracle_showcase.ipynb`, `advanced_multi_oracle_analysis.ipynb`:

- [ ] Fresh execution exit code 0 — **every cell completes**. **P0**
- [ ] **Zero errors** and **zero WARNING** lines in any cell output. **P1**
- [ ] Track counts printed by each oracle's `list_assay_types()` / `get_track_info()` match the README hardware matrix. **P1**
- [ ] Numbers in narrative cells (markdown) either exactly match the execution output or sit within the ±0.006 CPU non-determinism band. **P1**
- [ ] Every markdown link resolves in the committed repo (`[text](path)` targets exist; cross-notebook links correct; `applications/` never appears in live docs). **P1**
- [ ] No `/srv/local/<user>/...` or other machine-specific absolute paths appear in documented example commands (shipped **output** may contain them — cosmetic). **P2**
- [ ] Notebooks are committed with cleared metadata that doesn't leak the author's kernel path. **P2**

## 7. Shipped HTML reports — visual rendering + content

`selenium` and a Chrome/chromedriver binary are **not** declared in
`environment.yml` or any `environments/*.yml`, so this method does not work on
a documented install — install them explicitly first, or fall back to the
structural checks below (well-formed document, vendored `igv.min.js` present,
parsed IGV config carries real features).

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
- [ ] The **"How to read this report"** block is present with the log2FC/lnFC/Δ formula legend (rendered by `chorus/analysis/_report_glossary.py::render_how_to_read`; there is no element literally named "Glossary"). It intentionally lists only the formulas for layers present, so `lnFC` is absent from reports with no RNA layer. **P1**
- [ ] Every per-layer table has: Track · Cell Type · Ref · Alt · Effect [formula badge] · **Effect %ile** · Activity %ile · Interpretation. (The column is `Effect %ile`; a literal grep for `Ref %ile` returns nothing — the reference-signal percentile is `Activity %ile`.) **P1**
- [ ] Formula badges match layer: log2FC on chromatin/TF/histone/TSS, lnFC on RNA-seq/CAGE gene expression, Δ (alt−ref) on MPRA. **P0**
- [ ] Cell-type column doesn't duplicate text already in the track label (e.g. `CHIP:CEBPA:HepG2 · HepG2` is a known regression). **P1**
- [ ] The cross-oracle consensus section (multi-oracle reports only; the `h2` is "Cross-oracle consensus", not "Consensus matrix") uses single-voter `n=1` labels correctly. **P1**
- [ ] "How to read this report" defines every numeric column. It is an always-expanded `<section>`, **not** a collapsible, and it does not define the plain Ref / Alt value columns. Either implement the collapsible or drop the word. **P2** — ⚠ do **not** test this with `grep -c '<details'`: the multi-oracle report legitimately uses four `<details class='oracle-block'>` wrappers, one per oracle, so a zero-count assertion fails on a correct artefact. Scope the check to the how-to-read block.
- [ ] 👁 The Interpretation badge ("Strong opening", "Moderate binding gain", etc.) is consistent with the sign and magnitude of the effect and the assay convention.
- [ ] Every `README.md` number in the same walkthrough dir is within ±0.006 of the `example_output.md` it's derived from. **P1**
- [ ] **Report size is pushable.** `rs12740374_SORT1_legnet_report.html` and the consolidated multi-oracle report embed locus-wide 1-bp IGV arrays; with LegNet tiled per #99 they reach 137 MB / 145 MB, above GitHub's 100 MiB file limit, so the artefact cannot be committed at all. Check `find examples -name '*.html' -size +50M` is empty before regenerating. **P0** — now also enforced automatically by `test_no_tracked_example_artefact_is_oversized`, so this box is a backstop rather than the only check.
- [ ] **Every committed report still paints.** `pytest tests/test_committed_reports_render_in_a_browser.py -m integration` — opens all 19 in headless Chromium and asserts every canvas has ink, no console errors, no uncaught exceptions. Needs `playwright install chromium` plus the `chorus-browsertest` env for Chromium's shared libraries; skips cleanly without them. This is the check whose absence let a size ceiling stand in for a rendering check. **P0**

## 8. MCP server

- [ ] `chorus-mcp` subprocess starts cleanly on stdio.
- [ ] `list_oracles` returns exactly **9** oracles with spec fields matching the Python API (`sequence_length`, assay types, resolution). **P0**
- [ ] Exactly **24** tools registered via FastMCP (`await mcp._list_tools()`). The 22 historically documented plus `recommend_alphagenome_backend` and `score_ism`. **P1**
- [ ] MCP tool count matches what walkthrough READMEs & `docs/MCP_WALKTHROUGH.md` advertise.
- [ ] System-prompt instructions in `chorus/mcp/server.py` are in sync with real specs (track counts, assay names, recommended oracle per task). **P1**
- [ ] `analyze_variant_multilayer` end-to-end: spawn `chorus-mcp`, connect with `fastmcp.Client`, run rs12740374 against AlphaGenome HepG2 tracks, assert the returned dict shape matches what walkthroughs document. (Integration-marked; run on release host.) **P1**
- [ ] Error paths in MCP tools surface `{"error": ..., "error_type": ..., "tool": ...}` — not raw tracebacks. **P1**

### 8b. Drive the tools in plain English, as a user would

Inherited from the retired `AUDIT_PROMPT.md`. Paste each into a Claude Code session with the chorus
MCP server registered, and check the result against the committed walkthrough it corresponds to
(`examples/walkthroughs/<area>/<name>/example_output.md`). The point is to exercise the *natural
language* path, which the Python-API checks above never touch. **P1**

- **Variant analysis** — should reproduce a CEBPA/CEBPB binding gain in HepG2:
  > Load AlphaGenome and analyze rs12740374 (chr1:109274968 G>T) in HepG2 liver cells. Use DNASE,
  > CEBPA ChIP, CEBPB ChIP, H3K27ac, and CAGE tracks. Gene is SORT1.

  Expect `CHIP:CEBPB:HepG2` ≈ **+3.32 at percentile 0.9995** and `CHIP:CEBPA:HepG2` ≈ **+2.95 at
  0.9998** (`variant_analysis/SORT1_rs12740374/example_output.md`).
- **Discovery** — which cell types are affected:
  > Discover which cell types are most affected by rs12740374 (chr1:109274968 G>T) using
  > AlphaGenome.
- **Batch scoring** — five SORT1-locus SNPs:
  > Score these 5 variants in HepG2 with AlphaGenome and rank by effect: rs12740374
  > chr1:109274968 G>T, rs1626484 chr1:109275684 G>T, rs660240 chr1:109275216 T>C, rs4970836
  > chr1:109279175 G>A, rs7528419 chr1:109274570 A>G. Use DNASE, CEBPA, CEBPB, H3K27ac and CAGE.
  > Gene is SORT1.
- **Causal prioritization** — exercises the LDlink token path:
  > Fine-map the SORT1 LDL cholesterol GWAS locus. Lead variant is rs12740374. Auto-fetch LD
  > proxies from LDlink (population CEU, r²≥0.85). Score each variant in HepG2 with DNASE, CEBPA,
  > CEBPB, H3K27ac and CAGE. Gene is SORT1.

  Note the **committed** example supplies 11 LD variants directly rather than auto-fetching, so
  this prompt tests a path the shipped artefact does not: expect `composite=0.970`,
  `max_effect=+3.316`, 4 layers, `convergence=1.00` for the locus itself
  (`causal_prioritization/SORT1_locus/example_output.md`), not a byte match.
- **Region swap** and **integration simulation** — the two sequence-engineering tools. Use the
  prompts at the top of `sequence_engineering/region_swap/example_output.md` and
  `integration_simulation/example_output.md` **verbatim**; both were reworded (a 630 bp
  GFP/reporter construct, and a 378 bp CMV construct at chr19:55115000) and a paraphrase will not
  reproduce the committed numbers.

### 8c. Credentials for an audit run

`chorus setup` prompts for both tokens and persists them to `~/.chorus/config.toml`, so exporting
them is optional — but **required** if stdin is not a TTY, because the HF token resolves before
anything is built and the run aborts with zero progress otherwise:

```bash
export HF_TOKEN=hf_...        # gated AlphaGenome model; or pass chorus setup --hf-token
export LDLINK_TOKEN=...       # only for fine_map_causal_variant's auto-fetch
```

- [ ] Do **not** write real tokens into any file in the repo. A live LDlink token sat in a tracked
      audit report for four months because it was pasted into a "hygiene" note; see §16. **P0**

> **Browser rendering is no longer a manual step.** `AUDIT_PROMPT.md` carried a Selenium block for
> this; it is superseded by `tests/test_committed_reports_render_in_a_browser.py` (46 tests, run with
> `-m ""`), which drives headless Chromium via playwright and — unlike
> `document.querySelectorAll` — walks shadow roots, without which a perfectly painting IGV panel
> reads as blank. See §7.

## 9. Error messages — first-user friendliness

Trigger and inspect each:

- [ ] `create_oracle('fakeOracle')` → names the valid options.
- [ ] `predict(...)` pre-load → `ModelNotLoadedError` with the fix hint.
- [ ] Missing reference_fasta → names the kwarg and `chorus genome download hg38`.
- [ ] Missing oracle env → logs `Run chorus setup --oracle <name>` hint and downgrades to `use_environment=False` (graceful degradation — **not** a crash). Regression: `tests/test_error_recovery.py::TestEnvironmentFailurePaths::test_missing_oracle_env_falls_back_gracefully`.
- [ ] HF token missing (AlphaGenome) → names `HF_TOKEN`, the exact gated repo URL, and the `huggingface-cli login` alternative.
- [ ] Network drop during `download_pertrack_backgrounds` → returns 0 and logs a warning, does not raise. Regression: `tests/test_error_recovery.py::TestDownloadFailurePaths::test_hf_hub_download_failure_returns_zero_and_does_not_crash`.

## 10. Consistency of claims across the repo

Repo-wide drift grep — any match should be investigated:

```
grep -rn '5,930\|5930\|196 kbp\|examples/applications/' --include='*.md' --include='*.py' --include='*.ipynb' .
grep -rn '7,612' scripts/ examples/ --include='*.md'
grep -rn 'LegNet.*230 bp\|input_size_bp.*230' chorus/ scripts/ --include='*.py' --include='*.md'
```

- [ ] Canonical numbers: **AlphaGenome 5,731 model tracks** but **5,168 CDF-backed** (both figures are correct — always say which) / **Enformer 5,313** / **Borzoi 7,611** / **Sei 21,907** total but 40 CDF-backed classes / **LegNet 200 bp input, 3 CDFs** / **ChromBPNet 753 per-track CDFs** (9 human ATAC-DNASE + 744 CHIP; the 33 mouse mm10 models were dropped 2026-08-01 because their backgrounds were built on hg38 — the old "786" and "24 per-model" both predate that) / **Cherimoya 1,518** / **EPInformer-seq 33**. No doc may disagree. **P1**
- [ ] Formula conventions documented **once** and cited by every report/notebook: `log2FC` (default), `lnFC` (gene expression), `Δ (alt−ref)` (MPRA). **P1**
- [ ] Directory naming: live docs only reference `examples/walkthroughs/` and `examples/notebooks/`. The old `examples/applications/` path must not appear **as a path users are pointed at**. **P0** — three live mentions are legitimate and expected: two in `tests/test_rerender_refuses_to_degrade.py` (the test that exists *because* the directory was removed) and one comment in `scripts/rerender_examples.py` recording the rename.
- [ ] README "Hardware matrix per oracle" section is in sync with `chorus/mcp/server.py::ORACLE_SPECS`. **P1**
- [ ] 👁 No "TODO", "coming soon", "WIP" markers in live docs (`audits/` and git history excluded).

## 11. Test suite

No marker filter is needed any more, and that is the point: `pytest.ini` now sets
`addopts = -m "not integration"`, so one command means the same thing for a
contributor, for CI, and here.

It used to take two extra flags. `pytest.ini` set no `addopts`, so the "fast" suite
collected the integration tests and hit HuggingFace/ENCODE — red on any machine without
the per-oracle envs — while CI stayed green by passing `-m "not integration"` **and**
`--ignore=tests/test_smoke_predict.py`, whose fixtures were unmarked and unguarded and
so raised rather than skipped. Fixed at the source (2026-08-10): the smoke tests are
marked `integration` and guard their prerequisites, and
`tests/test_default_pytest_run_excludes_integration.py` fails if pytest.ini, the
workflow and this section ever drift apart again.

```
mamba run -n chorus python -m pytest tests/ -q          # the fast suite (default)
mamba run -n chorus python -m pytest tests/ -q -m integration   # needs the oracle envs
mamba run -n chorus python -m pytest tests/ -q -m ""            # everything, no filter
```

- [ ] Fast suite green, **0 fail, 0 error**. Current counts: **1,463 selected of 1,535**
      collected (72 deselected as `integration`), and the last full run was
      **1,501 passed / 33 skipped / 1 xfailed** with `-m ""`. Skips are
      import/weights-gated by design (alphagenome_pytorch absent, torch absent in the
      base env for epinformerseq, per-cell epinformerseq weights) plus the oracle-env
      gated integration tests when run on a machine without `chorus setup`.
- [ ] **Release gating needs both runs.** Excluding `integration` by default makes the
      common command honest; it does not retire the suite. `pytest -m integration` on a
      release host: SEI/LegNet CDF download, ChromBPNet fresh model download, MCP E2E,
      the 6 oracle smoke tests and the Cherimoya builder-vs-query invariant all pass.
- [ ] CI workflow at `.github/workflows/tests.yml` runs green on the PR, and runs
      `pytest tests/ -q --durations=10` with **no** marker filter and **no** `--ignore` —
      the same command a contributor runs.
      `tests/test_default_pytest_run_excludes_integration.py` fails if either comes back.
- [ ] Coverage of new code paths: any new oracle / normalizer / tool needs its own test.

## 12. Reproducibility

- [ ] Regen scripts in `scripts/` reproduce the committed walkthroughs **identically modulo the `generated_at` / `Generated:` timestamp** (or within ±0.006 numerically). Byte-identity is unattainable by construction: both `regenerate_multioracle.py` and `regenerate_examples.py` stamp a fresh UTC time. **P1**
- [ ] `scripts/regenerate_multioracle.py --consolidate` is idempotent and picks up fresh per-oracle JSONs. **P1**
  - ⚠ Each per-oracle run must happen **inside that oracle's env** (`mamba run -n chorus-chrombpnet …`), not the base env — the script builds its oracle with `use_environment=False`. Only `--consolidate` runs anywhere.
  - ⚠ `*_variant_report.pkl` is gitignored. Without all three present, `--consolidate` silently degrades to "loaded %s from JSON only (no IGV predictions)" and drops that oracle's IGV tracks from the shipped report. Count the tracks before and after. **P1**
- [ ] Reference-genome + annotation files can be reproduced by re-running the documented `chorus genome download` / `download_gencode` calls. **P2**

## 13. Scientific determinism

Same input → same output, run twice. One-shot check per oracle:

```python
import numpy as np
r1 = oracle.predict(('chr1', 1_000_000, 1_100_000), ['<track>'])
r2 = oracle.predict(('chr1', 1_000_000, 1_100_000), ['<track>'])
assert np.allclose(r1['<track>'].values, r2['<track>'].values, atol=1e-6)
```

- [ ] Same-machine back-to-back: identical predictions. Verified bitwise for borzoi, cherimoya, chrombpnet, enformer, epinformerseq, legnet, sei. **P1**
- [ ] ⚠ **AlphaGenome (JAX) is NOT deterministic run-to-run** on Linux/CUDA: two consecutive identical calls differed on all 64 raw values in the SORT1 multi-oracle report (e.g. `ref_value` 2573.0 → 2568.0, `raw_score` 1.33149 → 1.32977, ~0.1–0.4%). `quantile_score` was stable in all 58 cases. So any committed AlphaGenome artefact is not byte-reproducible, and this gate must exempt it or compare percentiles rather than raw values. **P1**
- [ ] Across machines: drift stays within the ±0.006 CPU non-determinism band documented in the walkthrough examples. **P2**

## 14. Genomics edge cases

Each is a common user scenario, not a theoretical corner:

- [ ] **Variant near a chromosome end** (< half window from telomere). The raw `extract_sequence` *raises* `InvalidRegionError` naming the chromosome and both lengths, which is correct; padding is `extract_sequence_with_padding`'s job and it must return exactly `total_length`. **P1** — regression: `tests/test_padding_never_returns_a_short_sequence.py`. Its wide-interval branch used to hand an out-of-bounds end to pysam, which clamps silently: 2,114 bp requested 40 bp from chr1's end returned **40 bp** with metadata claiming no padding (2026-08-12 audit, F2).
- [ ] **Soft-masked (lowercase) FASTA bases** — `extract_sequence` **upper-cases** its output (`chorus/utils/sequence.py:135` ends `return sequence.upper()`), and the ref-allele comparison that uses `.upper()` is at **`core/base.py:460`** (`:325` is now inside an unrelated `ValueError`). Either way a variant in a soft-masked region must not produce a spurious mismatch warning. **P1**
- [ ] **Multi-allelic site** (`alleles=['A','C','G','T']`) — the report renders one `### Allele: alt_N` **section** per alt (three tables), not three columns, and `effect_sizes` carries `alt_1..alt_3` with distinct values. **P1**
- [ ] **Non-SNV** (simple insertion / deletion): if not supported, `predict_variant_effect` should error **before** running the model, not after — and the message should say indels are unsupported. **P1**
- [ ] **Non-canonical chromosomes** (chrM, chrY): either predict or fail cleanly with a message that names the chromosome. **P2**

## 15. Offline / air-gapped behaviour

Many scientific compute environments cut outbound internet after setup. Once install + CDFs + genome are cached:

- [ ] `oracle.predict(...)` works with `HF_TOKEN` unset and no network, for the non-gated oracles. **P1**
- [ ] `oracle.analyze_gene_expression(predictions, 'GATA1')` works against the locally-cached GTF. (The signature takes the `OraclePrediction` first — `analyze_gene_expression('GATA1')` alone raises `TypeError`; see `core/base.py:591`.) **P1**
- [ ] Report HTML makes no network call but the reference sequence. **Do not use the grep** that used to be here: `googleapis` matches Google-Cloud-Storage support code *inside* the inlined `igv.min.js` and so fires on all 19 IGV reports, and `igv.org` matches its blat service URL — both false positives on correct artefacts. Measure the request inventory instead: `pytest tests/test_reports_bundle_their_genome.py -m integration`, which counts what Chromium actually fetches (9 requests to one host after #139) and proves the same-origin air-gap recipe renders with zero. **P1**

## 16. Logging hygiene

`HF_TOKEN` and other secrets should never land in logs, notebook outputs, or HTML reports.

- [ ] `pytest tests/test_no_committed_credentials.py` passes. **P0** — run the test, do **not** hand-roll a grep. Four consecutive audits reported "0 `hf_…` tokens, 0 AWS keys" while a live **LDlink** token sat in `audits/2026-04-23_v23_scorched_earth/report.md:299` for nearly four months. It was unmissable to a human and unmatchable by every sweep run, because an LDlink token is twelve **bare hex characters with no prefix** and every sweep searched for prefixes. A clean prefixed grep means "no secrets of the shapes we grep for", which is not the same claim. The test covers prefixed shapes *and* the contextual unprefixed case (a hex/base62 run within 40 characters of the word token/secret/api-key/password), and is mutation-tested against the wording that actually leaked. It also **scans by default** and skips only known-binary suffixes: an earlier version filtered by an allowlist of "text" suffixes and so never opened 59 `.log` files or 20 `.html` reports — the two artefacts a credential is most likely to reach, since a log captures the environment and a report captures URLs. Coverage is asserted by the test itself (>80% of tracked files), because a shrinking allowlist is how a guard quietly stops guarding.
- [ ] If a credential is found: **rotate it**. Redacting the file does not undo exposure — the value stays in git history, and history rewriting is the maintainer's call. Record the rotation, not just the redaction. **P0**
- [ ] Committed notebook outputs and test fixtures don't contain `HF_TOKEN=hf_…` or AWS-style keys. Known benign: per-machine absolute paths in shipped notebook outputs — documented as cosmetic in v16. Both forms occur: `/srv/local/<user>/…` (advanced_multi_oracle, comprehensive_showcase) and macOS `/Users/<user>/…`. Re-executing a notebook simply swaps in the current host's paths. **P1**

## 17. Dependency supply chain

- [ ] `environment.yml` pins every dep to a range or exact version — no bare dep names. **P1** — two entries read as unpinned to a naive check and are both fine: the literal `pip` (not a dependency) and `coolbox @ git+…@651b930…` (pinned to a commit, which is stronger than a version range).
- [ ] `pip-audit` on the base env flags no known CVEs above *medium*. **P1**
- [ ] Per-oracle envs use the same `chorus` editable install so they track the parent codebase (`EnvironmentManager.install_chorus_primitive`). **P1**

## 18. License / attribution

- [ ] `LICENSE` file at repo root matches the license Chorus claims. **P0**
- [ ] Each oracle's model weights + third-party code is attributed somewhere reachable from the README (Enformer → DeepMind, ChromBPNet → Kundaje lab, AlphaGenome → Google DeepMind, etc.) — a single `docs/THIRD_PARTY.md` is fine. **P1**
- [ ] Bundled vendor JS (`chorus/analysis/static/igv.min.js`) carries its upstream license header. **P1**

---

## 19. Cutting a release

Added 2026-08-10, because two things had drifted silently: v0.5.0–v0.5.6 were tagged and
published as GitHub Releases with **no CHANGELOG sections** (the notes lived only in the
Releases UI for three months), and 66 commits sat on `main` with **no tag at all** —
including one that moved every effect percentile, so the state users had was nameless.
`tests/test_release_bookkeeping.py` now fails on both, but a check is not a procedure.

**A release is a pair: (code tag, artefact revision).** Percentiles are a function of both,
and the artefacts live in a repo whose `main` moves. Skipping the second half is how the
2026-08-10 upload silently changed the behaviour of every already-released version.

- [ ] Decide the bump from **what moves for a user**, not from diff size. Any change to a
      null, a region set, a retention rule or an oracle's default fold moves percentiles →
      minor at least. Say so in the first line of the section, e.g. *"Effect percentiles
      change and are not comparable with any earlier release."* **P0**
- [ ] `[Unreleased]` → `## [X.Y.Z] — YYYY-MM-DD`, leaving `[Unreleased]` genuinely empty.
      If the branch and `main` both wrote to `[Unreleased]`, split by which bullets are
      already present in `git show origin/main:CHANGELOG.md`. **P0**
- [ ] Each section carries, in order: the **numbers-changed banner**, the **artefact
      revision** it pairs with, the Keep-a-Changelog buckets, and **Known limitations** —
      this project states its negatives and they must not be dropped at release time. **P1**
- [ ] Bump **both** `setup.py` and `chorus/__init__.py`. **P0**
- [ ] Tag the dataset repo at the revision this release was verified against, named
      `backgrounds-<date>-<slug>`, and set `_HF_REVISION` in
      `chorus/analysis/normalization.py` to it. Verify the tag resolves to the expected
      *content* (compare file sizes) — creating an HF tag makes a commit on the tag ref, so
      the sha you passed is not the sha you get back. **P0**
- [ ] Add the compare link to the footer. **P2**
- [ ] Both suites green on the commit being tagged — `pytest tests/ -q` **and**
      `pytest tests/ -q -m integration`. The second is the one that carries the release
      gates, and it is the one easy to forget. **P0**
- [ ] Annotated tag + a GitHub Release whose body is that CHANGELOG section. **P1**
- [ ] **Publish the Release only once the tag is final, and never move a tag under a published
      Release without saying so.** Both v0.7.2 and v0.7.3 were published seconds after their
      `vX.Y.Z` commit and then had the tag force-moved hours later — v0.7.3's by two commits —
      so for 8.5 h and 12 h respectively, anyone who ran `git fetch --tags` or downloaded the
      release tarball got a tree **missing the fixes the notes led with**. The tell is in the
      API and nowhere else: `gh release list --json tagName,createdAt,publishedAt` shows
      `createdAt` (which is tag-derived) *later* than `publishedAt`, and for a healthy release
      they differ by seconds. Check that before calling a release done. If the tag has to move,
      re-point it, regenerate the notes, and note the move on the release page. **P1**
- [ ] **A moved tag has _two_ independent copies of the notes; regenerate both.** The GitHub Release
      body is the obvious one. The **annotated tag object carries its own message**
      (`git tag -l --format='%(contents)' v<X.Y.Z>`) and editing the release does not touch it.
      Create it with **`--cleanup=verbatim`**, or git silently deletes every line starting with `#`:
      v0.7.3's tag message lost all four `###` headings that way and nobody noticed, because the
      release page looked correct. Verify with
      `git tag -l --format='%(contents)' v<X.Y.Z> | grep -c '^### '`. **P1**
- [ ] **The `## [X.Y.Z] — DATE` heading must match the day of the commit the tag points at.** Nothing
      enforces this — `test_released_sections_are_dated` only asserts that *an* ISO date is present —
      so a tag moved across midnight dates its own tree a day early. **P2**
- [ ] **CI cannot observe a tag move.** `.github/workflows/tests.yml` has no `tags:` trigger, so
      force-pushing a tag runs nothing at all. Any verification a release claims must be run locally
      against the exact commit the tag dereferences to, and the numbers recorded with that sha. **P1**
- [ ] `tests/test_release_bookkeeping.py` and `tests/test_artefact_revision_is_pinned.py`
      pass, including the `[Unreleased]`-is-empty check, which only activates once HEAD is
      tagged. **P0**

Retroactive tags are legitimate and better than leaving a state nameless — v0.6.0 was cut
this way at `3e7990a` five days after the fact. Say in the section that it is retroactive
and give the commit.

---

## Appendix — artefacts to produce per audit

A full audit should leave behind, in `audits/YYYY-MM-DD_vNN_<label>/`:

- `report.md` — findings summary (one bullet per finding: file:line, problem, fix).
- `screenshots/*.png` — one per shipped HTML (selenium-rendered, 1600×4500).
- `nb_fresh_output/*.ipynb` — fresh re-execution of every notebook.
- `cdf_check.txt` — output of the CDF-sanity script from §4.
- `device_probe.txt` — output of the per-env GPU probe from §3.

These let the next auditor diff your findings against theirs mechanically.
