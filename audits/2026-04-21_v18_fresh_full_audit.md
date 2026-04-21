# v18 fresh full audit — 2026-04-21

No cached / precomputed results. Exercised the library end-to-end from a
clean state: re-executed `single_oracle_quickstart.ipynb`, rendered
shipped HTML reports with **selenium** (so IGV's client-side JS actually
loads tracks), sanity-checked per-track CDFs for all 6 oracles,
enumerated the PyTorch / TF / JAX device matrix across all 6 oracle
envs on macOS arm64, and traced the HuggingFace gate path that
AlphaGenome users hit on first use.

Main deliverable: **`audits/AUDIT_CHECKLIST.md`** — a reusable
12-section checklist (Install → HF → GPU → CDFs → Python API →
Notebooks → HTML reports → MCP → Error paths → Repo-wide consistency →
Tests → Reproducibility) with P0/P1/P2 severity and exact commands to
run each check.

## What was actually run

### Notebook — fresh re-execution

Ran `jupyter nbconvert --execute examples/notebooks/single_oracle_quickstart.ipynb`
end-to-end on this macOS-arm64 host:

- Exit code 0, every cell completed.
- **Zero errors**, **zero WARNING lines** in any cell output.
- The old "Provided reference allele does not match the genome" warning
  (previously fired on cell 39 variant-effect) is gone — confirms the
  off-by-one fix from PR #32 flowed through to the notebooks.
- Numeric output drift vs the committed run: **≤ 0.0005** per value
  (e.g. `0.0248` → `0.0247`), well within CPU non-determinism across
  machines. No need to regenerate committed outputs.

### Per-track CDF sanity — all 6 oracles

Script in §4 of the checklist. Result:

| oracle | n_tracks | CDFs | monotonic effect_cdfs | p50 ≤ p95 ≤ p99 | signed% |
|---|---|---|---|---|---|
| enformer | 5,313 | effect, summary, perbin | ✓ | ✓ | 0% |
| borzoi | 7,611 | effect, summary, perbin | ✓ | ✓ | 20% |
| chrombpnet | 24 | effect, summary, perbin | ✓ | ✓ | 0% |
| sei | 40 | effect, summary | ✓ | ✓ | 100% |
| legnet | 3 | effect, summary | ✓ | ✓ | 100% |
| alphagenome | 5,168 | effect, summary, perbin | ✓ | ✓ | 13% |

Sei + LegNet CDFs were **auto-downloaded from
`huggingface.co/datasets/lucapinello/chorus-backgrounds`** on first use
— the on-demand path works without the user having to pre-build them.

### Device detection — macOS arm64

| env | backend | detected |
|---|---|---|
| chorus-enformer | TensorFlow | CPU:0, GPU:0 (Metal) |
| chorus-borzoi | PyTorch | cuda: False, **mps: True** |
| chorus-chrombpnet | TensorFlow | CPU:0, GPU:0 (Metal) |
| chorus-sei | PyTorch | cuda: False, **mps: True** |
| chorus-legnet | PyTorch | cuda: False, **mps: True** |
| chorus-alphagenome | JAX | device list populated |

All 6 envs correctly pick up Apple Silicon GPU acceleration. No env
pins to `cuda:0` — `grep -rn "cuda:0'" chorus/oracles/ …/templates/`
returns only docstring examples. `CUDA_VISIBLE_DEVICES` pinning works
(fix came in `6ebc996`).

### HTML reports — selenium (full-JS) render

Rendered 5 of the shipped reports with headless Chrome + selenium at
1600×4500, waited 12 s for IGV to hit the CDN, captured screenshots +
browser console logs:

- `SORT1_rs12740374/rs12740374_SORT1_alphagenome_report.html` —
  full IGV browser with SORT1 locus, ref/alt signal overlays, gene
  track.
- `validation/SORT1_rs12740374_multioracle/…_multioracle_report.html`
  — unified IGV across ChromBPNet / LegNet / AlphaGenome with per-layer
  tracks labelled by oracle.
- `causal_prioritization/SORT1_locus/…_causal_report.html` —
  composite-score bars + per-candidate DNASE / ChIP-TF tracks + ranked
  variants table.
- `validation/SORT1_rs12740374_with_CEBP/…_validation_report.html`
- `discovery/SORT1_cell_type_screen/…_LNCaP_…_report.html`

**Zero SEVERE/ERROR JS console messages** on any report. IGV tracks
load correctly; earlier audits that saw a "placeholder" were using
plain headless Chrome without `--allow-file-access-from-files`, which
blocks the CDN module fetches — documented in the checklist §7 so
future auditors don't hit the same dead end.

## Fixed in this PR

Three HF/alphagenome drifts from the live install flow:

1. **`chorus/oracles/alphagenome.py:133`** — the HF-missing-token error
   linked `huggingface.co/google/alphagenome`. The actual gated repo is
   `google/alphagenome-all-folds` (every live doc says so, including
   `README.md:631`, `README.md:926`, `environments/README.md:105`). A
   user clicking the link would land on the wrong page and not find the
   license form. Fixed.

2. **`chorus/oracles/alphagenome_source/templates/load_template.py:49`**
   — the env-runner load path also raises when HF auth is missing, but
   its error message stopped at "Set HF_TOKEN or run `huggingface-cli
   login`" with no URL at all. Appended the `alphagenome-all-folds`
   license URL so both code paths (direct and env-runner) give the
   same actionable guidance.

3. **`chorus/oracles/alphagenome_source/alphagenome_metadata.py:4`** —
   module docstring said "AlphaGenome predicts 5,930+ human functional
   genomic tracks"; the library reports 5,731 (matches v17 audit and
   the notebooks after v16). Updated to `5,731`.

Also tightened the assertion in
`tests/test_error_recovery.py:169` from `huggingface.co/google/alphagenome`
to `huggingface.co/google/alphagenome-all-folds` so the test actually
catches this drift in future.

## Known issues flagged but NOT fixed here

From the HF/install agent — left for a focused PR each:

- **Genome storage location is ambiguous** (`README.md:210`). Currently
  hardcoded to `<repo>/genomes/` in `chorus/core/globals.py:13`. Should
  be user-overridable (`CHORUS_GENOMES_DIR`) and default to
  `~/.chorus/genomes/` to match the backgrounds cache.
- **`EnvironmentManager.install_chorus_primitive()` silent stderr** —
  when a per-oracle `pip install` fails, the error message may be empty.
  Worth capturing stdout+stderr separately.
- **No documented download-resume hint** — `download_with_resume()`
  auto-resumes a stalled genome download, but users aren't told they
  can just re-run the same command.

## Delivered

- **`audits/AUDIT_CHECKLIST.md`** — 12-section runbook for future
  audits, every check has an exact command or `grep` to run.
- **`audits/2026-04-21_v18_fresh_full_audit.md`** — this report.
- **3 HF drift fixes** (wrong repo URL in 2 code paths + track count
  docstring).
- **1 test assertion tightened** to catch the drift in CI.

Tests: 334 passed / 1 skipped (fast suite). No behaviour changes except
error messages and docstring text.
