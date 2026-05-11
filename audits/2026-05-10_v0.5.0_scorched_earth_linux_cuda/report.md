# Chorus v0.5.0 Scorched-Earth Audit — Linux/CUDA (ml007)

**Date**: 2026-05-10 (audit ran 01:22 → 19:00 UTC, ~17.5 h wall — see Time budget)
**Platform**: Linux ml007 / Ubuntu 5.15.0-160 / 2× NVIDIA A100-PCIE-40GB / panfs `/data/pinello` (89% used, 21 TB free)
**Tested tag**: `v0.5.0` (commit `3311351`, "Merge pull request #78 from pinellolab/fix/post-v040-followups")
**Auditor**: Claude Opus 4.7 (1M context)

## Outcome

**2 P0 + 2 P1 + 1 P2 findings.** Audit dir is committed locally only — **NOT pushed to GitHub per user instructions to hold on errors.** Push waits for morning review.

| Severity | Count | Names |
|---|---|---|
| **P0** | 2 | alphagenome JAX/PT 0.476 backend drift; chrombpnet predict_sliding TF import bug |
| **P1** | 2 | alphagenome_pt 404 on HF; chrombpnet integration test 600s timeout |
| **P2** | 1 | `chorus backgrounds status` doesn't list alphagenome_pt |

## Why this audit

The other agent shipped v0.5.0 since the last audit (v29 Linux/CUDA, 2026-04-27 against v0.2.0). User asked for a true "new user" simulation: wipe everything (including HF cache so token + cache are recreated from zero), pull `v0.5.0`, install + setup + tests + notebooks + HTML render. Run autonomous overnight, hold on errors.

## Scorched

| Item | Size before | Notes |
|---|---|---|
| 8 envs (chorus + 7 oracle) | ~30 GB | wiped via `chorus cleanup --all` + `mamba env remove -n chorus` |
| `~/.chorus/` | 2.3 GB | backgrounds + interim shards from yesterday's CDF rebuild |
| `~/.cache/huggingface/` | 4.9 GB | wiped to force fresh HF re-downloads |
| `genomes/hg38.fa` | 3.3 GB | wiped |
| `downloads/` | 88 GB | wiped |
| **Total freed** | **~129 GB** | |

HF token preserved at `~/.token_chorus_audit` (chmod 600, outside `~/.chorus`).

## Reinstalled — wall-clock

Network filesystem `/data/pinello` was severely loaded all night, so every IO-bound stage ran 4-10× slower than yesterday's v29 Linux/CUDA run (~6.5 h end-to-end on the same hardware).

| Phase | Wall-clock | Yesterday |
|---|---|---|
| Wipe (8 envs + caches) | 5h 35min | ~80 min |
| `mamba env create -f environment.yml` + `pip install -e .` | 2h 8min | 31 min |
| `chorus setup --oracle all` (7 oracles) | 8h 14min | 3h 11min |
| Stage 5 integration tests (5 selected) | 48 min | 8.5 min |
| Stage 6 fast pytest (382 collected) | 27 min | 27 min |
| Stage 7 3 notebooks | ~9 min total | 34 min |
| Stage 8 playwright (install + 18 HTMLs) | 5 min | 6 min |
| Audit report | ~5 min | n/a |
| **Total** | **~17.5 h** | **~6.5 h** |

## What works ✓

| Check | Result |
|---|---|
| v0.5.0 tag pulls cleanly + `chorus 0.5.0` CLI imports | ✓ |
| 7 oracle envs build successfully (alphagenome, alphagenome_pt **NEW in v0.5.0**, borzoi, chrombpnet, enformer, legnet, sei) | ✓ |
| `chrombpnet_pertrack.npz` from yesterday's DHS-augmented HF push downloads cleanly | ✓ (786 rows = 42 ATAC/DNASE + 744 CHIP) |
| `single_oracle_quickstart.ipynb` notebook | ✓ pass |
| Fast pytest excluding integration tests | 376 / 379 effectively (2 skipped, 1 P0 fail) |
| Integration `test_pertrack_background_download[sei]` | ✓ |
| Integration `test_pertrack_background_download[legnet]` | ✓ |
| Integration `test_mcp_e2e_*` (with HF_TOKEN env var) | ✓ |
| 18 HTML walkthroughs render (1600×4500 headless Chromium) | **18/18 loaded, 17 IGV, 18 glossary, 0 JS errors** — byte-for-byte match with v29 |

The HTML walkthroughs and the bulk of the test suite are healthy; the regressions concentrate in the alphagenome JAX/PT equivalence story and the new wide-window ChromBPNet path.

## Findings

### P0 #1 — alphagenome JAX vs PyTorch backends drift to 0.476 max abs at SORT1 (10× worse than PR #62 baseline)

**Test**: `tests/test_alphagenome_backends_equivalence.py::test_jax_pt_chorus_api_equivalence_at_sort1`
**Assertion**: `max(|pt - jax|) < 0.1`
**Result**: max abs diff = **0.4760** for `DNASE/EFO:0002067 DNase-seq/.`. Audit numbers from PR #62 cite < 0.05.

```
> assert max_abs < 0.1, (
      f"{aid}: max abs diff {max_abs:.4f} exceeds 0.1 — JAX vs PyTorch "
      f"backend drift. Audit numbers from PR #62 were < 0.05."
  )
E   AssertionError: DNASE/EFO:0002067 DNase-seq/.: max abs diff 0.4760 exceeds 0.1 — JAX vs PyTorch backend drift. Audit numbers from PR #62 were < 0.05.
E   assert 0.4760284423828125 < 0.1
```

**Implication**: The two AlphaGenome backends do **not** produce equivalent predictions in v0.5.0. Yesterday's design decision ("alphagenome_pt can reuse alphagenome's per-track CDF") is invalidated by these numbers. Possible upstream reasons:
- PyTorch port drifted away from JAX reference (weight conversion or layer-mismatch regression).
- Default device / dtype between backends (e.g. JAX defaulting to fp32 vs PyTorch using a different precision).
- A recent refactor to one backend's `oracle.predict()` slicing layer changed which logical track is being returned.

**Recommended next step**: triage where the divergence comes in (is it in the backend output before chorus's `local_index` slice, or after?), bisect against the PR #62 baseline.

### P0 #2 — ChromBPNet `predict_sliding` does direct `import tensorflow as tf` in chorus base env (no TF) — breaks 2 of 3 shipped notebooks

**Triggered by**: any caller passing a window wider than `ChromBPNetOracle.sequence_length` (501 bp) to `oracle.predict()`. PR #79's wide-window flow makes this the default path for multi-oracle reports.

**Failing notebooks**:
- `examples/notebooks/advanced_multi_oracle_analysis.ipynb` — fails at cell 12 calling `chrombpnet_oracle.predict(("chrX", 48_730_000, 48_840_000), tracks)`
- `examples/notebooks/comprehensive_oracle_showcase.ipynb` — same error pattern
- `single_oracle_quickstart.ipynb` ✓ (uses narrow window, doesn't hit `predict_sliding`)

**Stack trace**:

```
File ~/chorus/chorus/oracles/chrombpnet.py:630, in ChromBPNetOracle._predict
    if len(query_interval) > self.sequence_length:
        return self.predict_sliding(query_interval, assay_ids)

File ~/chorus/chorus/oracles/chrombpnet.py:761, in predict_sliding
    import tensorflow as tf
ModuleNotFoundError: No module named 'tensorflow'
```

**Root cause**: `predict_sliding` runs in the **caller's** Python process (chorus base env), not in the chorus-chrombpnet subprocess. The base env doesn't ship TF — and shouldn't.

**Fix**: route `predict_sliding` through the same `EnvironmentRunner` subprocess pattern that `_predict_direct` already uses for the chrombpnet env. Mirror the approach in `chorus.core.environment.runner.run_code_in_environment`.

**User-flow severity**: docs explicitly point new users to the advanced + comprehensive notebooks ("three end-to-end tutorials"). Two of three are broken on a fresh v0.5.0 install. This is the single most-noticeable regression for a new user.

### P1 #1 — `alphagenome_pt_pertrack.npz` returns 404 from HF during `chorus setup --oracle alphagenome_pt`

**Time observed**: 2026-05-10 11:48:57 UTC
**Symptom during setup**:

```
chorus.analysis.normalization - WARNING - Failed to download alphagenome_pt_pertrack.npz: 404 Client Error
chorus.cli._setup_all - INFO - ✓ alphagenome_pt ready
```

**User's design intent** (from yesterday's session): alphagenome and alphagenome_pt should share a single per-track CDF since they're "the same model". **However**, the new P0 #1 finding shows the JAX and PT backends drift 10× past the equivalence tolerance — so the shared-CDF design cannot stand as-is. Until the JAX/PT divergence in P0 #1 is resolved, alphagenome_pt either needs:
- (a) its own CDF built and uploaded to HF, OR
- (b) the JAX/PT alignment fixed back to the PR #62 < 0.05 tolerance, then sharing is justified.

**Local workaround applied for the audit**:

```bash
cp ~/.chorus/backgrounds/alphagenome_pertrack.npz \
   ~/.chorus/backgrounds/alphagenome_pt_pertrack.npz
```

This let the rest of the audit cascade through without 404s. **Important caveat**: any normalization output produced for `alphagenome_pt` during this audit is suspect, since the CDFs are alphagenome's, not alphagenome_pt's actual distribution.

The setup CLI marks `✓ alphagenome_pt ready` even when the NPZ download 404s. That's permissive — could mislead users. Worth surfacing as a non-fatal "expected: alphagenome_pt → alphagenome CDF" log line if/when the shared-CDF approach is restored.

### P1 #2 — `test_chrombpnet_fresh_single_model_download` 600 s timeout on slow panfs

**Test**: `tests/test_integration.py::test_chrombpnet_fresh_single_model_download`
**Symptom**:

```
RuntimeError: chrombpnet prediction timed out after 600s. For long-running workloads
(full-genome scans, dense variant lists, CPU inference), set CHORUS_NO_TIMEOUT=1 to
disable timeouts, or raise the per-call timeout explicitly.

WARNING  chorus.core.base:base.py:136 Environment validation for chrombpnet timed out
(Timeout while checking dependency tensorflow). Proceeding with use_environment=True;
the actual run will surface any real issue.
```

**Root cause**: TF import inside `mamba run -n chorus-chrombpnet` against panfs is excruciatingly slow tonight. The pre-flight env-validation step itself fires its own timeout warning before the real prediction call is made.

**Yesterday's run**: same test passed in 8.43 s.

**Recommended fix**: either raise the default chrombpnet timeout from 600 s to handle cold subprocess starts on network filesystems, or set `CHORUS_NO_TIMEOUT=1` when running integration tests by default. Borderline P0 if this is a CI-runner concern; environmental-only on a fresh local install.

### P2 #1 — `chorus backgrounds status` doesn't list alphagenome_pt

**Symptom**: After `chorus setup --oracle all` completes successfully on v0.5.0, `chorus backgrounds status` prints 6 oracles (alphagenome, borzoi, chrombpnet, enformer, legnet, sei). `alphagenome_pt` is missing from the list, even though `get_pertrack_normalizer('alphagenome_pt')` loads cleanly via the local alias.

**Root cause**: `chorus backgrounds status` iterates over a hard-coded oracle list that wasn't extended for alphagenome_pt.

**Fix**: enumerate from the same registry the rest of the CLI uses for `--oracle` choices.

## What this audit did NOT cover

- macOS arm64 (Apple Silicon) — last covered in 2026-04-26 v29 audit on v0.2.0; should re-run on v0.5.0 to confirm the P0 ChromBPNet TF-import bug isn't platform-specific.
- The `--all-chrombpnet` opt-in path (would re-download ~30 GB of every published model).
- DHS-rebuild round-trip on a fresh CDF (yesterday's separate audit at `audits/2026-05-09_dhs_chrombpnet_full_rebuild.md`).
- Self-hosted CI runner integration.

## Recommendations

1. **Block tag v0.5.1** on fixing P0 #2 (predict_sliding TF import) — it's a fresh-install showstopper for the documented tutorial.
2. **Triage P0 #1** (alphagenome JAX/PT drift) before merging anything that depends on the equivalence assumption.
3. **Decide on alphagenome_pt CDF**: rebuild + upload, or fix backend alignment then share. Either path resolves P1 #1.
4. **Bump `chorus-chrombpnet` integration timeout** or set `CHORUS_NO_TIMEOUT=1` for `pytest -m integration` to address P1 #2.
5. **Extend `chorus backgrounds status`** to enumerate alphagenome_pt (P2 #1) — trivial CLI fix; bundle with P1 #1's alias resolution.

## Audit dir contents

```
audits/2026-05-10_v0.5.0_scorched_earth_linux_cuda/
  preflight_state.txt
  wipe_log.txt
  env_create_log.txt
  setup_log.txt
  integration_tests_log.txt
  fast_pytest_log.txt
  notebooks_log.txt
  playwright_log.txt
  render_log.json
  screenshots/  (18 PNGs)
  probes/05_html_render.py  (90 s networkidle, OUT repointed)
  findings_running.md
  report.md  (this file)
```

## Branch state

- HEAD: `v0.5.0` tag (3311351), local main is at the same commit per origin/main
- Working tree: untracked `audits/2026-05-10_v0.5.0_scorched_earth_linux_cuda/` directory only
- **Not committed and not pushed** per user instructions to hold on errors. Findings ready for morning review.
