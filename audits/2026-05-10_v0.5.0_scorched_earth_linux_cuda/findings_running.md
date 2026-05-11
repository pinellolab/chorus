# Running findings during v0.5.0 scorched-earth audit

## P1: alphagenome_pt CDF lookup is not aliased to alphagenome's CDF in code

**Stage**: Stage 4 (chorus setup --oracle alphagenome_pt)
**Time observed**: 2026-05-10 11:48:57 UTC
**Severity**: P1 — affects every new v0.5.0 user installing alphagenome_pt
**Symptom during setup**:
```
chorus.analysis.normalization - WARNING - Failed to download alphagenome_pt_pertrack.npz: 404 Client Error
chorus.cli._setup_all - INFO - ✓ alphagenome_pt ready
```
**Real problem**: The PyTorch and JAX versions of AlphaGenome produce equivalent predictions (per user design decision), so the per-track CDF should be SHARED between them. Instead, the codebase tries to fetch a separate `alphagenome_pt_pertrack.npz` from HF that doesn't exist (and shouldn't need to). The 404 is a code-wiring miss, not missing data.

**Local workaround applied so the audit cascade can continue**:
```bash
cp ~/.chorus/backgrounds/alphagenome_pertrack.npz \
   ~/.chorus/backgrounds/alphagenome_pt_pertrack.npz
```
After the cp, `get_pertrack_normalizer('alphagenome_pt')` loads cleanly: 5168 tracks. This unblocks downstream notebooks and integration tests that exercise alphagenome_pt normalization.

**Recommended fix (in code)**: In `chorus.analysis.normalization`, alias the alphagenome_pt per-track NPZ lookup to alphagenome — either by:
- (a) sharing the lookup key in `PerTrackNormalizer` so both oracle names resolve to the same NPZ, or
- (b) at HF download time, fall back to `alphagenome_pertrack.npz` when `alphagenome_pt_pertrack.npz` 404s.

Option (a) is cleaner — eliminates the redundant download and the misleading 404 warning entirely.

**Current setup behavior**: Marks `✓ alphagenome_pt ready` even though the NPZ download failed. That's permissive — could mislead a new user into thinking everything's fine. Worth surfacing this as a non-fatal "expected for alphagenome_pt → using alphagenome CDF" log line once (a) is implemented.

## P2: `chorus backgrounds status` doesn't list alphagenome_pt

**Stage**: Stage 4 post-setup verification
**Time observed**: 2026-05-10 17:24 UTC
**Severity**: P2 — surface-only, doesn't affect runtime
**Symptom**: Output of `chorus backgrounds status` shows 6 oracles (enformer / borzoi / chrombpnet / sei / legnet / alphagenome) — alphagenome_pt is missing from the list. But `get_pertrack_normalizer('alphagenome_pt')` loads cleanly (5168 tracks via the local alias copy).
**Root cause**: The `backgrounds status` CLI iterates over a hard-coded oracle list that hasn't been extended for alphagenome_pt.
**Fix**: When fixing the P1 (CDF aliasing for alphagenome_pt), also add it to the `backgrounds status` enumeration so users see it in the inventory.

## P0: alphagenome JAX vs PyTorch backends do NOT agree at SORT1 (max abs diff 0.4760, tolerance 0.1)

**Stage**: Stage 5 integration tests
**Time observed**: 2026-05-10 17:25 UTC, test session ran 48:40
**Severity**: P0 — invalidates the design assumption that the two backends produce equivalent predictions
**Failing test**: `tests/test_alphagenome_backends_equivalence.py::test_jax_pt_chorus_api_equivalence_at_sort1`
**Assertion**: `max(|pt - jax|) < 0.1`
**Actual values from this run**:
```
DNASE/EFO:0002067 DNase-seq/.:
  max abs diff: 0.4760  (tolerance: 0.1)
  PR #62 audit numbers were: < 0.05
```
**Implication**:
- The user's design assumption from yesterday — "alphagenome and alphagenome_pt produce equivalent predictions, so they can share one CDF" — does NOT hold on this v0.5.0 install.
- My local workaround `cp alphagenome_pertrack.npz alphagenome_pt_pertrack.npz` was based on that false premise. The CDFs alphagenome_pt is loading are WRONG for the actual PyTorch outputs.
- A real fix needs: either retraining/fixing the PyTorch backend so it matches JAX within the historical 0.05 tolerance, OR accepting the divergence and building a separate CDF for alphagenome_pt.
- This regressed: the test cites PR #62 audit numbers of <0.05; today we see 0.476 (~10× worse).

**Note**: The `cp` workaround was NOT removed — left in place so the rest of the audit (notebooks, playwright) doesn't fail with 404s on alphagenome_pt CDF. But any normalization output produced for alphagenome_pt during this audit should be considered suspect.

## P1: chrombpnet integration test timed out after 600s (slow panfs / TF re-init)

**Stage**: Stage 5 integration tests
**Time observed**: 2026-05-10 17:48 → 18:14 UTC (test_chrombpnet_fresh_single_model_download)
**Severity**: P1 — environmental / flake risk; would also affect any CI runner on a slow-disk node
**Failing test**: `tests/test_integration.py::test_chrombpnet_fresh_single_model_download`
**Assertion**: `chrombpnet prediction timed out after 600s`
**Symptom**:
```
RuntimeError: chrombpnet prediction timed out after 600s. For long-running workloads (full-genome scans, dense variant lists, CPU inference), set CHORUS_NO_TIMEOUT=1 to disable timeouts, or raise the per-call timeout explicitly.

WARNING  chorus.core.base:base.py:136 Environment validation for chrombpnet timed out (Timeout while checking dependency tensorflow). Proceeding with use_environment=True; the actual run will surface any real issue.
```
**Root cause**: TF import inside `mamba run -n chorus-chrombpnet` against panfs is excruciatingly slow tonight. Even the pre-flight `Timeout while checking dependency tensorflow` warning fires before the actual prediction runs out of budget.
**Yesterday's run**: same test passed in 8.43s on this same host with a less-loaded panfs.
**Recommended fix**: Either (a) raise the default chrombpnet timeout from 600s to something more forgiving for cold subprocess starts on network filesystems, or (b) set `CHORUS_NO_TIMEOUT=1` for integration tests by default.

## P0: ChromBPNet predict_sliding does direct `import tensorflow` in chorus base env (no TF) — breaks 2 of 3 shipped notebooks

**Stage**: Stage 7 notebooks
**Time observed**: 2026-05-10 18:50 UTC
**Severity**: P0 — every new user following the documentation tutorials will hit this
**Failing notebooks**: 2 of 3 shipped notebooks fail with `ModuleNotFoundError: No module named 'tensorflow'`:
- `examples/notebooks/advanced_multi_oracle_analysis.ipynb` — fails at cell 12 (`chrombpnet_oracle.predict(("chrX", 48_730_000, 48_840_000), tracks)`)
- `examples/notebooks/comprehensive_oracle_showcase.ipynb` — same error pattern
- `single_oracle_quickstart.ipynb` ✓ (uses narrow window, doesn't hit predict_sliding)

**Stack trace**:
```
File ~/chorus/chorus/oracles/chrombpnet.py:630, in ChromBPNetOracle._predict
   if len(query_interval) > self.sequence_length:
       return self.predict_sliding(query_interval, assay_ids)

File ~/chorus/chorus/oracles/chrombpnet.py:761, in predict_sliding
   import tensorflow as tf
ModuleNotFoundError: No module named 'tensorflow'
```

**Root cause**: PR #79's wide-window code path (`predict_sliding`) was added on top of the existing `_predict_direct` flow that uses `use_environment=True` to run in chorus-chrombpnet subprocess. But `predict_sliding` itself runs in the **caller's** Python process (chorus base env), and at line 761 it does `import tensorflow as tf` directly — assuming TF is available in the same env. The base chorus env does not (and should not) ship with TF.

**Fix options**:
- (a) Route `predict_sliding` through the same `EnvironmentRunner` subprocess pattern that `_predict_direct` uses for chorus-chrombpnet env.
- (b) Add the relevant TF-using code to the chrombpnet env's stub script that's already invoked via `mamba run -n chorus-chrombpnet`.
- (a) is cleaner — keeps the chorus base env TF-free.

**Impact on user flow**: Documentation says "run advanced_multi_oracle_analysis.ipynb on GPU" and that's literally the second tutorial new users open. Hitting this on tutorial 2 is a guaranteed bad first experience.

**Yesterday's audit** (v29 Linux/CUDA, 2026-04-27) had all 3 notebooks pass because v0.4.0 didn't yet have PR #79's wide-window predict_sliding logic.
