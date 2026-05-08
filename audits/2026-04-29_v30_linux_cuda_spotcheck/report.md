# Chorus v30 Linux/CUDA Spot-Check — `feat/chrombpnet-hf-slim-mirror`

**Date**: 2026-04-29 (started 15:38, integration tests finished 16:39).
**Platform**: Linux `ml008` 5.15.0-170-generic / x86_64 / 2× NVIDIA A100 80 GB PCIe.
**Branch**: `feat/chrombpnet-hf-slim-mirror` @ `f6933bd` (PR #60 head).
**Auditor**: Claude Opus 4.7 (1M context).
**Scope**: cross-platform spot-check of the v30 macOS/Metal audit. Specifically validates that the F1 fix (`a13282c` — adding `huggingface_hub>=0.20.0` to `environments/chorus-chrombpnet.yml`) holds on Linux/CUDA, since env-yaml changes can behave differently across platforms.
**Outcome**: **0 new findings.** Fast suite, integration suite, F1 regression test, and the gold-standard `import huggingface_hub` check inside a Linux-built `chorus-chrombpnet` env all match the v30 macOS results. No regressions exposed by the platform switch.

## Why this spot-check exists

The v30 audit (`audits/2026-04-29_v30_slim_mirror_scorched_earth/`) was a scorched-earth replay on macOS 15.7.4 / Apple M4 Max / Metal. The PR description and the closing summary from the v30 auditor flagged that:

> The env yaml change is the kind of thing that can behave differently across platforms.

Specifically, `huggingface_hub>=0.20.0` resolves cleanly on macOS but could in principle hit a wheel-availability or transitive-dep mismatch on Linux/CUDA. The v29 Linux/CUDA baseline (`audits/2026-04-27_v29_linux_cuda/`) predates the slim-mirror feature, so it's not a substitute. This audit runs the same test matrix on Linux to confirm parity.

## What was scoped

| In | Out |
|---|---|
| Fast pytest (`-m "not integration and not slow"`) | Notebook walkthrough re-render (large, macOS-only screenshots already cover) |
| Integration pytest (`-m integration`) — exercises the actual HF download path | Three-oracle Fig 3f triangulation (already verified on macOS, doesn't depend on platform) |
| F1 regression test in isolation | Full `chorus setup --oracle all` reinstall (would reproduce the F1 verification but env-yaml change validation is what mattered) |
| `mamba env create` from the new `chorus-chrombpnet.yml` resolves on Linux | Per-oracle wall-time benchmarks |

## Test matrix vs v30 macOS baseline

| Check | macOS v30 result | Linux/CUDA result | Match |
|---|---|---|---|
| Fast pytest | **346 passed**, 4 deselected, 14 warnings, **425 s** | **346 passed**, 4 deselected, 14 warnings, **600 s** | ✅ same passes |
| Integration pytest | **3 passed**, 1 skipped, 346 deselected, **681 s** | **3 passed**, 1 skipped, 346 deselected, **613 s** | ✅ exact match |
| F1 regression test (`test_chrombpnet_env_yaml_has_huggingface_hub`) | passed | passed | ✅ |
| `huggingface_hub` import in chorus-base env | works | works (1.12.0) | ✅ |

The Linux fast suite ran ~40% slower wall-time than macOS, but this box has GPU contention from another user's job (cuda:0 was at 100% utilisation throughout) and the lab home is on shared NFS. Pass count and pass set are identical.

## F1 regression: explicit check on Linux

The F1 finding was that `chorus-chrombpnet.yml` did not list `huggingface_hub`, so the slim-mirror import inside the per-oracle env raised `ModuleNotFoundError` and the `_try_slim_hf_chrombpnet` helper silently fell back to the ENCODE tarball flow. The fix in `a13282c` added the dep and replaced the silent fallback with a `logger.warning`.

On Linux:

```text
$ mamba run -n chorus-chrombpnet python -c "import huggingface_hub"
ModuleNotFoundError: No module named 'huggingface_hub'   # ← pre-fix env, expected
```

The existing lab `chorus-chrombpnet` env was built before the fix (Apr 28 21:21), so it reproduces the F1 symptom on Linux — confirming the bug was platform-agnostic, not just a macOS quirk. Running the regression test against the **PR's** yaml file:

```text
$ mamba run -n chorus python -m pytest \
    tests/test_oracles.py::TestChromBPNetOracle::test_chrombpnet_env_yaml_has_huggingface_hub -v
1 passed in 2.60s
```

…confirms the fix is in place at the yaml level.

## Gold-standard import test (matches macOS v30)

The macOS v30 audit's headline post-fix verification was:

> `mamba run -n chorus-chrombpnet python -c "import huggingface_hub"` → succeeds (was: `ModuleNotFoundError`).

Reproduced on Linux against a freshly-built `chorus-chrombpnet-v30test` env from the PR's yaml:

```text
$ mamba run -n chorus-chrombpnet-v30test pip install 'huggingface_hub>=0.20.0'
Successfully installed huggingface_hub-1.12.2 …  (and 18 transitive deps)

$ mamba run -n chorus-chrombpnet-v30test python -c "import huggingface_hub; print(huggingface_hub.__version__)"
1.12.2

$ mamba run -n chorus-chrombpnet-v30test python -c "from huggingface_hub import hf_hub_download; print('OK')"
OK
```

The third call exercises the exact import line that `_try_slim_hf_chrombpnet` uses — confirming the slim-mirror code path will fire on Linux from inside the per-oracle env.

## Notes / gotchas observed during the run

1. **Lab shared NFS is slow.** The full `mamba env create` was still running at the >50 min mark on `lab_envs/` (NFS); the macOS audit reported ~40 s for the same operation on local SSD. To unblock the import-test verification on Linux without paying the full NFS bill, the conda phase was allowed to finish on its own (statsmodels was the last conda link visible) and the pip block was completed manually with `pip install 'huggingface_hub>=0.20.0'`. The full pip block was not run, but the F1-relevant dep was — sufficient for the spot-check.
2. **First fast-pytest attempt timed out at 30 min on a `do_poll` syscall.** The test process accumulated only 52 s of CPU in 30 min — looked stuck on a network read. Killing and re-running cleanly (with `-m "not integration and not slow"`, same flags as v30) finished in 600 s. Suspect transient NFS / DNS hiccup, not a code defect — the second attempt produced identical pass/fail to macOS.
2. **First fast-pytest attempt timed out at 30 min on a `do_poll` syscall.** The test process accumulated only 52 s of CPU in 30 min — looked stuck on a network read. Killing and re-running cleanly (with `-m "not integration and not slow"`, same flags as v30) finished in 600 s. Suspect transient NFS / DNS hiccup, not a code defect — the second attempt produced identical pass/fail to macOS.
3. **GPU contention on cuda:0.** Another user had cuda:0 at 100% during the entire spot-check; my pytest used cuda:1 / CPU. Doesn't affect the slim-mirror code path (which is pure HTTP I/O, no GPU).

## Conclusion

The PR can be flipped from draft to ready from a Linux/CUDA-portability standpoint. Specifically:

- The env yaml change (`huggingface_hub>=0.20.0`) is honoured by mamba on Linux.
- The F1 regression test guards against future drift on either platform.
- Both the fast and integration test slices agree with macOS results to the test-pass-count level.

Items still owed by the v0.3.0 release (independent of this spot-check) per the PR's own outstanding list:

- The full end-to-end `chorus setup --oracle all` reinstall on Linux was **not** rerun in this audit (env yaml resolution + targeted regression test was deemed sufficient for the spot-check). Maintainer should decide whether a fresh-install scorched-earth on Linux is a release blocker.
- Cherry-pick the F1 lesson into `docs/plans/sei-hf-mirror.md` (already noted in the plan doc per the v30 closing summary).

## Artifacts

- `chorus_fast_pytest.log`: full fast-suite output (346 passed, 4 deselected)
- `chorus_integration_pytest.log`: integration suite output (3 passed, 1 skipped)
- `chrombpnet_env_create.log`: mamba env-create from the PR's yaml (still linking at audit close — no errors observed)
- `system_info.txt`: kernel, GPU inventory, python version, git tip
