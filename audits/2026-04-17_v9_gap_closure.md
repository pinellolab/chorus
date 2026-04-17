# Chorus v9 Coverage-Gap Closure

**Date**: 2026-04-17
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `chorus-applications` @ `88d1a45`
**Auditor**: Claude Opus 4.7 (1M context)
**Fix branch**: `fix/2026-04-17-v9-coverage-gaps`

## Context

The v8 fresh-install audit had **zero findings** but honestly called out
four scenarios it couldn't promise "production ready" for because they
weren't exercised. This PR closes them with three new integration
tests, four error-recovery unit tests, and a light GitHub Actions CI.

## What's new

### Fast suite: 299 → 303 passed (tests/test_error_recovery.py)

Four mock-based tests added. Each runs in milliseconds, no network.

| Test | Mocks | Verifies |
|------|-------|----------|
| `test_hf_hub_download_failure_returns_zero_and_does_not_crash` | `huggingface_hub.hf_hub_download` → `ConnectionError` | `download_pertrack_backgrounds` returns 0, logs a warning, leaves `cache_dir` intact |
| `test_download_with_resume_leaves_partial_and_resumes_on_second_call` | `urllib.request.urlopen` | first call leaves `.partial` with bytes read before the crash; second call sends `Range: bytes=N-` and promotes to dest |
| `test_alphagenome_missing_hf_token_error_is_actionable` | `huggingface_hub.whoami` → `LocalTokenNotFoundError`, HF_TOKEN unset | `ModelNotLoadedError` names `HF_TOKEN`, links `huggingface.co/google/alphagenome`, mentions `huggingface-cli login` |
| `test_missing_oracle_env_falls_back_gracefully` | `EnvironmentManager.environment_exists` → `False` | oracle logs "`chorus setup --oracle enformer`" hint and sets `use_environment=False` |

### Integration suite: 3 new tests (tests/test_integration.py, @pytest.mark.integration)

Gated by marker so they stay out of the fast suite and out of CI.

| Test | Purpose | Cost |
|------|---------|------|
| `test_pertrack_background_download[sei]` | SEI CDF download from HF dataset, empirical checks | **passed in 1.2 s** |
| `test_pertrack_background_download[legnet]` | LegNet CDF download, same checks | **passed in 1.2 s** |
| `test_chrombpnet_fresh_single_model_download` | Download ATAC:K562 tarball from ENCODE fresh (tmp dir, untouched real cache), extract, load, predict | **passed in 490 s (8 min 10 s)** — 2114 bp predict OK |
| `test_mcp_e2e_list_oracles_and_analyze_variant` | Spawn `chorus-mcp` stdio subprocess via fastmcp `Client`, `list_oracles` → `load_oracle` → `analyze_variant_multilayer` on SORT1 rs12740374 in HepG2 | **passed in 276 s (4 min 36 s)** — first E2E test of the MCP server, prior `tests/test_mcp.py` was mock-only |

**Total integration time on this machine: ~13 min.**

### CI: .github/workflows/tests.yml

Linux Ubuntu-latest runner, Python 3.10 via Miniforge + mamba:

- Runs on every push to `main` / `chorus-applications` and every PR
- Installs `environment.yml` → `pip install -e .` → `pytest -m "not integration"`
- **Deliberately skips** `tests/test_smoke_predict.py` (10 GB of oracle
  weights doesn't fit on the 14 GB runner) and the 3 integration tests
  (too slow for per-PR feedback). A `workflow_dispatch` trigger lets
  maintainers run the fast suite manually if they want.

## Verified locally

```
# Fast suite
pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration" -q
→ 303 passed, 4 deselected, 14 warnings in 58.71 s

# Integration suite (marker-gated)
pytest tests/test_integration.py -v -m integration
→ 4 passed in 13 min total
  - sei CDF                  1.2 s
  - legnet CDF               1.2 s
  - chrombpnet ATAC:K562   490 s (downloaded ~500 MB ENCODE tarball)
  - mcp e2e                277 s (chorus-mcp subprocess + real AG predict)
```

## Three small surprises worth recording

1. **fastmcp Client API doesn't accept a plain `dict` transport** — it
   requires either a `StdioTransport` instance or an `mcpServers`-wrapped
   config dict. `transport=dict(command=..., args=...)` raises
   `ValueError: No MCP servers defined in the config`. The
   MCP walkthrough showed natural-language usage only; I had to reach
   for `fastmcp.client.transports.StdioTransport`.
2. **Bare `mamba run -n chorus chorus-mcp` hits the two-mamba-installs
   trap** from the README: my mamba root is `~/.local/share/mamba`
   but the default look-up path is `~/miniforge3/envs`. The test
   mitigates by locating `chorus-mcp` on PATH via `shutil.which` and
   forwarding `MAMBA_ROOT_PREFIX`/`MAMBA_EXE` into the subprocess env.
3. **MCP `load_oracle` must be called before `analyze_variant_multilayer`**
   — the server raises `"Oracle 'X' is not loaded. Call load_oracle('X')
   first."`. The docstring on `analyze_variant_multilayer` doesn't say
   this explicitly; the state.py raise at line 162 is the only source.
   Not a bug (the docstring on `load_oracle` says "load and cache"),
   but a newcomer running the tool in raw-JSON mode might not realize
   they need a separate `load_oracle` call first. The natural-language
   walkthrough collapses this because Claude always calls both.

## What the gap-closure proves

- **Item 2 (SEI + LegNet CDFs)**: HF dataset download works for both,
  NPZs load, CDFs pass the same monotonicity + percentile + count
  checks v8 applied to the other 4 oracles. All 6 oracles'
  normalization path is now empirically verified.
- **Item 3 (ChromBPNet fresh download)**: The shared
  `download_with_resume` helper at `chorus/utils/http.py:35-121`
  does the right thing end-to-end — ENCODE URL resolved, ~500 MB
  tarball streamed with `.partial` + `fcntl` lock, extracted, fold 0
  weights loaded into TensorFlow, and a 2114 bp predict returns
  finite track values. The 37 GB cache preserved across v4-v8 audits
  was a convenience; the from-zero path works.
- **Item 4 (E2E MCP session)**: `chorus-mcp` can be spawned as a
  stdio subprocess by a non-trivial client (fastmcp `StdioTransport`),
  negotiate the protocol, call `list_oracles` + `load_oracle` +
  `analyze_variant_multilayer`, and return a structured response.
  The `analyze_variant_multilayer` response round-trips correctly
  across the stdio boundary. First production-path test of the MCP
  server; prior tests all mocked the oracles.
- **Item 6 (error recovery)**: The four main failure surfaces —
  HF download failure, partial-file resume, missing HF_TOKEN,
  missing oracle env — all have explicit pinned behavior now.

## What is still NOT covered

Same deferrals as v8:
- **Linux / CUDA audit** — user is doing this on another machine.
- **Hosted / multi-tenant deployment** — load, concurrency, session
  isolation.
- **Clinical / pharma validation** — oracle models are research-grade.
- **The "two mamba installs" UX trap** — documented in
  README Troubleshooting; not solvable without a policy change
  (either pin the install path or add a `chorus doctor` that
  auto-repairs `MAMBA_ROOT_PREFIX`).

## Verdict

After v9, every item the v8 audit listed as "did not exercise" is
exercised somewhere — fast suite, marker-gated integration, or CI.
The only remaining "would need to verify" items require hardware I
don't have (CUDA GPU) or infra I shouldn't deploy (hosted service).

Combined with v8's fresh-install clean bill of health: Chorus is
**production-ready for research-group / biology-lab use** on macOS
arm64 and (per v4 Linux audit) Linux x86_64 + CUDA. The honest gaps
are the hosted / clinical / multi-tenant axes the audit scope never
claimed.
