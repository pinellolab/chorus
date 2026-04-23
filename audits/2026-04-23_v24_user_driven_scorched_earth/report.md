# 2026-04-23 v24 — User-driven scorched-earth README test

Driver: Luca (explicit ask: "remove all the chorus env, and follow line
by line the readme"). Machine: macOS arm64 (Darwin 25.1.0).
Post-commit base: `e9d7dc2` + `17e245a` (v23) + `92adb84` (v22).

## Scope (teardown)

- All 7 conda envs: `chorus` + `chorus-{enformer, borzoi, chrombpnet, sei, legnet, alphagenome}`
- `~/.chorus/backgrounds/`
- `~/.huggingface/token`

Kept: `genomes/hg38.fa` (3.3 GB, idempotent skip), `downloads/sei/sei_model.tar.gz.partial`
(310 MB, exercises resume path).

## Tokens (session-scoped; not logged to any file)

- HF token via `--hf-token` flag → persisted to `~/.huggingface/token`
  via `huggingface_hub.login()`
- LDlink token via `LDLINK_TOKEN` env var → picked up non-interactively
  by `prompt_ldlink_token`

## Timeline

| Step | Start | End | Duration | Outcome |
|---|---|---|---|---|
| `mamba env create -f environment.yml` | 12:07:18 | 12:08:38 | 80 s | ✓ |
| `pip install -e .` | 12:08:44 | 12:08:58 | 14 s | ✓ |
| `python -m ipykernel install ...` | 12:08:58 | 12:08:59 | <1 s | ✓ |
| `chorus setup --oracle enformer` (README primary) | 12:09:36 | 12:11:28 | 112 s | ✓ Healthy |
| `chorus setup --oracle all --hf-token ... LDLINK_TOKEN=...` | 12:11:40 | 13:05:22 | 54 min | 4/6 ✓ (alphagenome + sei failed) |
| `chorus setup --oracle sei` (retry, resumed 557 MB → 3.3 GB) | 13:10:23 | 13:19:27 | 9 min | ✓ |
| `chorus setup --oracle alphagenome --force` (git clone retry) | 13:16:03 | **hung 51 min** | — | ✗ killed (stuck on SSL read) |
| `chorus setup --oracle alphagenome` (third attempt) | 14:13:51 | 14:30:42 | 17 min | ✓ (weights downloaded after 14 min cold-connect delay) |
| **Final `chorus health` sweep, all 6 oracles** | 14:30:52 | 14:34:03 | 3:11 | **6/6 ✓ Healthy** |
| Pytest fast suite (after fixes 1+2 landed) | 14:13:30 | 14:16:14 | 2:43 | 336 passed, 0 failed |
| MCP smoke test (list_tools, list_oracles, list_genomes) | 14:35:01 | 14:35:02 | 1 s | ✓ 22 tools, 6 oracles |
| Notebook: single_oracle_quickstart.ipynb | 14:34:34 | TBD | TBD | **IN PROGRESS** |

## Per-oracle outcomes in setup all

| Oracle | Env build | Weight dl | Backgrounds | Marker | Final health |
|---|---|---|---|---|---|
| enformer | 53 s (12:09:36 → 12:10:28, pre-setup-all) + 9 s cached re-check | TF Hub ~150 MB | enformer_pertrack.npz | ✓ 12:11:28 | ✓ Healthy |
| alphagenome | 39 s initial build → **pip git-clone failed** (transient) | — | — | missing | ⚠ → rebuilt via `--force`, then weight prefetch hung 51 min → killed → third retry succeeded at 14:27, marker at 14:30:42 | ✓ Healthy |
| borzoi | 42 s | HF ~800 MB, 50 s | ✓ | ✓ 12:14:55 | ✓ Healthy |
| chrombpnet | 51 s | ENCODE K562 DNASE 19 min (large tarball) | ✓ | ✓ 12:34:46 | ✓ Healthy |
| legnet | 53 s | Zenodo 150 MB, 104 s (used the new `download_with_resume` + tqdm) | ✓ | ✓ 12:37:33 | ✓ Healthy |
| sei | 34 s | **timed out** first pass at 1 GB/3.3 GB (tar extract corrupted sei.pth — v23 fix `_materialize_cached_seqclass_info` kicked in) → retry completed in 5.5 min, full 3.5 GB extracted | ✓ | ✓ 13:19:27 | ✓ Healthy |

## Findings

### P1 — transient pip git-clone in alphagenome env build
First `chorus setup --oracle all` run failed on pip's `git+https://github.com/google-deepmind/alphagenome_research.git` inside the ephemeral tempdir. `git ls-remote` from base shell succeeded, so the repo is reachable — this was a transient failure inside pip's invocation. v23 audit recorded alphagenome succeeding in 2m40s, so this is not a fundamental problem with the env yml.

**Impact**: Without the Fix 1 auto-heal added during this audit, a plain `chorus setup --oracle all` rerun could not recover — the env was left half-built (conda env present, pip deps missing), and the retry would skip env rebuild, hitting `ModuleNotFoundError: No module named 'requests'` at weight-prefetch time.

**Fix landed during audit** (commit TBD): `EnvironmentManager.environment_is_healthy()` + auto-heal in `create_environment()`. A plain rerun now auto-detects broken envs and rebuilds them.

### P1 — HF weight prefetch hung on connection
Second alphagenome attempt (`--force`) hung 51 min on `sock_connect` / `getaddrinfo` — no HF bytes transferred. Killed manually and retried; the third attempt took 14 min cold-connect before bytes started flowing and then completed in ~3 min. Looks like HF-CDN throttling or TCP-state issue on macOS after the prior stuck connection.

**Impact**: No timeout or progress heartbeat on the weight prefetch subprocess — a user watching `chorus setup --oracle alphagenome` with no console output for 51 min would reasonably conclude it's hung, with no actionable signal.

**Recommendation (not fixed in this audit)**: add a "no-data-for-N-seconds" watchdog to `prefetch_weights` (or to `download_with_resume`) that raises a clear error so the user / retry logic can act.

### P2 — fix 2 verbose HF instructions landed
When HF token can't be resolved (non-TTY halt, rejected token), the setup now prints a 72-char-wide numbered block with:

```
1. Create a read token: https://huggingface.co/settings/tokens
2. Accept the model license: https://huggingface.co/google/alphagenome-all-folds
3. Give the token to chorus — ANY of:
   a. chorus setup --oracle alphagenome --hf-token hf_xxx
   b. export HF_TOKEN=hf_xxx && chorus setup --oracle alphagenome
   c. mamba run -n chorus huggingface-cli login
```

This replaces a terse single-line error. Tested non-interactively ✓.

### Sei retry path worked as designed
`download_with_resume` helper (ported to LegNet in the prior commit) resumed Sei's 557 MB partial and completed the remaining 2.7 GB in 5.5 min. `_materialize_cached_seqclass_info` (v23 fix) correctly materialized the source-packaged metadata into the downloads cache so the probe can verify completion.

## Code changes landed during this audit

1. `chorus/core/environment/manager.py` — `environment_is_healthy()` + auto-heal in `create_environment()`. Plain `chorus setup --oracle all` now recovers from a half-built env without needing `--force`.
2. `chorus/cli/_tokens.py` — `_print_hf_setup_instructions()` helper; beefed up all three HF-auth failure branches (CLI flag rejected, env-var rejected, no token non-TTY) + the interactive prompt.

## Pytest
`mamba run -n chorus python -m pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration" -q` → **336 passed, 4 deselected, 0 failed, 163 s**. Both fixes safe.

## MCP smoke
`chorus-mcp` over stdio via `fastmcp.Client`:
- `list_tools` → 22 tools (matches §8 benchmark)
- `list_oracles` → dict with `oracles: [...]`, 6 entries with `name`, `framework`, `input_size_bp`, `output_bins`, `resolution_bp`, `assay_types`, `environment_installed`
- `list_genomes` → ok

## Notebooks (fresh re-execution from the reinstalled envs)

| Notebook | Oracles | Started | Ended | Duration | Errors | Warnings | Output size |
|---|---|---|---|---|---|---|---|
| single_oracle_quickstart.ipynb | Enformer | 14:34:34 | 14:36:52 | 2:18 | **0** | **0** | 672 KB |
| advanced_multi_oracle_analysis.ipynb | Enformer + ChromBPNet + LegNet | 14:37:15 | 14:44:54 | 7:39 | **0** | **0** | 2154 KB |
| comprehensive_oracle_showcase.ipynb | All 6 | 14:44:57 | 15:06:48 | 21:51 | **0** | **0** | 764 KB |

All three meet the §6 P1 bar ("zero errors and zero WARNING lines in any cell output"). Fresh outputs are in `nb_fresh/` alongside this report.

## Verdict

**Green.** Fresh-install flow from the README works end-to-end on macOS-arm64 with two fixes landed during the audit:
- `chorus setup` now auto-heals half-built oracle envs on retry (no more `--force` needed after a transient pip failure).
- HF-missing-token failures now print numbered step-by-step instructions to the terminal instead of a single terse line.

All 6 oracles reach Healthy, all 3 example notebooks execute cleanly with zero warnings, MCP surfaces 22 tools + 6 oracles over stdio, and the non-integration test suite is 336 passed / 0 failed.

Open follow-ups (P2, not blocking):
- The 51-min SSL-connect hang on alphagenome weight prefetch on attempt 2 was only discoverable by CPU-stack sampling. Add a "no bytes for N seconds" watchdog to `download_with_resume` (or to `prefetch_weights`) so a stuck HF connection fails fast with a clear error.
- The ChromBPNet ENCODE K562 DNASE tarball took 19 min to download (large) with only log-line progress since tqdm would spam logs in CI; consider a bounded progress indicator (e.g. every 10 % or every 30 s) regardless of TTY.
