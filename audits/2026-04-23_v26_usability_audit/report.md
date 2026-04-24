# 2026-04-23 v26 — Usability audit

Follow-up to v24/v25 install + feature audits, this pass focuses on
**usability** — the subjective quality of what a fresh user sees,
types, and reads. Scope: CLI, Python API, MCP server, and README
coherence. Three inputs: two Explore agents (error-sweep + README
claim verification) and ~14 live CLI/API probes on a working install.

## Verdict

**Broadly good for a research tool.** TLDR install works, the Python
API's top-level errors (`create_oracle` typo, `predict` before load)
are self-recovering, and the README accurately reflects the code —
all 7 README claims I spot-checked PASSED. One **P0** bug worth
fixing before wider announcement; a handful of **P1**s that are
quick wins; **P2**s are polish.

## 1. README ↔ behavior coherence

| Claim (README line) | Actual | Status |
|---|---|---|
| `git clone` + `mamba env create` + `mamba activate` + `pip install -e .` (L19-23) | `environment.yml` + `setup.py` with `chorus`/`chorus-mcp` entry points exist | ✓ |
| bare `chorus setup` installs all 6 (L30) | `cli/main.py:38` routes `args.oracle is None` → `setup_all_oracles` | ✓ |
| HF token 3-step flow (L35-38) | `_tokens.py:38-41` prints same URLs in same order; `getpass` for hidden paste | ✓ |
| `chorus setup --oracle enformer` starter (L41) | Argparse accepts it; tested in v25 | ✓ |
| Python snippet signatures (L45-70) | All calls match `__init__.py:109`, `base.py:144`, `base.py:275` | ✓ |
| `claude mcp add chorus -- mamba run -n chorus chorus-mcp` (L77) | `chorus-mcp` console script registered (setup.py:45) | ✓ |
| "22 tools" (L74) | Counted 22 user-facing @mcp.tool() decorators in `mcp/server.py` | ✓ |

Nothing to fix here.

## 2. Findings by surface

### CLI (`chorus …`)

| # | Severity | Issue | Evidence |
|---|---|---|---|
| 1 | **P1** | `chorus setup --oracle fakeoracle` errors with "Environment file not found: environments/chorus-fakeoracle.yml" and exits 0. Doesn't list valid oracle names. | Probe 4 |
| 2 | **P1** | `chorus health --oracle fakeoracle` reports "✗ fakeoracle: Unhealthy" + "Environment does not exist" but exits 0 despite the `✗`. Breaks `chorus health && echo ok`. | Probe 5 |
| 3 | **P1** | `chorus genome download fakegenome` logs error + "Available genomes: hg38, hg19, …" (good!) but still exits 0. | Probe 7 |
| 4 | **P1** | Every CLI command leads with 2 INFO lines ("Found mamba via MAMBA_EXE…", "Detected platform: Darwin arm64 …") before the user-relevant output. Noise on `--help`, `list`, `health --oracle X`. | Probes 2, 5, 10 |
| 5 | **P2** | Inconsistent error style across `_tokens.py`, `main.py`, oracle files: periods/no-periods, `logger.error + return False` vs raise, with/without fix hints. | Agent 1 sweep |
| 6 | ✓ | `chorus setup --help` output is comprehensive, names both tokens, documents every flag (`--force`, `--no-weights`, `--no-backgrounds`, `--no-genome`, `--hf-token`). | Probe 11 |
| 7 | ✓ | `chorus remove --oracle X` prompts `[y/N]`, defaults to N, prints "Removal cancelled." on n. Good model for destructive-op UX. | Probe 6 |
| 8 | ✓ | Fix 2 (new HF numbered instructions block) fires cleanly on non-TTY halt. | v25 run |

### Python API

| # | Severity | Issue | Evidence |
|---|---|---|---|
| 9 | **P0** | Invalid track ID (e.g. `'ENCFF999BADID'`) raises `TypeError: 'NoneType' object is not subscriptable` after a lone `WARNING - Identifier 'ENCFF999BADID' not found in metadata` line. User has no actionable message and no pointer to `list_tracks` / metadata search. | Probe 13 |
| 10 | **P1** | TF/absl startup noise in every oracle load: 3-5 lines of `pluggable_device_factory`, `Fingerprint not found`, `path_and_singleprint metric could not be logged`. Easy silence via `TF_CPP_MIN_LOG_LEVEL=3` set in oracle env. | Probes 13, 14 |
| 11 | **P1** | `OracleBase._setup_environment()` swallows `Exception` with a warning and continues — user may later call `predict()` on a half-initialized oracle and get downstream confusion, not a clean "env not available" raise. | Agent 1: `base.py:112` |
| 12 | **P1** | Subprocess template files (`enformer_source/templates/predict_template.py`, `borzoi_source/.../predict_template.py`) use bare `raise Exception("Some assay IDs not found …")`. Not wrapped in ChorusError → generic traceback. | Agent 1 |
| 13 | **P1** | `core/interval.py` contains `raise Exception('')` (empty message). | Agent 1 |
| 14 | **P2** | `result.py` uses `assert` for resolution constraints — user sees `AssertionError` not a custom exception. | Agent 1 |
| 15 | ✓ | `create_oracle('enfromer')` (typo) → `ValueError: Unknown oracle: enfromer. Available: ['enformer', 'borzoi', 'chrombpnet', 'sei', 'legnet', 'alphagenome']`. **Gold standard**. | Probe 8 |
| 16 | ✓ | `oracle.predict()` before load → `ModelNotLoadedError: Model not loaded. Call load_pretrained_model first.` | Probe 9 |
| 17 | ✓ | Invalid chromosome → `InvalidRegionError: Chromosome chrZZ not found in <fasta path>`. Names the file. | Probe 14 |

### MCP

| # | Severity | Issue | Evidence |
|---|---|---|---|
| 18 | **P1** | `@_safe_tool` decorator in `mcp/server.py:~150` catches all exceptions and returns `{"error": "...", "error_type": "..."}` — MCP client never sees the stack trace, hiding which line failed. Debugging relies on server stderr. | Agent 1 |
| 19 | **P2** | `_parse_region` / `_parse_position` raise `ValueError` (not `ChorusError`) with inconsistent messages — some say "expected chrN:start-end", others don't. | Agent 1 |
| 20 | **P2** | `list_tracks(oracle)` on a metadata import failure returns empty list silently (no log of which oracle failed). | Agent 1 |

### Cross-cutting

| # | Severity | Issue |
|---|---|---|
| 21 | **P2** | `--verbose` on `chorus health` / `list` prints extra info but doesn't actually enable `logging.DEBUG`. Users expecting more detail have to `export PYTHONPATH=… ; python -m logging…` instead. |
| 22 | **P2** | Subprocess timeout raises `CalledProcessError` with truncated stderr and no hint to set `CHORUS_NO_TIMEOUT=1`. |
| 23 | ✓ | `download_with_resume` uses `urlopen(timeout=60)` which already acts as a per-read socket watchdog; confirmed fires on SSL mid-stream failures (v25 sei download). |

## 3. Recommendations (in order of impact)

1. **P0 #9 — Invalid track ID.** Wrap the metadata lookup in `OracleBase.predict` (or per-oracle predict): if `lookup(assay_id)` returns None, raise `InvalidAssayError(f"Track '{assay_id}' not found. Use oracle.list_tracks() or the MCP list_tracks tool to discover valid IDs for '{oracle_name}'.")` *before* hitting the model. ~10 lines, covers all 6 oracles from one place.

2. **P1 #1-3 — Exit codes.** Wherever the CLI logs `✗` or `ERROR`, return 1. Four call sites in `chorus/cli/main.py`. Trivial fix; unblocks shell scripting.

3. **P1 #4 — Leading INFO noise.** Demote "Found mamba via MAMBA_EXE" and "Detected platform" to `logger.debug`. They're diagnostic, not user-facing. One-line `.setLevel(...)` or downgrade the log calls.

4. **P1 #10 — TF/absl noise.** Set `TF_CPP_MIN_LOG_LEVEL=3` in `chorus-enformer` and `chorus-chrombpnet` env activation scripts, or export it from the wrapper script that launches the subprocess. Silences 3-5 lines of boot spam per prediction.

5. **P1 #11 — Silent env swallow.** In `OracleBase._setup_environment`, if env setup fails, set a flag. On subsequent `predict()`/`load_pretrained_model()`, raise `EnvironmentNotReadyError(f"chorus-{name} env failed to initialize. Run 'chorus setup --oracle {name}' or check 'chorus health --oracle {name}'.")` instead of letting a misleading downstream error propagate.

6. **P1 #12-13 — Bare `raise Exception`.** Replace with domain-specific exceptions (`InvalidAssayError`, `ChorusError`). ~5 call sites.

7. **P1 #18 — MCP `_safe_tool`.** Include `"traceback": traceback.format_exc()` in the error dict (or optionally, guarded by `CHORUS_MCP_DEBUG=1` env var). The MCP client can choose whether to display it; currently it can't.

8. **P2** — everything else. Polish pass; do once the above are in.

## 4. Nothing needs fixing

- README TLDR coherence (all 7 claims PASS).
- Python API's three top-level error paths (`create_oracle`, `predict` pre-load, bad chromosome) are gold-standard.
- `chorus remove` confirmation UX.
- `download_with_resume` socket watchdog (already present via `urlopen(timeout=60)`).
- MCP tool count matches README's claim of 22.

## Sign-off

Nothing in this audit blocks shipping to the current research-user
audience. The P0 (invalid track ID → bare TypeError) is worth fixing
before wider announcement since it hits users on their second or third
prediction call. The other P1s are polish that would pay off
disproportionately in first-impression quality.
