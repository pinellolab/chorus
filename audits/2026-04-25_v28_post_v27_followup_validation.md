# v28 — Post-v27-followup full scorched-earth validation, 2026-04-25

Validation pass on Linux/CUDA after the v27 follow-up commits
(`6fcd82d`, `f60573a`, `9100964`) landed. Full teardown → README
TLDR install → 6-oracle setup → all 3 notebooks → MCP smoke → tests.

**No new findings.** Every fix from v27/v27-followup behaves as
documented on a truly clean state.

## Scope

- **Linux x86_64 + CUDA**, fresh repo HEAD = `9100964`.
- True scorched earth before start: 7 conda envs removed,
  `~/.chorus/`, `genomes/`, `downloads/`, `~/.cache/huggingface/hub/`
  (chorus-relevant entries only). HF token preserved separately and
  re-injected via `HF_TOKEN` env var.

## Timeline

| Step | Started | Ended | Δ | Notes |
|---|---|---|---|---|
| Nuke 7 envs + caches | 04:04 | 04:57 | 53 min | PanFS small-file cost (ext4 would be ~3 min) |
| `mamba env create -f environment.yml` | 04:57 | ~05:24 | 27 min | Linking phase dominated by metadata |
| `python -m pip install -e .` (1st try) | 05:24 | 05:24 | 16 s | **transient OSError on importlib_metadata mid-upgrade**; pip retry succeeded in 54 s |
| `chorus setup` (all 6) | 05:25 | 08:22 | **2:57** | All 6 oracles ready end-to-end with HF token |
| `chorus health` (all 6) | 08:25 | 08:27 | 2 min | 6/6 ✓ Healthy |
| Quickstart notebook | 08:31 | 08:34 | 3 min | 34 cells, 0 err, 7 plots |
| Advanced notebook | 08:34 | 08:44 | 10 min | 57 cells, 0 err, 16 plots |
| Comprehensive notebook | 08:44 | 09:01 | 17 min | 38 cells, 0 err, 9 plots |
| Pytest fast suite | 08:32 | 08:35 | 3 min | 337 / 1 / 0 / 4 deselected |
| MCP smoke | 09:02 | 09:02 | <1 s | 22 tools, list_oracles → 6 oracles |

## v27 follow-up fixes — all VALIDATED

### `9100964` ChromBPNet HepG2 prefetch — ✓ confirmed

After fresh `chorus setup --oracle chrombpnet`:

```
$ ls /PHShome/lp698/chorus/downloads/chrombpnet/
DNASE_HepG2
DNASE_K562
```

Both cell-type subdirs present. Before this fix, only `DNASE_K562`
would land. The `comprehensive_oracle_showcase.ipynb` notebook (which
loads ChromBPNet HepG2 mid-run) executed without timeouts on the
fresh state — direct evidence the prefetch fix works.

### `9100964` `chorus --version` — ✓ confirmed

```
$ chorus --version
chorus 0.1.0
```

(Was `error: unrecognized arguments: --version` before the fix.)

### `f60573a` Advanced + comprehensive notebooks pass — ✓ confirmed

| Notebook | cells | executed | errors | plots |
|---|---|---|---|---|
| single_oracle_quickstart | 34 | 34 | **0** | 7 |
| advanced_multi_oracle_analysis | 57 | 57 | **0** | 16 |
| comprehensive_oracle_showcase | 38 | 38 | **0** | 9 |

All 32 plots render inline. Zero CellExecutionError, zero
WARNING-as-error tracebacks.

### `5acee71` README `python -m pip install -e .` — ✓ confirmed

The bare `pip install -e .` is no longer in README/CONTRIBUTING.
Following the README literally now picks up the env's pip.

### `2ecd998` FANTOM CAGE track validator — ✓ confirmed

The Quickstart notebook's `['ENCFF413AHU', 'CNhs11250']` cell ran
without `InvalidAssayError`. Track index 4828 (CAGE: K562) is
reachable.

### `68f5cc3` chorus-sei.yml relaxed pin — ✓ confirmed

`chorus setup --oracle sei` env-create finished in **9 min** (vs
50+ min hang on the old `pytorch>=1.13.0,<2.0.0 + cudatoolkit=11.7`
constraints). Resolved to a modern PyTorch 2.x + cuda toolchain.

## One transient observation (not a chorus bug)

**Pip OSError mid-install** on first attempt:
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such
file or directory: '<env>/site-packages/importlib_metadata-8.8.0.dist-info/METADATA'
```

Pip was upgrading importlib_metadata 8.7.1 → 8.8.0; the unpack of
the new dist-info hit a missing-file race. Retry succeeded on the
same env, no manual cleanup needed.

Likely PanFS small-file races during pip's atomic-rename dance. Not
chorus's fault. Worth noting in a "if you hit this on HPC, retry"
README footnote.

## Side-by-side: v27 vs v28 timings

| Step | v27 (this Linux/CUDA) | v28 (this Linux/CUDA) | Note |
|---|---|---|---|
| Nuke 7 envs | ~60 min | 53 min | similar |
| Fresh env create | 30 min | 27 min | similar |
| `chorus setup` 5 oracles | ~75 min | (n/a — single-pass v28) | v27 ran serially |
| `chorus setup` 6 oracles | (sei broken) | **2:57** | v27 hung on sei before fix |
| Setup → Healthy | n/a | 9 min after setup exits | clean |

## Scorecard

| Surface | Result |
|---|---|
| `mamba env create -f environment.yml` | ✓ |
| `python -m pip install -e .` | ✓ (one transient pip OSError, retry ok) |
| `chorus --help` / `--version` | ✓ |
| `chorus setup` (HF token via env var) | ✓ 6/6 in 2:57 |
| `chorus health` | ✓ 6/6 |
| `downloads/chrombpnet/{K562,HepG2}` | ✓ both present |
| `chorus genome download` (auto by setup) | ✓ |
| Per-track CDFs (auto-download from HF) | ✓ all 6 oracles |
| `single_oracle_quickstart.ipynb` | ✓ 0 err / 7 plots |
| `advanced_multi_oracle_analysis.ipynb` | ✓ 0 err / 16 plots |
| `comprehensive_oracle_showcase.ipynb` | ✓ 0 err / 9 plots |
| MCP `chorus-mcp` smoke (fastmcp Client) | ✓ 22 tools, 6 oracles |
| Pytest fast suite (`-m "not integration"`) | ✓ 337 / 1 / 0 |

## Bottom line

Every fix from v27 + v27-followup is reproducible on a clean
Linux/CUDA install. All 6 oracles healthy, all 3 notebooks pass with
inline plots, MCP server registers 22 tools. **No new findings.**
The v22 → v27 → v27-followup arc has driven the new-user install
path from "breaks 3 ways in the first 30 min" to "works end-to-end,
2:57 fresh".
