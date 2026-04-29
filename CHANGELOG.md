# Changelog

All notable changes to Chorus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Cell-type discovery default ranking is now `alt_x_abs_effect`**
  (was `abs_effect` = raw |log2FC|). Investigating the SORT1 rs12740374
  example revealed that `|log2FC|` over a 501 bp window with
  `pseudocount=1.0` systematically rewards cell types with closed
  baseline chromatin: when `ref` is near zero, creating a *de novo* TF
  binding site produces a huge fold-change even when the absolute
  alt-allele activity is modest. For SORT1, the well-known HepG2
  enhancer (alt_sum=1571, top-2 of 472 cell types by absolute alt
  signal) was buried at rank #59, while three closed-baseline
  fibroblast/epithelial cell types (alt_sum 161–378) took the top
  three slots.

  The new default `alt_x_abs_effect = alt_value × |log2FC|` rewards
  both effect magnitude and final activity, recovering HepG2 #1 and
  liver lobes #2–#5 — matching the canonical SORT1 biology. The old
  metric is still available via `ranking_metric="abs_effect"`, and a
  filtered variant `ranking_metric="abs_effect_min_ref"` lets callers
  apply a baseline-activity floor.

  The MCP tool `discover_variant_cell_types` and the Python helpers
  `discover_cell_types` / `discover_and_report` all expose the new
  parameter; the SORT1 cell-type-screen example was regenerated with
  the new default.

## [0.2.1] — 2026-04-28

### Fixed

- **Disk-size claim for `--all-chrombpnet` was off by ~2.5×.** The
  v0.2.0 README and CLI help text said the opt-in full ChromBPNet
  prefetch needed ~30 GB additional / ~60 GB total. A user actually
  running it had to kill the install when disk filled up. Re-measured
  on a freshly-extracted ENCODE model: per-model is ~720 MB tarball
  + ~1.1 GB extracted (5 fold ensembles) = **~1.8 GB on disk**. With
  42 ChromBPNet ATAC/DNase models that's **~76 GB** for the full
  ChromBPNet weights alone, plus ~410 MB for all 744 BPNet/CHIP TF
  models. **Total `--all-chrombpnet` install footprint is now
  documented as ~100 GB** (was ~60 GB). Default fast-path install
  (~25 GB, K562+HepG2 DNase only) is unaffected.

  Updated everywhere the old number appeared: `README.md`,
  `chorus/cli/main.py` (`--all-chrombpnet` --help text),
  `chorus/cli/_setup_prefetch.py`, and the two multi-oracle
  notebook intros.

## [0.2.0] — 2026-04-27

This release is the cumulative output of the v22 → v29 audit chain
(spring 2026): six fresh-install audits, three scorched-earth replays
on macOS arm64 + Linux/CUDA, and the BPNet/CHIP CDF rebuild that brings
ChromBPNet's percentile-normalisation coverage from 24 → 786 tracks.

### Added

- **ChromBPNet/BPNet ENCODE catalogue expansion (PR #50)** — `chrombpnet_globals.py`
  now exposes all 42 ENCODE-published ChromBPNet ATAC/DNase models
  (was 24) plus the full 744 BPNet TF×cell-type models from the
  JASPAR_DeepLearning 2026 release. New `iter_unique_models()` and
  `iter_unique_bpnet_models()` helpers dedupe by ENCFF / (TF,cell)
  for callers that want to iterate the catalogue.
- **786-track ChromBPNet CDF NPZ on HuggingFace (PR #52, PR #53)** —
  `lucapinello/chorus-backgrounds @ c1e5fc1` now contains effect /
  summary / perbin CDFs for all 786 tracks (42 ATAC/DNASE + 744
  BPNet/CHIP). Auto-downloaded by `chorus setup --oracle chrombpnet`.
- **Sharded background-build pipeline (PR #51, PR #53)** — new
  `--shard N --shard-of M` flags on `build_backgrounds_chrombpnet.py`
  + `--part merge-shards` aggregator + the `scripts/run_bpnet_cdf_build.sh`
  6-GPU orchestrator. Cuts a full BPNet rebuild from ~37 h on 1 GPU
  to ~6 h across 6 GPUs.
- **Incremental CDF append (PR #51, PR #52)** —
  `PerTrackNormalizer.append_tracks()` deduplicates new track-IDs
  against the existing NPZ and stitches new rows in place. Drives
  the new `chorus backgrounds add-tracks --oracle X --npz <path>`
  CLI subcommand.
- **`chorus backgrounds` CLI subcommand group (PR #52)** — `status`,
  `build`, and `add-tracks` for managing per-track CDF backgrounds
  without leaving the shell.
- **`chorus setup --all-chrombpnet` opt-in flag** — pre-cache every
  one of the 786 ChromBPNet/BPNet models during setup (~76 GB on
  disk, 3–4 h). Each of the 42 ChromBPNet ATAC/DNase models is
  ~720 MB tarball + ~1.1 GB extracted = ~1.8 GB; the 744 BPNet/CHIP
  models are tiny (~410 MB combined). Default behaviour stays on
  the v0.1 fast path (K562 + HepG2 DNase only, ~3.5 GB).
- **`chorus --version` flag** — was missing in 0.1.
- **`EnvironmentNotReadyError`** — predict / load now raise a clear
  actionable error pointing to `chorus setup` / `chorus health` when
  `use_environment=True` was requested but the env wasn't built.
  Replaces the earlier silent `use_environment=False` swallow.
- **`docs/NORMALIZATION_GUIDE.md`** — full walkthrough of the per-track
  CDF design, layer configs, and three end-to-end "bring your own
  model" recipes (ChromBPNet, LegNet, new oracle from scratch).
- **GitHub Actions CI** (`.github/workflows/tests.yml`) — runs the
  fast pytest suite on every push and PR.
- **End-to-end integration tests** (`tests/test_integration.py`) —
  marker-gated SEI / LegNet CDF download, ChromBPNet fresh download,
  and `chorus-mcp` stdio MCP E2E.
- **Error-recovery unit tests** (`tests/test_error_recovery.py`) — 12
  mock-based tests covering download/auth/env-missing failure paths.
- **HTML walkthrough render audit** —
  `audits/2026-04-26_v29_scorched_earth/probes/05_html_render.py`
  renders all 18 shipped walkthroughs at 1600×4500 in headless
  Chromium and audits each against the §7 audit checklist (IGV
  block, glossary, percentile columns, formula badges, JS errors).

### Changed

- **README quickstart** rewritten to four numbered steps that read
  in one lunch break (PR #42 + later refinements). Disk requirement
  reduced from "~80 GB" (which itself was an under-estimate) to
  **~25 GB default / ~100 GB with `--all-chrombpnet`** after the
  prefetch revert in PR #55. Note: the `--all-chrombpnet` figure
  was originally documented as ~60 GB; an audit on 2026-04-28
  re-measured each ENCODE ChromBPNet model on disk
  (~720 MB tarball + ~1.1 GB extracted = ~1.8 GB per model × 42
  models ≈ 76 GB just for ChromBPNet weights) and corrected the
  claim to ~100 GB total.
- **`chorus setup --oracle <X>` exit codes** — `chorus setup`,
  `chorus health`, `chorus genome download`, `chorus remove` all
  now return non-zero on bad input and surface the valid-name list.
- **All "Failed to load X" exceptions** point at
  `chorus health --oracle X` for diagnosis and end with a period.
  HuggingFace-rejected-token errors point at
  `https://huggingface.co/settings/tokens`.
- **`--verbose` on `chorus health` / `list` / `genome`** now sets
  the root logger to DEBUG (was previously a no-op aside from a few
  extra print lines).
- **Subprocess timeouts** raise `RuntimeError` with a pointer to
  `CHORUS_NO_TIMEOUT=1` instead of bare `subprocess.TimeoutExpired`.
- **`InvalidSequenceError`, `InvalidAssayError`, `InvalidRegionError`**
  now multiply inherit from `ChorusError, ValueError` so legacy
  `except ValueError` handlers still catch them.
- **CLI noise demoted to DEBUG** — "Found mamba via MAMBA_EXE…" /
  "Detected platform: Darwin arm64…" are no longer printed by every
  command.
- **TF/absl boot spam silenced** for Enformer + ChromBPNet via
  `TF_CPP_MIN_LOG_LEVEL=3` set automatically inside the env runner.
- **`chorus setup --oracle chrombpnet` default behaviour** reverted
  to the v0.1 fast path (K562 + HepG2 DNase, ~9 min) after PR #55
  found that PR #51's "all 786 models by default" change had silently
  20×'d the default disk footprint and 20×'d setup time.

### Fixed

- **P0: track-ID validator rejected FANTOM CAGE identifiers**
  (`CNhs11250` etc.) — Enformer/Borzoi `_validate_assay_ids` only
  treated `ENCFF*` as identifier candidates, so the shipped
  `single_oracle_quickstart.ipynb` broke on the first multi-track
  cell for every new user. Fixed in PR #48.
- **P0: `chorus-sei.yml` solver explosion** — old `cudatoolkit=11.7`
  + `pytorch<2.0.0` pins triggered a 50-minute libsolv hang on
  fresh installs. Removed in PR #46.
- **P0: stale `pip install -e .` in README** — picked up a Python
  2.7 `pip` from `~/.local/bin` on HPC PATHs. Replaced with
  `python -m pip install -e .` in PR #46.
- **P0: `_setup_environment` silently swallowed errors** — flipped
  `use_environment=False` and continued. Now raises
  `EnvironmentNotReadyError` on next predict.
- **ChromBPNet HepG2 prefetch (P1)** — `chorus setup --oracle
  chrombpnet` now pre-caches both K562 and HepG2 DNase models so
  `advanced_multi_oracle_analysis` and `comprehensive_oracle_showcase`
  notebooks don't block mid-run on a 720 MB ENCODE tarball.
- **MCP `--help` listed 20 of 22 tools** — missed `discover_variant`
  and `fine_map_causal_variant`. Reorganised into 4 logical groups
  with explicit `(22)` count.
- **Dead `#mcp-server-ai-assistant-integration` anchor** in
  `MCP_WALKTHROUGH.md` — fixed to `#mcp-server`.
- **Numerous error-message inconsistencies** — periods, fix hints,
  `raise ChorusError` vs `logger.error + return False`, etc.

### Documentation

- **`docs/NORMALIZATION_GUIDE.md`** added (~700 lines).
- **`docs/MCP_WALKTHROUGH.md`** — both install paths (`.mcp.json`
  per-project + `claude mcp add` global) documented; added a
  "verify the connection" first-prompt sanity check.
- **README**: signed/unsigned tracks introduced before percentile
  range; AlphaGenome 5,731-vs-5,168 disambiguated inline; Sei
  21,907-vs-40 disambiguated inline; Apple Metal support claim
  reconciled with the actual macOS GPU table.

### Audits

Six dated reports documenting the v22 → v29 cycle live under
[`audits/`](audits/). The latest cross-platform validation is
[`audits/2026-04-27_v29_linux_cuda/report.md`](audits/2026-04-27_v29_linux_cuda/report.md)
(Linux/CUDA replay) which mirrors
[`audits/2026-04-26_v29_scorched_earth/report.md`](audits/2026-04-26_v29_scorched_earth/report.md)
(macOS arm64). Both returned 0 chorus findings.

### Migration notes from 0.1.x

- `chorus setup --oracle chrombpnet` will only pre-cache K562 + HepG2
  DNase by default (~1.4 GB). Any other ChromBPNet cell type still
  downloads on first `load_pretrained_model(...)`. To restore the
  "everything up front" behaviour, pass `--all-chrombpnet`.
- The default ChromBPNet CDF NPZ now has 786 rows (was 24). Any code
  that hard-coded `track_ids` indices into the old NPZ should switch
  to the dict-style `track_index[<id>]` lookup that
  `PerTrackNormalizer` exposes.
- `chorus-sei` env yml dropped `cudatoolkit=11.7` and bumped
  `pytorch>=2.0.0`. If you have a manually pinned env, rebuild with
  `chorus setup --oracle sei --force`.

## [0.1.0] — 2025-09-XX

Initial release: unified Python API + MCP server over six genomic
deep-learning oracles (Enformer, Borzoi, ChromBPNet, Sei, LegNet,
AlphaGenome). Per-oracle conda envs, per-track CDF normalization,
HTML report generation with embedded IGV, and the `chorus` CLI
(`setup`, `list`, `health`, `validate`, `remove`, `genome`).
