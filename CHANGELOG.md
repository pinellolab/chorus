# Changelog

All notable changes to Chorus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Three backgrounds rebuilt against a gene-anchored effect region set (AlphaGenome, Borzoi, Enformer).** The shipped effect nulls were drawn from uniformly random genomic positions, which is the wrong reference class for a TSS-localised assay: a random position has essentially no CAGE signal, so the `+1` pseudocount damps its log-ratio toward zero and the null's body sits far below where real regulatory effects live. Positions are now sampled per stratum from protein-coding annotation (GENCODE v48 basic) — 20 % within ±1 kb of a TSS, 20 % at 1–10 kb, 33 % within ±100 bp of an exon/intron boundary, 12 % elsewhere in a gene body, 15 % uniformly random. The random tail is deliberate: without near-zero mass, genuinely small effects would receive artificially *low* percentiles, the mirror of the failure being fixed. All three oracles drew from one seeded region set — each build logged an identical `tss_near 1200, tss_far 1200, junction 1980, gene_body 720, random 849` of 6,000 sampled positions — so the three are directly comparable. New `build_config` provenance is stamped into each NPZ.

  Measured against strong TSS-proximal liver eQTLs from GTEx v8 (`|slope| >= 0.5`, `maf >= 0.05`, `p <= 1e-10`), scored in tissue-matched tracks:

  | layer | eQTL percentile p50, before | after | saturated |
  |---|---|---|---|
  | RNA (232 rows, 8 tracks) | 0.899 | **0.781** | 0 % |
  | CAGE (100 rows, 4 tracks) | 0.857 | **0.659** | 0 % |

  Both moved down, which is the intended direction — the reference class now contains variants that actually perturb these assays, so a given eQTL is less extreme against it.

- **Effect of the whole cycle on the committed examples.** Across the four AlphaGenome/Enformer variant walkthroughs, saturated rows (percentile pinned at exactly 1.0000, where the column has stopped discriminating) fell from **47 to 16** with row counts unchanged at 369 and distinct percentile values up from 280 to 284.

  Attribution, and the honest limits of it. These were measured separately and are attributable:

  | change | measured effect |
  |---|---|
  | RNA denominator: exon *intervals* → bins actually summed (#149) | numerator was overstated 251–1736×; median \|effect percentile\| 0.99+ → 0.062 on the unchanged population |
  | Enformer `effect_cdfs` grid repair (#143) | reachable percentile ceiling 0.9605 → 0.9998; the top 4 % of the scale did not previously exist |
  | Cross-process determinism (#127, #145) | a full report is now bit-exact: 603 numeric fields, 0 differing, 0 sign flips, worst relative delta 0.0 — against 454 differing fields with 36 sign flips before |
  | Gene-anchored null (this entry) | the eQTL table above |

  The per-layer walkthrough diff is a **combined** effect of all of the above plus the CHIP window classifier (#122/#146) and window-span parity (#147/#148); it is not decomposed per change, because doing so honestly would require re-running each rebuild in isolation. It is reported as a fused diff rather than split by guesswork.

- **One layer did not improve, and is not described as fixed.** Enformer `chromatin_accessibility` at SORT1 went from 4/12 saturated to **6/12**, median percentile 0.960 → 1.000. This is not a new regression: 0.960 *was* the padded-grid artefact ceiling (#143), so those rows were already pinned and the repair only made the pinning visible instead of disguising it as a plausible 0.96. The underlying fact is that Enformer's accessibility effect null is genuinely too narrow for a variant this strong. It clears the release gate (7 distinct values across 12 rows, so the column is not constant) and is the next thing to look at.

### Fixed

- **Walkthrough TSVs silently dropped every per-gene row after the first.** `scripts/regenerate_remaining_examples.py` carried its own report flattener that de-duplicated on `(allele, assay_id, layer)` — a key omitting the region. RNA and CAGE emit one row per *gene* per track, so all but one gene were discarded: `validation/TERT_chr5_1295046` shipped 18 rows where its JSON had 99 (one `tss_activity` row where there were fifteen, one per nearby gene TSS), `discovery/SORT1_cell_type_screen` 39 of 347, `sequence_engineering/region_swap` 4 of 32, `integration_simulation` 3 of 55. The same writer also put `region_label` in a column named `description`, which already means the *track* description in `to_dict()` — one name for two things across two artefacts of the same report. Fixed by deletion rather than repair: everything now routes through `report.to_dataframe()`, the canonical writer `scripts/regenerate_examples.py` already used. All 14 walkthrough (JSON, TSV) pairs now agree on both counts and row identities, pinned by `tests/test_json_tsv_parity.py`. Long-standing rather than a regression — the counts were identical before and after the rebuild.

- **Docs overstated AlphaGenome's usable track count by 563.** Ten places across `README.md`, `docs/variant_analysis_framework.md`, `docs/MCP_WALKTHROUGH.md` and `docs/API_DOCUMENTATION.md` advertised **5,731 tracks**, including the README's headline sentence. That is the row count of AlphaGenome's metadata table; 563 of those rows are `padding` placeholders whose only purpose is keeping `local_index` aligned with the model's output array. They carry no assay, `iter_tracks()` skips them, and the shipped background has no row for any of them. The queryable count is **5,168** — verified both directions: 5,168 metadata tracks have a background row and 0 do not, and 5,168 + 563 = 5,731 exactly. `tests/test_documented_track_counts.py` now compares live-doc prose against the shipped NPZs so this cannot drift again. (An earlier entry below claims this was "disambiguated inline"; it was not, in any of the four live docs.)

- **The background grid guard blocked a healthy rebuild.** The `distinct == count` fingerprint — reported as perfect, 5,313/5,313 with zero false positives — fired on AlphaGenome `effect_cdfs` row 3966 (CHIP_TF ARID3A) and refused an 11-hour merge. That row is not padded: 913 of its 5,949 samples are exact zeros, so interpolating the remainder lands on exactly 5,949 distinct values by coincidence, and its maximum first appears at index 9998 — precisely where `np.interp` puts it, where padding would put it at 5,948. The raising condition is now the mechanical one alone (`first_max == n - 1`, unreachable by `np.interp`, whose `source_q` stops at `(n-1)/n`); `distinct == count` is a `logger.warning` that says outright it is usually coincidence. Tally before the demotion was three false positives to one true catch.

- **ChromBPNet recovers counts with `expm1`, not `exp`.** ChromBPNet's count head is trained against `log(1 + count)` (upstream `batchgen_generator.py` feeds `np.log(1+batch_cts.sum(-1, keepdims=True))` as the target), but chorus inverted it with `np.exp`, so every recovered count was high by exactly +1 — negligible at a peak (~0.1 % at 1,000 counts) but up to 100 % at a low-activity site, which is precisely the regime the activity CDFs are built from. Corrected at the three count-inversion sites: `oracles/chrombpnet.py:579` (`_transform_predictions_to_tracks`), `oracles/chrombpnet.py:802` (`predict_sliding`), and `scripts/build_backgrounds_chrombpnet.py:348` (`predict_profiles_batch`). The profile softmax `np.exp` calls at `:577`, `:801` and `:347` are a different transform and are unchanged. The bug was self-consistent — oracle and CDF builder made the same error — so ChromBPNet percentiles were internally valid, which is why it went unnoticed; raw counts and cross-oracle comparability were not. Cherimoya already did this correctly (`cherimoya_source/scoring.py`) and is unaffected. New regression suite `tests/test_chrombpnet_counts.py` covers all three sites, including an oracle/builder consistency check so the two cannot drift apart again.

## [0.4.0] — 2026-04-30

### Added

- **Rebuilt ChromBPNet per-track CDFs against `chrombpnet_nobias`** to match the 0.3+ default model variant. Prior 0.2.x CDFs were built against the bias-aware `chrombpnet`; effect-percentile lookups now point at the matching empirical distribution. New NPZ on [`lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds): 786 tracks (22 ATAC + 20 DNASE + 744 CHIP), all CDFs monotone, every reservoir filled (effect_count=9609 per track), sha256 `be61e9e8...`. ATAC/DNase percentiles shift 13.5–29.3% at p95 (the bias correction strips enzymatic motif preferences); CHIP/BPNet percentiles ~unchanged (already nobias-equivalent in the old NPZ). Built on A100 in ~10 h. Audit at `audits/2026-04-29_chrombpnet_cdf_rebuild/report.md`.

- **HuggingFace mirror consolidation for Enformer, Borzoi, Sei, and LegNet weights.** Chorus now ships a chorus-controlled HF mirror for each non-AlphaGenome oracle, so the install path doesn't depend on third-party hosts that have shown lifecycle volatility (TFHub deprecation in particular). Each loader prefers the chorus mirror and falls back to the original source on any failure — no behavior change in the happy path, redundancy on the unhappy.

  | Oracle | New chorus mirror | Original source |
  |---|---|---|
  | Enformer | [`lucapinello/chorus-enformer`](https://huggingface.co/lucapinello/chorus-enformer) | TFHub `deepmind/enformer/1` (redirects to Kaggle) |
  | Borzoi | [`lucapinello/chorus-borzoi`](https://huggingface.co/lucapinello/chorus-borzoi) (4 folds) | `johahi/borzoi-replicate-{0..3}` |
  | Sei | [`lucapinello/chorus-sei`](https://huggingface.co/lucapinello/chorus-sei) | Zenodo `4906997` |
  | LegNet | [`lucapinello/chorus-legnet`](https://huggingface.co/lucapinello/chorus-legnet) | Zenodo `17863550` |

  Each mirror's README explicitly identifies (a) where the weights came from, (b) who owns the weights, (c) which model terms apply to the weights regardless of mirror, (d) which code license applies to the chorus loader. License terms applying to the *weights* are unchanged by mirroring (Sei stays CC-BY-NC, etc.). `huggingface_hub>=0.20` added to the four corresponding env yamls so users get the HF path by default; older installs predating this fall back to the original source via `ImportError`-handling. Sei tarball verified byte-and-md5 against Zenodo's published `4297aafb711aec4ecccb645b8928ea26`. See `audits/2026-04-29_hf_mirror_consolidation/report.md` for the full inventory and verification trail.

- **AlphaGenome PyTorch backend (second AlphaGenome oracle, installed by default)** —
  `AlphaGenomePTOracle` wraps the upstream
  [`genomicsxai/alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch)
  port. **Same model, same weights** as the default JAX `alphagenome`
  oracle: [`gtca/alphagenome_pytorch`](https://huggingface.co/gtca/alphagenome_pytorch)
  is the official JAX checkpoint converted to safetensors. Outputs
  agree within fp32 implementation noise (1–2 % per-track on chorus-API
  scoring, verified on M3 Ultra + A100). Both backends now install by
  default in `chorus setup` (~1.7 GB env + ~880 MB weights for the PT
  side, ~10–13 min extra wall-clock). The PT mirror's HF repo is
  public, but Google's non-commercial AlphaGenome model terms still
  apply to the weights regardless of which mirror they came from —
  read https://deepmind.google.com/science/alphagenome/model-terms.
  New conda env `chorus-alphagenome_pt`. Use via
  `chorus.create_oracle('alphagenome_pt', use_environment=True)`.
  - 5–8× faster than the JAX default on Apple Silicon for windows
    ≤ 600 kb via MPS; **slower than JAX CPU past a sharp cliff at
    768→896 kb** (GPU on-die cache spillover, not RAM swap — verified
    against memory traces on a 96 GB M3 Ultra). See
    `audits/2026-04-29_alphagenome_pytorch_spike/` for the full speed
    table, root-cause investigation, and decision discussion.
  - JAX backend (`alphagenome`) remains the default and is unchanged.
    Track metadata, assay identifiers, and CDF backgrounds are shared
    between backends.
  - Variant scoring, fine-tuning hooks, and CONTACT_MAPS /
    SPLICE_JUNCTIONS exposure available upstream are **not yet wired
    through chorus** — accessible via direct `alphagenome_pytorch`
    import for users who need them.

- **AlphaGenome backend-routing helper** —
  `chorus.recommend_alphagenome_backend(window_size_bp)` (also exposed
  as `oracle.recommend_backend(window_size_bp)` on both AlphaGenome
  oracles) returns a dict with the suggested oracle (`alphagenome` vs
  `alphagenome_pt`), suggested device, a one-line reason, confidence,
  and supporting benchmark numbers. Logic (verified on M3 Ultra +
  A100):
  - Linux + CUDA → `alphagenome` on CUDA — counter-intuitively, JAX
    with CUDA is 1.2–2.8× *faster* than the PyTorch port at every
    window length. PT remains useful for portability (smaller install,
    looser CUDA-version pinning) but not for raw speed.
  - macOS + MPS, window ≤ 600 kb → `alphagenome_pt` on MPS (5–8× over
    JAX CPU; safe-zone is conservative under the empirical
    768→896 kb GPU on-die cache cliff)
  - macOS + MPS, window > 600 kb → `alphagenome` on CPU
  - No GPU → `alphagenome` on CPU
  Suggestion-only, no auto-routing — users always know which backend
  produced their predictions.

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

## [0.3.0] — 2026-04-28

### ⚠️ Breaking change

**ChromBPNet default `model_type` flipped from `'chrombpnet'` (bias-aware)
to `'chrombpnet_nobias'` (bias-corrected).** Predictions from
`load_pretrained_model(assay='DNASE', cell_type='K562')` (i.e. the
default call shape used in every shipped notebook and the README
quickstart) **will shift in magnitude and shape** because the
bias-corrected model removes Tn5 / DNase-cleavage motif bias. The new
default is what the Kundaje paper recommends for variant analysis,
motif discovery, and region-swap predictions — which is what chorus
is used for.

To preserve old behaviour exactly, pass `model_type='chrombpnet'`:
```python
oracle.load_pretrained_model(
    assay="DNASE", cell_type="K562", model_type="chrombpnet",
)
```
The bias-aware variant is no longer in the default cache; chorus falls
back to the full ENCODE tarball flow for it (~1.8 GB on disk per
model). All other API shapes are unchanged.

### Added

- **HuggingFace slim mirror** at
  [`lucapinello/chorus-chrombpnet-slim`](https://huggingface.co/lucapinello/chorus-chrombpnet-slim)
  containing only the artifacts chorus actually loads at inference
  time: 42 fold-0 ChromBPNet `chrombpnet_nobias` h5's
  (1,074 MB) + 744 BPNet/CHIP h5's (419 MB) = **1.49 GB total**.
  Replaces the previous ~100 GB ENCODE-tarball-based prefetch path.
  See `audits/2026-04-28_chrombpnet_slim_mirror/` for the design,
  build pipeline, manifest, and round-trip verification.
- New `_try_slim_hf_chrombpnet()` and `_try_slim_hf_bpnet()` helpers
  on `ChromBPNetOracle`. `load_pretrained_model()` now tries the HF
  slim mirror first for the common case (fold=0 +
  `chrombpnet_nobias`, or any CHIP/BPNet model) and falls back to
  ENCODE / JASPAR transparently when:
  - The mirror is missing the requested ENCFF / BP_BASE_ID.
  - The user requested fold ≠ 0 or `model_type='chrombpnet'`.
  - `huggingface_hub` is unavailable or the network is down.

### Changed

- **Default `chorus setup --oracle chrombpnet` footprint:** ~3.5 GB
  → ~50 MB. Two slim h5's (K562 + HepG2 DNase) replace two ENCODE
  tarballs as the fast-path default.
- **`chorus setup --all-chrombpnet` footprint:** ~100 GB → **~1.5 GB**
  (67× reduction). All 786 fold-0 nobias h5's stream from HF in ~5 min
  vs ~3-4 h of ENCODE tarball downloads + extraction.
- README, CLI `--help` text, and internal docstrings updated to the
  new disk-size figures.

### Migration notes

- **Numerical comparisons against pre-0.3 ChromBPNet outputs will
  change.** The bias-corrected model removes systematic enzymatic
  motif preferences. Re-baseline any analysis that pinned exact
  values; rankings + relative effects are largely preserved but exact
  magnitudes shift.
- If you have manual model paths set on the oracle (`oracle.model_path
  = ...`), nothing changes.
- If you load with `model_type='chrombpnet'` explicitly, nothing
  changes — that path still goes through the ENCODE tarball flow.
- If you load with no `model_type` kwarg, you now get the
  bias-corrected variant. Pass `model_type='chrombpnet'` to opt back in.

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
