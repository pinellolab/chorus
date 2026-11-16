# Pre-release full audit of `main` at `4258b6f` (v0.7.5 + #246, #249–#252) — 2026-11-15

This audit ran [`AUDIT_CHECKLIST.md`](AUDIT_CHECKLIST.md) §1–§18, except §3 and §12, against `4258b6f` on the release host before the next stable release. Five agents ran their sections in parallel, and this report compiles their results. The run was read-only: nothing was fixed, and every finding below is deferred to a fix PR.

**Verdict: not ready to tag.**

- **P0:** none.
- **P1:** 18. Three were introduced since v0.7.5, which makes them release blockers, and all three come from #246's conservation work. One pre-existing P1 (P1-17) was fixed on `main` by #253 during the run.
- **P2:** 33. Six were introduced since v0.7.5 and two more partly.
- **Gates:** two release gates are unmet. CI is red on `4258b6f`, and neither test suite is green as written on this host. See [Release blockers](#release-blockers).

What holds up:

- every row of all 9 shipped nulls;
- the report structure (columns, formula badges, IGV payloads);
- the MCP tool surface and its specs;
- offline prediction;
- bit-exact cross-process determinism for ChromBPNet, LegNet and AlphaGenome.

What does not:

- numbers and wording in shipped walkthroughs and notebooks, including two walkthroughs regenerated after v0.7.5;
- two scoring edge cases: variants near a chromosome's 3' end, and LegNet insertions;
- an LDlink token that reaches server logs;
- a test harness that cannot find this host's genome.

## What was run

- **Code under test:** `main` at `4258b6f`, the #252 merge. That is v0.7.5 (tag at `cc8aabe`) plus 18 commits, from:
  - #246: conservation tracks, AnnotationStore and 3 new MCP tools, with #247's review fixes folded in.
  - #249: alphagenome v1 import compat.
  - #250: alphagenome env pins.
  - #251: MCP agent-facing text, the padding filter, `load_oracle` routing and LegNet `cell_type`.
  - #252: the browser outage guard.

  The tree was clean throughout.
- **Moving target:** at 21:45 UTC, during the run, `origin/main` advanced to `eab109c` (#253, AlphaGenome-PT env pins). It changes `CHANGELOG.md`, `environments/chorus-alphagenome_pt.yml` and `tests/test_alphagenome_deps_are_pinned.py`. It was not audited, except that for this report its extended pin test was run on an extracted copy (5 passed; see P1-17).
- **Host:** the release host, a Linux x86_64 GCP VM: kernel 6.17.0-1020-gcp, 208 CPUs, 1,842 GiB RAM.
  - GPUs: the audit used only GPUs 0–3. GPUs 4–7 belonged to another user's job and were not touched.
  - Envs: miniforge3, with base `chorus` (Python 3.10.20) and the 9 per-oracle envs. All 10 have chorus 0.7.5 as an editable install of the checkout.
  - Data dir: `/data/chorus_data`, selected via `chorus_data_dir.txt`. It holds hg38 with its `.fai`, GENCODE v48 basic and all 9 `*_pertrack.npz`.
- **When:** 2026-11-15, 21:35–22:51 UTC (from agent log timestamps). Compiled at about 23:00 UTC.
- **Who ran what:**

| agent key | checklist sections | GPU |
|---|---|---|
| `install-api-errors` | 1, 2, 5, 9, 14, 15 | none (CPU only) |
| `cdf-consistency-hygiene` | 4, 10, 16, 17, 18 | none (CPU only) |
| `integration-suite` | 11 | 0–1 |
| `notebooks-determinism` | 6, 13 | 2 |
| `mcp-reports` | 7, 8 | 3 |

§3 and §12 had no owner and were not run. §19 is the maintainer's and is in progress.

- **Constraints:**
  - No repo writes: no edits, checkouts, commits or stashes.
  - No installs or env creation, so no `chorus setup`, no Playwright Chromium and no `pip-audit`.
  - No outward actions: no push and no GitHub or Hugging Face writes.
  - No `mcp__chorus__*` tools; each agent started its own `chorus-mcp`.
  - Secrets were reported by presence only.
  - Every Bash call stayed under 10 minutes. Long runs were split into chunks of 540 s or less, and anything longer was recorded as not run.
  - Checks that write into the tree ran from scratch copies or a scratch clone of `4258b6f`: notebook execution, notebook codegen, `rerender_examples.py --check` and the determinism gate.

### Side effects on the shared install

- **Data-dir marker (P1-15).** `tests/test_data_dir_resolution.py` overwrote and then restored `chorus_data_dir.txt`, the gitignored live data-dir pointer.
  - Rewrites were seen at 21:47, 21:53, 21:54 and 22:30 UTC.
  - Two are attributed: 21:53:07 to `install-api-errors`, and 22:30:11 to `integration-suite` (window 22:30:07–22:30:11). No agent's report attributes the other two.
  - Each time, the content was restored to `/data/chorus_data`. Any chorus process that started inside one of those windows resolved its data dir to a pytest tmp path, so results from those minutes deserve a sanity check.
- **Tracked track list (P2-2).** Chorus's own AlphaGenome metadata loader rewrote the tracked `chorus/oracles/alphagenome_source/alphagenome_tracks.json` at least seven times, between 21:46:08 and 22:48:15 UTC. Every write was byte-identical, and `git status` stayed clean.
- **Conservation downloads.** About 71 GB of conservation bigwigs were downloaded into `/data/chorus_data/downloads`, by the quickstart's cell 44 and by `chorus conservation download`. They are now cached.
- **Lock files (P2-29).** The fast suite touched two stray lock files under `/data/chorus_data/backgrounds/.cache/huggingface/download/`.
- **Git index.** One `git status` refreshed `.git/index` (stat cache only). Every later git call used `GIT_OPTIONAL_LOCKS=0`.
- **Final state.** Re-checked for this report: `git status --porcelain` is empty and the marker reads `/data/chorus_data`.

## Per-section status

| § | area | owner | status | key results |
|---|---|---|---|---|
| 1 | Installation & environment | install-api-errors | **Partial:** 6 pass, 4 not run, P2 only | `chorus --help` and all 26 subcommand `--help` pages return rc=0. All 9 env names exist. hg38 and its `.fai` (455 contigs) are present, GENCODE v48 is cached, and the 9 NPZs match the pinned revision in size. `legnet_pertrack.npz` auto-downloaded with no token (200,956 B, 2.6 s). Not run because they need installs: env solve, `pip install -e .`, `chorus setup` and its idempotency. P2-10, P2-11. |
| 2 | HF authentication | install-api-errors | **Fail** (P1-18) | A cached login loads AlphaGenome (66.1 s, CPU), but the documented `huggingface-cli login` exits 1 under huggingface_hub 1.26.0. The no-token error names `HF_TOKEN`, the gated URL and the now-dead alternative. The gated URL is identical in all four places. `list_tracks('alphagenome')` with no token returns 331 K562 hits with 0 network attempts. `HF_TOKEN` path not run. Export hint missing (P2-8). |
| 3 | GPU / device detection | none | **Not run** | The per-env device probe was not run. Indirect evidence only: AlphaGenome loaded on `cuda:0` under `CUDA_VISIBLE_DEVICES=3`, the smoke tests ran on GPUs 0–1 and the notebooks on GPU 2. Related: P1-16, P2-27. |
| 4 | Per-track CDFs | cdf-consistency-hygiene | **Pass on every P0 item;** 1 P2 | All 9 NPZs load (schema 4). Every row was checked, not just the first 10: 0 of 43,864 effect rows are non-monotone, p50 ≤ p95 ≤ p99 holds on every row, and there are 0 NaN/Inf. Signed fractions: borzoi 20.27%, alphagenome 12.91%, sei and legnet 100%, the rest 0%. Perbin layers are present exactly where expected, and track counts are exact. Pinned-revision test: 2 passed. expm1 test: 23 passed. P2-14. |
| 5 | Python API | install-api-errors | **No P0/P1 findings;** 2 P2; the direct-path P0 item only partly assessed | All 9 names construct, and an invalid name lists all 9. Sequence lengths match the README. Calling before load raises `ModelNotLoadedError` (9/9), and a bad chromosome raises `InvalidRegionError`. Correct dbSNP alleles give 0 warnings, and a wrong ref raises `ReferenceAlleleMismatchError`. `extract_sequence` returns G and T. The direct path ran live for LegNet, ChromBPNet, Enformer and the AlphaGenome load. No direct-path predict test exists for chrombpnet, sei or alphagenome_pt. P2-6, P2-9. |
| 6 | Notebooks | notebooks-determinism | **Fail** (P1-2 new; P1-9 to P1-12) | `single_oracle_quickstart` cold: not completed (exit 124 at 541 s inside the ~71 GB download). Warm: 232 s, 36/36 cells, 0 errors, 0 WARNING lines. `advanced`: 397 s, 57/57, 0/0. `comprehensive` needs about 585 s, more than the cap; 33/38 code cells passed, the cells 51–53 slice is identical to the committed outputs, and cells 56–57 are unverified. `cherimoya_quickstart` fails as committed. With the genome symlinked it passes, but every number differs. The kernelspec is registered and the links resolve. P2-12, P2-22 to P2-24. |
| 7 | HTML reports | mcp-reports | **Fail** (P1-1 new; P1-7, P1-8); browser checks not run | All 21 reports are well-formed; the largest is 25.70 MiB. All 20 IGV reports inline a bundle with 0 CDN tags, and 481/481 wig tracks carry non-zero data. "How to read this report" is present in 21/21. All 52 per-layer tables (827 rows) have the right columns. There are 0 formula-badge mismatches, badges agree with the effect's sign in 858/858 rows, and the consensus n=1 labels are correct. P2-15 to P2-21. |
| 8 | MCP server | mcp-reports | **No P0/P1 findings;** 2 P2 (1 new) | `chorus-mcp` came up in 3.3 s and 2.8 s. Its 27 tools match the README list, and `list_oracles` returns 9 with every field equal to ORACLE_SPECS and to the Python API. `_INSTRUCTIONS` is byte-identical to the source, and the MCP tests give 115 passed. End-to-end, rs12740374 DNASE:HepG2 gave +1.33446 / 0.9964 / 0.9725 (load 73.4 s, analysis 204.7 s), with the same keys as the committed example and a clean unload. §8b not run; §8c pass. P2-1, P2-3. |
| 9 | Error messages | install-api-errors | **Fail** (P1-18); P2-5, P2-7 | `fakeOracle`, calling before load, the missing-env fallback and the network drop (returns 0) all pass. On a missing reference, `predict()` names the kwarg and `chorus genome download hg38`, but three other entry points do not. |
| 10 | Repo-wide consistency | cdf-consistency-hygiene | **No P0/P1 findings;** 3 P2 | All 3 drift greps are clean: the only hits are false positives and the 3 expected `examples/applications/` mentions. `test_documented_track_counts`: 6 passed. The 10-file consistency batch: 173 passed / 6 skipped. The README matrix matches ORACLE_SPECS (9 rows), and 27 tools is stated everywhere. P2-25, P2-4, P2-26. |
| 11 | Test suite | integration-suite | **Fail:** gates not met | Fast suite: 2,293 passed / 61 skipped / 1 failed. Integration as written: 88 passed / 2 failed / 77 skipped, 1 not run. With the genome reachable and the MCP child confined: 112 / 0 / 55, 1 not run. CI is red on `4258b6f`. Every code path added since v0.7.5 has its own tests, and all are green. P1-13 to P1-16, P2-27 to P2-29. |
| 12 | Reproducibility | none | **Not run** | Regeneration writes into `examples/`. In a scratch clone, notebook codegen reproduced (2 passed) and `rerender_examples.py --check` passed (1). Related: P1-1, two committed walkthroughs that do not reproduce from the pinned null. |
| 13 | Determinism | notebooks-determinism | **Pass except Enformer** (P2-13); across-machine not run | Two predicts per oracle, each in a fresh process (`use_environment=True`), were bit-exact (max abs diff 0.0): ChromBPNet over 100,000 bins, LegNet over 2,000 and AlphaGenome over 1,048,576. AlphaGenome was also bit-exact inside `predict_session()`. Enformer: max abs diff 1.750e-3, with 896/896 bins differing (F8). |
| 14 | Genomics edge cases | install-api-errors | **Fail** (P1-4, P1-5) | Padding returns the full length. Soft-masked bases give 0 warnings. Multi-allelic G>A,C,T renders 3 sections with distinct effects. `predict_variant_effect` fails within half a window of a 3' end and across chrM, and LegNet insertions are mis-scored. |
| 15 | Offline | install-api-errors | **Pass;** request inventory not run | LegNet and ChromBPNet predict with no token while every non-loopback socket is refused. With `HF_HUB_OFFLINE=1` they make 0 network attempts; without it, ChromBPNet makes 2 before falling back to the cache. Enformer `analyze_gene_expression(pred, 'GATA1')` works offline: 112.11, 6 TSS. The report structural tests give 33 passed / 3 skipped. |
| 16 | Logging hygiene | cdf-consistency-hygiene | **Fail** (P1-6) | `test_no_committed_credentials`: 16 passed. There is no committed credential, so nothing to rotate. The LDlink token reaches the server logs through exception chaining. Per-machine paths in notebook outputs: P2-22. |
| 17 | Dependency supply chain | cdf-consistency-hygiene | **Fail** (P1-17, fixed on `eab109c`); P2-32; `pip-audit` not run | The root `environment.yml` is fully versioned. All 10 envs use the editable chorus 0.7.5. The vendored `igv.min.js` is byte-identical to upstream 3.8.4. Pin tests: 7 passed, plus 2 integration tests passed. |
| 18 | License / attribution | cdf-consistency-hygiene | **Fail** (P1-3 new); P2-30 (new), P2-31 | LICENSE is MIT, which matches `setup.py`, `CITATION.cff` and the README. `docs/THIRD_PARTY.md` covers the oracles and is linked from the README. GPN-Star, phyloP and phastCons are credited nowhere. |
| 19 | Cutting a release | maintainer | **In progress** | Being done in parallel and not audited here. Facts from this audit that bear on it: the artefact half of the release pair checks out (local NPZs match `backgrounds-2026-08-16-sei-full`, and the pinned-revision test passed 2), CI is red on `4258b6f`, and `main` moved to `eab109c` during the run. |

## Test results

| run | result |
|---|---|
| Fast suite, `pytest tests/ -q`, CPU, in three chunks (166 s, 335 s, 4 s) | **2,293 passed / 61 skipped / 1 failed / 0 errors** of 2,355 selected; 2,523 collected, of which 168 integration tests were deselected. The failure is P1-15. 22 of the skips come from P1-14's repo-relative genome paths. |
| Integration, `-m integration`, as written | **88 passed / 2 failed / 77 skipped / 0 errors / 1 not run** (168 tests in 28 files). Both failures are P1-14's `OSError`. The skips are 23 genome path, 49 no Chromium, 4 torch-gated and 1 missing AlphaGenome effect shards. The MCP E2E test was not run as written, because it is unsafe on a shared host (P1-16). |
| Integration with the genome reachable and the MCP child confined (scratch clone of `4258b6f`) | **112 passed / 0 failed / 55 skipped / 1 not run.** The skips are 49 no Chromium, 4 torch-gated, 1 cherimoya forge and 1 AlphaGenome shards. Not run: the ChromBPNet fresh ENCODE download, which hit the 540 s timeout at 717,225,984 bytes. |
| CI `tests.yml` run 43764177834 on `4258b6f` | **Red.** The fast-suite job passed. `browser smoke (2 reports)` failed with 1 failed / 6 passed / 5 skipped (P1-13). The #249 merge (7724aa5) was also red; #251 (fdf56e9) and #253 (eab109c) were green. |

Targeted runs:

| tests | result |
|---|---|
| Error-recovery and related files | 233 passed / 1 failed (the P1-15 test) / 1 skipped |
| 10-file repo-consistency batch | 173 passed / 6 skipped |
| MCP tests | 115 passed |
| `test_no_committed_credentials` | 16 passed |
| Pin tests | 7 passed, plus 2 integration passed |
| `test_chrombpnet_counts` | 23 passed |
| `test_documented_track_counts` | 6 passed |
| Walkthrough README guards | 20 passed / 5 skipped |
| Report-bundling structural tests | 33 passed / 3 skipped |
| `eab109c`'s `test_alphagenome_deps_are_pinned.py` (run for this report) | 5 passed |

## What was fixed, and what was deferred

Nothing was fixed. The audit ran read-only by design, and the report itself was not committed. Every finding below is deferred to a fix PR, except P1-17, which #253 fixed on `main` during the run.

## Findings

### P0

None.

### P1 (18)

| # | new since v0.7.5 | finding | file |
|---|---|---|---|
| P1-1 | **yes** | Two ChromBPNet walkthroughs' percentiles do not reproduce from the pinned null, and the conservation README's headline claim is false | `examples/walkthroughs/conservation/SORT1_rs12740374/README.md` |
| P1-2 | **yes** | The quickstart notebook downloads ~71 GB of conservation bigwigs on a cold run | `examples/notebooks/single_oracle_quickstart.ipynb` |
| P1-3 | **yes** | GPN-Star, phyloP and phastCons are credited nowhere reachable from the README | `docs/THIRD_PARTY.md` |
| P1-4 | no | `predict_variant_effect` fails within half a window of a chromosome's 3' end, and across chrM for the large-window oracles | `chorus/core/base.py` |
| P1-5 | no | LegNet insertions are silently mis-scored | `chorus/oracles/legnet.py` |
| P1-6 | no | The LDlink token reaches MCP server logs and debug tracebacks | `chorus/utils/ld.py` |
| P1-7 | no | README result tables in four walkthroughs are stale | `examples/walkthroughs/validation/README.md` and 3 more |
| P1-8 | no | Causal reports call the effect null "a background of common variants" | `chorus/analysis/causal.py` |
| P1-9 | no | `cherimoya_quickstart` hardcodes `../../genomes/hg38.fa` | `examples/notebooks/cherimoya_quickstart.ipynb` |
| P1-10 | no | `cherimoya_quickstart` describes the 5-fold ensemble, but the default is fold 0 | same |
| P1-11 | no | `comprehensive_oracle_showcase` prints 5731 AlphaGenome tracks against the README's 5,168 | `examples/notebooks/comprehensive_oracle_showcase.ipynb` |
| P1-12 | no | The same notebook's "H3K27ac +80" insight matches no computed number | same |
| P1-13 | no | CI is red on `4258b6f` because the browser meta-test skips the outage guard | `tests/test_committed_reports_render_in_a_browser.py` |
| P1-14 | no | Release-gate tests hardcode a repo-relative `genomes/hg38.fa` | `tests/test_smoke_predict.py` and 10 more files |
| P1-15 | no | `test_data_dir_resolution` fails on configured installs and rewrites the live marker | `tests/test_data_dir_resolution.py` |
| P1-16 | no | The MCP E2E test's env whitelist drops `CUDA_VISIBLE_DEVICES` and `CHORUS_MCP_OUTPUT_DIR` | `tests/test_integration.py` |
| P1-17 | no (**fixed on `eab109c`**) | `chorus-alphagenome_pt` installs alphagenome-pytorch and alphagenome unpinned | `environments/chorus-alphagenome_pt.yml` |
| P1-18 | no | The `huggingface-cli login` advice is dead under huggingface_hub 1.26 | `chorus/oracles/alphagenome.py` |

#### P1-1 · new since v0.7.5 · Two ChromBPNet walkthroughs carry percentiles that no published null reproduces, and the conservation walkthrough's headline claim is false
*mcp-reports*

- **File:** `examples/walkthroughs/conservation/SORT1_rs12740374/README.md`, plus both walkthroughs' HTML and `example_output.{json,md,tsv}`. The walkthroughs are:
  - `variant_analysis/SORT1_chrombpnet`, regenerated in 3d0b6b1 ("Generated 2026-08-28 06:54 UTC");
  - `conservation/SORT1_rs12740374`, added in 01aae63 (13:03 UTC).
- **Problem:** Both report DNASE:HepG2 (+1.3758) with Effect %ile "≥99th (1.18× null max)" (quantile_score 1.0, effect_exceedance 1.1778, implying a null max of about 1.168) and Activity %ile 0.885.
  - With the pinned background, current code gives effect 0.9995, which is inside the null's support (null max 1.6057), and activity 0.906.
  - The conservation README's central claim is false under the pinned null. That claim is that the effect "exceeds every one of the ~18,000 background variants … about as strong as this scoring system can register" (lines 42–43), along with the "saturating regulatory effect" framing (line 59).
  - `SORT1_chrombpnet/README.md` ("effect percentile 0.9995 … stronger than 99.95 %") contradicts its own `example_output.md`.
  - `example_output.json` records no background revision, so no test can notice.
- **Evidence:**
  - With `PerTrackNormalizer` on `/data/chorus_data/backgrounds`:
    - `effect_percentile('chrombpnet','DNASE:HepG2',1.3758442)` = 0.9995
    - `effect_exceedance` = None
    - `activity_percentile(287.2576)` = 0.906
    - effect-null max = 1.605698
  - The local `chrombpnet_pertrack.npz` (sha256 `d4d6488c…`) is identical to the copy at the pinned revision `backgrounds-2026-08-16-sei-full`.
  - These all give the same 0.9995 / 0.906: the older 08-06-schema4 and 08-01-preunified ChromBPNet NPZs, v0.7.5's committed output, and the 08-13 multi-oracle ChromBPNet report.
  - Re-checked for this report: line 24 of `SORT1_chrombpnet/example_output.md` reads `| DNASE:HepG2 | 287 | 747 | +1.376 | 0.9995 | 0.906 | Very strong opening |` at v0.7.5, and `… | ≥99th (1.18× null max) | 0.885 | …` at HEAD. The conservation walkthrough's line 24 matches HEAD.
- **Fix:**
  - Regenerate both walkthroughs (HTML plus json/md/tsv) on a host whose `chrombpnet_pertrack.npz` matches the pin (sha256 `d4d6488c…`).
  - Rewrite the conservation README's saturation narrative: 0.9995 lies inside the null.
  - Stamp the background revision and sha256 into `example_output.json`.
  - Add a test that recomputes committed percentiles from the pinned NPZ.

#### P1-2 · new since v0.7.5 · The quickstart notebook downloads ~71 GB of conservation bigwigs on first run
*notebooks-determinism*

- **File:** `examples/notebooks/single_oracle_quickstart.ipynb`, cell 44 (added in 3d0b6b1).
- **Problem:** The cell calls `build_variant_report(..., show_conservation=True)` unconditionally, so on a fresh data dir "Run All" pulls seven genome-wide bigwigs: GPN-Star entropy, four GPN-Star LLR tracks, phyloP100way and phastCons100way. Together they are 66.0 GiB (~71 GB).
  - The only warning is in markdown cell 43, while `examples/notebooks/README.md` presents this as the first-timer notebook ("15 min", "runs fully on CPU with 8 GB RAM").
  - The cell's comment says the report embeds "several MB"; it writes 103.5 MB.
- **Evidence:**
  - The prescribed `nbconvert` run was killed at 541 s (exit 124) inside cell 44. By then the GPN-Star entropy track (9.9 GB) and the four LLR tracks (~45 GB) had finished, and phyloP was at 230 MB.
  - `chorus conservation download --track phylop100way` (448 s) and `--track phastcons100way` (337 s) then finished the set.
  - After that, the same command passed in 232 s: 36/36 cells, 0 errors, 0 WARNING lines.
  - `chorus conservation status` lists 9.2 + 10.3 + 10.8 + 10.8 + 10.2 + 9.2 + 5.5 GiB.
- **Fix:**
  - Gate the cell behind a flag that defaults to off (e.g. `RUN_CONSERVATION = False`), or check the `has_*_bigwig()` helpers and print the `chorus conservation download` command instead of downloading.
  - Add the ~70 GB requirement to the notebooks README row and to the "Scaling up" list.
  - Change the "several MB" comment to about 100 MB.

#### P1-3 · new since v0.7.5 · The conservation data sources added in #246 are credited nowhere reachable from the README
*cdf-consistency-hygiene*

- **File:** `docs/THIRD_PARTY.md`
- **Problem:** #246 added three third-party data sources. Chorus downloads them and draws them in every `show_conservation` report:
  - GPN-Star entropy and LLR bigwigs, from `songlab/gpn-star-scores` at pinned revision `a7b13bbf…`;
  - UCSC phyloP 100-way;
  - UCSC phastCons 100-way (both UCSC sources from `hgdownload.soe.ucsc.edu`).

  None is credited in `docs/THIRD_PARTY.md` or in the README Acknowledgments. `THIRD_PARTY.md` was last changed in 6c9d5c8 (2026-08-14), which is inside v0.7.5. The GPN-Star dataset licence is recorded nowhere. The only GPN-Star citation in the repo is in `examples/notebooks/cherimoya_gpn_tutorial.ipynb` ("Ye et al., 2026"). The GENCODE v48 bullet in `THIRD_PARTY.md` points at an "annotation section below" that does not exist.
- **Evidence:** `grep -i 'gpn|phylop|phastcons' docs/THIRD_PARTY.md` finds 0 matches. The README Acknowledgments (`README.md:1477-1488`) list the 8 oracles plus coolbox.
- **Fix:** Add an "Annotation and conservation data" section to `docs/THIRD_PARTY.md`, and link it from the README Acknowledgments and from the conservation walkthrough README. It should cover:
  - GPN-Star: Ye et al., Nature 2026; `songlab/gpn-star-scores` @ `a7b13bbf0d2338d74a7e5f0f8466e41ac0722f50`, with the licence from its dataset card.
  - phyloP (Pollard et al., Genome Res 2010) and phastCons (Siepel et al., Genome Res 2005), on the UCSC multiz100way alignment, under UCSC's data-use terms.
  - GENCODE v48.

#### P1-4 · `predict_variant_effect` cannot score a variant within half a window of a chromosome's 3' end (all of chrM for the large-window oracles)
*install-api-errors*

- **File:** `chorus/core/base.py`, about lines 482–488.
- **Problem:** The region is widened to the oracle window as `region_start = max(0, var - half)` and `region_end = region_start + seqlen`. That clamps the 5' end but never caps `region_end` at the chromosome length.
  - Any variant closer than half a window to a q-terminus therefore raises `InvalidRegionError`. Half a window is 524 kb for AlphaGenome, 262 kb for Borzoi and 196 kb for Enformer, and every chrM position fails whenever the window exceeds 16,569 bp.
  - The error quotes an end coordinate the user never supplied.
  - `predict()` on the same region works, because `Interval.extend` clamps and N-pads.
  - `predict_variant_effect` underpins `analyze_variant_multilayer`, `score_variant_batch`, fine-mapping and discovery.
- **Evidence:** A weights-free stub was used for all but the last item.
  - rs1805007 (MC1R, chr16:89,919,709, 418,636 bp from the end) raises for alphagenome and alphagenome_pt.
  - SHANK3 chr22:50,721,500 and ARSA chr22:50,625,000 raise for enformer, borzoi and both AlphaGenome backends.
  - chrM:100, 8000 and 16519 raise for all four large-window oracles ("End position 393216 exceeds chromosome chrM length 16569").
  - chrM:16519 also raises for legnet, chrombpnet, cherimoya, epinformerseq and sei.
  - Live LegNet: chrM:16519 T>C raises "End position 16618 exceeds chromosome chrM length 16569".
  - The SORT1 control works on every oracle.
- **Fix:**
  - Clamp the widened window to the contig: when `region_end > chrom_len`, set `region_end = chrom_len` and `region_start = max(0, chrom_len - seqlen)`.
  - Let `Interval.extend` N-pad contigs shorter than the window.
  - Add regressions for rs1805007 on AlphaGenome, SHANK3 on Enformer and chrM:16519 on LegNet.

#### P1-5 · LegNet insertions are silently mis-scored: sign flips and false 98th-percentile calls
*install-api-errors*

- **File:** `chorus/oracles/legnet.py:276-277`
- **Problem:** An insertion makes the alt interval longer than the 200 bp window. `query_interval.extend(self.sequence_length)` returns that interval unchanged, so the alt is tiled into 5 mostly N-padded windows while the ref gets 1.
  - `effect_sizes` broadcasts to 5 values, and the report scorer (`scorers.py:478-481`) compares their mean with the ref's single window.
  - No error is raised, although the checklist expects an unsupported non-SNV to fail before the model runs.
  - Deletions and MNVs are unaffected.
  - Sei uses the same `extend()` pattern (`sei.py:540-541`) and is likely affected, but that was not measured.
- **Evidence:** rs12740374 on LegNet HepG2, compared with a correctly centred 200 bp pair from `get_centered_window`:

  | variant | report | correct |
  |---|---|---|
  | G>GT | −0.0886 | +0.4404 |
  | G>GA | −0.4085 (effect percentile −0.985) | +0.0198 |
  | G>G+5A | −0.3109 (percentile −0.97) | +0.3039 |

  The SNV G>T and the deletion GT>G match exactly (+0.3471 and +0.1111). Ref/alt value counts are 1/5 for insertions and 1/1 otherwise.
- **Fix:**
  - Build every allele at exactly `sequence_length`, centred on the variant (as `get_centered_window` does), or crop the alt back to the window.
  - Failing that, raise before the model runs for insertions on scalar-output oracles.
  - Assert `len(ref.values) == len(alt.values)` before computing `effect_sizes`.
  - Add regressions against `get_centered_window` for LegNet and Sei.

#### P1-6 · The LDlink token leaks into MCP server logs and debug tracebacks through exception chaining
*cdf-consistency-hygiene*

- **File:** `chorus/utils/ld.py:169-171`; `chorus/mcp/server.py:2151-2155` and `_safe_tool`.
- **Problem:** `fetch_ld_variants` removes the token from the `LDLinkError` message but raises it `from exc`. That keeps the original `requests` exception attached, and its message is the full request URL, including `token=<value>`.
  - **How it reaches the logs:** `_resolve_rsid_to_position` re-raises `from exc`. `_safe_tool`'s `logger.exception` then prints the whole chain to the server's stderr, which MCP clients save to log files.
  - **Other exposure:** with `CHORUS_MCP_DEBUG=1` the chain is also returned in `payload['traceback']`. A second path: urllib3 2.7.0 logs the full URL at DEBUG.
  - **Reachability:** `analyze_variant_multilayer`, `discover_variant` and `discover_variant_cell_types`, whenever `position` is an rsID and LDlink returns an HTTP error or times out. `fine_map_causal_variant` does not leak. Both chains were already in v0.7.5.
- **Evidence:** An offline probe used a fake token and a monkeypatched `requests.get` that raises `HTTPError`.
  - **`payload['error']`:** clean.
  - **Server log:** contains `…/LDlinkRest/ldproxy?var=rs12740374&pop=CEU&r2_d=r2&token=<FAKE-TOKEN>&genome_build=grch38`.
  - **With `CHORUS_MCP_DEBUG=1`:** `payload['traceback']` contains it too.
  - **Script:** `s4_10_16_17_18/scripts/s16_ldlink_chain_probe.py`.
- **Fix:**
  - Raise the redacted `LDLinkError` `from None`, or scrub `exc.args` first.
  - Redact `LDLINK_TOKEN` and `HF_TOKEN` values in the tracebacks `_safe_tool` logs and returns.
  - Keep the `urllib3` logger at INFO or above around the LDlink request.
  - Add a regression test asserting the token is absent from `caplog.text` and from `traceback.format_exc()`.

#### P1-7 · README result tables in four walkthroughs disagree with `example_output.md` by far more than ±0.006
*mcp-reports*

- **Files:** `examples/walkthroughs/validation/README.md`, `variant_analysis/SORT1_rs12740374/README.md`, `batch_scoring/README.md` and `sequence_engineering/integration_simulation/README.md`.
- **Problem:** The tables violate §7's ±0.006 rule. README value first, artefact value second:
  - **`validation/README.md`:**
    - C/EBP table: CEBPB +2.981 vs +3.316, CEBPA +2.709 vs +2.945, CEBPG +2.269 vs +2.460, CEBPD +1.739 vs +2.033.
    - "RNA:HepG2 (best of 58 rows) +0.718" vs a best RNA row of +0.465.
    - The summary row "CEBPA +0.379, CEBPB +0.269, DNASE +0.450, CAGE +0.239" vs +2.945 / +3.316 / +1.334 / +1.502.
    - TERT "CAGE +0.34" (lines 33 and 89) vs CAGE:K562 — TERT TSS +1.462.
  - **`SORT1_rs12740374/README.md`, lines 33–37:**
    - CEBPB +2.981 with activity 0.977, vs +3.316 and 0.964.
    - CEBPA +2.626 vs +2.945.
    - H3K27ac activity 0.999 vs 0.946.
    - The "+2.98 log2FC … 8-fold" prose and the "Why all ≥99th?" box are out of date: the artefact prints 0.9964–0.9998.
  - **`batch_scoring/README.md`, lines 123–127:**
    - DNASE Effect %ile 0.90 / 0.91 / 0.73 / 0.18 vs 0.79 / 0.81 / 0.59 / 0.08.
    - CEBPA Ref/Alt 2.57e+03 / 1.75e+04 vs 2.07e+03 / 1.6e+04.
    - CEBPA %ile 0.39 / 0.46 / 0.35 / 0.32 vs 0.36 / 0.28 / 0.47 / 0.62.
  - **`integration_simulation/README.md:28`:** H3K27ac:K562 +1.097 vs +1.304.
- **Evidence:** The README lines were last touched between 2026-03-25 and 2026-08-05; the artefacts were regenerated on 08-12 and 08-13. The repo's README guards still pass these (20 passed, 5 skipped); P2-17 explains why.
- **Fix:** Regenerate the tables from each sibling `example_output.md`, or quote that file by reference. Update the TERT row, the summary row and the "8-fold" / "Why all ≥99th?" prose in the same commit.

#### P1-8 · Causal reports describe the effect null as "a background of common variants"
*mcp-reports*

- **File:** `chorus/analysis/causal.py:1608-1611`
- **Problem:** Every per-layer drill-down table carries the footnote "Percentile is the variant effect vs. a background of common variants (when available)." It is the only definition of that column, and it is false. The effect null is about 18,000 synthetic SNPs at sampled regulatory positions, with the alt allele chosen uniformly (`docs/BACKGROUND_NULL_PROTOCOL.md` §6.1). `tests/test_reports_describe_the_real_reference_class.py` exists to forbid exactly this misdescription, but its FORBIDDEN patterns do not match this wording.
- **Evidence:** The phrase appears 11 times in `causal_prioritization/SORT1_locus/rs12740374_SORT1_locus_causal_report.html` and 56 times in `CDYL_rs9504151/rs9504151_CDYL_locus_causal_report.html`. The text dates from 97fdcad, which is in v0.7.5.
- **Fix:**
  - Reuse the how-to-read "Effect %ile" wording.
  - Add Percentile and r² to the causal how-to-read (P2-19).
  - Add "common variants" to FORBIDDEN.
  - Regenerate the two causal reports.

#### P1-9 · `cherimoya_quickstart` hardcodes `../../genomes/hg38.fa` and fails whenever the data dir is relocated
*notebooks-determinism*

- **File:** `examples/notebooks/cherimoya_quickstart.ipynb`: cells 2 and 15, and markdown cell 0.
- **Problem:** `REFERENCE = "../../genomes/hg38.fa"` exists only when the data dir is the install dir.
  - With `CHORUS_DATA_DIR` or the marker file, which is the setup CLAUDE.md recommends, the first predict (cell 11) dies with a bare `OSError`.
  - Cell 15 also opens `REFERENCE` directly with pysam.
  - Markdown cell 0 says to run `chorus genome get hg38`, which is not a valid subcommand. The choices are list, download, remove and info.
  - The hardcoded path dates from dfa0a5e (#107, 2026-07-30).
- **Evidence:** The run fails at cell 11 after 20.6 s with "OSError: file `../../genomes/hg38.fa` not found". The repo's `genomes/` directory holds only `.gitkeep`. With the path symlinked into place, the notebook completes 10/10 cells in 108.3 s.
- **Fix:** Set `REFERENCE = str(get_genome('hg38'))`, as the other notebooks do, and change markdown cell 0 to `chorus genome download hg38`.

#### P1-10 · `cherimoya_quickstart`'s narrative and outputs describe the 5-fold ensemble, but the default has been fold 0 since 0.7.2
*notebooks-determinism*

- **File:** `examples/notebooks/cherimoya_quickstart.ipynb`: markdown cell 23 and every committed output.
- **Problem:** Markdown cell 23 says "**The 5-fold ensemble by default.**", and the committed outputs (2026-08-09) were all produced with the ensemble. The 0.7.2 CHANGELOG and the README make fold 0 the default, with `fold='ensemble'` as an opt-in. Readers are told they get the ensemble and are shown numbers they cannot reproduce.
- **Evidence:** A fresh run logs "Loading Cherimoya DNASE:ENCSR000EOT (K562, fold 0)", where the committed copy says "(K562, 5-fold ensemble)". Every printed value moves:

  | value | committed (ensemble) | fresh (fold 0) |
  |---|---|---|
  | total signal | 2228.4 | 2808.3 |
  | ref / alt | 1769.0 / 1765.8 | 2290.7 / 2298.8 |
  | raw log2FC | −0.0026 | +0.0051 (sign flip) |
  | effect percentile | 0.058 | 0.108 |
  | biosample table, K562 | 1770.1 | 2293.4 |
  | biosample table, GM12878 | 151.0 | 121.1 |
  | biosample table, HepG2 | 333.1 | 278.0 |
  | biosample table, H1 | 67.9 | 75.7 |
- **Fix:**
  - Rewrite the caveat to say:
    - fold 0 is the default;
    - `fold='ensemble'` gives the 5-fold mean;
    - each mode is ranked against its own null.
  - Re-execute on GPU and commit the outputs.

#### P1-11 · `comprehensive_oracle_showcase` prints "AlphaGenome total tracks: 5731", contradicting the README's 5,168
*notebooks-determinism*

- **File:** `examples/notebooks/comprehensive_oracle_showcase.ipynb`: cell 25, markdown cells 1 and 24, and a comment in cell 50.
- **Problem:** Cell 25 prints `len(ag_meta._track_index_map)`. That count includes the 563 padding rows that #251 excluded, so the printed figure fails §6's P1 printed-counts check.
- **Evidence:**
  - The fresh cell 25 prints "AlphaGenome total tracks: 5731" and "ATAC tracks: 167" to stdout, and "Loaded 5168 AlphaGenome tracks (5731 metadata rows incl. 563 padding)" to stderr.
  - `README.md:147,192,549` say 5,168, as they already did at v0.7.5.
- **Fix:** Print `sum(1 for _ in ag_meta.iter_tracks())`, and change markdown cells 1 and 24 and the cell 50 comment to 5,168.

#### P1-12 · The same notebook's hardcoded "H3K27ac +80" insight matches no computed number
*notebooks-determinism*

- **File:** `examples/notebooks/comprehensive_oracle_showcase.ipynb`, cell 53.
- **Problem:** Cell 53 prints a fixed string: ">> KEY INSIGHT: The variant affects chromatin marks locally (H3K27ac +80)". None of the values it summarises is 80, which fails §6's P1 narrative-numbers check. The string has been there since 93cd4c0 (2026-04-21).
- **Evidence:**
  - Cell 52 prints "L2_H3K27ac 763.71 952.00 +188.2857 Active enhancer GAIN <<<".
  - Cell 51 gives an at-variant value of −64.0000 and a peak of +416.0000.
  - The slice run reproduces the committed outputs exactly.
- **Fix:** Format the number from `variant_scores`, e.g. `f"H3K27ac {variant_scores['alt_1'][tid]['effect']:+.0f}"`, or drop it.

#### P1-13 · CI is red on `4258b6f`: the browser meta-test skips the outage guard #252 extended
*integration-suite*

- **File:** `tests/test_committed_reports_render_in_a_browser.py`, `test_the_check_actually_detects_a_broken_panel` (around lines 196–246).
- **Problem:** #252 taught `browser_harness.unreachable_external_host` to treat a CORS-shaped UCSC failure as an outage, but this meta-test never consults it.
  - When UCSC's `hg38.2bit` is CORS-blocked or down, igv.js paints nothing (canvases 0/0).
  - The "features emptied → blank" mutation can then not be detected, and the test fails.
  - This makes the release commit's CI depend on UCSC.
  - The test dates from a3a176b (2026-08-12).
- **Evidence:** gh run 43764177834 (`tests.yml`, headSha `4258b6f`):
  - The fast-suite job passed. `browser smoke (2 reports)` failed: 1 failed / 6 passed / 5 skipped.
  - Failure message: "features emptied: the blank channel did not notice … canvases 0/0 painted (NOT converged)".
  - Browser console: "Access to XMLHttpRequest at 'https://hgdownload.soe.ucsc.edu/…/hg38.2bit' … has been blocked by CORS policy".
  - #253 changed no test code, and its CI is green, which points to a transient outage.
  - It cannot be reproduced locally because there is no Chromium on this host.
- **Fix:**
  - Make the meta-test hermetic: serve the sequence locally with `CHORUS_IGV_SEQUENCE_URL` and a self-hosted hg38, as `test_reports_bundle_their_genome` does.
  - Failing that, skip when `bh.unreachable_external_host(r)` returns a reason.
  - Re-run CI on the final release sha.

#### P1-14 · Release-gate tests and the determinism gate hardcode a repo-relative `genomes/hg38.fa`, so they skip or fail on any install that uses a data dir
*install-api-errors, integration-suite*

- **Files:**
  - `tests/test_smoke_predict.py:28`
  - `tests/test_cherimoya_integration.py:51`
  - `test_alphagenome_backends_equivalence.py:49`
  - `test_integration.py:107`
  - `test_strict_ref.py:33` and `:182`
  - `test_region_strata_are_what_they_claim.py:40`
  - `test_gene_anchored_regions.py:42`
  - `test_genome_is_asserted_not_assumed.py:130`, `:147` and `:170`
  - `test_reports_bundle_their_genome.py:102` and `:289`
  - `scripts/gate_end_to_end_determinism.py:78`
  - `test_padding_never_returns_a_short_sequence.py:32` hardcodes this host's `/data/chorus_data` instead.
- **Problem:** The genome lives at `<data_dir>/genomes/hg38.fa`, but these files look for it relative to the repo or the cwd.
  - On a host configured the way CLAUDE.md describes, that skips or breaks the 6 oracle smoke tests, all 11 Cherimoya tests, the ChromBPNet fresh download, the AlphaGenome JAX-vs-PyTorch pair and the bit-exact release gate.
  - A green `pytest -m integration` here verifies none of them.
  - The skip text recommends `chorus genome get hg38`, which does not exist.
- **Evidence:**
  - As written, 23 integration tests skip and 2 fail. The failures are `test_jax_pt_agree_across_every_output_type` and `test_a_report_is_bit_exact_across_two_processes`, both with "OSError: file …/genomes/hg38.fa not found".
  - 22 fast-suite tests also skip for the same reason.
  - With the genome reachable through gitignored symlinks in a scratch clone, all of them pass:
    - smoke 6/6;
    - Cherimoya 10 passed, plus 1 unrelated skip;
    - AlphaGenome equivalence 2/2;
    - release gate 1/1;
    - strict_ref 1/1;
    - region strata 3/3.
  - The correct pattern already exists in `test_every_shipped_track_is_reachable.py`, `test_cherimoya_ensemble.py` and `test_track_geometry_is_self_consistent.py:114`.
- **Fix:**
  - Resolve the FASTA in one shared conftest fixture (`GenomeManager().get_genome_path('hg38')` or `chorus.core.globals.CHORUS_GENOMES_DIR`) and use it in every file listed.
  - Give `test_jax_pt_agree_across_every_output_type` a FASTA guard.
  - Under `CHORUS_RELEASE_GATE=1`, make a missing prerequisite fail rather than skip.
  - Change the skip message to `chorus genome download hg38`.

#### P1-15 · `test_data_dir_resolution` fails on configured installs and rewrites the live data-dir marker
*install-api-errors, integration-suite*

- **File:** `tests/test_data_dir_resolution.py`, line 76 and lines ~161–181.
- **Problem:** Two related defects:
  - **(a)** `test_default_is_the_installation_directory_not_home` strips `CHORUS_DATA_DIR` but not `<install>/chorus_data_dir.txt`. The fast suite is therefore red on any install set up the documented way (`chorus setup --data-dir` or `chorus config data-dir --set`), which blocks §11's "0 fail" gate on this host.
  - **(b)** `test_marker_file_redirects_and_env_var_beats_it` writes a pytest tmp path into the real marker and restores it in `finally`.
    - For about 4 s, any chorus process that starts resolves its data dir to that tmp path. That includes every env-mode oracle subprocess and the MCP server.
    - A run killed inside the window leaves the install pointing at a deleted tmp dir, because Python runs no `finally` on `timeout`'s SIGTERM.
  - Both come from 4b9d33d (2026-08-10).
- **Evidence:**
  - Fast-suite run: "AssertionError: default data dir is /data/chorus_data, expected the installation directory `<repo>`" (1 failed).
  - The marker was rewritten at 21:53:07 and 22:30:11 and restored byte-identical both times.
  - In the clone the test took 4.02 s.
- **Fix:** Never touch the real marker. Give the globals a test-only override, such as `CHORUS_DATA_DIR_MARKER` or a root override, or run `_layout()` against a temp copy of the package. Use it in both tests.

#### P1-16 · The MCP E2E release-gate test drops `CUDA_VISIBLE_DEVICES` and `CHORUS_MCP_OUTPUT_DIR` from the child's environment
*integration-suite*

- **File:** `tests/test_integration.py`, `test_mcp_e2e_list_oracles_and_analyze_variant` (around lines 246–262).
- **Problem:** The `StdioTransport` env is built from a fixed whitelist: `HF_TOKEN`, `CHORUS_NO_TIMEOUT`, `PATH`, `MAMBA_ROOT_PREFIX`, `MAMBA_EXE`, `HOME` and `HF_*`. No cwd is set. The whitelist dates from 6bae3db (2026-04-17).
  - **No `CUDA_VISIBLE_DEVICES`:** the child, and its AlphaGenome env child, see every GPU on the host. Nothing in chorus sets `XLA_PYTHON_CLIENT_PREALLOCATE`, so JAX's default preallocation applies on initialisation; here that includes GPUs 4–7. This is from reading the code: the test was deliberately not run unconfined.
  - **No `CHORUS_MCP_OUTPUT_DIR`:** `chorus/mcp/state.py:36` falls back to `cwd/chorus_mcp_output`. When the test runs from the repo root, that writes into the checkout.
  - **No `CHORUS_DATA_DIR`:** an install configured by env var resolves the wrong data dir.
- **Evidence:**
  - Recorded whitelist keys: `CHORUS_NO_TIMEOUT`, `HF_HUB_CACHE`, `HF_TOKEN`, `HOME`, `MAMBA_EXE`, `MAMBA_ROOT_PREFIX`, `PATH`.
  - With `CUDA_VISIBLE_DEVICES=0,1`, `CHORUS_MCP_OUTPUT_DIR` and `CHORUS_DATA_DIR` forwarded by an audit-only plugin, the unchanged test passed in 283.41 s.
  - The checkout's gitignored `chorus_mcp_output/` already holds two reports dated 2026-11-15 14:18 and 14:20 UTC, from before this audit.
- **Fix:**
  - Forward `CUDA_VISIBLE_DEVICES`, `CHORUS_DATA_DIR` and `TMPDIR` when they are set.
  - Always pass `CHORUS_MCP_OUTPUT_DIR=str(tmp_path)` and `cwd=str(tmp_path)`.
  - Assert that the report landed there.

#### P1-17 · `chorus-alphagenome_pt` installs alphagenome-pytorch from git HEAD and alphagenome from PyPI, both unpinned — fixed on `main` by #253 (`eab109c`)
*cdf-consistency-hygiene*

- **File:** `environments/chorus-alphagenome_pt.yml:38`
- **Problem at `4258b6f`:**
  - **No ref on the git dep.** The line reads `alphagenome-pytorch[scoring] @ git+https://github.com/genomicsxai/alphagenome-pytorch.git`, so a fresh `chorus setup` builds whatever is at HEAD.
  - **Unbounded transitive dep.** The `[scoring]` extra declares an unbounded `alphagenome`, so PyPI's latest would arrive. That is the 2.x line, which removed the legacy import paths.
  - **Why it matters.** The two backends share one CDF via `_CDF_ALIASES`.
  - **No test caught it.** `tests/test_alphagenome_deps_are_pinned.py` read only `chorus-alphagenome.yml`.
- **Evidence:**
  - The working env has alphagenome-pytorch 0.3.2.dev72+g473d3d226 (commit `473d3d226b3a0aec754fb47cb6fcab1028829435`) and alphagenome 0.7.0, installed from pip.
  - The file was last changed in d1b0bcd (2026-04-29).
- **Status:** #253, merged during the audit, makes three changes:
  - pins `alphagenome-pytorch[scoring]` at `@473d3d226b3a0aec754fb47cb6fcab1028829435`;
  - adds `alphagenome==0.7.0`;
  - adds a scan requiring every git dep in every env file to name a full sha.

  For this report, that test file was run on an extracted copy of `eab109c`'s tests and env files: 5 passed. No env was built from the new pins.
- **Fix:** Nothing further at the git-dep level. The floating non-git deps are P2-32.

#### P1-18 · The documented `huggingface-cli login` path no longer works in the shipped envs
*install-api-errors*

- **File:**
  - Code: `chorus/oracles/alphagenome.py:199`, `alphagenome_source/templates/load_template.py:51`, `cli/main.py:98`, `cli/_setup_all.py:79`, `core/weights_probe.py:127` and `cli/_tokens.py:46`.
  - Docs: `README.md:1162` and `README.md:1306`.
  - Test: `tests/test_error_recovery.py:172` pins the string.
- **Problem:** huggingface_hub resolves to 1.26.0 in `chorus`, `chorus-alphagenome` and `chorus-alphagenome_pt`. It is unpinned in the two env files and `>=0.20.0` in `environment.yml`, so every fresh install gets 1.26.0. In that version `huggingface-cli` refuses to run, yet chorus's error messages and the README's "Option B" still tell users to run it.
- **Evidence:**
  - In `chorus-alphagenome`, `huggingface-cli --help` exits 1 with "Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead."
  - `hf auth login --help` works.
  - A live load with no token prints "run `huggingface-cli login`".
  - Loading from an existing cached login works (66.1 s, CPU), so only the advice is broken.
- **Fix:**
  - Replace it with `hf auth login` in the code, the README and the test assertion. Optionally add "or `huggingface-cli login` on huggingface_hub < 1.0".
  - Consider pinning `huggingface_hub>=1.0` in the AlphaGenome envs, so the advice always matches the installed binary.

### P2 (33)

#### Library, CLI and MCP

**P2-1 · new since v0.7.5 · AnnotationStore and the new `list_annotations` tool report GENCODE as not downloaded and list its index files as annotations** *(notebooks-determinism, mcp-reports)*

- **File:** `chorus/utils/annotations.py`, about lines 280–296.
- **Problem:** `AnnotationManager.list_annotations()` has two bugs.
  - It checks for the URL basename `gencode.v48.basic.annotation.gtf.gz`, which chorus deletes after decompressing. So `gencode_v48_basic` reports `downloaded=False` even though `download_gencode()` uses the `.gtf`.
  - Its `*.gtf*` glob keys entries by `Path.stem`, so the tabix `.gtf.bgz` and its `.tbi` show up as "Local GTF file" entries with no genome build.

  The code itself is old (ee1ca96). #246 makes the bugs user- and agent-visible through AnnotationStore, quickstart cell 45 and the three new MCP tools.
- **Evidence:**
  - Quickstart cell 45, fresh and committed, prints "gencode_v48_basic [gtf] hg38 not downloaded". The same run logs "Annotation file already exists: /data/chorus_data/annotations/gencode.v48.basic.annotation.gtf".
  - Live `list_annotations` returns three extra ids: `gencode.v48.basic.annotation`, `.gtf` and `.gtf.bgz`. The last one points at the `.gtf.bgz.tbi`.
  - The directory holds `.gtf`, `.gtf.bgz` and `.gtf.bgz.tbi`, and no `.gtf.gz`.
- **Fix:**
  - Use `annotation_exists()` for the downloaded flag.
  - Exclude `.tbi`, `.csi`, `.fai` and the `.bgz` siblings of known sources from the glob.
  - Derive ids by stripping every GTF suffix.
  - Add a test with this file layout.

**P2-2 · AlphaGenome metadata load rewrites the tracked `alphagenome_tracks.json` on every load, non-atomically** *(all of cdf-consistency-hygiene, integration-suite, notebooks-determinism, mcp-reports)*

- **File:** `chorus/oracles/alphagenome_source/alphagenome_metadata.py`, lines 52, 103–107 and 158–162.
- **Problem:** `_load_metadata()` rebuilds the track list from `alphagenome_research` and always calls `_save_cache()`. That does `open(…, 'w')` on the git-tracked 1.74 MB `alphagenome_tracks.json`.
  - **Where it runs:** every `chorus-alphagenome` child, the predict worker and the determinism gate.
  - **Who reads the file:** base-env processes (the MCP server, `list_tracks`) and alphagenome_pt use it as their track list.
  - **Consequences:**
    - A different `alphagenome_research` would silently rewrite the committed list and dirty the checkout.
    - A reader that lands mid-write gets truncated JSON, leaving AlphaGenome metadata empty for that process's lifetime. This is plausible from the code but was not reproduced.
    - Read-only installs log a warning on every call.
  - **Age:** since 53558dc (2026-03-13).
- **Evidence:** The file's mtime moved at least seven times during the audit (21:46:08, 21:57:39, 22:01:08, 22:04:32, 22:17:42, 22:20:47 and 22:48:15 UTC). Every write was byte-identical: `git hash-object` equals HEAD's blob `7b65582e…`.
- **Fix:**
  - Write only when the content differs, via a temp file and `os.replace`.
  - Better, keep the runtime cache under the data dir and treat the shipped JSON as read-only.

**P2-3 · Three MCP error returns bypass the `{error, error_type, tool}` shape** *(mcp-reports)*

- **File:** `chorus/mcp/server.py`, lines 424, 613 and 2078.
- **Problem:** These three error returns use a different shape:
  - `list_tracks` with an unknown oracle returns `{error, oracle}`.
  - The `list_tracks` fall-through returns `{error}`.
  - `fine_map_causal_variant`'s `LDLinkError` branch returns `{error}`.

  None of them leaks a traceback, but clients that key on `error_type` or `tool` miss them.
- **Evidence:** The live probe of `list_tracks('not_an_oracle')` returns keys `['error','oracle']`. The `load_oracle`, `analyze_variant_multilayer`, `predict_variant_effect` and `describe_annotation` probes return exactly `['error','error_type','tool']`. Separately, `get_gene_tss('NOT_A_REAL_GENE_XYZ')` returns success with 0 transcripts and no hint.
- **Fix:** Raise and let `_safe_tool` format the error, or add `error_type` and `tool` to the three dicts. Assert the shape in `tests/test_mcp.py`.

**P2-4 · `list_oracles` gives Sei's assay types as `sequence-class` only** *(cdf-consistency-hygiene, mcp-reports)*

- **File:** `chorus/mcp/server.py`, `ORACLE_SPECS['sei']` (line 98).
- **Problem:** Agents filtering by assay type will conclude that Sei has no DNase or histone outputs.
  - `assay_types` has been `['sequence-class']` since 33c9ad69 (2026-03-15).
  - The same entry's description, updated in #251, and `list_tracks('sei')` both expose 21,907 chromatin profiles: DNase/ATAC, histone and TF ChIP.
  - `test_list_oracles_assay_types_match_the_metadata` covers only enformer, borzoi, alphagenome and alphagenome_pt.
- **Evidence:** `sei_pertrack.npz` holds 21,947 track_ids. `list_tracks('sei','DNase')` and `('sei','H3K27ac')` each return 200 (truncated), and `('sei','sequence-class')` returns 40.
- **Fix:** Derive `assay_types` from the metadata, or list Sei's profile families, and parametrize the test over every oracle.

**P2-5 · Three `base.py` entry points still raise the old missing-reference message** *(install-api-errors)*

- **File:** `chorus/core/base.py`, lines 362, 403 and 447.
- **Problem:** `predict_region_replacement`, `predict_region_insertion_at` and `predict_variant_effect` raise "No reference genome provided. Either pass genome parameter or initialize oracle with reference_fasta." That message omits `chorus genome download hg38`. `predict()` already uses `missing_reference_fasta_error` (`utils/genome.py:486`).
- **Evidence:** With LegNet loaded without `reference_fasta`, `predict(tuple)` gives the new message and the three entry points give the old one.
- **Fix:** Raise `missing_reference_fasta_error(self.oracle_name)` at all three sites, keeping a mention of `genome=`.

**P2-6 · `predict()` given a region string fails with `KeyError: 'H'`** *(install-api-errors)*

- **File:** `chorus/core/base.py`, lines 259–266.
- **Problem:** `predict()` passes a `str` to `_predict` as literal DNA, without calling `_validate_dna_sequence`. A region string, the same form `predict_variant_effect` accepts, therefore crashes in the one-hot encoder. This is the trap CLAUDE.md documents for alphagenome_pt.
- **Evidence:** Calling LegNet directly with `predict('chr1:109274468-109275468', ['LentiMPRA:HepG2'])` raises `KeyError 'H'` at `chorus/oracles/legnet_source/transforms.py:26`.
- **Fix:** If the string matches `^\w+:\d+-\d+$`, raise `InvalidSequenceError` saying it looks like a genomic region and a `(chrom, start, end)` tuple is needed. Otherwise, call `_validate_dna_sequence` before `_predict`.

**P2-7 · `extract_sequence` relabels region errors as "not indexed" whenever the path contains "index", and prints a literal `{genome}`** *(install-api-errors)*

- **File:** `chorus/utils/sequence.py`, lines 144–150.
- **Problem:** The catch-all converts any exception whose text contains "index" into `FileFormatError` "… is not indexed", and its second line lacks the f-prefix.
  - Because `InvalidRegionError`'s message includes the genome path, a FASTA stored under `genome_index/`, `bowtie2_index/` or `indexes/` turns a mistyped chromosome into wrong advice.
  - A FASTA that genuinely cannot be indexed raises pysam's "error when opening file" instead, so the helpful branch never fires.
- **Evidence:** For `<scratch>/genome_index/hg38.fa` (a symlink to hg38), `extract_sequence('chrZZ:1-5')` raises "…is not indexed. Please run 'samtools faidx {genome}' to create index." An unindexed FASTA in a read-only directory raises a raw `OSError`.
- **Fix:**
  - Re-raise `InvalidRegionError` and `FileFormatError` untouched.
  - Detect a missing index by checking for `<genome>.fai`.
  - Add the f-prefix.

**P2-8 · The AlphaGenome no-token error differs by path and never says to export `HF_TOKEN`** *(install-api-errors)*

- **File:** `chorus/oracles/alphagenome_source/templates/load_template.py`, lines 59–64.
- **Problem:** The two load paths fail differently:
  - The direct path raises `ModelNotLoadedError`.
  - The env path surfaces as `builtins.RuntimeError` "Code execution failed: …" with two chained tracebacks, so callers that catch `ModelNotLoadedError` miss it.

  Neither message tells a user whose token lives only in `~/.zshrc` to export it in the shell that starts claude or the MCP client (checklist §2, last item). Only `README.md:1165` says so.
- **Evidence:** With no credential, the direct path gives `ModelNotLoadedError` in 2.2 s. The env path gives a `RuntimeError` in 4.2 s: about 1.5 KB of text with 2 "Traceback" blocks. Neither message contains "export" or "shell".
- **Fix:** Map the template's auth failure to `ModelNotLoadedError` carrying only the first line, and add the export hint to both messages.

**P2-9 · Sei's `fine_tune` still points users at AlphaGenome and Borzoi** *(install-api-errors)*

- **File:** `chorus/oracles/sei.py`
- **Problem:** `NotImplementedError` reads "Sei fine-tuning is not supported. Use AlphaGenome or Borzoi for workflows that need on-the-fly track adaptation." Both of those raise "Fine-tuning is not yet implemented". Checklist §5 already flags this, and it is unchanged.
- **Evidence:** Live calls on all 9 oracles.
- **Fix:** Reword it like LegNet's message ("train a new model externally").

**P2-10 · `chorus backgrounds status` omits cherimoya_ensemble and says alphagenome_pt has no backgrounds** *(install-api-errors)*

- **File:** `chorus/cli/_backgrounds.py`, lines 27–30.
- **Problem:**
  - `_KNOWN_ORACLES` has no `cherimoya_ensemble`, although its NPZ has shipped since `backgrounds-2026-08-12-cherimoya-fold0`.
  - For alphagenome_pt, the command looks up a nonexistent `alphagenome_pt_pertrack.npz`. It then prints exactly the misleading line its own code comment set out to avoid.
- **Evidence:** The output has 9 rows, none for cherimoya_ensemble, and shows "alphagenome_pt  --  no backgrounds found". `cherimoya_ensemble_pertrack.npz` is present (161,658,353 B, 1518 tracks).
- **Fix:**
  - Add `cherimoya_ensemble` to the list.
  - Resolve aliases via `PerTrackNormalizer._CDF_ALIASES`, and print e.g. "alphagenome_pt -> uses alphagenome_pertrack.npz (5168 tracks)".

**P2-11 · `download_gencode` returns a `str` on a cache hit, has no timeout, and leaves partial downloads that look cached** *(install-api-errors)*

- **File:** `chorus/utils/annotations.py`, lines 190–205.
- **Problem:**
  - On a cache hit it returns a `str`, although the function is annotated `-> Path`.
  - `requests.get(url, stream=True)` has no timeout, so a firewall that silently drops packets hangs it forever.
  - It streams straight to the final `.gtf.gz` name, so an interrupted download is treated as cached on the next call.
- **Evidence:** `download_gencode(version='v48', annotation_type='basic')` returns `'/data/chorus_data/annotations/gencode.v48.basic.annotation.gtf'`, a `str`.
- **Fix:**
  - Return a `Path`.
  - Download through `chorus.utils.http.download_with_resume`, which writes a `.partial` file, resumes with Range requests and has timeouts, and rename to the final name only on success.

**P2-12 · AlphaGenome env-path predict costs ~90 s per call, against the README's "~30 s (GPU)"** *(notebooks-determinism)*

- **File:** `chorus/oracles/alphagenome.py`, the `_predict_alleles` path.
- **Problem:** With `use_environment=True`, every `predict()` spawns a fresh `mamba run` process (`runner.py:224-229`) that reloads the model and re-JITs.
  - `predict_variant_effect` makes one such call per allele, so two alleles cost about 184 s.
  - `predict_session()` would amortise the load, but neither the notebook nor `_predict_alleles` uses it.
  - This is why the comprehensive notebook overruns the time cap.
- **Evidence:**
  - `load_pretrained_model`: 71.9 s.
  - Fresh-process predicts: 91.8 s and 86.7 s.
  - Inside `predict_session()`: 15.5 s, then 0.8 s, with identical output.
  - Comprehensive cell 51: 183.8 s.
- **Fix:** When `use_environment=True`, run all alleles inside one `predict_session()`, or wrap the notebook's multi-call cells in one. State the env-path cost in the README matrix.

**P2-13 · Enformer is still non-deterministic across processes (F8 still open)** *(notebooks-determinism)*

- **File:** `chorus/core/determinism.py`
- **Problem:**
  - Two back-to-back `use_environment=True` predicts differ on every bin.
  - The fix measured in `audits/2026-08-14_f8_localisation.md` (`TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`) has not been adopted, because it requires rebuilding the Enformer and ChromBPNet nulls.
- **Evidence:** For ENCFF413AHU at chr1:1,000,000–1,100,000:
  - `allclose(atol=1e-6)` is False.
  - max abs diff 1.750e-3, max rel diff 9.95e-4.
  - 896/896 bins differ.

  Notebook drift agrees: CNhs11250's peak moved from −29.3518 to −29.2754.
- **Fix:**
  - Either adopt the flags for Enformer, with a null rebuild and a two-arm measurement as BACKGROUND_NULL_PROTOCOL requires, or state the cross-process wobble in the release notes and README.
  - Either way, the release notes must not claim all-oracle determinism.

#### Nulls

**P2-14 · ChromBPNet background nulls are 38.6% negative, not "a few small negatives", and four TF rows are entirely negative** *(cdf-consistency-hygiene)*

- **File:** `docs/NORMALIZATION_GUIDE.md` (lines 223–227) and `audits/AUDIT_CHECKLIST.md` §4 (line 97). The display path is `perbin_floor_rescale_batch`.
- **Problem:** Negative entries are widespread in ChromBPNet's summary CDFs.
  - **Scale:** 2,903,382 of 7,530,000 summary entries are negative. They span 599 of the 744 BPNet CHIP rows, but only 1 of the 9 ATAC/DNase rows.
  - **Per row:** 312 rows have a negative median, 102 a negative p95 and 55 a negative p99.
  - **Entirely negative:** four rows, CHIP:HEK293:ZSCAN21, KLF7, ZNF664 and ZNF24.
  - **Cause:** the shared two-strand transform exp(C) − n_strands (`scripts/build_backgrounds_chrombpnet.py:378`).
  - **What is fine:** builder and oracle agree, and the effect CDFs have no negatives. Cherimoya matches the guide: 195 entries in 146 rows, min −0.38.
  - **User impact:** on these rows, a near-empty prediction ranks as top genome-wide activity and fills the IGV colour scale.
- **Evidence:** On CHIP:HEK293:ZSCAN21, `activity_percentile(count=0.0)` = 1.0 and `activity_percentile(count=-0.3)` = 0.903. `perbin_floor_rescale_batch` maps bins [−0.001, 0.0, 0.001] to [0, 1.36, 3.0]. `tests/test_chrombpnet_counts.py` gives 23 passed.
- **Fix:** Document the measured figures and the exp(C) − 2 cause. Any behavioural change is a maintainer decision requiring a two-arm measurement. Options include declining or log-scaling the floor rescale for rows with p99 ≤ 0, or a "null entirely ≤ 0" note on their activity percentiles.

#### Reports and walkthroughs

**P2-15 · new since v0.7.5 · 18 of the 20 IGV reports still inline the pre-#246 igv.js 3.1.1 bundle** *(mcp-reports, cdf-consistency-hygiene)*

- **File:** `examples/walkthroughs/**/*.html` and `chorus/analysis/static/igv.min.js`.
- **Problem:** #246 (3b59ca9) replaced the vendored bundle, moving it from 3.1.1 to 3.8.4.
  - Only `rs12740374_SORT1_chrombpnet_report.html` and the conservation report inline 3.8.4, so the shipped corpus mixes two library versions.
  - CI's browser smoke renders one 3.8.4 report.
  - `CHANGELOG.md`'s `[Unreleased]` already records this as "Still open".
- **Evidence:**
  - By sha256 of the inlined script block, 18 reports match the bundle at 3b59ca9^ (`9977ee77…`) and 2 match the current one (`91bda9aa…`).
  - The 18 are jQuery-era (jQuery 3.3.1, DOMPurify 3.2.1). The 2 carry DOMPurify 3.4.11. 1 report has no IGV panel.
  - A freshly generated MCP report inlines 3.8.4.
- **Fix:** Regenerate the 18 reports, or state the mixed corpus in the release notes.

**P2-16 · new since v0.7.5 · The conservation walkthrough's `example_output.md` and `.tsv` omit the conservation scores its README quotes** *(mcp-reports)*

- **File:** `examples/walkthroughs/conservation/SORT1_rs12740374/example_output.md`
- **Problem:**
  - The README's phyloP −0.046, phastCons 0.001 and GPN-Star 0.772 bits exist only under `example_output.json['conservation']`.
  - The `.md` is byte-identical to `variant_analysis/SORT1_chrombpnet/example_output.md` apart from the Generated line.
  - So the ±0.006 rule has nothing to check the walkthrough's actual subject against.
- **Evidence:** Diffing the two `.md` files shows only line 9, the timestamp.
- **Fix:** Render the conservation block into the `.md` and `.tsv` when `show_conservation=True`.

**P2-17 · The walkthrough README guard does not match numbers to track labels** *(mcp-reports)*

- **File:** `tests/test_walkthrough_readmes_match_artefacts.py`
- **Problem:** The test accepts a README number if any numeric field anywhere in the sibling or linked JSON/TSV is within 0.005.
  - With thousands of fields, stale values match by accident.
  - Unsigned percentiles are never checked.
  - `test_summary_tables_match_their_artefact_by_label.py` covers only the discovery summary.
  - This is why P1-7 passes CI.
- **Evidence:** Three stale values passed by coincidence:
  - CEBPB +2.981 matched a CAGE alt_value of 2.9811.
  - CEBPA +2.626 matched a CAGE alt_value of 2.626.
  - H3K27ac +1.097 matched a CAGE raw_score of 1.0967.

  Both guard files: 20 passed, 5 skipped.
- **Fix:** Compare each README row with the artefact row that has the same track label and column, as the agent's `s7_readme_tables.py` does, and include the unsigned percentile and activity columns.

**P2-18 · Per-layer tables show strand-ambiguous track labels: 519 rows across 8 reports** *(mcp-reports)*

- **File:** `chorus/analysis/variant_report.py`
- **Problem:** The label is `ts.description` plus `region_label`, so + and − strand CAGE/RNA tracks render identically, with no tooltip. IGV track names repeat the same way. The batch report already shows (+)/(−).
- **Evidence:** In the SORT1 AlphaGenome report, "CAGE:HepG2 — variant site" appears twice: the `/+` track at +1.203 and the `/-` track at +1.502. Rows sharing a label within one table:

  | report | rows |
  |---|---|
  | discovery HepG2 | 205 |
  | CEBP validation | 116 |
  | discovery MCF_10A | 60 |
  | amniotic | 58 |
  | SORT1 AlphaGenome | 58 |
  | FTO | 12 |
  | BCL11A | 8 |
  | Enformer | 2 |
- **Fix:** When a description is not unique, append the strand or the assay_id tail.

**P2-19 · "How to read this report" does not define every numeric column** *(mcp-reports, §7 check)*

- **File:** `chorus/analysis/causal.py` and `chorus/analysis/_report_glossary.py`
- **Problem:**
  - The causal how-to-read defines Composite, Max Effect, Layers, Convergence and Per-layer effect, but not r² or the drill-down Percentile. Percentile's only definition is the P1-8 footnote.
  - Multi-oracle tables label the column "%ile", while the block defines "Effect %ile", plus an "Activity %ile" that no multi-oracle column shows.
- **Evidence:** The §7 structural check across all 21 reports.
- **Fix:** Add r² and Percentile to the causal block, and align the multi-oracle column label with its definitions.

**P2-20 · The multi-oracle README contradicts its own table, and the recorded prompt says three oracles where four are scored** *(mcp-reports)*

- **File:** `examples/walkthroughs/validation/SORT1_rs12740374_multioracle/README.md` and `scripts/regenerate_multioracle.py:73`.
- **Problem:**
  - The table (line 36) says "Cherimoya (fold 0, the default) … +1.793, 3.463", which matches the artefact.
  - Lines 61–67 say Cherimoya "reports the average of its five cross-validation folds" and that "Ensembling narrowed the three-way spread … to 2.52–2.75".
  - `USER_PROMPT` says "three independent deep-learning oracles", while the report says it was "scored independently by 4".
- **Evidence:** The README lines blame to 2026-08-10/11. The HTML lists chrombpnet, cherimoya, legnet and alphagenome.
- **Fix:** Rewrite the ensemble paragraph for the fold-0 default, and update `USER_PROMPT` at the next regeneration.

**P2-21 · The validation README converts an RNA lnFC with base 2 and labels it log2FC** *(mcp-reports)*

- **File:** `examples/walkthroughs/validation/README.md`, lines 26, 60 and 69–75.
- **Problem:** RNA uses ln((alt+0.001)/(ref+0.001)), as the README's own formula table says.
  - So +0.718 is e^0.718 = 2.05-fold, not 2^0.718 = 1.65-fold.
  - The table header "Effect (log2FC)" covers the RNA row, and the RNA background is described as "|log2FC| = 0.1".
  - The +0.718 is itself stale: the best RNA row is +0.465, which is 1.59-fold.
- **Evidence:** The report's HTML badge for RNA is lnFC.
- **Fix:** Label the RNA row lnFC, convert with e^x, and refresh the value from the artefact.

#### Notebooks

**P2-22 · new since v0.7.5 · Committed notebook outputs are stale and expose a contributor's home directory** *(notebooks-determinism, cdf-consistency-hygiene)*

- **File:** `examples/notebooks/single_oracle_quickstart.ipynb`, plus `advanced_multi_oracle_analysis` and `comprehensive_oracle_showcase`.
- **Problem:** The quickstart was re-executed on another machine and committed in 3d0b6b1.
  - **Warnings:** its outputs carry 10 lines of "WARNING - enformer background … is superseded (no effect_retained … predates the retention rebuild)". A fresh run here shows none.
  - **Home path:** a contributor's absolute home path appears 10 times, as `/home/<contributor>/.chorus/backgrounds/enformer_pertrack.npz`. That is the legacy location, not the documented data dir.
  - **Version strings:** `advanced` and `comprehensive` print "Chorus version: 0.7.3".
  - **Stale counts:** `comprehensive` still shows "Loaded 5731 AlphaGenome tracks" and "ATAC tracks: 256"; a fresh run gives 5168 and 167.
  - **Older host paths:** `/home/nvidia` paths, present before v0.7.5, appear in advanced (6), comprehensive (8), klf1 (1) and epinformerseq_testing (1).
- **Evidence:** Scanning the committed quickstart finds 10 warn/error lines; a fresh run finds 0.
- **Fix:** Pre-download the conservation tracks, re-execute on GPU with current backgrounds, and commit. Add an output scrubber that replaces `$HOME` with `~` to the regeneration step.

**P2-23 · new since v0.7.5 · `examples/notebooks/README.md` misstates the committed outputs and the inventory** *(notebooks-determinism)*

- **File:** `examples/notebooks/README.md`, lines 8, 25 and 53.
- **Problem:**
  - Line 8 says 16 cells across three notebooks lack results because they need a second oracle env or coolbox, and that they fill in when run. That is false: a full fresh run leaves the same 16 cells empty (advanced 13, single 2, comprehensive 1), because they are imports, definitions and assignments.
  - Line 53 says "All 20 shipped notebooks". There are 21: 7 in `examples/notebooks/` and 14 walkthroughs.
  - `cherimoya_gpn_tutorial.ipynb` (058eceb) is missing from both tables.
  - Line 25 lists six oracles for `advanced`, which loads only Enformer, ChromBPNet and LegNet. `README.md:771` already has this right.
- **Evidence:** The fresh empty-output cells are identical to the committed ones, and `find examples -name '*.ipynb'` counts 21.
- **Fix:** Rewrite or delete the paragraph, change 20 to 21, add the tutorial row, and correct the oracle list.

**P2-24 · Notebook narrative counts are stale, and the Borzoi CAGE search silently returns nothing** *(notebooks-determinism)*

- **Files:** `examples/notebooks/comprehensive_oracle_showcase.ipynb` (markdown cell 1, cell 9), `cherimoya_quickstart` (markdown cell 0) and `advanced_multi_oracle_analysis` (markdown cell 70).
- **Problem:**
  - The markdown cell 1 table gives "ChromBPNet 786 (42 ChromBPNet ATAC/DNASE + 744 BPNet/CHIP)", where the count is 753 (9 + 744).
  - The same table gives "Sei 21,907 (… sequence classes)", where the count is 21,947 (21,907 + 40).
  - Cell 9 searches Borzoi for `CAGE:K562`, which matches 0 tracks, because Borzoi descriptions read "CAGE:chronic myelogenous leukemia cell line:K562". The Borzoi section therefore drops the CAGE track its comment promises.
  - The cherimoya notebook's markdown cell 0 says "ChromBPNet ships 42 ATAC/DNase models"; the README says 9 human.
  - `advanced`'s markdown cell 70 attributes the "54x–1307x … inverted its sign in 5 of 6" measurement to `single_oracle_quickstart`. The numbers come from comprehensive cell 35.
- **Evidence:**
  - Fresh cell 9 prints "Borzoi CAGE:K562 tracks: 0" and "Using tracks: [ENCFF899YDP]".
  - The search `CAGE:.*K562` finds 4 tracks (CNhs11250± and CNhs12336±).
  - Comprehensive cell 35 gives dilutions of 130/585/206/1,307/54/577x, with the window-mean sign inverted in 5 of 6.
- **Fix:** Use `CAGE:.*K562`, update the counts, and point `advanced`'s markdown cell 70 at `comprehensive_oracle_showcase`.

#### Docs and counts

**P2-25 · partly new since v0.7.5 · About 25 stale per-oracle counts remain in notebooks, scripts/README, code comments and tests** *(cdf-consistency-hygiene; one item from mcp-reports)*

- **File:** `examples/notebooks/comprehensive_oracle_showcase.ipynb` and others, in five categories.
- **Problem:** The README and the four test-guarded docs are clean. The stale counts are elsewhere:
  - **(a) AlphaGenome "5,731 tracks", 9 instances:**
    - comprehensive cells 1, 24 and 50, plus two cell-25 outputs;
    - klf1 cells 18–19;
    - epinformerseq_testing cell 17;
    - `tests/test_alphagenome_backends_equivalence.py:25`.
  - **(b) "6 oracles", 5 instances:**
    - advanced cell 0;
    - comprehensive cells 0, 1 and 58;
    - `examples/walkthroughs/variant_analysis/README.md:120`.
  - **(c) Sei "40" as the number of tracks with backgrounds, 6 instances:**
    - comments in `_igv_report.py:1154`, `causal.py:1225` and `multi_oracle_report.py:444`;
    - `tests/test_display_scale_is_measured_not_declared.py:495`, which says "2,253 shipped tracks are signed" (now 24,160);
    - `scripts/README.md:41`, "Sei | 40 | 2.9 MB" (now 21,947 tracks, ~1.5 GB);
    - `scripts/verify_rebuilt_backgrounds.py:39`.
  - **(d) ChromBPNet "786" as a track count** in the comprehensive table. The 786 in `README.md:539` counts weight files and is correct.
  - **(e) Report-corpus counts.** There are 21 committed reports, 20 of them with IGV.
    - "19-report corpus": `docs/THIRD_PARTY.md:47`, `tests/browser_harness.py:197`, and `tests/test_browser_outage_guard.py:4` and `:93`. The last two were added in #252.
    - `tests/test_committed_reports_render_in_a_browser.py`'s floor `expected = 19` and its docstring "18 committed IGV reports".
    - `tests/test_error_recovery.py:344`, "~1.3 MB".
- **Evidence:**
  - Logs: `s10_categories.log` and `s10_rawnums.log`.
  - The repo's own claim-finder, run over every .md and .ipynb, finds 21 mismatches, 8 of them in CHANGELOG.
  - `test_documented_track_counts.py`: 6 passed.
  - An earlier estimate of ~250 stale instances could not be reproduced. A raw grep for 5,731, 786 and 21,907 returns 235 hits: 108 in audits/ and 22 in CHANGELOG, which are historical by design. Most of the rest are correct as written or false positives.
- **Fix:**
  - Fix the stale text:
    - the comprehensive table and prose;
    - the three code comments;
    - the scripts/README row and `verify_rebuilt_backgrounds.py:39`;
    - the test docstrings and the walkthrough sentence.
  - Refresh the three notebooks' outputs at their next GPU run.
  - Extend `LIVE_DOCS` in `tests/test_documented_track_counts.py` to cover `examples/**/*.md`, `scripts/README.md` and notebook markdown cells.

**P2-26 · Doc drift: Enformer's TensorFlow version, and a plan doc that says "not yet started" for a feature that ships** *(cdf-consistency-hygiene)*

- **Files:** `environments/README.md:106` and `docs/plans/chrombpnet-hf-slim-mirror.md:3`.
- **Problem:**
  - `environments/README` says Enformer uses TensorFlow 2.14. `chorus-enformer.yml` pins `tensorflow==2.13.*` (installed: 2.13.1), with a comment that 2.14.1 breaks.
  - The plan doc reads "Status: planning, not yet started. Branch: TBD", but the slim mirror ships (`README.md:539`, `chorus-chrombpnet.yml`).
- **Evidence:** The installed version in `chorus-enformer`.
- **Fix:** Change the README to TensorFlow 2.13, and mark the plan done or archive it.

#### Test harness

**P2-27 · The determinism gate overwrites `CUDA_VISIBLE_DEVICES`, and `test_release_gates` hardcodes `--gpu 0`** *(integration-suite)*

- **Files:** `scripts/gate_end_to_end_determinism.py`, lines 67–68, and `tests/test_release_gates.py:145`.
- **Problem:**
  - `_child()` sets `os.environ['CUDA_VISIBLE_DEVICES'] = gpu`, where `--gpu` defaults to '1'. The test then passes `'--gpu', '0'`.
  - So the gate always lands on physical GPU 0 (or 1), whatever GPUs the caller was granted.
  - This is the bug class CLAUDE.md says was fixed in the builders.
  - It dates from 457a3a2 (2026-08-04).
- **Evidence:** The lines quoted above. It was harmless in this audit only because GPU 0 was assigned to that agent.
- **Fix:** Treat `--gpu` as an ordinal into the inherited mask, as `chorus/core/platform.py` does for `cuda:N`, or set it only when the variable is unset. Have the test pass no `--gpu`.

**P2-28 · Five integration tests can never execute on this host, because they need torch or cherimoya in the pytest interpreter** *(integration-suite)*

- **Files:** `tests/test_track_geometry_is_self_consistent.py:105`, `tests/test_cherimoya_ensemble.py:217` and `tests/test_cherimoya_integration.py:423`.
- **Problem:** pytest exists only in the base env, and torch and cherimoya only in per-oracle envs that lack pytest. Three tests therefore always skip:
  - `test_legnet_resolution_times_length_equals_its_interval` (3 params);
  - the ensemble dispatch test, including the `use_environment=True` half its docstring says it guards;
  - `test_geometry_guard_rejects_a_mismatched_checkpoint`.

  They date from 4b9d33d (2026-08-10).
- **Evidence:** A per-env probe found pytest 9.1.1 and no torch in `chorus`. `chorus-legnet`, `-cherimoya`, `-sei`, `-borzoi`, `-alphagenome_pt` and `-epinformerseq` have torch and no pytest.
- **Fix:** Run the in-process halves as a subprocess through the oracle env's python, as `test_predict_matches_direct_window_scoring_in_process` already does. Drop the torch gate from the `use_environment=True` half. Alternatively, record them in the checklist as expected skips.

**P2-29 · Fast-suite MCP normalizer tests reach Hugging Face and write lock files into the user's data dir** *(integration-suite)*

- **File:** `tests/test_mcp.py`, lines 222–235 and about 274–295.
- **Problem:**
  - `test_normalizer_none_when_no_backgrounds` and `test_normalizer_cleaned_on_unload` call `_auto_load_normalizer` with fake oracle names.
  - That reaches `get_pertrack_normalizer` with the default cache dir.
  - The supposedly offline fast suite therefore makes HF download attempts and leaves stray lock files in the real `CHORUS_BACKGROUNDS_DIR`.
  - Both tests pass either way.
  - The names date from 64505f0 (2026-04-12).
- **Evidence:** `/data/chorus_data/backgrounds/.cache/huggingface/download/{nonexistent_oracle_xyz,myoracle}_pertrack.npz.lock` were touched at 22:33:49 and 22:33:50.
- **Fix:** Monkeypatch `chorus.analysis.normalization.CHORUS_BACKGROUNDS_DIR`, or the download function, to `tmp_path` or a raising stub.

#### Supply chain and licence

**P2-30 · new since v0.7.5 (the stale description; the missing notice predates it) · `THIRD_PARTY.md` still describes the igv.js 3.1.1 bundle, and igv.js's own MIT notice ships nowhere** *(cdf-consistency-hygiene)*

- **File:** `docs/THIRD_PARTY.md`
- **Problem:** Since 3b59ca9 (#246), the vendored `igv.min.js` is 3.8.4. It is byte-identical to upstream (sha256 `91bda9aa8268cbd9642713bad4a48e9089cb8cfab9b5b4aaf7e648a3a759a6fc`). `THIRD_PARTY.md` is out of date in three ways:
  - **Stale facts.** It gives the bundle's size as "1.3 MB" (now 1,501,402 bytes) and lists "DOMPurify 3.2.1" (now 3.4.11).
  - **Incomplete component table.** Its "bundled inside igv.min.js" table lists only DOMPurify and pako. Both the old and new bundles also contain:
    - vanilla-picker (ISC);
    - Canvas2Svg (MIT);
    - code derived from GMOD twobit-js;
    - a cram bundle whose banner points at a `cram-bundle.js.LICENSE.txt` that is not shipped.
  - **No igv.js notice.** Neither upstream nor the repo ships igv.js's own MIT copyright and permission notice; `THIRD_PARTY.md` only links to it.
- **Evidence:** A banner scan of the current bundle against the v0.7.5 one.
- **Fix:**
  - Update the size and the DOMPurify version, and complete the banner table.
  - Ship igv.js's LICENSE as `chorus/analysis/static/igv.min.js.LICENSE.txt`, and add a `*.txt` pattern to `package_data` so it is packaged.
  - Record the version and sha256 next to `_IGV_CDN`, with a test.

**P2-31 · Vendored Sei model code lacks its upstream BSD-3 notice, THIRD_PARTY says chorus vendors no model code, and README and THIRD_PARTY cite oracles differently** *(cdf-consistency-hygiene)*

- **File:** `chorus/oracles/sei_source/sei.py`
- **Problem:**
  - **Missing notice.** `sei_source/sei.py` ("Sei architecture", with the patsy-style `bs()` spline helper) appears to be the upstream FunctionLab sei-framework model file, and it carries no copyright or licence notice. `THIRD_PARTY` lists Sei as BSD-3-Clause, and BSD-3 requires source redistributions to keep the notice. That the file is an upstream copy was not verified by diff, because fetching upstream source is not a request chorus makes.
  - **Inaccurate claim.** `THIRD_PARTY`'s "Chorus does not modify the upstream model code beyond the adapter layer" is wrong: `sei_source/sei.py` and `legnet_source/model.py` are vendored model code.
  - **Conflicting citations.**
    - `README.md:1485` credits EPInformer-seq to "Lin et al., Nature Communications, 2026", while `THIRD_PARTY` calls it first-party.
    - README cites ChromBPNet as "Agarwal et al., 2021"; `THIRD_PARTY` cites "Pampari et al., bioRxiv 2024".
- **Evidence:** `grep -rli 'copyright|license'` over the oracle `_source` dirs matches only two alphagenome templates. `sei.py` was added in e9d42fa (2025-12-11).
- **Fix:**
  - Add the upstream copyright line and BSD-3 text to `sei.py`, and an MIT header to `legnet_source/model.py` if it was copied.
  - Correct the `THIRD_PARTY` sentence.
  - Reconcile the two citation lists.

**P2-32 · Model-critical deps are unpinned in the per-oracle env files, and weights and UCSC bigwigs are fetched without pinned revisions or checksums** *(cdf-consistency-hygiene)*

- **File:** `environments/chorus-borzoi.yml` and the other per-oracle files.
- **Problem:** The root `environment.yml` passes: 25 lower-bound ranges, `python=3.10`, and coolbox at a full sha. The per-oracle files do not:
  - **Bare deps:** 49 deps across 8 files carry no version: alphagenome 18, alphagenome_pt 12, borzoi 5, sei 4, epinformerseq 3, legnet 3, base 3, cherimoya 1. Model-critical ones include:
    - borzoi-pytorch (working env: 0.5.1);
    - AlphaGenome's JAX stack: jax[cpu], dm-haiku, optax, chex, orbax-checkpoint, jaxtyping and einshape (working env: jax 0.10.2, dm-haiku 0.0.17, optax 0.2.8, chex 0.1.92, orbax-checkpoint 0.12.1, jaxtyping 0.3.11);
    - safetensors and huggingface_hub in `chorus-alphagenome_pt.yml`;
    - numba in `chorus-sei.yml`.
  - **Lower bounds only:** most other deps; for example `tensorflow>=2.15.0` in the JAX env resolved to 2.21.0.
  - **No revision pinning:** 19 of 22 `hf_hub_download`/`snapshot_download` call sites pass no `revision=`, 13 of them outside templates. They include every oracle's weights, the `igv.min.js` HF fallback (`_igv_report.py:363`) and the DHS vocabulary (`utils/annotations.py:783`).
  - **No checksums:** the UCSC phyloP/phastCons bigwigs are fetched by URL with only a chr1-length assembly check.
  - **Not fixed by #253:** it pinned the git deps and alphagenome; the bare names above remain on current `main`.
  - **pyBigWig:** needed by #246's conservation tracks and absent from `environment.yml`. It does reach a fresh install, because `setup.py` reads `requirements.txt` into `install_requires`, so the gap is on the conda side only.
- **Evidence:** `s17_pins.log` and `s17_installed_versions.log`. The pyBigWig result was checked for this report.
- **Fix:**
  - Pin the model-critical bare deps to the working-env versions, or add upper bounds.
  - Add a test that fails on bare names in `environments/*.yml`, with an allowlist.
  - Decide a policy for weight revisions.
  - Record sha256 values for the UCSC bigwigs.
  - Optionally, add `pybigwig>=0.3.18` to `environment.yml`.

#### The checklist itself

**P2-33 · partly new since v0.7.5 · `AUDIT_CHECKLIST.md` is stale in 15 sections, and the CI workflow comment carries old counts** *(all five agents)*

- **File:** `audits/AUDIT_CHECKLIST.md` and `.github/workflows/tests.yml:4`.
- **Problem:** By section:
  - **§1:** line 19 says there is no download-dir override; `CHORUS_DATA_DIR`, the marker and `chorus config data-dir` exist, and `CHORUS_BACKGROUNDS_REVISION` is undocumented. Line 20 says "8 NPZs ship"; there are 9, including cherimoya_ensemble.
  - **§2 and §9:** lines 25–26 and 249 require `huggingface-cli login`, which is dead (P1-18).
  - **§4:**
    - Line 84 and the script cover 8 oracles; 9 NPZs ship.
    - The script checks only the first 10 rows.
    - Line 93 says "sei 40"; the count is 21,947.
    - Line 97's "a few small negative entries" is wrong for ChromBPNet (P2-14).
  - **§5:** line 106 says the check "does still warn"; since #128 it raises `ReferenceAlleleMismatchError` by default.
  - **§6:**
    - Lines 113–115 say `chorus setup` does not register the kernelspec; it has since #214 (v0.7.3).
    - Line 126 gives 6 and 13 notebooks; there are 7 and 14.
    - Nothing warns about the ~71 GB conservation download, or that nbconvert's kernel cwd is the notebook's directory.
  - **§7:**
    - Line 166's "lnFC on RNA-seq/CAGE" is wrong: CAGE carries log2FC.
    - Line 173 says "opens all 19"; there are 21.
  - **§8:** line 179 says "Exactly 24 tools"; there are 27. Line 236 says "46 tests"; 48 are collected.
  - **§10:** line 262 says "AlphaGenome 5,731 model tracks"; that is metadata rows, including 563 padding, and there are 5,168 real tracks. It says "Sei 21,907 total but 40 CDF-backed"; all 21,947 are CDF-backed.
  - **§11:**
    - Lines 289–291 say "1,463 selected of 1,535 … 1,501 passed / 33 skipped / 1 xfailed"; there are now 2,355 selected of 2,523, with 168 integration tests.
    - The gate bullet (lines 296–298) omits the host prerequisites: the genome path, Chromium and GPU confinement.
    - `tests.yml:4` says "1,451 selected of 1,525 collected, the other 74".
  - **§13:** line 324 does not name the execution path. "Verified bitwise … enformer" holds only for `use_environment=False`.
  - **§14:** line 333 cites `sequence.py:135` and `base.py:460`, now `:140` and `:539`. Line 335 assumes indels are unsupported; they have been accepted since v0.5.5.
  - **§15:** line 343 cites `base.py:591`, now `:682`.
  - **§17:** it checks only the root `environment.yml`.
  - **§18:** it asks for `igv.min.js` to carry an upstream licence header, but upstream has none.
- **Evidence:** Each agent measured the current values directly; they are listed in the Problem entries above.
- **Fix:**
  - **Counts:** replace with measured values, or with commands that print them.
  - **§4 script:** loop over `glob('<data-dir>/backgrounds/*_pertrack.npz')` and check every row.
  - **§17:** cover every env file and require 40-hex git refs.
  - **§18:** reword to "byte-identical to upstream (record the sha256), and igv.js's LICENSE ships alongside".
  - **§11:** list the release-host prerequisites under the gate bullet, and require a review of its skip list.
  - **§14:** reframe the indel check as "indel effects equal the `get_centered_window` pair", and extend the chromosome-end check to `predict_variant_effect`.
  - **§13:** name the execution path.
  - **§6:** add a conservation pre-download step and "execute a copy outside the repo".

## Not run, and why

| § | check (checklist severity) | why |
|---|---|---|
| 1 | `mamba env create --dry-run` (P0), `pip install -e .` from a fresh clone (P0), `chorus setup --oracle X` (P0) and its idempotency (P2) | Installs and env creation were forbidden. All 11 env YAMLs parse, and `pip check` in the base env is clean. |
| 2 | `HF_TOKEN` env-var load (P0) | It would mean copying the stored credential into `HF_TOKEN`, and no separate token was provided. By code reading the path is equivalent, because `get_token()` reads `HF_TOKEN` first. |
| 3 | The whole section (P0 items) | No agent was assigned it, and the compilation step had no GPU. |
| 5 | Direct-path predict for every oracle (P0) | Only partly assessed. It ran live for LegNet, ChromBPNet, Enformer and the AlphaGenome load, but not for Sei or alphagenome_pt. No direct-path predict test was found for chrombpnet, sei or alphagenome_pt. |
| 6 | `comprehensive_oracle_showcase` end to end (P0) | It needs about 585 s, more than the 540 s cap. Cells 56–57 are unverified. |
| 6 | `single_oracle_quickstart` on a cold cache (P0) | The ~71 GB download does not fit in 540 s (P1-2). It passed on a warm cache. |
| 6 | `epinformerseq_testing`, `klf1_validated_enhancer_profiles`, `cherimoya_gpn_tutorial` and the 14 walkthrough notebooks | They are outside the checklist's minimum of three, so they were not executed. All 21 declare kernel `chorus`. |
| 7 | Browser render of all 21 reports (P0), and with it §15's request inventory (P1) | There is no Chromium on the host, and `playwright install chromium` was forbidden. 47 of the 48 browser tests and the 2 bundling tests skipped. CI's 2-report smoke ran and failed (P1-13). |
| 8b | Drive the tools in plain English | It needs an interactive Claude Code session with the chorus MCP server registered. |
| 11 | ChromBPNet fresh ENCODE model download | It is one test and cannot be split. It timed out at 540 s with 717,225,984 bytes downloaded (~1.3 MB/s, while other agents were pulling conservation bigwigs). |
| 11 | 4 torch-gated tests and 1 cherimoya-forge test | No env has both pytest and torch (P2-28). |
| 11 | 1 AlphaGenome effect-shard test | The build intermediates are not on disk. |
| 12 | The whole section | The regeneration scripts write into `examples/`. Notebook codegen and `rerender_examples.py --check` passed in a scratch clone. |
| 13 | Drift across machines (P2) | Only one host was available. |
| 14 | Sei insertions | The pattern behind P1-5 likely applies to Sei too, but it was not measured. |
| 17 | `pip-audit` on the base env (P1) | It is not installed, and installing it was forbidden. |
| 19 | Release bookkeeping | The maintainer is doing it in parallel. |

## Notes on the agents' results

- **Corrections.** The compilation step made three:
  - **igv bundle commit.** The `igv.min.js` swap is 3b59ca9 (#246, 2026-08-19), not 3d0b6b1 as `cdf-consistency-hygiene` wrote. 3d0b6b1 (2026-08-28) is the commit that regenerated SORT1_chrombpnet and re-executed the quickstart.
  - **Genome-path skip count.** `integration-suite`'s summary says 24 integration tests skip on the genome path. Its per-file table gives 23 (6 smoke + 11 Cherimoya + 1 AlphaGenome equivalence + 1 ChromBPNet download + 1 strict_ref + 3 region strata), and 23 is the figure that makes its 77 total add up. This report uses 23.
  - **pyBigWig.** It is missing from `environment.yml`, but `setup.py` reads `requirements.txt` into `install_requires`, so the documented `pip install -e .` step installs it.
- **Re-checked for this report, read-only:**
  - the v0.7.5 and HEAD versions of SORT1_chrombpnet's `example_output.md` line 24;
  - the conservation README's lines 42–43 and 59;
  - the last change to `docs/THIRD_PARTY.md` (6c9d5c8, inside v0.7.5);
  - that 3b59ca9, 3d0b6b1 and 01aae63 all came in through #246's branch;
  - #253's diff, and its extended pin test (5 passed on an extracted copy of `eab109c`);
  - the clean tree and the intact marker at the end.
- **Merged duplicates:**

  | finding | merged from |
  |---|---|
  | P1-14, genome paths | 2 agents |
  | P1-15, data-dir marker | 2 agents |
  | P2-1, AnnotationStore | 2 agents |
  | P2-2, `alphagenome_tracks.json` | 4 agents |
  | P2-4, Sei assay types | 2 agents |
  | P2-15, mixed igv corpus | 2 agents |
  | P2-22, committed notebook outputs | 2 agents |
  | P2-33, checklist staleness | all 5 agents |

- **Severity where agents differed.** The AlphaGenome "5731" that the comprehensive notebook prints is P1-11, because it fails §6's P1 printed-counts check. The same figure elsewhere stays in P2-25.
- **Failed checks filed below their checklist rating.** Three checks rated P1 in the checklist were filed as P2 by their agents:
  - §8's error-payload shape: no traceback leaks.
  - §10's canonical numbers: the README and the guarded docs are clean.
  - §18's igv licence header: upstream has none, and the notice gap predates v0.7.5.

  None would block at P1 either. Their new-since-v0.7.5 parts are documentation drift.
- **A pattern across agents.** Every artefact regenerated in #246's branch shows signs of a generating host with off-pin backgrounds:
  - wrong ChromBPNet percentiles in both walkthroughs, from 3d0b6b1 and 01aae63 (P1-1);
  - 10 "superseded" Enformer-background warnings, plus paths under the legacy `~/.chorus/backgrounds`, in 3d0b6b1's quickstart outputs (P2-22).

  Regenerate the post-v0.7.5 artefacts together, on a host whose NPZs match `backgrounds-2026-08-16-sei-full`.
- **Observed, not filed:**
  - `/data/chorus_data/huggingface/` still holds `token` and `stored_tokens` (mode 600, dated Aug 1). They are leftovers of the old `HF_HOME` behaviour, and CLAUDE.md says credentials do not belong in the shared data dir. Only their presence was checked.
  - `~/.cache/huggingface/token` and an LDlink key in `~/.chorus/config.toml` are present. No token environment variables were set.
- **Evidence.** It is in the ephemeral scratch tree `<scratch>/audit076/`:

  | subdirectory | contents |
  |---|---|
  | `s1_2_5_9_14_15/` | install-api-errors |
  | `s4_10_16_17_18/` | cdf-consistency-hygiene, including `cdf_check.txt` |
  | `integration-suite/` | including `results_per_file.tsv` |
  | `notebooks-determinism/` | fresh notebooks and `determinism/*_result.json` |
  | `s7s8/` | mcp-reports: `s7_results.json`, `s8_e2e.json` and the fresh MCP report |
  | `report/` | this compilation's `eab109c` check |

  Archive whatever is worth keeping, as the checklist appendix describes.

## Release blockers

The release rule: a P0 or P1 introduced since v0.7.5 blocks unless it is fixed. There are no P0s and three such P1s:

1. **P1-1.** Two ChromBPNet walkthroughs ship percentiles that the pinned null does not reproduce, and the conservation walkthrough's headline claim is false. Regenerate both on a host whose `chrombpnet_pertrack.npz` matches the pin, and rewrite the README.
2. **P1-2.** "Run All" on `single_oracle_quickstart.ipynb` downloads ~71 GB on a cold cache. Gate the conservation cell.
3. **P1-3.** GPN-Star, phyloP and phastCons are credited nowhere reachable from the README. Add the THIRD_PARTY section.

Two gates are not met (§11 and §19):

4. **CI is red on `4258b6f`** (P1-13). `main` is now `eab109c`, whose CI is green but which this audit tested only through #253's pin test. Tag a sha with green CI, and measure any numbers quoted in the release notes against that exact sha.
5. **Neither suite is green as written on this host.** The fast suite has 1 failure (P1-15). The integration suite has 2 failures and 23 genome-path skips (P1-14); with the genome reachable it gives 112 passed / 0 failed / 55 skipped. Either fix the harness, or record a genome-reachable run against the final sha and review its 55 skips.

P0 checks this audit could not run. Run them on the release host before tagging:

6. **§7 browser render** of all 21 reports, which also covers §15's request inventory. It needs Playwright Chromium.
7. **§1 install:** a fresh `chorus setup`, an env solve and `pip install -e .`.
8. **§2 HF_TOKEN:** an AlphaGenome load with `HF_TOKEN` set.
9. **§3 devices:** device detection in every env.
10. **§5 direct path:** direct-path predict for Sei and alphagenome_pt.
11. **§6 notebook:** a full `comprehensive_oracle_showcase` run, including cells 56–57.

Pre-existing P1s do not block under that rule. The checklist rates P1 as "fix before release", so either fix them or list them under the release's Known limitations, as §19 requires. The ones users meet in normal use are P1-4 (variants near a chromosome's 3' end fail), P1-5 (LegNet insertions are mis-scored) and P1-6 (the LDlink token reaches the logs). P1-17 is already fixed on `main`.

## Outcome: what was fixed before tagging v0.8.0 (added 2026-11-16)

The maintainer decided to fix the three blockers, the two gates and the smaller P1s that users
meet. The conservation walkthrough (P1-1) is pulled from the release, and every other deferred
P1 is listed under *Known limitations* in the `[0.8.0]` CHANGELOG section. Everything below
landed in the release PR, on top of `eab109c` (main after #253, which fixed P1-17).

| Finding | Resolution |
|---|---|
| P1-1 ChromBPNet walkthrough percentiles | `variant_analysis/SORT1_chrombpnet` regenerated against the pinned null; it now reads 0.9995 / 0.906, matching v0.7.5. The conservation walkthrough is pulled until it is regenerated. New guard: `tests/test_committed_percentiles_match_the_pinned_null.py` |
| P1-2 Quickstart ~71 GB download | Conservation cell gated behind `RUN_CONSERVATION = False`; `tests/test_quickstart_conservation_cell_is_gated.py` |
| P1-3 Conservation sources uncredited | `docs/THIRD_PARTY.md` *Annotation and conservation data* section, linked from the README; igv.js entry updated to 3.8.4 |
| P1-6 LDlink token in logs | Redacted raises `from None`; `_safe_tool` scrubs tracebacks; urllib3 held at INFO for the call |
| P1-13 CI red on a UCSC outage | Browser meta-test consults the outage guard |
| P1-14 Repo-relative hg38 in tests | `tests/reference_genome.py`, resolved through the data dir, at all listed sites plus the determinism gate |
| P1-15 Data-dir test fails and rewrites the marker | Runs against a throwaway install root; never touches the real marker |
| P1-18 `huggingface-cli login` | `hf auth login` everywhere it is user-facing, with a guard |
| P1-8 Causal null wording | Footnote, README and NORMALIZATION_GUIDE corrected, with a guard |
| (new) rsID input never resolved | `v.pos` → `v.position` in `_resolve_rsid_to_position`; `tests/test_rsid_position_resolves.py` |
| (new) `regenerate_examples.py` silent failure | Resolves hg38 through the data dir, and exits 1 when an example fails |

Final verification on the release head is recorded in the PR.
