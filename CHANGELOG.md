# Changelog

All notable changes to Chorus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [0.7.3] — 2026-08-13

**No percentile changes.** No background null, region set, retention rule or default fold moved,
so effect and activity percentiles are directly comparable with 0.7.2. What changed is enforcement,
report rendering and two real defects — see the banner on each entry.

**Background artefacts.** Unchanged: this release is verified against dataset revision
`backgrounds-2026-08-12-cherimoya-fold0`, the same pair 0.7.2 shipped with, and
`_HF_REVISION` still pins it. Nothing was rebuilt or re-uploaded.

### Fixed
- **⚠ `device='cuda:N'` no longer takes a GPU you were not given — on **either** load path.** Six oracle *templates* (the `use_environment=True` subprocess path) set `CUDA_VISIBLE_DEVICES` straight from the ordinal, and so did `ChromBPNetOracle._load_direct` and Enformer's in-process load — which is the `create_oracle` **default**. Under a scheduler granting `CUDA_VISIBLE_DEVICES=4,5`, `cuda:1` means *the second GPU I was granted* (physical 5); overwriting the mask sent the process to **physical GPU 1, somebody else's job**. Not hypothetical: GPU 4 was another tenant's throughout the audit that found it.

  Worse on the PyTorch side, where it was a live crash rather than a hazard: Cherimoya's templates masked to `N` and then handed the same `cuda:N` string to torch, which indexes *within* the now-one-device visible set, so every ordinal but 0 died with `CUDA error: invalid device ordinal`. A documented parameter was simply broken.

  The two frameworks need different handling and that is the point: torch resolves the ordinal itself, so the assignment is gone; TensorFlow selects *through* the mask, so `cuda:N` is remapped by one shared `chorus.core.platform.resolve_visible_ordinal`, which raises naming what `N` indexes when it is out of range. Verified under `CUDA_VISIBLE_DEVICES=2,3`: `cuda:0` → physical 2, `cuda:1` → physical 3.

  The direct path was missed by the first fix and by its own guard, which enumerated only `*_source/templates/*template.py` — the search was scoped to what the checklist item named rather than to the property. The guard now covers `chorus/oracles/*.py` too and is mutation-tested. The eight builders under `scripts/` were swept in the same pass and were already correct. Both halves are audit findings F1 and G1, in `audits/2026-08-12_post_v0.7.2_audit.md` and `audits/2026-08-13_v0.7.3_release_audit.md`.

- **`extract_sequence_with_padding` no longer returns a short sequence near a chromosome end.** Its wide-interval branch handed an out-of-bounds end straight to pysam, which clamps silently, and the metadata it returned hardcoded `leftN/rightN = 0` — reporting that no padding was needed. Measured on chr1: **2,114 bp requested 40 bp from the end returned 40 bp.** The narrow-interval branch always padded correctly, so the two disagreed about the same question; both are now pinned to agree. Sei reaches this path with `total_length=SEI_WINDOW`, so a short one-hot could reach the model. Audit F2.

- **"Reference FASTA required" now says what to do about it.** Ten sites across nine oracles raised it in two wordings that differed only by punctuation, and neither named the `reference_fasta` kwarg or `chorus genome download hg38`. One helper now carries both. Audit F3.

- **The Cell Type column no longer repeats the track label.** `CHIP:CEBPB:IMR-90` beside a column reading `IMR-90` said it twice in **13 of 20 committed reports**, up to 22 rows in one. Blanked only on an exact case-insensitive match of the label's final colon-delimited component, so `DNASE:fibroblast of lung` beside the same string is suppressed while `CHIP:CEBPB:HepG2_treated` beside `HepG2` is not, and Enformer's opaque ids keep the column that carries all their meaning. All reports regenerated; no score moved. Audit F7.

- **Attribution gaps:** `igv.min.js` bundles **DOMPurify** (Cure53) and **pako**, both shipped in the wheel and credited nowhere; both are now in `docs/THIRD_PARTY.md` with the licence notices that survive inside the bundle. Borzoi was described as 7,612 tracks in a docstring where it is **7,611**. And `THIRD_PARTY.md` now states explicitly that **EPInformer-seq is first-party** — no vendored upstream code, weights from this project's own HuggingFace repo — because "7 of 8 oracles attributed" reads like a missing attribution otherwise. Audit F4–F6.

- **Reports no longer resolve their genome through igv.org's hosted registry ([#139](https://github.com/pinellolab/chorus/issues/139)).** `genome: "hg38"` reads like a setting but is a *registry lookup*: igv.js resolves the string against its catalogue and follows the result, so every shipped report opened **six remote resources across two hosts** — the catalogue, chromosome aliases, `hg38.chrom.sizes`, `cytoBandIdeo.txt.gz`, `ncbiRefSeq.txt.gz` and ranged reads of `hg38.2bit`. Fourteen requests, and the catalogue fetch is **fatal**: with the network cut the panel did not degrade, it never appeared. The docstring claiming that inlining `igv.min.js` made reports "viewable offline, on air-gapped hosts" was therefore false in the strongest available sense.

  Chromosome lengths and the ideogram now come from one vendored 6.1 kB table (UCSC's `cytoBandIdeo`, primary chromosomes only — the per-chromosome maximum band end *is* the chromosome length, verified against the FASTA index at chr1 = 248,956,422 both ways), and the gene track from chorus's own GENCODE v48 annotation scoped to the drawn window, which also means the panel's genes agree with the gene names printed beside them. Measured on one report:

  | | requests | hosts | load to paint | with no network |
  |---|---|---|---|---|
  | before | 14 | 2 | 9.6 s | **fatal** (`genomes.json`) |
  | after | 9 | 1 | **2.2 s** | fatal (`hg38.2bit`) |
  | after, sequence self-hosted same-origin | **0** | 0 | **0.8 s** | **works** |

  Cost: **+46.7 kB** per report, 2.8% of a 1.65 MiB one.

  **The reference sequence is the one thing that cannot be bundled, and that is igv.js's rule rather than a choice.** Every version requires a sequence source: omit it and 3.1.1 dies in `Ec.loadAll` on `undefined.startsWith`, while 3.8.5 dies on `url must be either a 'File', 'string', 'function', or 'Promise'`. A `data:` URI does not substitute either — igv decodes data URIs inline and treats them as a *non-indexed* FASTA, taking chromosome lengths from the body, so a stub declaring the real lengths in its index renders a perfect ideogram and ruler while every feature track silently draws nothing (3 of 5 canvases painted, against 5 of 5 with a real reference). hg38 is 3 GB.

  For a genuinely air-gapped site, `CHORUS_IGV_SEQUENCE_URL` points the report at a self-hosted copy, and it accepts a **FASTA** as well as a 2bit because every chorus install already downloads `hg38.fa`:

  ```bash
  # serve the report beside the genome — same origin, no CORS, no internet
  # `chorus config data-dir` prints a human-readable block, so resolve the path itself:
  GENOMES=$(python -c 'from chorus.core.globals import describe_layout; print(describe_layout()["genomes"])')
  cp report.html "$GENOMES/" && python -m http.server -d "$GENOMES" 8000
  ```

  Same-origin matters: a page opened from `file://` has origin `null` and cannot read a served FASTA, and a second port needs CORS headers — both measured, both fail with `Access to XMLHttpRequest ... has been blocked`.

  All three render paths (`_igv_report`, `multi_oracle_report`, `causal`) now share one config builder, since three copies of the same dict is precisely how the display-pooling defect shipped with only one of them fixed. Every committed report was regenerated; a test reads the artefacts rather than the generator, because regeneration is what makes the fix real.


### Changed
- **`docs/BACKGROUND_NULL_PROTOCOL.md` is current with the count-head extraction ([#186](https://github.com/pinellolab/chorus/pull/186)).** The protocol is the living document a new oracle's background build is written against, and §8 said nothing about which quantity a CDF must be built from — the distinction #125 exists to enforce. It now carries §8 Step 2b (use `chorus.core.count_head.expected_counts_profile` from **both** the oracle and the builder, because a CDF is only meaningful if it was built from the quantity `predict()` returns), the three count-head conventions side by side (`log1p` per track → `expm1`; `log1p` pooled → `exp(C) − n_tracks`; `log10` → `10**C`, which differ by 26× at log-count 2.5), a §7 equivalence-guard row, and a §9 decision-log entry.

- **CI renders reports on every PR now, a reduced set of them.** The full 19-report browser suite stays on the release host, but running none of it in CI is how a blank panel reaches `main` between audits — which is the gap that let a size ceiling stand in for a rendering check in the first place. `CHORUS_BROWSER_SMOKE=1` selects the two smallest IGV reports plus the panel-less table: 12 tests instead of 46, enough to catch the failures that hit every report at once.

  This needed `test_ci_runs_the_same_command_a_contributor_runs` narrowed, since it correctly refused the new job. Its property is about the run that stands for "the tests pass" — a whole-suite invocation must carry no selection of its own — so a single-file job with an explicit marker is now allowed while a widened one still fails. Mutation-tested three ways.

- **Six items in `audits/AUDIT_CHECKLIST.md` corrected**, because they produced false findings against a correct tree: the `<details>` count (the multi-oracle report legitimately has four), the CDN grep (it matches `googleapis` and `igv.org` *inside* the inlined igv.js, firing on all 19 reports — the measured request inventory replaces it), the P0 forbidding the pre-rename examples directory (its three live mentions are the test and the comment that record the rename, not paths anyone is sent to), `list_tracks` being an MCP tool rather than a Python API, the dependency-pin check, and which function §14's padding item refers to.

- **⚠️ `AlphaGenomeOracle(organism="mouse")` now raises instead of being silently ignored ([#124](https://github.com/pinellolab/chorus/issues/124)).** The parameter was accepted, assigned to `self.organism`, and read by nothing — the metadata loader hardcodes `Organism.HOMO_SAPIENS` and the PyTorch port passes `organism_index=0` — so the one oracle whose upstream API genuinely supports mouse had a switch that looked functional and returned human predictions under a mouse label. Same fix on `AlphaGenomePTOracle`. `organism="human"` (any capitalisation, plus `homo_sapiens`) is unchanged; anything else raises `NotImplementedError` naming what mouse would actually require. Of make-it-work / remove-it / raise, only raising is both honest and affordable: mouse needs an mm10 reference in the genome manager, an mm10 reference class for the background null (SCREEN publishes mm10 cCREs; the Meuleman DHS index has no mouse equivalent), and a background pass over ~4,300 further tracks.

- **Report size is no longer a proxy for report health.** The ceiling stands at 50 MiB — GitHub's advisory threshold, under its hard 100 MiB wall — and it now exists alongside a check that actually loads the file. Measured across the corpus, size barely predicts load time: **20× the bytes costs 1.3× the wait** (25.7 MiB → 11.3 s; 1.3 MiB → 8.8 s), because most of those seconds go on ~14 network round-trips for genome resources ([#139](https://github.com/pinellolab/chorus/issues/139)) rather than on parsing the payload. `CHORUS_WRITE_LARGE_HTML` is gone.

- **The BPNet-family count-head arithmetic now has one implementation ([#125](https://github.com/pinellolab/chorus/issues/125)).** Turning a profile head and a count head into a per-position expected-count track is four operations — centre the logits, softmax, invert the count head, scale — and it existed in five places at once. All three defects fixed on 2026-07-31 were two of those copies disagreeing: `exp` vs `expm1` across four call sites (+1 read, ~0.1% at a peak but up to **100%** at a quiet site, which is the regime the activity CDFs are built from); per-strand vs joint softmax (the two emitted tracks together claimed **2.00×** the predicted counts); and a count bias hardcoded `(N, 1)` that Keras broadcast silently (every log-count shifted by 0.5885, i.e. 1.80× low at a peak and 3.04× at a quiet site). None crashed; each produced a plausible number and each shipped.

  `chorus/core/count_head.py` is now the only implementation, used by `ChromBPNetOracle`'s three paths, Cherimoya's `scoring.py`, and the ChromBPNet background builder — the last of which had to be factored into a `profiles_from_heads()` function, because its arithmetic was inline in a TensorFlow-dependent routine and could therefore only ever be compared against the oracle by grepping both files for matching source text. That is exactly the check `exp`/`expm1` walked past four times.

  **No number moved, and that was verified rather than assumed.** The pre-extraction expressions are written out verbatim in the tests and compared with `array_equal`, not `allclose`, in both float32 and float64. End to end, regenerating the ChromBPNet and Cherimoya examples produced JSON and HTML differing from the committed files **only in the timestamp**, so no example needed re-committing and the shipped CDFs are untouched.

  The one thing that did drift was caught by that end-to-end run and by no unit test: the two copies disagreed about *precision* — Cherimoya cast to float64 before doing anything, ChromBPNet used whatever TensorFlow returned. The shared helper therefore preserves its caller's dtype, and Cherimoya promotes on the way in; without that, leaving `log_counts` in float32 moved the SORT1 example's `ref_value` from 603.3464052301788 to 603.3464123072064 — 1.2e-8, and with nothing behind it. Whether ChromBPNet *should* compute this in float64 is a real question and deliberately a separate one.

  **Two copies stay where they are, on purpose, with their equivalence pinned.** The torch expression in `build_backgrounds_cherimoya.py` runs on the accelerator inside the batch loop, where a numpy round-trip per batch would be charged to a multi-hour job — verified in `chorus-cherimoya` (torch 2.13.0+cu130) at a **1.25e-07** maximum relative difference. And **EPInformer-seq is not routed through here at all**: it scales by `10 ** log_count`, a different convention from a differently-trained model, and at a log-count of 2.5 the two differ by **26×** — so a tidy-up that unified them would be the same class of mistake as the three above, in the opposite direction. A test pins that difference rather than leaving it to be noticed.


### Added
- **The installation instructions were audited against the code and re-measured; the disk figure was
  wrong by more than 2× and is the reason to re-read them.** The prerequisite said **~38 GB free
  disk** while a default all-oracle install on Linux + CUDA measures **~85 GB** — so anyone who
  provisioned a 40 GB volume on that advice ran out of space partway through an unattended
  55–75-minute setup, with a half-built set of envs and no hint that 2× was needed. Two independent
  causes: the per-env row claimed "~3 GB each" when the smallest oracle env is 6.1 GB and the largest
  is 11 GB (every env carries its own CUDA payload — pip `nvidia_*` wheels in six of them, conda-side
  `libtorch_cuda`/`libcu*` in Borzoi, LegNet and Sei, and pip does not hardlink between envs); and the
  table had **no row at all** for the weights `chorus setup` prefetches by default, ~11 GB of which
  Sei alone is 6.5 GB because its 3.1 GB tarball is kept beside the 3.4 GB it extracts to. The
  breakdown is now itemised to 14 buckets and `tests/test_disk_claims_add_up.py` fails if the rows
  stop summing to the stated total or the prerequisite drops below it.

  Fixed in the same pass, each verified against the code rather than re-read:

  - **`chorus setup all` is not a command** — the subparser defines only flags, so it exits 2 with
    "unrecognized arguments". It was the documented prerequisite in
    `examples/walkthroughs/README.md`, and appeared in `chorus/utils/ld.py`'s user-facing hint and in
    `_setup_all.py`'s own failure banner.
  - **`chorus setup` needs a TTY.** The token resolves before anything is built and aborts if stdin
    is not a TTY, so `nohup chorus setup &` — the pattern the TLDR recommends — died instantly with
    zero progress. `HF_TOKEN` / `--hf-token` / `--no-weights` are now documented there.
  - **Notebooks needed a step that did not exist in the docs.** 16 of 19 shipped notebooks declare
    kernel name `chorus`, nothing in the package registers it (`grep -rn ipykernel chorus/` → 0), and
    the README claimed all three "work as soon as `chorus setup` finishes". Without
    `python -m ipykernel install --user --name chorus`, `nbconvert` raises `NoSuchKernel`.
  - **`claude mcp add chorus …` does not register globally** — it defaults to `--scope local`, so the
    documented "available in every project" recipe registered chorus for the clone directory only.
    Now `-s user`.
  - **`chorus cleanup --all` leaves the HuggingFace cache**, which is where most weights live (~20 GB),
    while the docs called it "Remove everything".
  - **`~/.chorus/backgrounds/` was wrong in every live place it appeared** — 32 lines across 30 files, including two CLI help strings and four
    `normalization.py` docstrings stating it as the default. The data directory has defaulted to the
    *installation tree* since 2026-08; `CHORUS_DATA_DIR` — the one switch that relocates all 85 GB —
    appeared in no user-facing doc at all, and now has its own section, including the legacy
    backgrounds-only rule that lets `backgrounds` resolve outside `data_dir`.
  - **`$(chorus config data-dir)` was used as a path** in this file's own air-gap recipe; the command
    prints a human-readable block.
  - **Stale by 28×**: three `_setup_prefetch.py` comments called the 2-model ChromBPNet default
    "~1.4 GB" where the slim mirror serves 25 MB each. The giveaway was that the same line put the
    786-model catalogue at ~1.5 GB.
  - `environments/chorus-base.yml` is documented as **vestigial** — no code reads it (the manager
    explicitly excludes it), it is not the subset the docs claimed, and it declares `name: chorus`,
    so installing it collides with the documented base env.
  - Three dead in-page doc anchors, now guarded by `tests/test_doc_links_resolve.py`.

- **The genome assembly is now asserted rather than assumed ([#124](https://github.com/pinellolab/chorus/issues/124)).** Chorus is human hg38 everywhere, and until now that held *by accident*: Enformer and Borzoi exclude their 1,643 and 2,608 mouse tracks because someone selected `enformer_human_targets.txt` / `borzoi_human_targets.txt`, AlphaGenome because `Organism.HOMO_SAPIENS` is hardcoded in its metadata loader. Those are file and literal choices, not assertions — nothing connected any of them to `genomes/hg38.fa`, which every builder opens, so nothing would have caught a future `*_mouse_targets.txt`. ChromBPNet, whose registry had no organism field at all, is where it actually went wrong: 33 mm10 models were scored against hg38 sequence using the hg38 DHS vocabulary, removed in [#121](https://github.com/pinellolab/chorus/issues/121).

  A hard failure rather than a warning, because there is no symptom to notice: mm10 `chr1:1,000,000` exists in hg38 too, so every coordinate resolves, every prediction returns, and every percentile lands in [0, 1]. The answer is simply about a different piece of DNA. Four mechanisms:

  - every oracle **declares** `training_genome` on its own class — deliberately not inherited, since `OracleBase` defaulting to `"hg38"` would be one more silent choice, and a test enumerates the subclasses to make sure a new oracle says;
  - all 8 builders **check** it against the FASTA they open, at all 10 open sites, via `require_reference_assembly` — at preflight, so a 14-hour build fails in its first second;
  - both stamp scripts **observe** the assembly from the reference's chromosome lengths instead of writing `"genome": "hg38"` as a literal, which had made that field a restatement of the stamper's own assumption;
  - the loader **refuses** an artefact declaring a genome chorus does not rank against, as `BackgroundGenomeMismatch`.

  Identity comes from chr1's length, which is provider-independent: UCSC, Ensembl and GENCODE disagree about chromosome naming, line width and scaffolds but agree on chromosome lengths, where the `fasta_sha256_prefix64mb` the artefacts already carry would reject a correct Ensembl GRCh38 as loudly as it rejects mm10. An *unrecognised* reference warns and proceeds — refusing it would break anyone on a legitimate custom build to enforce a lookup table's completeness — while a recognised, wrong one raises. A typo in the expected value (`"GRCh38"` for `"hg38"`) also raises, so it cannot silently disable the check it was added to perform.

  **No published number changes and no artefact was re-uploaded.** All nine shipped backgrounds already declared `genome: hg38`, and all nine are; the fix is that this is now checked at both ends rather than asserted at neither. Deliberately *not* done: the per-row `genome` field the issue also proposed, which would mean re-stamping and re-uploading all nine artefacts and re-cutting the pinned dataset tag, and only earns its keep if chorus ever ships mixed-species artefacts.

- **`BackgroundGenomeMismatch` and `BackgroundFoldMismatch` now share a `BackgroundArtefactMismatch` base**, and `get_normalizer` re-raises the base class. Every *other* reason a per-track artefact fails to load means "no percentiles available", which its legacy `.npy` fallback is right to absorb; these mean "the percentiles would be **wrong**", ranked against a different distribution than they name. Catching the family rather than listing subclasses means the next guard of this kind inherits the non-swallowing contract instead of depending on someone remembering to widen the clause.

- **Every committed report is now opened in a real browser and checked that it paints ([#135](https://github.com/pinellolab/chorus/issues/135)).** Nothing in the suite had ever loaded one: the checks were on bytes and JSON, so a report could be regenerated, sized, diffed, committed and shipped while rendering a blank page. `tests/test_committed_reports_render_in_a_browser.py` loads all 19 in headless Chromium and asserts every canvas IGV laid out has ink, with no console errors and no uncaught exceptions. Marked `integration`; skips cleanly where Chromium is unavailable.

  Three things had to be got right, each of which produces a failure indistinguishable from a broken panel. **IGV renders inside a shadow root**, so `document.querySelectorAll('canvas')` returns 0 for a panel that is painting perfectly — which is what led #139 to report "the panel never renders", and to the conclusion that headless Chromium cannot be used as an oracle for this. It can; the query just has to pierce shadow boundaries. **Chromium's shared libraries are scattered across conda envs** (`libgbm`/`libatk` only in `chorus-browsertest`, `libXcomposite`/`libcups` only in the oracle and base envs), so the path is assembled in the harness and passed to the browser process rather than exported by whoever runs pytest. And **"loaded" is not "painted"**: polling until the canvas count stops changing reported 61/62 and 39/42 on reports that reach 64/64 and 44/44 a second later, so the harness polls to convergence instead.

  Calibrated rather than guessed: "painted" is **any** non-white pixel, because a 0.05% floor was too high — the causal report's point tracks are ~20 marks across a 3288 px canvas (0.000238 of it) and were being reported as blank while drawing exactly what they should. And the check is shown to work: three mutations of a real report (features emptied, genome name broken, config truncated) are each caught by a different one of the three channels.

- **The CDYL fine-map report ships its IGV panel** — `examples/walkthroughs/causal_prioritization/CDYL_rs9504151/rs9504151_CDYL_locus_causal_report.html`, 25.7 MiB and now the largest artefact in the repo. It was the one example with no browser panel, skipped behind `CHORUS_WRITE_LARGE_HTML=1`, because 21 lung-fibroblast tracks × 2 alleles exceeded a 20 MiB ceiling that had been chosen as "headroom over today's largest artefact" rather than derived from any failure. Adding it moved no numbers: regenerating the example produced a JSON differing from the committed one in **1 of 4,001 leaf values**, and that one is the timestamp.

### Known limitations

- **The `SORT1_enformer` example cannot be regenerated reproducibly.** Two runs of identical code
  differ by 400 values at a 0.0159% median, while Enformer's forward pass is bitwise identical
  in-process *and* across processes — so the cause is above the model and is not localised. A diff
  on that one example after an unrelated change is expected to show ~400 moved values.
  `scripts/gate_end_to_end_determinism.py` cannot catch it, because what it compares is
  deterministic. Measured in `audits/2026-08-12_post_v0.7.2_audit.md` (F8).
- **A report still fetches one thing from the internet: the reference sequence.** Every igv.js
  version requires a sequence source and hg38 is 3 GB, so it cannot be bundled. Everything else is
  inline. Point `CHORUS_IGV_SEQUENCE_URL` at a self-hosted copy served **same-origin** with the
  report and it needs no network at all (measured: 0 requests, 0.8 s).
- **Cherimoya and ChromBPNet remain incomparable at raw magnitude** — CATv1 has no bias model and
  tracks bias-*aware* ChromBPNet, while chorus loads `chrombpnet_nobias`. Percentiles are
  unaffected, since each is ranked against its own null. Unchanged from 0.7.2.
- **One AlphaGenome MCP end-to-end test never runs on an authenticated host**:
  `tests/test_integration.py:196` gates on the `HF_TOKEN` environment variable, so it skips when
  credentials come from the stored token. That path stays unverified.

## [0.7.2] — 2026-08-12

### Changed
- **The display ceiling is 4.0 instead of 3.0, which reveals peaks that were being flattened.** Nothing is rescaled: the band is still `(v − floor) / (peak − floor)`, so **1.0 still means the track's genome-wide p99** and every value below 3.0 is the number it always was. Only the clip point moves. Measured share of each track's clipped mass that this recovers: ChromBPNet DNase 43%, Cherimoya DNase 41%, AlphaGenome H3K27ac 33%, CAGE 25%, AlphaGenome DNase 24%. On the SORT1 panel the fraction of bins sitting at the ceiling roughly halves — H3K27ac 0.0046 → 0.0010, ChromBPNet 0.0030 → 0.0018, CEBPA 0.0066 → 0.0042, Cherimoya 0.0103 → 0.0080, CAGE 0.0126 → 0.0097.

  The cost is uniform: a value of 3.0 now occupies 75% of the axis rather than all of it, so every track reads correspondingly shorter. Uniform is the point — raising the ceiling for everything preserves cross-oracle comparability exactly, where a per-track ceiling would destroy the one property that makes stacked panels readable together.

  Going higher does not pay. To clip nothing a track needs its genome-wide maximum on the axis, which is ~196 for ChromBPNet DNase and ~160 for AlphaGenome DNase in band units; at that ceiling a p99-level peak occupies 0.5% of the height and every panel reads flat.

  One consequence worth knowing: the ceiling clips *before* pooling, so mean-pooled tracks move too. AlphaGenome DNase's displayed peak went 2.92 → 3.81 despite never touching the old ceiling, because its native values were being clipped at 3.0 before being averaged. The new value is the mean of unclipped data, which is more faithful, but it is a change rather than a pure display tweak.

- **igv.js now reduces every track to pixels with `max`, not a two-name oracle list.** A report emits ~349 bp display features, and igv.js reduces *those* to pixels whenever there are more features than pixels — about 3:1 on a 1 Mb panel. That second reduction was `"max" if source_model in ["chrombpnet", "legnet"] else "mean"`, so Cherimoya and AlphaGenome's 1 bp CAGE got mean-reduced in the browser immediately after the feature stage had measured them into max: the original 5.5× dilution, one stage further down.

  Measured on the SORT1 panel at the browser's 3:1 ratio, the peak height `mean` costs: LegNet **2.33×**, AlphaGenome DNase 1.56×, ChromBPNet 1.38×, CAGE 1.31×, Cherimoya 1.14×, H3K27ac 1.00×. Two things make that unacceptable rather than merely lossy — it costs **unequally**, and 1.38× for ChromBPNet against 1.14× for Cherimoya is a 1.2× relative distortion between the two tracks a cross-oracle panel exists to compare; and `mean` cancels signed tracks against themselves, which is why LegNet is the worst case.

  `max` loses no peak anywhere. Its cost is one small floor lift (Cherimoya 0.000 → 0.074 of a 0–3 axis) and roughly doubled saturation, still well under the 0.075 readability limit (CAGE 0.013 → 0.032 is the highest).

  **One exception: log-scaled tracks keep `mean`.** The log band compresses the top of the range, so many more bins sit just under the ceiling and `max` promotes them over it — AlphaGenome CAGE's saturation went 0.003 → 0.023 (7.7×) and the clipped flat tops read as RNA-seq gene-body coverage rather than TSS spikes. Ink barely moved (0.186 → 0.192), so it is the ceiling and not the density doing the damage. `mean` costs that track 1.31× of peak height, the cheaper of the two harms.

  Note this takes the **opposite** default from the feature stage, and deliberately: that stage collapses ~349 native bins, where max lifted AlphaGenome DNase's floor to 0.707 and so has to be decided per track; the browser collapses 2–3 already-pooled features, where max has almost no opportunity to promote background. Mirroring the feature stage's per-track choice was considered and is also wrong here — it would send AlphaGenome DNase and the ChIP tracks to mean, costing 1.1–1.6× of peak for floor protection a 3:1 collapse does not need.
- **⚠️ Cherimoya now uses fold 0 by default instead of the 5-fold ensemble, so its scores move.** Anyone comparing numbers against 0.7.1 will see Cherimoya percentiles and log2FCs differ. This is the default model changing, not a regression.

  Agreed with CATv1's author, [@jmschrei](https://github.com/jmschrei): five models complicate and slow most analyses, so fold 0 is the right default for an interactive tool and the ensemble is there when you want to dig deeper. Fold 0 also matches ChromBPNet's default fold, and the two nulls are built on the same reference sets — the Cherimoya null reproduces ChromBPNet's `effect_counts=18672` and `summary_counts=34004` exactly — so a **percentile** from one means what a percentile from the other means. Raw magnitudes are *not* comparable; see the bias note below.

  The 5-fold mean stays available as `fold="ensemble"` and has its own null (`cherimoya_ensemble_pertrack.npz`); a percentile is a rank against a background, so each mode is ranked against a background built with the same model. Selection is automatic — every Cherimoya prediction records its fold — and a mismatch raises rather than returning a plausible wrong number.

  Folds 1–4 now raise: no null ships for them, and ranking one against another fold's null returns a plausible wrong number rather than an approximation.

- **Documented: Cherimoya and ChromBPNet are not comparable at raw magnitude, because one is bias-corrected and the other is not.** chorus loads ChromBPNet as `chrombpnet_nobias`, the "TF Model" that predicts the *bias-corrected* accessibility profile — ChromBPNet trains a Tn5/DNase bias model first and regresses its effect out. CATv1 does no such correction: the shipped checkpoints carry `n_control_tracks: 0`, its training config has `controls = None`, and its ATAC recipe uses GC-matched negatives with no control file. Measured on the four shared DNase experiments, fold 0 both sides:

  | CATv1 vs | window sum | peak | peak/sum |
  |---|---|---|---|
  | `chrombpnet_nobias` (what chorus loads) | 1.32× | **3.40×** | **2.19×** |
  | `chrombpnet` (bias-aware) | 1.14× | **1.02×** | **0.80×** |

  So CATv1 tracks the bias-*aware* model almost exactly and differs from the bias-corrected one by ~3.4× on peak height. Profile shape agrees regardless (rank correlation 0.95, peaks 18 bp apart) — it is height and sharpness that separate. **Percentiles are unaffected**, since each oracle is ranked against its own null (0.9325 vs 0.9550 at the SORT1 locus), which is why the cross-oracle panel reads consistently. This asymmetry is pre-existing, not introduced here, and switching the panel to bias-aware ChromBPNet is deliberately left as separate work.

- **Cherimoya is 3.6× faster on the 5-fold ensemble, and 17× faster in-process** — contributed by [@jmschrei](https://github.com/jmschrei) in [#165](https://github.com/pinellolab/chorus/pull/165). Triton benchmarks its `autotune` candidates the first time a kernel sees a shape, which is right for training but was being re-run in *every* subprocess to serve a single forward pass, because `run_code_in_environment` spawns a fresh process per call. Chorus now enables Triton's on-disk autotune cache before importing `cherimoya`, so the winner is reused across processes. Measured on one H200, `DNASE:ENCSR149XIL`, single 2,114 bp window:

  | mode | before | after | |
  |---|---|---|---|
  | `use_environment=True`, 5-fold ensemble | 89.5 s | 25.0 s | 3.6× |
  | in-process (`use_environment=False`) | 13.6 s | 0.79 s | 17.3× |

  **Predictions are bit-identical**, verified in both modes rather than assumed: the ensemble window sum is 954.37564392006766 and the peak 11.103147742142609 before and after, to the last digit. It reuses the config `autotune` already chose; an unseen shape still falls back to benchmarking. The cherimoya integration suite got 3–4× faster on its own as corroboration (`test_variant_effect_runs_end_to_end` 178 s → 50 s).

### Added
- **The Cherimoya documentation gaps are filled** — contributed by [@jmschrei](https://github.com/jmschrei) in [#166](https://github.com/pinellolab/chorus/issues/166), landed via [#170](https://github.com/pinellolab/chorus/pull/170). Cherimoya was missing from the `chorus setup --oracle` list and from the `download_pertrack_backgrounds` pre-download loop, so a reader following the README could not have set it up; it was also absent from `docs/THIRD_PARTY.md` attribution, the `CLAUDE.md` env list and `examples/notebooks/README.md`. Adds the macOS caveat (Cherimoya is CPU-only on Apple Silicon: no MPS path in the model, and its `triton>=3.5.1` pin ships no macOS wheel) and corrects the background size to ~162 MB. Verified against the artefacts: 1,518 tracks = 369 ATAC + 1,149 DNASE, `cherimoya_pertrack.npz` 161.7 MB.
- **The README's Cherimoya timing table is re-measured**, because #165 invalidated it — its "~12 s per fold per call" *was* the autotune benchmark that change removes. The headline row moves from ~60 s to ~25 s. The section now also states, with numbers, why pinning a single fold is not a cheaper approximation of the default: on one window the five fold peaks are 8.24 / 15.47 / 15.34 / 11.08 / 7.65 against an ensemble peak of 11.10, so they disagree by 2.02× among themselves and any one lands between 0.69× and 1.39× of the ensemble — and the shipped background CDFs were built against the ensemble, so a single fold would be ranked against the wrong null.

### Fixed
- **Two IGV display decisions were hardcoded per oracle name, and both were wrong for at least one track.** They are now measured per track from the data being drawn.

  **Pooling.** A display bin covers ~349 bp of a 1 Mb panel, so the renderer must reduce 349 native values to one — mean or max. That was chosen from a list of oracle names, which is how Cherimoya (a BPNet-family 1 bp model, absent from the list) rendered its peak at **0.547 instead of 3.000** — a 5.5× display-only dilution on the same 0–3 axis as ChromBPNet, in a report whose purpose is cross-oracle comparison. Five candidate predictors were measured and all get the sign wrong on at least one oracle: `resolution <= 1`, per-bin `max/p99` from the artefact (22 for Cherimoya against 65 for AlphaGenome — backwards), the artefact's signal mass above p99 (0.122 for ChromBPNet against 0.243 for AlphaGenome — backwards), profile density, and density × collapse factor (AlphaGenome and Cherimoya both emit DNase at 1 bp and both collapse 349 bins, yet max lifts one floor to 0.707 and the other to 0.000 — and *Cherimoya* is the denser). So it is measured instead: max-pooling can never lose a peak and mean-pooling can never lift a floor, so the only question is whether max lifts *this* track's floor, which costs one reduce over an array already in memory. It decides per track, which matters because AlphaGenome needs opposite answers for its own 1 bp and 128 bp tracks.

  **Scale.** `floor=p95, peak=p99, linear` assumes signal decays smoothly out of the background. That holds for accessibility and fails completely for base-resolution TSS assays, where the genome is a huge mass of near-zero plus a tiny population of enormous peaks. AlphaGenome CAGE has p95 = 0.0050 and p99 = 0.0405 against a max of **852**, so **every real TSS from strength 1 to 3000 rendered at exactly 3.00** — no dynamic range among peaks at all. Measured on the shipped SORT1 panel, **13.1% of CAGE's display bins were pinned at the ceiling**, against 0.0–1.3% for the panels that read well.

  A track is now re-rendered on a log band (`log1p`, anchored p99.5/p99.9) when the linear band is *measured* to clip more than 4% of the bins it will actually draw, and the log band is kept only if it both brings saturation down **and** leaves the strongest feature at or above 1.0 — that second condition is what an earlier attempt lacked, where p99.9/p99.99 anchors "fixed" saturation by dropping CAGE's peak to 1.24 of 3.0 and erasing the track.

  It is measured on the panel because no genome-wide CDF statistic can predict it. Four were tried across the 20,366 tracks that carry a CDF, split into "must get the log band" (1,296) and "was working, must not move" (19,070), and every one overlaps: `max/p99.9` (must-log down to 172, must-stay-linear p95 20.5 and **max 4212**), `p99.9/p99` (5.7 vs 15.6), `p99/p95` (3.0 vs 10.0), and predicted clip fraction (0.0028 vs 0.0045). `max/p99.9` at a threshold of 50 briefly looked clean at 41× separation — only because ChromBPNet's ChIP tracks had been left out of the protected set. It would have log-scaled **130 tracks outside the must-log set: 102 ChromBPNet ChIP, 10 Enformer and 8 Borzoi CAGE, 7 AlphaGenome TF-ChIP, 2 ChromBPNet DNase and 1 Cherimoya DNase** — including AlphaGenome's own TF-ChIP tracks, which is the most awkward case, since AlphaGenome is the oracle the rule exists for. And on a 10,000-point grid `int(0.9999 × n)` is the last slot, so that statistic is a ratio to the single extreme order statistic the null protocol explicitly warns against.

  Saturation also has to be measured **as drawn, not natively**: pooling is what creates it. CAGE's *native* clip rate is 0.005–0.014, indistinguishable from the ChIP tracks at 0.001–0.008, because CAGE is 1 bp and collapses 349 native bins into each display bin while ChIP is 128 bp and collapses 2. The separation only exists after pooling — 0.131 against ≤0.013, a 10× gap — which is where the trigger is applied.

  Saturation — not ink — is what makes a panel unreadable, and the two concerns stay separate: pooling protects the floor, the scale protects the peaks. Cherimoya inks 41% of its display bins and reads correctly, because only 1.3% of them clip. An "ink fraction" criterion was tried for pooling and flipped Cherimoya and ChromBPNet to mean, re-creating the original defect.

  The 7.5% limit is calibrated on the **corpus**, not on one panel. An earlier 4% came from the geometric midpoint of a single panel's gap; measured across all 346 subtracks of the 19 committed IGV panels it cuts through the middle of the population — 45 subtracks (13%) exceed it, including seven Enformer CAGE tracks at 0.042–0.063 that render acceptably and whose peaks the log band would compress, silently invalidating committed panels that were never regenerated. The real gap has nothing in it: the 22 subtracks at or above **0.0899** are exactly the AlphaGenome CAGE/ATAC/DNase panels at the SORT1 locus, and the next one down is **0.0656**. The limit sits inside that gap, and the calibration is conservative in the right direction because it was measured under the *old* pooling — the pooling fix lowers saturation for exactly these dense 1 bp tracks.

  Acceptance also requires the log band to **clear the limit or at least halve the clipping**, not merely reduce it: a bare "did it go down" test is satisfied by an epsilon, so a track going 0.550 → 0.500 would be re-rendered, relabelled and still ship with half the panel pinned. A log band whose anchors collapse — reachable from shipped data, `chrombpnet CHIP:HEK293:ZNF24` has p99.5 = −7.4e-07 and p99.9 = −3.3e-10, which `max(x, 0)` maps to the same 0.0 — renders a two-level barcode that would pass both tests, since clipping guarantees the peak; it is now rejected outright.

  **Signed tracks are excluded from both measured decisions.** "Does max-pooling lift the floor" is meaningless for a track with no floor at zero, and the answer is actively harmful: max over a bin holding a strong repression and a weak activation returns the activation, so the repressive half of the panel disappears. Measured on borzoi `ENCFF734OLC+` (signed, 32 bp, 11 native bins per display bin) the measured choice flips mean → max and takes displayed saturation 0.000 → 0.138. 2,253 shipped tracks are signed (borzoi 1,543, AlphaGenome 667, Sei 40, LegNet 3); they keep the static geometry-based choice.

  A re-rendered track is **labelled** `(log scale)`, because its 1.0 means genome-wide p99.9 rather than p99. Two same-assay panels in one report can legitimately land on different bands — BCL11A's two CAGE:K562 tracks measured 0.053 and 0.036 saturation and only the first escalated. The 0–3 axis was always per-track (1.0 is *that* track's percentile, never a shared raw value), so mixing bands is not new; shipping it unlabelled would be. Follows LegNet's existing `(per-track norm)` precedent, and every IGV legend now carries the same caveat — it previously asserted p95/p99 anchors and "tracks comparable" for every track unconditionally.

  Deliberately **not** extended to the matplotlib/CoolBox figure paths, which share `rescale_for_display`: they mean-smooth rather than max-pool, so they cannot manufacture ceiling bins, and CAGE's native clip rate (0.005–0.014) is already under the limit — wiring the escalation in there would be a no-op that churns every committed figure.

  Verified on the shipped SORT1 multi-oracle panel: **exactly two tracks change** — AlphaGenome CAGE (saturation 0.131 → 0.013, ink 0.503 → 0.132, peak still 3.00) and AlphaGenome DNase (0.090 → 0.000, ink 0.982 → 0.423, peak 3.00 → 2.92, from the pooling fix). The other ten, including every ChromBPNet, Cherimoya, LegNet and AlphaGenome ChIP track, are bit-identical. `CHIP:K562:ZBTB11` — the ChromBPNet track the discarded CDF statistic ranked as *more* bimodal than CAGE — measures 0.000 saturation as drawn and is left untouched. `tests/test_display_scale_is_measured_not_declared.py` pins the scope of both rules, including that all three duplicated render paths (`_igv_report`, `multi_oracle_report`, `causal`) go through the measured decision — patching one of the three is how a change to this logic came back reporting byte-identical output.

  Not changed, deliberately: **LegNet**. Its genome-wide maximum maps to 2.6 of a 3.0 axis, so its scale is correct and the SORT1 window simply contains no strong promoter. MPRA activity is a compact bounded score, not a heavy-tailed count; making it look dramatic would overstate the data.

## [0.7.1] — 2026-08-10

Bookkeeping release. **No CDF changed** — the matrices are byte-identical to 0.7.0 — so no
percentile moves. Append-in-place, no rebuild, no GPU.

**Background artefacts.** Pinned to dataset revision [`backgrounds-2026-08-10-layers`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/tree/backgrounds-2026-08-10-layers). Identical CDFs to `backgrounds-2026-08-06-schema4`; the only difference is the per-row layer array and `build_config.layers_present`. The 0.7.0 revision is left intact, so 0.7.0 stays reproducible.

### Fixed
- **`layers_per_row` shipped on three oracles of eight, and ChromBPNet was one of the five without it.** AlphaGenome (6 distinct layers), Borzoi (5) and Enformer (4) carried the array because their builders construct `track_info` with a `'layer'` key; Cherimoya, ChromBPNet, EPInformer-seq, LegNet and Sei never had that concept in their builders, so the array was absent and `build_config.layers_present` was `null`. For four of the five that costs nothing — they are single-layer, so the array carries no information. ChromBPNet is not: its **753 rows span ATAC (4), DNASE (5) and CHIP (744)**, accessibility *and* TF binding, and code keying on the array had to fall back to re-deriving each row's layer from the track-id string.

  Now stamped on all eight. Every value is produced by the same `classify_track_layer` the query path calls — on a shim carrying the assay_type that oracle emits — and then asserted equal to it, because a stored array that disagreed with the query path would be worse than no array. What had to be supplied per oracle is only the assay_type, since the ids do not all carry one: from the id prefix for ChromBPNet, Cherimoya and EPInformer-seq, and as a constant for Sei (`sequence-class`) and LegNet (whose ids are bare, e.g. `K562`). Result: ChromBPNet 9 + 744, Cherimoya 1,518 accessibility, EPInformer-seq 33 enhancer activity, LegNet 3 promoter activity, Sei 40 regulatory classification.

  Verified against values recorded before the stamp: all three CDFs bit-match for Cherimoya `DNASE:ENCSR149XIL` and ChromBPNet `DNASE:HepG2`.

  It stayed invisible because `tests/test_canonical_layer_vocabulary.py` validates the array *when present* and never required presence — a shape of test worth naming, since it reads like coverage and is not. `tests/test_every_background_carries_its_layers.py` now requires presence, per-row length, canonical values, no `'other'`, agreement with `build_config.layers_present`, and equality with what `classify_track_layer` computes.

  One defect introduced and fixed on the way, recorded because the tag moved: the first stamp wrote `build_config` as a **0-d** array where every artefact stores shape `(1,)`, which raises `IndexError` in any reader doing `build_config[0]` — it broke `test_the_shipped_null_records_that_it_was_built_from_the_ensemble` on five artefacts at once. The stamper's own reader tolerated both shapes, which is exactly why it did not notice. Repaired, and `test_build_config_storage_shape_is_uniform` now asserts the shape rather than tolerating it. The `backgrounds-2026-08-10-layers` dataset tag was deleted and recreated at the corrected head; nothing referenced it yet, and `backgrounds-2026-08-06-schema4` was untouched throughout, so 0.7.0 stays reproducible.

### Changed
- **Five GitHub Releases told users to run a command that has never worked.** v0.5.2 through v0.5.6 ended with `pip install --upgrade chorus-genomics`. Chorus is not on PyPI: `chorus-genomics` is unregistered and the `chorus` name belongs to an unrelated chemistry package. All five release bodies now give the source install, with a note saying what was corrected and why. Found by verifying the fix rather than trusting the first three I had read — the check turned up two more.

## [0.7.0] — 2026-08-10

**Effect percentiles change, and are not comparable with any earlier release.** The
background sampler was discarding the tail it exists to measure: a uniform *m*-of-*N*
reservoir subsample retains the population maximum with probability exactly *m*/*N*, and
that maximum is what a percentile clamps against. 9 of 19 (oracle, layer) reservoir pairs
were thinned, 1.36x to 43.5x; AlphaGenome `gene_expression` ceilings were understated by up
to **8.3x** while p99 stayed right to 0.6%, which is why no percentile test caught it.

**Background artefacts.** This release is pinned to dataset revision [`backgrounds-2026-08-06-schema4`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/tree/backgrounds-2026-08-06-schema4) of `lucapinello/chorus-backgrounds`. `schema_version 4`, one `build_id` across all eight oracles. Older cached copies are moved aside and refetched automatically. This is the first release to pin a revision at all — see the *Added* entry below for why that matters.

### Added
- **A release now names the artefact revision it was verified against.** A percentile is a function of *(code, artefacts)*, and the artefacts live in a separate HuggingFace dataset whose `main` moves independently — so until now the same chorus commit produced different numbers depending on when the user happened to download. That is demonstrated rather than hypothetical: the 2026-08-10 upload replaced every file in place, which silently changed the behaviour of every already-released version that fetched afterwards. Both states are now tagged on the dataset (`backgrounds-2026-08-01-preunified`, `backgrounds-2026-08-06-schema4`), every download and listing call passes `revision=`, and `CHORUS_BACKGROUNDS_REVISION` overrides it for anyone developing a new oracle's background — with a note saying what that costs. `tests/test_artefact_revision_is_pinned.py` fails if a call site loses the pin or the pin stops matching the artefacts on disk.

### Changed
- **`pytest` now means the same thing for a contributor, for CI and in the audit checklist.** `pytest.ini` set no `addopts`, so a bare `pytest tests/` collected the integration tests — which spawn oracle subprocesses and download weights — and failed on any machine without the per-oracle environments, while CI stayed green by passing `-m "not integration"` *and* `--ignore=tests/test_smoke_predict.py`. Both flags were working around defects: the exclusion now lives in `pytest.ini`, and the smoke tests are marked `integration` (which they always were in substance) with prerequisite guards so a machine without the envs gets skips instead of a wall of fixture ERRORs. Default suite 1,443 passed in 5:23 against 28:49 for the everything run.
- **The build documentation described the defect this release fixed as the design.** `scripts/README.md`, which README and API_DOCUMENTATION both cite as the pipeline reference, predated the rebuild: it documented capacity-50,000 reservoir subsampling as intended behaviour, listed 6 oracles of 8, gave ChromBPNet as "24 models, 2.4 MB" against a real 753 tracks and 79.5 MB, described one baseline mixture where there are three, and pointed at a build-script path that does not resolve for EPInformer-seq. Rewritten from values read out of the artefacts. README also claimed the downloader "will not overwrite local files — it only fetches when the file is missing", which is exactly the behaviour that would have made this release's corrected backgrounds reach nobody.

### Fixed

- **Two notebook examples showed nothing, and the reason was different in each case.** Reported from the rendered output of `examples/notebooks/single_oracle_quickstart.ipynb`. Neither was a regression from this release — the figure has been blank since 2026-08-01, the first re-execution after the 2026-05-08 change that made CDF rescaling with a fixed 0–3 axis the default (ink fraction 0.07034 before, 0.01066 after, identical at every revision since including the merge-base).

  *The synthetic-sequence panel* was blank because the sequence had almost no regulatory content. Its builder appended variable-length blocks inside a fixed-stride loop, so the blocks summed to **56.2%** of the target length, `[:context_size]` never fired, and **43.8% of the final sequence was poly-A padding** — leaving 53.4% `ACGT` tandem repeat, 1.94% literal `N`, and **0.88% real motif bases**, with element positions drifting up to 171 kb from where the code claimed. Raw maxima of 0.0983 and 0.2468 then sat *at* the genome-wide display floors (p90 0.09101, p95 0.21096), rendering at 0.20% and 1.49% of the axis. Fixing the builder alone was measured to be insufficient (display max still 0.367): Enformer predicts accessibility from promoter context, not isolated motifs, and a GATA cluster with no CpG island reproduces the blank panel exactly. The rebuilt example plants a 1 kb CpG island with a GATA cluster and a TATA box at the centre of the input window; it now renders at **56.8%** and **98.1%** of the axis, with the peak inside the planted element across all five seeds tested.

  *The variant-effect example* reported `np.mean(values)` over the whole 114,688 bp output window, differenced between alleles. The effect was real — DNase Δ −0.0338 at the variant's own bin — but the window mean diluted it **54×–1307×** and **inverted its sign in 5 of 6 (track, allele) combinations**, printing +0.00026 where the variant's bin had dropped by 0.0338. The per-bin `effect_sizes` array that chorus already computes was fetched and never used. The example now reports the change at the variant bin, the largest per-bin change and its offset, and adds a difference track — an absolute overlay cannot show a small effect at any y-scale, and ref/alt were separated by less than one pixel. Its variant was also near-neutral, so it taught that variants do nothing; it now uses chrX:48,783,008 A>C, which breaks a WGATAR site 3.5 kb upstream of *GATA1* and ablates **97.6%** of the local DNase peak while dropping the distal *GATA1* promoter CAGE peak by 28.6%.

  The same whole-window metric was found and fixed in three further notebooks (`advanced_multi_oracle_analysis`, `comprehensive_oracle_showcase`, `epinformerseq_testing`, the last of which contradicted a correct cell in the same file — +1.067 against +1.426 log2FC), along with one other near-blank panel whose prediction covered 11.4% of its plotted x-range. **ChromBPNet is a documented exception**: its output is `softmax(profile) × expm1(counts)`, so the sum over its predicted bins *is* the count head's total-accessibility prediction and a single base rescales every bin — its window statistic is a legitimate published readout, not a dilution, and both notebooks now say so.

  New `tests/test_notebook_figures_are_not_blank.py` decodes all 41 committed figures. Nothing in the suite had ever examined a notebook's pixels (`grep -rl image/png tests/` returned zero files), and all 13 assertions of the three existing notebook-figure tests pass against a blank panel. The test also records why the obvious check fails: blank panels carry **more** ink (0.0049–0.0107) than legitimately sparse ones (0.0019–0.0038), so the threshold is trace peak height as a fraction of its axis, set at the geometric midpoint of the measured gap.

- **Cherimoya's IGV track rendered at a fifth of its true height, because display pooling was keyed on oracle *name*.** `_calculate_track_bin_size` max-pooled ChromBPNet and LegNet by name and mean-pooled everything else. Cherimoya is a BPNet-family model with the same 1 bp point-profile output as ChromBPNet, but it was not in the list, so it fell through to mean-pooling — which dilutes a one-base peak by the width of the display bin.

  Measured on the SORT1 multi-oracle panel (`DNASE:ENCSR149XIL`, 1,048,396 bp window, 349 bp bins): the ensemble 1 bp profile peaks at **11.10**, which max-pools to a rendered **3.000** — the same ceiling ChromBPNet reaches — but mean-pooled to **0.547**. A **5.5× display-only dilution**, drawn on the same 0–3 axis as ChromBPNet, in a report whose entire purpose is cross-oracle comparison.

  **No score was ever affected**, and that is worth stating precisely because the panel invited the opposite conclusion: the 501 bp window sum is linear, so Cherimoya's log2FC was `1.4576` against ChromBPNet's `1.3756` throughout, with quantiles 0.9997/0.9995 and Cherimoya's reference window at the *higher* activity percentile (0.957 vs 0.906).

  Two candidate universal rules were measured and both are wrong, so the fix does not pretend one exists. `resolution <= 1` would also flip AlphaGenome, which emits DNase at 1 bp as well — it must not flip, because a point profile is sparse on a near-zero floor (Cherimoya's null: p50 0.075, p99 3.38) where max recovers the peak without lifting the floor, whereas AlphaGenome's 1 bp DNase is dense coverage (p50 0.020, p99 0.285) where max over 349 dense bins would inflate the whole track. And artefact "spikiness" points the wrong way: per-bin max/p99 is 22 for Cherimoya against 65 for AlphaGenome. So pooling is now a **declared per-oracle property**, with `tests/test_igv_pooling_is_declared_per_oracle.py` failing for any oracle that is neither declared a point-profile nor declared coverage — a silent fall-through is what caused this.

  Both affected artefacts regenerated. ChromBPNet and LegNet feature values are bit-identical across the change.

- **AlphaGenome is now max-pooled for display too, by maintainer decision, and the trade is measured.** The rule above initially kept AlphaGenome on mean-pooling: it emits DNase/CAGE at 1 bp and so was suffering the same bin-width division as Cherimoya, but its 1 bp output is *dense coverage* (per-bin null p50 0.020, p99 0.285) rather than sparse spikes on a near-zero floor (Cherimoya p50 0.075, p99 3.38), so the max of ~349 dense samples lands near the upper tail almost everywhere and lifts the baseline as well as the peak. The maintainer's call was to treat it like the BPNet-family models anyway, so that every panel in a cross-oracle report is computed the same way. Measured on the SORT1 panel over a 1,048,396 bp window:

  | AlphaGenome track | peak | bins above 1.0 (its own genome-wide p99) | mean displayed |
  |---|---|---|---|
  | `DNASE:HepG2` | 2.918 → 3.000 | 1.96% → **32.61%** | 0.0800 → **0.9915** |
  | `CAGE:HepG2` | 2.567 → 3.000 | 1.40% → 22.06% | 0.0630 → 0.6417 |
  | `CHIP:H3K27ac:HepG2` (128 bp) | 3.000 → 3.000 | 2.08% → 2.39% | 0.0709 → 0.0838 |

  So the peaks became comparable with ChromBPNet and Cherimoya on the shared axis, which is what was wanted, and the average displayed bin on the 1 bp tracks rose to roughly the genome-wide p99 — those panels now read as broadly hot rather than as peaks against a floor. The 128 bp histone tracks are effectively untouched, which is the control: the effect comes from collapsing many native bins, not from the pooling choice as such. Reverting is moving two names between two frozensets in `_igv_report.py`; `tests/test_igv_pooling_is_declared_per_oracle.py` carries the table so the trade stays visible.

- **The background sampler was throwing away the tail it existed to measure.** Every percentile Chorus reports is a rank against a per-track empirical null, and `effect_percentile` is `min(rank/denominator, 1.0)` — so it clamps the moment an effect reaches the largest *sampled* background value. That ceiling had been patched three times (re-anchoring, union-at-2N, the read-side `effect_exceedance` ratio). None of them addressed the cause, because the cause was a defect, not a limitation:

  **`ReservoirSampler` keeps a uniform subsample once a track's offered count exceeds its capacity, and a uniform *m*-of-*N* subsample retains the population maximum with probability exactly *m/N*.** The maximum is precisely what the clamp is computed against. So the sampler was, by construction, discarding the statistic the whole mechanism depends on — and doing it silently, at a rate nobody had measured.

  It was not one oracle. **9 of 19 (oracle, layer) reservoir pairs were thinned**, from 1.36× to 43.5×:

  | oracle | layer | offered | cap | thinning | retained now |
  |---|---|---|---|---|---|
  | alphagenome | effect | 148,367 → 225,253 | 20,000 | **7.42×** | **exact** |
  | alphagenome | summary | 104,033 → 319,642 | 20,000 | **5.20×** | **exact** |
  | alphagenome | perbin | 328,992 → 987,776 | 20,000 | **16.45×** | 20,000 + exact top 19,740 |
  | borzoi | summary | 75,021 | 50,000 | **1.50×** | **exact** |
  | borzoi | perbin | 991,552 | 50,000 | **19.83×** | 50,000 + exact top 19,832 |
  | enformer | perbin | 992,160 | 50,000 | **19.84×** | 50,000 + exact top 19,844 |
  | chrombpnet | summary | 68,008 | 50,000 | **1.36×** | **exact** |
  | chrombpnet | perbin | 2,176,256 | 50,000 | **43.53×** | 50,000 + exact top 43,526 |
  | cherimoya | perbin | 1,088,128 | 50,000 | **21.76×** | 50,000 + exact top 21,763 |

  Borzoi's `effect` layer is the near-miss worth naming: at the old position count it offered 34,482 against a 50,000 cap and was never thinned, but the new count takes it to 51,831. **The rebuild meant to fix this defect would have introduced it**, on the one layer that had been clean. A preflight catches it in two seconds; nothing before this release would have.

  **The mechanism is confirmed, not inferred.** `1 - m/N` predicts what share of tracks should gain a higher ceiling under exact retention. Across the six layers whose position population is unchanged — so the only variable is retention — prediction and measurement agree to within 1.4 points over a 32-fold range of thinning:

  | oracle.layer | thinning | 1 − m/N predicts | measured |
  |---|---|---|---|
  | chrombpnet.summary | 1.4× | 26.5% | **25.9%** |
  | borzoi.summary | 1.5× | 33.4% | **34.2%** |
  | borzoi.perbin | 19.8× | 95.0% | **95.0%** |
  | enformer.perbin | 19.8× | 95.3% | **95.3%** |
  | cherimoya.perbin | 21.8× | 95.4% | **96.8%** |
  | chrombpnet.perbin | 43.5× | 97.7% | **97.5%** |

  And the control holds: **0 of Borzoi's 6,068 unthinned tracks moved.** AlphaGenome's three layers are deliberately excluded from that table — their position count grew too, so retention is not the only variable and the identity does not apply. Quoting them as agreement would have been dishonest.

  Why it survived every existing test. Re-unioning AlphaGenome's raw shards showed `gene_expression`'s maximum was wrong by a **median 1.33×, up to 8.34×**, while p99 was right to **0.6%**. Every percentile test looked at the body. The tail was the only thing broken, and the tail was the only thing that mattered. It also explains instability previously written off as sampling variance: after an earlier re-anchoring, 12/12 Enformer TF tracks got a wider p99 while 11/12 reported a *lower* maximum — which is not noise, it is the subsample.

  What the fix is. `effect` and `summary` are now **exact** for all eight oracles. `perbin` — 2.2 M values on ChromBPNet, and display-only — keeps a uniform body plus an **exact top-K**, with `K` derived (`ceil(200 × N_expected / n_points)`, 2% margin) rather than picked, so at least the top 2% of the grid is exact. The hybrid degenerates **bit-identically** to the old implementation when `N ≤ C`, so every pre-existing grid-integrity test still describes real behaviour. `from_flat_samples`' `capacity` is now keyword-only with no default: the original defect site (`merge_effect_shards.py` inheriting `DEFAULT_CAPACITY=50_000` from a bare call) is now a `TypeError`.

  **User-visible effect**, measured on the committed corpus:

  | | before | after |
  |---|---|---|
  | Enformer, 168 committed effects pinned at 1.0000 | 9.5% | **3.6%** |
  | AlphaGenome, 1,792 committed effects pinned | 2.5% | **2.0%** |
  | SORT1 rs12740374 × C/EBP validation, 246 rows | CEBPA at **1.11× its null's maximum** | **0 rows pinned** |

  That last row is the one to read. `CHIP:CEBPA:HepG2` had an effect *above* everything its null contained, so it reported 1.0 and carried no ranking information at the exact locus the walkthrough exists to explain. It now resolves to **0.9998** (raw +2.945), CEBPB to **0.9995** (+3.316), CEBPG to **0.9997** (+2.460).

  **Honest negatives, all measured:**

  - **Meuleman DHS was tested and rejected.** It was the obvious candidate for widening the ChIP tail and was added to the plan on that basis. Two-arm measurement at n=6,000/arm: it raised **no** ceiling anywhere on Sei (max ratio exactly 1.000 across all 40 tracks) and *diluted* Enformer `tf_binding` worse than any other layer (p99 **0.858**; 744 of 2,101 tracks gained a ceiling, 1,217 lost one). The prior argument that "an additive union cannot hurt" is wrong as stated — `max(union) = max(max_a, max_b)` protects the **maximum**, not the quantiles, and p99 is what a user reads. No composition change ships without a two-arm measurement; every unmeasured composition guess in this project has been wrong.
  - **Motif-creation saturation is unfixed and unfixable this way.** AlphaGenome `histone_marks` (18.2%) and Enformer `tf_binding` (25.0%) still pin. A null over random regulatory positions contains almost no single-base changes that *complete* a specific factor's motif, which is exactly the saturating case. Correcting the ceiling cannot manufacture draws that were never in the population. `effect_exceedance` — the ratio past the end of support — remains the answer above the ceiling, and remains a ratio rather than a percentile on purpose.
  - **Extrapolation is not the answer, and this was measured rather than assumed.** A GPD fit overshoots the far tail by **3.8×** and an exponential undershoots by **0.27×**, while the plain empirical maximum is within **13%**. A GPD is well calibrated for modest extrapolation (1.05× at q=0.999) and useless beyond it. Percentiles stay strictly empirical.
  - **LegNet's effect null got narrower at the top, on one of three tracks.** LegNet was never thinned; it moved to the shared `snps_promoter` reference class (11,913 → 17,805 positions), and that composition change took K562's ceiling from 0.906 to **0.789** (0.871×) while HepG2 and WTC11 were flat or wider (0.991×, 1.051×). A narrower ceiling raises percentiles, the opposite of the intent. Verified consequence-free on the shipped artefacts — 0 of LegNet's committed rows pin — but it is a composition effect on n=3 and is recorded as such, not smoothed over.

  **Guards added**, because every defect in this cycle was one a guard should have caught. The pre-existing write-time check passed `offered` counts to a function whose geometry assertions describe `retained`, and which did `if n >= n_points: continue` — **skipping every thinned row by construction**. Its docstring's promise was false precisely when it mattered. Now: `thinning_violations` (independent of the geometry check, not folded into it), `yield_violations` (an all-zero build previously wrote a valid file), `scope_violations` (a ChromBPNet run built 9 of 753 tracks, exited 0, reported 100% yield and exact retention — only a track-set comparison caught it), `abort_if_nothing_loads` (a wrong-env run logged 1,518 identical warnings over 75 minutes instead of stopping), and a per-row **and file-level** array-preservation check (`layers_per_row` was lost twice; `build_config` once, invisibly, because the first version of that guard only checked arrays whose first dimension was the track count).

  Provenance is now `schema_version: 4`, one `build_id` across all eight oracles, stamped **from the artefacts rather than the build logs**, and records the reference-set sha256s so a rebuild can be shown to reproduce the population. All eight verified against the committed reference sets; backups at `/data/chorus_data/pre_unified_rebuild/`.

- **`list_tracks` returned 200 tracks of 1,504 without saying so.** All four search branches returned `{"num_results": len(results), "tracks": results[:200]}`. The true count was present, in a sibling field — but a caller reading `tracks`, the field named after the thing it asked for, got a silent 13% sample. For an MCP tool that caller is usually a model, and "the list I was handed is the list that exists" is the natural reading; the failure mode is an agent concluding a track is unavailable and narrowing its own analysis. Responses now always carry `showing` (== `len(tracks)`, so the two cannot drift) and `truncated`, plus a `note` naming `num_results` when rows were dropped. Unconditional, because a flag that appears only when set is a flag you must already know to look for — which is how this shipped. Same shape as the reservoir defect above, and the same remedy: make the loss visible where it happens.

- **The report collapsed every top-ranked percentile into one `≥99th` bucket, hiding the ordering this release restored.** `_fmt_percentile` bucketed everything at or above 0.99. That was **correct while the nulls were thinned**: past the ceiling a percentile is clamped, so an effect 1.11× beyond it is arithmetically identical to one 10× beyond, and only the exceedance ratio can separate them — more decimals would be fabricated precision. Exact retention moved these rows *inside* support, where 0.9998 genuinely orders above 0.9995, and the escape valve (`(N× null max)`) fires only when an effect is past the ceiling — so with nothing past it, all five C/EBP rows rendered as an identical bare `≥99th`. Measured: **127 committed rows collapsing 81 distinct values**, including the case the C/EBP walkthrough exists to explain (CEBPA 0.9998 outranking CEBPB 0.9995 on a *smaller* raw effect). The rule is now "bucket exactly when the number is not real": four decimals in the tails, two in the body, bucket plus ratio only when clamped. Generalisable lesson — **a display policy tuned to a broken statistic becomes wrong when the statistic is fixed, and nothing fails.** No test caught it because the tests asserted the bucketing, encoding the old regime as the contract.

- **Signed layers rendered their entire negative half as `≤1st`.** Found while fixing the above. Signed layers span [−1, 1] — sign is direction, magnitude is unusualness — so the `q <= 0.01` test captured every negative value. The C/EBP vignette showed nine `gene_expression` rows as `≤1st` whose real percentiles were **−0.74 to −0.96**: moderately to strongly down-regulated, not bottom-1%. Beside a `≥99th` three rows above, that reads as a variant which both strongly represses *and* is indistinguishable from noise. `_fmt_percentile` now takes the `layer` and tests `|q| >= 0.99` for signed layers while keeping both ends as tails for unsigned ones, since the same number needs opposite treatment depending on the layer. Long-standing, and it survived because no test passed a negative percentile.

- **Cherimoya added to the SORT1 multi-oracle walkthrough**, pointed deliberately at the *same* question ChromBPNet answers — HepG2 DNase accessibility (`DNASE:ENCSR149XIL`) — rather than at a new assay. Two independently trained models agreeing on one variant is a stronger statement than one asserting it, and since they share a 2,114 bp window and base-pair-resolution output the rows and IGV tracks are directly comparable. They concur on direction and differ on magnitude: ChromBPNet **+1.069** (0.9977), Cherimoya **+1.793** (0.9999).

- **A test left the MCP state singleton unable to resolve the reference genome, breaking seven later tests.** `OracleStateManager` is a singleton that resolves hg38 exactly once, in `__init__`. Seven tests in `test_mcp.py` need one built under a mocked `GenomeManager`, so each sets `_instance = None` and reconstructs inside the patch — where `is_genome_downloaded()` returns False. The rebuilt singleton takes `_reference_fasta = None` and **keeps it after the patch lifts**, because nothing saved the old value. That field is what the state manager hands an oracle as `reference_fasta`, so every subsequent test scoring a genomic interval raised `ValueError: Reference FASTA required for genomic coordinates`. Fixed with an autouse snapshot/restore fixture in a new `tests/conftest.py`, covering any future test that reaches for the same pattern rather than the seven current call sites. It restores the same object rather than resetting to None, so module-scoped fixtures that loaded an oracle into the singleton keep seeing it instead of reloading a model per test. Worth recording for how it presented: the seven failures appeared only in the full suite, passed in isolation, and the box happened to be sharing GPUs with another job — so they were initially and wrongly written off as contention. Bisecting the file list identified `test_mcp.py` as the sole cause. Three tests now guard the fixture and were verified to fail without it.

## [0.6.0] — 2026-08-05

Retroactive tag for the 66 commits that landed on `main` between v0.5.6 and 2026-08-05 and
were never released. Cut on 2026-08-10 at `3e7990a` so that the state users had before the
0.7.0 rebuild has a name.

**Effect percentiles change.** Every oracle's effect null moved onto the regions its assay
actually measures, so no effect percentile from this release is comparable with 0.5.x.

**Background artefacts.** This release is pinned to dataset revision [`backgrounds-2026-08-01-preunified`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds/tree/backgrounds-2026-08-01-preunified) of `lucapinello/chorus-backgrounds`. Those artefacts predate `schema_version 4` and carry the reservoir-thinned ceilings 0.7.0 corrected. The tag exists because the dataset was overwritten in place on 2026-08-10; without it this release would silently produce 0.7.0 numbers from 0.6.0 code. 0.6.0 itself does not pin the revision — the pinning arrived in 0.7.0 — so reproducing it exactly needs `CHORUS_BACKGROUNDS_REVISION=backgrounds-2026-08-01-preunified`.

### Changed

- **Every oracle's variant-effect null is now drawn from the regions its assay actually measures.** The effect null answers "how unusual is this effect, compared to what?", and for a localised assay a uniformly random genomic position is the wrong comparison: it carries almost no signal, so the pseudocount damps its log-ratio toward zero and the null's body collapses below where real regulatory effects live.

  Five of the eight oracles already anchored on peaks — ChromBPNet drew 10,000 DHS-summit variants alongside 10,000 uniform, Cherimoya unioned `random + dhs` explicitly. The other three did not, **even though two of them already used cCREs for their baseline pass** — an asymmetry inside a single oracle, which is harder to defend than any difference between oracles. That is now closed, 8 of 8:

  | oracle | effect reference population | tracks |
  |---|---|---|
  | AlphaGenome, Enformer, Borzoi, Sei, EPInformer-seq | gene-anchored ∪ ENCODE SCREEN cCREs | 18,145 |
  | LegNet | promoter-anchored (TSS ±250 bp, PLS, pELS) | 3 |
  | ChromBPNet, Cherimoya | uniform ∪ DHS summits (unchanged) | 2,271 |

  Measured per track, as the ratio of the new null's tail to the old one's — median over tracks, and the share of tracks that got wider:

  | oracle | tracks | p99 | p99.9 | wider |
  |---|---|---|---|---|
  | Sei | 40 | **2.05×** | 1.80× | 100% |
  | EPInformer-seq | 33 | 1.38× | 1.28× | 76% |
  | LegNet | 3 | 1.30× | 1.17× | 67% |
  | Enformer | 5,313 | 1.26× | 1.33× | 84% |
  | Borzoi | 7,611 | 1.19× | 1.19× | 82% |

  The concrete win: at SORT1, **half of Enformer's chromatin-accessibility rows had a percentile pinned at exactly 1.0000** — the column had stopped discriminating precisely where it mattered. That is now zero.

- **It is a union at doubled N, not a mixture — and the difference is the whole finding.** The first attempt held the position count fixed and gave cCRE 25% of it. It made things *worse*. The statistic that decides whether a percentile still discriminates is the null **maximum**, and a maximum grows with the number of draws, so splitting a fixed budget shortens *every* component's tail. Measured on one Enformer accessibility track and one TF track:

  | reference set | accessibility | tf_binding |
  |---|---|---|
  | gene-anchored, 5,949 positions | 1.653 | 3.539 |
  | cCRE-only, 5,986 positions | 2.754 | 3.301 |
  | 25/75 mixture, 5,962 total | 1.697 | **2.937** ← below both |

  TF saturation went from 25% of rows to 92%. Keeping each component at full size instead makes the union's maximum exactly `max(max_gene, max_cCRE)`, so it is **provably never worse than the better component for any layer**. The gene-anchored half reproduces the previously shipped counts exactly (1,200 / 1,200 / 1,980 / 720), so the 6,000 cCRE positions are purely additive: nothing that already worked can get worse.

- **Calibration held.** Against strong TSS-proximal liver eQTLs (GTEx v8, tissue-matched tracks), the median AlphaGenome effect percentile moved from 0.781 to **0.778** for RNA and 0.659 to **0.625** for CAGE — both inside the acceptance band, both 0% saturated. The tail widened for the peak layers without disturbing the layers that were already well calibrated.

- **Not fixed, and not claimed to be.** AlphaGenome `histone_marks` and Enformer `tf_binding` keep whatever their better component gives (20% and 25% of rows still pinned at 1.0). Both would need a *per-track* reference population — that mark's own broad domains, that factor's own ChIP peaks — which is a different design, not a different fraction.

- **Weaker evidence for three of the eight, stated as such.** Sei, LegNet and EPInformer-seq have no committed walkthrough rows and no positive set (there is no eQTL equivalent for MPRA activity or Sei sequence classes). Unlike the accessibility fix, where saturation was measured at 50% and dropped to 0%, those three are justified by tail width and by matching the assay's biology — **not** by a calibration check.

- **Three backgrounds rebuilt against a gene-anchored effect region set (AlphaGenome, Borzoi, Enformer).** The shipped effect nulls were drawn from uniformly random genomic positions, which is the wrong reference class for a TSS-localised assay: a random position has essentially no CAGE signal, so the `+1` pseudocount damps its log-ratio toward zero and the null's body sits far below where real regulatory effects live. Positions are now sampled per stratum from protein-coding annotation (GENCODE v48 basic) — 20 % within ±1 kb of a TSS, 20 % at 1–10 kb, 33 % within ±100 bp of an exon/intron boundary, 12 % elsewhere in a gene body, 15 % uniformly random. The random tail is deliberate: without near-zero mass, genuinely small effects would receive artificially *low* percentiles, the mirror of the failure being fixed. All three oracles drew from one seeded region set — each build logged an identical `tss_near 1200, tss_far 1200, junction 1980, gene_body 720, random 849` of 6,000 sampled positions — so the three are directly comparable. New `build_config` provenance is stamped into each NPZ.

  Measured against strong TSS-proximal liver eQTLs from GTEx v8 (`|slope| >= 0.5`, `maf >= 0.05`, `p <= 1e-10`), scored in tissue-matched tracks:

  | layer | eQTL percentile p50, before | after | saturated |
  |---|---|---|---|
  | RNA (232 rows, 8 tracks) | 0.899 | **0.781** | 0 % |
  | CAGE (100 rows, 4 tracks) | 0.857 | **0.659** | 0 % |

  Both moved down, which is the intended direction — the reference class now contains variants that actually perturb these assays, so a given eQTL is less extreme against it. (These are the figures after the gene-anchored rebuild. The cCRE union that followed moved them again, to 0.778 and 0.625 — see the entry above for the final values.)

- **Effect of the whole cycle on the committed examples.** Across the four AlphaGenome/Enformer variant walkthroughs, saturated rows (percentile pinned at exactly 1.0000, where the column has stopped discriminating) fell from **47 to 16** with row counts unchanged at 369 and distinct percentile values up from 280 to 284.

  Attribution, and the honest limits of it. These were measured separately and are attributable:

  | change | measured effect |
  |---|---|
  | RNA denominator: exon *intervals* → bins actually summed (#149) | numerator was overstated 251–1736×; median \|effect percentile\| 0.99+ → 0.062 on the unchanged population |
  | Enformer `effect_cdfs` grid repair (#143) | reachable percentile ceiling 0.9605 → 0.9998; the top 4 % of the scale did not previously exist |
  | Cross-process determinism (#127, #145) | a full report is now bit-exact: 603 numeric fields, 0 differing, 0 sign flips, worst relative delta 0.0 — against 454 differing fields with 36 sign flips before |
  | Gene-anchored null (this entry) | the eQTL table above |

  The per-layer walkthrough diff is a **combined** effect of all of the above plus the CHIP window classifier (#122/#146) and window-span parity (#147/#148); it is not decomposed per change, because doing so honestly would require re-running each rebuild in isolation. It is reported as a fused diff rather than split by guesswork.

- **One layer did not improve in this step.** Enformer `chromatin_accessibility` at SORT1 went from 4/12 saturated to **6/12**, median percentile 0.960 → 1.000. That was not a new regression: 0.960 *was* the padded-grid artefact ceiling (#143), so those rows were already pinned and the repair only made the pinning visible instead of disguising it as a plausible 0.96. The underlying fact was that Enformer's accessibility effect null was genuinely too narrow for a variant this strong.

  **Fixed later in the same release** by the cCRE union described in the entry above — that layer is now 0/12 saturated. The two entries are sequential, not contradictory: this one records the state after the gene-anchored rebuild, and the entry above records the state after the union.

### Fixed

- **Walkthrough TSVs silently dropped every per-gene row after the first.** `scripts/regenerate_remaining_examples.py` carried its own report flattener that de-duplicated on `(allele, assay_id, layer)` — a key omitting the region. RNA and CAGE emit one row per *gene* per track, so all but one gene were discarded: `validation/TERT_chr5_1295046` shipped 18 rows where its JSON had 99 (one `tss_activity` row where there were fifteen, one per nearby gene TSS), `discovery/SORT1_cell_type_screen` 39 of 347, `sequence_engineering/region_swap` 4 of 32, `integration_simulation` 3 of 55. The same writer also put `region_label` in a column named `description`, which already means the *track* description in `to_dict()` — one name for two things across two artefacts of the same report. Fixed by deletion rather than repair: everything now routes through `report.to_dataframe()`, the canonical writer `scripts/regenerate_examples.py` already used. All 14 walkthrough (JSON, TSV) pairs now agree on both counts and row identities, pinned by `tests/test_json_tsv_parity.py`. Long-standing rather than a regression — the counts were identical before and after the rebuild.

- **Docs overstated AlphaGenome's usable track count by 563.** Ten places across `README.md`, `docs/variant_analysis_framework.md`, `docs/MCP_WALKTHROUGH.md` and `docs/API_DOCUMENTATION.md` advertised **5,731 tracks**, including the README's headline sentence. That is the row count of AlphaGenome's metadata table; 563 of those rows are `padding` placeholders whose only purpose is keeping `local_index` aligned with the model's output array. They carry no assay, `iter_tracks()` skips them, and the shipped background has no row for any of them. The queryable count is **5,168** — verified both directions: 5,168 metadata tracks have a background row and 0 do not, and 5,168 + 563 = 5,731 exactly. `tests/test_documented_track_counts.py` now compares live-doc prose against the shipped NPZs so this cannot drift again. (An earlier entry below claims this was "disambiguated inline"; it was not, in any of the four live docs.)

- **The background grid guard blocked a healthy rebuild.** The `distinct == count` fingerprint — reported as perfect, 5,313/5,313 with zero false positives — fired on AlphaGenome `effect_cdfs` row 3966 (CHIP_TF ARID3A) and refused an 11-hour merge. That row is not padded: 913 of its 5,949 samples are exact zeros, so interpolating the remainder lands on exactly 5,949 distinct values by coincidence, and its maximum first appears at index 9998 — precisely where `np.interp` puts it, where padding would put it at 5,948. The raising condition is now the mechanical one alone (`first_max == n - 1`, unreachable by `np.interp`, whose `source_q` stops at `(n-1)/n`); `distinct == count` is a `logger.warning` that says outright it is usually coincidence. Tally before the demotion was three false positives to one true catch.

- **ChromBPNet recovers counts with `expm1`, not `exp`.** ChromBPNet's count head is trained against `log(1 + count)` (upstream `batchgen_generator.py` feeds `np.log(1+batch_cts.sum(-1, keepdims=True))` as the target), but chorus inverted it with `np.exp`, so every recovered count was high by exactly +1 — negligible at a peak (~0.1 % at 1,000 counts) but up to 100 % at a low-activity site, which is precisely the regime the activity CDFs are built from. Corrected at the three count-inversion sites: `oracles/chrombpnet.py:579` (`_transform_predictions_to_tracks`), `oracles/chrombpnet.py:802` (`predict_sliding`), and `scripts/build_backgrounds_chrombpnet.py:348` (`predict_profiles_batch`). The profile softmax `np.exp` calls at `:577`, `:801` and `:347` are a different transform and are unchanged. The bug was self-consistent — oracle and CDF builder made the same error — so ChromBPNet percentiles were internally valid, which is why it went unnoticed; raw counts and cross-oracle comparability were not. Cherimoya already did this correctly (`cherimoya_source/scoring.py`) and is unaffected. New regression suite `tests/test_chrombpnet_counts.py` covers all three sites, including an oracle/builder consistency check so the two cannot drift apart again.

## [0.5.6] — 2026-05-15

*Per-walkthrough reproduction notebooks.* No library code touched; 389 passed, 2 skipped.

### Added
- **Every walkthrough ships a `notebook.ipynb`** that reproduces the same result as the matching MCP query — 13 of them, generated by one declarative script, `scripts/generate_walkthrough_notebooks.py`, whose `WALKTHROUGHS` list holds the path, MCP tool, oracle and arguments per spec. Eight cell-template builders cover the seven distinct MCP-tool flows plus multi-oracle consolidation.
- Notebook contract: a single imports cell, no `pip install` cells, one logical step per cell, all arguments explicit, and a dedicated save cell writing `example_output.md` / `.json` / `.tsv` / HTML. Top-to-bottom execution reproduces the MCP output.
- `nbformat>=5.0` made explicit in the dev extras.

### Changed
- `use_environment=True` throughout the notebooks, so they stay in the base `chorus` kernel and each oracle delegates its model load to its own mamba env by subprocess — no notebook-level env switching.
- The causal-prioritization notebook inlines 11 LD proxies so it has no LDlink network dependency, with a commented-out `fetch_ld_variants` block for fresh proxies. The multi-oracle notebook is the only one that runs three oracles end to end (ChromBPNet → LegNet → AlphaGenome).

## [0.5.5] — 2026-05-14

*Indel and multi-allelic variant prioritization.* 389 passed, 2 skipped (+10 tests).

### Added
- **`predict_variant_effect` accepts indels everywhere.** The SNV-only gate at `chorus/core/base.py:415-424` was the only thing blocking them — `apply_variant`, `Interval.replace` and `parse_ld_response` already supported any-length swaps but never saw them. Replaced with a permissive validator backed by `normalize_allele` (`-` / `None` / case-folded ACGTN), auto-widening the internal region to fit `len(ref)`.
- **`get_centered_window` returns a `length`-bp window for any variant kind** — SNV, insertion, deletion, MNV, VCF-style anchored. For deletions it fetches extra right flank so `len(alt_seq) == length`.
- **LDlink parsing fans out `Correlated_Alleles`.** A row with `Alleles=(CT/-)` and `T=CT,A=-` now emits **two** LDVariant records, one per `SENT=PROXY` pair, both at the same coordinate and both scored. Multi-allelic alts in `(A/G,T)` fan out too. `LDVariant.kind` classifies each.
- **`snvs_only` filter** on `fetch_ld_variants`, `prioritize_causal_variants` and the MCP `fine_map_causal_variant` tool, defaulting to `False`.
- `CausalVariantScore.kind` surfaced in `to_dict()` and the markdown ranking table.

### Changed
- **Backward-compatible only if you opt in.** Callers that relied on `InvalidRegionError("…single-nucleotide variant…")` to filter indels will now have them scored instead. Migration: pass `snvs_only=True`.

## [0.5.4] — 2026-05-13

*Collaborator round 2: rsID input, LDlink config, AlphaGenome example IDs.* 379 passed, 2 skipped.

### Added
- **rsID input** in `analyze_variant_multilayer`, `discover_variant` and `discover_variant_cell_types`: a `position` starting with `rs` is resolved via LDlink (sentinel-only fetch, no full proxy lookup), with new `ldlink_token` and `genome_build` parameters. `score_variant_batch` deliberately excluded — its contract takes explicit coordinates per row for the VCF case.
- `fine_map_causal_variant` takes `ldlink_timeout` (default 30.0) and `genome_build`, which accepts `hg19` / `grch37` / `GRCh37` / `hg38` / `grch38` / `GRCh38`; it had been hardcoded to `grch38`.

### Fixed
- **Four walkthrough READMEs used display-name assay ids** (`"DNASE:HepG2"`) that raise `ValueError("Assay ID not found in metadata")` at runtime. Replaced with real AlphaGenome identifiers (`"DNASE/EFO:0001187 DNase-seq/."`) plus an inline comment showing the `metadata.search_tracks()` lookup.

### Known limitations
- **AlphaGenome normalization is over-optimistic**, confirmed by direct CDF inspection: at effect 0.05 the mean percentile is already 0.896 (ATAC), 0.884 (DNASE), 0.720 (CHIP_HISTONE). Deferred to #83 with three implementation options rather than patched — it is the problem the 0.6.0 region re-anchoring exists to address.

## [0.5.3] — 2026-05-12

*Collaborator-audit followups.* Eight findings plus a ranking sweep; two re-shaped on verification.

### Added
- `TrackScore.low_effective_bins` — a diagnostic flag when scoring window / native bin resolution < 8. Fires on AlphaGenome CHIP-TF (501 bp / 128 bp ≈ 4 bins). **No window change**, deliberately: widening `tf_binding` would invalidate every published CDF *and* dilute narrow TF footprints, so the quantization risk is surfaced instead of silently traded away.
- `AlphaGenomeMetadata.iter_tracks()`, a public iterator excluding padding rows; `_tracks` documented as internal (kept for output-array index alignment).
- `chorus.oracles.bpnet` exposing `load_bpnet_model`, `encode_sequence`, `predict_bpnet`, encapsulating the `sys.path` + `BPNet.arch` + `tasks.json` + bias-tensor recipe.
- `chorus.analysis.normalization.is_ready_for_oracle(name)`, unified across both normalizer layouts and honouring `_CDF_ALIASES`.
- `chorus.utils.get_centered_window(...)` — 1-based-safe centred ref/alt window with strict ref-base validation.

### Changed
- `discover_variant` / `discover_variant_effects` / `_score_all_tracks` / `_rank_and_select` / `_rank_cell_types` all default to `ranking_metric="alt_x_abs_effect"`, matching `discover_variant_cell_types`. Ranking outputs now carry `ref_value`, `alt_value`, `ranking_score`, `ranking_metric` and `low_baseline_warning`.

### Fixed
- **A wrong-class normalizer failed silently.** `build_variant_report`'s type hint said `QuantileNormalizer | None` when `PerTrackNormalizer` is what IGV rescale requires. Now widened, raising `TypeError` on anything else and warning clearly when a `QuantileNormalizer` is passed (tables score, IGV will not).
- **MCP `analyze_variant_multilayer` crashed with `StopIteration`.** The docstring invited `assay_ids=[]` to mean "all tracks" but `_predict()` only handled `None`, so an empty list gave empty predictions and `next(iter(...))` blew up. Fixed in all three multi-track oracles; new `EmptyPredictionsError` replaces the opaque failure.
- **`VariantReport.to_html()` silently stripped IGV above 50 tracks.** Now truncates to the top 50 by `|effect|` with a warning and an HTML callout; the table still shows every track.

## [0.5.2] — 2026-05-11

*Test-only release closing #81 (AlphaGenome JAX vs PyTorch equivalence).* No production code changed. 379 passed, 2 skipped.

### Fixed
- **The equivalence test's absolute bound was measuring bf16 quantization, not model drift.** `max(|pt - jax|) < 0.1` failed on Mac Metal (0.9963) and Linux/CUDA (0.4760), but per-track Pearson correlation is **1.0000** and mean absolute difference over 1M positions × 3 tracks is 0.0005–0.0008. JAX runs `params=float32,compute=bfloat16,output=bfloat16`; at signal magnitude ~55 adjacent bf16 values are 0.25 apart, so ~0.4 peak differences are intrinsic. The PR #62 baseline of <0.05 had been measured on a lower-magnitude window — the peaks grew, the model did not drift. Replaced with Pearson > 0.99, `max(|Δ|)/peak < 2%`, and `mean(|Δ|)/mean(|jax|) < 5%`.

## [0.5.1] — 2026-05-10

*Patch: three findings from the v0.5.0 scorched-earth audit.* 375 passed on macOS Metal.

### Fixed
- **ChromBPNet wide-window `predict` crashed in the base env** with `ModuleNotFoundError: No module named 'tensorflow'` — two of the three shipped notebooks hit it. The v0.5.0 auto-route to `predict_sliding` was the cause; the env-runner template path was already correct. Also fixed a parallel bug in `_predict_direct` that used `// sequence_length` where it needed `// output_length`, undersizing `one_hot` and crashing on inputs above 2,114 bp.
- **`chorus setup --oracle alphagenome_pt` logged a 404 and then "✓ ready".** It aliases to AlphaGenome's CDFs at lookup time, but `download_pertrack_backgrounds` still tried to fetch a separate NPZ. Now short-circuits for anything in `_CDF_ALIASES` with an alias-aware message.
- `chorus backgrounds status` omitted `alphagenome_pt` from its hardcoded oracle list.

### Known limitations
- AlphaGenome JAX vs PyTorch backend drift at SORT1 confirmed pre-existing (`git log v0.4.0..v0.5.0 -- alphagenome*` is empty) and platform-amplified: 0.476 max absolute difference on Linux/CUDA, 0.996 on macOS Metal. Diagnosed and closed in 0.5.2 as bf16 quantization.

## [0.5.0] — 2026-05-10

*Unified track rescale and a DHS-augmented ChromBPNet CDF.* 376 tests pass cold; fresh-install end-to-end verified from a wiped machine (~33 min); 18 walkthrough HTMLs inspected.

### Added
- **One rescale helper for every rendering path.** `chorus.analysis._igv_report.rescale_for_display()` now drives IGV, matplotlib, CoolBox and the notebooks, so `track.get_coolbox_representation()` with no arguments gives CDF-rescaled output; `normalize=False` opts out. *(This default is what later made a low-signal synthetic-sequence panel render blank on a 0–3 axis — see 0.7.0.)*
- **Symmetric signed rescale** for Borzoi RNA, Sei and LentiMPRA: negative values render on `[-3, +3]` against `p99(|cdf|)` instead of clipping to 0.
- **`predict_sliding` for ChromBPNet**, so the multi-oracle IGV panel shows it across AlphaGenome's full 1 Mb window rather than a 0.2% stripe.
- **Max-pooling for high-resolution oracles** (ChromBPNet, LegNet) so 1 bp peaks survive zoom-out — carries forward and corrects PR #79. *(Keyed on oracle name, which is how Cherimoya was later missed; fixed in 0.7.0.)*
- **DHS vocabulary mirrored to HuggingFace** with auto-fetch in `load_dhs_vocabulary()`, removing the `gdown` step for anyone rebuilding CDFs.
- Per-layer CDF sampling guide in `docs/NORMALIZATION_GUIDE.md`.

### Changed
- **ChromBPNet CDF rebuilt DHS-augmented** (786 tracks at the time): ~10K SNPs at random offsets within ±150 bp of Meuleman 2020 DHS summits, making percentiles more discriminating for cell-type-specific peaks.
- Per-layer display floors lowered so a peak's base and shoulder stay visible: `chromatin_accessibility` 0.95 → 0.90, `promoter_activity` 0.95 → 0.85.

### Fixed
- `OraclePrediction.add()` backfills `track.assay_id` from the dict key, fixing silent `None` assay_ids on ChromBPNet.
- `is_signed()` and `_match_track_id()` share fuzzy matching including CHIP `:+`/`:-` strand-suffix stripping, so `LentiMPRA:HepG2` resolves.
- `_predict()` auto-routes wide queries to `predict_sliding`, also fixing a pre-existing `IndexError` in `_predict_direct`'s sliding formula.

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

---

[Unreleased]: https://github.com/pinellolab/chorus/compare/v0.7.3...HEAD
[0.7.3]: https://github.com/pinellolab/chorus/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/pinellolab/chorus/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/pinellolab/chorus/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/pinellolab/chorus/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/pinellolab/chorus/compare/v0.5.6...v0.6.0
[0.5.6]: https://github.com/pinellolab/chorus/compare/v0.5.5...v0.5.6
[0.5.5]: https://github.com/pinellolab/chorus/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/pinellolab/chorus/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/pinellolab/chorus/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/pinellolab/chorus/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/pinellolab/chorus/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/pinellolab/chorus/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pinellolab/chorus/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pinellolab/chorus/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/pinellolab/chorus/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/pinellolab/chorus/releases/tag/v0.2.0
