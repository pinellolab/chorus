# 2026-08-06 — Fixing the background null model, and ten defects found on the way

Scope: the per-track background nulls (`effect_cdfs`, `summary_cdfs`, `perbin_cdfs`)
that every percentile in chorus is computed against. Prompted by a user asking for a
plan to "fix this null model for good" after three earlier partial fixes.

## The headline defect: the sampler discarded the tail it exists to measure

`percentile = min(rank/denominator, 1.0)` clamps once an effect passes the largest
sampled background value. That had been patched three times — re-anchoring on gene/cCRE
features, union-at-2N, and a read-side exceedance ratio. None was the cause.

`scripts/merge_effect_shards.py` called `ReservoirSampler.from_flat_samples(*parts)`
with **no capacity**, silently inheriting `DEFAULT_CAPACITY = 50_000`. Every AlphaGenome
`gene_expression` track offers **148,367** effect values, so the union was subsampled
2.97×. A uniform *m*-of-*N* subsample retains the population maximum with probability
exactly *m/N*: 50,000/148,367 = **0.3370**, and **33.9%** of the 667 RNA rows were
measured to have kept theirs — the mechanism confirmed to three digits.

| AlphaGenome effect null | tracks | max ratio (true/shipped) | p99 ratio |
|---|---|---|---|
| `gene_expression` | 667 | median **1.332**, p90 3.18, worst **8.34** | 1.006 |
| every other layer | 4,501 | 1.0000 | 1.0000 |

**The tail was wrong by up to 8.3× while p99 was right to 0.02%.** That asymmetry is why
it survived every calibration gate: reservoir sampling is unbiased for the *body*, and
every gate measures the body.

Fixed by exact retention — **no GPU**, because the 8 shards still held every offered
value. 23 seconds. Verified against the raw samples: all 4,501 never-thinned rows came
back bit-identical, and the release gates were unchanged (CAGE 0.6250, RNA 0.7770).

Scope was wider than first measured: **9 of 19 (oracle, layer) reservoir pairs were
thinned** — `perbin` 16–43× on five oracles, `summary` up to 5.2× on three. An initial
claim that only AlphaGenome was affected was wrong; only `effect_counts` had been checked.

## Why no guard caught it

`cdf_grid_violations` is handed the **offered** count while the geometry it validates is
set by the **retained** count, and its first act is `if n >= n_points: continue`. Offered
is always ≥ 10,000 in a real build, so it skipped every thinned row **by construction** —
while its docstring promises it "refuses to write a CDF matrix that could not have been
produced by `to_cdf_matrix`".

`thinning_violations` is now a separate, independent check (geometry and retention are
different questions, and the early return that makes the geometry check correct is what
makes it blind here), wired into both `build_and_save` and the merge script. Retention
is now persisted into the shipped file, so the question is answerable from a published
artefact — previously only the offered count was stored.

## DHS: a user-directed change, measured and reversed

The user proposed sampling the Meuleman DHS index, reasoning that DHS regions contain TF
footprints. Sound a priori. Three Sei builds, differing only as labelled:

| | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| **+DHS, additive** (the proposal) | 0.971 | 0.937 | 0.954 | 0.936 | **1.000** |
| same budget, more cCRE + gene | 1.035 | 1.030 | 1.042 | 0.992 | **1.261** |

`max = 1.000` is decisive: across all 40 tracks **not one DHS position produced a larger
effect than the best cCRE- or gene-anchored position already in the set.** DHS added
nothing to the ceiling while lowering every quantile.

Tested again where the idea was actually aimed, because Sei outputs chromatin-state
classes rather than TF ChIP tracks. Enformer, n=6,000 per arm, +DHS / no-DHS:

| layer | n | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| chromatin_accessibility | 684 | 0.942 | 0.980 | 0.972 | 0.947 |
| histone_marks | 1890 | 0.953 | 0.917 | 0.916 | 0.931 |
| **tf_binding** | **2101** | **0.904** | **0.858** | 0.888 | 0.953 |
| tss_activity | 638 | 0.747 | 0.822 | 0.760 | 0.810 |

`tf_binding` — the layer that saturates, and the target of the proposal — is diluted
**most of any layer**; 744 of 2,101 tracks gained a ceiling, 1,217 lost one. The likely
reason is redundancy: the SCREEN cCRE catalogue already carries `CA-TF`, `CA-CTCF`,
`CA-H3K4me3` and `TF`, so DHS summits largely duplicated positions already sampled while
skewing distal (3.6% TSS-proximal, median 68.7 kb). Redundant draws dilute a mixture
without extending it.

**An argument of mine was also wrong.** I claimed DHS could not hurt because
`max(union) = max(max_a, max_b)`. That protects the *maximum* and nothing else — a
percentile is a quantile of the *mixture*, so adding smaller-effect positions lowers the
whole upper body. On LegNet's promoter null it diluted p50/p90/p99 by 8–19% on all three
cell types. The pre-existing comment at `annotations.py:1288` ("right family, wrong
member") was correct.

What does work is more positions from the populations already in use: **n = 12,000 →
18,000**, which bought a 26–31% higher ceiling with the body unmoved.

## Seven further defects

1. **Sei was entirely dark.** `sei.py` set `assay_type = info.name` for its sequence
   classes; `classify_track_layer` expects the literal `"sequence-class"`, returned
   `"other"`, whose `LAYER_CONFIGS` entry is `None`, so **all 40 tracks scored
   `raw_score=None`**. A built, verified, zero-degeneracy null no query could reach; its
   absence from every example read as "we didn't include Sei". One line.

2. **Anchored positions were clamped onto contig margins.** `usable` selects whole
   *chromosomes* long enough for the margin, but the tss/junction/gene-body populations
   were filtered only by `chrom in usable`. 2,515 of 20,083 PC TSS (12.5%) sit within
   5 Mb of a contig end and were clamped onto the boundary — up to 5 Mb from the TSS they
   were labelled as within 1 kb of. Before: 12.1–14.6% of each stratum on a boundary,
   5,265 of 6,000 positions distinct, `chr16:5,000,000` appearing **64 times**. After:
   0.0–0.1%, 6,000/6,000 distinct. Duplicates give identical effect values, padding the
   sample count and manufacturing the tied CDF runs `_rank_with_tie_breaking` exists to
   compensate for. **This affected the shipped backgrounds.**

3. **A stratum landmine.** The dispatch ended in a bare `else` drawing a uniformly random
   position, doubling as the `random` handler, the empty-pool fallback *and* the
   catch-all for unknown names. Adding `"dhs"` without a branch would have shipped 6,000
   uniformly random positions tagged, tallied and stamped as DHS, with entirely plausible
   numbers. Both samplers now raise; the decisive guard is an annotation round-trip
   (≥99% of each stratum verified against the annotation it names) — which is what found
   defect 2.

4. **Builders overwrote `CUDA_VISIBLE_DEVICES`** with their `--gpu` default. Two ablation
   arms landed on one GPU; the first took 78 GB, the second could not allocate a cuBLAS
   handle, and all 5,968 positions failed. A fleet rebuild sharded by env var would have
   silently serialised.

5. **A build where every position failed still wrote a valid background** — 5,313 tracks,
   all-zero rows, all counts 0. It merges cleanly, and `_has_samples` then suppresses
   those tracks, so the symptom is an oracle that silently stops ranking.

6. **Merges exited 0 having done nothing** — six builders did
   `logger.error("Missing interim files"); return`. A driver keying off exit codes
   recorded success for steps that wrote nothing.

7. **A per-track loader warned 1,518 times instead of stopping.** Cherimoya ran 75
   minutes loading nothing in an env without the `cherimoya` package.

8. **All 8 builders and 5 scripts hardcoded `~/.chorus/backgrounds`**, so a chorus
   installed with `CHORUS_DATA_DIR=/data/...` still wrote every background it built into
   the home directory the data dir exists to avoid. The guard for that defect scanned
   only `chorus/`, never `scripts/` — the same half-fix shape it was written for.

Items 5, 6, 7 and the two guards above are one pattern: **work reporting progress while
failing.** Every one was found by *running* something small, never by reading code.

## Deliberately not done

- **Vectorising Algorithm R.** The plan estimated ~11 GPU-hours saved, derived by
  dividing a baseline pass's wall-clock by its sample count. Benchmarked directly the
  reservoir runs at 2.4M values/s — ~19% of a pass — so the saving is under an hour
  across the fleet, while changing every retained sample and invalidating reproducibility
  of any background not rebuilt.
- **Extrapolating percentiles past the data.** A GPD fit overshoots the far tail by
  **3.8×**, an exponential undershoots by **0.27×**, and the plain empirical maximum is
  within **13%**. So the fix is to make the empirical ceiling correct, not to model past
  it. `effect_exceedance` reports the ratio to that ceiling instead.
- **A motif-aware ChIP null.** Motif-creation saturation is irreducible with an empirical
  ceiling: no null over random positions contains many single-base changes that complete
  a specific factor's motif. Confirmed by measurement — the DHS-anchored null ChromBPNet
  has always used still pins on rs12740374 CEBPA at 1.11×. The only statistically valid
  route is importance sampling with reweighting, which needs per-TF PWM proposals.

## Verification

`scripts/verify_rebuilt_backgrounds.py` gates every swap and exits non-zero. Its first
run refused LegNet on "ceiling fell to 0.88×" — and **the gate was wrong, not the
build**: the ceiling is one extreme order statistic per track, so a median over 3 tracks
cannot support a pass/fail, and "adding positions can only raise the ceiling" ignores
that this rebuild also *removed* the 12% clamped ones. Now track-count aware, gating p99
instead.

Verified at the time of writing (staged, not yet swapped):

| oracle | p50 | p90 | p99 | max | retention |
|---|---|---|---|---|---|
| sei | 1.021 | 1.014 | 1.031 | **1.308** | exact, 0 thinned |
| epinformerseq | 1.032 | 1.021 | 1.038 | 0.931 | exact, 0 thinned |
| legnet | 0.993 | 0.996 | 0.930 | 0.881 | exact (3 tracks, advisory) |

Tests: **1,220 fast tests pass.** Every new guard was checked to fail on the pre-fix
code. Also added: 10 tests for `score_ism` (previously zero, and a published vignette
rests on it) and 9 for the oracle-free MCP tools.

## Still open

- The fleet rebuild is running; AlphaGenome, Borzoi, Enformer, ChromBPNet and Cherimoya
  are not yet verified or swapped.
- Regeneration of committed artefacts, the CHANGELOG before/after table, and the HF
  republish — the last on hold pending user review.
- AlphaGenome `histone_marks` and Enformer `tf_binding` remain ~20%/25% pinned. DHS does
  not fix it; a per-track reference population would be a different design.
- `oracle_status`, `score_prediction_region`, `score_variant_effect_at_region` remain
  untested (all need a loaded model).
- Two HF tokens exposed in plaintext earlier in this work still need rotating.

## Addendum: the builder is bit-reproducible, verified accidentally

Cherimoya was rebuilt twice — once with `--n-variants 18000` (my error: that is the
gene-anchored oracles' setting, and cherimoya uses random ∪ DHS-summit, so it shifted the
composition from 50:50 to 64:36) and once with its shipped defaults.

| statistic | diluted (n=18k) | correct (defaults) |
|---|---|---|
| p50 | 0.963 | **1.000** |
| p90 | 0.962 | **1.000** |
| p99 | 0.983 | **1.000** |
| max | 1.000 | **1.000** |

The correct build reproduces the shipped effect null **bit-identically on all 1,518
rows**, with identical offered counts. Two things follow, neither of which was the point
of the exercise:

1. **The builder is deterministic end to end** — region sampling, GPU forward passes,
   reservoir, gridding and NPZ write all reproduce exactly across a 75-minute run on a
   different day. That is a stronger reproducibility statement than any test in the
   suite makes, and it is worth keeping: a future rebuild of an oracle whose inputs have
   not changed should be bit-identical, and if it is not, something moved that nobody
   declared.

2. **My composition flag was the sole cause of the 3.7% narrowing**, not the retention
   change, not the margin fix, not the sampler rewrite. Isolating that took one rebuild
   because the two runs differed in exactly one flag.

The same reflex produced both of this cycle's wrong calls: adding DHS to LegNet's
promoter null, and applying n=18,000 to cherimoya. Both are "treat the oracles as
interchangeable", and both were caught only by measuring per-oracle rather than reasoning
about the fleet.


## Addendum 2: the mechanism, confirmed across six layers

A uniform *m*-of-*N* subsample retains the population maximum with probability exactly
*m/N*, so **1 − m/N** predicts the share of tracks whose ceiling exact retention should
raise. Measured across every thinned (oracle, layer) pair, spanning a 30-fold range of
thinning:

| oracle.layer | thinning | 1 − *m/N* predicts | measured |
|---|---|---|---|
| chrombpnet.summary | 1.4× | 26.5% | **25.9%** |
| borzoi.summary | 1.5× | 33.4% | **34.2%** |
| borzoi.perbin | 19.8× | 95.0% | **95.0%** |
| enformer.perbin | 19.8× | 95.0% | **95.3%** |
| cherimoya.perbin | 21.8× | 95.4% | **96.8%** |
| chrombpnet.perbin | 43.5× | 97.7% | **97.5%** |

Worst deviation 1.4 points, over predictions ranging from 26% to 98%. Two controls make
this more than a curve fit:

* **0 of borzoi's 6,068 unthinned summary tracks changed ceiling.** The fix moved exactly
  the tracks the arithmetic says were affected and nothing else.
* The body is untouched throughout — p50 and p90 ratios of 1.000 — which is what
  distinguishes "restored a lost tail" from "changed the distribution".

This is the strongest evidence in the effort that the diagnosis was right and the fix does
what it claims. It also retrospectively explains the instability that started the whole
investigation: after re-anchoring, 12 of 12 Enformer `tf_binding` tracks gained a wider
p99 while 11 of 12 reported a *lower* maximum. Judged on the ceiling that looked like a
regression. The ceiling was a coin flip weighted by *m/N*.

Generated by `scripts/rebuild_before_after_table.py`, which reads the artefacts rather
than transcribing numbers, so it can be re-run as each oracle lands.

---

# Addendum C — post-swap verification sweep (2026-08-07)

Everything above concerns the rebuild itself. This records the audit run *after* the swap,
covering the checklist sections that need the live artefacts in place, and the seven defects
it found. Five were in this cycle's own work; two were long-standing (the `list_tracks`
truncation and the test-isolation bug).

## What was verified

| checklist § | check | result |
|---|---|---|
| §4 | CDF integrity, all 8 oracles, all 20,439 rows | pass |
| §5 | normalizer API loads all 8 | pass |
| §6 | all 6 notebooks executed on GPU | **0 errors, 0 pinned rows** |
| §7 | 18 shipped HTML reports vs sibling JSON/TSV | consistent in all 21 directories |
| §10 | doc claims vs live artefacts | 2 defects, both fixed |
| §13 | cross-process determinism | **603 fields, 0 differing, 0 sign flips** |
| tests | full suite, quiet machine | see below |

**§6 detail.** `rc=0` is not sufficient evidence a notebook ran: a cherimoya notebook in
the wrong env logs `Failed to load <track>` per track and exits 0. So each executed
notebook was re-opened and checked for error outputs, suspicious strings, and whether an
oracle was actually loaded. All six genuinely predicted, across 7 per-oracle envs.
`comprehensive_oracle_showcase` loads **sei** — the first user-facing artefact in which
Sei's 40 tracks resolve, since they were dead at the query path until `sei.py:448`.

Prerequisite worth recording: `chorus setup` does not register the `chorus` kernelspec, so
every `nbconvert` dies with `NoSuchKernel` until
`python -m ipykernel install --user --name chorus` is run.

**§13 detail.** The gate must run under `chorus-alphagenome`; run in the base env it fails
with `No module named 'jax'`, which looks like a determinism failure and is not one.

## Defects found

**1. Seven walkthrough directories were still shipping pre-swap percentiles.** The earlier
regeneration covered only what `regenerate_examples.py` owns; `batch_scoring`,
`causal_prioritization/*`, `discovery/*`, `sequence_engineering/*`,
`validation/SORT1_rs12740374_multioracle` and `validation/TERT_chr5_1295046` belong to
`regenerate_remaining_examples.py` and `regenerate_multioracle.py`. All regenerated.

Detecting this needed a *dated* comparison, not a diff: HTML and its sibling JSON agreed
inside every directory, because both were equally stale. Three files then came out
byte-identical, which is also correct — the two causal `example_output.tsv` and
`discovery_summary.json` carry no percentile column, and the only JSON fields that moved
were `quantile_score` and the timestamp.

**2. Three reservoir tests were diffing two different builds.** They re-union interim
shards from the backgrounds directory and compare against the shipped rows. The shards
there are leftovers from the *previous* build, so the failure read as corruption — "ships a
ceiling of 0.800 but the true maximum over all 148367 offered values is 0.875" — when
148,367 is simply the old offered count against a shipped 225,253. Now checks build
identity via the stamped `effect_counts` first.

Two premises of one of those tests had also expired: it recorded borzoi and enformer as
"never thinned and deliberately NOT rebuilt", and **borzoi's effect layer is now past
capacity** (34,482 → 51,831 offered against a 50,000 cap). It is the only shipped effect
layer that exercises the exact-retention path, so it is now an explicit canary against the
retention assertions becoming vacuous — enformer, at 17,907, cannot serve that role.

**3. 24 stale interim shards, 2.4 GB, sitting in the live backgrounds directory.** Beyond
the test noise this is an active hazard: `merge_effect_shards.py` once defaulted to that
directory, found 8 stale `of8` shards and merged those instead of the new ones. Quarantined
to `pre_unified_rebuild/stale_effect_shards/` with a README recording offered counts per
build. ChromBPNet's two shards were deliberately **left** — different artefact
(track-sharded pre-built CDFs for the 744 BPNet/CHIP tracks), and their counts agree with
the shipped file on all 744 common tracks. A blanket sweep would have discarded a valid
artefact.

**4. `list_tracks` returned 200 of 1,504 tracks with nothing saying so.** Long-standing,
not from this cycle, and found only because verifying a doc claim gave 200 for two
different queries. `num_results` carried the truth in a sibling field; `tracks` carried a
silent 13% sample. For an MCP tool the consumer is usually a model, so the failure is an
agent concluding a track does not exist. Now always carries `showing`/`truncated`, plus a
`note` when rows were dropped. Same shape as the reservoir defect — a subsample presented
as the population — and the same remedy.

**5. `SORT1_locus/README.md` quoted percentile-scale numbers as effect sizes.** Four
per-track values in the 0.18–0.45 range where the real log2FC are +3.316 / +1.502 / +1.334
/ +1.251, and CEBPA named as top TF where the artefact has CEBPB. Second instance of this
exact confusion (the C/EBP README was the first). Caught by
`test_walkthrough_readmes_match_artefacts.py`.

**6. CHANGELOG arithmetic.** The gene-anchored family was given as 18,145 tracks; it is
18,165 (5,168 + 5,313 + 7,611 + 40 + 33). Not staleness — track counts did not move in
this rebuild.

## Two claims of my own that were wrong, and are corrected in place

**The mechanism check over-claimed.** `docs/BACKGROUND_NULL_PROTOCOL.md` §10 said 1 − *m/N*
was confirmed across "every thinned layer". It was not. AlphaGenome's position count grew
in the same rebuild, so retention is not the only variable for its three layers, and they
deviate accordingly — 70.8% measured against 86.5% predicted for `effect`, 97.0% against
80.8% for `summary`. They are now explicitly excluded, because quoting them as agreement
would be dishonest and quoting them as disagreement equally wrong. The six layers whose
population is unchanged are the evidence, and they agree to within 1.4 points.

**A near-miss regression report.** Global distinct AlphaGenome percentile values fell
257 → 215 across the committed corpus, which looks like lost resolution and was very nearly
written up as one. It is not. Two hypotheses were tested and refuted before the right
measurement was found:

* *CDF plateaus from exact retention?* No — the new rows are **finer** (median 10,000
  distinct values per row vs 9,845; longest plateau 1 vs 5).
* *The CAGE null narrowed?* No — it **widened** (p50 1.042, p99 1.061, body width 1.061).

The decisive measurement is per track, because percentiles are only comparable within a
track: **within-track collisions fell 25 → 22, with 0 rank inversions across 3,448
comparable pairs**, on 292 rows whose raw scores are bit-identical. The global drop is
cross-track collisions at 4 decimal places, which carry no meaning.

Consequence for the release notes: **"distinct percentile values" is not a resolution
metric** and should not be quoted as one. An existing CHANGELOG entry does ("up from 280 to
284"); it is left as written because it is accurate for what it measured, but within-track
ties and rank inversions are what the metric should be.

## Test result

Full suite on a quiet machine after the fixes. One test is a known flake under concurrent
GPU load and is documented as such rather than having its tolerance loosened:
`test_cherimoya_integration.py::test_predict_matches_direct_window_scoring` showed 980/1000
elements mismatched at max **relative** 2.3e-3 while four other GPUs sat at 99%, and passes
3/3 quiet. The device assert passed, so it is not the CPU fallback — consistent with Triton
autotuning under occupancy pressure. Its tolerance stays at kernel-agreement strictness,
because that is the only setting at which it can detect the path divergence it exists for.

**Correction to the paragraph above, which I got wrong the first time.** I wrote that seven
of the twelve failures in the first sweep were GPU contention. They were not. They were
**order dependence**, and they are defect 7 below. The box was in fact shared with another
user's 4-way DDP job, which made "contention" a plausible-looking explanation — and
plausible-looking explanations for order-dependent failures are how they survive. The only
failure genuinely attributable to load is the cherimoya one.

**7. `test_mcp.py` left the MCP state singleton holding `reference_fasta=None`.** Seven
tests in `test_mcp_scoring_tools.py` failed in the full sweep and passed in isolation with

    ValueError: Reference FASTA required for genomic coordinates.

`OracleStateManager` is a singleton and resolves the genome exactly once, in `__init__`.
Seven tests in `TestOracleStateManager` need a manager built under a mocked
`GenomeManager`, so each sets `_instance = None` and reconstructs inside the patch, where
`is_genome_downloaded()` returns False. The fresh singleton takes `_reference_fasta = None`
and **keeps it after the patch lifts** — nothing restored it, because nothing saved it.
That field is what the state manager passes to an oracle as `reference_fasta`.

Located by bisecting the file list: `test_mcp.py` alone reproduces all seven, and the four
other candidates in the same range do not. Fixed with an autouse snapshot/restore fixture
in a new `tests/conftest.py` — restoring the same object rather than resetting to None, so
module-scoped fixtures that loaded an oracle into the singleton keep seeing it instead of
reloading a model per test. Three tests guard the fixture and were verified to fail
without it.

**Release gates should still be run on an unloaded GPU** — the cherimoya flake is real —
but "the machine was busy" must not be the first hypothesis for a failure that reproduces
deterministically under a fixed test order.

---

# Addendum D — visual inspection of the rendered artefacts (2026-08-08)

Addendum C recorded §7 as "18 shipped HTML reports vs sibling JSON/TSV: consistent in all
21 directories". That was true and it was not a visual check. It compared *text*: file
dates, greps for a clamped `1.0000`, NaN counts, plot-marker counts. It could not have told
you whether a chart was blank, mislabelled, or misleading. Prompted to actually look, I
rendered all 18 reports in Chromium via Playwright and extracted all 41 notebook figures.

## One real finding, fixed

**127 committed rows rendered as a single `≥99th` bin, hiding 81 distinct values.**

`_fmt_percentile` bucketed everything at or above 0.99. Its docstring gives the rationale,
and the rationale was **correct for the regime it was written in**: past the null's
maximum the percentile is clamped, so an effect 1.11× beyond the ceiling is arithmetically
identical to one 10× beyond, and only the exceedance ratio can separate them. More decimal
places there would be fabricated precision.

The rebuild moved these rows out of that regime. With exact retention the C/EBP effects sit
*inside* support: CEBPA 0.9998, CEBPG 0.9997, CEBPB 0.9995 — real, orderable ranks. The
escape valve (`(N× null max)`) only fires when an effect is genuinely past the ceiling, so
with nothing past it, all five C/EBP rows rendered as an identical bare `≥99th`. The
walkthrough README explains at length that CEBPA outranks CEBPB on a *smaller* raw effect;
the report contradicted it.

So the release's headline benefit was invisible in the artefact users open, while being
present in the JSON all along. The rule is now "bucket exactly when the number is not
real" rather than "bucket the ends of the scale": four decimals in the tails, where
percentiles bunch and two decimals cannot separate anything; two in the body; bucket plus
ratio only when clamped.

Worth stating plainly because it generalises: **a display policy tuned to a broken
statistic becomes wrong when the statistic is fixed, and nothing fails.** No test caught
this — the tests asserted the bucketing, so they encoded the old regime as the contract.

## A second display bug, found by fixing the first

Signed layers span [-1, 1] -- the sign is direction, the magnitude is how unusual -- so the
old `q <= 0.01` test captured **the entire negative half of every signed layer**. The C/EBP
vignette rendered nine `gene_expression` rows as `≤1st` whose real percentiles were
**-0.7374 to -0.9634**: moderately to strongly down-regulated, nowhere near the bottom
percentile. Three rows above them sat a `≥99th`. Read together that describes a variant
which both strongly represses transcription and is indistinguishable from noise.

It surfaced only because the first fix changed those cells, and the diff was reviewed
rather than assumed. `_fmt_percentile` now takes the `layer` and tests `|q| >= 0.99` for
signed layers while keeping both ends as tails for unsigned ones -- necessary because the
same number needs opposite treatment depending on the layer, and the function cannot tell
-0.74 (signed, mid-body) from 0.005 (unsigned, genuine low tail) without being told.

Long-standing, and it survived because **no test ever passed a negative percentile**. Two
test classes covered this helper and both used unsigned values only.

One further bug of my own, caught by a test I had just written: my first version chose the
clamp label with `q >= 0`, which is correct for a signed layer (clamps at ±1) and wrong for
an unsigned one (clamps at 0 and 1), so an unsigned bottom-clamp reported `≥99th`. The
midpoint has to depend on the range.

## One false alarm, and how it was caught

I reported that the embedded IGV panel renders nothing — ~1,180 px blank under a heading
promising an interactive browser, `#igv-div` with 0 children, console logging "IGV browser
created successfully", reproduced across cold and warm profiles, 60 s waits, and in **both**
`chrome-headless-shell` and the full Chromium build after installing the missing system
libraries. All 14 external fetches (igv.org genome metadata, UCSC `hg38.2bit`, 7.3 MB
`ncbiRefSeq.txt.gz`) returned 200/206 with real payloads.

**It renders correctly in a real browser.** The user downloaded the multi-oracle report and
saw the tracks. The finding was an artefact of this headless environment — no display,
software GL, possibly proxied network — not of the report.

Recorded rather than deleted, for two reasons. First, so nobody spends the same hours
re-deriving it: headless Chromium on this box is not a valid oracle for whether igv.js
renders, however many ways you probe it. Second, because the shape of the mistake is worth
remembering — I had a reproducible, multi-configuration, zero-error negative result and it
was still wrong, because every configuration shared the one confound I could not vary from
inside the box. The check that settled it took someone opening the file.

The one substantive thing that survives: the reports **do** require live network access to
`igv.org` and `hgdownload.soe.ucsc.edu` for the genome and RefSeq track. `igv.js` itself is
inlined, and the code comment "no network needed" refers only to that script — not to the
browser as a whole. Relevant to §15 (offline), where a user without internet gets tables
but no genome browser.

## Two notebook figures that are honest but read wrong

Not fixed; flagged for a presentation pass rather than silently accepted.

* `comprehensive_oracle_showcase`, GATA1 fold-change bar chart: three bars at exactly 1.0
  on a 0→1.0 axis, drawn in alarm-red, with the dashed "No change" line sitting exactly at
  the bar tops. Reaching the top of the chart *means no effect*. A reader skimming sees
  three maxed-out red bars and concludes the opposite of the data. Centring the axis on 1.0
  (or stating "no detectable change") would fix it.
* `advanced_multi_oracle_analysis`, four stacked signal panels (G/A/C/T) on a fixed 0–8
  axis: real signal peaks at roughly 1/8 of the range, three of four panels read as flat
  lines, about 85% of the canvas is white, the data stops short of the axis extent, and
  there is no title, y-label or legend.

Both are pre-existing and neither affects a number. They are the kind of thing only looking
finds, which is the point of this addendum.

---

# Addendum E — why two oracles on the same ENCODE experiment disagreed by 33% (2026-08-08)

Prompted by a user question after Cherimoya was added to the SORT1 multi-oracle
walkthrough: "why cherimoya and chrombnet different, they should give very similar
results". A fair challenge — they are the same assay, the same biosample and, as it turns
out, the same ENCODE experiment.

Investigated with five independent hypothesis agents, one synthesis, and three adversarial
verifiers under different lenses. **0 of 3 lenses refuted the conclusion**, and each
reproduced the decisive measurement from scratch rather than reusing the diagnostics.

## Answer: not a chorus defect

| oracle | ref | alt | linear ratio | log2FC | percentile |
|---|---|---|---|---|---|
| Cherimoya | 603.3 | 2093.2 | 3.469 | +1.793 | 0.9999 |
| ChromBPNet | 287.2 | 746.9 | 2.600 | +1.376 | 0.9995 |
| AlphaGenome | 660.2 | 1666.3 | 2.524 | +1.334 | 0.9964 |

Ruled out, each with a measurement:

* **Sequence construction** — REFUTED. Byte-identical model input, one md5 for both
  2,114 bp windows, variant at 0-based index 1057 in both, forward strand, single
  substitution. Neutralising the two off-by-ones found (below) moves the ratio by ≤3e-4.
* **Window/offset mismatch** — REFUTED as stated. Both oracles go through one
  `LAYER_CONFIGS['chromatin_accessibility']` and one call site, integrating the identical
  `values[808:1309]`.
* **Different data or fold** — REFUTED. Both resolve to ENCODE `ENCSR149XIL`
  (ChromBPNet's mirror manifest → `model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5`;
  Cherimoya → `models/ENCSR149XIL/cherimoya.fold_0.torch`), and the two projects' fold-0
  chromosome partitions match exactly — **chr1 is held out for both**.
* **Count-recovery transform** — REFUTED. Both invert `log(1+count)` with `expm1`. Worst
  case had one used `exp`: 0.1% of the ratio.
* **Systematic scale difference** — REFUTED. Over all 18,672 `snps_accessibility`
  variants, mean signed difference **−0.001 log2**, r = 0.888. Cherimoya is systematically
  *quieter* (|log2FC| ratio 0.736 at median), i.e. the trend runs **opposite** to this
  locus.

## Where the gap actually lives

Exact decomposition into a count-head term and a profile-shape term:

| | count-head FC | × shape term | = reported |
|---|---|---|---|
| Cherimoya | 3.211 | 1.081 | 3.469 |
| ChromBPNet | 2.114 | 1.230 | 2.600 |

The count heads disagree by **52%** (total predicted counts over the shared 1,000 bp
output: ref 755.5 vs 448.9, alt 2425.8 vs 949.0); the shape term pulls **14%** the other
way. The reported 33% is the residue of two larger, partly cancelling disagreements.

Two contributors, both measured:

1. **They are not predicting the same quantity.** ChromBPNet loads the
   `chrombpnet_nobias` head — Tn5/DNase enzymatic bias subtracted. CATv1 has no
   bias-model concept at all (zero hits for "bias" in its source), so it predicts total
   observed counts. Consistent with the 2.1× difference in absolute ref counts. Measured
   on a matched ATAC K562 triple, adding the bias component *shrinks* a variant effect by
   ~20%, so neutralising it **widens** this gap rather than closing it.
2. **Fold-0 is a sample, not the model.** Cherimoya `ENCSR149XIL` across its own five
   folds: **3.469** (fold 0, shipped), 2.393, 2.716, 2.765, 2.768 — ChromBPNet's 2.600
   sits inside that range. Absolute ref counts vary **2.49×** across folds for the
   identical sequence. The 5-fold ensemble that CATv1's README recommends gives 2.749,
   closing **80%** of the gap; folds 1–4 close 92%.

## Two claims of mine that were wrong

**"AlphaGenome agrees with ChromBPNet, so Cherimoya is the outlier."** Wrong, and the way
it is wrong is instructive. The gap is a monotone function of the aggregation window:

| window | Cherimoya | ChromBPNet | AlphaGenome |
|---|---|---|---|
| 51 bp | 3.62 | 3.57 | 2.51 |
| **501 bp** (shipped) | **3.47** | **2.60** | **2.52** |
| 1001 bp | 3.21 | 2.11 | 2.42 |

At 51 bp the two BPNet-family models agree to 1.6% and **both** disagree with AlphaGenome;
the curves cross at 47 bp; at 2001 bp ChromBPNet falls *below* AlphaGenome. The apparent
corroboration is an artefact of where the curves intersect at the window width we happen
to ship. Any "X is the outlier" conclusion drawn from one window width is unsound.

**"33% is a lot."** It is inside the normal spread for these two models. rs12740374 sits at
the **83rd percentile** of the |log2FC| ≥ 0.5 stratum, where **18–22% of loci disagree by
more than 33%**.

## Verification quality note

One verifier found that the `effect_sha256` provenance I had been citing is **circular**:
`stamp_provenance_v4.py:140` copies it out of the reference-set artefact into every
oracle's npz post-hoc, so it cannot independently prove two oracles drew the same
population. The verifier instead regenerated the 18,672-variant set from both builders'
samplers and confirmed exact identity including order and stratum labels. The conclusion
held; the evidence I had offered for it was weaker than it looked. Worth fixing the stamp's
semantics or its documentation.

## Two real defects found on the way

1. **`regenerate_multioracle.py` built a 2,115 bp region** — `pos-half … pos+half` is
   `seqlen+1` bases. A query longer than `sequence_length` pushes ChromBPNet down its
   **sliding** branch (`num_windows=2`), tiling the model twice and populating `values`
   outside the central 1,000 bp with a second window. Present at two sites, including the
   Cherimoya runner added in this cycle by copying the first. **Fixed.** Effect on the
   reported score is 4th-decimal (ref 287.2173 → 287.2176, log2FC unchanged); the real
   effect was on the IGV values array.
2. **`chorus/core/result.py:275`** computes `(position - prediction_interval.reference.start)
   // resolution` with a 1-based variant position against a 0-based interval start, so
   **every** chromatin window is centred 1 bp right of the variant. Identical for both
   oracles and numerically negligible (~3e-4 on the ratio), so it is not this gap — but it
   is a genuine off-by-one in core scoring. **Deferred**, batched with the LegNet
   `resolution` issue, because both are correct-but-move-every-committed-artefact.

## Recommendation, and what was done

**Document, do not "fix".** Narrowing `window_bp` 501 → 51 closes 95% of the gap and is the
wrong move: it invalidates every 501 bp background CDF and moves both BPNet-family models
away from AlphaGenome. Curve-fitting to one variant.

Recorded as `docs/BACKGROUND_NULL_PROTOCOL.md` §12, with three rules: never compare
`ref_value`/`alt_value` across oracles (model-specific depth-normalised scales; only
`raw_score` and `quantile_score` are comparable, and all three agree here); the aggregation
window is doing more work than it looks; and a single-fold checkpoint is a sample, not the
model.

**Open decision for the maintainer:** should Cherimoya use the 5-fold ensemble CATv1
recommends? It closes 80% of the gap and is the upstream-recommended usage, but the
background CDFs were built on fold 0, so it carries a rebuild.

---

# Addendum F — the release audit's last eight findings, and a flake I had misdiagnosed (2026-08-09)

Closes the 12-dimension release audit. The eight remaining findings are recorded in full in
the commit message of `db4f312`; the short version is that **seven of the eight were reports
or tools describing something other than what they had actually done**, which is the same
failure mode as the headline sampler defect at the top of this document:

| # | what it claimed | what was true |
|---|---|---|
| 1 | summary box named the subject gene's effect | the winning track's region label was stripped; `region_swap` (subject SORT1) reported **−7.96** — the *GSTM2* TSS row, while SORT1's own TSS is **+0.752**, opposite sign. 8 of 13 rehydratable reports affected → 0 of 13 |
| 2 | batch detail rows reconcile with the headline | `track_scores` keyed on `assay_id` alone, dropping the per-gene CAGE rows from the detail while they still counted toward `max_effect`. rs7528419's headline cited a row its own file gives as the opposite sign |
| 3 | `score_ism` succeeded | returned an all-zero motif profile as SUCCESS when every substitution failed |
| 4 | "no score available" | a false note on **every** call — an allele-keyed dict indexed by assay id, so the guard always fell through. Enformer shipped `{ref 2.07, alt 2.46, effect 0.39}` beside "no score" |
| 5 | `--verify-against` verified content | a cardinality check: 5 fake track_ids with all-zero CDFs and no `build_config` earned "VERIFIED". **I had been citing this as proof of population identity** |
| 6 | one activity population, 31,500 positions | false for 5 of 8 oracles, and arithmetically impossible from the artefact alone (chrombpnet offered 34,004, sei 29,004). Now three derived populations with their own hashes |
| 7–8 | three notebook figures | one printed "Max signal: 1.0000" (true by construction) beside a raw 22.46; one compared two panels of **the same data** in different units; one **inverted** its own Sei ranking, rendering the 2.76 tracks flat and the 0.35 tracks full-height |

Each fix was re-measured by an independent checker that also reverted it in a scratch copy to
confirm the new test fails without it: 8 of 8 hold, 8 of 8 have load-bearing tests, 0
out-of-remit edits. All 14 walkthroughs, 4 multi-oracle passes and 6 notebooks regenerated on
GPU (13 of 13 steps rc=0, 0 notebook errors). The 21 unique pinned rows that remain all carry
an exceedance ratio, which is the intended end state.

## A flake I had misdiagnosed, twice over

`test_cherimoya_integration.py::test_predict_matches_direct_window_scoring` was the single
failure in the final 1,482-test sweep. Its committed docstring blamed **concurrent GPU load**
and instructed the reader not to loosen the tolerance, citing "on a quiet machine it passes
repeatedly (3/3 verified)".

That explanation was wrong. On an **idle** GPU the test fails about half the time — 4 runs
gave pass, pass, then max *relative* differences of **5.6e-4** and **4.2e-4** against
`rtol=1e-5`. Everything the docstring offered as mechanism was then ruled out:

- **Not the CPU fallback** — the `_last_device` assert passed on every run.
- **Not cross-process nondeterminism as such** — three consecutive `_forward_windows` calls,
  each five fresh subprocesses, agreed **bit-exactly** (0.0e+00).
- **Not batch shape or window cutting** — `_window_sequences` returns one window identical to
  the query sequence, so both paths submit the same input.
- **Not a persistent-worker warm-up** — there is no worker; every call is a fresh subprocess.
- **In-process the two paths agree exactly** — 0.0e+00 on both the per-position profile and
  the 501 bp sum, 6/6 iterations.

A mirrored probe localised it: iteration 0 differed at 8.5e-5 and iterations 1–3 were
bit-identical. So the residual is **sporadic, low-amplitude non-reproducibility inside
Cherimoya's Triton path across fresh processes** (≤5.6e-4 relative observed), landing on
whichever call happens to hit it. `rtol=1e-5` was asserting cross-process bit reproducibility
that this path does not provide — a coin flip presented as a gate.

**Fix: two arms rather than one loosened one.**

1. The env-mode test — the mode users get by default — keeps its full structure at
   `rtol=5e-3`, ~9× the measured noise floor. This does not blind it: averaging per-fold
   log2FCs instead of profiles moves rs12740374 by ~2%, `exp` for `expm1` shifts the sum by
   exactly 1.0 count, and a trim-offset error is gross. 4 of 4 runs pass.
2. A new `test_predict_matches_direct_window_scoring_in_process` asserts the invariant where
   it is exactly true, at `rtol=1e-7`. `chorus-cherimoya` has no `pytest` and installing dev
   deps into a pinned oracle env was refused, so the comparison runs *there* as a subprocess
   and is asserted *here* — the same split `_pinned_device` already used. It therefore runs in
   the ordinary base-env sweep, in 17 s, rather than being an exactness claim nothing executes.
   Negative control: a 1 bp trim shift trips it at **6.6e-1**, six orders above the noise floor.

Two notes on scope. The builder is single-process and was separately verified bit-reproducible
(`b0d4ae6`), so this does not implicate the shipped CDFs. And this is the **second** time in
this cycle I attributed a test failure to GPU contention that was not GPU contention —
`3b9e98d` corrected the first, where 7 failures were `test_mcp.py` leaving the state singleton
holding `reference_fasta=None`. Both times the real cause was reachable in under an hour of
measurement. The pattern to distrust is *"the machine was busy"* as a first explanation.

## Still open at the end of this cycle

- **Two HuggingFace tokens need rotating by the maintainer** — they were pasted into a session
  transcript in plaintext. The authenticated token ("Integration chorus", write, `lucapinello`)
  was never pasted and is clean. The LDlink token was additionally printed by the code path
  fixed in `edc344c`, so it was exposed in any transcript where an LDlink call failed.
- **The documented test command is red while CI is green**, because CI deselects the
  integration tests. That is a real inconsistency in what "the tests pass" means, and it is a
  maintainer decision whether to change the docs or the CI selection.
- `effect_sha256` remains circular (Addendum E) — the stamp is copied post-hoc, so it cannot
  independently prove two oracles drew the same population. The conclusion it was cited for
  was separately re-established; the stamp's semantics or its documentation should be fixed.

---

# Addendum G — two notebook defects reported from the rendered output (2026-08-09/10)

Reported by the maintainer while reviewing: *"prediction on synthetic sequences shows a white
plot, also the analysis of variant in this notebook doesn't show differences"* —
`examples/notebooks/single_oracle_quickstart.ipynb`. Both reproduce. Neither is a regression
from this branch, and both are the same failure mode as the rest of this document: **a figure
or a number that describes something other than what it appears to.**

## Not a regression, and that was checked before anything was changed

Cell 47's panel has been blank since **2026-08-01** — first committed in `b0a5731`, squash-merged
as `8651432`, which was the first re-execution of the notebook after the 2026-05-08 change
(`0c2b8e6`) made `normalize=True` with a fixed 0–3 axis the default for
`get_coolbox_representation`. Ink fraction is **0.07034 at `9cc6587` (2026-04-27) and earlier**,
where the axes autoscaled to 0–0.10 / 0–0.25 and ~80 spikes were visible, against **0.01066** at
`8651432`, `049ba77`, the merge-base `3e7990a`, `3599fd2`, `db4f312` and HEAD. The branch's
re-execution reproduced the same image: ink masks bit-identical, 26 pixels of 1,098,064 differing.

## Example E — the white plot had two stacked causes, and fixing one was not enough

The builder was broken in a way that made its own `element_positions` list fiction. It appended
blocks of 87 bp, 40 bp or 1,000 bp inside `for i in range(0, context_size, 1000)`, so:

| measured | value |
|---|---|
| joined length before truncation | **221,073 bp = 56.2%** of 393,216 → `[:context_size]` never fired |
| poly-A appended by the padding branch | **172,143 bp = 43.8%** of the final sequence |
| `ACGT` tandem repeat | 210,000 bp = 53.4% |
| literal `N` (one-hot encodes to all-zeros) | 7,630 bp = 1.94% |
| **real motif bases** | **3,443 bp = 0.88%** |
| drift of the 5th "promoter every 5 kb" | −8,452 bp, reaching −171,054 bp at the last |

The prediction was therefore honest and useless: raw max **0.0983** (DNase) and **0.2468** (CAGE)
against genome-wide display floors of p90 **0.09101** and p95 **0.21096**, rendering at **0.20%**
and **1.49%** of a 0–3 axis.

**Fixing the builder alone does not fix the figure**, and an independent arm measured that:
uniform-random filler, no `N`, no padding, every motif retained still gives a display max of
**0.367** — still blank. Enformer predicts accessibility from promoter context, not isolated
motifs. A GATA cluster alone (design C) reproduces the failure exactly: display **0.13**, peak
408 bins from where it was planted.

What works is a CpG island around the cluster. Four designs were measured; the shipped one is
a 1 kb CpG island + 20 GATA motifs at 45 bp spacing + a TATA box at the centre of the input,
because the output window is the central 114,688 bp:

| design | DNase display | CAGE display | peak vs centre |
|---|---|---|---|
| GATA cluster only | 0.13 | 1.52 | 408 bins away |
| CpG island only | 0.32 | 1.05 | 9 / 401 bins |
| **1 kb CpG + GATA + TATA (shipped)** | **1.71** | **2.93** | **5 / 0 bins** |
| 2 kb CpG + denser GATA | 0.84 | 3.00 (capped) | 6 / 3 bins |

Robust across five seeds: DNase display 1.71–2.40, CAGE 2.59–3.00, peak always at bin 443 —
inside the planted island. After re-execution the committed figure peaks at **56.8%** and
**98.1%** of its axis.

## Example D — the effect was real; the statistic hid it

`np.mean(pred.values)` over all 896 bins × 128 bp = 114,688 bp, differenced between alleles.
Measured on the old variant (chrX:48786129):

- a real, deterministic local change — DNase ref 3.0744 at the variant bin, Δ **−0.03384**;
  CAGE ref 129.31 at bin 451, Δ **−0.38765** — reproduced bitwise across two GPU processes;
- diluted **54×–1307×** by the window mean;
- **sign-inverted in 5 of 6 (track, allele) combinations** — the notebook printed **+2.57e-4**
  for DNase C>A while the variant's own bin had *dropped* by 0.0338;
- the ref/alt overlay separated by **≤0.98 px** on a 280 px axis, i.e. sub-pixel;
- `variant_results['effect_sizes']` — a per-bin `(896,)` array that is exactly `alt − ref` —
  fetched in cell 40 and **never used**.

The variant was also close to non-functional (−1.1% of its local peak), so the example taught
that variants do nothing. A scan of 86 positions replaced it with **chrX:48783008 A>C**, inside
`TTATCT` at chrX:48783006-48783011 — revcomp `AGATAA`, a canonical WGATAR site — 3.5 kb upstream
of the *GATA1* TSS. Reference base verified as `A` from hg38 before use. Measured after
re-execution:

| track | ref at variant bin | Δ | local | largest \|Δ\| |
|---|---|---|---|---|
| DNASE:K562 | 11.1984 | **−10.9264** | **−97.6%** | at the variant bin |
| CAGE:K562 | 24.2055 | −24.1154 | −99.6% | **−29.68 at +3,456 bp** — the *GATA1* promoter, 103.7 → 74 |

The cell now reports the change at the variant bin, the largest per-bin change and its offset,
and labels the window mean as the diluted summary it is. The figure gained a difference track,
because an absolute overlay cannot show a small effect at any y-scale.

## The same two defects elsewhere, and what a sweep of all six notebooks found

| notebook | cell | defect | measured |
|---|---|---|---|
| `comprehensive_oracle_showcase` | 16 | near-blank panel | 0.49% ink, trace 7.8% of axis; the prediction covers 13,060 bp of a 114,688 bp x-range — **88.6% of the panel empty by construction** |
| `comprehensive_oracle_showcase` | 35/36/51 | window-mean effect | 132× to **139,130×** understatement; cell 36 *plotted* the diluted value |
| `advanced_multi_oracle_analysis` | 69 | byte-identical expression | see below — the mean is not diluted here |
| `epinformerseq_testing` | 29/30 | whole-window sum | +1.067 vs +1.426 log2FC against cell 16 of the *same notebook*, which does it correctly |

## Two things I got wrong, caught by the adversarial reviewers

**1. ChromBPNet's window sum is not a diluted statistic, and I told the fixers it was.** I
briefed them that `advanced_multi_oracle_analysis` cell 69 "happens to survive because the
destroyed CTCF peak holds most of the window's signal mass." Two independent reviewers refuted
it from the source: ChromBPNet emits `softmax(profile) * expm1(counts)`
(`chorus/oracles/chrombpnet.py:600-612`), and the softmax sums to 1 across positions, so the sum
over its 1,000 predicted bins **is** `expm1(counts)` — the count head's total-accessibility
prediction, and ChromBPNet's own published counts readout. A single base moves that scalar and
rescales **every** bin, which is why the window mean drops ~66% for every alt allele. The prose
in both notebooks now states the real mechanism; the wrong one would have taught a plausible
falsehood in the very cell written to prevent one. Also recorded: `Window_mean` divides by all
2,114 bins while `_insert_into_output` leaves 1,114 of them as structural zeros, so it is ~2.1×
below a mean over the predicted span; and the largest per-bin change sits **33 bp upstream** at
−4.76, ~72× the −0.066 in the variant's own bin.

**2. A fix silently redefined a column its own figure labels.** In `epinformerseq_testing` the
pseudocount change was applied to `linear_fold_change` as well, making it `(alt+1)/(ref+1)` =
**2.69×** — while cell 27 calls that column "ALT/REF in linear space" and the section-6 figure's
axis reads "linear ALT/REF fold change", where the true ratio is 7.012/1.983 = **3.54×**. Reverted
to the plain ratio. The same review caught the comment quoting "2.69×" as if it were that ratio,
a log2 gap (0.357) computed across two different pseudocount conventions (0.359 held at one,
0.753 at neither), and cell 27's "log₂-counts for ChromBPNet" becoming false once the statistic
changed to a central-256-bp max. All corrected.

## The test that was missing

`grep -rl "image/png" tests/` returned **zero files**: nothing in the suite had ever looked at a
notebook's pixels, and all 13 assertions in the three notebook-figure tests pass against a blank
panel, because axis limits, titles and normalization flags can all be correct while the trace
between them is invisible.

`tests/test_notebook_figures_are_not_blank.py` decodes all 41 committed PNGs. Its useful finding
is that **the obvious check is inverted**: measured ink fractions are 0.0049–0.0107 for the two
blank panels and 0.0019–0.0038 for four legitimately sparse ones (narrow ChIP/DNase footprints in
a 100 kb window), so any ink threshold that fails the blanks also fails four correct figures. The
separating signal is trace peak height as a fraction of its own axis — 0.32%/1.29%/7.80% for the
blanks against 14.05%–24.59% for the sparse ones — and the threshold is the geometric midpoint of
that measured gap, `sqrt(0.0780 * 0.1405) = 0.1047`, 1.34× clear on each side. The docstring
records why an ink threshold cannot work, so it is not "simplified" back into a broken test.

## The second blank panel was the wrong tissue, not a rendering bug

`comprehensive_oracle_showcase.ipynb` cell 16 survived the first fix. Restricting the plot to
the span the prediction actually covers raised its coloured ink 33× (0.006% → 0.198%) but left
the trace at **6.72% of its axis** — still under the new test's 10.47% floor, and still unreadable.

The x-range was a real defect but not the cause. The cause is that the notebook profiles the
*GATA1* locus with **ChromBPNet DNASE:HepG2**, while every other panel in the same notebook uses
K562 (`enformer_tracks = ['ENCFF413AHU', 'CNhs11250']`, both K562). *GATA1* is erythroid; HepG2
is a hepatocyte line. A cell-type-specific accessibility model asked about a promoter that is
closed in its cell type correctly answers "almost nothing":

| ChromBPNet DNASE model over GATA1_REGION | raw mean | raw max | display max | % of a 0–3 axis |
|---|---|---|---|---|
| HepG2 (what shipped) | 0.0716 | 1.126 | 0.236 | **7.9%** |
| **K562 (correct tissue)** | 0.2633 | **6.077** | 0.930 | **31.0%** |

Switched to K562, with the reason written into the cell rather than left as a coincidence. Worth
noting what the blank-panel test bought here: it did not merely catch a bad figure, it caught a
showcase silently comparing a liver model against erythroid models at an erythroid locus — and
the first, plausible fix did not move the measurement past the threshold, which is exactly the
outcome a pixel-level assertion is for.
