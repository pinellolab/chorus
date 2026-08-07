# 2026-08-06 — Fixing the background null model, and eight defects found on the way

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
covering the checklist sections that need the live artefacts in place, and the six defects
it found. Five were in this cycle's own work; one was long-standing.

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

**Release gates must be run on an unloaded GPU.** Seven of the twelve failures in the first
sweep were contention, not defects.
