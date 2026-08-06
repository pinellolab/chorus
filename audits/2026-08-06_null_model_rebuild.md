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
