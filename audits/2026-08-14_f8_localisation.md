# F8 — localising the `SORT1_enformer` regeneration drift, 2026-08-14

Time-boxed bisect of the one deferred item from v0.7.3 that was a real defect rather than a stale
number: two runs of identical code produce a different `SORT1_enformer` example. Three previous
passes recorded it and left the mechanism unlocalised.

**Outcome: reproducible in one command, and narrowed to a specific asymmetry that contradicts the
code.** Two hypotheses are refuted with measurements, one amplifier is confirmed and quantified, and
the remaining suspect is named precisely enough to test in a single run.

---

## What the number actually is

The CHANGELOG said "400 values". Measured against the committed file:

* **596 differing numeric fields**, ≈ **298 distinct values** — every track is serialised twice, once
  under `all_scores` and once under `scores_by_layer`.
* The file has **845 numeric leaves**, so ~35% of them move.
* Magnitudes, from `audits/2026-08-12_post_v0.7.2_audit.md`: median **0.0159%**, `ref_value` max
  **0.067%**, `raw_score` max **4.29%** (log2FC over a near-zero denominator).

Both the count and the "an in-process check cannot catch this" wording are corrected in the
CHANGELOG. The gate *is* cross-process — it spawns two children — it was simply pointed at
**AlphaGenome with `use_environment=False`**, an oracle and an execution model the failing example
does not use.

## It is reproducible now

`scripts/gate_end_to_end_determinism.py` gained `--oracle enformer` (the `use_environment=True`
subprocess path `regenerate_examples.py` actually takes) and `--strings`:

```
$ python scripts/gate_end_to_end_determinism.py --oracle enformer --strings --gpu 4
[gate] differing values      : 32
[gate] worst relative delta  : 1.446e-02
[gate] FAIL
```

No string leaves differed in that run, so this instance is numeric drift with no top-N identity
flip. `generated_at` is excluded from the comparison, and `AnalysisRequest` now honours
`SOURCE_DATE_EPOCH`, so a regeneration diff is readable at all instead of always showing at least a
timestamp.

## Refuted, with evidence

**1. cuDNN/cuBLAS autotune varying by available workspace.** This was the leading hypothesis: every
`_predict` spawns a fresh process, each redoing autotune, whose kernel choice depends on free memory.
Measured — two `use_environment=True` `predict()` calls over the same region, hashed across **all**
896 × 5,313 = 4,760,448 float32 values:

| condition | sha256 (first 16) | verdict |
|---|---|---|
| idle GPU 4 | `067b6ee7103d7bac` | identical across both runs |
| **40 GB held on GPU 4** by a second process | `067b6ee7103d7bac` | identical, **and the same hash as idle** |

Contention producing the *identical* hash is the informative part: if workspace budget drove kernel
selection, 40 GB of pressure on the same card is where it would appear. Checked for a confound —
neither `enformer.py` nor its predict template caches a model or a prediction, so the second run is a
genuine second forward pass.

**2. The percentile tie-break.** `_rank_with_tie_breaking` draws from a blake2b hash keyed on the raw
value, so a 1-ULP change could in principle re-hash across a whole tied run and move the percentile
by the tie's full width. Measured on all 84 committed tracks against their shipped CDF rows: **0**
change rank under `np.nextafter`. Not the mechanism here — those tracks' values do not land inside
tied runs.

## Confirmed amplifier: the top-N cutoff

`discover_variant_effects` cuts hard at `top_n_per_layer`, so a sub-percent wobble can change *which
track ships*. Adjacent-rank gaps inside the committed top-12, per layer:

| layer | tightest adjacent gap | gap at the cutoff |
|---|---|---|
| `tss_activity` | 0.14% | 87.7% |
| `chromatin_accessibility` | 0.25% | 21.6% |
| `tf_binding` | 1.42% | **4.25%** |
| `histone_marks` | 2.38% | 44.9% |

Against a measured `raw_score` max drift of **4.29%**, `tf_binding`'s 4.25% cutoff gap is inside
range: a rank swap at the boundary is possible, and it would present as a large diff rather than a
small numeric one. `--strings` now names that case when it happens.

*Scope note:* the committed file holds only the survivors of the cut, so the 12↔13 gap is not
measurable from it. The adjacent gaps above are a proxy for how tightly packed that neighbourhood is,
not a direct measurement of the boundary.

## The drift is intermittent — and that is the finding

Four pairs of consecutive `use_environment=True` `predict()` calls, same region, same GPU:

| # | tracks requested | condition | result |
|---|---|---|---|
| 1 | 5,313 | idle | bitwise identical |
| 2 | 5,313 | 40 GB held on the same GPU | bitwise identical, same hash as #1 |
| 3 | **4** | idle | **3,583 of 3,584 values differ** — max abs 2.2e-02, median 5.6e-05 |
| 4 | 4 *and* 5,313, one process | idle | **both bitwise identical** |

**A correction to my own reading.** Pair #3 alone looked like a clean track-count asymmetry, and it
was tempting: it would have contradicted the earlier pass that declared batch composition "refuted by
reading" (`predict_template.py` runs `predict_on_batch` and only then slices
`[:, track_indices]`, so the forward pass genuinely does not depend on the requested set). Pair #4
re-ran the same 4-track comparison and got **identical** results, so #3 was **chance, not a
mechanism**. The asymmetry does not exist; I nearly published it on n=1.

What the four pairs do establish is that the drift is **intermittent** — roughly 1 pair in 4 here —
and that is itself the explanation for why three previous passes recorded "Enformer's forward pass is
bitwise identical in-process *and* across processes" while real regenerations kept differing. Both
observations are true. A single clean probe is simply not evidence of determinism when the failure
rate is well under 1.

**Consequences for how this gets chased.** Any future bisect must run each configuration **several
times** and report a rate, not a verdict; a one-shot comparison will confirm whichever answer the
experimenter expects about three times in four. The gate is the right instrument because it is one
command and can be looped. Ruled out so far: workspace-driven autotune (identical hash under 40 GB
of contention), the percentile tie-break (0 of 84 tracks ULP-sensitive), and track-set size (pair
#4). Not yet excluded, and each testable by looping the gate with the stage stubbed: the reference
sequence extraction, `low_effective_bins`, and per-track object construction.

## Landed regardless of the cause

* **`discovery.py`'s `layers_affected` is `sorted()`.** It was `list({...})` over a set of `str`, so
  its order followed per-process `hash(str)`. It reaches MCP replies rather than the committed
  examples, so it is a separate defect of the same class — and free to remove.
* **`AnalysisRequest.generated_at` honours `SOURCE_DATE_EPOCH`**, so a regeneration can be diffed.
* **The gate can see this class of failure**: `--oracle enformer`, `--strings`, and `generated_at`
  excluded.
* **Two CHANGELOG corrections**: the field count, and the false claim that the gate's design
  prevents it from catching this.

## Deliberately not done

`TF_DETERMINISTIC_OPS` is **not** set. `grep` finds it nowhere in the tree — `chorus/core/determinism.py`
pins `--xla_gpu_deterministic_ops=true` for JAX only, so the TF-backed oracles never got that
treatment. It is a plausible hardening, but `determinism.py`'s own rule is that a background and its
queries must run under the same setting, so flipping it would invalidate the Enformer and ChromBPNet
nulls and require rebuilding both. That is far outside this scope, and it should not be flipped
without measuring whether it moves any number first. Recorded with the cost attached.
