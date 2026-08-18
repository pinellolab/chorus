# Focused post-v0.7.4 audit — determinism closed, one new defect

Scope agreed up front: what v0.7.4 changed, plus the two determinism questions left open
(ChromBPNet and AlphaGenome). Run against `main` at and after `8ffb842` (= `v0.7.4`).

## Cross-process determinism is now measured for all nine oracles

This was the open question. In-process determinism had been allowed to stand for more than it
measured — Enformer is bit-exact within one process and drifts up to 3.4% *between* processes, and
between processes is the path `regenerate_examples.py` takes.

| oracle | framework | cross-process | evidence |
|---|---|---|---|
| chrombpnet | TF | **bit-exact** | 3/3 pairs, `max diff 0.000e+00` |
| alphagenome | JAX | **bit-exact** | gate, 3/3 runs, `worst relative 0.000e+00` |
| borzoi | PyTorch | **bit-exact** | 2/2 pairs |
| cherimoya | PyTorch + Triton | **bit-exact** | 2 runs |
| legnet | PyTorch | **bit-exact** | 2 runs |
| epinformerseq | PyTorch | **bit-exact** | 2 runs |
| sei | PyTorch | **bit-exact** | 2/2 pairs |
| **enformer** | TF | **drifts** | 3.4% / 1.4% worst relative; bit-exact 4/4 with `TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1` |
| alphagenome_pt | PyTorch | **bit-exact** | 2/2 pairs, 9 tracks spanning all 9 heads, 288 values — measurable once #239 landed |

**The headline: Enformer is the only oracle that drifts — and this is now a complete table, not a
partial one.** All nine are measured. F8 is not a TensorFlow problem: ChromBPNet is the other TF oracle
and is bit-exact. That materially shrinks the accepted risk: it is one oracle wide, and the one way it
can reach a reader (a changed top-N membership) is already covered by `near_ties_at_cutoff`.

`alphagenome_pt` was the last unmeasured row and stayed that way for most of this pass, because the
defect below blocked it. Fixing that defect closed it: **bit-exact, 2/2 pairs.** Worth noting how the
blockage read — "not measured" looked like a scheduling gap in the table, when it was actually a live
bug in the default code path.

Two corrections to the record followed:

* `AUDIT_CHECKLIST.md` said **"AlphaGenome (JAX) is NOT deterministic run-to-run"**, on the strength of
  two calls differing on all 64 raw values. Three gate runs give `0.000e+00`. The finding predates
  `determinism.py` pinning `--xla_gpu_deterministic_ops=true`, which is the likely cause of the change.
* The adjacent line claimed predictions were "verified bitwise" for seven oracles. That was
  **same-process only**, which Enformer shows is necessary but not sufficient. Now qualified, pointing
  at the cross-process table as the real record.

## New defect: `assay_ids=None` breaks alphagenome_pt

`predict_variant_effect(..., assay_ids=None)` raises
`RuntimeError: Code execution failed: invalid literal for int() with base 10: 'logits'`. With explicit
ids the same call succeeds, so this is specifically the default-to-all-tracks path.

Same *class* as the Sei defect fixed in #237 — an omitted `assay_ids` mishandled — in the other
AlphaGenome backend, which is largely a copy of the JAX one. It is what blocked the last determinism
row.

**My guard for the Sei defect does not catch it, and that is worth stating.**
`test_no_oracle_raises_when_assay_ids_is_omitted` asserts `_validate_assay_ids(None)` does not raise,
at the validation layer so it needs no GPU. alphagenome_pt passes that: its failure is deeper, inside
the child process where the default track list is resolved. A validation-layer guard cannot see a
defect in the execution path it precedes — the cheap test bought less coverage than it appeared to.

**Fixed in #239** (after this report was first written), and the fix found two more things:

* The splice-site tracks were being read as **logits, not probabilities**. The crash masked it. The JAX
  reference returns `{'logits', 'predictions'}` and treats the softmax as the prediction, so the two
  AlphaGenome backends would have disagreed on those 4 tracks.
* **The first fix was incomplete and the unit tests did not notice.** `use_environment=True` — this
  backend's default — runs a subprocess *template* that carried its own copy of the same six lines. All
  13 unit tests passed while the default path stayed broken. That is the second time in this session a
  duplicated copy of one rule hid a defect, and it is a stronger version of the lesson below: a test can
  fail to catch a bug not only by asserting at the wrong *layer* but by asserting against the wrong
  *copy*.

## Everything else v0.7.4 changed, re-verified

| area | result |
|---|---|
| Sei end-to-end scoring | all **21,947** tracks produce a `raw_score` |
| `describe_tracks()` | 9/9 oracles, every count matching its published figure |
| MCP `list_tracks` | sei/legnet/chrombpnet/epinformerseq derive from the catalogue; chrombpnet no longer implies 41,280 models where 744 exist |
| near-tie reporting | fires on a drift-sized gap, silent on a real one, does not alter selection |
| release artefacts | tag → `8ffb842`, `isDraft=false`, tag message and release body byte-identical (10,665 chars), 5 `#` headings intact |
| fast / browser / integration | **2,124** / **58** / **158 passed, 0 failed** |

## Two documentation gaps this pass closed

Both were cases of shipped behaviour the README did not describe.

1. **`describe_tracks()` was undocumented in the README.** It shipped in 0.7.4 specifically to end the
   "four attempts to get one track id" friction, and the README's *Discovering tracks* section still
   opened with `from chorus.oracles.enformer_source.enformer_metadata import get_metadata` — a
   per-oracle import, i.e. the friction itself. Now leads with `describe_tracks()`, every snippet
   copied from a real run (5,313 total; `query='K562', limit=3` → the three DNASE:K562 ids shown).
2. **The accepted Enformer drift was invisible to users.** It was recorded in the CHANGELOG and
   `BACKGROUND_NULL_PROTOCOL.md`, both of which a user hitting it would not think to open. The README's
   Enformer section now carries the measured numbers, the fact that it is the only affected oracle, why
   conclusions hold, and the env-var opt-in. Accepting a known limitation is defensible; not telling
   users about it is not.

## One follow-up created by this pass

`TrackRecord.has_background` is declared, documented as tri-state, and **never populated** — it is
`None` on every record from every oracle. The docstring claimed "a caller can now see it", which was
aspirational. Corrected to state the truth and the reason: answering it needs the background NPZ and a
`_match_track_id` pass, and `describe_tracks()` is deliberately load-free — no model, no genome, no
GPU, no multi-GiB download — which is what makes it cheap enough to explore with. Callers who need the
answer should ask `NormalizationLoader.has_background`. Left `None` rather than defaulted to `False`,
which would read as "checked and absent".

## Process notes worth keeping

* **I caused two false failures** by running the determinism gate on other GPUs while the integration
  suite was live. `CLAUDE.md` warns about exactly this, and it produced a false `CUDA_ERROR_OUT_OF_MEMORY`
  the same way during the v0.7.3 audit. I had read that warning earlier the same session. Rerunning in
  isolation was the only thing that distinguished my mistake from a regression — on the run I was about
  to gate a release on.
* **Two probe failures were mine, not the code's**: measuring Sei from a branch that lacked the fix, and
  choosing a GPU without headroom (other tenants held ~53 GiB per card; alphagenome_pt needed 6 GiB
  more). Both looked like defects until rerun.
* **Counts passing is not fields passing.** cherimoya's catalogue reported the right 1,518 tracks with
  `cell_type` `None` on every one, because the code read a metadata key that does not exist. Every count
  assertion passed. It surfaced only when a consumer tried to group by biosample and got 1.
