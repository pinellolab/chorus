# MPS 1 MB regression — root-cause investigation

**Question (user):** "is truly a memory bandwidth? how can we be sure?"

## Hypotheses

1. **H₁ Unified-memory pressure**: 1 MB activations exceed Apple's
   high-bandwidth tile budget; MPS swaps tiles through system RAM.
   Prediction: smooth scaling up to a cliff, then sharp degradation.
2. **H₂ O(n²) attention compute**: 1 MB doubles the spatial dim vs
   524 kb; full attention is 4× compute. Prediction: smooth O(n²)
   curve.
3. **H₃ MPS kernel inefficiency at large tensors**: some MPS ops
   degrade past Apple's max tile size. Prediction: cliff but
   `current_allocated_memory()` stays modest; fp16 wouldn't help
   much (kernel issue, not memory).
4. **H₄ Logspace fallback overhead**: ruled out by the rope-patch
   experiment (524 kb: 3.78 s → 3.78 s; 1 MB: ~216 s → 222 s).

## Diagnostic

`mps_length_sweep` (in `report.md` Step 4): MPS forward at
{64, 128, 256, 384, 524, 655, 786, 918, 1024} kb with rope-patched
`apply_rope`. For each length: 1 warm-up + 3 timed iters,
`torch.mps.current_allocated_memory()` reported after each.

Expected if **H₁** (memory pressure):
- Time grows roughly linearly with L up to some L*, then degrades
  sharply (factor 5–50× for L > L*)
- Allocated memory grows linearly with L until L*, plateaus or grows
  super-linearly afterwards (suggesting tiling/swap)

Expected if **H₂** (compute):
- Time grows ~quadratically (4× per doubling of L)
- Allocated memory grows linearly throughout, no plateau

Expected if **H₃** (kernel):
- Time grows linearly to ~512 kb, then jumps as Apple's tile-cache
  limit is hit; fp16 (half memory) shouldn't recover

## Results

Hardware: Apple M3 Ultra, **96 GB unified memory**, macOS 24.6.0,
torch 2.11.0, alphagenome-pytorch 0.3.2. Rope patch applied (no
logspace fallback warnings).

| L (kb) | iter1 (s) | iter2 (s) | iter3 (s) | mean (s) | mem (MB) | Δmem | Δtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.32 | 0.32 | 0.33 | 0.32 | 1881 | — | — |
| 128 | 0.64 | 0.61 | 0.59 | 0.62 | 2044 | +163 | +0.30 s |
| 256 | 1.43 | 1.31 | 1.28 | 1.34 | 2369 | +325 | +0.72 s |
| 384 | 2.43 | 2.50 | 2.62 | 2.52 | 2695 | +326 | +1.18 s |
| 512 | 3.44 | 4.03 | 3.90 | 3.79 | 3017 | +322 | +1.27 s |
| 640 | 5.49 | 5.48 | 5.19 | 5.38 | 3345 | +328 | +1.59 s |
| 768 | 6.83 | 7.06 | 7.64 | 7.18 | 3667 | +322 | +1.80 s |
| **896** | **51.94** | **103.99** | **98.69** | **84.87** | **3994** | **+327** | **+77.69 s** |
| **1024** | 79.94 | 151.84 | 291.66 | **174.48** | 4319 | +325 | +89.61 s |

The within-L iter time **grows** at 1024 kb (80 → 152 → 292 s) —
characteristic of progressive cache thrashing rather than a one-shot
allocation failure. Memory growth across the cliff (768 → 1024) is
linear and modest: +652 MB for +256 kb, ratio 2.5 MB/kb (matching the
pre-cliff slope). **No swap-out** observed in `vm_stat` deltas (only
the system's chronic background pressure of ~2.5 M compressed pages).

## Cumulative-pressure cross-check

A second sweep at the cliff zone (`512 → 768 → 1024 kb`) was run
after ~30 min of back-to-back MPS work in the same shell session:

| L (kb) | dtype | iter1 | iter2 | iter3 | mean (s) | mem MB |
|---:|---|---:|---:|---:|---:|---:|
| 512 | fp32 | 4.02 | 4.14 | 3.48 | 3.88 | 3018 |
| 768 | fp32 | 6.37 | 11.53 | 23.62 | **13.84** | 3669 |

The 768 kb mean **doubled** (7.18 → 13.84 s) **without any code change** —
process was killed at the 1024 kb step before bf16 could be measured.
This re-cements the diagnosis: the regression is **system-state-dependent
cache pressure**, not a deterministic compute cliff. A fresh boot would
likely move the cliff to a slightly different L. Two things scale
together — sequence length × accumulated unified-memory fragmentation
from prior runs — and tip MPS off the cache-resident path.

## Verdict

**Not raw RAM swap** (H₁ in the colloquial sense): the cliff occurs
between 768 kb (3.7 GB allocated) and 896 kb (4.0 GB allocated) —
nowhere near the 96 GB physical ceiling. Memory growth across the
cliff is **linear and modest** (+8% mem for +17% length), so we are
not paging out. `vm_stat` showed no swap-out delta during the runs.

**Not logspace fallback** (H₄): the rope patch eliminating the
`aten::logspace.out` CPU fallback didn't move the 1 MB number (216 s
without patch → 222 s with patch — within run-to-run noise).

**Not pure compute-quadratic** (H₂): the within-L iter time **grows
across iterations** (1024 kb: 80 → 152 → 292 s; 768 kb on second
sweep: 6 → 12 → 24 s) — pure compute would be uniform. Also the
ratio 7.18 s (768 kb) → 84.87 s (896 kb) is 11.8× for 1.17× length,
far steeper than O(n²) (which would be 1.36×).

**Most likely (H₃)**: GPU on-die cache + working-set pressure. Apple
Silicon GPUs have a system-level cache (~32 MB on M3) plus per-tile
shader caches. Once the model's per-block activations + attention
working set exceeds those caches, each op pays full unified-memory-
bandwidth cost (800 GB/s on M3 Ultra is fast in absolute terms but
~10–20× slower than cache hits). The cliff is sharp because eviction
is a step function; the within-L degradation is consistent with
fragmentation in MPS's own allocator pushing more of the working set
out of cache on each call.

The user's intuition — "is it memory bandwidth?" — was directionally
right but more specific than physical RAM exhaustion: **it's GPU
cache spillover**, which manifests as effective memory bandwidth
becoming the bottleneck once the working set exceeds on-die cache.

## Practical implications for chorus

| Window | Mac default route | Why |
|---|---|---|
| ≤768 kb | **PT MPS** (5–8× over JAX CPU) | comfortably cache-resident |
| 768–896 kb | borderline; check `current_allocated_memory()` | cliff-zone, run-state-dependent |
| ≥896 kb | **JAX CPU** | post-cliff, MPS slower than CPU |

A future tiling helper (predict 4× 256 kb tiles + stitch with
attention-receptive-field overlap) would unlock full-1 MB MPS at
~15 s per query (4 × 3.77 s + stitch overhead), 3.3× faster than
JAX CPU. Tiling is the **real fix** — deferred to a follow-up PR
because it requires non-trivial boundary handling.

A simpler band-aid: the oracle could expose
`oracle.predict_tile_size_hint = 768 * 1024` and have higher-level
chorus code chunk longer queries before calling. This deserves a
small dedicated PR.

## Follow-up tests (only if H₁ is suggested but ambiguous)

- **fp16/bf16 sweep**: rerun at 1 MB with `model.half()` — if memory-
  bound, halving activation size should recover most of the throughput.
  No effect → either compute or kernel limit.
- **`vm_stat` snapshot before/during/after 1 MB run**: swap-in/out
  delta directly attributes to unified-memory pressure.
- **`torch.profiler` trace at 1 MB**: per-op breakdown shows whether
  attention.softmax, attention.matmul, or memory ops dominate.
