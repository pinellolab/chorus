# AlphaGenome PyTorch backend spike — audit report

**Date**: 2026-04-29
**Branch**: `feat/alphagenome-pytorch-backend`
**Plan**: `/Users/lp698/.claude/plans/flickering-seeking-valley.md`
**Upstream port**: [`genomicsxai/alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch) (v0.3.2)
**Weights**: [`gtca/alphagenome_pytorch`](https://huggingface.co/gtca/alphagenome_pytorch) (public, 5 safetensors, 450M params)

## TL;DR

Landed `AlphaGenomePTOracle` as an **opt-in side-by-side backend** (the
JAX path stays the default). The upstream port works end-to-end on Apple
Silicon, but the speed picture is **mixed**, not the clean Mac win we
hoped for:

| Window | JAX CPU | PT CPU | PT MPS | Mac winner |
|---|---:|---:|---:|---|
| 128 kb | 5.09 s | 6.66 s | **0.61 s** | PT MPS (8.3× over JAX) |
| 524 kb | 21.96 s | 32.69 s | **3.77 s** | PT MPS (5.8× over JAX) |
| **1 MB** | **50.29 s** | 84.46 s | 216.39 s | **JAX CPU** (PT MPS regresses) |

**MPS is 5–8× faster than the current default (JAX CPU) at sub-1 MB**,
but **regresses badly at the full 1 MB context**. Cause: 1 MB inference
exceeds Apple unified-memory budget, forcing MPS↔system-RAM swaps. The
slowdown is reproducible across multiple runs (255 s, 226 s, 167 s).

Verdict: **don't flip the macOS default** in this PR. Ship the PyTorch
backend as opt-in. Variant-scoring on smaller windows (≤524 kb) gets a
real Mac speedup; 1 MB notebooks should keep using JAX CPU until either
upstream lands an MPS `logspace` kernel or chorus ships a tiling helper.
Follow-ups in `## Decisions` below.

## What landed in this PR

| File | Purpose |
|---|---|
| `environments/chorus-alphagenome_pt.yml` | New conda env (Python 3.12 + torch + alphagenome-pytorch[scoring]) |
| `chorus/oracles/alphagenome_pt.py` | `AlphaGenomePTOracle` — mirrors `AlphaGenomeOracle` shape; load + forward swapped to PyTorch port |
| `chorus/oracles/alphagenome_pt_source/__init__.py` | Empty package marker |
| `chorus/oracles/alphagenome_pt_source/templates/{__init__,load_template,predict_template}.py` | Subprocess templates run inside `chorus-alphagenome_pt` |
| `chorus/oracles/__init__.py` | Register `alphagenome_pt` in `ORACLES` + `get_oracle` |
| `chorus/__init__.py` | Wire `create_oracle('alphagenome_pt', ...)` |
| `tests/test_alphagenome_backends_equivalence.py` | `@pytest.mark.integration` JAX↔PyTorch DNase equivalence pin |
| `scripts/benchmark_alphagenome_backends.py` | Cross-backend cross-device latency table generator |
| `audits/2026-04-29_alphagenome_pytorch_spike/report.md` | This report |

The JAX-path `AlphaGenomeOracle` is **unchanged** — still the default
when `chorus.create_oracle('alphagenome', ...)` is called. The 5,731-track
metadata cache (`alphagenome_tracks.json`) is shared between backends, so
walkthroughs and CDF backgrounds port over without schema work.

## Verification trail

### Step 1 — env build ✓

```bash
mamba env create -f environments/chorus-alphagenome_pt.yml
```

Built cleanly on macOS arm64 after dropping the `pytorch` conda channel
(SSL connect errors against `conda.anaconda.org/pytorch`); torch is now
pip-installed (torch 2.11.0, MPS available). The conda-forge `compilers`
package ships its own `libomp.dylib` that conflicts with the libomp
inside the pip torch wheel — both templates set
`KMP_DUPLICATE_LIB_OK=TRUE` before `import torch` to mute the diagnostic
(harmless on Linux/CUDA).

### Step 2 — load + single forward on MPS ✓

```python
weights = hf_hub_download("gtca/alphagenome_pytorch", "model_all_folds.safetensors")
model = AlphaGenome.from_pretrained(weights, device=torch.device("mps"))
model.eval()
out = model(dna_onehot, organism_index=torch.tensor([0]), heads=("atac","dnase"))
```

Required two macOS-specific shims:

1. `PYTORCH_ENABLE_MPS_FALLBACK=1` — `torch.logspace` is not implemented
   on MPS; the rope-attention layer falls back to CPU for that single op
   (per-attention-block cost; warms up cleanly).
2. `organism_index` must be passed as a torch.Tensor — passing `int`
   crashes inside the embedding layer (`forward()` doesn't auto-convert
   like `predict()` does, but `forward()` is what we call to get the
   `heads=` filter).

Output shapes verified:

```
atac[1]:   (1, 1048576, 256)   # 1bp resolution
atac[128]: (1, 8192, 256)      # 128bp pooled
dnase[1]:  (1, 1048576, 384)
dnase[128]: (1, 8192, 384)
```

These match the chorus track schema exactly — local_index slicing into
`out[ot_key][res][0, :, idx]` reuses the existing
`alphagenome_tracks.json` cache.

### Step 3 — speed benchmarks (Apple M3 Ultra, macOS arm64)

| Window | Backend | Device | Mean (s) | Min (s) | Notes |
|---|---|---|---:|---:|---|
| 32 kb | PyTorch | CPU | 1.63 | 1.62 | |
| 32 kb | PyTorch | MPS | 0.20 | 0.20 | **8.2× over CPU PyTorch** |
| 128 kb | PyTorch | CPU | 6.66 | 6.62 | |
| 128 kb | PyTorch | MPS | 0.61 | 0.60 | **10.9× over CPU PyTorch** |
| 524 kb | PyTorch | CPU | 32.69 | 32.50 | |
| 524 kb | PyTorch | MPS | 3.77 | 3.75 | **8.7× over CPU PyTorch** |
| **1 MB** | **PyTorch** | **CPU** | **84.46** | **84.37** | |
| **1 MB** | **PyTorch** | **MPS** | **216.39** | **167.00** | **0.39× — slower than CPU** |
| 128 kb | JAX | CPU | 5.09 | 5.06 | macOS baseline (`JAX_PLATFORMS=cpu`) |
| 524 kb | JAX | CPU | 21.96 | 21.78 | |
| **1 MB** | **JAX** | **CPU** | **50.29** | **49.96** | macOS default — best 1 MB number on this machine |

JAX (`chorus-alphagenome`) is ~30–65 % **faster than PyTorch CPU** on
this M3 Ultra (likely better matmul kernels via XLA), so the comparison
that matters is **PT MPS vs JAX CPU**. PT MPS wins clearly at ≤524 kb
(5.8–8.3× speedup over the current default), loses at 1 MB.

The 1 MB MPS regression is reproducible across three back-to-back runs
(255 s, 226 s, 167 s — slowdown decreasing as MPS cache warms but
never matching CPU). The likely cause is unified-memory exhaustion:
the model's attention activations at 1 MB exceed what fits in MPS's
hot-cached pool, so each step round-trips data through main memory.
The `aten::logspace.out` CPU fallback aggravates this by forcing
dozens of MPS↔system memory copies per attention block.

### Step 4 — equivalence (test scaffolded; not yet run on real data)

`tests/test_alphagenome_backends_equivalence.py::test_jax_pt_dnase_equivalence_at_sort1`
runs both backends on a 1 MB SORT1 window and asserts:

- Same output shape for `OutputType.DNASE` at resolution 1
- `max(|pt - jax|) < 0.1`
- `mean(|pt - jax|) / mean(|jax|) < 5%`

Tolerances are deliberately generous — upstream claims per-head equivalence
but our pin is at the user-facing log-counts scale, not raw logits.
Test is `@pytest.mark.integration` and runs both backends inside their
respective conda envs via `mamba run`.

> _Test execution pending HF_TOKEN handshake on this machine; deferred
> to user's CUDA box where the 1 MB JAX side completes in ~3 s instead
> of ~80 s on macOS CPU._

## Mapping table — PyTorch vs JAX exposure

| Capability | JAX path (today) | PyTorch path (this PR) |
|---|---|---|
| Load API | `create_from_huggingface(fold, device)` | `AlphaGenome.from_pretrained(local_path, device)` — needs `hf_hub_download` first |
| Predict API | `predict_sequence(seq, requested_outputs, ontology_terms)` | `model(dna_onehot, organism_index, heads=..., resolutions=...)` |
| Heads in chorus | 9 (skips CONTACT_MAPS, SPLICE_JUNCTIONS) | Same 9 wired in this PR (port has 11; we keep parity for now) |
| Folds | All 4 (single `all_folds` checkpoint via JAX) | All 4 — separate safetensors per fold (`model_fold_{0..3}.safetensors`) plus `model_all_folds.safetensors` (default) |
| Variant scoring | chorus generic two-predict-and-diff | Upstream `VariantScoringModel` + 7 scorers (`CenterMaskScorer`, `GeneMaskLFCScorer`, `ContactMapScorer`, `GeneMaskSplicingScorer`, `SpliceJunctionScorer`, `PolyadenylationScorer`, `GeneMaskActiveScorer`) — **not yet wired through chorus**; opt-in via direct upstream import |
| Fine-tuning | ❌ none | ✅ LoRA + linear probe via `alphagenome_pytorch.scripts.finetune` — **not wired through chorus** |
| Apple Silicon | Forced CPU (JAX-Metal crashes with `default_memory_space`) | MPS works for sub-1 MB; CPU fallback at 1 MB |
| HF gating | Gated (`google/alphagenome-all-folds`) | Public (`gtca/alphagenome_pytorch`) — same auth flow inherited but not strictly required |

## Decisions / open follow-ups

### Will NOT do in this PR (but should track)

1. **Flip macOS default to `alphagenome_pt`** — blocked by the 1 MB MPS
   regression. Most variant-effect notebooks use the full 1 MB context
   and would slow down. Reconsider after one of:
   - Upstream adds an MPS-native `logspace` (track via
     [pytorch/pytorch#141287](https://github.com/pytorch/pytorch/issues/141287))
   - We ship a tiling helper that splits 1 MB → 4× 256 kb on MPS
   - We add a `device='mps' if length < 600_000 else 'cpu'` heuristic
2. **Wire `VariantScoringModel`** into chorus's `predict_variant_effect`
   path — biggest API improvement upstream offers. Belongs in a
   dedicated PR with side-by-side equivalence checks against the
   current two-predict-and-diff results.
3. **Expose CONTACT_MAPS / SPLICE_JUNCTIONS** — PyTorch port returns
   them; current chorus `alphagenome_metadata.py` strips them. Wire-up
   is one entry in `OUTPUT_TYPE_TO_CHORUS` plus a small change to track
   schema; needs walkthrough validation.
4. **Rebuild CDFs against PyTorch outputs** — only if the equivalence
   test fails when run end-to-end. Deferred until the user's CUDA box
   completes the test.

### Already deferred (per plan)

- Mirroring weights to a chorus-controlled HF repo
  (`gtca/alphagenome_pytorch` is currently fine)
- Linux/CUDA verification beyond build + load (user's other machine)
- Replacing the JAX path entirely

## How to use

```python
import chorus
oracle = chorus.create_oracle('alphagenome_pt', use_environment=True)
oracle.load_pretrained_model()
result = oracle.predict(('chr1', 109_274_000, 109_275_500))
```

The oracle auto-picks MPS on Mac, CUDA on Linux, CPU otherwise. Pass
`device='cpu'` to force CPU (recommended for 1 MB queries on Mac until
the MPS regression is fixed). The conda env `chorus-alphagenome_pt`
needs `KMP_DUPLICATE_LIB_OK=TRUE` and `PYTORCH_ENABLE_MPS_FALLBACK=1`
in the subprocess environment — both are set automatically by the
chorus subprocess templates.

## Numbers source

All benchmark numbers: Apple M3 Ultra, macOS 24.6.0 (Darwin), 32 GB
unified memory, torch 2.11.0, alphagenome-pytorch 0.3.2.dev15+gc5e77a37a.
3 timed iterations after 2 warm-ups, mean reported. Random
1-MB-of-bases input, no FASTA dependency.
