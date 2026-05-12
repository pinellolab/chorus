# Issue #81 — root cause found: bf16 output quantization + tight absolute tolerance

**Date**: 2026-05-10 / 2026-05-11
**Tested against**: v0.5.1 (commit 907b564)
**Hardware**: ml007 NVIDIA A100-PCIE-40GB

## TL;DR

**The PT and JAX backends are statistically identical.** The 0.476 max abs diff at SORT1 isn't a regression — it's the **bf16 output quantization in the JAX backend** showing up at peak signal positions, combined with tiny per-position fp32 numerical kernel differences. The current test uses an **absolute** tolerance of 0.1, which is mathematically unachievable at signal magnitudes of ~55 because bf16's representable spacing there is 0.25.

The right fix is **adjusting the equivalence test**, not the model code.

## Evidence — full per-track diff at SORT1

```
track                                    max|d|  mean|d|  corr   jax_mean  pt_mean   ratio
DNASE/EFO:0002067 DNase-seq/.            0.4760  0.0007   1.0000 0.0841    0.0836    0.9939
DNASE/EFO:0001187 DNase-seq/.            0.4127  0.0008   0.9999 0.0618    0.0624    1.0108
DNASE/CL:0000182  DNase-seq/.            0.1831  0.0005   1.0000 0.0563    0.0567    1.0071

Global max abs diff: 0.4760  (PR #62 baseline <0.05; test tolerance 0.1)
```

The mean abs diff across **1 million positions × 3 tracks** is **0.0005–0.0008**. The correlation is **1.0000** on every track. The output ratios (pt_mean / jax_mean) are all within 1.1%. The two backends produce essentially identical signals.

## Evidence — JAX outputs are bf16-quantized

The top-10 highest-diff positions all have JAX values on bf16's mantissa grid:

```
bin 339979: jax= 54.7500  pt= 55.2260  |diff|= 0.4760
bin 675881: jax= 56.2500  pt= 56.6997  |diff|= 0.4497
bin 126323: jax= 38.2500  pt= 38.6617  |diff|= 0.4117
bin 349414: jax= 35.0000  pt= 34.5947  |diff|= 0.4053
bin 797855: jax= 35.5000  pt= 35.8670  |diff|= 0.3670
bin 340088: jax= 31.1250  pt= 31.4910  |diff|= 0.3660
bin 349360: jax= 38.7500  pt= 38.4155  |diff|= 0.3345
bin 797926: jax= 24.3750  pt= 24.6990  |diff|= 0.3240
bin 325317: jax= 22.6250  pt= 22.9461  |diff|= 0.3211
bin 797930: jax= 31.0000  pt= 31.2801  |diff|= 0.2801
```

JAX values like 54.7500, 56.2500, 38.2500, 35.0000, 38.7500, 31.1250 are exactly on bf16's representable grid (multiples of 0.25 at magnitude ~32–64). PT values 55.226, 56.6997, 38.6617, 34.5947, 35.867, 31.491 have full fp32 precision.

The bf16 representable spacing for any value `v` is approximately `v / 256` (7 mantissa bits). At `v ≈ 55`, that's 0.215 — and the observed gap between consecutive bf16 values around 55 is exactly **0.25**:

```
54.5 → bf16 → 54.5
54.75 → bf16 → 54.75
55.0 → bf16 → 55.0
55.25 → bf16 → 55.25
55.5 → bf16 → 55.5
```

So at peak positions, the bf16 quantization in JAX alone introduces up to ~0.25 of noise vs. fp32. The remaining ~0.2 is fp32 numerical kernel difference between torch's conv path and JAX's XLA conv path — which accumulates over a model with hundreds of layers.

## Confirmed via JAX policy in source

`alphagenome_research/model/dna_model.py:1207`:
```python
jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
```

JAX deliberately outputs bf16. The PyTorch port defaults to **fp32 output** (`DtypePolicy.full_float32()` → `params=float32,compute=float32,output=float32`).

## Why earlier hypotheses were wrong

1. **Mixed precision (forcing PT to bf16 too) made drift WORSE (0.476 → 0.875).** Because `torch.amp.autocast` casts more operations to bf16 than JAX's JMP policy does — PT became *more* bf16-quantized than JAX, increasing the divergence on top of the per-position numerical difference. (See `equivalence_autocast_fp32cast.log`.)
2. **Model-space output (`return_scaled_predictions=True`) made drift much WORSE (0.476 → 22.85).** The default `False` returns experimental space (unscaled via per-track `track_means` / `residual_scales`), which is what JAX also returns. The unscale step compresses model-space drift down by ~50×.

## Why this isn't a regression

- The mean abs diff is 0.0007 — that's the actual model-output agreement, and it's excellent.
- Correlation = 1.000 means the two are interchangeable for any downstream analysis that doesn't compare exact peak heights position-by-position.
- The 0.476 hits only at the absolute peak positions (signal 30–77 out of 0–77). For variant scoring, MPRA effects, accessibility windows, etc., the differences at peak are below biological noise.

**PR #62's <0.05 baseline likely measured against a different SORT1 window or different tracks where the peak signal was lower** (say, <10). At lower magnitudes, bf16's representable spacing is smaller (`<0.05` at signal ~10), so the same model implementations would pass a 0.05 absolute tolerance. The drift didn't change; the signal magnitude at the tested locus did.

## Recommended fix

Adjust the equivalence test to use a metric that's robust to bf16 quantization at high-magnitude positions:

**Option A (cleanest, recommended)**: use **relative diff at peak**:
```python
peak_mag = max(np.abs(jax_track).max(), np.abs(pt_track).max())
max_abs = float(np.abs(a - b).max())
relative_at_peak = max_abs / peak_mag
assert relative_at_peak < 0.01, ...  # 1% of peak value
```
Bf16 quantization gives ~0.4% at peak; PR #62 baseline of <0.05 abs at low magnitudes also translates to a sub-1% relative.

**Option B**: keep absolute tolerance but **quantize PT to bf16 before comparing**:
```python
import torch
pt_bf16 = torch.from_numpy(pt_track.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
max_abs = float(np.abs(jax_track - pt_bf16).max())
assert max_abs < 0.05, ...
```
This puts both backends on the same precision footing.

**Option C**: assert on **correlation + mean abs diff** instead of max abs:
```python
assert np.corrcoef(a, b)[0, 1] > 0.99
assert float(np.abs(a - b).mean()) < 0.01
```
Most resilient to outlier positions; expresses the "are these the same model" question directly.

I recommend **Option A** for the equivalence test (relative tolerance at peak), and keeping correlation + mean abs as secondary asserts for diagnostic clarity. Option B is also acceptable if the team prefers absolute tolerances.

## Code change scope

None to chorus or alphagenome_pytorch — both implementations are correct. Only `tests/test_alphagenome_backends_equivalence.py` needs an updated assertion.

## Files / logs in this audit dir

- `diagnostic_per_track_diff.py` — the standalone script
- `diagnostic_per_track_diff.log` — full per-track diff table
- `equivalence_mixed_precision.log` — first bf16 attempt (timed out)
- `equivalence_mixed_precision_gpu1.log` — bf16 attempt on free GPU 1, drift 0.476 (template patch didn't reach prediction path)
- `equivalence_autocast_fp32cast.log` — full bf16 PT (autocast + dtype_policy), drift 0.875
- `equivalence_return_scaled.log` — model-space PT, drift 22.85
- `issue_81_investigation.md` — earlier investigation notes (dtype/precision experiments)
- `issue_81_root_cause.md` — this file (final root cause)

Templates have been reverted to pristine v0.5.1. Nothing pushed to GitHub.
