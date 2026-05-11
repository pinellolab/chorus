# Issue #81 (alphagenome JAX vs PT drift) — investigation notes from the ml007 GPU agent

**Date**: 2026-05-10 / 2026-05-11
**Tested against**: v0.5.1 tag (commit 907b564)
**Hardware**: ml007, NVIDIA A100-PCIE-40GB

## TL;DR

**Mixed-precision is NOT the cause.** Switching the PyTorch backend from full float32 to JAX-matching bfloat16 compute (via `DtypePolicy.mixed_precision()` + `torch.amp.autocast`) made drift **worse**, not better:

| PT precision config | max abs diff vs JAX at SORT1 |
|---|---|
| `DtypePolicy.full_float32()` (v0.5.1 default) | **0.4760** |
| `DtypePolicy.mixed_precision()` + `torch.amp.autocast(bfloat16)` | **0.8750** ← worse |
| PR #62 audit baseline (Apr 30, ~7a3ef7d) | < 0.05 |

This strongly implies that **JAX is also running fp32 at inference time** — it's NOT bf16 as the alphagenome-pytorch documentation hints. So step 3 of the Issue #81 handoff ("Investigate dtype/device defaults") is ruled out as the regression source.

## What I tested

Same `tests/test_alphagenome_backends_equivalence.py::test_jax_pt_chorus_api_equivalence_at_sort1` reproducer that the issue cites. All runs on ml007 GPU 1 (an idle A100-40), with `CHORUS_NO_TIMEOUT=1` and the HF token sourced from `~/.token_chorus_audit`.

Three configs, identical chorus code otherwise:

1. **Baseline (unchanged v0.5.1)**:
   - `AlphaGenome.from_pretrained(weights_path, device=device)` in both load and predict templates
   - No autocast wrapper
   - Result: max abs diff `0.4760` (same as Issue #81 report on Linux/CUDA)

2. **DtypePolicy.mixed_precision() — model only**:
   - Edit `load_template.py` AND `predict_template.py`: pass `dtype_policy=DtypePolicy.mixed_precision()` to `from_pretrained`
   - No autocast wrapper
   - Result: `Input type (CUDABFloat16Type) and weight type (torch.cuda.FloatTensor) should be the same` — the policy is just a config, doesn't autocast on its own.

3. **DtypePolicy.mixed_precision() + torch.amp.autocast(bfloat16)**:
   - Same template edit as (2)
   - Wrapped the `model(...)` call inside `torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)` in predict_template
   - Also cast output back to fp32 before numpy: `tensor.detach().to(torch.float32).cpu().numpy()` (numpy doesn't support bf16)
   - Result: max abs diff `0.8750` — **drift gets worse, not better**.

## What this rules in / rules out

- ✅ **Ruled out**: dtype/precision is the cause. Both backends already operate in fp32; forcing bf16 only adds quantization error.
- ✅ **Ruled out (from prior investigation in this audit's `report.md`)**: alphagenome-pytorch upstream commit drift. Installed commit `c5e77a37a` IS the latest on main (`Add Locon support to finetune CLI`, 2026-04-28) and was already main when PR #62 audited the <0.05 baseline.
- ❓ **Still suspect**: weight-conversion mismatch (the `gtca/alphagenome_pytorch` HF weights vs the live JAX model). If the PT weight bundle was converted from JAX months ago and JAX has since been retrained, you'd see exactly this kind of "same architecture, drifted predictions" pattern.
- ❓ **Still suspect**: torch version. `torch>=2.0.0` allows anything; installed is torch 2.11.0+cu130. Some torch kernel may have changed semantics (e.g. conv determinism flags, default flash attention path).
- ❓ **Still suspect**: alphagenome (JAX) package version. Pinned to `alphagenome==0.6.1` here; was that the version PR #62 measured against?

## Suggested next moves for the agent that picks this up

1. **Verify the JAX backend isn't quietly producing different numbers than PR #62.** Run JAX-only at SORT1 with PR #62's reference snapshot (if there is one) — confirms JAX side is stable. If JAX has drifted too, then the regression is in alphagenome/alphagenome-research/jax/jaxlib.
2. **Pin torch.** Test against torch 2.7 / 2.8 / 2.9 / 2.10 / 2.11 by rebuilding `chorus-alphagenome_pt` with the env yml pinning `torch==2.10.0` etc. Bisect.
3. **Compare PT vs JAX weights numerically.** Load the HF PT weight bundle (`gtca/alphagenome_pytorch`'s `alphagenome.pt`) and the JAX checkpoint, compare layer-by-layer. If a layer's weights mismatch by >1e-4 anywhere, that's the source.
4. **Operator-level diff.** Insert hooks in both backends and dump the activation at each layer for the SORT1 input. Find the first layer where they diverge. Identify which operator behaves differently (conv stride, attention scaling, layernorm epsilon, etc.).

## Decision frame for the user

If the drift turns out to be permanent (the PT weights are stale or torch kernels genuinely changed), the options remain:

- **Tag alphagenome_pt as experimental** in docs and on `chorus.create_oracle`. Add a runtime warning when called.
- **Remove alphagenome_pt** from v0.6.0; keep just the JAX backend (which is canonical).
- **Accept the drift and re-baseline.** Change the test's tolerance from `0.1` to `1.0` (would mask future regressions but reflects reality).

My recommendation is option (a) for now — keep alphagenome_pt available for users who specifically want PT compatibility (MPS users on Apple Silicon), but warn loudly. Then schedule the operator-level diff investigation as a separate follow-up.

## Log files

- `equivalence_mixed_precision_gpu1.log` — config (2), the dtype_policy-only attempt
- `equivalence_autocast.log` — config (3), autocast but with the bf16-numpy error
- `equivalence_autocast_fp32cast.log` — config (3) with the fp32 output cast, this is the `0.8750` result

All on `/PHShome/lp698/chorus/audits/2026-05-10_v0.5.0_scorched_earth_linux_cuda/`.

Working tree restored to pristine v0.5.1; no code changes pushed.
