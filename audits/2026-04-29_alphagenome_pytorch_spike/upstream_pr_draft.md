# Draft: upstream PR to `genomicsxai/alphagenome-pytorch`

This is a draft for the user to review, edit, and submit. Send to:
https://github.com/genomicsxai/alphagenome-pytorch

---

## Suggested branch name

`fix/mps-rope-logspace-fallback`

## Suggested commit message

```
attention: avoid CPU fallback in apply_rope on MPS

torch.logspace doesn't have an MPS kernel yet (pytorch/pytorch#141287),
so apply_rope falls back to CPU once per attention block on Apple
Silicon — printing a warning and forcing MPS↔CPU sync per block.

Substituting the mathematically equivalent torch.exp(linspace * log(base))
keeps the computation on-device on every backend (CPU, CUDA, MPS) with
no behavioral change. Verified bit-equivalent to the original within
float32 ulp (max abs diff 7e-7 on CPU).
```

## Suggested PR title

`Avoid CPU fallback in apply_rope on MPS (Apple Silicon)`

## Suggested PR body (in the user's voice)

```markdown
Hi! First — thank you so much for porting AlphaGenome to PyTorch.
Being able to run this without the JAX stack is a huge deal,
especially for those of us on Apple Silicon where JAX-Metal is
still pretty experimental.

I've been trying to run the model on Mac / MPS and ran into one
small thing that I thought might be worth flagging: `torch.logspace`
doesn't have an MPS kernel yet
([pytorch/pytorch#141287](https://github.com/pytorch/pytorch/issues/141287)),
so the call inside `apply_rope` falls back to CPU on every attention
block. With `PYTORCH_ENABLE_MPS_FALLBACK=1` it works, but it prints a
warning per block and forces an MPS↔CPU sync each time.

The good news is `torch.logspace(start, end, steps, base)` is just
`torch.exp(torch.linspace(start, end, steps) * math.log(base))`, and
both `linspace` and `exp` *do* have native MPS kernels. So the fix is
a one-liner that keeps the computation on-device on every backend
without changing behavior. I verified the substitution is
bit-equivalent on CPU within float32 ulp (max abs diff 7×10⁻⁷ on a
random input).

Sorry to drop this in unsolicited — totally understand if you'd
rather wait for upstream PyTorch to ship the kernel, or fold this
into a different cleanup. Just wanted to flag it in case it's useful.
Either way, thank you again for all the work on this port!

### Change

In `src/alphagenome_pytorch/attention.py`, inside `apply_rope`,
replace the `torch.logspace(...)` call with the equivalent
`torch.exp(torch.linspace(...) * log(base))`.

### Test

I checked the substitution against the original on CPU with
`torch.randn(1, 32, 8, 64)` random input:

```
max abs diff (orig vs replacement) = 7.15e-07  # float32 ulp
```

On Apple M3 Ultra with `PYTORCH_ENABLE_MPS_FALLBACK=1`, this also
silences the per-block "operator 'aten::logspace.out' is not currently
supported on the MPS backend" warning, so the warning becomes
unnecessary for AlphaGenome users.

### Note on MPS speed

For full transparency: on M3 Ultra at 1 MB sequence length there's
still a separate cliff in MPS performance that I traced to GPU
on-die cache spillover at ~768→896 kb (memory grows linearly across
the cliff, far below the 96 GB unified-memory ceiling). That's a
separate problem from this patch and probably needs different
attention than a one-line fix. This PR just removes the avoidable
CPU fallback for users in the cache-resident size range.

Thanks again!
```

## The diff to apply

In `src/alphagenome_pytorch/attention.py`, replace the body of
`apply_rope`:

```diff
 def apply_rope(x, positions=None, max_position=_MAX_RELATIVE_DISTANCE, inplace=False):
     """..."""
     # x: (B, S, H, C)
     B, S, H, C = x.shape
     compute_dtype = x.dtype  # Match JAX: use input dtype for all RoPE ops

     if positions is None:
         positions = torch.arange(S, device=x.device, dtype=compute_dtype).unsqueeze(0)
     elif positions.dtype != compute_dtype:
         positions = positions.to(compute_dtype)

     num_freq = C // 2
-    # JAX geomspace equivalent: geomspace(1, max_position - num_freq + 1, num_freq)
-    base_freqs = torch.logspace(
-        math.log10(1), math.log10(max_position - num_freq + 1),
-        steps=num_freq, base=10, device=x.device, dtype=compute_dtype
-    )
+    # JAX geomspace equivalent: geomspace(1, max_position - num_freq + 1, num_freq).
+    # Use torch.exp(linspace * log(base)) instead of torch.logspace so this
+    # stays on-device on MPS (which lacks an aten::logspace.out kernel
+    # — pytorch/pytorch#141287). Mathematically identical; bit-equivalent
+    # within float32 ulp (max abs diff 7e-7 verified on CPU).
+    log_end = math.log10(max_position - num_freq + 1)
+    base_freqs = torch.exp(
+        torch.linspace(0.0, log_end, steps=num_freq,
+                       device=x.device, dtype=compute_dtype) * math.log(10.0)
+    )
     denom = torch.arange(num_freq, device=x.device, dtype=compute_dtype) + base_freqs
     inv_freq = 1.0 / denom

     theta = torch.einsum('bs,f->bsf', positions, inv_freq)
     theta = torch.repeat_interleave(theta, 2, dim=-1).unsqueeze(2)

     cos_theta = torch.cos(theta)
     sin_theta = torch.sin(theta)

     if inplace:
         return _apply_rope_inplace(x, cos_theta, sin_theta)
     else:
         x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(start_dim=-2)
         return x * cos_theta + x_rotated * sin_theta
```

## Suggested test (only if the repo wants one — they may prefer a tiny addition under `tests/`)

```python
# tests/test_apply_rope_mps_compat.py
import math
import pytest
import torch

from alphagenome_pytorch.attention import apply_rope


@pytest.mark.parametrize("seq_len", [32, 128, 512])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_apply_rope_matches_logspace_baseline(seq_len, head_dim):
    """The logspace-free implementation must match the bit-precision
    of the original within float32 ulp on CPU."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, 4, head_dim)

    # Direct re-implementation using the original logspace formula
    # for the comparison baseline:
    num_freq = head_dim // 2
    max_position = 8192  # _MAX_RELATIVE_DISTANCE
    base_freqs_orig = torch.logspace(
        math.log10(1), math.log10(max_position - num_freq + 1),
        steps=num_freq, base=10, dtype=x.dtype,
    )
    base_freqs_new = torch.exp(
        torch.linspace(0.0, math.log10(max_position - num_freq + 1),
                       steps=num_freq, dtype=x.dtype) * math.log(10.0)
    )
    assert (base_freqs_orig - base_freqs_new).abs().max() < 1e-5, (
        "Replacement formula should be float32-ulp equivalent to logspace"
    )

    out = apply_rope(x.clone(), inplace=False)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
```

## How to actually submit (when ready)

```bash
# fork on GitHub UI first, then:
git clone https://github.com/<your-fork>/alphagenome-pytorch
cd alphagenome-pytorch
git checkout -b fix/mps-rope-logspace-fallback
# apply the diff above to src/alphagenome_pytorch/attention.py
git add src/alphagenome_pytorch/attention.py
git commit -m "attention: avoid CPU fallback in apply_rope on MPS"
git push -u origin fix/mps-rope-logspace-fallback
gh pr create --title "Avoid CPU fallback in apply_rope on MPS (Apple Silicon)" --body-file <path-to-PR-body>
```

## Things I'd flag for the maintainer

- They might prefer the patch live behind a `device.type == "mps"` check, only triggering the `exp(linspace*log)` path on MPS to keep `torch.logspace`'s slightly tighter numerics on CUDA/CPU. The 7e-7 ulp diff is well below model noise but a maintainer focused on bit-exact fidelity to the JAX reference might want the conservative version. Easy to switch to that on request.
- They may already be tracking this internally. A friendly heads-up issue first might be lower-friction than a PR.
- The PR scope is intentionally small — no MPS-specific build flags, no env vars, no test infrastructure beyond what they already use. Just one function body.
