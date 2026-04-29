"""MPS compatibility shim for the alphagenome-pytorch port.

PyTorch's MPS backend doesn't yet implement ``aten::logspace.out``
(tracked at https://github.com/pytorch/pytorch/issues/141287), so the
upstream port falls back to CPU for the rope-attention base-frequency
computation on every attention block. At 1 MB context this means
dozens of MPS↔system-RAM round trips per forward pass, dominating
runtime and producing a ~3× regression vs CPU.

The fix is mathematical, not architectural: ``torch.logspace`` simply
computes ``base ** torch.linspace(start, end, steps)``, both of which
ARE implemented on MPS. Using ``torch.exp(linspace * log(base))`` is
bit-equivalent within IEEE float32 ulp.

Apply by importing this module before invoking the model:

    import chorus.oracles.alphagenome_pt_source._mps_compat  # noqa: F401

The patch is idempotent and safe on non-MPS devices (the substitute
formula is mathematically identical to ``torch.logspace`` on any
backend; we just bypass the unimplemented kernel).
"""

from __future__ import annotations

import math

import torch

import alphagenome_pytorch.attention as _agpt_attn

_orig_apply_rope = _agpt_attn.apply_rope
_MAX_RELATIVE_DISTANCE = getattr(_agpt_attn, "_MAX_RELATIVE_DISTANCE", 8192)
_patched_marker = "_chorus_mps_compat_patched"


def _patched_apply_rope(x, positions=None, max_position=_MAX_RELATIVE_DISTANCE, inplace=False):
    B, S, H, C = x.shape
    compute_dtype = x.dtype
    if positions is None:
        positions = torch.arange(
            S, device=x.device, dtype=compute_dtype
        ).unsqueeze(0)
    elif positions.dtype != compute_dtype:
        positions = positions.to(compute_dtype)

    num_freq = C // 2

    log10_n = math.log10(max(max_position - num_freq + 1, 1))
    base_freqs = torch.exp(
        torch.linspace(
            0.0, log10_n, num_freq, device=x.device, dtype=compute_dtype
        )
        * math.log(10.0)
    )
    denom = (
        torch.arange(num_freq, device=x.device, dtype=compute_dtype)
        + base_freqs
    )
    inv_freq = 1.0 / denom

    theta = torch.einsum("bs,f->bsf", positions, inv_freq)
    theta = torch.repeat_interleave(theta, 2, dim=-1).unsqueeze(2)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    if inplace:
        return _agpt_attn._apply_rope_inplace(x, cos_theta, sin_theta)

    x_rotated = torch.stack(
        [-x[..., 1::2], x[..., ::2]], dim=-1
    ).flatten(start_dim=-2)
    return x * cos_theta + x_rotated * sin_theta


def install() -> None:
    """Install the rope patch (idempotent)."""
    if getattr(_agpt_attn, _patched_marker, False):
        return
    _agpt_attn.apply_rope = _patched_apply_rope
    setattr(_agpt_attn, _patched_marker, True)


# Apply on import — chorus's templates simply import this module.
install()
