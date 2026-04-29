"""Static regression tests for the AlphaGenome PyTorch predict template.

These tests grep the predict_template.py source for known-bad patterns
caught during the v62 stress-test audit (2026-04-29). They're fast (no
env, no model load, no network) so they run in the default fast suite
and would have caught F2 in CI.

The integration test in test_alphagenome_backends_equivalence.py hits
the upstream port directly (not chorus's predict_template), so it
won't catch chorus-side template regressions like this one.
"""
from __future__ import annotations

from pathlib import Path

PREDICT_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "chorus" / "oracles" / "alphagenome_pt_source"
    / "templates" / "predict_template.py"
)


def test_predict_template_passes_organism_index_as_tensor():
    """v62-audit F2 (2026-04-29): the production predict template was
    passing ``organism_index=0`` (int) to the AlphaGenome PyTorch port,
    which raised ``TypeError: embedding(): argument 'indices' (position 2)
    must be Tensor, not int`` at the first model call. The model expects
    a torch.Tensor of shape (batch,) with dtype long.

    Pin the fix so future hand-edits to the template don't accidentally
    revert to passing a Python int.
    """
    src = PREDICT_TEMPLATE.read_text()
    assert "organism_index=0" not in src, (
        "predict_template.py must not pass organism_index=0 — the PyTorch "
        "port's organism_embed Embedding layer requires a Tensor input. "
        "Use organism_index=torch.tensor([0], dtype=torch.long, device=device) "
        "(see v62-audit F2)."
    )
    # The fix should use torch.tensor wrapped around the organism int.
    assert "organism_index=torch.tensor(" in src, (
        "predict_template.py should pass organism_index as a torch.Tensor "
        "(see v62-audit F2 in audits/2026-04-29_alphagenome_pt_stress_test/)."
    )


def test_predict_template_organism_index_uses_long_dtype():
    """The organism embedding indexes a small lookup table — needs long
    (int64) dtype for torch.embedding."""
    src = PREDICT_TEMPLATE.read_text()
    # Permissive — different formattings of the dtype kwarg are OK as
    # long as one of them appears near the organism_index call.
    snippet_idx = src.find("organism_index=torch.tensor(")
    if snippet_idx == -1:
        # The other test will fail with a clearer message; bail here.
        return
    nearby = src[snippet_idx:snippet_idx + 200]
    assert "torch.long" in nearby, (
        "organism_index tensor should be dtype=torch.long; see v62-audit F2."
    )


# ---------------------------------------------------------------------------
# Direct unit test of the MPS rope substitution (lives in chorus-alphagenome_pt
# env — skips in chorus base env via importorskip).
# ---------------------------------------------------------------------------

def test_mps_compat_apply_rope_matches_upstream_on_cpu():
    """The MPS-compat shim replaces ``alphagenome_pytorch.attention.apply_rope``
    with a substitute that swaps the unimplemented-on-MPS ``torch.logspace``
    for ``torch.exp(linspace * log(10))``. The substitute should be
    bit-equivalent to the original (within float32 ulp) on any backend.

    This test runs on CPU (where the upstream original works fine) and
    asserts the patched and original produce indistinguishable outputs.
    Two value adds:

    1. Catches a bad refactor of ``_patched_apply_rope`` itself — if the
       substitute drifts from the upstream math, this test fails fast
       without needing a 4 GB model load.

    2. **Tells us when the shim is no longer needed**: if upstream
       PyTorch ships ``aten::logspace.out`` for MPS (or upstream
       alphagenome-pytorch removes the logspace dependency), the test
       still passes — but at that point we can also drop ``_mps_compat``
       and simplify. The audit at
       ``audits/2026-04-29_alphagenome_pt_stress_test/`` documents the
       indirect evidence (524 kb chorus-API equivalence on MPS); this
       test is the direct guarantee.
    """
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import pytest
    pytest.importorskip("alphagenome_pytorch")

    import torch
    import alphagenome_pytorch.attention as agpt_attn

    # Capture the un-patched original BEFORE chorus's shim runs install().
    original_apply_rope = agpt_attn.apply_rope

    # Now install the chorus shim — it stashes original_apply_rope on the
    # module and replaces ``apply_rope`` with the substitute.
    from chorus.oracles.alphagenome_pt_source import _mps_compat
    patched_apply_rope = _mps_compat._patched_apply_rope

    torch.manual_seed(0)
    B, S, H, C = 1, 64, 4, 8  # tiny — model dim 8 keeps test under 100 ms
    x = torch.randn(B, S, H, C, dtype=torch.float32, device="cpu")

    # Case 1: positions=None (uses default arange)
    a = original_apply_rope(x.clone(), positions=None,
                            max_position=_mps_compat._MAX_RELATIVE_DISTANCE)
    b = patched_apply_rope(x.clone(), positions=None,
                           max_position=_mps_compat._MAX_RELATIVE_DISTANCE)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
        f"shim diverges from upstream apply_rope (positions=None): "
        f"max abs diff = {(a - b).abs().max().item():.3e}"
    )

    # Case 2: explicit positions
    pos = torch.arange(S, dtype=torch.float32).unsqueeze(0)
    a = original_apply_rope(x.clone(), positions=pos)
    b = patched_apply_rope(x.clone(), positions=pos)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
        f"shim diverges from upstream apply_rope (custom positions): "
        f"max abs diff = {(a - b).abs().max().item():.3e}"
    )

    # Case 3: float16 input — chorus's predict path may downcast
    x16 = x.to(torch.float16)
    a = original_apply_rope(x16.clone(), positions=None)
    b = patched_apply_rope(x16.clone(), positions=None)
    # fp16 has worse ulp; loosen rtol modestly
    assert torch.allclose(a, b, rtol=2e-3, atol=1e-3), (
        f"shim diverges from upstream apply_rope (fp16): "
        f"max abs diff = {(a - b).abs().max().item():.3e}"
    )
