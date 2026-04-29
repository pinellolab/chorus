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
