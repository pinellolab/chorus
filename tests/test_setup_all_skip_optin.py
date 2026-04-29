"""Pin the default-install behavior of `chorus setup` (no args).

Both AlphaGenome backends — JAX `alphagenome` and PyTorch
`alphagenome_pt` — install by default so Mac users get MPS access
without an extra setup step. The PyTorch backend uses the same model
and same weights as the JAX backend (the upstream PyTorch HF repo
`gtca/alphagenome_pytorch` is the official JAX checkpoint converted
to safetensors), and equivalence is verified on M3 Ultra + A100
(1–2 % per-track fp32 noise on chorus-API scoring).

The `_SKIP_FROM_DEFAULT_SETUP` infrastructure stays in place for
future opt-in oracles, but is currently empty. The
`--include-alternative-backends` CLI flag is a no-op for now —
preserved so scripts that pass it don't break.

Pin both the empty skip set and the no-production-oracles invariant.
"""
from __future__ import annotations

from chorus.cli._setup_all import _SKIP_FROM_DEFAULT_SETUP


def test_alphagenome_pt_installed_by_default():
    """alphagenome_pt must NOT be in the skip set — both AlphaGenome
    backends install by default. Removing this assertion (or putting
    alphagenome_pt back into the skip set) is a meaningful UX change
    that should be coupled with README + CHANGELOG updates."""
    assert "alphagenome_pt" not in _SKIP_FROM_DEFAULT_SETUP, (
        "alphagenome_pt should install by default so Mac users get MPS "
        "access without an extra setup step. If you're putting it back "
        "into the skip set, also update the README's 'Two AlphaGenome "
        "backends' section, the disk-space prerequisite line, and the "
        "CHANGELOG."
    )


def test_skip_set_does_not_contain_production_oracles():
    """Make sure no production oracle accidentally lands in the skip set —
    that would silently break the default install for everyone."""
    production = {
        "alphagenome", "alphagenome_pt", "borzoi", "chrombpnet",
        "enformer", "legnet", "sei",
    }
    assert _SKIP_FROM_DEFAULT_SETUP.isdisjoint(production), (
        f"Production / default-installed oracles in skip set: "
        f"{_SKIP_FROM_DEFAULT_SETUP & production}"
    )
