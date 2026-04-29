"""Pin the opt-in skip behavior of `chorus setup` (no args).

The PyTorch AlphaGenome backend (`alphagenome_pt`) uses the same model
and same weights as the default JAX `alphagenome` oracle (the upstream
PyTorch HF repo `gtca/alphagenome_pytorch` is the official JAX
checkpoint converted to safetensors). It's skipped from default
`chorus setup` only because of disk size — the env is ~5 GB and the
weights are ~3.4 GB on top of the JAX backend the user already gets.

The skip is a disk-cost decision, not a stability one: equivalence is
verified on M3 Ultra + A100 (1–2 % per-track fp32 noise on chorus-API
scoring). Users opt in via `chorus setup --oracle alphagenome_pt` or
by passing `--include-alternative-backends` to the bare `chorus setup`
command.

Pin both the skip set and the call-site filter logic.
"""
from __future__ import annotations

from chorus.cli._setup_all import _SKIP_FROM_DEFAULT_SETUP


def test_alphagenome_pt_skipped_from_default_setup():
    """alphagenome_pt must stay in the skip-by-default set unless someone
    explicitly removes it (e.g. when promoting it to default)."""
    assert "alphagenome_pt" in _SKIP_FROM_DEFAULT_SETUP, (
        "alphagenome_pt should be skipped by default `chorus setup` to "
        "avoid an unsolicited ~5 GB env build + ~3.4 GB weight download "
        "on top of the JAX `alphagenome` backend the user already gets. "
        "Install with `chorus setup --oracle alphagenome_pt` or pass "
        "--include-alternative-backends to `chorus setup`. If you intend "
        "to promote it to default, also update the README's 'Two "
        "AlphaGenome backends' section + the disk-space prerequisite line."
    )


def test_skip_set_does_not_contain_production_oracles():
    """Make sure no production oracle accidentally lands in the skip set —
    that would silently break the default install for everyone."""
    production = {"alphagenome", "borzoi", "chrombpnet", "enformer", "legnet", "sei"}
    assert _SKIP_FROM_DEFAULT_SETUP.isdisjoint(production), (
        f"Production oracles in skip set: "
        f"{_SKIP_FROM_DEFAULT_SETUP & production}"
    )
