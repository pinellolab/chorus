"""Pin the skip-experimental behavior of `chorus setup` (no args).

The PyTorch AlphaGenome backend (`alphagenome_pt`) is opt-in. The default
`chorus setup` flow must not auto-build its conda env or download its
~3.4 GB weights — users opt in explicitly via `chorus setup --oracle
alphagenome_pt` or by passing `--include-experimental` to the bare
`chorus setup` command.

Pin both the skip set and the call-site filter logic.
"""
from __future__ import annotations

from chorus.cli._setup_all import _SKIP_FROM_DEFAULT_SETUP


def test_alphagenome_pt_skipped_from_default_setup():
    """alphagenome_pt must stay in the skip-by-default set unless someone
    explicitly removes it (e.g. when promoting it from opt-in to default)."""
    assert "alphagenome_pt" in _SKIP_FROM_DEFAULT_SETUP, (
        "alphagenome_pt should be skipped by default `chorus setup` to "
        "avoid an unsolicited ~5 GB env build + ~3.4 GB weight download. "
        "Install with `chorus setup --oracle alphagenome_pt` or pass "
        "--include-experimental to `chorus setup`. If you intend to "
        "promote it to default, also update the README's 'Two AlphaGenome "
        "backends' section + the disk-space prerequisite line."
    )


def test_skip_set_does_not_contain_production_oracles():
    """Make sure no production oracle accidentally lands in the skip set —
    that would silently break the default install for everyone."""
    production = {"alphagenome", "borzoi", "chrombpnet", "enformer", "legnet", "sei"}
    assert _SKIP_FROM_DEFAULT_SETUP.isdisjoint(production), (
        f"Production oracles in skip set: "
        f"{_SKIP_FROM_DEFAULT_SETUP & production}"
    )
