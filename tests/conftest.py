"""Suite-wide fixtures.

Currently one job: stop a test that reconstructs the MCP `OracleStateManager`
singleton from corrupting every test that runs after it.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_oracle_state_singleton():
    """Snapshot and restore `OracleStateManager._instance` around every test.

    `OracleStateManager` is a singleton, and it resolves the reference genome
    exactly once, in `__init__`:

        gm = GenomeManager()
        if gm.is_genome_downloaded("hg38"):
            self._reference_fasta = str(gm.get_genome_path("hg38"))

    `tests/test_mcp.py` has seven tests that need a state manager built under a
    mocked `GenomeManager`. Each does the only thing that can force a rebuild --
    `OracleStateManager._instance = None`, then constructs one inside the patch
    context, where `is_genome_downloaded` returns False. So the fresh singleton
    gets `_reference_fasta = None`, and when the patch is lifted at the end of the
    test, **the singleton keeps that None**. Nothing restores it, because nothing
    saved it.

    Every later test that scores a genomic *interval* then fails, because
    `_reference_fasta` is what the state manager passes to an oracle as
    `reference_fasta`, and without it `LegNetOracle._predict` raises

        ValueError: Reference FASTA required for genomic coordinates.

    Measured cost before this fixture: 7 failures in `test_mcp_scoring_tools.py`,
    every one of them passing in isolation. Verified by bisecting the file list --
    `test_mcp.py` alone reproduces it, and the four other candidates in the same
    range do not. It looked like GPU contention and was briefly mislabelled as
    such, which is the real reason this is a fixture and not a note: an
    order-dependent failure that only appears in the full sweep is exactly the
    kind that gets waved through as flake.

    Snapshot-and-restore rather than reset-to-None on purpose. Restoring the same
    object preserves identity, so a module-scoped fixture that loaded an oracle
    into the singleton -- `test_mcp_scoring_tools.loaded` does -- still sees its
    own loaded oracle across the tests in its file. Resetting to None would force
    a real model reload per test.
    """
    try:
        from chorus.mcp.state import OracleStateManager
    except Exception:
        # Nothing to protect if the MCP layer is not importable in this env.
        yield
        return

    before = getattr(OracleStateManager, "_instance", None)
    try:
        yield
    finally:
        if getattr(OracleStateManager, "_instance", None) is not before:
            OracleStateManager._instance = before
