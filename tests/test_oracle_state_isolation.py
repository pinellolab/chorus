"""The MCP state singleton must survive a test that rebuilds it under a mock.

`OracleStateManager` resolves the reference genome once, in `__init__`. Any test
that sets `_instance = None` and reconstructs it inside a patched
`GenomeManager` therefore leaves a singleton whose `_reference_fasta` is None,
and every later test that scores a genomic interval fails with

    ValueError: Reference FASTA required for genomic coordinates.

That happened: 7 failures in `test_mcp_scoring_tools.py`, all of which passed in
isolation, caused by `test_mcp.py` running first. `tests/conftest.py` now
snapshots and restores the singleton around every test.

These tests guard the fixture itself. Without them the fixture could be deleted
or narrowed and nothing would notice until the next full-suite run -- and an
order-dependent failure that only shows up in the full sweep is the kind that
gets dismissed as flake. It was, briefly, mislabelled as GPU contention here.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

state_mod = pytest.importorskip("chorus.mcp.state")
OracleStateManager = state_mod.OracleStateManager


def test_reconstructing_under_a_mocked_genome_does_not_outlive_the_test():
    """This is the polluting pattern, run deliberately.

    If the conftest fixture is working, the damage is confined to this test: the
    singleton this test leaves behind gets replaced by the one that was there
    before it ran.
    """
    with patch("chorus.mcp.state.GenomeManager") as gm:
        gm.return_value.is_genome_downloaded.return_value = False
        OracleStateManager._instance = None
        mgr = OracleStateManager()
        assert mgr.reference_fasta is None, (
            "the mock did not take effect, so this test is not exercising the "
            "pollution it exists to reproduce"
        )


def test_the_singleton_still_resolves_a_genome_after_that():
    """Runs immediately after the polluting test, in file order.

    Before the fixture, this is where the corruption became visible. `_instance`
    is whatever the fixture restored, so constructing a manager must once again
    see the real `GenomeManager`.
    """
    mgr = OracleStateManager()
    if not state_mod.GenomeManager().is_genome_downloaded("hg38"):
        pytest.skip("hg38 not downloaded in this environment")
    assert mgr.reference_fasta is not None, (
        "the state singleton is still carrying reference_fasta=None from the test "
        "above -- tests/conftest.py::_restore_oracle_state_singleton is not "
        "restoring it, and every interval-scoring test after test_mcp.py will fail"
    )
    assert "hg38" in str(mgr.reference_fasta)


def test_restore_preserves_object_identity_not_just_a_fresh_instance():
    """Why the fixture snapshots rather than resetting to None.

    A module-scoped fixture that loads an oracle into the singleton -- as
    `test_mcp_scoring_tools.loaded` does -- must keep seeing *its own* loaded
    oracle across the tests in its file. Resetting to None between tests would
    discard it and force a real model reload each time. So restoring the
    identical object is part of the contract, not an implementation detail.
    """
    first = OracleStateManager()
    second = OracleStateManager()
    assert first is second, (
        "OracleStateManager stopped being a singleton, which changes what the "
        "conftest fixture needs to restore"
    )
