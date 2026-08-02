"""Reservoir adds must be all-or-nothing per sampled position (#123).

Every builder wrapped its model calls **and** its per-track loop in one
``try/except``. An exception raised part-way through the loop left the tracks
visited before it incremented and the rest not — so tracks were ranked against
*different variant sets*.

The damage is in the shipped ``enformer_pertrack.npz``: ``effect_counts`` takes
**7** values, 9600-9606. That tight run of consecutive integers is the
fingerprint, and it is what distinguishes the bug from a legitimate spread:

===============  ==================  ==================  =========
oracle           effect_counts       adjacent pairs      verdict
===============  ==================  ==================  =========
enformer         9600...9606 (7)     **6**               partial credit
alphagenome      1697, 1909          0                   RNA vs windowed path
borzoi           6563, 9609          0                   different paths
chrombpnet       18672, 37344        0                   1x vs 2x sampling
cherimoya        18672               0                   uniform
===============  ==================  ==================  =========

So ``n_distinct`` alone cannot detect it — ``adjacent_pairs`` is the signal.

An AST scan finds the pattern at **11 sites across 6 builders** — enformer (2,
fixed here), alphagenome (2), borzoi (2), chrombpnet (2), sei (2), cherimoya (1).
Only enformer's data shows it having fired.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.background_sampling import (
    ReservoirSampler,
    StagedSamples,
    report_sampling_uniformity,
    sampling_uniformity,
)


# ---------------------------------------------------------------------------
# StagedSamples
# ---------------------------------------------------------------------------


def test_nothing_is_committed_until_commit_is_called():
    res = ReservoirSampler(n_tracks=3, capacity=100)
    staged = StagedSamples()
    for i in range(3):
        staged.add(i, float(i))
    assert res.total_samples() == 0
    staged.commit(res)
    assert list(res.get_counts()) == [1, 1, 1]


def test_a_dropped_position_leaves_every_track_untouched():
    """The actual bug: a mid-loop failure must not credit the earlier tracks."""
    res = ReservoirSampler(n_tracks=5, capacity=100)
    staged = StagedSamples()
    try:
        for i in range(5):
            if i == 3:
                raise RuntimeError("model blew up mid-loop")
            staged.add(i, 1.0)
    except RuntimeError:
        pass  # staged simply discarded
    else:  # pragma: no cover
        staged.commit(res)
    assert res.total_samples() == 0, "partial credit leaked"
    assert list(res.get_counts()) == [0, 0, 0, 0, 0]


def test_repeated_positions_keep_counts_uniform():
    res = ReservoirSampler(n_tracks=4, capacity=1000)
    for pos in range(50):
        staged = StagedSamples()
        for i in range(4):
            staged.add(i, float(pos))
        staged.commit(res)
    counts = list(res.get_counts())
    assert counts == [50] * 4
    assert sampling_uniformity(res.get_counts())["n_distinct"] == 1


def test_multiple_reservoirs_commit_together():
    """Baseline loops feed two reservoirs; both must be all-or-nothing."""
    summary = ReservoirSampler(n_tracks=2, capacity=100)
    perbin = ReservoirSampler(n_tracks=2, capacity=100)
    staged = StagedSamples()
    for i in range(2):
        staged.add(i, 1.0, reservoir=0)
        staged.add_batch(i, np.arange(4, dtype=np.float64), reservoir=1)
    assert summary.total_samples() == perbin.total_samples() == 0
    staged.commit(summary, perbin)
    assert list(summary.get_counts()) == [1, 1]
    assert list(perbin.get_counts()) == [4, 4]


def test_partial_failure_across_two_reservoirs_commits_neither():
    summary = ReservoirSampler(n_tracks=3, capacity=100)
    perbin = ReservoirSampler(n_tracks=3, capacity=100)
    staged = StagedSamples()
    try:
        for i in range(3):
            staged.add(i, 1.0, reservoir=0)
            if i == 2:
                raise ValueError("boom after staging summary but before perbin")
            staged.add_batch(i, np.ones(3), reservoir=1)
    except ValueError:
        pass
    else:  # pragma: no cover
        staged.commit(summary, perbin)
    assert summary.total_samples() == 0 and perbin.total_samples() == 0


def test_staged_len_reports_pending_work():
    staged = StagedSamples()
    assert len(staged) == 0
    staged.add(0, 1.0)
    staged.add_batch(0, [1.0, 2.0])
    assert len(staged) == 2


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------


def test_consecutive_run_is_flagged_as_suspect():
    """Enformer's real shape: 9600..9606."""
    stats = sampling_uniformity(np.array([9600, 9601, 9602, 9603, 9604, 9605, 9606]))
    assert stats["n_distinct"] == 7
    assert stats["adjacent_pairs"] == 6
    assert stats["suspect"] is True


@pytest.mark.parametrize("counts,label", [
    ([1697, 1909], "alphagenome RNA vs windowed"),
    ([6563, 9609], "borzoi"),
    ([18672, 37344], "chrombpnet 1x vs 2x"),
    ([18672], "cherimoya uniform"),
    ([9609], "legnet uniform"),
])
def test_legitimate_spreads_are_not_flagged(counts, label):
    """Separated clusters are design, not damage — must stay quiet."""
    stats = sampling_uniformity(np.array(counts))
    assert stats["suspect"] is False, f"{label} wrongly flagged"


def test_zero_count_tracks_are_ignored():
    stats = sampling_uniformity(np.array([0, 0, 500, 500]))
    assert stats["n_tracks"] == 2 and stats["n_distinct"] == 1
    assert sampling_uniformity(np.array([0, 0]))["n_tracks"] == 0


def test_reporter_escalates_only_on_the_fingerprint(caplog):
    res = ReservoirSampler(n_tracks=2, capacity=10_000)
    for i, n in enumerate((9605, 9606)):
        for _ in range(n):
            res.add(i, 1.0)
    logger = logging.getLogger("test-uniformity")
    with caplog.at_level(logging.INFO, logger="test-uniformity"):
        stats = report_sampling_uniformity(res, {"RuntimeError": 2}, "effect", logger)
    assert stats["suspect"] is True
    assert "partial-credit fingerprint" in caplog.text
    assert "dropped 2 position(s)" in caplog.text


def test_reporter_stays_quiet_on_a_legitimate_spread(caplog):
    res = ReservoirSampler(n_tracks=2, capacity=10_000)
    for i, n in enumerate((100, 400)):
        for _ in range(n):
            res.add(i, 1.0)
    logger = logging.getLogger("test-uniformity-ok")
    with caplog.at_level(logging.INFO, logger="test-uniformity-ok"):
        stats = report_sampling_uniformity(res, {}, "summary", logger)
    assert stats["suspect"] is False
    assert "fingerprint" not in caplog.text


# ---------------------------------------------------------------------------
# The builders
# ---------------------------------------------------------------------------


def _try_blocks_wrapping_reservoir_loops(path: str) -> list[int]:
    """Line numbers of ``try`` blocks whose body loops and adds to a reservoir."""
    tree = ast.parse(Path(path).read_text())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.For) and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("add", "add_batch")
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id.endswith(("reservoir", "res"))
                for n in ast.walk(sub)
            ):
                hits.append(node.lineno)
                break
    return hits


def test_enformer_no_longer_credits_tracks_partially():
    """The oracle whose shipped data shows the bug."""
    path = "scripts/build_backgrounds_enformer.py"
    assert _try_blocks_wrapping_reservoir_loops(path) == [], (
        "a try block still wraps a per-track loop that writes straight to a "
        "reservoir; stage the samples and commit after the loop"
    )
    src = Path(path).read_text()
    assert "StagedSamples()" in src
    assert "staged.commit(" in src
    assert "report_sampling_uniformity(" in src


@pytest.mark.xfail(
    reason="latent at 9 more sites across alphagenome, borzoi, chrombpnet, sei "
           "and cherimoya; fixed before the rebuild, since a transient failure "
           "there would recreate #123 in fresh data",
    strict=True,
)
def test_every_builder_stages_its_samples():
    offenders = {
        p: _try_blocks_wrapping_reservoir_loops(p)
        for p in sorted(Path("scripts").glob("build_backgrounds_*.py"))
        if _try_blocks_wrapping_reservoir_loops(str(p))
    }
    assert not offenders, offenders
