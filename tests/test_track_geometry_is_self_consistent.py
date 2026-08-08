"""A track's declared resolution must describe the array it actually holds.

Two fixes are pinned here, and one deliberate non-fix.

**LegNet declared the sliding stride, not the bin width.** ``self.bin_size`` is
the step (default 50), which is right for a multi-window query -- a 300 bp region
yields 6 values at stride 50 -- but wrong for the single-window case. And that
case is the default conversational/MCP path: ``base.py`` widens a point query to
exactly 200 bp, LegNet returns ONE scalar, and resolution 50 over a 200 bp
interval implies four bins that do not exist. Consequences measured before the
fix: ``pos2bin`` returned **2** for a length-1 array, so ``score_region`` and
``score_variant_effect(at_variant=True)`` both answered ``None``, and the IGV
feature was drawn 50 bp wide and 76 bp left-shifted. ``mcp/server.py`` documents
that arithmetic in two places and adds an explanatory note instead of fixing it.

**``pos2bin`` had no array-bounds guard.** Its checks were on genomic coordinates
only, so any track whose declared resolution overstated its sampling got an index
past the end of ``values`` -- silently, because callers turn a bad index into a
``None`` score rather than an error.

**And the thing NOT to fix:** ``pos2bin`` returns a bin one to the right of the
base the caller names (1-based ``position`` against a 0-based ``reference.start``).
Every background builder carries the identical slip, so the query window and the
null window are the same genomic span and a percentile ranks like against like.
Correcting only one side would move the numerator off its null. The test at the
bottom exists so that nobody removes half of a matched pair.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.core.interval import GenomeRef, Interval, Sequence
from chorus.core.result import OraclePredictionTrack


def _track(values, resolution, chrom="chr1", start=1_000_000):
    """A minimal track with a genomic prediction interval."""
    iv = Interval.make(GenomeRef(
        chrom=chrom, start=start, end=start + len(values) * resolution, fasta=None,
    ))
    return OraclePredictionTrack(
        source_model="legnet", assay_id="LentiMPRA:HepG2",
        track_id="T", assay_type="LentiMPRA", cell_type="HepG2",
        query_interval=iv, prediction_interval=iv, input_interval=iv,
        resolution=resolution, values=np.asarray(values, dtype=float),
        metadata=None,
    )


# ---------------------------------------------------------------------------
# pos2bin bounds
# ---------------------------------------------------------------------------


def test_pos2bin_never_returns_an_index_past_the_array():
    """The guard that was missing.

    A length-1 array declared at resolution 50 spans 50 bp by the interval's own
    arithmetic, but a caller asking about a position 100 bp in used to get bin 2.
    """
    t = _track([1.0], resolution=50)
    inside = t.pos2bin("chr1", 1_000_010)
    assert inside == 0
    # Anything the interval admits must land in range or be refused outright.
    start = t.prediction_interval.reference.start
    end = t.prediction_interval.reference.end
    for p in range(start, end + 2):
        b = t.pos2bin("chr1", p)
        assert b is None or 0 <= b < len(t.values), f"pos2bin({p}) = {b}"


def test_pos2bin_refuses_another_chromosome_and_far_positions():
    t = _track([1.0, 2.0, 3.0], resolution=10)
    assert t.pos2bin("chr2", 1_000_005) is None
    assert t.pos2bin("chr1", 1) is None
    assert t.pos2bin("chr1", 9_999_999) is None


def test_a_track_whose_resolution_overstates_its_sampling_is_refused_not_mangled():
    """4x overstatement -- exactly LegNet's old shape -- must not index past the end."""
    t = _track([0.37], resolution=50)
    t.prediction_interval = Interval.make(GenomeRef(
        chrom="chr1", start=1_000_000, end=1_000_200, fasta=None,
    ))
    for p in (1_000_000, 1_000_100, 1_000_199):
        b = t.pos2bin("chr1", p)
        assert b is None or b == 0, f"pos2bin({p}) = {b} for a length-1 array"


# ---------------------------------------------------------------------------
# LegNet's declared resolution, against the real model
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("span,expect_n,expect_res", [
    (1, 1, 200),      # native/point query: ONE value over a 200 bp window
    (300, 6, 50),     # multi-window: stride 50 is correct here
    (400, 8, 50),
])
def test_legnet_resolution_times_length_equals_its_interval(span, expect_n, expect_res):
    """``n_values * resolution == len(prediction_interval)``, the invariant that
    makes ``positions`` and ``pos2bin`` mean anything."""
    pytest.importorskip("torch")
    try:
        from chorus.oracles.legnet import LegNetOracle
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"legnet not importable here: {exc}")

    from pathlib import Path

    from chorus.core.globals import CHORUS_DATA_DIR
    genome = CHORUS_DATA_DIR / "genomes" / "hg38.fa"
    if not Path(genome).exists():
        pytest.skip("hg38.fa missing")

    oracle = LegNetOracle(use_environment=False, reference_fasta=str(genome))
    try:
        oracle.load_pretrained_model()
    except Exception as exc:
        pytest.skip(f"legnet weights unavailable ({type(exc).__name__}); needs chorus-legnet")

    pos = 109_274_968
    half = span // 2
    interval = ("chr1", pos - max(half, 1), pos - max(half, 1) + max(span, 1))
    pred = oracle.predict(interval, assay_ids=["LentiMPRA:HepG2"])

    for _, track in pred.items():
        n, res = len(track.values), track.resolution
        assert n == expect_n, f"expected {expect_n} values for a {span} bp query, got {n}"
        assert res == expect_res, f"expected resolution {expect_res}, got {res}"
        assert n * res == len(track.prediction_interval), (
            f"{n} values x {res} bp != {len(track.prediction_interval)} bp interval -- "
            f"the declared resolution does not describe the array"
        )
        b = track.pos2bin("chr1", pos)
        assert b is None or 0 <= b < n, f"pos2bin returned {b} for a length-{n} array"


# ---------------------------------------------------------------------------
# The deliberate non-fix
# ---------------------------------------------------------------------------


def test_pos2bin_keeps_its_one_based_offset_on_purpose():
    """Do NOT "fix" this in isolation. It is half of a matched pair.

    ``position`` is 1-based and ``reference.start`` is 0-based, and pos2bin does
    not convert, so the bin is one to the right of the named base. Every
    background builder makes the same slip -- fetching ``(pos - half, pos + half)``
    with a 1-based pos against 0-based pysam, then centring on ``L // 2`` -- so
    the query window and the null window cover the identical genomic span:

        builder null : [109274718, 109275219)
        query        : [109274718, 109275219)

    Converting here alone shifts the query 1 bp off its null and makes every
    percentile slightly wrong. If the convention is corrected it must be one
    commit touching the builders too, re-deriving the CDFs. Cost measured if
    done: ~0.03% on ref/alt, ~2e-4 on log2FC, across 15 committed artefacts.

    This test fails if someone converts here without the other half, which is the
    only outcome worse than the present inconsistency.
    """
    t = _track([0.0] * 2114, resolution=1, start=109_273_910)
    # The variant's own 0-based index inside this window is 1057.
    assert t.pos2bin("chr1", 109_274_968) == 1058, (
        "pos2bin's 1-based/0-based offset changed. If that was deliberate, the "
        "background builders must be re-centred and the CDFs rebuilt in the SAME "
        "commit -- otherwise the query no longer ranks against a matching null. "
        "See docs/BACKGROUND_NULL_PROTOCOL.md and this test's docstring."
    )
