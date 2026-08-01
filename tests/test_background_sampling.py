"""The shared sampling primitives must reproduce every builder's copy exactly.

This is the gate for #125's extract-then-migrate: a builder may only be switched
to ``chorus.analysis.background_sampling`` once the shared implementation is
proved to behave identically to the copy it replaces. Proving it here, before any
builder is touched, means a later migration PR that changes a number is a
regression rather than an ambiguity.

The comparison is against the *live source* of each builder, extracted by AST and
executed in an isolated namespace. That is deliberate: a test that compared
against a pasted-in snapshot would pass forever while the real copies drifted.
The idiom follows ``tests/test_cherimoya.py:581``, which parses the builder
rather than importing it — several builders do work at module scope and cannot
be imported.
"""

from __future__ import annotations

import ast
import glob
import math
import random
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.background_sampling import (
    DEFAULT_CAPACITY,
    DEFAULT_CDF_POINTS,
    DEFAULT_MAX_N_FRACTION,
    ReservoirSampler,
    compute_effect,
    get_window_slice,
    one_hot_encode,
    score_window_sum,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDERS = sorted(glob.glob(str(REPO_ROOT / "scripts" / "build_backgrounds_*.py")))


def _extract_class(path: str, name: str):
    """Exec one class out of a module's source, without importing the module."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            ns: dict = {"np": np, "numpy": np, "random": random, "math": math}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns[name]
    return None


def _builder_ids():
    return [Path(p).stem.replace("build_backgrounds_", "") for p in BUILDERS]


def test_there_are_builders_to_compare_against():
    """Guard the guard — a bad glob would make every comparison below vacuous."""
    assert len(BUILDERS) >= 8, f"found only {len(BUILDERS)} builders"


@pytest.mark.parametrize("path", BUILDERS, ids=_builder_ids())
def test_reservoir_matches_each_builders_copy(path):
    """Same stream, same seed, same capacity -> byte-identical CDF and counts.

    Runs enough values to cross the capacity boundary, so the replacement branch
    of algorithm R is exercised rather than just the fill branch. If a builder's
    copy diverged in *which* samples it retains, the CDF differs here even though
    no arithmetic changed — which is exactly the failure mode a migration must
    not introduce.
    """
    theirs_cls = _extract_class(path, "ReservoirSampler")
    if theirs_cls is None:
        pytest.skip("no ReservoirSampler in this builder")

    n_tracks, capacity, n_values = 3, 50, 500
    rng = random.Random(99)
    stream = [rng.gauss(0, 1) for _ in range(n_values)]

    # Some copies take (n_tracks, capacity), some only n_tracks.
    try:
        theirs = theirs_cls(n_tracks, capacity=capacity)
    except TypeError:
        theirs = theirs_cls(n_tracks)
        capacity = getattr(theirs, "capacity", DEFAULT_CAPACITY)
    ours = ReservoirSampler(n_tracks, capacity=capacity)

    for i, v in enumerate(stream):
        t = i % n_tracks
        theirs.add(t, v)
        ours.add(t, v)

    # Read `.counts` rather than get_counts(): EPInformer-seq's copy omits the
    # accessor and touches the attribute directly. That is an API difference,
    # not a behavioural one, and the shared module keeps the accessor.
    assert np.asarray(ours.counts).tolist() == np.asarray(theirs.counts).tolist(), (
        "sample counts differ — *_counts in the shipped NPZ would change"
    )
    np.testing.assert_array_equal(
        ours.to_cdf_matrix(n_points=256),
        theirs.to_cdf_matrix(n_points=256),
        err_msg="CDF matrices differ; the stored background would move",
    )


@pytest.mark.parametrize("path", BUILDERS, ids=_builder_ids())
def test_add_batch_matches_where_a_builder_has_one(path):
    """AlphaGenome hand-vectorises add_batch; it must still agree.

    A vectorised reservoir insert is only safe if it makes the same
    accept/reject decisions in the same order as the scalar loop. This is the
    test that has to pass before that 37-line version is allowed to replace the
    shared 3-line one.
    """
    theirs_cls = _extract_class(path, "ReservoirSampler")
    if theirs_cls is None:
        # Migrated — assert the import rather than skipping, for the same
        # reason as the test above: otherwise coverage drains silently to zero
        # as builders move across.
        assert "from chorus.analysis.background_sampling import" in Path(path).read_text(), (
            f"{Path(path).name} has no local ReservoirSampler and does not "
            f"import the shared one"
        )
        return
    if not hasattr(theirs_cls, "add_batch"):
        pytest.skip("this builder's sampler has no add_batch")

    capacity, n_values = 40, 400
    rng = random.Random(7)
    stream = [rng.gauss(0, 1) for _ in range(n_values)]

    try:
        theirs = theirs_cls(1, capacity=capacity)
    except TypeError:
        pytest.skip("constructor does not take capacity")
    ours = ReservoirSampler(1, capacity=capacity)

    # Feed in chunks so batching actually differs from a single call.
    for i in range(0, n_values, 37):
        chunk = stream[i:i + 37]
        theirs.add_batch(0, chunk)
        ours.add_batch(0, chunk)

    assert np.asarray(ours.counts).tolist() == np.asarray(theirs.counts).tolist()
    np.testing.assert_array_equal(
        ours.to_cdf_matrix(n_points=128), theirs.to_cdf_matrix(n_points=128),
        err_msg=(
            "add_batch diverges from the scalar loop — a different traversal "
            "order changes which samples survive, so the CDF moves without any "
            "arithmetic changing"
        ),
    )


# ---------------------------------------------------------------------------
# The four divergences must survive as parameters, not be unified away
# ---------------------------------------------------------------------------

def test_capacity_is_a_parameter_not_a_constant():
    """AlphaGenome retains 20,000; everyone else 50,000."""
    assert DEFAULT_CAPACITY == 50_000
    assert ReservoirSampler(1, capacity=20_000).capacity == 20_000
    # And capacity genuinely bounds retention, so the two are not interchangeable.
    small = ReservoirSampler(1, capacity=10)
    for v in range(100):
        small.add(0, float(v))
    assert len(small.data[0]) == 10
    assert small.get_counts()[0] == 100, "counts must record what was offered"


def test_n_fraction_is_a_parameter_not_a_constant():
    """LegNet rejects at 0.3 over a 200 bp window; everyone else at 0.5.

    Unifying these silently would change which positions LegNet samples, and so
    its shipped background.
    """
    assert DEFAULT_MAX_N_FRACTION == 0.5

    from chorus.analysis.background_sampling import get_sequence

    class _Fasta:
        def __init__(self, seq):
            self.seq = seq
        def __getitem__(self, chrom):
            return _Slice(self.seq)

    class _Slice:
        def __init__(self, seq):
            self.seq = seq
        def __getitem__(self, sl):
            return self.seq[sl]

    seq = "N" * 4 + "ACGTACGTAC"  # 14 bp, 4 N -> 28.6% N
    fa = _Fasta(seq)
    # 28.6% N: accepted at 0.5, accepted at 0.3, rejected at 0.2
    assert get_sequence(fa, "chr1", 8, 14, max_n_fraction=0.5) is not None
    assert get_sequence(fa, "chr1", 8, 14, max_n_fraction=0.3) is not None
    assert get_sequence(fa, "chr1", 8, 14, max_n_fraction=0.2) is None


def test_effect_formula_is_a_parameter():
    """AlphaGenome/Borzoi/Enformer need three conventions; the others one."""
    assert compute_effect(10, 20) == pytest.approx(math.log2(21 / 11))
    assert compute_effect(10, 20, formula="logfc") == pytest.approx(math.log(21 / 11))
    assert compute_effect(10, 20, formula="diff") == 10
    with pytest.raises(ValueError, match="Unknown effect formula"):
        compute_effect(1, 2, formula="nonsense")


def test_one_hot_channel_order_is_a_parameter():
    """EPInformer-seq's builder needs (4, L); everyone else (L, 4)."""
    assert one_hot_encode("ACGT").shape == (4, 4)
    assert one_hot_encode("ACGTAC").shape == (6, 4)
    assert one_hot_encode("ACGTAC", channels_first=True).shape == (4, 6)
    np.testing.assert_array_equal(
        one_hot_encode("ACGTAC", channels_first=True),
        one_hot_encode("ACGTAC").T,
    )


# ---------------------------------------------------------------------------
# Behaviour the copies relied on but never documented
# ---------------------------------------------------------------------------

def test_effect_matches_the_cherimoya_precedent_exactly():
    """The log2fc branch must equal cherimoya_source/scoring.py bit-for-bit.

    That module is the one place the builder/oracle sharing is already enforced,
    so it is the reference the shared implementation has to match.
    """
    from chorus.oracles.cherimoya_source.scoring import (
        compute_effect as cherimoya_effect,
    )
    for ref, alt in [(0, 0), (0, 1), (1, 0), (48.57, 100.2), (1e-6, 3.3), (1e4, 1e4)]:
        assert compute_effect(ref, alt) == cherimoya_effect(ref, alt), (
            f"diverges from the precedent at ref={ref}, alt={alt}"
        )


def test_non_acgt_becomes_an_all_zero_column():
    """`get_sequence` tolerates up to max_n_fraction N, so this path is live."""
    oh = one_hot_encode("ANCG")
    assert oh[1].sum() == 0.0
    assert oh.sum() == 3.0


def test_short_reservoir_interpolates_rather_than_pads():
    """A short row must span the whole grid — the premise #119 depends on.

    If short rows were padded, the last entry would repeat and the percentile
    denominator would legitimately be the sample count. They are interpolated,
    so every entry is a real quantile estimate and the denominator must be the
    grid width.
    """
    rs = ReservoirSampler(1, capacity=1_000)
    for v in range(50):
        rs.add(0, float(v))
    row = rs.to_cdf_matrix(n_points=DEFAULT_CDF_POINTS)[0]
    assert len(row) == DEFAULT_CDF_POINTS
    assert row[0] == pytest.approx(0.0)
    assert row[-1] == pytest.approx(49.0)
    # Strictly increasing over most of the row — not a flat padded tail.
    assert (np.diff(row) > 0).mean() > 0.9


def test_empty_track_stays_all_zero():
    """`_has_samples` detects an unfilled track by the row being all-zero."""
    rs = ReservoirSampler(2, capacity=10)
    rs.add(0, 1.0)
    m = rs.to_cdf_matrix(n_points=32)
    assert m[0].any()
    assert not m[1].any()


def test_window_slice_is_centred_and_clamped():
    values = np.arange(1000, dtype=np.float64)
    assert len(get_window_slice(values, 100, 1)) == 100
    assert get_window_slice(values, 100, 1)[0] == 450
    # A window wider than the prediction returns everything rather than raising.
    assert len(get_window_slice(values, 10_000, 1)) == 1000
    # Resolution divides the window into bins.
    assert len(get_window_slice(values, 128, 32)) == 4
    assert score_window_sum(np.ones(100), 10, 1) == 10.0
