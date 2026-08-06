"""A position-sharded build must equal the unsharded one, exactly.

AlphaGenome, Borzoi and Enformer produce every track from ONE forward pass, so
sharding them by *track* — which is how ``build_backgrounds_chrombpnet.py`` shards,
and which is correct there because its tracks are separate model files — saves no GPU
time: each shard would still run every pass. Splitting one of these across 8 GPUs
means sharding by *position*, and that changes what a merge is. Each shard then holds
a partial reservoir for EVERY track rather than a complete one for some tracks.

The tempting merge is to pool the shards' 10,000-point CDF grids. That is an
approximation — a good one for equal-sized shards, since each grid is a
representative sample of its shard's distribution — but an approximation, and this
codebase has already shipped one artefact whose approximation looked exact (the
padded enformer grid, #143). So shards serialise raw samples and the CDF is built
once from the union.

These tests pin the property that makes the whole scheme trustworthy: for any
partition of the input stream, the merged sampler is **identical** to one fed the
whole stream, in both retained values and reported counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis.background_sampling import ReservoirSampler

N_TRACKS = 6


def _stream(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).exponential(1.0, n)


def _feed(values, *, capacity: int, n_tracks: int = N_TRACKS) -> ReservoirSampler:
    """One sampler fed every value into every track."""
    s = ReservoirSampler(n_tracks=n_tracks, capacity=capacity)
    for v in values:
        for t in range(n_tracks):
            s.add(t, float(v))
    return s


def _shards(values, n_shards: int, *, capacity: int) -> list[dict]:
    """Stride-partition the stream, exactly as ``i % n_shards == k`` would."""
    out = []
    for k in range(n_shards):
        s = _feed(values[k::n_shards], capacity=capacity)
        out.append(s.to_flat_samples())
    return out


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_flat_samples_round_trip_one_shard_is_the_identity():
    whole = _feed(_stream(500), capacity=10_000)
    back = ReservoirSampler.from_flat_samples(whole.to_flat_samples(), capacity=10_000)
    assert np.array_equal(back.get_counts(), whole.get_counts())
    for t in range(N_TRACKS):
        np.testing.assert_array_equal(back.get_sorted(t), whole.get_sorted(t))


def test_flat_samples_survive_an_npz_without_pickle():
    """Ragged data stored flat, so the interim still loads under
    ``allow_pickle=False`` — which every reader in chorus uses."""
    import io

    payload = _feed(_stream(300), capacity=10_000).to_flat_samples()
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as data:
        assert set(data.files) == {"values", "offsets", "counts", "n_tracks"}
        rebuilt = ReservoirSampler.from_flat_samples(
            {k: data[k] for k in data.files}, capacity=10_000)
    assert int(rebuilt.get_counts()[0]) == 300


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_shards", [2, 3, 4, 8])
def test_sharded_merge_equals_the_unsharded_build(n_shards):
    """Below capacity, the merge must be EXACT, not merely close.

    Every shipped percentile is read off this distribution, so an approximate merge
    would move numbers for a reason unrelated to the science.
    """
    values = _stream(1_200, seed=3)
    capacity = 10_000                      # far above 1,200: nothing is evicted

    whole = _feed(values, capacity=capacity)
    merged = ReservoirSampler.from_flat_samples(
        *_shards(values, n_shards, capacity=capacity), capacity=capacity)

    np.testing.assert_array_equal(merged.get_counts(), whole.get_counts())
    for t in range(N_TRACKS):
        np.testing.assert_array_equal(merged.get_sorted(t), whole.get_sorted(t))
    # and therefore the CDF, which is what actually ships
    np.testing.assert_allclose(merged.to_cdf_matrix(1_000),
                               whole.to_cdf_matrix(1_000), rtol=0, atol=0)


@pytest.mark.parametrize("n_shards", [2, 4])
def test_counts_add_across_shards_even_when_values_are_evicted(n_shards):
    """``counts`` is values *offered*, not retained — the shipped ``*_counts``
    meaning. It must add across shards regardless of capacity pressure."""
    values = _stream(900, seed=5)
    capacity = 100                          # forces eviction inside every shard

    merged = ReservoirSampler.from_flat_samples(
        *_shards(values, n_shards, capacity=capacity), capacity=capacity)
    assert int(merged.get_counts()[0]) == 900
    # retained is capped, and the cap is honoured after merging
    assert len(merged.data[0]) == capacity


def test_shards_with_disagreeing_track_counts_are_refused():
    """A silent mismatch here would misalign every row of the merged matrix."""
    a = _feed(_stream(50), capacity=1_000, n_tracks=4).to_flat_samples()
    b = _feed(_stream(50), capacity=1_000, n_tracks=5).to_flat_samples()
    with pytest.raises(ValueError, match="track counts disagree"):
        ReservoirSampler.from_flat_samples(a, b, capacity=1_000)


def test_no_shards_is_an_error_not_an_empty_sampler():
    with pytest.raises(ValueError, match="no shards"):
        ReservoirSampler.from_flat_samples(capacity=10_000)


def test_capacity_must_be_passed_explicitly():
    """The defect: ``from_flat_samples`` defaulted to DEFAULT_CAPACITY = 50,000.

    ``merge_effect_shards.py`` called it without a capacity, so every AlphaGenome
    RNA track's 148,367-value union was silently subsampled to 50,000 and the grid's
    maximum became the max of that subsample rather than of the population. Making
    the argument required turns that from a silent 2.97x thinning into a TypeError.
    """
    s = ReservoirSampler(n_tracks=1, capacity=10)
    for v in range(10):
        s.add(0, float(v))
    with pytest.raises(TypeError):
        ReservoirSampler.from_flat_samples(s.to_flat_samples())  # no capacity=


def test_a_track_with_no_samples_in_any_shard_stays_empty():
    """``_has_samples`` detects an all-zero row; the merge must not invent one."""
    s = ReservoirSampler(n_tracks=3, capacity=100)
    for v in _stream(20):
        s.add(0, float(v))          # only track 0 ever receives anything
    merged = ReservoirSampler.from_flat_samples(s.to_flat_samples(), capacity=100)
    assert int(merged.get_counts()[0]) == 20
    assert int(merged.get_counts()[1]) == 0
    assert merged.data[1] == []
    assert merged.to_cdf_matrix(100)[1].max() == 0.0


def test_merge_is_deterministic_under_capacity_pressure():
    """Same shards in, same sampler out — #127's requirement, at the merge step."""
    values = _stream(900, seed=9)
    shards = _shards(values, 4, capacity=100)
    a = ReservoirSampler.from_flat_samples(*shards, capacity=100)
    b = ReservoirSampler.from_flat_samples(*shards, capacity=100)
    for t in range(N_TRACKS):
        np.testing.assert_array_equal(a.get_sorted(t), b.get_sorted(t))
