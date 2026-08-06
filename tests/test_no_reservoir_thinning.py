"""The null's ceiling must be the true maximum, not the maximum of a subsample.

`ReservoirSampler` keeps a bounded uniform sample. Past capacity that is the whole
point -- but a *uniform* m-of-N subsample retains the population maximum with
probability exactly m/N, and the top of the grid is what `effect_percentile` clamps
against and what `effect_exceedance` divides by. Thinning the sample therefore moves
the ceiling, silently and by a random amount per track.

That is what happened. `scripts/merge_effect_shards.py` called
`from_flat_samples(*parts)` with no capacity, inheriting `DEFAULT_CAPACITY = 50_000`,
while every AlphaGenome `gene_expression` track offers 148,367 effect values. Measured
consequences, all reproduced by the gates below:

* 50,000/148,367 = 0.3370, and 33.9% of the 667 RNA rows kept their true maximum --
  the m/N arithmetic, confirmed to three digits on real data;
* the ceiling was understated by a median **1.332x**, p90 3.18x, worst **8.34x**;
* while p99 was correct to 0.02% and p50/p90 did not move at all.

That last line is why no percentile test caught it for months: reservoir sampling is
unbiased for the *body*, and every calibration gate measures the body. The eQTL
fixture's effects sit near grid index 8500-9000, so RNA p50 read 0.778 before the fix
and 0.7770 after it.

Neither existing guard could have caught it, and both are worth stating because the
instinct is to assume one did:

* the builders' own `capacity` was never exceeded -- each of the 8 shards offers only
  ~18.5k values per track, under the 20,000 it was given. The thinning is in the
  *merge*, a second and independent call site.
* `cdf_grid_violations` is handed the **offered** count while the geometry it checks
  is set by the **retained** count, and it does `if n >= n_points: continue` --
  offered is always >= 10,000, so it skips every thinned row by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis.background_sampling import DEFAULT_CAPACITY, ReservoirSampler

BG = None


def _bg():
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
    return CHORUS_BACKGROUNDS_DIR


# ---------------------------------------------------------------------------
# The mechanism, on synthetic data. Fast.
# ---------------------------------------------------------------------------


def test_capacity_none_retains_everything():
    s = ReservoirSampler(n_tracks=1, capacity=10_000)
    s.add_batch(0, np.arange(10_000, dtype=float))
    merged = ReservoirSampler.from_flat_samples(s.to_flat_samples(), capacity=None)
    assert merged.retained_counts()[0] == 10_000
    assert merged.get_counts()[0] == 10_000
    assert max(merged.data[0]) == 9_999.0


def test_a_capped_union_loses_the_maximum_at_the_predicted_rate():
    """m/N is not a metaphor: it is the probability the ceiling survives.

    Two shards of 600 values each, unioned under a capacity of 300, keep the
    population max with probability 300/1200 = 0.25. Averaged over many seeds the
    empirical rate must match -- which is the arithmetic that identified the defect
    (33.9% measured vs 50,000/148,367 = 0.337 predicted).
    """
    kept = 0
    trials = 200
    for seed in range(trials):
        a, b = ReservoirSampler(1, capacity=600), ReservoirSampler(1, capacity=600)
        a.add_batch(0, np.arange(0, 600, dtype=float))
        b.add_batch(0, np.arange(600, 1200, dtype=float))
        m = ReservoirSampler.from_flat_samples(
            a.to_flat_samples(), b.to_flat_samples(), capacity=300, seed=seed)
        if max(m.data[0]) == 1199.0:
            kept += 1
    rate = kept / trials
    assert 0.15 < rate < 0.35, (
        f"kept the true maximum in {rate:.1%} of unions; m/N predicts 25%. If this "
        f"drifts far from 0.25 the subsampling is no longer uniform."
    )


def test_the_body_survives_thinning_but_the_tail_does_not():
    """Why every calibration gate passed while the ceiling was wrong."""
    rng = np.random.default_rng(0)
    vals = rng.pareto(1.5, 120_000) + 1.0          # heavy tail, like an RNA null
    s = ReservoirSampler(1, capacity=120_000)
    s.add_batch(0, vals)
    flat = s.to_flat_samples()

    exact = ReservoirSampler.from_flat_samples(flat, capacity=None)
    capped = ReservoirSampler.from_flat_samples(flat, capacity=40_000, seed=7)
    e = np.sort(np.array(exact.data[0])); c = np.sort(np.array(capped.data[0]))

    for q in (0.5, 0.9, 0.99):
        assert abs(np.quantile(c, q) / np.quantile(e, q) - 1) < 0.05, (
            f"the body moved at q={q}; reservoir sampling should be unbiased there"
        )
    assert c.max() <= e.max()
    assert c.max() / e.max() < 0.95, (
        "a 3x thinning of a heavy tail should visibly lower the ceiling; if it does "
        "not, this fixture is no longer heavy-tailed enough to model the defect"
    )


def test_capacity_is_required_so_the_defect_cannot_recur():
    s = ReservoirSampler(1, capacity=10)
    s.add_batch(0, np.arange(10, dtype=float))
    with pytest.raises(TypeError):
        ReservoirSampler.from_flat_samples(s.to_flat_samples())
    assert DEFAULT_CAPACITY == 50_000, (
        "the value that was silently inherited; if it changes, update this module's "
        "measured numbers, which are all relative to a 50,000 cap"
    )


# ---------------------------------------------------------------------------
# The shipped artefacts. Integration: these read multi-GB shard files.
# ---------------------------------------------------------------------------


def _union(oracle: str, n_shards: int = 8, capacity=None):
    parts, ids, layers = [], None, None
    for k in range(n_shards):
        p = _bg() / f"{oracle}_effect_cdfs_interim.shard{k}of{n_shards}.npz"
        if not p.exists():
            return None, None, None
        with np.load(p, allow_pickle=False) as d:
            if "values" not in d.files:
                return None, None, None
            parts.append({q: d[q] for q in ("values", "offsets", "counts", "n_tracks")})
            if ids is None:
                ids = [str(x) for x in d["track_ids"]]
                layers = ([str(x) for x in d["layers_per_row"]]
                          if "layers_per_row" in d.files else None)
    return ReservoirSampler.from_flat_samples(*parts, capacity=capacity), ids, layers


@pytest.mark.integration
def test_alphagenome_shipped_ceilings_are_the_exact_population_maxima():
    """The fix, verified against the raw samples rather than against itself."""
    path = _bg() / "alphagenome_pertrack.npz"
    if not path.exists():
        pytest.skip("no downloaded background for alphagenome")
    merged, ids, layers = _union("alphagenome")
    if merged is None:
        pytest.skip("alphagenome effect shards not on disk")

    assert int((merged.retained_counts() < merged.get_counts()).sum()) == 0, (
        "the exact union is still thinning tracks"
    )
    with np.load(path, allow_pickle=True) as d:
        shipped = d["effect_cdfs"]
        assert [str(x) for x in d["track_ids"]] == ids

    # Spot-check the heaviest layer: the shipped ceiling must be the true maximum.
    lay = np.array(layers)
    rna = np.where(lay == "gene_expression")[0][:40]
    for i in rna:
        true_max = max(merged.data[int(i)])
        assert shipped[i].max() == pytest.approx(true_max, rel=1e-6), (
            f"row {i} ({ids[i]}) ships a ceiling of {shipped[i].max()} but the true "
            f"maximum over all {merged.get_counts()[i]} offered values is {true_max}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("oracle", ["borzoi", "enformer"])
def test_borzoi_and_enformer_effect_unions_are_a_no_op(oracle):
    """Recorded as a permanent negative result so nobody re-derives it.

    Their unions are below DEFAULT_CAPACITY, so `from_flat_samples` never takes the
    `rng.choice` path and the merge is a pure concatenate-then-sort. They were
    therefore never thinned and deliberately NOT rebuilt -- worth pinning, because
    "we rebuilt AlphaGenome but not these two" otherwise looks like an omission.
    """
    path = _bg() / f"{oracle}_pertrack.npz"
    if not path.exists():
        pytest.skip(f"no downloaded background for {oracle}")
    merged, ids, _ = _union(oracle)
    if merged is None:
        pytest.skip(f"{oracle} effect shards not on disk")

    counts, retained = merged.get_counts(), merged.retained_counts()
    assert int((retained < counts).sum()) == 0, (
        f"{oracle} IS thinned at the union ({int((retained < counts).sum())} tracks), "
        f"so it does need a rebuild -- max offered {counts.max()} vs capacity "
        f"{DEFAULT_CAPACITY}"
    )
    with np.load(path, allow_pickle=True) as d:
        shipped = d["effect_cdfs"]
    # A few rows including the highest-count ones, where thinning would show first.
    probe = list(np.argsort(counts)[-3:]) + [0, len(ids) // 2, len(ids) - 1]
    for i in probe:
        row = np.sort(np.array(merged.data[int(i)]))
        grid = row[np.linspace(0, len(row) - 1, shipped.shape[1], dtype=int)]
        assert np.allclose(grid.astype(np.float32), shipped[int(i)], rtol=0, atol=0), (
            f"{oracle} row {i} does not reproduce from its shards bit-exactly"
        )
