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


# ---------------------------------------------------------------------------
# The write-time guard
# ---------------------------------------------------------------------------


def test_thinning_violations_passes_an_exact_sample():
    from chorus.analysis.background_sampling import thinning_violations
    assert thinning_violations(np.array([100, 148_367]),
                               np.array([100, 148_367])) == []


def test_thinning_violations_catches_the_alphagenome_case():
    from chorus.analysis.background_sampling import thinning_violations
    problems = thinning_violations(np.array([148_367]), np.array([50_000]))
    assert len(problems) == 1
    assert "2.97x thinned" in problems[0]
    assert "0.337" in problems[0], (
        "the message should state the probability the ceiling survived, since that "
        "is the whole mechanism"
    )


def test_an_exact_top_k_tail_is_accepted_but_a_token_one_is_not():
    from chorus.analysis.background_sampling import (
        MIN_EXACT_TAIL_SLOTS, thinning_violations,
    )
    # A tail wide enough to fill >= MIN_EXACT_TAIL_SLOTS grid slots is fine: the
    # region the percentile saturates in is then exact even though the body is not.
    assert thinning_violations(np.array([148_367]), np.array([69_832]),
                               tail_k=19_832) == []
    # A token tail is not. 100 of 148,367 fills 6 slots of 10,000.
    problems = thinning_violations(np.array([148_367]), np.array([50_100]),
                                   tail_k=100)
    assert len(problems) == 1 and "6 of 10000" in problems[0]
    assert str(MIN_EXACT_TAIL_SLOTS) in problems[0]


def test_build_and_save_refuses_to_write_a_thinned_matrix(tmp_path):
    """The point of the guard: it has to actually block, not just exist.

    The previous guard existed, was called, and could not see this class of defect --
    it is fed the OFFERED count while validating geometry set by the RETAINED count,
    and returns early on every row with n >= n_points.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    n_points = 10_000
    row = np.linspace(0.0, 1.0, n_points)
    matrix = np.vstack([row, row])
    counts = np.array([148_367, 148_367])

    with pytest.raises(ValueError, match="thinned sample"):
        PerTrackNormalizer.build_and_save(
            oracle_name="synthetic",
            track_ids=["a", "b"],
            effect_cdfs=matrix,
            effect_counts=counts,
            cache_dir=str(tmp_path),
            sampling={"effect": {"offered": counts,
                                 "retained": np.array([50_000, 50_000])}},
        )
    assert not list(tmp_path.glob("*.npz")), "a rejected build must write nothing"


def test_build_and_save_accepts_the_same_matrix_when_retention_was_exact(tmp_path):
    from chorus.analysis.normalization import PerTrackNormalizer

    n_points = 10_000
    row = np.linspace(0.0, 1.0, n_points)
    counts = np.array([148_367, 148_367])
    out = PerTrackNormalizer.build_and_save(
        oracle_name="synthetic",
        track_ids=["a", "b"],
        effect_cdfs=np.vstack([row, row]),
        effect_counts=counts,
        cache_dir=str(tmp_path),
        sampling={"effect": {"offered": counts, "retained": counts}},
    )
    assert out.exists()


def test_omitting_the_sampling_block_is_logged_as_an_error(tmp_path, caplog):
    """Silence is how both previous guards were bypassed.

    A builder that forgets `sampling=` gets no thinning protection whatsoever, so the
    absence has to be noisy rather than a default-None that quietly disables the
    check.
    """
    import logging

    from chorus.analysis.normalization import PerTrackNormalizer

    row = np.linspace(0.0, 1.0, 10_000)
    with caplog.at_level(logging.ERROR):
        PerTrackNormalizer.build_and_save(
            oracle_name="synthetic",
            track_ids=["a"],
            effect_cdfs=row.reshape(1, -1),
            effect_counts=np.array([11_934]),
            cache_dir=str(tmp_path),
        )
    assert any("WITHOUT a sampling= block" in r.getMessage()
               for r in caplog.records), (
        f"no error logged; records={[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# The opt-in exact tail
# ---------------------------------------------------------------------------


def _heavy(n, seed=3):
    """A heavy-tailed stream, which is what an RNA or splice null looks like."""
    return np.random.default_rng(seed).pareto(1.5, n) + 1.0


def _fill(sampler, vals, chunk=8192):
    for i in range(0, len(vals), chunk):
        sampler.add_batch(0, vals[i:i + chunk])
    return sampler


def test_the_exact_tail_captures_the_true_extremes():
    vals = _heavy(200_000)
    s = _fill(ReservoirSampler(1, capacity=50_000, tail_k=2_000), vals)
    bot, top = s.exact_tail(0)
    assert top.max() == vals.max()
    assert bot.min() == vals.min()
    assert len(top) == len(bot) == 2_000


def test_both_ends_are_tracked_because_signed_layers_need_the_floor():
    """12.9% of AlphaGenome's rows and 20.3% of Borzoi's are signed.

    For those, a strongly repressive effect crosses the LOWER bound, so tracking only
    the maximum would leave exactly those rows with an estimated floor -- and
    ``effect_exceedance`` divides by whichever end was crossed.
    """
    vals = np.concatenate([_heavy(100_000), -_heavy(100_000, seed=4)])
    s = _fill(ReservoirSampler(1, capacity=20_000, tail_k=1_000), vals)
    bot, top = s.exact_tail(0)
    assert top.max() == vals.max()
    assert bot.min() == vals.min() < 0


def test_the_grid_row_recovers_the_population_maximum():
    vals = _heavy(300_000)
    hybrid = _fill(ReservoirSampler(1, capacity=50_000, tail_k=20_000),
                   vals).to_cdf_matrix(n_points=10_000)[0]
    plain = _fill(ReservoirSampler(1, capacity=50_000),
                  vals).to_cdf_matrix(n_points=10_000)[0]
    truth = np.sort(vals)

    assert hybrid[-1] == truth[-1], "the hybrid row's ceiling is not the true maximum"
    assert hybrid[0] == truth[0]
    # And the failure it replaces: the plain thinned row understates badly. Measured
    # 0.40x on this fixture; assert only that it is materially low, so the test does
    # not depend on one RNG draw.
    assert plain[-1] < 0.75 * truth[-1], (
        f"the thinned row's ceiling is {plain[-1] / truth[-1]:.2f}x of the true "
        f"maximum; if this is no longer materially low, the fixture stopped modelling "
        f"the defect"
    )


def test_the_tail_slots_are_exact_order_statistics_not_estimates():
    """Exactness in the top K*n_points/N slots is the whole point of the design."""
    vals = _heavy(300_000)
    n, n_points, k = len(vals), 10_000, 20_000
    row = _fill(ReservoirSampler(1, capacity=50_000, tail_k=k),
                vals).to_cdf_matrix(n_points=n_points)[0]

    truth = np.sort(vals)
    ranks = np.rint(np.arange(n_points) * (n - 1) / (n_points - 1)).astype(np.int64)
    ref = truth[ranks]
    n_top = int(np.sum(ranks >= n - k))
    n_bot = int(np.sum(ranks < k))
    assert n_top >= 200 and n_bot >= 200

    assert np.array_equal(row[-n_top:], ref[-n_top:]), (
        "the top tail slots are not the population's own order statistics"
    )
    assert np.array_equal(row[:n_bot], ref[:n_bot])


def test_an_unthinned_build_is_bit_identical_with_or_without_a_tail():
    """The degeneracy guarantee: nothing changes where nothing was discarded.

    Without this, turning the feature on would silently move every background whose
    stream fits in capacity -- i.e. most of them -- and no existing grid-integrity
    test would still be describing real behaviour.
    """
    vals = _heavy(30_000)
    with_tail = _fill(ReservoirSampler(1, capacity=50_000, tail_k=5_000),
                      vals).to_cdf_matrix(n_points=10_000)
    without = _fill(ReservoirSampler(1, capacity=50_000),
                    vals).to_cdf_matrix(n_points=10_000)
    assert np.array_equal(with_tail, without)


def test_the_default_is_off_so_todays_builders_are_unaffected():
    s = ReservoirSampler(4, capacity=100)
    assert s.tail_k is None
    bot, top = s.exact_tail(0)
    assert len(bot) == len(top) == 0


def test_a_spliced_row_is_non_decreasing():
    """A non-monotone CDF row silently corrupts every searchsorted downstream."""
    for seed in range(4):
        vals = np.concatenate([_heavy(80_000, seed), -_heavy(80_000, seed + 10)])
        row = _fill(ReservoirSampler(1, capacity=20_000, tail_k=4_000),
                    vals).to_cdf_matrix(n_points=10_000)[0]
        assert np.all(np.diff(row) >= 0), f"seed {seed}: row is not sorted"


def test_the_interior_is_still_the_same_uniform_estimate():
    """Only the ends become exact; the body must not be perturbed."""
    vals = _heavy(300_000)
    n, n_points, k = len(vals), 10_000, 20_000
    row = _fill(ReservoirSampler(1, capacity=50_000, tail_k=k),
                vals).to_cdf_matrix(n_points=n_points)[0]
    truth = np.sort(vals)
    for q in (0.25, 0.5, 0.75, 0.9):
        got, want = np.quantile(row, q), np.quantile(truth, q)
        assert abs(got / want - 1) < 0.05, f"q={q}: {got} vs {want}"


# ---------------------------------------------------------------------------
# A build where everything failed must not write a well-formed empty file
# ---------------------------------------------------------------------------


def test_yield_violations_catches_a_build_where_every_position_failed():
    from chorus.analysis.background_sampling import yield_violations
    problems = yield_violations(np.zeros(5313, dtype=np.int64), label="enformer.effect")
    assert len(problems) == 1
    assert "0 of 5313" in problems[0]
    assert yield_violations(np.full(5313, 17_900, dtype=np.int64)) == []


def test_build_and_save_refuses_an_all_zero_background(tmp_path):
    """The failure this is drawn from, which produced a file rather than an error.

    Two TensorFlow processes were launched onto the same GPU because the builder
    overwrote CUDA_VISIBLE_DEVICES with its --gpu default. The second could not
    allocate a cuBLAS handle, so EVERY forward pass raised InternalError, and the
    per-position try/except -- correctly there so one bad locus cannot lose a run --
    dropped all 5,968 positions and wrote a well-formed interim: 5,313 tracks, every
    row all-zero, every count 0.

    That file merges cleanly. `_has_samples` would then suppress those tracks at query
    time, so the symptom is an oracle that silently stops ranking anything: the same
    shape as Sei's 40 dark rows, reached by a different route.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    n = 64
    with pytest.raises(ValueError, match="almost every position failed"):
        PerTrackNormalizer.build_and_save(
            oracle_name="synthetic",
            track_ids=[f"t{i}" for i in range(n)],
            effect_cdfs=np.zeros((n, 10_000)),
            effect_counts=np.zeros(n, dtype=np.int64),
            cache_dir=str(tmp_path),
        )
    assert not list(tmp_path.glob("*.npz"))


def test_a_partial_build_is_still_allowed(tmp_path):
    """The per-position tolerance must survive: some loci legitimately fail."""
    from chorus.analysis.normalization import PerTrackNormalizer

    n = 64
    counts = np.full(n, 17_900, dtype=np.int64)
    counts[:20] = 0                      # 69% of tracks have data -- above the floor
    row = np.linspace(0.0, 1.0, 10_000)
    matrix = np.tile(row, (n, 1))
    matrix[:20] = 0.0
    out = PerTrackNormalizer.build_and_save(
        oracle_name="synthetic", track_ids=[f"t{i}" for i in range(n)],
        effect_cdfs=matrix, effect_counts=counts, cache_dir=str(tmp_path),
        sampling={"effect": {"offered": counts, "retained": counts}},
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# T2: thinning must be provable from a shipped artefact, not just an interim
# ---------------------------------------------------------------------------


def test_retention_is_persisted_into_the_written_file(tmp_path):
    """Only the OFFERED count was ever stored, so nobody could check.

    ``*_counts`` is the offered count. Without the retained count beside it, "was this
    track's tail thinned?" is unanswerable from a published background -- which is
    exactly how AlphaGenome's 2.97x thinning survived a republish and months of use.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    n, n_points = 8, 10_000
    row = np.linspace(0.0, 1.0, n_points)
    counts = np.full(n, 222_551, dtype=np.int64)
    out = PerTrackNormalizer.build_and_save(
        oracle_name="synthetic", track_ids=[f"t{i}" for i in range(n)],
        effect_cdfs=np.tile(row, (n, 1)), effect_counts=counts,
        cache_dir=str(tmp_path),
        sampling={"effect": {"offered": counts, "retained": counts}},
    )
    with np.load(out, allow_pickle=True) as d:
        assert "effect_retained" in d.files, (
            f"retention absent from the shipped file; keys={sorted(d.files)}"
        )
        assert np.array_equal(d["effect_retained"], counts)
        assert int((d["effect_retained"] < d["effect_counts"]).sum()) == 0


def test_a_hybrid_layer_records_its_tail_k(tmp_path):
    """So a reader can reconstruct how many top slots are exact."""
    from chorus.analysis.background_sampling import MIN_EXACT_TAIL_SLOTS
    from chorus.analysis.normalization import PerTrackNormalizer

    n, n_points = 4, 10_000
    row = np.linspace(0.0, 1.0, n_points)
    offered = np.full(n, 991_552, dtype=np.int64)
    retained = np.full(n, 50_000 + 2 * 19_832, dtype=np.int64)
    out = PerTrackNormalizer.build_and_save(
        oracle_name="synthetic", track_ids=[f"t{i}" for i in range(n)],
        perbin_cdfs=np.tile(row, (n, 1)), perbin_counts=offered,
        cache_dir=str(tmp_path),
        sampling={"perbin": {"offered": offered, "retained": retained,
                             "tail_k": 19_832}},
    )
    with np.load(out, allow_pickle=True) as d:
        assert int(d["perbin_tail_k"]) == 19_832
        slots = int(19_832 * n_points // 991_552)
        assert slots >= MIN_EXACT_TAIL_SLOTS
