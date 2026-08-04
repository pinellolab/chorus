"""Guard the one invariant that made enformer's effect percentiles unreachable.

Every shipped ``*_pertrack.npz`` stores three ``(n_tracks, n_points)`` CDF
matrices, and ``PerTrackNormalizer._get_denominator`` divides by the stored
*width* rather than the sample count (#119, correct). That makes the width
load-bearing: if a matrix is gridded at one width and stored at another, every
percentile in it is silently rescaled.

Which is exactly what shipped. ``enformer_pertrack.npz`` had its ``effect_cdfs``
built on a 9,606-point grid — ``max(effect_counts)`` — then padded to 10,000 by
repeating each row's maximum 395 times. All 5,313 rows. Consequences:

* every enformer effect percentile compressed by ~0.9606;
* ``(0.9605, 1.0)`` unreachable, so the top 4 % of the scale did not exist;
* #83's per-track floor at any ``q >= 0.96`` resolves to the *null maximum*.

Nothing caught it. ``to_cdf_matrix`` interpolates short rows onto the full grid
and *cannot* produce that shape, so the file was not reproducible from repo code;
``build_and_save`` only resampled when ``shape[1] > n_points``, so a narrow matrix
was written verbatim; and ``normalization.py``'s own docstring asserts rows are
never padded. The artefact contradicted an invariant its code documented.

The unit tests here run in CI without any download. The integration test checks
the shipped files and is skipped when they are absent.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.background_sampling import (
    ReservoirSampler,
    cdf_grid_violations,
    expected_first_max_index,
)

N_POINTS = 10_000
_BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"
_CDF_KEYS = ("effect_cdfs", "summary_cdfs", "perbin_cdfs")


def _build_row(n_samples: int, n_points: int = N_POINTS) -> tuple[np.ndarray, int]:
    """A genuine one-track CDF row via the real sampler, plus its count."""
    sampler = ReservoirSampler(n_tracks=1, capacity=max(n_samples, 1))
    rng = np.random.default_rng(0)
    for value in rng.exponential(1.0, n_samples):
        sampler.add(0, float(value))
    return sampler.to_cdf_matrix(n_points=n_points), n_samples


# ---------------------------------------------------------------------------
# The derivation, pinned. These are the two branches of to_cdf_matrix.
# ---------------------------------------------------------------------------


def test_expected_first_max_long_row_is_last_slot():
    """``n >= n_points`` subsamples, so the largest sample lands last."""
    assert expected_first_max_index(N_POINTS, N_POINTS) == N_POINTS - 1
    assert expected_first_max_index(18_672, N_POINTS) == N_POINTS - 1


@pytest.mark.parametrize("n_samples", [2, 100, 1_697, 1_909, 9_606, 9_999])
def test_expected_first_max_matches_a_real_interpolated_row(n_samples):
    """The formula must describe what ``to_cdf_matrix`` actually produces.

    ``source_q = arange(n)/n`` stops at ``(n-1)/n``, not 1.0, so ``np.interp``
    clamps beyond it. Any off-by-one here would make the guard reject healthy
    rows — a naive ``n_points - 1`` is red on every short row in every oracle.
    """
    matrix, n = _build_row(n_samples)
    row = matrix[0]
    expected = expected_first_max_index(n, N_POINTS)
    assert int(np.argmax(row)) == expected
    assert row[expected] == row.max()


def test_short_rows_legitimately_end_in_a_plateau():
    """Trailing duplicates are normal, so they cannot be the padding signal.

    A 100-sample row clamps for its last 100 slots. Any guard that flags
    "repeated maxima at the tail" would reject this healthy row.
    """
    matrix, n = _build_row(100)
    row = matrix[0]
    plateau = N_POINTS - int(np.nonzero(row != row[-1])[0][-1]) - 1
    assert plateau == N_POINTS - expected_first_max_index(n, N_POINTS) == 100
    assert cdf_grid_violations(matrix, np.array([n])) == []


# ---------------------------------------------------------------------------
# The guard: healthy rows pass, the shipped enformer shape does not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 2, 100, 1_909, 9_606, 10_000, 18_672])
def test_genuine_rows_pass_the_guard(n_samples):
    matrix, n = _build_row(n_samples)
    assert cdf_grid_violations(matrix, np.array([n])) == []


def test_guard_catches_the_enformer_padding_shape():
    """Reproduce the real defect: grid at ``count``, then pad to 10,000."""
    n = 9_606
    narrow, _ = _build_row(n, n_points=n)
    padded = np.concatenate(
        [narrow, np.repeat(narrow[:, -1:], N_POINTS - n, axis=1)], axis=1
    )
    assert padded.shape == (1, N_POINTS)
    # The fingerprints the shipped file showed, before asserting the guard fires.
    assert int(np.argmax(padded[0])) == n - 1 == 9_605
    assert int((padded[0] == padded[0].max()).sum()) == 395
    assert np.unique(padded[0]).size == n

    problems = cdf_grid_violations(padded, np.array([n]))
    assert problems, "padded row must be rejected"
    assert "distinct" in problems[0]


def test_guard_catches_a_non_monotonic_row():
    matrix, n = _build_row(1_909)
    matrix[0, 500] = matrix[0, 400] - 1.0
    problems = cdf_grid_violations(matrix, np.array([n]))
    assert problems and "non-decreasing" in problems[0]


def test_zero_count_rows_are_skipped_not_flagged():
    """A track that failed to build ships all-zero and is handled by
    ``_has_samples``, not by this guard."""
    assert cdf_grid_violations(np.zeros((1, N_POINTS)), np.array([0])) == []


def test_single_sample_row_is_not_flagged():
    """One sample interpolates to a flat row, making ``distinct == count == 1``.

    That collides with the distinct-vs-count fingerprint, so a constant row has
    to be exempt or the guard refuses a legitimate single-sample track.
    """
    matrix, n = _build_row(1)
    assert np.unique(matrix[0]).size == 1
    assert cdf_grid_violations(matrix, np.array([n])) == []


def test_all_identical_samples_are_not_flagged():
    """A track whose every prediction is identical is degenerate, not corrupt.

    Its plateau spans the whole grid, which the plateau check would read as a
    width mismatch. Degeneracy is the scale census's business, not this guard's.
    """
    constant = np.full((1, N_POINTS), 0.42)
    assert cdf_grid_violations(constant, np.array([9_606])) == []


# ---------------------------------------------------------------------------
# The write-time guard. This is the half that stops a rebuild recreating it.
# ---------------------------------------------------------------------------


def test_build_and_save_accepts_a_narrow_matrix_but_warns(tmp_path, caplog):
    """A narrow matrix is self-consistent, so it is stored, not rejected.

    ``_get_denominator`` reads ``shape[1]``, so a 9,606-wide matrix stored at
    9,606 yields correct percentiles. Enformer's defect was the *opposite* shape
    — gridded at 9,606 and padded up to 10,000, so ``shape[1] == n_points`` and no
    width check could ever have caught it. Rejecting narrow matrices would only
    have broken honest callers (``test_analysis`` builds 100-wide fixtures).
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    n = 9_606
    narrow, _ = _build_row(n, n_points=n)
    with caplog.at_level("WARNING"):
        path = PerTrackNormalizer.build_and_save(
            oracle_name="guardtest",
            track_ids=["t0"],
            effect_cdfs=narrow,
            effect_counts=[n],
            cache_dir=str(tmp_path),
            n_points=N_POINTS,
        )
    assert "columns wide" in caplog.text
    with np.load(path) as data:
        assert data["effect_cdfs"].shape == (1, n)


def test_build_and_save_refuses_a_padded_matrix(tmp_path):
    from chorus.analysis.normalization import PerTrackNormalizer

    n = 9_606
    narrow, _ = _build_row(n, n_points=n)
    padded = np.concatenate(
        [narrow, np.repeat(narrow[:, -1:], N_POINTS - n, axis=1)], axis=1
    )
    with pytest.raises(ValueError, match="to_cdf_matrix"):
        PerTrackNormalizer.build_and_save(
            oracle_name="guardtest",
            track_ids=["t0"],
            effect_cdfs=padded,
            effect_counts=[n],
            cache_dir=str(tmp_path),
            n_points=N_POINTS,
        )


def test_build_and_save_accepts_a_genuine_matrix(tmp_path):
    from chorus.analysis.normalization import PerTrackNormalizer

    matrix, n = _build_row(1_909)
    path = PerTrackNormalizer.build_and_save(
        oracle_name="guardtest",
        track_ids=["t0"],
        effect_cdfs=matrix,
        effect_counts=[n],
        cache_dir=str(tmp_path),
        n_points=N_POINTS,
    )
    assert Path(path).exists()
    with np.load(path) as data:
        assert data["effect_cdfs"].shape == (1, N_POINTS)


# ---------------------------------------------------------------------------
# The shipped artefacts.
# ---------------------------------------------------------------------------


def _shipped() -> list[tuple[str, Path]]:
    if not _BACKGROUNDS.is_dir():
        return []
    return [(p.name.replace("_pertrack.npz", ""), p)
            for p in sorted(_BACKGROUNDS.glob("*_pertrack.npz"))]


@pytest.mark.integration
@pytest.mark.parametrize("oracle,path", _shipped() or [("none", Path("none"))])
def test_shipped_backgrounds_have_reproducible_grids(oracle, path):
    """Every shipped CDF matrix must be reproducible from repo code.

    Validated across all eight backgrounds: 19,393 short rows, of which the only
    violations are enformer's 5,313 ``effect_cdfs``. Notably clean is
    AlphaGenome's ``summary_cdfs`` row 2,452, whose top two H3K4me1 windows tie
    at exactly 2480.0 — real saturation, and the reason the plateau check is
    restricted to ``n < n_points``.
    """
    if not path.exists():
        pytest.skip("no downloaded backgrounds")
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        problems: list[str] = []
        for key in _CDF_KEYS:
            count_key = key.replace("_cdfs", "_counts")
            if key not in keys or count_key not in keys:
                continue
            problems += cdf_grid_violations(
                data[key], data[count_key], label=f"{oracle}.{key}"
            )
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Ties at the maximum: the false positive a real rebuild exposed
# ---------------------------------------------------------------------------


def _row_with_tied_maximum(n_samples: int, n_tied: int, n_points: int = 10_000):
    """A genuine to_cdf_matrix row whose top ``n_tied`` samples are equal.

    Real: a fresh Borzoi build produced 10 of these and Enformer 152. Borzoi row
    3011's top effect value 0.689308 recurs 9 times because several sampled
    variants hit the same clipped ceiling.
    """
    sampler = ReservoirSampler(n_tracks=1, capacity=n_samples)
    rng = np.random.default_rng(3)
    values = np.sort(rng.exponential(1.0, n_samples))
    values[-n_tied:] = values[-n_tied]          # tie the top
    for v in values:
        sampler.add(0, float(v))
    return sampler.to_cdf_matrix(n_points=n_points)


@pytest.mark.parametrize("n_tied", [2, 5, 9, 40])
def test_ties_at_the_maximum_are_not_padding(n_tied):
    """The regression. A per-row plateau check flagged all of these.

    np.interp holds a tied value from the q-position of the first tied sample
    onward, so ties lengthen the trailing run of maxima exactly as padding does.
    The distinguishing fact is WHERE the first maximum sits: a tied row keeps it
    near the interpolation clamp, while padding puts it at n-1.
    """
    matrix = _row_with_tied_maximum(5_949, n_tied)
    counts = np.array([5_949])
    # The per-row check is what must be tie-immune. The file-level one deliberately
    # refuses to judge a single row — see test_file_level_check_needs_enough_rows.
    assert cdf_grid_violations(matrix, counts) == []






