"""A background must describe itself, not depend on whatever metadata is on disk (#124).

``append_tracks`` loaded **every** key from the existing NPZ but forwarded only the
canonical eight to ``build_and_save``, so anything else — a per-row ``layer``, a
file-level ``build_config`` — vanished on the first merge. Cherimoya works around
it by re-stamping ``build_config`` after every append, which is why it is the only
one of nine shipped NPZs that has one.

Why it matters concretely: Borzoi's ``track_ids`` are opaque FANTOM accessions
(``CNhs10608+``), so its 1,276 CAGE and 1,543 RNA rows are identifiable *today*
only by joining against ``borzoi_metadata.py``. All 7,611 resolve, so the join
works — but it binds a stored row to whatever version of that file is on disk at
read time. If the metadata ever gains, drops or reorders tracks, previously-built
rows are silently reinterpreted. That is #144's failure mode applied to metadata
instead of arithmetic.

The subtle part is the merge. A per-row array must be **concatenated**; passing it
through unchanged leaves an ``(n_old,)`` array against ``n_old + n_new`` rows,
mis-attributing every added row — worse than dropping it, because it looks
authoritative.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

N_POINTS = 64


def _cdf(n_tracks: int, n_points: int = N_POINTS) -> np.ndarray:
    return np.sort(np.random.default_rng(0).standard_normal((n_tracks, n_points)), axis=1)


def _save(tmp_path, oracle="provtest", n=3, **kw):
    return PerTrackNormalizer.build_and_save(
        oracle_name=oracle,
        track_ids=[f"T{i}" for i in range(n)],
        effect_cdfs=_cdf(n),
        effect_counts=[100] * n,
        cache_dir=str(tmp_path),
        n_points=N_POINTS,
        **kw,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_file_level_provenance_round_trips(tmp_path):
    prov = {
        "genome": "hg38",
        "xla_flags": "--xla_gpu_deterministic_ops=true",
        "region_strata": {"tss_near": 0.2, "random": 0.15},
        "schema_version": 2,
    }
    path = _save(tmp_path, provenance=prov)
    with np.load(path, allow_pickle=True) as data:
        assert json.loads(str(data["build_config"])) == prov


def test_per_row_provenance_round_trips(tmp_path):
    path = _save(tmp_path, per_row={
        "layer": np.array(["tss_activity", "gene_expression", "tf_binding"]),
        "window_bp": np.array([501, 0, 2001]),
        "resolution": np.array([1, 1, 128]),
    })
    with np.load(path, allow_pickle=True) as data:
        assert list(data["layer"]) == ["tss_activity", "gene_expression", "tf_binding"]
        assert list(data["window_bp"]) == [501, 0, 2001]


def test_a_wrong_length_per_row_array_is_rejected(tmp_path):
    """Validated, not trusted: a short array mis-attributes every later row."""
    with pytest.raises(ValueError, match="mis-attributes rows"):
        _save(tmp_path, per_row={"layer": np.array(["a", "b"])})  # 2 vs 3 tracks


def test_per_row_cannot_shadow_a_cdf_array(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        _save(tmp_path, per_row={"effect_counts": np.array([1, 2, 3])})


def test_no_provenance_still_writes_the_canonical_eight(tmp_path):
    """Provenance is additive — a builder that supplies none must be unaffected."""
    path = _save(tmp_path)
    with np.load(path, allow_pickle=True) as data:
        assert "build_config" not in data.files
        assert "effect_cdfs" in data.files


# ---------------------------------------------------------------------------
# The merge — where it was being dropped
# ---------------------------------------------------------------------------


def test_append_preserves_file_level_provenance(tmp_path):
    """The live reproduction of #124: this used to come back empty."""
    _save(tmp_path, n=3, provenance={"genome": "hg38", "schema_version": 2})
    PerTrackNormalizer.append_tracks(
        oracle_name="provtest",
        new_track_ids=["T3", "T4"],
        new_effect_cdfs=_cdf(2),
        new_effect_counts=[100, 100],
        cache_dir=str(tmp_path),
    )
    with np.load(tmp_path / "provtest_pertrack.npz", allow_pickle=True) as data:
        assert "build_config" in data.files, "provenance dropped on merge (#124)"
        assert json.loads(str(data["build_config"]))["genome"] == "hg38"
        assert len(data["track_ids"]) == 5


def test_append_concatenates_per_row_provenance(tmp_path):
    _save(tmp_path, n=3, per_row={"layer": np.array(["a", "b", "c"])})
    PerTrackNormalizer.append_tracks(
        oracle_name="provtest",
        new_track_ids=["T3", "T4"],
        new_effect_cdfs=_cdf(2),
        new_effect_counts=[100, 100],
        cache_dir=str(tmp_path),
        new_per_row={"layer": np.array(["d", "e"])},
    )
    with np.load(tmp_path / "provtest_pertrack.npz", allow_pickle=True) as data:
        assert list(data["layer"]) == ["a", "b", "c", "d", "e"]
        assert len(data["layer"]) == len(data["track_ids"])


def test_per_row_length_always_matches_the_row_count_after_append(tmp_path):
    """The invariant that makes per-row provenance trustworthy at all."""
    _save(tmp_path, n=3, per_row={"layer": np.array(["a", "b", "c"])})
    for i in range(3):
        PerTrackNormalizer.append_tracks(
            oracle_name="provtest",
            new_track_ids=[f"X{i}"],
            new_effect_cdfs=_cdf(1),
            new_effect_counts=[100],
            cache_dir=str(tmp_path),
            new_per_row={"layer": np.array([f"L{i}"])},
        )
        with np.load(tmp_path / "provtest_pertrack.npz", allow_pickle=True) as data:
            assert len(data["layer"]) == len(data["track_ids"])


def test_unsupplied_per_row_key_is_dropped_loudly_not_misaligned(tmp_path, caplog):
    """Dropping beats mis-aligning.

    An ``(n_old,)`` array left against ``n_old + n_new`` rows would look
    authoritative while being wrong for every added row.
    """
    _save(tmp_path, n=3, per_row={"layer": np.array(["a", "b", "c"])})
    with caplog.at_level("WARNING"):
        PerTrackNormalizer.append_tracks(
            oracle_name="provtest",
            new_track_ids=["T3"],
            new_effect_cdfs=_cdf(1),
            new_effect_counts=[100],
            cache_dir=str(tmp_path),
        )
    assert "layer" in caplog.text and "dropping" in caplog.text
    with np.load(tmp_path / "provtest_pertrack.npz", allow_pickle=True) as data:
        assert "layer" not in data.files
        assert len(data["track_ids"]) == 4


def test_new_provenance_overrides_the_stored_one(tmp_path):
    _save(tmp_path, n=3, provenance={"genome": "hg38", "schema_version": 1})
    PerTrackNormalizer.append_tracks(
        oracle_name="provtest",
        new_track_ids=["T3"],
        new_effect_cdfs=_cdf(1),
        new_effect_counts=[100],
        cache_dir=str(tmp_path),
        new_provenance={"genome": "hg38", "schema_version": 2},
    )
    with np.load(tmp_path / "provtest_pertrack.npz", allow_pickle=True) as data:
        assert json.loads(str(data["build_config"]))["schema_version"] == 2


# ---------------------------------------------------------------------------
# Why Borzoi needs this at all
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_borzoi_layers_are_recoverable_today_but_only_by_joining(tmp_path):
    """Documents the dependency this replaces.

    All 7,611 track_ids resolve against the vendored metadata, so the join works —
    Borzoi is *not* blocked on provenance, contrary to what I claimed earlier. But
    the join binds a stored row to whatever version of borzoi_metadata.py is on
    disk, and nothing detects a mismatch.
    """
    from pathlib import Path

    npz = Path("/data/chorus_data/backgrounds/borzoi_pertrack.npz")
    if not npz.exists():
        pytest.skip("borzoi background not downloaded")

    from chorus.oracles.borzoi_source.borzoi_metadata import BorzoiMetadata

    frame = BorzoiMetadata().tracks_df
    by_id = dict(zip(frame["identifier"].astype(str), frame["description"].astype(str)))
    with np.load(npz, allow_pickle=True) as data:
        ids = [str(t) for t in data["track_ids"]]
        has_layer = "layer" in data.files

    unresolved = [t for t in ids if t not in by_id]
    assert not unresolved, f"{len(unresolved)} track_ids do not resolve"
    if not has_layer:
        pytest.xfail("shipped borzoi NPZ predates per-row layer; rebuild adds it")
