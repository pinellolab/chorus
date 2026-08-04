"""The stamped ``build_config`` has to be read, or it is documentation not an invariant.

``build_and_save`` and ``append_tracks`` learned to write and preserve a file-level
``build_config`` (#124), and the three rebuilt backgrounds carry one. But nothing in
``chorus/`` read it: a grep for ``build_config`` found only the writer, the
append-path preservation, and docstrings. Provenance nobody consults cannot catch
anything.

What it should catch is #122, which is the reason the field exists. AlphaGenome
histone CHIP tracks had their null built over 501 bp while the query summed 2001 bp.
Both artefacts were internally consistent, so neither looked wrong; the defect only
existed *between* them, and it shipped across 1,075 of 5,168 tracks. A stamped build
geometry plus a load-time comparison makes that a detectable disagreement rather
than an invisible one.

The check warns rather than raises, and that is deliberate. Every background built
before the provenance work is unstamped, and an unstamped file makes no claim to
contradict. Refusing to load one would break working installs to enforce a metadata
convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"
REBUILT = ("alphagenome", "borzoi", "enformer")


def _make_npz(tmp_path: Path, oracle: str, config: dict | None) -> Path:
    """A minimal but genuine one-track background, optionally stamped."""
    row = np.linspace(0.0, 1.0, 1_000)[None, :]
    return Path(PerTrackNormalizer.build_and_save(
        oracle_name=oracle,
        track_ids=["CHIP_HISTONE/EFO:1 Histone ChIP-seq H3K27ac/."],
        effect_cdfs=row,
        effect_counts=[1_000],
        cache_dir=str(tmp_path),
        n_points=1_000,
        provenance=config,
    ))


# ---------------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------------


def test_provenance_round_trips_through_build_and_save(tmp_path):
    config = {"schema_version": 2, "genome": "hg38", "histone_window_bp": 2001,
              "other_window_bp": 501, "resolution": 1}
    _make_npz(tmp_path, "provtest", config)
    got = PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert got == config


def test_provenance_is_none_for_an_unstamped_background(tmp_path):
    _make_npz(tmp_path, "provtest", None)
    assert PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest") is None


def test_unparseable_provenance_is_treated_as_absent(tmp_path, caplog):
    """A corrupt stamp must not take the whole background down with it."""
    path = _make_npz(tmp_path, "provtest", {"genome": "hg38"})
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["build_config"] = np.array(["{not json"])
    np.savez_compressed(path, **payload)
    with caplog.at_level("WARNING"):
        assert PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest") is None
    assert "will not parse" in caplog.text


# ---------------------------------------------------------------------------
# The check that makes it load-bearing
# ---------------------------------------------------------------------------


def test_a_stamped_window_mismatch_warns_loudly(tmp_path, caplog):
    """The #122 shape: built at 501 bp, queried at 2001 bp."""
    _make_npz(tmp_path, "provtest", {
        "schema_version": 2, "genome": "hg38",
        "histone_window_bp": 501,       # WRONG on purpose — the #122 defect
        "other_window_bp": 501, "resolution": 1,
    })
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over 501 bp" in caplog.text
    assert "histone_marks" in caplog.text
    assert "#122" in caplog.text


def test_a_matching_stamp_is_silent(tmp_path, caplog):
    from chorus.analysis.scorers import LAYER_CONFIGS

    _make_npz(tmp_path, "provtest", {
        "schema_version": 2, "genome": "hg38",
        "histone_window_bp": LAYER_CONFIGS["histone_marks"].window_bp,
        "other_window_bp": LAYER_CONFIGS["tf_binding"].window_bp,
        "resolution": 1,
    })
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over" not in caplog.text


def test_an_unstamped_background_does_not_warn(tmp_path, caplog):
    """Silence is the correct behaviour: it makes no claim to contradict."""
    _make_npz(tmp_path, "provtest", None)
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over" not in caplog.text


# ---------------------------------------------------------------------------
# The shipped artefacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", REBUILT)
def test_the_rebuilt_backgrounds_are_stamped_and_agree_with_the_query(oracle, caplog):
    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    from chorus.analysis.scorers import LAYER_CONFIGS

    norm = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    with caplog.at_level("WARNING"):
        config = norm.provenance(oracle)
    assert isinstance(config, dict), f"{oracle} carries no build_config"
    assert config["genome"] == "hg38"
    assert config["histone_window_bp"] == LAYER_CONFIGS["histone_marks"].window_bp
    assert config["other_window_bp"] == LAYER_CONFIGS["tf_binding"].window_bp
    assert "was BUILT over" not in caplog.text, "shipped geometry disagrees"

    # The stamp must describe the file it is attached to, not a different build.
    assert config["n_tracks"] == norm.n_tracks(oracle)
    assert config["cdf_points"] == norm._ensure_loaded(oracle)["effect_cdfs"].shape[1]


@pytest.mark.parametrize("oracle", REBUILT)
def test_the_rebuilt_backgrounds_record_the_region_set_they_logged(oracle):
    """The strongest field in the stamp: what the build itself reported sampling.

    It outranks mtimes and commit shas — borzoi and enformer began eight minutes
    BEFORE the commit that introduced gene-anchored sampling, so file mtimes read as
    changed-after-build for both, while their logs show the strata from the first
    minute.
    """
    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    config = PerTrackNormalizer(cache_dir=str(BACKGROUNDS)).provenance(oracle)
    logged = config.get("effect_region_set_as_logged") or {}
    assert logged.get("available") is True, f"{oracle}: {logged.get('reason')}"
    counts = logged["strata_counts"]
    assert set(counts) == {"tss_near", "tss_far", "junction", "gene_body", "random"}
    # All three oracles drew from ONE seeded region set, which is what makes their
    # percentiles comparable. Identical counts is the evidence.
    assert counts == {"tss_near": 1200, "tss_far": 1200, "junction": 1980,
                      "gene_body": 720, "random": 849}
    assert logged["n_positions"] == 6_000
