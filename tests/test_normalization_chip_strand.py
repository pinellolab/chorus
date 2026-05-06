"""Tests that CHIP strand-suffixed track IDs (e.g. CHIP:K562:REST:+) resolve
to the correct CDF row even though the NPZ stores the strand-less key
CHIP:K562:REST.  This covers the silent fallback bug where all CHIP
normalization lookups returned None and IGV reports fell back to raw values.
"""

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer


def _make_normalizer(monkeypatch):
    """Return a PerTrackNormalizer whose _ensure_loaded is monkeypatched to
    return a minimal in-memory dict with one CHIP track (no strand suffix).
    """
    norm = PerTrackNormalizer.__new__(PerTrackNormalizer)
    norm.cache_dir = None
    norm._loaded = {}

    cdf = np.linspace(0.0, 10.0, 100)  # simple monotone CDF
    entry = {
        "track_ids": ["CHIP:K562:REST"],
        "track_index": {"CHIP:K562:REST": 0},
        "effect_cdfs": cdf[np.newaxis, :],
        "summary_cdfs": cdf[np.newaxis, :],
        "perbin_cdfs": cdf[np.newaxis, :],
        "effect_counts": np.array([100], dtype=np.int64),
        "summary_counts": np.array([100], dtype=np.int64),
        "perbin_counts": np.array([100], dtype=np.int64),
        "signed_flags": np.array([False]),
    }

    def _fake_ensure_loaded(oracle_name):
        return entry

    monkeypatch.setattr(norm, "_ensure_loaded", _fake_ensure_loaded)
    return norm


@pytest.mark.parametrize("strand", ["+", "-"])
def test_effect_percentile_chip_strand(monkeypatch, strand):
    norm = _make_normalizer(monkeypatch)
    track_id = f"CHIP:K562:REST:{strand}"
    result = norm.effect_percentile("chrombpnet", track_id, 5.0)
    assert result is not None, f"effect_percentile returned None for {track_id}"
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("strand", ["+", "-"])
def test_activity_percentile_chip_strand(monkeypatch, strand):
    norm = _make_normalizer(monkeypatch)
    track_id = f"CHIP:K562:REST:{strand}"
    result = norm.activity_percentile("chrombpnet", track_id, 5.0)
    assert result is not None, f"activity_percentile returned None for {track_id}"
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("strand", ["+", "-"])
def test_perbin_floor_rescale_chip_strand(monkeypatch, strand):
    norm = _make_normalizer(monkeypatch)
    track_id = f"CHIP:K562:REST:{strand}"
    result = norm.perbin_floor_rescale_batch(
        "chrombpnet", track_id, np.array([3.0, 5.0, 8.0])
    )
    assert result is not None, f"perbin_floor_rescale_batch returned None for {track_id}"
    assert result.shape == (3,)


def test_strandless_key_still_works(monkeypatch):
    """Strand-less CHIP keys (e.g. from direct lookup) must still resolve."""
    norm = _make_normalizer(monkeypatch)
    result = norm.effect_percentile("chrombpnet", "CHIP:K562:REST", 5.0)
    assert result is not None
