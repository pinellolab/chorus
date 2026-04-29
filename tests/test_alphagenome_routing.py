"""Unit tests for the AlphaGenome backend-routing helper.

These are fast (no network, no model load) — checks that the
recommendation logic returns the right oracle/device for each
host × window-size combination grounded in the audit data.
"""
from __future__ import annotations

import pytest

from chorus import recommend_alphagenome_backend


# ---------------------------------------------------------------------------
# macOS + MPS (the chorus dev machine — most-tested path)
# ---------------------------------------------------------------------------

class TestMacOSWithMPS:
    common = {"system": "Darwin", "has_cuda": False, "has_mps": True}

    def test_small_window_picks_pt_mps(self):
        rec = recommend_alphagenome_backend(128_000, **self.common)
        assert rec["oracle"] == "alphagenome_pt"
        assert rec["device"] == "mps"
        assert rec["confidence"] == "high"

    def test_524kb_picks_pt_mps(self):
        rec = recommend_alphagenome_backend(524_288, **self.common)
        assert rec["oracle"] == "alphagenome_pt"
        assert rec["device"] == "mps"

    def test_just_above_safe_zone_picks_jax(self):
        rec = recommend_alphagenome_backend(601_000, **self.common)
        assert rec["oracle"] == "alphagenome"
        assert rec["device"] == "cpu"

    def test_1mb_picks_jax(self):
        """1 MB is post-cliff on Mac MPS — must route to JAX."""
        rec = recommend_alphagenome_backend(1_048_576, **self.common)
        assert rec["oracle"] == "alphagenome"
        assert rec["device"] == "cpu"
        assert rec["confidence"] == "high"
        assert "cache cliff" in rec["reason"]


# ---------------------------------------------------------------------------
# Linux + CUDA
# ---------------------------------------------------------------------------

class TestLinuxWithCUDA:
    def test_picks_pt_cuda_for_any_window(self):
        for L in (128_000, 524_288, 1_048_576):
            rec = recommend_alphagenome_backend(
                L, system="Linux", has_cuda=True, has_mps=False,
            )
            assert rec["oracle"] == "alphagenome_pt"
            assert rec["device"] == "cuda"


# ---------------------------------------------------------------------------
# Fallback: anything without GPU acceleration → JAX CPU
# ---------------------------------------------------------------------------

class TestNoGPU:
    def test_mac_intel_falls_back_to_jax(self):
        rec = recommend_alphagenome_backend(
            524_288, system="Darwin", has_cuda=False, has_mps=False,
        )
        assert rec["oracle"] == "alphagenome"

    def test_linux_no_cuda_falls_back_to_jax(self):
        rec = recommend_alphagenome_backend(
            524_288, system="Linux", has_cuda=False, has_mps=False,
        )
        assert rec["oracle"] == "alphagenome"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_zero_window_size_rejected():
    with pytest.raises(ValueError):
        recommend_alphagenome_backend(0, system="Darwin", has_mps=True, has_cuda=False)


def test_negative_window_size_rejected():
    with pytest.raises(ValueError):
        recommend_alphagenome_backend(-1, system="Darwin", has_mps=True, has_cuda=False)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_result_keys():
    rec = recommend_alphagenome_backend(
        524_288, system="Darwin", has_cuda=False, has_mps=True,
    )
    for key in ("oracle", "device", "reason", "confidence", "benchmarks"):
        assert key in rec
    assert rec["oracle"] in {"alphagenome", "alphagenome_pt"}
    assert rec["confidence"] in {"high", "medium"}
