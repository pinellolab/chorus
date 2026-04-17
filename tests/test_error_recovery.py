"""Error-recovery tests for download / auth / env-missing paths.

These exercise the failure paths that existing infrastructure
(`download_with_resume`, `hf_hub_download`, `huggingface_hub.whoami`,
`EnvironmentManager.environment_exists`) already handle. Prior audits
verified the happy paths but not the error messages, so this suite
pins the behavior a new user would hit when something goes wrong.

All tests are mock-based and run in milliseconds — no network, no
filesystem beyond ``tmp_path``, no subprocesses. They stay in the fast
suite (no ``@pytest.mark.integration`` marker).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# TestDownloadFailurePaths
# ---------------------------------------------------------------------------

class TestDownloadFailurePaths:
    """Cover the two HTTP surfaces: huggingface_hub + stdlib urlopen."""

    def test_hf_hub_download_failure_returns_zero_and_does_not_crash(self, tmp_path, caplog):
        """``download_pertrack_backgrounds`` swallows HF failures and
        returns 0 so a caller (e.g. ``get_pertrack_normalizer``) can
        fall back to legacy ``.npy`` scan without blowing up.
        """
        from chorus.analysis import normalization

        # Mock the hf_hub_download that's imported inside the function
        with patch("huggingface_hub.hf_hub_download", side_effect=ConnectionError("network down")):
            with caplog.at_level(logging.WARNING, logger="chorus.analysis.normalization"):
                n = normalization.download_pertrack_backgrounds("sei", cache_dir=str(tmp_path))
        assert n == 0
        # The warning message should surface the oracle+reason for debugging
        assert any("Failed to download" in rec.message for rec in caplog.records)
        # Cache dir should still exist and be usable for a retry
        assert tmp_path.is_dir()
        # No partial file dropped
        assert not (tmp_path / "sei_pertrack.npz").exists()

    def test_download_with_resume_leaves_partial_and_resumes_on_second_call(self, tmp_path):
        """``download_with_resume`` writes to ``<dest>.partial`` while
        downloading. If ``urlopen`` raises mid-read, the partial file
        persists; the next call resumes via ``Range: bytes=<offset>-``.
        """
        from chorus.utils.http import download_with_resume

        dest = tmp_path / "model.tar.gz"
        url = "https://example.test/model.tar.gz"
        first_half = b"A" * 1024
        second_half = b"B" * 1024

        # First call: returns a response that yields first_half then raises.
        class RaisingStream:
            def __init__(self):
                self._served = False
            def read(self, n):
                if not self._served:
                    self._served = True
                    return first_half
                raise IOError("network dropped")

        fake_resp_1 = MagicMock()
        fake_resp_1.read = RaisingStream().read
        fake_resp_1.status = 200
        fake_resp_1.getcode.return_value = 200
        fake_resp_1.headers = {"Content-Length": str(len(first_half) + len(second_half))}

        with patch("urllib.request.urlopen", return_value=fake_resp_1):
            with pytest.raises(IOError, match="network dropped"):
                download_with_resume(url, dest)

        partial = dest.with_suffix(dest.suffix + ".partial")
        assert partial.exists(), "first call should leave a .partial file behind"
        assert partial.stat().st_size == len(first_half), (
            f"partial should contain the bytes we actually read "
            f"({len(first_half)}), got {partial.stat().st_size}"
        )
        assert not dest.exists(), "dest must not be promoted when download failed"

        # Second call: returns only the remaining half (server honored Range).
        class Stream2:
            def __init__(self):
                self._served = False
            def read(self, n):
                if not self._served:
                    self._served = True
                    return second_half
                return b""

        fake_resp_2 = MagicMock()
        fake_resp_2.read = Stream2().read
        fake_resp_2.status = 206
        fake_resp_2.getcode.return_value = 206
        fake_resp_2.headers = {"Content-Length": str(len(second_half))}

        # Capture the Request so we can verify the Range header was set.
        captured = {}
        def spy_urlopen(req, timeout=60):
            captured["range"] = req.get_header("Range")
            return fake_resp_2

        with patch("urllib.request.urlopen", side_effect=spy_urlopen):
            download_with_resume(url, dest)

        assert captured["range"] == f"bytes={len(first_half)}-"
        assert dest.exists()
        assert not partial.exists(), "partial should be promoted to dest"
        assert dest.read_bytes() == first_half + second_half


# ---------------------------------------------------------------------------
# TestAuthFailurePaths
# ---------------------------------------------------------------------------

class TestAuthFailurePaths:
    """AlphaGenome requires HF auth for the gated model. The error
    message must tell the user what to do."""

    def test_alphagenome_missing_hf_token_error_is_actionable(self, monkeypatch):
        """When both ``huggingface_hub.whoami()`` raises
        LocalTokenNotFoundError *and* HF_TOKEN / HUGGING_FACE_HUB_TOKEN
        are absent, ``AlphaGenomeOracle._load_direct`` must raise a
        message that names the env var and links to the license page.
        """
        from chorus.core.exceptions import ModelNotLoadedError
        from chorus.oracles.alphagenome import AlphaGenomeOracle

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        oracle = AlphaGenomeOracle.__new__(AlphaGenomeOracle)
        oracle.device = None

        import huggingface_hub
        import huggingface_hub.errors as hf_errors

        fake_hub = MagicMock()
        fake_hub.whoami.side_effect = hf_errors.LocalTokenNotFoundError("no token")
        fake_hub.errors = hf_errors
        fake_hub.login = huggingface_hub.login  # unused on the raise path

        # Patch both jax and alphagenome_research imports so _load_direct
        # gets to the whoami check. We just need to reach line 123.
        fake_jax = MagicMock()
        fake_jax.devices.return_value = [MagicMock()]
        fake_ag = MagicMock()

        with patch.dict("sys.modules", {
            "jax": fake_jax,
            "huggingface_hub": fake_hub,
            "alphagenome_research": MagicMock(),
            "alphagenome_research.model": MagicMock(),
            "alphagenome_research.model.dna_model": fake_ag,
        }):
            with pytest.raises(ModelNotLoadedError) as excinfo:
                oracle._load_direct(weights="all_folds")

        msg = str(excinfo.value)
        assert "HF_TOKEN" in msg, "error must name the env var the user should set"
        assert "huggingface.co/google/alphagenome" in msg, (
            "error must link to the license page so the user knows where to click"
        )
        assert "huggingface-cli login" in msg, "alternative auth path should be mentioned"


# ---------------------------------------------------------------------------
# TestEnvironmentFailurePaths
# ---------------------------------------------------------------------------

class TestEnvironmentFailurePaths:
    """When the oracle env doesn't exist, OracleBase must degrade
    gracefully (use_environment=False) and log a 'Run chorus setup'
    hint so the user knows what to do."""

    def test_missing_oracle_env_falls_back_gracefully(self, caplog):
        """Mock ``environment_exists`` to return False and verify the
        oracle (a) disables environment mode, (b) logs the actionable
        hint with the exact command to run.
        """
        from chorus.oracles.enformer import EnformerOracle

        with patch(
            "chorus.core.environment.EnvironmentManager.environment_exists",
            return_value=False,
        ):
            with caplog.at_level(logging.WARNING):
                oracle = EnformerOracle(use_environment=True)

        assert oracle.use_environment is False, (
            "oracle must degrade to in-process when env is missing"
        )
        msgs = " ".join(rec.message for rec in caplog.records)
        assert "chorus setup --oracle enformer" in msgs, (
            "log must quote the exact command the user needs to run"
        )
