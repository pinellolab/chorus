"""Oracle state manager — singleton that caches loaded oracles across MCP tool calls."""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from chorus.utils.genome import GenomeManager

logger = logging.getLogger(__name__)


class OracleStateManager:
    """Manages loaded oracle instances and shared state for the MCP server.

    Oracles take 30s–5min to load, so we cache them in-memory and reuse across
    tool calls within the same server process.
    """

    _instance: Optional["OracleStateManager"] = None

    def __new__(cls) -> "OracleStateManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._oracles: dict = {}          # name → loaded oracle instance
        self._load_times: dict = {}       # name → seconds it took to load
        self._normalizers: dict = {}      # name → QuantileNormalizer or None
        self._reference_fasta: str | None = None
        self._output_dir: str = os.environ.get("CHORUS_MCP_OUTPUT_DIR", str(Path.cwd() / "chorus_mcp_output"))
        self._initialised = True

        # Auto-detect hg38 reference
        try:
            gm = GenomeManager()
            if gm.is_genome_downloaded("hg38"):
                self._reference_fasta = str(gm.get_genome_path("hg38"))
                logger.info("Auto-detected hg38 at %s", self._reference_fasta)
        except Exception as exc:
            logger.warning("Could not auto-detect hg38 reference: %s", exc)

    # ------------------------------------------------------------------
    # Oracle lifecycle
    # ------------------------------------------------------------------

    def load_oracle(
        self,
        name: str,
        device: str | None = None,
        **kwargs,
    ) -> dict:
        """Create an oracle, load its pretrained model, and cache it.

        Returns a summary dict with name, device, and load time.
        """
        name = name.lower()
        if name in self._oracles:
            return {
                "name": name,
                "status": "already_loaded",
                "load_time_seconds": self._load_times.get(name),
            }

        from chorus import create_oracle

        # Separate constructor kwargs from load-time kwargs (assay/cell_type
        # are only used by load_pretrained_model, not the oracle constructor).
        _load_only_keys = {"assay", "cell_type", "TF", "fold", "model_type"}
        oracle_kwargs: dict = {}
        if device:
            oracle_kwargs["device"] = device
        if self._reference_fasta:
            oracle_kwargs["reference_fasta"] = self._reference_fasta
        oracle_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in _load_only_keys}
        )

        t0 = time.time()
        oracle = create_oracle(name, use_environment=True, **oracle_kwargs)

        # Pass load-time kwargs (assay, cell_type, TF, fold, model_type)
        load_kwargs: dict = {}
        for key in ("assay", "cell_type", "TF", "fold", "model_type"):
            if key in kwargs:
                load_kwargs[key] = kwargs[key]
        oracle.load_pretrained_model(**load_kwargs)

        elapsed = time.time() - t0
        self._oracles[name] = oracle
        self._load_times[name] = round(elapsed, 1)

        # Auto-load background distributions for quantile normalization
        self._auto_load_normalizer(name)

        resolved_device = getattr(oracle, "device", None)
        if resolved_device is None:
            resolved_device = self._detect_device(oracle)

        result = {
            "name": name,
            "status": "loaded",
            "device": resolved_device,
            "load_time_seconds": self._load_times[name],
        }
        normalizer = self._normalizers.get(name)
        if normalizer is not None:
            from chorus.analysis.normalization import PerTrackNormalizer
            if isinstance(normalizer, PerTrackNormalizer):
                summary = normalizer.summary(name)
                result["backgrounds"] = {
                    "status": "loaded",
                    "type": "per_track",
                    "n_tracks": summary.get("n_tracks", 0),
                    "cdfs": list(summary.get("cdfs", {}).keys()),
                }
            else:
                summary = normalizer.summary()
                result["backgrounds"] = {
                    "status": "loaded",
                    "type": "per_layer",
                    "n_layers": len(summary),
                    "layers": {k: v["n_scores"] for k, v in summary.items()},
                }
        else:
            result["backgrounds"] = {"status": "none"}
        return result

    @staticmethod
    def _detect_device(oracle) -> str:
        """Best-effort detection of the compute device when auto-detect was used.

        Oracles run in isolated envs so we can't import their frameworks here.
        Instead, check OS-level GPU availability.
        """
        import shutil
        import subprocess
        import sys
        try:
            if shutil.which("nvidia-smi"):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return "gpu (auto-detected via nvidia-smi)"
        except Exception:
            pass
        if sys.platform == "darwin":
            return "mps (auto-detected, macOS)"
        return "cpu (no GPU detected)"

    def get_oracle(self, name: str):
        """Return a loaded oracle or raise if not loaded."""
        name = name.lower()
        if name not in self._oracles:
            raise RuntimeError(
                f"Oracle '{name}' is not loaded. Call load_oracle('{name}') first."
            )
        return self._oracles[name]

    def _auto_load_normalizer(self, name: str) -> None:
        """Auto-discover and cache normalizers for the named oracle.

        Tries per-track normalizer first (preferred), then falls back to
        legacy per-layer normalizer.
        """
        from chorus.analysis.normalization import get_pertrack_normalizer, get_normalizer

        # Try per-track normalizer first
        try:
            pt_norm = get_pertrack_normalizer(name)
            if pt_norm is not None:
                self._normalizers[name] = pt_norm
                summary = pt_norm.summary(name)
                logger.info(
                    "Loaded per-track normalizer for '%s': %d tracks, CDFs: %s",
                    name, summary.get("n_tracks", 0),
                    list(summary.get("cdfs", {}).keys()),
                )
                return
        except Exception as exc:
            logger.warning("Failed to load per-track normalizer for '%s': %s", name, exc)

        # Fall back to legacy per-layer normalizer
        try:
            normalizer = get_normalizer(name)
            self._normalizers[name] = normalizer
            if normalizer is not None:
                logger.info(
                    "Loaded legacy normalizer for '%s' with %d backgrounds",
                    name, len(normalizer.summary()),
                )
        except Exception as exc:
            logger.warning("Failed to load normalizer for '%s': %s", name, exc)
            self._normalizers[name] = None

    def get_normalizer(self, name: str):
        """Return the cached normalizer for an oracle, or None.

        May return a :class:`PerTrackNormalizer` or a legacy
        :class:`QuantileNormalizer` depending on what's available.
        """
        return self._normalizers.get(name.lower())

    def unload_oracle(self, name: str) -> bool:
        """Remove an oracle from the cache and free memory."""
        name = name.lower()
        if name not in self._oracles:
            return False
        del self._oracles[name]
        self._load_times.pop(name, None)
        self._normalizers.pop(name, None)
        return True

    def list_loaded(self) -> list[dict]:
        """Return info on every loaded oracle."""
        result = []
        for name, oracle in self._oracles.items():
            info: dict = {
                "name": name,
                "device": getattr(oracle, "device", None),
                "load_time_seconds": self._load_times.get(name),
            }
            normalizer = self._normalizers.get(name)
            if normalizer is not None:
                from chorus.analysis.normalization import PerTrackNormalizer
                if isinstance(normalizer, PerTrackNormalizer):
                    info["backgrounds_loaded"] = normalizer.n_tracks(name)
                else:
                    info["backgrounds_loaded"] = len(normalizer.summary())
            else:
                info["backgrounds_loaded"] = 0
            result.append(info)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def reference_fasta(self) -> str | None:
        return self._reference_fasta

    @property
    def output_dir(self) -> str:
        return self._output_dir
