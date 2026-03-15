"""Oracle state manager — singleton that caches loaded oracles across MCP tool calls."""

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
        self._reference_fasta: str | None = None
        self._output_dir: str = str(Path.cwd() / "chorus_mcp_output")
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

        oracle_kwargs: dict = {}
        if device:
            oracle_kwargs["device"] = device
        if self._reference_fasta:
            oracle_kwargs["reference_fasta"] = self._reference_fasta
        oracle_kwargs.update(kwargs)

        t0 = time.time()
        oracle = create_oracle(name, use_environment=True, **oracle_kwargs)

        # ChromBPNet needs assay/cell_type passed to load_pretrained_model
        load_kwargs: dict = {}
        if name == "chrombpnet":
            if "assay" in kwargs:
                load_kwargs["assay"] = kwargs["assay"]
            if "cell_type" in kwargs:
                load_kwargs["cell_type"] = kwargs["cell_type"]
        oracle.load_pretrained_model(**load_kwargs)

        elapsed = time.time() - t0
        self._oracles[name] = oracle
        self._load_times[name] = round(elapsed, 1)

        return {
            "name": name,
            "status": "loaded",
            "device": getattr(oracle, "device", None),
            "load_time_seconds": self._load_times[name],
        }

    def get_oracle(self, name: str):
        """Return a loaded oracle or raise if not loaded."""
        name = name.lower()
        if name not in self._oracles:
            raise RuntimeError(
                f"Oracle '{name}' is not loaded. Call load_oracle('{name}') first."
            )
        return self._oracles[name]

    def unload_oracle(self, name: str) -> bool:
        """Remove an oracle from the cache and free memory."""
        name = name.lower()
        if name not in self._oracles:
            return False
        del self._oracles[name]
        self._load_times.pop(name, None)
        return True

    def list_loaded(self) -> list[dict]:
        """Return info on every loaded oracle."""
        return [
            {
                "name": name,
                "device": getattr(oracle, "device", None),
                "load_time_seconds": self._load_times.get(name),
            }
            for name, oracle in self._oracles.items()
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def reference_fasta(self) -> str | None:
        return self._reference_fasta

    @property
    def output_dir(self) -> str:
        return self._output_dir
