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

    #: Model-selecting load arguments, and the oracle attribute each one is recorded under once a
    #: model is loaded (ChromBPNet stores ``TF`` as ``tf``). Read back from the oracle rather than
    #: from the arguments of the original call, because ``discover_variant`` reloads per-model
    #: oracles in place and leaves whichever model it visited last.
    _SELECTION_ATTRS = {
        "assay": "assay", "cell_type": "cell_type", "TF": "tf",
        "fold": "fold", "model_type": "model_type", "encode_id": "encode_id",
    }

    @staticmethod
    def _selection_params(fn):
        """The model-selecting argument names ``fn`` accepts, or None if it takes ``**kwargs``
        (then it cannot be vetted, so everything is passed through as before)."""
        import inspect

        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return None
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return None
        return set(params) & set(OracleStateManager._SELECTION_ATTRS)

    @classmethod
    def _refuse_unknown_selection(cls, name: str, oracle, requested: dict) -> None:
        """Refuse, by name, a selection argument this oracle takes neither at load nor at init.

        Without this the argument reached ``load_pretrained_model`` and surfaced as a bare
        ``TypeError`` from inside the load -- or, on a repeat call, as advice to unload and load
        again, which then failed the same way and lost the working model.
        """
        at_load = cls._selection_params(oracle.load_pretrained_model)
        at_init = cls._selection_params(type(oracle).__init__)
        if at_load is None or at_init is None:
            return
        takes = at_load | at_init
        unknown = sorted(k for k, v in requested.items() if v is not None and k not in takes)
        if unknown:
            raise ValueError(
                f"{name} does not take {', '.join(unknown)}. Its model is chosen with "
                f"{', '.join(sorted(takes)) or 'no arguments'}."
            )

    def _loaded_selection(self, oracle) -> dict:
        """Which model a loaded oracle holds, read from the oracle itself."""
        loaded = {}
        for key, attr in self._SELECTION_ATTRS.items():
            value = getattr(oracle, attr, None)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                loaded[key] = value
        return loaded

    def _already_loaded(self, name: str, requested: dict) -> dict:
        """Answer a repeat ``load_oracle`` call without silently serving a different model.

        This used to return ``already_loaded`` whatever was asked for. ChromBPNet, Cherimoya,
        LegNet and EPInformer-seq hold one model per load, so ``load_oracle('chrombpnet',
        assay='DNASE', cell_type='K562')`` after HepG2 reported success and kept HepG2 — and the
        reply did not say which model was loaded, so a caller had no way to notice.
        """
        oracle = self._oracles[name]
        self._refuse_unknown_selection(name, oracle, requested)
        loaded = self._loaded_selection(oracle)

        def _same(a, b) -> bool:
            if isinstance(a, str) and isinstance(b, str):
                return a.lower() == b.lower()
            return a == b

        # Compare against the attribute itself, not only the displayed values: a ChromBPNet
        # DNASE model has `tf = None`, and a request for TF='CTCF' is a different model, not an
        # unverifiable one. Only an attribute the oracle does not have at all is unverifiable.
        _missing = object()
        mismatched = {}
        unverified = []
        for k, v in requested.items():
            if v is None:
                continue
            have = getattr(oracle, self._SELECTION_ATTRS[k], _missing)
            if have is _missing:
                unverified.append(k)
            elif not _same(have, v):
                mismatched[k] = v
        if mismatched:
            asked = ", ".join(f"{k}={v!r}" for k, v in mismatched.items())
            have = ", ".join(f"{k}={v!r}" for k, v in loaded.items())
            raise ValueError(
                f"{name} is already loaded with a different model ({have}); this call asked for "
                f"{asked}. One model is held per oracle name: call unload_oracle({name!r}) and "
                f"then load_oracle again to switch."
            )
        result = {
            "name": name,
            "status": "already_loaded",
            "load_time_seconds": self._load_times.get(name),
            "loaded_model": loaded,
        }
        if unverified:
            result["note"] = (
                f"Could not confirm {', '.join(sorted(unverified))} against the loaded model; if you need "
                f"a different one, call unload_oracle({name!r}) first."
            )
        return result

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
            return self._already_loaded(name, kwargs)

        from chorus import create_oracle

        # Separate constructor kwargs from load-time kwargs (assay/cell_type
        # are only used by load_pretrained_model, not the oracle constructor).
        # `encode_id` (Cherimoya) selects a specific CATv1 ENCODE
        # experiment. Listed here rather than only in the loop below --
        # this set and that loop must stay in sync, and a key missing from
        # *this* one silently lands in the constructor and raises
        # `TypeError: __init__() got an unexpected keyword argument`.
        _LOAD_ONLY_KEYS = ("assay", "cell_type", "TF", "fold", "model_type", "encode_id")

        oracle_kwargs: dict = {}
        if device:
            oracle_kwargs["device"] = device
        if self._reference_fasta:
            oracle_kwargs["reference_fasta"] = self._reference_fasta
        oracle_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in _LOAD_ONLY_KEYS}
        )

        t0 = time.time()
        oracle = create_oracle(name, use_environment=True, **oracle_kwargs)

        # Route each selection argument to where this oracle takes it. Most per-model oracles
        # choose their model in load_pretrained_model, but EPInformer-seq takes `assay` (and
        # AlphaGenome/Borzoi take `fold`) only in the constructor -- sent to the load call, those
        # raised TypeError, so 22 of EPInformer-seq's 33 tracks were unreachable over MCP.
        selection = {k: kwargs[k] for k in _LOAD_ONLY_KEYS if k in kwargs}
        self._refuse_unknown_selection(name, oracle, selection)
        at_load = self._selection_params(oracle.load_pretrained_model)
        if at_load is None:
            at_load = set(selection)
        load_kwargs = {k: v for k, v in selection.items() if k in at_load}
        at_init = {k: v for k, v in selection.items() if k not in at_load}
        if at_init:
            oracle = create_oracle(name, use_environment=True, **oracle_kwargs, **at_init)
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
            "loaded_model": self._loaded_selection(oracle),
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
                # Read live: discover_variant reloads per-model oracles in place, so the model
                # held now can differ from the one load_oracle was asked for.
                "loaded_model": self._loaded_selection(oracle),
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
