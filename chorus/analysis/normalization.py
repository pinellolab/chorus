"""Quantile normalization against background distributions.

Per-Track Normalization (primary system)
========================================

Each oracle track gets its own CDF, stored in a single ``.npz`` file per
oracle: ``{oracle}_pertrack.npz``.  Three CDF matrices are stored:

- ``effect_cdfs``  — ``(n_tracks, N)`` sorted |effect| values → Effect %ile
- ``summary_cdfs`` — ``(n_tracks, N)`` sorted window-sum values → Activity %ile
- ``perbin_cdfs``  — ``(n_tracks, N)`` sorted per-bin values → IGV visualization

A ``track_ids`` array maps row index → track identifier (e.g. ``ENCFF123ABC``).

Per-Layer Normalization (legacy)
================================

Two kinds of background are supported:

**1. Variant effect backgrounds** (``{oracle}_{layer}.npy``)
    Maps raw variant effect scores to quantile scores.

**2. Baseline signal backgrounds** (``{oracle}_{layer}_baseline.npy``)
    Maps raw predicted signal levels to genome-wide activity percentiles [0, 1].

Background distributions are cached to ``~/.chorus/backgrounds/`` for reuse.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BackgroundDistribution:
    """Pre-computed score distribution for quantile normalization.

    Stores a sorted 1-D array of background scores (e.g. from common
    variants or random genomic regions).  Maps any raw score to its
    quantile position in the background via empirical CDF.
    """

    def __init__(self, scores: np.ndarray):
        self.sorted_scores = np.sort(scores.ravel()).astype(np.float64)

    @classmethod
    def from_scores(cls, scores: np.ndarray, signed: bool = True) -> "BackgroundDistribution":
        """Create a background distribution with correct sign handling.

        For **unsigned** layers (chromatin, ChIP, CAGE, splicing), pass
        ``signed=False`` — the scores will be converted to absolute values
        before sorting, so that quantiles reflect effect magnitude only.

        For **signed** layers (MPRA, expression, Sei), pass ``signed=True``
        (the default) — scores are stored as-is.

        Args:
            scores: 1-D array of raw effect scores from background variants.
            signed: Whether the layer uses signed normalization.
        """
        arr = scores.ravel()
        if not signed:
            arr = np.abs(arr)
        return cls(arr)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def raw_to_quantile(self, raw_score: float, signed: bool = True) -> float:
        """Map a raw score to its quantile in the background.

        Args:
            raw_score: The raw effect score to normalise.
            signed: If ``True`` map to [-1, 1]; otherwise [0, 1].
        """
        rank = np.searchsorted(self.sorted_scores, raw_score, side="right")
        quantile = rank / len(self.sorted_scores)
        if signed:
            return 2.0 * quantile - 1.0
        return quantile

    def raw_to_quantile_batch(
        self, raw_scores: np.ndarray, signed: bool = True,
    ) -> np.ndarray:
        """Map an array of raw scores to quantiles."""
        ranks = np.searchsorted(self.sorted_scores, raw_scores, side="right")
        quantiles = ranks.astype(np.float64) / len(self.sorted_scores)
        if signed:
            return 2.0 * quantiles - 1.0
        return quantiles

    # ------------------------------------------------------------------
    # CDF compression
    # ------------------------------------------------------------------

    def to_compact_cdf(self, n_points: int = 10_000) -> "BackgroundDistribution":
        """Compress the distribution to a compact CDF lookup table.

        Reduces millions of stored values to *n_points* evenly-spaced
        percentile values (default 10,000 → ~80 KB).  The resulting
        distribution produces the same ``raw_to_quantile`` results within
        ±1/n_points precision — indistinguishable for visualization.

        Returns a new BackgroundDistribution backed by the compact CDF.
        """
        n = len(self.sorted_scores)
        if n <= n_points:
            return self  # already compact
        # Sample at evenly-spaced percentile positions
        indices = np.linspace(0, n - 1, n_points, dtype=int)
        compact = self.sorted_scores[indices].copy()
        return BackgroundDistribution(compact)

    @property
    def is_compact(self) -> bool:
        """Whether this distribution is already compact (≤10K points)."""
        return len(self.sorted_scores) <= 10_000

    @property
    def nbytes(self) -> int:
        """Memory/storage size in bytes."""
        return self.sorted_scores.nbytes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, compact: bool = False) -> None:
        """Save the sorted background to a ``.npy`` file.

        Args:
            path: File path.
            compact: If ``True``, compress to a 10K-point CDF table first.
                Reduces file size from hundreds of MB to ~80 KB with
                negligible loss of precision.
        """
        dist = self.to_compact_cdf() if compact else self
        np.save(path, dist.sorted_scores)

    @classmethod
    def load(cls, path: str) -> "BackgroundDistribution":
        """Load a background distribution from a ``.npy`` file."""
        return cls(np.load(path))

    def __len__(self) -> int:
        return len(self.sorted_scores)


class QuantileNormalizer:
    """Manages background distributions for normalization across layers.

    Looks up pre-computed distributions by key
    (e.g. ``"alphagenome_chromatin_accessibility"``) and normalises raw
    scores against them.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = str(Path.home() / ".chorus" / "backgrounds")
        self.cache_dir = Path(cache_dir)
        self._distributions: dict[str, BackgroundDistribution] = {}

    # ------------------------------------------------------------------
    # Look-ups
    # ------------------------------------------------------------------

    def has_background(self, key: str) -> bool:
        """Check if a background distribution exists (in memory or on disk)."""
        if key in self._distributions:
            return True
        return (self.cache_dir / f"{key}.npy").exists()

    def get_background(self, key: str) -> BackgroundDistribution | None:
        """Get a background distribution, loading from cache if needed."""
        if key not in self._distributions:
            bg_path = self.cache_dir / f"{key}.npy"
            if bg_path.exists():
                self._distributions[key] = BackgroundDistribution.load(str(bg_path))
        return self._distributions.get(key)

    def set_background(
        self,
        key: str,
        distribution: BackgroundDistribution,
        persist: bool = True,
    ) -> None:
        """Store a background distribution, optionally saving to disk."""
        self._distributions[key] = distribution
        if persist:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            distribution.save(str(self.cache_dir / f"{key}.npy"))

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(
        self, key: str, raw_score: float, signed: bool = True,
    ) -> float | None:
        """Normalise a raw score.  Returns ``None`` when no background exists."""
        bg = self.get_background(key)
        if bg is None:
            return None
        return bg.raw_to_quantile(raw_score, signed=signed)

    def summary(self) -> dict:
        """Return summary stats for all loaded backgrounds.

        Returns a dict keyed by background name with n_scores, median, p95, p99.
        """
        result = {}
        for key, dist in self._distributions.items():
            scores = dist.sorted_scores
            result[key] = {
                "n_scores": len(scores),
                "median": float(np.median(scores)),
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
            }
        return result

    # ------------------------------------------------------------------
    # Baseline (activity percentile) normalization
    # ------------------------------------------------------------------

    def normalize_baseline(
        self, oracle_name: str, layer: str, raw_signal: float,
    ) -> float | None:
        """Map a raw signal level to its genome-wide activity percentile [0, 1].

        Uses baseline background distributions built from ~25K genomic
        positions.  Returns ``None`` when no baseline background exists.
        """
        key = self.baseline_key(oracle_name, layer)
        bg = self.get_background(key)
        if bg is None:
            return None
        return bg.raw_to_quantile(raw_signal, signed=False)

    def normalize_baseline_batch(
        self, oracle_name: str, layer: str, raw_signals: np.ndarray,
    ) -> np.ndarray | None:
        """Batch version of :meth:`normalize_baseline`.

        Returns an array of [0, 1] percentiles, or ``None`` when no
        baseline background exists.
        """
        key = self.baseline_key(oracle_name, layer)
        bg = self.get_background(key)
        if bg is None:
            return None
        return bg.raw_to_quantile_batch(raw_signals, signed=False)

    def has_baseline(self, oracle_name: str, layer: str) -> bool:
        """Check if a baseline distribution exists for this oracle/layer."""
        return self.has_background(self.baseline_key(oracle_name, layer))

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def background_key(oracle_name: str, layer: str) -> str:
        """Generate a standard cache key for a variant effect background."""
        return f"{oracle_name}_{layer}"

    @staticmethod
    def baseline_key(oracle_name: str, layer: str) -> str:
        """Generate a standard cache key for a baseline signal background."""
        return f"{oracle_name}_{layer}_baseline"

    @staticmethod
    def perbin_key(oracle_name: str, layer: str) -> str:
        """Generate a cache key for a per-bin baseline distribution."""
        return f"{oracle_name}_{layer}_perbin"

    # ------------------------------------------------------------------
    # Per-bin (visualization) normalization
    # ------------------------------------------------------------------

    def normalize_perbin_batch(
        self, oracle_name: str, layer: str, raw_values: np.ndarray,
    ) -> np.ndarray | None:
        """Map per-bin values to genome-wide percentiles [0, 1].

        Uses per-bin baseline distributions built from individual bin values
        at ~25K SCREEN cCRE positions.  Unlike :meth:`normalize_baseline`
        (which uses window-sum statistics for scoring), this method compares
        each bin value against the per-bin distribution — correct for
        visualization and heatmaps.

        Returns ``None`` when no per-bin background exists.
        """
        key = self.perbin_key(oracle_name, layer)
        bg = self.get_background(key)
        if bg is None:
            return None
        return bg.raw_to_quantile_batch(raw_values, signed=False)

    def has_perbin(self, oracle_name: str, layer: str) -> bool:
        """Check if a per-bin baseline distribution exists."""
        return self.has_background(self.perbin_key(oracle_name, layer))


# ======================================================================
# Per-track normalization
# ======================================================================

class PerTrackNormalizer:
    """Per-track quantile normalization using compact CDF matrices.

    Stores three ``(n_tracks, n_points)`` matrices in a single ``.npz``
    file per oracle.  Each row is a sorted CDF for one track, enabling
    track-specific percentile lookups via binary search.

    CDF types:

    - **effect_cdfs** — sorted ``|effect|`` scores from background
      variants.  Used by :meth:`effect_percentile` to map a raw variant
      effect score to [0, 1] (unsigned layers) or [-1, 1] (signed layers).

    - **summary_cdfs** — sorted window-sum (or exon-mean) signal levels
      from baseline genomic positions.  Used by :meth:`activity_percentile`
      to map a raw signal level to its genome-wide percentile [0, 1].

    - **perbin_cdfs** — sorted individual bin values from baseline
      positions.  Used by :meth:`perbin_percentile_batch` for IGV
      visualization normalization.

    Parameters
    ----------
    cache_dir : str or None
        Directory for ``.npz`` files.  Defaults to ``~/.chorus/backgrounds/``.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = str(Path.home() / ".chorus" / "backgrounds")
        self.cache_dir = Path(cache_dir)
        # Keyed by oracle name
        self._loaded: dict[str, dict] = {}  # oracle -> {track_ids, effect_cdfs, ...}

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------

    @staticmethod
    def npz_filename(oracle_name: str) -> str:
        """Standard filename for a per-track NPZ file."""
        return f"{oracle_name}_pertrack.npz"

    def npz_path(self, oracle_name: str) -> Path:
        return self.cache_dir / self.npz_filename(oracle_name)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def has_oracle(self, oracle_name: str) -> bool:
        """Check if per-track CDFs exist for an oracle (in memory or on disk)."""
        if oracle_name in self._loaded:
            return True
        return self.npz_path(oracle_name).exists()

    def _ensure_loaded(self, oracle_name: str) -> dict | None:
        """Lazy-load the NPZ for *oracle_name*.  Returns the data dict or None."""
        if oracle_name in self._loaded:
            return self._loaded[oracle_name]
        path = self.npz_path(oracle_name)
        if not path.exists():
            return None
        data = np.load(str(path), allow_pickle=False)
        entry = {
            "track_ids": [str(x) for x in data["track_ids"]],
            "track_index": {},  # built below
        }
        # Build track_id → row index map
        for i, tid in enumerate(entry["track_ids"]):
            entry["track_index"][tid] = i
        # Load CDF matrices (may be absent for some oracles)
        for key in ("effect_cdfs", "summary_cdfs", "perbin_cdfs"):
            entry[key] = data[key] if key in data else None
        # Load per-track sample counts (actual number of background
        # samples before CDF compaction — used for correct percentile
        # calculation instead of dividing by the CDF length).
        for key in ("effect_counts", "summary_counts", "perbin_counts"):
            entry[key] = data[key].astype(np.int64) if key in data else None
        # Load signed flags
        if "signed_flags" in data:
            entry["signed_flags"] = data["signed_flags"].astype(bool)
        else:
            entry["signed_flags"] = None
        self._loaded[oracle_name] = entry
        n_tracks = len(entry["track_ids"])
        cdfs_present = [k for k in ("effect_cdfs", "summary_cdfs", "perbin_cdfs") if entry[k] is not None]
        logger.info(
            "Loaded per-track CDFs for '%s': %d tracks, CDFs: %s",
            oracle_name, n_tracks, ", ".join(cdfs_present),
        )
        return entry

    def n_tracks(self, oracle_name: str) -> int:
        """Number of tracks for an oracle."""
        entry = self._ensure_loaded(oracle_name)
        return len(entry["track_ids"]) if entry else 0

    def track_ids(self, oracle_name: str) -> list[str]:
        """List of track IDs for an oracle."""
        entry = self._ensure_loaded(oracle_name)
        return entry["track_ids"] if entry else []

    # ------------------------------------------------------------------
    # Core percentile lookups
    # ------------------------------------------------------------------

    def _get_denominator(self, entry: dict, cdf_key: str, idx: int) -> int:
        """Get the correct denominator for percentile calculation.

        When sample counts are stored AND smaller than the CDF width,
        use the actual sample count to avoid artificial compression from
        padding short CDFs.  When the sample count is larger (reservoir
        sampling compacted many samples into the CDF), use the CDF width
        since the CDF is a proper subsample.
        """
        cdf_width = entry[cdf_key].shape[1]
        counts_key = cdf_key.replace("_cdfs", "_counts")
        counts = entry.get(counts_key)
        if counts is not None and 0 < counts[idx] < cdf_width:
            return int(counts[idx])
        return cdf_width

    def _has_samples(self, entry: dict, cdf_key: str, idx: int) -> bool:
        """True iff the track has at least one background sample stored.

        Tracks that failed to build (e.g. an ENCODE download that timed out
        during the ChromBPNet background build — see
        ``audits/2026-04-16_application_and_normalization_audit.md``
        finding #1) can land in the committed NPZ with ``counts[idx] == 0``
        and a zero-filled CDF row. Without this guard, every lookup returns
        ``rank / cdf_width`` where ``rank`` collapses to the top of the row,
        silently producing ``quantile = 1.0`` for every raw_score including 0.
        Treat those tracks as "no background available" and let callers
        render them as "—" instead of a false 100th-percentile.
        """
        counts_key = cdf_key.replace("_cdfs", "_counts")
        counts = entry.get(counts_key)
        if counts is None:
            return True  # pre-count-tracking NPZs assumed valid
        return bool(int(counts[idx]) > 0)

    def _lookup(
        self,
        oracle_name: str,
        track_id: str,
        cdf_key: str,
        raw_value: float,
        signed: bool = False,
    ) -> float | None:
        """Binary-search a single value against one track's CDF row.

        Returns percentile in [0, 1] (unsigned) or [-1, 1] (signed),
        or ``None`` if the CDF is missing / the track has no background
        samples stored.
        """
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return None
        cdf_matrix = entry.get(cdf_key)
        if cdf_matrix is None:
            return None
        idx = entry["track_index"].get(track_id)
        if idx is None:
            return None
        if not self._has_samples(entry, cdf_key, idx):
            return None
        row = cdf_matrix[idx]
        rank = np.searchsorted(row, raw_value, side="right")
        denom = self._get_denominator(entry, cdf_key, idx)
        quantile = min(rank / denom, 1.0)
        if signed:
            return float(2.0 * quantile - 1.0)
        return float(quantile)

    def _lookup_batch(
        self,
        oracle_name: str,
        track_id: str,
        cdf_key: str,
        raw_values: np.ndarray,
        signed: bool = False,
    ) -> np.ndarray | None:
        """Vectorised binary-search for an array of values."""
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return None
        cdf_matrix = entry.get(cdf_key)
        if cdf_matrix is None:
            return None
        idx = entry["track_index"].get(track_id)
        if idx is None:
            return None
        if not self._has_samples(entry, cdf_key, idx):
            return None
        row = cdf_matrix[idx]
        ranks = np.searchsorted(row, raw_values, side="right")
        denom = self._get_denominator(entry, cdf_key, idx)
        quantiles = np.minimum(ranks.astype(np.float64) / denom, 1.0)
        if signed:
            return 2.0 * quantiles - 1.0
        return quantiles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def effect_percentile(
        self,
        oracle_name: str,
        track_id: str,
        raw_score: float,
        signed: bool = False,
    ) -> float | None:
        """Map a raw variant effect score to its per-track percentile.

        For **unsigned** layers (chromatin, ChIP, CAGE, splicing):
            pass ``signed=False`` and ``abs(raw_score)`` — returns [0, 1].
        For **signed** layers (MPRA, expression, Sei):
            pass ``signed=True`` and the raw score — returns [-1, 1].
        """
        return self._lookup(oracle_name, track_id, "effect_cdfs", raw_score, signed=signed)

    def activity_percentile(
        self,
        oracle_name: str,
        track_id: str,
        raw_signal: float,
    ) -> float | None:
        """Map a raw window-sum signal to its genome-wide activity percentile [0, 1]."""
        return self._lookup(oracle_name, track_id, "summary_cdfs", raw_signal, signed=False)

    def perbin_percentile_batch(
        self,
        oracle_name: str,
        track_id: str,
        raw_values: np.ndarray,
    ) -> np.ndarray | None:
        """Map per-bin values to genome-wide percentiles [0, 1] for visualization."""
        return self._lookup_batch(oracle_name, track_id, "perbin_cdfs", raw_values, signed=False)

    def perbin_floor_rescale_batch(
        self,
        oracle_name: str,
        track_id: str,
        raw_values: np.ndarray,
        floor_pctile: float = 0.95,
        peak_pctile: float = 0.99,
        max_value: float = 1.5,
    ) -> np.ndarray | None:
        """Rescale raw bin values using CDF-derived noise floor and peak threshold.

        Maps raw values into a [0, max_value] display range using:

        .. math::

            display = \\frac{raw - cdf[floor\\_pctile]}{cdf[peak\\_pctile] - cdf[floor\\_pctile]}

        with values below ``floor_pctile`` clipped to 0 and values above
        ``max_value`` clipped to ``max_value``.

        This **preserves raw peak shape** (linear scaling above floor)
        while making tracks **comparable across cell types** (1.0 always
        means "at the genome-wide top 1% threshold for this track").

        Default ``floor_pctile=0.95``, ``peak_pctile=0.99`` works well
        for most genomic tracks.  Sharp signals (CAGE, TF) and broad
        signals (histone marks) both render correctly because the
        transform is linear in raw space — peak shapes are preserved.
        """
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return None
        cdf_matrix = entry.get("perbin_cdfs")
        if cdf_matrix is None:
            return None
        idx = entry["track_index"].get(track_id)
        if idx is None:
            return None
        cdf = cdf_matrix[idx]
        n = len(cdf)
        floor = float(cdf[min(int(floor_pctile * n), n - 1)])
        peak = float(cdf[min(int(peak_pctile * n), n - 1)])
        denom = max(peak - floor, 1e-9)
        out = (raw_values.astype(np.float64) - floor) / denom
        return np.clip(out, 0.0, max_value)

    def is_signed(self, oracle_name: str, track_id: str) -> bool:
        """Whether a track uses signed normalization (default: False)."""
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return False
        flags = entry.get("signed_flags")
        if flags is None:
            return False
        idx = entry["track_index"].get(track_id)
        if idx is None:
            return False
        return bool(flags[idx])

    # ------------------------------------------------------------------
    # Building / saving
    # ------------------------------------------------------------------

    @staticmethod
    def build_and_save(
        oracle_name: str,
        track_ids: list[str],
        effect_cdfs: np.ndarray | None = None,
        summary_cdfs: np.ndarray | None = None,
        perbin_cdfs: np.ndarray | None = None,
        signed_flags: np.ndarray | None = None,
        effect_counts: np.ndarray | None = None,
        summary_counts: np.ndarray | None = None,
        perbin_counts: np.ndarray | None = None,
        cache_dir: str | None = None,
        n_points: int = 10_000,
    ) -> Path:
        """Save per-track CDF matrices to a compressed ``.npz`` file.

        Each CDF matrix has shape ``(n_tracks, n_points)`` where each row
        is a sorted array of values sampled from the background distribution.

        Args:
            oracle_name: Oracle identifier (e.g. ``"enformer"``).
            track_ids: List of track identifiers (length = n_tracks).
            effect_cdfs: ``(n_tracks, n_points)`` sorted |effect| values.
            summary_cdfs: ``(n_tracks, n_points)`` sorted window-sum values.
            perbin_cdfs: ``(n_tracks, n_points)`` sorted per-bin values.
            signed_flags: ``(n_tracks,)`` bool — True for signed layers.
            effect_counts: ``(n_tracks,)`` int — actual sample count per track
                for effect CDFs (used for correct percentile calculation).
            summary_counts: ``(n_tracks,)`` int — actual sample count per track
                for summary CDFs.
            perbin_counts: ``(n_tracks,)`` int — actual sample count per track
                for per-bin CDFs.
            cache_dir: Output directory.  Defaults to ``~/.chorus/backgrounds/``.
            n_points: Number of CDF points per track (default 10,000).

        Returns:
            Path to the saved ``.npz`` file.
        """
        if cache_dir is None:
            cache_dir = str(Path.home() / ".chorus" / "backgrounds")
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_tracks = len(track_ids)
        arrays: dict[str, np.ndarray] = {
            "track_ids": np.array(track_ids, dtype="U"),
        }

        for name, matrix in [
            ("effect_cdfs", effect_cdfs),
            ("summary_cdfs", summary_cdfs),
            ("perbin_cdfs", perbin_cdfs),
        ]:
            if matrix is not None:
                assert matrix.shape[0] == n_tracks, (
                    f"{name} has {matrix.shape[0]} rows but {n_tracks} track_ids"
                )
                # Compact: subsample columns to n_points if needed
                if matrix.shape[1] > n_points:
                    indices = np.linspace(0, matrix.shape[1] - 1, n_points, dtype=int)
                    matrix = matrix[:, indices]
                arrays[name] = matrix.astype(np.float32)

        if signed_flags is not None:
            arrays["signed_flags"] = np.array(signed_flags, dtype=bool)

        # Store actual sample counts for correct percentile calculation
        for name, counts in [
            ("effect_counts", effect_counts),
            ("summary_counts", summary_counts),
            ("perbin_counts", perbin_counts),
        ]:
            if counts is not None:
                arrays[name] = np.array(counts, dtype=np.int64)

        path = out_dir / PerTrackNormalizer.npz_filename(oracle_name)
        np.savez_compressed(str(path), **arrays)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved per-track CDFs for '%s': %d tracks, %.1f MB → %s",
            oracle_name, n_tracks, size_mb, path,
        )
        return path

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self, oracle_name: str | None = None) -> dict:
        """Return summary stats for loaded oracles' CDFs.

        When *oracle_name* is given, returns a dict describing that
        oracle's CDFs. When omitted, returns a dict keyed by every
        already-loaded oracle so callers can iterate without knowing the
        oracle name up front — matches the signature shape of
        :meth:`QuantileNormalizer.summary`.
        """
        if oracle_name is None:
            return {name: self._summary_one(name) for name in sorted(self._loaded)}
        return self._summary_one(oracle_name)

    def _summary_one(self, oracle_name: str) -> dict:
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return {}
        result = {
            "n_tracks": len(entry["track_ids"]),
            "cdfs": {},
        }
        for key in ("effect_cdfs", "summary_cdfs", "perbin_cdfs"):
            m = entry.get(key)
            if m is not None:
                result["cdfs"][key] = {
                    "shape": list(m.shape),
                    "dtype": str(m.dtype),
                    "nbytes_mb": round(m.nbytes / (1024 * 1024), 1),
                }
        return result


def get_pertrack_normalizer(
    oracle_name: str,
    cache_dir: str | None = None,
) -> PerTrackNormalizer | None:
    """Auto-discover and load per-track CDFs for an oracle.

    Looks for ``{oracle_name}_pertrack.npz`` in *cache_dir* (default
    ``~/.chorus/backgrounds/``).  If not found locally, attempts to
    download from the ``lucapinello/chorus-backgrounds`` HuggingFace
    dataset.

    Returns a ready-to-use :class:`PerTrackNormalizer`, or ``None`` if
    no per-track CDFs exist for *oracle_name*.
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".chorus" / "backgrounds")
    norm = PerTrackNormalizer(cache_dir=cache_dir)
    if norm.has_oracle(oracle_name):
        norm._ensure_loaded(oracle_name)
        return norm

    # Auto-download from HuggingFace
    n = download_pertrack_backgrounds(oracle_name, cache_dir=cache_dir)
    if n > 0 and norm.has_oracle(oracle_name):
        norm._ensure_loaded(oracle_name)
        return norm

    return None


_HF_REPO = "lucapinello/chorus-backgrounds"


def download_pertrack_backgrounds(
    oracle_name: str,
    cache_dir: str | None = None,
) -> int:
    """Download per-track background CDFs from HuggingFace for an oracle.

    Fetches ``{oracle_name}_pertrack.npz`` from the
    ``lucapinello/chorus-backgrounds`` dataset repo and saves it to
    *cache_dir* (default ``~/.chorus/backgrounds/``).

    Returns 1 if downloaded, 0 if already exists or not available.
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".chorus" / "backgrounds")
    bg_dir = Path(cache_dir)
    bg_dir.mkdir(parents=True, exist_ok=True)

    fname = PerTrackNormalizer.npz_filename(oracle_name)
    local_path = bg_dir / fname
    if local_path.exists():
        return 0  # already cached

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — cannot download backgrounds. "
            "Install it with: pip install huggingface_hub"
        )
        return 0

    try:
        logger.info("Downloading %s from HuggingFace ...", fname)
        hf_hub_download(
            _HF_REPO,
            filename=fname,
            repo_type="dataset",
            local_dir=str(bg_dir),
        )
        logger.info("Downloaded %s", fname)
        return 1
    except Exception as exc:
        logger.warning("Failed to download %s: %s", fname, exc)
        return 0


def download_backgrounds(
    oracle_name: str,
    cache_dir: str | None = None,
) -> int:
    """Download pre-computed backgrounds from HuggingFace for an oracle.

    Downloads both per-track ``.npz`` files (preferred) and legacy
    per-layer ``.npy`` files from the ``lucapinello/chorus-backgrounds``
    dataset repo.

    Returns the number of files downloaded.  Skips files that already
    exist locally.
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".chorus" / "backgrounds")
    bg_dir = Path(cache_dir)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # Try per-track NPZ first
    downloaded = download_pertrack_backgrounds(oracle_name, cache_dir=cache_dir)

    # Also try legacy per-layer NPY files
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return downloaded

    api = HfApi()
    try:
        files = api.list_repo_files(_HF_REPO, repo_type="dataset")
    except Exception as exc:
        logger.warning("Failed to list backgrounds on HuggingFace: %s", exc)
        return downloaded

    target_files = [
        f for f in files
        if f.startswith(f"{oracle_name}_") and (f.endswith(".npy") or f.endswith(".npz"))
    ]

    for fname in target_files:
        local_path = bg_dir / fname
        if local_path.exists():
            continue
        try:
            logger.info("Downloading %s ...", fname)
            hf_hub_download(
                _HF_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(bg_dir),
            )
            downloaded += 1
        except Exception as exc:
            logger.warning("Failed to download %s: %s", fname, exc)

    if downloaded:
        logger.info("Downloaded %d background files for '%s'", downloaded, oracle_name)
    return downloaded


def get_normalizer(
    oracle_name: str, cache_dir: str | None = None,
) -> "PerTrackNormalizer | QuantileNormalizer | None":
    """Auto-discover and load pre-computed backgrounds for an oracle.

    **Preferred path**: if ``{oracle_name}_pertrack.npz`` exists (or can be
    downloaded from the public ``lucapinello/chorus-backgrounds`` HuggingFace
    dataset), returns a :class:`PerTrackNormalizer` — the newer per-track
    CDF format that supports effect / activity / perbin percentiles.

    **Legacy fallback**: if only old-format ``{oracle_name}_*.npy``
    per-layer backgrounds are present, returns a
    :class:`QuantileNormalizer` loaded with them.

    Returns ``None`` only when neither format is available.

    Both return types are accepted by
    :meth:`chorus.core.result.OraclePrediction.to_percentile`, so callers
    don't need to branch.
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".chorus" / "backgrounds")
    bg_dir = Path(cache_dir)

    # ── 1. Try the per-track NPZ first (new format, auto-downloaded from HF).
    try:
        pertrack = get_pertrack_normalizer(oracle_name, cache_dir=cache_dir)
        if pertrack is not None and pertrack.n_tracks(oracle_name) > 0:
            return pertrack
    except Exception as exc:
        logger.debug(
            "Per-track normalizer unavailable for %s (%s); falling back to legacy .npy scan.",
            oracle_name, exc,
        )

    # ── 2. Legacy per-layer .npy scan.
    if bg_dir.is_dir():
        matched = sorted(bg_dir.glob(f"{oracle_name}_*.npy"))
    else:
        matched = []

    if not matched:
        n = download_backgrounds(oracle_name, cache_dir=cache_dir)
        if n > 0 and bg_dir.is_dir():
            matched = sorted(bg_dir.glob(f"{oracle_name}_*.npy"))

    if not matched:
        return None

    normalizer = QuantileNormalizer(cache_dir=cache_dir)
    for path in matched:
        key = path.stem  # drop .npy
        normalizer.get_background(key)  # lazy load
    logger.info(
        "Auto-loaded %d legacy-format backgrounds for '%s' "
        "(new per-track format not yet available for this oracle)",
        len(matched), oracle_name,
    )
    return normalizer
