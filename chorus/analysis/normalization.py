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

Background distributions are cached under ``CHORUS_BACKGROUNDS_DIR`` for reuse
(the installation directory by default; see ``chorus/core/globals.py``).
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from chorus.analysis.background_sampling import (
    cdf_grid_violations,
    thinning_violations,
    yield_violations,
)
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

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
            cache_dir = str(CHORUS_BACKGROUNDS_DIR)
        self.cache_dir = Path(cache_dir)
        self._distributions: dict[str, BackgroundDistribution] = {}

    # ------------------------------------------------------------------
    # Look-ups
    # ------------------------------------------------------------------

    def has_background(self, key: str) -> bool:
        """Check if a background distribution exists (in memory or on disk).

        ``key`` is the full ``{oracle}_{layer}`` filename stem (e.g.
        ``"alphagenome_chromatin_accessibility"``). See
        :meth:`has_oracle_quantiles` for an oracle-level check that
        glob-matches any layer.
        """
        if key in self._distributions:
            return True
        return (self.cache_dir / f"{key}.npy").exists()

    def has_oracle_quantiles(self, oracle_name: str) -> bool:
        """True if any per-layer Quantile NPY for *oracle_name* exists on disk.

        Mirrors :meth:`PerTrackNormalizer.has_oracle` at the oracle level
        so callers can ask "is something usable on disk?" without picking
        which normalizer flavor they want. See module-level
        :func:`is_ready_for_oracle` for the union over both flavors.
        """
        return any(self.cache_dir.glob(f"{oracle_name}_*.npy"))

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
        Directory for ``.npz`` files.  Defaults to ``<data-dir>/backgrounds/``.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = str(CHORUS_BACKGROUNDS_DIR)
        self.cache_dir = Path(cache_dir)
        # Keyed by oracle name
        self._loaded: dict[str, dict] = {}  # oracle -> {track_ids, effect_cdfs, ...}
        # Oracles already warned about a superseded artefact, so the message is
        # emitted once rather than on every percentile lookup.
        self._warned_superseded: set[str] = set()

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

    # Oracles that share an identical CDF with another oracle.
    #
    # This alias makes one backend's percentiles depend on another backend's null, so the premise —
    # "the two produce the same predictions" — is a correctness requirement, not a convenience note.
    # It is verified by `tests/test_alphagenome_backends_equivalence.py`, which compares one track per
    # output type at correlation > 0.99 and peak-relative < 8%. Keep that test covering every head: it
    # previously compared three DNASE tracks and, while it passed, `alphagenome_pt` was returning raw
    # logits for 738 splice tracks against this sigmoid-space null (correlation 0.20). Fixed in #240.
    _CDF_ALIASES: dict[str, str] = {"alphagenome_pt": "alphagenome"}

    #: CDF keys that are not oracles, and the fold each one's artefact must be built on.
    #:
    #: Cherimoya ships two nulls because a percentile is a rank against a null and the folds
    #: are not interchangeable -- on DNASE:ENCSR149XIL the five fold peaks span 7.65 to 15.47
    #: (2.02x), and any single fold lands between 0.69x and 1.39x of the ensemble. fold 0 is
    #: the default and takes the plain filename; the ensemble is a separate key.
    #: ``CherimoyaOracle.normalization_key`` is how a caller picks.
    _EXPECTED_FOLD: dict[str, object] = {"cherimoya": 0, "cherimoya_ensemble": "ensemble"}

    def _ensure_loaded(self, oracle_name: str) -> dict | None:
        """Lazy-load the NPZ for *oracle_name*.  Returns the data dict or None."""
        if oracle_name in self._loaded:
            return self._loaded[oracle_name]
        path = self.npz_path(oracle_name)
        if not path.exists():
            # Fall back to a known alias (e.g. alphagenome_pt → alphagenome)
            alias = self._CDF_ALIASES.get(oracle_name)
            if alias:
                return self._ensure_loaded(alias)
            return None
        # Refuse an artefact built on a different fold than the key promises.
        #
        # This is the guard that makes a fold mismatch impossible to ship silently. The
        # failure it prevents is not a crash: it is a percentile that looks entirely normal
        # and is ranked against a different model's distribution. It is reachable three ways
        # -- a cache from before fold 0 became the default (that file holds ensemble CDFs
        # under the plain name), a hand-copied or mirrored artefact, or a caller that asks
        # for the wrong key. Checked here rather than at the call sites because there is one
        # loader and about twenty callers.
        expected = self._EXPECTED_FOLD.get(oracle_name)
        if expected is not None:
            got = _stamped_fold(path)
            if got is not None and got != expected:
                raise BackgroundFoldMismatch(
                    f"{path.name} was built on CATv1 fold {got!r} but the "
                    f"'{oracle_name}' CDF key requires fold {expected!r}. A percentile is a "
                    f"rank against a null, so using this file would rank predictions from "
                    f"one model against another model's distribution -- the folds disagree "
                    f"by ~2x on the same sequence. If this is a cache from before fold 0 "
                    f"became the default, delete it and let chorus re-download: "
                    f"rm {path}"
                )

        # Say something when the artefact predates the retention rebuild. The
        # downloader now refuses to keep a superseded cache, but a user who
        # obtained the file another way -- a copy, a pinned mirror, a restored
        # backup -- would otherwise get silently wrong ceilings. Warned once per
        # oracle per process, not per call.
        if oracle_name not in self._warned_superseded:
            why = _cached_artefact_is_superseded(path)
            if why:
                self._warned_superseded.add(oracle_name)
                logger.warning(
                    "%s background at %s is superseded (%s). Effect ceilings are "
                    "draws from a uniform subsample, so strong effects will stay "
                    "clamped at percentile 1.0 with an inflated exceedance ratio. "
                    "Delete the file and re-run to refetch.",
                    oracle_name, path, why,
                )

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
        # As-built provenance (#124). Stored as a 1-element unicode array, so it
        # reads under allow_pickle=False.
        entry["build_config"] = None
        if "build_config" in data:
            entry["build_config"] = self._read_build_config(
                data["build_config"], oracle_name,
            )
        self._require_ranking_genome(oracle_name, path, entry["build_config"])
        self._loaded[oracle_name] = entry
        n_tracks = len(entry["track_ids"])
        cdfs_present = [k for k in ("effect_cdfs", "summary_cdfs", "perbin_cdfs") if entry[k] is not None]
        logger.info(
            "Loaded per-track CDFs for '%s': %d tracks, CDFs: %s",
            oracle_name, n_tracks, ", ".join(cdfs_present),
        )
        self._warn_on_geometry_mismatch(oracle_name, entry)
        return entry

    @staticmethod
    def _require_ranking_genome(oracle_name: str, path: Path, config: dict | None) -> None:
        """Refuse an artefact whose declared assembly is not the one chorus ranks against.

        Checked here, after ``build_config`` is parsed, rather than before ``np.load`` like
        the fold guard: the genome is in the config the loader already reads, so this costs
        no second open. Raised before the entry reaches ``self._loaded``, so a refused
        artefact is not cached as usable.

        An **unstamped** artefact is silent, matching :meth:`_warn_on_geometry_mismatch`:
        every background built before the provenance work has no ``genome`` key, and it
        makes no claim to contradict. Refusing those would break working installs to enforce
        a metadata convention. What is *not* tolerated is a stated assembly that disagrees —
        that is a claim, and it is wrong.
        """
        if not isinstance(config, dict):
            return
        declared = config.get("genome")
        if declared is None or declared == RANKING_GENOME:
            return
        raise BackgroundGenomeMismatch(
            f"{path.name} declares genome {declared!r}, but chorus ranks against "
            f"{RANKING_GENOME!r} (build_config.schema_version "
            f"{config.get('schema_version')!r}). A percentile is a rank within a reference "
            f"class of {RANKING_GENOME} regions, so this artefact's CDFs describe different "
            f"DNA than the predictions being ranked -- and every coordinate would still "
            f"resolve, so nothing else would notice. If you are developing a "
            f"{declared} oracle, the query-side reference class has to be built for "
            f"{declared} first; see docs/BACKGROUND_NULL_PROTOCOL.md."
        )

    @staticmethod
    def _read_build_config(raw, label: str) -> dict | None:
        """Parse a stored ``build_config``, accepting both shapes it ships in.

        0-d is what ``build_and_save(provenance=...)`` wrote before 2026-08-04;
        1-element is what Cherimoya's builder and the stamp script wrote, and what
        every file on disk actually has. Reading only one of the two silently
        discarded provenance from the other.
        """
        import json as _json

        try:
            arr = np.asarray(raw)
            text = str(arr.item()) if arr.ndim == 0 else str(arr.reshape(-1)[0])
            parsed = _json.loads(text)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError):
            logger.warning(
                "'%s' has a build_config that will not parse; treating it as "
                "absent", label,
            )
            return None
        return parsed if isinstance(parsed, dict) else None

    def provenance(self, oracle_name: str) -> dict | None:
        """The as-built ``build_config`` for *oracle_name*, or None if unstamped.

        Public because a stamp nobody reads is documentation, not an invariant —
        which is what it was until this method existed. See
        :meth:`_warn_on_geometry_mismatch` for the one thing chorus itself does
        with it.
        """
        entry = self._ensure_loaded(oracle_name)
        return (entry or {}).get("build_config")

    def _warn_on_geometry_mismatch(self, oracle_name: str, entry: dict) -> None:
        """Compare the stamped build geometry against what the query will use.

        This is #122's whole class turned into a data-carried check. #122 was
        AlphaGenome histone CHIP tracks whose null was built over 501 bp while the
        query summed 2001 bp — a mismatch invisible in both artefacts, because each
        was internally consistent. Nothing compared them, so it shipped across
        1,075 of 5,168 tracks.

        A warning rather than an exception, deliberately: every background built
        before the provenance work is unstamped, and older stamped files may
        legitimately predate a query-side change. Refusing to load them would break
        working installs to enforce a metadata convention. An unstamped file is
        silent — it makes no claim to contradict.
        """
        config = entry.get("build_config")
        if not isinstance(config, dict):
            return
        try:
            from chorus.analysis.scorers import LAYER_CONFIGS
        except Exception:                                  # pragma: no cover
            return

        expected = {
            "histone_marks": config.get("histone_window_bp"),
            "chromatin_accessibility": config.get("other_window_bp"),
            "tf_binding": config.get("other_window_bp"),
            "tss_activity": config.get("other_window_bp"),
        }
        for layer, built_bp in expected.items():
            spec = LAYER_CONFIGS.get(layer)
            if spec is None or built_bp is None or spec.window_bp is None:
                continue
            if int(spec.window_bp) != int(built_bp):
                logger.warning(
                    "'%s' background for layer %s was BUILT over %d bp but this "
                    "query sums %d bp. Percentiles for that layer compare two "
                    "different statistics (#122/#144). Rebuild the background or "
                    "align LAYER_CONFIGS.",
                    oracle_name, layer, int(built_bp), int(spec.window_bp),
                )

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
        """Denominator for a percentile: always the CDF grid width.

        ``rank`` comes from ``np.searchsorted`` over the stored row, so it
        lives on ``[0, cdf_width]`` and the denominator has to be measured
        against the *same* population — the grid — or the ratio is not a
        quantile at all.

        This used to return ``counts[idx]`` whenever the sample count was
        below the grid width, on the theory that short CDFs are padded and
        only the first ``counts[idx]`` entries are real. They are not
        padded: ``ReservoirSampler.to_cdf_matrix`` **interpolates** a short
        sample onto the full grid (``np.interp(target_q, source_q, arr)``)
        and subsamples a long one (``arr[np.linspace(...)]``), so all
        ``cdf_width`` entries are meaningful quantile estimates either way.

        Mixing the two populations inflated every percentile by
        ``cdf_width / counts[idx]`` and clamped the top
        ``1 - counts[idx]/cdf_width`` of the range to 1.0. It hit the effect
        CDF of every oracle except ChromBPNet and Cherimoya (whose 18,672
        samples exceed the grid), and it was severe for AlphaGenome, whose
        ``effect_counts`` are ~1,700-1,900 against a 10,000-point grid: a
        value at the *median* of its own background reported the 100th
        percentile, and 80.9 % of the range pinned to 1.0.

        ``counts`` remains the right thing to consult for "was anything
        sampled at all" — see :meth:`_has_samples`.
        """
        return int(entry[cdf_key].shape[1])

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
        idx = self._resolve_row(track_id, entry)
        if idx is None:
            return None
        if not self._has_samples(entry, cdf_key, idx):
            return None
        row = cdf_matrix[idx]
        rank = self._rank_with_tie_breaking(row, raw_value, track_id)
        denom = self._get_denominator(entry, cdf_key, idx)
        quantile = min(rank / denom, 1.0)
        if signed:
            return float(2.0 * quantile - 1.0)
        return float(quantile)

    @staticmethod
    def has_tied_quantiles(row: np.ndarray) -> bool:
        """Does this CDF row contain adjacent equal values?

        AlphaGenome precomputes exactly this per track, as
        ``has_duplicate_quantiles = np.any(np.diff(quantiles) == 0, axis=1)``
        (``alphagenome_research/.../calibration/calibration.py:63``), because a
        degenerate track is expected rather than exceptional and is better flagged
        at load time than discovered downstream.
        """
        return bool(np.any(np.diff(np.asarray(row)) == 0))

    @staticmethod
    def _rank_with_tie_breaking(
        row: np.ndarray, raw_value: float, track_id: str,
    ) -> int:
        """Insertion rank, spread uniformly across any run of tied values.

        Plain ``searchsorted`` collapses everything landing inside a tied run to
        one end of it. Where a CDF's upper tail is a long tie — which is what a
        near-degenerate background looks like — effects of 0.05, 0.5 and 5.0 all
        return the same percentile, and the column stops discriminating exactly
        where users need it to.

        AlphaGenome solves this by drawing uniformly between the left and right
        insertion points (``break_quantile_ties=True`` by default,
        ``calibration.py:157-160``). This does the same thing, with one deliberate
        difference: their draw comes from a sequential ``np.random.Generator``, so
        the result depends on call ORDER and varies run to run unless a seed is
        threaded through. Here the draw is derived from a stable hash of
        ``(track_id, raw_value)``, so it is uniform across the tie *and* identical
        for the same query every time — which keeps the reproducibility #127 was
        fixed to obtain.

        What this does and does not buy: it restores *distributional* correctness,
        so percentiles are uniform under the null again. It does not make an
        individual row more informative — the raw effect is still where the
        resolution lives, which is why both are reported.
        """
        arr = np.asarray(row)
        lo = int(np.searchsorted(arr, raw_value, side="left"))
        hi = int(np.searchsorted(arr, raw_value, side="right"))
        # Only a genuine RUN of tied entries is broken. A single exact match
        # (hi - lo == 1) keeps the previous side="right" answer exactly.
        #
        # This narrows AlphaGenome's rule deliberately: theirs draws from
        # [lo, hi] inclusive, so it randomises on a single exact match too. That is
        # defensible for them — an exact hit does sit at a quantile boundary, and
        # the randomised-quantile convention spreads it. But in chorus a value
        # landing exactly on a stored quantile is common (a 10,000-point grid over
        # a few thousand samples repeats values), and randomising all of those
        # perturbs every previously-exact percentile for no gain: it does not touch
        # the saturation problem, which is caused by LONG runs. Measured: the wider
        # rule broke 13 existing percentile tests; this one breaks none.
        if hi - lo <= 1:
            return hi
        digest = hashlib.blake2b(
            f"{track_id}\x00{raw_value!r}".encode(), digest_size=8,
        ).digest()
        return lo + int.from_bytes(digest, "big") % (hi - lo + 1)

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
        idx = self._resolve_row(track_id, entry)
        if idx is None:
            return None
        if not self._has_samples(entry, cdf_key, idx):
            return None
        row = cdf_matrix[idx]
        # Tie-broken exactly as in _lookup. Vectorised where there are no ties —
        # the common case — and per-value only inside a tied run, so the batch and
        # single paths return the SAME percentile for the same input. Letting them
        # diverge here would be another instance of #144, in the code that computes
        # the number rather than the code that builds the null.
        arr = np.asarray(row)
        values = np.asarray(raw_values)
        lo = np.searchsorted(arr, values, side="left")
        hi = np.searchsorted(arr, values, side="right")
        ranks = hi.astype(np.int64)
        tied = np.nonzero(hi - lo > 1)[0]
        for j in tied:
            ranks[j] = self._rank_with_tie_breaking(arr, float(values[j]), track_id)
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

    def effect_null_support(
        self,
        oracle_name: str,
        track_id: str,
    ) -> tuple[float, float] | None:
        """The (most negative, most positive) effect the null actually contains.

        These are the first and last entries of the track's ``effect_cdfs`` row —
        the smallest and largest of the sampled background effects. They are
        already in the shipped artefacts, so nothing here needs a rebuild.

        Unsigned rows start at 0.0 (the row holds ``abs`` effects); signed rows —
        every Sei and LegNet track, 12.9% of AlphaGenome's and 20.3% of Borzoi's —
        run negative, so **both** ends are live and a caller must not assume the
        maximum is the only bound that can be crossed.
        """
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return None
        cdf_matrix = entry.get("effect_cdfs")
        if cdf_matrix is None:
            return None
        idx = self._resolve_row(track_id, entry)
        if idx is None or not self._has_samples(entry, "effect_cdfs", idx):
            return None
        row = np.asarray(cdf_matrix[idx])
        if row.size == 0:
            return None
        return (float(row[0]), float(row[-1]))

    def effect_exceedance(
        self,
        oracle_name: str,
        track_id: str,
        raw_score: float,
        signed: bool = False,
    ) -> float | None:
        """How far past the null's most extreme sample an effect lies, as a ratio.

        ``effect_percentile`` is ``min(rank / denominator, 1.0)``, so it reaches
        exactly 1.0 the moment the effect reaches the largest of the ~10–12k
        sampled background effects, and stays there however much further it goes.
        At rs12740374, ``CHIP:HepG2:CEBPA:+`` scores +1.865 against a null maximum
        of 1.682: it pins at 1.0, indistinguishable from an effect ten times
        larger. That is a real loss of information, and it is not fixable by
        resampling — the ceiling is a single extreme order statistic, so its
        position carries large sampling variance. Measured across Enformer's 12
        ``tf_binding`` tracks after re-anchoring, 12 of 12 got a *wider* p99
        (+63%) while 11 of 12 reported a *lower* maximum.

        So report the distance instead. Returns ``|raw| / |bound|`` where *bound*
        is the end of the support the effect crossed, or ``None`` when the effect
        is inside the support and the percentile is already exact. A returned
        1.11 means "11% beyond the most extreme background effect sampled for
        this track".

        This is deliberately **not** an extrapolated percentile. Fitting a
        generalised Pareto tail to Enformer's TF nulls gives shape c = −0.190,
        i.e. a *bounded* tail whose endpoint (4.245) sits above the empirical
        maximum (2.956) but still below the observed effect (4.372) — so the
        fitted model calls the measurement impossible. Forcing an exponential
        tail (c = 0) does extrapolate monotonically and never saturates, but it
        turns a measurement into a modelling assumption and prints it to eight
        decimal places. The ratio is a fact about the sample.
        """
        support = self.effect_null_support(oracle_name, track_id)
        if support is None:
            return None
        lo, hi = support
        value = raw_score if signed else abs(raw_score)
        if value > hi and hi > 0:
            return float(value / hi)
        if signed and value < lo and lo < 0:
            return float(value / lo)
        return None

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
    
    def _find_matching_cdf(self, entry: dict, idx: int, track_id: str) -> np.ndarray | None:
        """Retrieve the CDF array for track at *idx*, falling back through CDF types.

        Tries perbin_cdfs → summary_cdfs → effect_cdfs, returning the first
        valid array whose corresponding ``*_counts`` > 0.  Skipping CDF
        types with zero samples means a track that failed to build at the
        per-bin stage but succeeded at summary still gets the summary CDF
        (rather than landing on the all-zero perbin row, which would
        saturate every display value to ``max_value``).
        """
        for cdf_key in ("perbin_cdfs", "summary_cdfs", "effect_cdfs"):
            cdf_matrix = entry.get(cdf_key)
            if cdf_matrix is None:
                continue

            try:
                cdf = cdf_matrix[idx]
            except (IndexError, TypeError):
                continue

            if cdf is None or len(cdf) == 0:
                continue

            # Skip CDF types where this track has no background samples
            # (counts[idx] == 0 → all-zero CDF row from a failed build).
            if not self._has_samples(entry, cdf_key, idx):
                continue

            logger.debug(f"Using {cdf_key} for '{track_id}'")
            return cdf

        logger.warning(f"No valid CDF found for '{track_id}' (index {idx})")
        return None
    
    def _resolve_row(self, track_id: str, entry: dict) -> int | None:
        """Row index for *track_id*, using the one shared matcher.

        Every path that reads a CDF row — percentile lookups and display
        rescaling alike — must resolve an id the same way, or the two halves
        of a report end up describing different background rows.

        This used to be an inline exact-match-plus-strand-strip duplicated in
        `_lookup` and `_lookup_batch`, while `_match_track_id` (which handles
        strictly more cases, including LegNet's ``"LentiMPRA:HepG2"`` against a
        row keyed ``"HepG2"``) was reached only from the display helpers. The
        result was an oracle whose three shipped CDF rows were unreachable from
        `effect_percentile`, rendering an em dash that looked deliberate. See
        https://github.com/pinellolab/chorus/issues/126 and
        tests/test_normalization_track_ids.py.
        """
        matched = self._match_track_id(track_id, entry["track_index"])
        if matched is None:
            return None
        return entry["track_index"].get(matched)

    def _match_track_id(self, track_id: str, track_index: dict) -> str | None:
        """Find *track_id* in *track_index*, trying common alternative formats.

        Returns the matched key, or None if no match is found.
        """
        if track_id in track_index:
            return track_id

        # Build candidate list from track_id components
        parts = track_id.split(":")
        candidates = [
            track_id.replace(":", "_"),
            track_id.replace("_", ":"),
        ]
        # CHIP strand suffix: BPNet/CHIP track IDs end in ":+" or ":-" but
        # the CDF row is keyed without the strand (e.g. "CHIP:HepG2:CEBPA"
        # not "CHIP:HepG2:CEBPA:+").  Strip the suffix so per-strand
        # predictions still hit the merged CDF.
        if track_id.endswith((":+", ":-")):
            candidates.append(track_id.rsplit(":", 1)[0])
        if len(parts) >= 2:
            candidates.append(parts[-1])  # Last component only

        for candidate in candidates:
            if candidate in track_index:
                logger.debug(f"Track ID matched: '{track_id}' → '{candidate}'")
                return candidate

        logger.warning(f"Track '{track_id}' not found (candidates: {candidates})")
        return None

    def perbin_floor_rescale_batch(
        self,
        oracle_name: str,
        track_id: str,
        raw_values: np.ndarray,
        floor_pctile: float = 0.95,
        peak_pctile: float = 0.99,
        max_value: float = 3.0,
        log_scale: bool = False,
    ) -> np.ndarray | None:
        """Rescale raw bin values using CDF-derived noise floor and peak threshold.

        ``log_scale`` applies ``log1p`` to the values and to both band anchors, for tracks
        whose linear rendering clips. **Whether** a track needs it is not decided here: it
        is a property of the rendered panel, not of the CDF, and the caller measures it --
        see ``chorus.analysis._igv_report.escalate_scale_if_saturated``. Four genome-wide
        CDF statistics were tried as a proxy first and every one of them overlaps between
        the tracks that need the log band and the tracks that must not move.

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

        # Match track ID with possible alternative formats
        track_index = entry.get("track_index", {})
        matched_id = self._match_track_id(track_id, track_index)
        if matched_id is None:
            return None

        idx = track_index[matched_id]

        # Find appropriate CDF (perbin → summary → effect fallback).
        # ``_find_matching_cdf`` skips CDF types with zero samples per
        # track, so failed-build perbin rows fall through to summary
        # rather than saturating every display bin to ``max_value``.
        cdf = self._find_matching_cdf(entry, idx, matched_id)
        if cdf is None:
            return None

        # Compute thresholds and rescale.
        n = len(cdf)
        floor = float(cdf[min(int(floor_pctile * n), n - 1)])
        peak = float(cdf[min(int(peak_pctile * n), n - 1)])

        values = raw_values.astype(np.float64)
        if log_scale:
            # Clamp before log1p: this is the unsigned path, so a negative here is a model
            # artefact rather than repression, and log1p of it is undefined.
            values = np.log1p(np.maximum(values, 0.0))
            floor = float(np.log1p(max(floor, 0.0)))
            peak = float(np.log1p(max(peak, 0.0)))

        denom = max(peak - floor, 1e-9)
        out = (values - floor) / denom
        return np.clip(out, 0.0, max_value)

    def is_signed(self, oracle_name: str, track_id: str) -> bool:
        """Whether a track uses signed normalization (default: False)."""
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return False
        flags = entry.get("signed_flags")
        if flags is None:
            return False
        # Use the same fuzzy track-id matching as perbin_floor_rescale_batch
        # so callers don't have to know the exact NPZ key (e.g. LegNet
        # passes "LentiMPRA:HepG2" but the row is keyed "HepG2").
        track_index = entry.get("track_index", {})
        matched_id = self._match_track_id(track_id, track_index)
        if matched_id is None:
            return False
        return bool(flags[track_index[matched_id]])

    def signed_floor_rescale_batch(
        self,
        oracle_name: str,
        track_id: str,
        raw_values: np.ndarray,
        peak_pctile: float = 0.99,
        max_value: float = 3.0,
    ) -> np.ndarray | None:
        """Rescale **signed** raw values to ``[-max_value, +max_value]``.

        Uses ``p99(|cdf|)`` as the unit so ``±1.0`` corresponds to the
        genome-wide top-1% absolute effect for the track.  Both the
        positive and negative tails render symmetrically, unlike
        :meth:`perbin_floor_rescale_batch` which clips negatives to 0.

        Same CDF-fallback chain as the unsigned version
        (perbin → summary → effect via :meth:`_find_matching_cdf`),
        so e.g. LegNet (no ``perbin_cdfs``) uses ``summary_cdfs``.
        """
        entry = self._ensure_loaded(oracle_name)
        if entry is None:
            return None
        track_index = entry.get("track_index", {})
        matched_id = self._match_track_id(track_id, track_index)
        if matched_id is None:
            return None
        idx = track_index[matched_id]
        cdf = self._find_matching_cdf(entry, idx, matched_id)
        if cdf is None:
            return None
        # |cdf|.p99 — the magnitude that defines "1.0" on either side.
        # The original CDF is sorted (ascending); for signed values we
        # need to compute the abs-percentile from the underlying samples.
        abs_cdf = np.sort(np.abs(cdf))
        n = len(abs_cdf)
        unit = float(abs_cdf[min(int(peak_pctile * n), n - 1)])
        if unit < 1e-9:
            return None
        out = raw_values.astype(np.float64) / unit
        return np.clip(out, -max_value, max_value)

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
        provenance: dict | None = None,
        per_row: dict | None = None,
        sampling: dict | None = None,
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
            cache_dir: Output directory.  Defaults to ``<data-dir>/backgrounds/``.
            n_points: Number of CDF points per track (default 10,000).
            provenance: File-level facts about how this background was built —
                genome, formula, pseudocount, region-set composition, XLA flags,
                builder git sha, FASTA sha256, schema version. Stored as a JSON
                string under ``build_config``.
            per_row: ``{name: array}`` with ``len(array) == n_tracks``, for facts
                that vary by track: ``layer``, ``window_bp``, ``resolution``.
                Length is validated, because a per-row array of the wrong length
                silently mis-attributes every row after the first mismatch.

        Returns:
            Path to the saved ``.npz`` file.

        Note:
            Provenance exists because a background is otherwise only interpretable
            by joining it against whatever metadata happens to be on disk at read
            time. Borzoi's ``track_ids`` are opaque FANTOM accessions
            (``CNhs10608+``), so its CAGE and RNA rows are identifiable *today*
            only via ``borzoi_metadata.py`` — and if that file ever gains, drops or
            reorders tracks, previously-built rows get silently reinterpreted. That
            is the #144 failure mode applied to metadata instead of arithmetic
            (#124).
        """
        if cache_dir is None:
            cache_dir = str(CHORUS_BACKGROUNDS_DIR)
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_tracks = len(track_ids)
        arrays: dict[str, np.ndarray] = {
            "track_ids": np.array(track_ids, dtype="U"),
        }

        for name, matrix, counts in [
            ("effect_cdfs", effect_cdfs, effect_counts),
            ("summary_cdfs", summary_cdfs, summary_counts),
            ("perbin_cdfs", perbin_cdfs, perbin_counts),
        ]:
            if matrix is not None:
                assert matrix.shape[0] == n_tracks, (
                    f"{name} has {matrix.shape[0]} rows but {n_tracks} track_ids"
                )
                # Compact: subsample columns to n_points if needed
                if matrix.shape[1] > n_points:
                    indices = np.linspace(0, matrix.shape[1] - 1, n_points, dtype=int)
                    matrix = matrix[:, indices]
                # A width below n_points is not itself a defect: _get_denominator
                # reads shape[1], so a 100-wide matrix stored at 100 is entirely
                # self-consistent. Worth a note only because it means the caller's
                # requested grid and its actual grid diverged. Enformer's defect
                # was the opposite shape — gridded at 9,606 and *padded* to 10,000,
                # so shape[1] == n_points and no width check could have caught it.
                # The geometry check below is what catches padding.
                if matrix.shape[1] < n_points:
                    logger.warning(
                        "%s: %s is %d columns wide but n_points=%d; percentiles "
                        "will be computed against the stored width (%d)",
                        oracle_name, name, matrix.shape[1], n_points, matrix.shape[1],
                    )
                if counts is not None:
                    problems = yield_violations(
                        np.asarray(counts), label=f"{oracle_name}.{name}"
                    )
                    if problems:
                        raise ValueError(
                            "refusing to write a background where almost every "
                            "position failed:\n  " + "\n  ".join(problems)
                        )
                    problems = cdf_grid_violations(
                        matrix, np.asarray(counts), label=f"{oracle_name}.{name}"
                    )
                    if problems:
                        raise ValueError(
                            "refusing to write a CDF matrix that could not have "
                            "been produced by ReservoirSampler.to_cdf_matrix:\n  "
                            + "\n  ".join(problems)
                        )

                # A SECOND, independent check: was the sample thinned before it was
                # gridded? cdf_grid_violations cannot answer that -- it is handed the
                # OFFERED count while the geometry it validates is set by the RETAINED
                # count, and it skips every row with n >= n_points. Offered is always
                # >= n_points for a real build, so it skipped every thinned row, which
                # is how AlphaGenome shipped 667 RNA ceilings drawn from a 33.7%
                # subsample (median 1.33x understated, worst 8.34x).
                layer_key = name.replace("_cdfs", "")
                spec = (sampling or {}).get(layer_key)
                if spec is not None:
                    problems = thinning_violations(
                        np.asarray(spec["offered"]),
                        np.asarray(spec["retained"]),
                        n_points=matrix.shape[1],
                        tail_k=spec.get("tail_k"),
                        label=f"{oracle_name}.{name}",
                    )
                    if problems:
                        raise ValueError(
                            "refusing to write a CDF matrix whose top grid slots came "
                            "from a thinned sample -- the row maximum would be a draw "
                            "from a subsample, and that maximum is what "
                            "effect_percentile clamps against:\n  "
                            + "\n  ".join(problems)
                        )
                if spec is not None:
                    # Persist retention INTO the shipped file, not just the interim.
                    # Only the offered count was ever stored, so "was this track's tail
                    # thinned?" was unanswerable from a published background -- which is
                    # precisely why AlphaGenome's 2.97x thinning went unnoticed through
                    # a republish. One extra int64 per track per layer.
                    arrays[f"{layer_key}_retained"] = np.asarray(
                        spec["retained"], dtype=np.int64
                    )
                    if spec.get("tail_k"):
                        arrays[f"{layer_key}_tail_k"] = np.int64(spec["tail_k"])

                elif counts is not None and np.any(np.asarray(counts) > 0):
                    # Loud, not silent. A builder that forgets to pass `sampling` gets
                    # NO thinning protection at all, and "a guard nobody wired up" is
                    # how both the padded enformer grid and the AlphaGenome thinning
                    # reached users past guards that already existed.
                    logger.error(
                        "%s.%s written WITHOUT a sampling= block: thinning cannot be "
                        "checked for this matrix. Pass sampling={%r: {'offered': ..., "
                        "'retained': ..., 'tail_k': ...}} from the builder.",
                        oracle_name, name, layer_key,
                    )
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

        # Per-row provenance. Length is validated rather than trusted: an array of
        # the wrong length would silently mis-attribute every row past the first
        # mismatch, which is worse than having no provenance at all.
        for name, values in (per_row or {}).items():
            if name in arrays:
                raise ValueError(f"per_row key {name!r} collides with a CDF array")
            arr = np.asarray(values)
            if arr.shape[0] != n_tracks:
                raise ValueError(
                    f"per_row[{name!r}] has {arr.shape[0]} entries but there are "
                    f"{n_tracks} tracks; a length mismatch mis-attributes rows"
                )
            arrays[name] = arr

        if provenance:
            import json as _json

            # A 1-ELEMENT array, not 0-d. This wrote 0-d until 2026-08-04 while
            # both real producers -- Cherimoya's builder and
            # scripts/stamp_background_provenance.py -- wrote (1,), so a reader
            # written against the shipped files raised IndexError on anything this
            # function produced. Three producers, two conventions, nothing
            # comparing them: the same shape of defect as #122 and the duplicate
            # TSV writer. _read_build_config accepts both so old files still load.
            arrays["build_config"] = np.array(
                [_json.dumps(provenance, sort_keys=True, default=str)], dtype="U",
            )

        path = out_dir / PerTrackNormalizer.npz_filename(oracle_name)
        np.savez_compressed(str(path), **arrays)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved per-track CDFs for '%s': %d tracks, %.1f MB → %s",
            oracle_name, n_tracks, size_mb, path,
        )
        return path

    @staticmethod
    def append_tracks(
        oracle_name: str,
        new_track_ids: list[str],
        new_effect_cdfs: np.ndarray | None = None,
        new_summary_cdfs: np.ndarray | None = None,
        new_perbin_cdfs: np.ndarray | None = None,
        new_signed_flags: np.ndarray | None = None,
        new_effect_counts: np.ndarray | None = None,
        new_summary_counts: np.ndarray | None = None,
        new_perbin_counts: np.ndarray | None = None,
        cache_dir: str | None = None,
        new_provenance: dict | None = None,
        new_per_row: dict | None = None,
    ) -> tuple[Path, int]:
        """Append new tracks to an existing per-track NPZ file.

        Loads the existing ``{oracle}_pertrack.npz``, skips any
        *new_track_ids* that already exist, concatenates the rest,
        and saves back.  If no existing file is found, creates one
        from scratch.

        Returns:
            ``(path, n_added)`` — path to the saved file and the number
            of tracks that were actually appended (after dedup).
        """
        if cache_dir is None:
            cache_dir = str(CHORUS_BACKGROUNDS_DIR)
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / PerTrackNormalizer.npz_filename(oracle_name)

        # Load existing data (if any)
        existing: dict[str, np.ndarray] = {}
        existing_ids: set[str] = set()
        if npz_path.exists():
            with np.load(str(npz_path), allow_pickle=False) as data:
                for key in data.files:
                    existing[key] = data[key]
            existing_ids = set(str(t) for t in existing.get("track_ids", []))

        # Filter out duplicates
        keep = [i for i, tid in enumerate(new_track_ids) if tid not in existing_ids]
        if not keep:
            logger.info(
                "append_tracks('%s'): all %d tracks already present, nothing to add.",
                oracle_name, len(new_track_ids),
            )
            return npz_path, 0

        keep_idx = np.array(keep)
        added_ids = [new_track_ids[i] for i in keep]

        def _concat(existing_key, new_matrix):
            """Concatenate existing rows with new rows (filtered by keep_idx)."""
            if new_matrix is None:
                return existing.get(existing_key)
            new_rows = np.asarray(new_matrix)[keep_idx]
            old = existing.get(existing_key)
            if old is not None:
                return np.concatenate([old, new_rows], axis=0)
            return new_rows

        def _concat_1d(existing_key, new_arr):
            """Concatenate 1-D arrays (flags, counts).

            ``asarray`` because ``build_and_save`` accepts plain lists for these —
            ``[100, 100]`` is a perfectly ordinary way to pass counts — but fancy
            indexing with ``keep_idx`` needs an ndarray. Without it, a caller who
            builds fresh and a caller who appends have different accepted types
            for the same argument.
            """
            if new_arr is None:
                return existing.get(existing_key)
            new_vals = np.asarray(new_arr)[keep_idx]
            old = existing.get(existing_key)
            if old is not None:
                return np.concatenate([old, new_vals])
            return new_vals

        merged_ids = list(existing.get("track_ids", [])) + added_ids

        # Carry through any key outside the canonical eight, instead of dropping it.
        #
        # This is #124's actual defect: `existing` is loaded with EVERY key, but only
        # the canonical eight were forwarded, so a per-row `layer` or a file-level
        # `build_config` vanished on the first merge. Cherimoya works around it by
        # re-stamping build_config after every append — the only NPZ of nine that
        # has one.
        #
        # A per-row array is one whose first axis matches the existing row count; it
        # must be CONCATENATED. Passing it through unchanged would leave an
        # (n_old,) array against (n_old + n_new) rows, silently mis-attributing
        # every added row — worse than dropping it. Anything else is file-level and
        # passes through.
        CANONICAL = {
            "track_ids", "effect_cdfs", "summary_cdfs", "perbin_cdfs",
            "signed_flags", "effect_counts", "summary_counts", "perbin_counts",
        }
        n_existing = len(existing.get("track_ids", []))
        extra_per_row: dict = {}
        extra_file: dict = {}
        for key, value in existing.items():
            if key in CANONICAL:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape and arr.shape[0] == n_existing and n_existing:
                extra_per_row[key] = arr
            else:
                extra_file[key] = value

        merged_per_row: dict = {}
        for key, old_arr in extra_per_row.items():
            new_arr = (new_per_row or {}).get(key)
            if new_arr is None:
                logger.warning(
                    "append_tracks('%s'): per-row key %r exists on disk but was not "
                    "supplied for the %d new tracks; dropping it rather than "
                    "mis-aligning existing rows.",
                    oracle_name, key, len(added_ids),
                )
                continue
            merged_per_row[key] = np.concatenate(
                [old_arr, np.asarray(new_arr)[keep_idx]],
            )
        # keys supplied for the new tracks that the file did not have yet
        for key, new_arr in (new_per_row or {}).items():
            if key in merged_per_row or key in CANONICAL:
                continue
            if n_existing:
                logger.warning(
                    "append_tracks('%s'): per-row key %r supplied for new tracks but "
                    "absent for the %d existing ones; skipping.",
                    oracle_name, key, n_existing,
                )
                continue
            merged_per_row[key] = np.asarray(new_arr)[keep_idx]

        provenance = new_provenance
        if provenance is None and "build_config" in extra_file:
            provenance = PerTrackNormalizer._read_build_config(
                extra_file["build_config"], oracle_name,
            )
        extra_file.pop("build_config", None)

        path = PerTrackNormalizer.build_and_save(
            oracle_name=oracle_name,
            track_ids=merged_ids,
            effect_cdfs=_concat("effect_cdfs", new_effect_cdfs),
            summary_cdfs=_concat("summary_cdfs", new_summary_cdfs),
            perbin_cdfs=_concat("perbin_cdfs", new_perbin_cdfs),
            signed_flags=_concat_1d("signed_flags", new_signed_flags),
            effect_counts=_concat_1d("effect_counts", new_effect_counts),
            summary_counts=_concat_1d("summary_counts", new_summary_counts),
            perbin_counts=_concat_1d("perbin_counts", new_perbin_counts),
            cache_dir=cache_dir,
            provenance=provenance,
            per_row=merged_per_row or None,
        )
        logger.info(
            "append_tracks('%s'): added %d new tracks (%d total).",
            oracle_name, len(added_ids), len(merged_ids),
        )
        return path, len(added_ids)

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
    ``<data-dir>/backgrounds/``).  If not found locally, attempts to
    download from the ``lucapinello/chorus-backgrounds`` HuggingFace
    dataset.

    Returns a ready-to-use :class:`PerTrackNormalizer`, or ``None`` if
    no per-track CDFs exist for *oracle_name*.
    """
    if cache_dir is None:
        cache_dir = str(CHORUS_BACKGROUNDS_DIR)
    norm = PerTrackNormalizer(cache_dir=cache_dir)
    if norm.has_oracle(oracle_name):
        norm._ensure_loaded(oracle_name)
        return norm

    # Auto-download from HuggingFace
    n = download_pertrack_backgrounds(oracle_name, cache_dir=cache_dir)
    if n > 0 and norm.has_oracle(oracle_name):
        norm._ensure_loaded(oracle_name)
        return norm

    # alphagenome_pt runs the same model and weights through a different backend, so it shares the JAX
    # CDF rather than requiring a separate 5,168-track upload. Agreement is ~1-5% per head, measured, not
    # assumed — see `_CDF_ALIASES` above for the test that holds this up and what happened when its
    # coverage was narrower than the claim.
    if oracle_name == "alphagenome_pt":
        return get_pertrack_normalizer("alphagenome", cache_dir=cache_dir)

    return None


# Dataset repo holding the per-track background CDFs. Override with the
# CHORUS_BACKGROUNDS_REPO environment variable so contributors can self-host
# their own backgrounds (e.g. while developing a new oracle) before they are
# mirrored into the canonical repo. Default is unchanged.
_HF_REPO = os.environ.get("CHORUS_BACKGROUNDS_REPO", "lucapinello/chorus-backgrounds")


# Artefacts below this provenance schema predate the retention rebuild, so their
# ceilings are draws from a uniform subsample rather than population maxima.
MIN_ARTEFACT_SCHEMA = 4


#: The assembly every shipped background is built on, and the only one the query side
#: knows how to rank against.
#:
#: Not merely "what we happen to ship". The region sets a percentile is a rank within are
#: hg38-specific and have no mouse equivalent — the Meuleman DHS vocabulary in particular,
#: where SCREEN publishes an mm10 cCRE registry but the DHS index does not. So an mm10
#: artefact is not a drop-in even once one exists: the reference class needs its own mm10
#: variant and its own validation. Until that work happens, an artefact declaring anything
#: else is refused rather than ranked against.
RANKING_GENOME = "hg38"


#: Dataset revision this release of chorus was verified against.
#:
#: A percentile is a function of (code, artefacts), and the artefacts live in a separate
#: repo whose ``main`` moves. Without a pin, the same chorus commit gives different numbers
#: depending on when the user happened to download -- which is precisely what a version tag
#: is supposed to rule out. Demonstrated rather than hypothetical: the 2026-08-10 upload
#: replaced every file in place, so a v0.6.0 checkout silently changed behaviour that day.
#:
#: Pinned to a dataset **tag**, not a commit sha, so the pairing reads as a decision.
#: Release ↔ revision:
#:
#:   chorus ≤ 0.6.0   backgrounds-2026-08-01-preunified   (schema < 4, thinned ceilings)
#:   chorus 0.7.0     backgrounds-2026-08-06-schema4      (exact effect/summary retention)
#:   chorus 0.7.1     backgrounds-2026-08-10-layers        (+ layers_per_row on all eight)
#:   chorus 0.7.2     backgrounds-2026-08-12-cherimoya-fold0  (+ a cherimoya null per fold mode)
#:   chorus 0.7.3     backgrounds-2026-08-12-cherimoya-fold0  (unchanged: no artefact was rebuilt)
#:   chorus 0.7.4     backgrounds-2026-08-16-sei-full      (sei 40 -> 21,947 tracks, nucleosome-
#:                                                          normalized; see CHANGELOG #224/#225/#226)
#:
#: Override with ``CHORUS_BACKGROUNDS_REVISION`` -- set it to ``main`` to track the
#: dataset's head, which is what you want while developing a new oracle's background and
#: not what you want in an analysis you intend to reproduce.
_HF_REVISION = os.environ.get(
    "CHORUS_BACKGROUNDS_REVISION", "backgrounds-2026-08-16-sei-full"
)


def _hf_endpoint_reachable(timeout: float = 5.0) -> "tuple[bool, str]":
    """Can we open a TCP connection to the HuggingFace endpoint within *timeout*?

    Deliberately a socket connect rather than an HTTP request: the failure being
    guarded against is a silently-dropping firewall, where the connect never
    completes and every higher-level timeout is applied to a request that has not
    started. Returns ``(ok, detail)`` with *detail* naming the host and the reason,
    so the caller's message can say which host and why rather than "download failed".
    """
    import socket
    from urllib.parse import urlparse

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    parsed = urlparse(endpoint if "//" in endpoint else f"https://{endpoint}")
    host = parsed.hostname or "huggingface.co"
    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port}"
    except OSError as exc:
        return False, f"{host}:{port} — {type(exc).__name__}: {exc}"


def normalization_key(oracle_name: str, track=None, fold=None) -> str:
    """Which CDF key ranks a prediction from *oracle_name*, given its fold.

    A percentile is a rank against a null, so the null must come from the same model. Only
    Cherimoya has more than one: ``cherimoya`` (fold 0, the default) and
    ``cherimoya_ensemble`` (the 5-fold mean). Every other oracle returns its own name.

    The fold is read from the prediction's own metadata when a track is given -- every
    Cherimoya ``TrackPrediction`` stamps ``metadata["fold"]`` -- so a caller that already has
    the track does not need to know about folds at all. Pass ``fold=`` directly when only the
    oracle instance is in scope (``oracle.fold``).

    Getting this wrong is not a crash: it is a percentile that looks entirely normal and is
    ranked against a different model's distribution. The folds disagree by ~2x on the same
    sequence. ``tests/test_fold_selects_its_own_null.py`` enumerates the lookup sites so a new
    one cannot skip this resolution silently.
    """
    if oracle_name not in ("cherimoya", "cherimoya_ensemble"):
        return oracle_name
    if fold is None and track is not None:
        meta = getattr(track, "metadata", None)
        if isinstance(meta, dict):
            fold = meta.get("fold")
    return "cherimoya_ensemble" if fold == "ensemble" else "cherimoya"


class BackgroundArtefactMismatch(ValueError):
    """The artefact loaded is not the null this CDF key is supposed to rank against.

    Deliberately NOT swallowed by ``get_normalizer``'s legacy fallback, and the base class
    exists so that stays true for the next guard as well. Every other reason a per-track
    artefact fails to load means "no percentiles available", which is a degradation the
    fallback is right to absorb. These mean "the percentiles would be WRONG" -- ranked
    against a different distribution than the one they name -- and falling back silently
    would hide the single condition the guard was added to surface.
    """


class BackgroundFoldMismatch(BackgroundArtefactMismatch):
    """An artefact was built on a different model fold than the CDF key requires.

    The five CATv1 folds disagree by ~2x on the same sequence, so this is a plausible
    wrong number rather than an approximation.
    """


class BackgroundGenomeMismatch(BackgroundArtefactMismatch):
    """An artefact was built on a different genome assembly than chorus ranks against.

    Same shape of failure as the fold case and a worse magnitude: an mm10 null describes a
    different species' regulatory landscape, but every hg38 coordinate still resolves
    against it and every percentile still returns a number in [0, 1].

    Reachable today only by hand — all nine shipped artefacts are hg38 — which is exactly
    when to install the guard. #124's finding was that human-only holds by *accident*:
    Enformer and Borzoi are human because someone chose ``*_human_targets.txt``, and
    ChromBPNet, whose registry had no organism field, shipped 33 mm10 models scored against
    hg38 sequence. The next mouse artefact should meet a refusal, not a percentile.
    """


def _stamped_fold(path) -> object | None:
    """CATv1 fold recorded in an artefact's ``build_config``, or None if unstamped.

    Opened lazily and defensively: an unreadable or unstamped file must not break loading,
    because only Cherimoya has a fold at all and every other oracle's artefact has no such
    key. Returns None in that case so the caller skips the check rather than failing.
    """
    try:
        import json

        with np.load(str(path), allow_pickle=True) as z:
            if "build_config" not in z.files:
                return None
            raw = z["build_config"]
            raw = raw.item() if getattr(raw, "shape", None) == () else raw[0]
            cfg = json.loads(str(raw)) if isinstance(raw, (str, np.str_)) else raw
        return cfg.get("fold") if isinstance(cfg, dict) else None
    except Exception:
        return None


def _cached_artefact_is_superseded(path: Path) -> str | None:
    """Why this cached NPZ should be refetched, or None if it is current.

    Deliberately conservative: an unreadable or unrecognised file is reported as
    superseded rather than trusted, because the failure mode of accepting a bad
    cache is silent wrong percentiles, while the failure mode of refetching a good
    one is a download.
    """
    import json as _json

    try:
        with np.load(path, allow_pickle=True) as d:
            files = set(d.files)
            layers = [k[:-len("_cdfs")] for k in files if k.endswith("_cdfs")]
            missing = [f"{lay}_retained" for lay in layers
                       if f"{lay}_retained" not in files]
            if missing:
                return f"no {', '.join(sorted(missing))} -- predates the retention rebuild"
            if "build_config" not in files:
                return "no build_config"
            cfg = _json.loads(str(d["build_config"][0]))
    except Exception as exc:  # unreadable, truncated, or not an NPZ
        return f"unreadable ({type(exc).__name__})"

    schema = cfg.get("schema_version")
    if not isinstance(schema, int) or schema < MIN_ARTEFACT_SCHEMA:
        return f"provenance schema {schema!r} < {MIN_ARTEFACT_SCHEMA}"
    return None


def download_pertrack_backgrounds(
    oracle_name: str,
    cache_dir: str | None = None,
) -> int:
    """Download per-track background CDFs from HuggingFace for an oracle.

    Fetches ``{oracle_name}_pertrack.npz`` from the
    ``lucapinello/chorus-backgrounds`` dataset repo and saves it to
    *cache_dir* (default ``<data-dir>/backgrounds/``).

    Returns 1 if downloaded, 0 if already usable-as-cached or not available.

    A cached file is NOT automatically accepted. "The file exists" was the only
    check here, and it is the wrong one for any release that changes artefact
    CONTENT rather than adding a new oracle: publishing rebuilt CDFs would then
    reach nobody who had ever run ``chorus setup``, silently and permanently.

    Demonstrated before this guard existed: a pre-2026-08 ``legnet_pertrack.npz``
    dropped into an empty cache made this function return 0 and the normalizer
    load it without a word, with a K562 effect ceiling of 0.9057 against the
    rebuilt 0.7887. Percentiles in the body barely move, so nothing looks wrong —
    the damage is at the clamp, where the strong motif-creating variants this
    release exists to un-pin keep reading 1.0 with an exceedance multiplier
    inflated by up to 10x.

    So the cache is checked for the two properties that distinguish a rebuilt
    artefact from a superseded one, and a stale file is moved aside and refetched:

    * ``{layer}_retained`` present — added by the retention rebuild, and the only
      way to answer "was any track thinned" from the artefact;
    * ``build_config.schema_version >= MIN_ARTEFACT_SCHEMA``.

    Neither is a substitute for a published checksum manifest, which is the real
    fix and does not exist yet — but both are derivable from the file itself,
    which means this works without any new infrastructure or server round-trip.
    """
    if cache_dir is None:
        cache_dir = str(CHORUS_BACKGROUNDS_DIR)
    bg_dir = Path(cache_dir)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # alphagenome_pt aliases to alphagenome's NPZ — there is no separate
    # alphagenome_pt_pertrack.npz on HF.  Skip the download attempt (the
    # 404 is expected) and let the caller's alias-aware lookup take over.
    if oracle_name in PerTrackNormalizer._CDF_ALIASES:
        return 0

    fname = PerTrackNormalizer.npz_filename(oracle_name)
    local_path = bg_dir / fname
    if local_path.exists():
        stale = _cached_artefact_is_superseded(local_path)
        if not stale:
            return 0  # cached and current
        aside = local_path.with_suffix(".npz.superseded")
        logger.warning(
            "Cached %s is superseded (%s); moving it to %s and refetching. "
            "A superseded background ranks against the wrong null -- percentiles "
            "in the body look normal while extreme effects stay clamped.",
            fname, stale, aside.name,
        )
        try:
            local_path.replace(aside)
        except OSError as exc:
            logger.error(
                "Could not move the superseded %s aside (%s). Delete it manually "
                "and re-run, or percentiles will be computed against it.", fname, exc,
            )
            return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — cannot download backgrounds. "
            "Install it with: pip install huggingface_hub"
        )
        return 0

    # Bound the wait BEFORE handing off to huggingface_hub. Its timeout settings
    # (HF_HUB_ETAG_TIMEOUT / HF_HUB_DOWNLOAD_TIMEOUT) do not cover the TCP connect and
    # are multiplied by an internal retry loop, so on a network that DROPS packets to
    # huggingface.co -- the ordinary firewalled-cluster case -- this sat in SYN-SENT
    # indefinitely. Measured: still connecting after 18 minutes with one log line and
    # no diagnostic, and setting both env vars did not bound it either (still running
    # at a 300 s cap). The cost is also paid repeatedly, because get_normalizer()
    # retries and `chorus setup` loops over every oracle, so the user sees an apparent
    # hang for hours with nothing saying the host is unreachable.
    #
    # A short TCP preflight converts that into a clear failure in seconds. Set the env
    # defaults too, for the case where the host answers but then stalls mid-transfer.
    for var, default in (("HF_HUB_ETAG_TIMEOUT", "10"),
                         ("HF_HUB_DOWNLOAD_TIMEOUT", "30")):
        os.environ.setdefault(var, default)

    reachable, detail = _hf_endpoint_reachable()
    if not reachable:
        logger.warning(
            "Cannot reach the HuggingFace endpoint (%s), so %s was not downloaded. "
            "Percentiles need this file: analyses will return raw scores with no "
            "percentile columns until it is present. Fetch it on a connected machine "
            "and copy it into %s, or set HF_ENDPOINT to a reachable mirror.",
            detail, fname, bg_dir,
        )
        return 0

    try:
        logger.info("Downloading %s from HuggingFace ...", fname)
        hf_hub_download(
            _HF_REPO,
            filename=fname,
            repo_type="dataset",
            revision=_HF_REVISION,
            local_dir=str(bg_dir),
        )
        logger.info("Downloaded %s", fname)
        # If what we just fetched is ALSO superseded, the published dataset is behind
        # the code. Say so plainly instead of re-downloading it on every call: a
        # silent loop would look like a slow start-up, and a silent accept would give
        # wrong ceilings. Keep the file (it is the best available) and keep the
        # superseded copy aside so nothing is destroyed.
        why = _cached_artefact_is_superseded(local_path)
        if why:
            logger.warning(
                "The PUBLISHED %s is itself superseded (%s). The dataset at %s is "
                "behind this version of chorus, so effect ceilings will be draws from "
                "a uniform subsample until it is republished. Percentiles in the body "
                "are unaffected; strong effects will stay clamped at 1.0.",
                fname, why, _HF_REPO,
            )
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
        cache_dir = str(CHORUS_BACKGROUNDS_DIR)
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
        # Same revision as the download below. Listing `main` while fetching a tag would
        # ask for files that need not exist at that revision.
        files = api.list_repo_files(
            _HF_REPO, repo_type="dataset", revision=_HF_REVISION,
        )
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
                revision=_HF_REVISION,
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
        cache_dir = str(CHORUS_BACKGROUNDS_DIR)
    bg_dir = Path(cache_dir)

    # ── 1. Try the per-track NPZ first (new format, auto-downloaded from HF).
    try:
        pertrack = get_pertrack_normalizer(oracle_name, cache_dir=cache_dir)
        if pertrack is not None and pertrack.n_tracks(oracle_name) > 0:
            return pertrack
    except BackgroundArtefactMismatch:
        # Not a "no percentiles available" condition -- a "the percentiles would be wrong"
        # condition. Let it out. Catching the base class rather than BackgroundFoldMismatch
        # so the next guard in this family inherits the non-swallowing contract instead of
        # having to remember to widen this clause.
        raise
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


def is_ready_for_oracle(
    oracle_name: str, cache_dir: str | None = None,
) -> bool:
    """True if either a per-track NPZ or per-layer Quantile NPYs are on disk.

    Unified readiness check across both normalizer flavors. Useful for
    setup / health-check code that doesn't care which on-disk layout
    is present, only whether normalization will work for the oracle.

    - :class:`PerTrackNormalizer` looks for ``{oracle}_pertrack.npz``.
    - :class:`QuantileNormalizer` looks for ``{oracle}_*.npy`` per-layer files.

    Returns True if at least one exists locally (does not trigger a
    HuggingFace download).
    """
    pt = PerTrackNormalizer(cache_dir=cache_dir)
    if pt.has_oracle(oracle_name):
        return True
    # Honor the same alias map the loader uses (e.g. alphagenome_pt → alphagenome)
    aliased = PerTrackNormalizer._CDF_ALIASES.get(oracle_name)
    if aliased is not None and pt.has_oracle(aliased):
        return True
    q = QuantileNormalizer(cache_dir=cache_dir)
    if q.has_oracle_quantiles(oracle_name):
        return True
    if aliased is not None and q.has_oracle_quantiles(aliased):
        return True
    return False
