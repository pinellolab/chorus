"""Quantile normalization against background distributions.

Maps raw variant effect scores to quantile scores, making them comparable
across oracles, tracks, and modalities.

Two normalization modes determined by the layer's ``signed`` flag:

- **Unsigned [0, 1]** — for chromatin, ChIP, CAGE, splicing layers.
  Background stores ``|effect|`` values.  Higher quantile = larger effect
  magnitude.  Direction (gain/loss) is preserved in the raw score sign.
- **Signed [-1, 1]** — for MPRA, expression, Sei layers.
  Background stores raw effect values (positive and negative).
  Quantile reflects both magnitude and direction.

**Important**: background distributions must match the normalization mode.
For unsigned layers, compute backgrounds with ``np.abs(scores)``.
Use :meth:`BackgroundDistribution.from_scores` for automatic handling.

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
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the sorted background to a ``.npy`` file."""
        np.save(path, self.sorted_scores)

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

    @staticmethod
    def background_key(oracle_name: str, layer: str) -> str:
        """Generate a standard cache key for a background distribution."""
        return f"{oracle_name}_{layer}"
