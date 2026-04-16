"""Multi-layer genomic analysis framework.

Provides modality-specific scoring, quantile normalization, and
multi-layer variant interpretation tools built on top of Chorus
oracle primitives.
"""

from .scorers import (
    LAYER_CONFIGS,
    LayerConfig,
    classify_track_layer,
    score_track_effect,
    score_variant_multilayer,
)
from .normalization import (
    BackgroundDistribution,
    QuantileNormalizer,
    PerTrackNormalizer,
    get_normalizer,
    get_pertrack_normalizer,
    download_backgrounds,
    download_pertrack_backgrounds,
)
from .variant_report import (
    TrackScore,
    VariantReport,
    build_variant_report,
)
from .discovery import (
    CellTypeHit,
    TrackEffect,
    discover_cell_types,
    discover_and_report,
    discover_variant_effects,
)
from .region_swap import analyze_region_swap
from .integration import simulate_integration
from .batch_scoring import BatchVariantScore, BatchResult, score_variant_batch
from .causal import (
    CausalVariantScore,
    CausalWeights,
    CausalResult,
    prioritize_causal_variants,
)
from .build_backgrounds import (
    build_variant_backgrounds,
    build_baseline_backgrounds,
    get_common_snps,
)

__all__ = [
    "LAYER_CONFIGS",
    "LayerConfig",
    "classify_track_layer",
    "score_track_effect",
    "score_variant_multilayer",
    "BackgroundDistribution",
    "QuantileNormalizer",
    "PerTrackNormalizer",
    "get_normalizer",
    "get_pertrack_normalizer",
    "download_backgrounds",
    "download_pertrack_backgrounds",
    "TrackScore",
    "VariantReport",
    "build_variant_report",
    "CellTypeHit",
    "TrackEffect",
    "discover_cell_types",
    "discover_and_report",
    "discover_variant_effects",
    "analyze_region_swap",
    "simulate_integration",
    "BatchVariantScore",
    "BatchResult",
    "score_variant_batch",
    "CausalVariantScore",
    "CausalWeights",
    "CausalResult",
    "prioritize_causal_variants",
    "build_variant_backgrounds",
    "build_baseline_backgrounds",
    "get_common_snps",
]
