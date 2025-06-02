"""Utility functions for the Chorus library."""

from .sequence import (
    extract_sequence,
    parse_vcf,
    apply_variant,
    reverse_complement,
    validate_sequence,
    get_gc_content,
    split_sequence_into_windows,
    pad_sequence
)

from .normalization import (
    normalize_tracks,
    quantile_normalize,
    minmax_normalize,
    zscore_normalize,
    robust_zscore_normalize,
    log_transform,
    percentile_normalize,
    normalize_by_reference
)

from .visualization import (
    visualize_tracks,
    plot_track_heatmap,
    plot_track_comparison
)

__all__ = [
    # Sequence utilities
    'extract_sequence',
    'parse_vcf',
    'apply_variant',
    'reverse_complement',
    'validate_sequence',
    'get_gc_content',
    'split_sequence_into_windows',
    'pad_sequence',
    
    # Normalization utilities
    'normalize_tracks',
    'quantile_normalize',
    'minmax_normalize',
    'zscore_normalize',
    'robust_zscore_normalize',
    'log_transform',
    'percentile_normalize',
    'normalize_by_reference',
    
    # Visualization utilities
    'visualize_tracks',
    'plot_track_heatmap',
    'plot_track_comparison'
]