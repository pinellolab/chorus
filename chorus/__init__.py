"""
Chorus: A unified interface for genomic sequence oracles.

Chorus provides a consistent API for working with various genomic deep learning
models including Enformer, Borzoi, ChromBPNet, and Sei.
"""

__version__ = "0.1.0"

# Import core classes
from .core import (
    OracleBase,
    Track,
    ChorusError,
    ModelNotLoadedError,
    InvalidSequenceError,
    InvalidAssayError,
    InvalidRegionError,
    FileFormatError
)

# Import oracles
from .oracles import (
    EnformerOracle,
    BorzoiOracle,
    ChromBPNetOracle,
    SeiOracle,
    get_oracle,
    ORACLES
)

# Import utilities
from .utils import (
    # Sequence utilities
    extract_sequence,
    parse_vcf,
    apply_variant,
    reverse_complement,
    validate_sequence,
    get_gc_content,
    
    # Normalization utilities
    normalize_tracks,
    quantile_normalize,
    minmax_normalize,
    zscore_normalize,
    
    # Visualization utilities
    visualize_tracks,
    plot_track_heatmap,
    plot_track_comparison
)

# Convenience function to create oracle instances
def create_oracle(oracle_name: str, **kwargs):
    """
    Create an oracle instance by name.
    
    Args:
        oracle_name: Name of the oracle (enformer, borzoi, chrombpnet, sei)
        **kwargs: Additional arguments passed to oracle constructor
        
    Returns:
        Oracle instance
        
    Example:
        >>> oracle = chorus.create_oracle('enformer')
        >>> oracle.load_pretrained_model()
    """
    oracle_class = get_oracle(oracle_name)
    return oracle_class(**kwargs)

__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'OracleBase',
    'Track',
    
    # Exceptions
    'ChorusError',
    'ModelNotLoadedError',
    'InvalidSequenceError',
    'InvalidAssayError',
    'InvalidRegionError',
    'FileFormatError',
    
    # Oracle classes
    'EnformerOracle',
    'BorzoiOracle',
    'ChromBPNetOracle',
    'SeiOracle',
    
    # Oracle utilities
    'get_oracle',
    'create_oracle',
    'ORACLES',
    
    # Sequence utilities
    'extract_sequence',
    'parse_vcf',
    'apply_variant',
    'reverse_complement',
    'validate_sequence',
    'get_gc_content',
    
    # Normalization utilities  
    'normalize_tracks',
    'quantile_normalize',
    'minmax_normalize',
    'zscore_normalize',
    
    # Visualization utilities
    'visualize_tracks',
    'plot_track_heatmap',
    'plot_track_comparison'
]