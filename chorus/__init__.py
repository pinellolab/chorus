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

# Import oracles - make them optional to avoid dependency issues
import os
import warnings

if not os.environ.get('CHORUS_DISABLE_ORACLE_IMPORTS'):
    try:
        from .oracles import (
            EnformerOracle,
            BorzoiOracle,
            ChromBPNetOracle,
            SeiOracle,
            get_oracle,
            ORACLES
        )
    except ImportError as e:
        warnings.warn(
            f"Some oracle imports failed: {e}\n"
            "This is expected if oracle-specific dependencies are not installed.\n"
            "Use 'chorus setup --oracle <name>' to set up oracle environments."
        )
        # Provide dummy implementations
        EnformerOracle = None
        BorzoiOracle = None
        ChromBPNetOracle = None
        SeiOracle = None
        ORACLES = {}
        
        def get_oracle(name: str):
            raise ImportError(
                f"Oracle '{name}' requires its environment to be set up.\n"
                f"Run: chorus setup --oracle {name}"
            )
else:
    # Testing mode - skip oracle imports
    EnformerOracle = None
    BorzoiOracle = None
    ChromBPNetOracle = None
    SeiOracle = None
    ORACLES = {}
    get_oracle = None

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
def create_oracle(oracle_name: str, use_environment: bool = False, **kwargs):
    """
    Create an oracle instance by name.
    
    Args:
        oracle_name: Name of the oracle (enformer, borzoi, chrombpnet, sei)
        use_environment: If True, use isolated conda environment for the oracle
        **kwargs: Additional arguments passed to oracle constructor, including:
            - model_load_timeout: Timeout for model loading in seconds (default: 600)
            - predict_timeout: Timeout for predictions in seconds (default: 300)
            - reference_fasta: Path to reference genome FASTA file
            - device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        Oracle instance
        
    Example:
        >>> # Default (auto-detect GPU)
        >>> oracle = chorus.create_oracle('enformer', use_environment=True)
        >>> 
        >>> # Force CPU usage
        >>> oracle = chorus.create_oracle('enformer', 
        ...                              use_environment=True,
        ...                              device='cpu')
        >>> 
        >>> # Use specific GPU
        >>> oracle = chorus.create_oracle('enformer',
        ...                              use_environment=True,
        ...                              device='cuda:1')  # Use second GPU
        >>> 
        >>> # Custom timeouts for slower systems
        >>> oracle = chorus.create_oracle('enformer', 
        ...                              use_environment=True,
        ...                              model_load_timeout=1200,  # 20 minutes
        ...                              predict_timeout=600,      # 10 minutes
        ...                              device='cpu')             # Force CPU
        >>> 
        >>> # Disable all timeouts
        >>> oracle = chorus.create_oracle('enformer',
        ...                              use_environment=True,
        ...                              model_load_timeout=None,
        ...                              predict_timeout=None)
        >>> oracle.load_pretrained_model()
        
    Environment Variables:
        - CHORUS_NO_TIMEOUT=1: Disable all timeouts globally
        - CHORUS_DEVICE=cpu: Set default device (can be overridden by device parameter)
    """
    if use_environment:
        # Use the environment-aware oracle implementations
        if oracle_name.lower() == 'enformer':
            from .oracles.enformer import EnformerOracle
            return EnformerOracle(use_environment=True, **kwargs)
        elif oracle_name.lower() == 'sei':
            from .oracles.sei_oracle import SeiOracle
            return SeiOracle(use_environment=True, **kwargs)
        else:
            raise NotImplementedError(
                f"Environment-isolated version of {oracle_name} not yet implemented.\n"
                f"You can use the base version with use_environment=False"
            )
    else:
        # Use direct oracle (requires dependencies in current environment)
        if get_oracle is None:
            raise ImportError(
                "Direct oracle usage requires oracle dependencies.\n"
                "Either:\n"
                "1. Use use_environment=True for isolated execution, or\n"
                "2. Install oracle dependencies in the current environment"
            )
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