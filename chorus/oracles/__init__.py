"""Oracle implementations for the Chorus library."""

from .enformer import EnformerOracle
from .borzoi import BorzoiOracle
from .chrombpnet import ChromBPNetOracle
from .sei import SeiOracle

# Dictionary for easy oracle access
ORACLES = {
    'enformer': EnformerOracle,
    'borzoi': BorzoiOracle,
    'chrombpnet': ChromBPNetOracle,
    'sei': SeiOracle
}

def get_oracle(name: str) -> type:
    """
    Get oracle class by name.
    
    Args:
        name: Oracle name (enformer, borzoi, chrombpnet, sei)
        
    Returns:
        Oracle class
    """
    name = name.lower()
    if name not in ORACLES:
        raise ValueError(f"Unknown oracle: {name}. Available: {list(ORACLES.keys())}")
    return ORACLES[name]

__all__ = [
    'EnformerOracle',
    'BorzoiOracle', 
    'ChromBPNetOracle',
    'SeiOracle',
    'ORACLES',
    'get_oracle'
]