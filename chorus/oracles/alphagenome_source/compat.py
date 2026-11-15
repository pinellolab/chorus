"""Version-tolerant imports from the ``alphagenome`` package.

alphagenome 0.10 added the ``alphagenome.models.v1`` namespace, and 2.0 removed
the legacy ``alphagenome.models.*`` aliases. Import these names from here rather
than from ``alphagenome`` directly so the fallback lives in one place.
"""

try:  # alphagenome >= 0.10 (required from 2.0)
    from alphagenome.models.v1.dna_model import Organism
    from alphagenome.models.v1.dna_output import OutputType
except ImportError:  # alphagenome < 0.10
    from alphagenome.models.dna_model import Organism
    from alphagenome.models.dna_output import OutputType

__all__ = ["Organism", "OutputType"]
