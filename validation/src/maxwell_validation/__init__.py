"""
Maxwell Validation
==================

Validation and testing suite for the Complementarity-Aware Processor.

This package provides:
- Unit tests for categorical and kinetic engines
- Integration tests for complementarity constraints
- Validation of the seven-fold dissolution
- Numerical verification of phase-lock networks
"""

from .categorical import CategoricalValidator
from .kinetic import KineticValidator
from .complementarity import ComplementarityValidator
from .dissolution import DissolutionValidator

__version__ = "0.1.0"
__all__ = [
    "CategoricalValidator",
    "KineticValidator",
    "ComplementarityValidator",
    "DissolutionValidator",
]

