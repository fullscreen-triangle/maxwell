"""
Categorical Memory System (S-RAM)

A virtual RAM implementation based on S-entropy navigation and precision-by-difference
addressing. This system doesn't replace existing memory - it provides an intelligent
layer that uses categorical completion to determine optimal data placement.

Key Concepts:
- History IS the address (precision-by-difference trajectory)
- Navigation through 3^k recursive hierarchy
- Categorical completion instead of prediction
- Zero backaction memory access through snapshots
"""

from .s_entropy_address import SEntropyAddress, SCoordinate
from .precision_calculator import PrecisionByDifferenceCalculator
from .categorical_hierarchy import CategoricalHierarchy, HierarchyNode
from .memory_controller import CategoricalMemoryController
from .hardware_oscillator import HardwareOscillatorCapture

__all__ = [
    'SEntropyAddress',
    'SCoordinate', 
    'PrecisionByDifferenceCalculator',
    'CategoricalHierarchy',
    'HierarchyNode',
    'CategoricalMemoryController',
    'HardwareOscillatorCapture',
]


