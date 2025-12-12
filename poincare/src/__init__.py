"""
Poincaré: Virtual Categorical Gas Chamber
==========================================

This is NOT a simulation.

The computer's hardware oscillations ARE the categorical gas.
Each timing measurement IS a molecule's categorical state.
The spectrometer IS the molecule being measured.

Key principles:
1. The hook defines the catch - your apparatus shapes what can be measured
2. No surprise - you catch exactly what your tackle can catch
3. Cursor = Molecule = Spectrometer - they are the same categorical state
4. No propagation - categorical distance is independent of spatial distance
"""

from .virtual_molecule import CategoricalState, VirtualMolecule
from .virtual_spectrometer import VirtualSpectrometer, FishingTackle
from .virtual_chamber import VirtualChamber, CategoricalGas
from .molecular_dynamics import CategoricalDynamics
from .maxwell_demon import MaxwellDemon
from .thermodynamics import CategoricalThermodynamics
from .visualization import CategoricalVisualizer

__all__ = [
    'CategoricalState',
    'VirtualMolecule', 
    'VirtualSpectrometer',
    'FishingTackle',
    'VirtualChamber',
    'CategoricalGas',
    'CategoricalDynamics',
    'MaxwellDemon',
    'CategoricalThermodynamics',
    'CategoricalVisualizer',
]

__version__ = '0.1.0'

