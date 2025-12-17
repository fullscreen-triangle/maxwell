"""
Poincaré Computing
==================

A unified computational framework where processor, memory, and semantic
processing are the same categorical state viewed from different perspectives.

Named for Henri Poincaré and his recurrence theorem: in a bounded phase space,
a dynamical system returns to its initial state. In Poincaré Computing,
**recurrence is solution**—computation completes when the categorical state
returns to its origin. The trajectory through S-entropy space IS the answer.

Core Principles:
1. The computer IS the gas (not a device that processes gas)
2. Molecule = Address = Processor State = Meaning (simultaneously)
3. Computation is navigation, not execution
4. Return to origin = problem solved
5. No separation between processor/memory/semantics
6. Hardware oscillations create real categorical states

The Poincaré Recurrence Connection:
- S-entropy space is bounded ([0,1]³) → recurrence guaranteed
- Problem specification = initial state
- Categorical completion = phase space dynamics
- Solution = trajectory that returns to origin
- No return = unsatisfiable constraints
"""

from .virtual_molecule import CategoricalState, VirtualMolecule
from .virtual_spectrometer import VirtualSpectrometer, FishingTackle
from .virtual_chamber import VirtualChamber, CategoricalGas
from .molecular_dynamics import CategoricalDynamics
from .maxwell_demon import MaxwellDemon
from .thermodynamics import CategoricalThermodynamics
from .visualization import CategoricalVisualizer
from .virtual_capacitor import VirtualCapacitor, GenomeCapacitor, ChargeState
from .virtual_aperture import (
    CategoricalAperture, ChargeFieldAperture, ApertureCascade,
    temperature_independence_experiment, categorical_exclusion_experiment,
    cascade_amplification_experiment
)
from .virtual_partition import (
    VirtualPartition, PartitionCompositionCycle, CategoricalAggregate,
    PartitionResult, entropy_equivalence_experiment, millet_paradox_experiment
)
from .virtual_element_synthesizer import (
    ElementSynthesizer, ShellResonator, AngularAnalyzer, OrientationMapper,
    ChiralityDiscriminator, ExclusionDetector, EnergyProfiler,
    SpectralLineAnalyzer, IonizationProbe, ElectronegativitySensor,
    AtomicRadiusGauge, PartitionCoordinate, ElementSignature,
    periodic_table_from_partition_geometry
)

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
    'VirtualCapacitor',
    'GenomeCapacitor',
    'ChargeState',
    'CategoricalAperture',
    'ChargeFieldAperture',
    'ApertureCascade',
    'temperature_independence_experiment',
    'categorical_exclusion_experiment',
    'cascade_amplification_experiment',
    'VirtualPartition',
    'PartitionCompositionCycle',
    'CategoricalAggregate',
    'PartitionResult',
    'entropy_equivalence_experiment',
    'millet_paradox_experiment',
    # Exotic instruments for element synthesis
    'ElementSynthesizer',
    'ShellResonator',
    'AngularAnalyzer',
    'OrientationMapper',
    'ChiralityDiscriminator',
    'ExclusionDetector',
    'EnergyProfiler',
    'SpectralLineAnalyzer',
    'IonizationProbe',
    'ElectronegativitySensor',
    'AtomicRadiusGauge',
    'PartitionCoordinate',
    'ElementSignature',
    'periodic_table_from_partition_geometry',
]

__version__ = '0.1.0'

