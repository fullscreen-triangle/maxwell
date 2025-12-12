"""
Core Types for Maxwell Processor
================================

Fundamental data structures representing:
- S-coordinates (S_k, S_t, S_e)
- Categorical and Kinetic states
- Phase-lock networks
- Complementarity structures

Based on the theoretical framework from:
- Categorical Resolution of Gibbs' Paradox
- Biological Oscillatory Semiconductors
- Information Complementarity
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import numpy as np


class ObservableFace(Enum):
    """The observable face of information - only one can be observed at a time"""
    CATEGORICAL = auto()  # Phase-lock networks, topology, completion
    KINETIC = auto()      # Velocities, temperatures, thermodynamics


class MolecularType(Enum):
    """Type of molecule affecting phase-lock formation"""
    POLAR = auto()
    NON_POLAR = auto()
    DIPOLAR = auto()


class InteractionType(Enum):
    """Type of intermolecular interaction"""
    VAN_DER_WAALS = auto()    # ~r^-6 dependence
    DIPOLE_DIPOLE = auto()    # ~r^-3 dependence
    INDUCED_DIPOLE = auto()   # ~r^-4 dependence
    VIBRATIONAL = auto()      # Frequency coupling


class CarrierType(Enum):
    """Type of charge carrier in biological semiconductor"""
    P_TYPE = auto()  # Oscillatory holes (functional absences)
    N_TYPE = auto()  # Molecular carriers (pharmaceutical molecules)


class DissolutionArgument(Enum):
    """The seven arguments dissolving Maxwell's Demon"""
    TEMPORAL_TRIVIALITY = 1
    PHASE_LOCK_TEMPERATURE_INDEPENDENCE = 2
    RETRIEVAL_PARADOX = 3
    DISSOLUTION_OF_OBSERVATION = 4
    DISSOLUTION_OF_DECISION = 5
    DISSOLUTION_OF_SECOND_LAW = 6
    INFORMATION_COMPLEMENTARITY = 7


@dataclass
class SCoordinates:
    """
    S-entropy coordinates (S_k, S_t, S_e) in tri-dimensional S-space.
    
    These coordinates represent position in the information-theoretic space
    where BMDs operate and navigate.
    """
    s_k: float  # Knowledge entropy (information dimension)
    s_t: float  # Temporal entropy (time dimension)  
    s_e: float  # Evolutionary entropy (entropy dimension)
    
    def conjugate(self) -> 'SCoordinates':
        """Compute the conjugate (back face) coordinates"""
        return SCoordinates(-self.s_k, -self.s_t, -self.s_e)
    
    def phase_conjugate(self) -> 'SCoordinates':
        """Phase conjugate (inverts knowledge only)"""
        return SCoordinates(-self.s_k, self.s_t, self.s_e)
    
    def distance(self, other: 'SCoordinates') -> float:
        """Euclidean distance in S-space"""
        dk = self.s_k - other.s_k
        dt = self.s_t - other.s_t
        de = self.s_e - other.s_e
        return np.sqrt(dk**2 + dt**2 + de**2)
    
    def sums_to_zero_with(self, other: 'SCoordinates', tolerance: float = 1e-10) -> bool:
        """Check if coordinates sum to zero (conjugate verification)"""
        return (abs(self.s_k + other.s_k) < tolerance and
                abs(self.s_t + other.s_t) < tolerance and
                abs(self.s_e + other.s_e) < tolerance)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.s_k, self.s_t, self.s_e])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SCoordinates':
        """Create from numpy array"""
        return cls(arr[0], arr[1], arr[2])
    
    @classmethod
    def origin(cls) -> 'SCoordinates':
        """Return origin coordinates"""
        return cls(0.0, 0.0, 0.0)


@dataclass
class OscillatorySignature:
    """
    Oscillatory signature of a molecular or biological system.
    
    Ω(t) = (A, ω, φ) - amplitude, frequency, phase
    These signatures determine interaction specificity through resonance.
    """
    amplitude: float      # A - oscillation amplitude
    frequency: float      # ω - angular frequency (rad/s)
    phase: float          # φ - phase offset (rad)
    harmonics: List[Tuple[float, float, float]] = field(default_factory=list)  # Higher harmonics
    
    def resonates_with(self, other: 'OscillatorySignature', bandwidth: float = 0.1) -> bool:
        """Check if this signature resonates with another"""
        freq_diff = abs(self.frequency - other.frequency)
        return freq_diff < bandwidth * max(self.frequency, other.frequency)
    
    def overlap_integral(self, other: 'OscillatorySignature', duration: float = 1.0) -> float:
        """Compute the therapeutic overlap integral"""
        # Simplified: using frequency matching
        freq_match = np.exp(-((self.frequency - other.frequency) / self.frequency)**2)
        phase_match = np.cos(self.phase - other.phase)
        amplitude_product = self.amplitude * other.amplitude
        return amplitude_product * freq_match * (1 + phase_match) / 2


@dataclass
class OscillatoryHole:
    """
    An oscillatory hole - a functional absence that behaves as an active carrier.
    
    Analogous to positive holes in semiconductor physics.
    These represent missing oscillatory components in biological pathways.
    """
    id: int
    missing_signature: OscillatorySignature  # The missing oscillatory component
    therapeutic_charge: float = 1.0          # q_h - effective therapeutic charge
    mobility: float = 0.0123                 # μ_h - mobility (cm²/(V·s))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @property
    def concentration(self) -> float:
        """Hole concentration (would be computed from context)"""
        return 2.80e12  # cm^-3 (from paper measurements)
    
    def drift_velocity(self, therapeutic_field: float) -> float:
        """Compute drift velocity under therapeutic field"""
        return self.mobility * therapeutic_field
    
    def diffusion_coefficient(self, temperature: float = 300.0) -> float:
        """Compute diffusion coefficient via Einstein relation"""
        k_b = 1.380649e-23
        return (k_b * temperature / self.therapeutic_charge) * self.mobility


@dataclass 
class MolecularCarrier:
    """
    A molecular carrier (N-type) - pharmaceutical or endogenous molecule.
    
    These complete oscillatory holes through resonance matching.
    """
    id: int
    signature: OscillatorySignature
    molecular_mass: float  # Da
    concentration: float   # M (molar)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def can_fill(self, hole: OscillatoryHole, threshold: float = 0.5) -> bool:
        """Check if this carrier can fill the given hole"""
        overlap = self.signature.overlap_integral(hole.missing_signature)
        return overlap > threshold
    
    def recombination_rate(self, hole: OscillatoryHole, coefficient: float = 1e-10) -> float:
        """Compute recombination rate with a hole"""
        return coefficient * self.concentration * hole.concentration


@dataclass
class PhaseLockNode:
    """A node in the phase-lock network"""
    id: int
    frequency: float      # Oscillatory frequency (Hz)
    phase: float          # Current phase (rad)
    amplitude: float      # Oscillation amplitude
    molecular_type: MolecularType = MolecularType.NON_POLAR
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def phase_difference(self, other: 'PhaseLockNode') -> float:
        """Compute phase difference with another node"""
        return (self.phase - other.phase) % (2 * np.pi)


@dataclass
class PhaseLockEdge:
    """An edge (coupling) in the phase-lock network"""
    source_id: int
    target_id: int
    coupling_strength: float
    interaction_type: InteractionType = InteractionType.VAN_DER_WAALS


@dataclass
class CategoricalState:
    """
    A categorical state in phase-lock space.
    
    This represents what's ACTUALLY happening (ground truth),
    as opposed to the kinetic projection (what Maxwell saw).
    """
    id: int
    coordinates: SCoordinates
    cluster_id: Optional[int] = None
    completed: bool = False
    accessible: Set[int] = field(default_factory=set)
    phase_locks: Dict[int, float] = field(default_factory=dict)  # node_id -> coupling
    
    def complete(self):
        """Mark this state as completed (irreversible)"""
        self.completed = True
    
    def can_access(self, other_id: int) -> bool:
        """Check if another state is accessible from this one"""
        return other_id in self.accessible
    
    def add_phase_lock(self, other_id: int, coupling: float):
        """Add a phase-lock edge"""
        self.phase_locks[other_id] = coupling
        self.accessible.add(other_id)


@dataclass
class KineticState:
    """
    A kinetic state (what Maxwell observed).
    
    This is the PROJECTION of categorical dynamics onto
    the observable kinetic face.
    """
    id: int
    velocity: float       # m/s
    kinetic_energy: float # J
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @classmethod
    def from_velocity(cls, id: int, velocity: float, position: np.ndarray, mass: float = 1.0):
        """Create from velocity"""
        ke = 0.5 * mass * velocity**2
        return cls(id=id, velocity=velocity, kinetic_energy=ke, position=position)
    
    @property
    def temperature_contribution(self) -> float:
        """Temperature contribution (proportional to KE)"""
        return self.kinetic_energy
    
    def classify(self, threshold: float) -> str:
        """Classify as fast or slow"""
        return "fast" if self.velocity > threshold else "slow"


@dataclass
class PNJunction:
    """
    A biological P-N junction enabling therapeutic rectification.
    
    Forms at the interface between P-type (holes) and N-type (molecules) regions.
    """
    p_region_concentration: float  # N_A - acceptor concentration
    n_region_concentration: float  # N_D - donor concentration
    temperature: float = 300.0     # K
    
    @property
    def intrinsic_carrier_density(self) -> float:
        """Compute intrinsic carrier density n_i"""
        return np.sqrt(self.p_region_concentration * self.n_region_concentration)
    
    @property
    def built_in_potential(self) -> float:
        """Compute built-in potential V_bi (V)"""
        k_b = 1.380649e-23
        e = 1.602e-19
        n_i = self.intrinsic_carrier_density
        return (k_b * self.temperature / e) * np.log(
            self.p_region_concentration * self.n_region_concentration / n_i**2
        )
    
    @property
    def depletion_width(self, permittivity: float = 8.85e-12) -> float:
        """Compute depletion width W (m)"""
        e = 1.602e-19
        na = self.p_region_concentration
        nd = self.n_region_concentration
        return np.sqrt(
            2 * permittivity / e * 
            ((na + nd) / (na * nd)) * 
            self.built_in_potential
        )
    
    def current(self, voltage: float) -> float:
        """Compute therapeutic current (diode equation)"""
        k_b = 1.380649e-23
        e = 1.602e-19
        i_0 = 1e-12  # Reverse saturation current
        return i_0 * (np.exp(e * voltage / (k_b * self.temperature)) - 1)
    
    def rectification_ratio(self, voltage: float = 0.1) -> float:
        """Compute rectification ratio at given voltage"""
        i_forward = self.current(voltage)
        i_reverse = abs(self.current(-voltage))
        return i_forward / max(i_reverse, 1e-20)


@dataclass
class ProcessorConfig:
    """Configuration for the Maxwell Processor"""
    num_molecules: int = 1000
    temperature: float = 300.0           # K
    coupling_distance: float = 1e-9      # ~1 nm
    vdw_coefficient: float = 1e-77       # J·m^6
    collision_frequency: float = 1e10    # 10 GHz
    tolerance: float = 1e-10
    
    # Biological semiconductor parameters
    hole_mobility: float = 0.0123        # cm²/(V·s)
    carrier_mobility: float = 0.001      # cm²/(V·s)
    therapeutic_conductivity: float = 7.53e-8  # S/cm


@dataclass
class CompletionResult:
    """Result of recursive categorical completion"""
    states_completed: int
    depth_reached: int
    decomposition_count: int  # 3^k decomposition
    entropy_change: float
    completed_ids: List[int]
    cascade_path: List[List[int]]


@dataclass
class DemonExplanation:
    """Explanation of why a categorical operation appears as demon behavior"""
    operation: str
    kinetic_observation: str
    apparent_intelligence: str
    actual_mechanism: str
    dissolution_argument: DissolutionArgument

