"""
Categorical Molecular Dynamics: How Molecules Move in S-Space
=============================================================

Molecules in the categorical gas don't move through physical space.
They move through S-entropy space.

Movement is driven by:
1. Hardware oscillation changes (new timing → new S-coordinates)
2. Harmonic coincidences (molecules that resonate can "interact")
3. Categorical navigation (the demon/spectrometer moving to different coordinates)

This is NOT physics simulation. This is categorical dynamics.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    from .virtual_molecule import VirtualMolecule, SCoordinate
except ImportError:
    from virtual_molecule import VirtualMolecule, SCoordinate


class InteractionType(Enum):
    """Types of categorical interactions."""
    NONE = "none"
    HARMONIC = "harmonic_coincidence"
    RESONANCE = "resonance"
    COLLISION = "categorical_collision"


@dataclass
class CategoricalTrajectory:
    """A trajectory through S-space."""
    points: List[SCoordinate] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    def add_point(self, coord: SCoordinate, timestamp: Optional[float] = None):
        self.points.append(coord)
        self.timestamps.append(timestamp or time.perf_counter())
    
    @property
    def length(self) -> int:
        return len(self.points)
    
    @property
    def total_distance(self) -> float:
        """Total S-distance traveled."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i-1].distance_to(self.points[i])
        return total
    
    @property
    def mean_velocity(self) -> float:
        """Mean velocity in S-space (distance/time)."""
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt == 0:
            return 0.0
        return self.total_distance / dt


@dataclass
class CategoricalInteraction:
    """Record of an interaction between molecules."""
    molecule1: VirtualMolecule
    molecule2: VirtualMolecule
    interaction_type: InteractionType
    strength: float
    s_distance: float
    harmonic_order: Optional[Tuple[int, int]] = None


class CategoricalDynamics:
    """
    Manages molecular dynamics in categorical space.
    
    Unlike physical dynamics:
    - No forces, no potentials
    - Movement is through S-coordinate changes
    - Interactions are harmonic coincidences
    - "Collisions" are categorical proximity events
    """
    
    def __init__(self):
        self.trajectories: Dict[str, CategoricalTrajectory] = {}
        self.interactions: List[CategoricalInteraction] = []
    
    def track(self, molecule: VirtualMolecule) -> None:
        """Start tracking a molecule's trajectory."""
        mol_id = molecule.identity
        if mol_id not in self.trajectories:
            self.trajectories[mol_id] = CategoricalTrajectory()
        self.trajectories[mol_id].add_point(molecule.s_coord)
    
    def update_position(self, molecule: VirtualMolecule, 
                       new_coord: SCoordinate) -> None:
        """Record a position update for a molecule."""
        mol_id = molecule.identity
        if mol_id not in self.trajectories:
            self.track(molecule)
        self.trajectories[mol_id].add_point(new_coord)
    
    def check_interaction(self, mol1: VirtualMolecule, 
                         mol2: VirtualMolecule) -> CategoricalInteraction:
        """
        Check for categorical interaction between two molecules.
        
        Interaction occurs through harmonic coincidence:
        If n·f1 ≈ m·f2 for small integers n, m, they can "interact".
        """
        s_distance = mol1.s_coord.distance_to(mol2.s_coord)
        
        # Check harmonic coincidence
        if mol1.frequency > 0 and mol2.frequency > 0:
            ratio = mol1.frequency / mol2.frequency
            best_harmonic = self._find_best_harmonic(ratio)
            
            if best_harmonic:
                n, m, strength = best_harmonic
                interaction_type = InteractionType.HARMONIC
                return CategoricalInteraction(
                    molecule1=mol1,
                    molecule2=mol2,
                    interaction_type=interaction_type,
                    strength=strength,
                    s_distance=s_distance,
                    harmonic_order=(n, m)
                )
        
        # Check for categorical collision (very close in S-space)
        if s_distance < 0.01:
            return CategoricalInteraction(
                molecule1=mol1,
                molecule2=mol2,
                interaction_type=InteractionType.COLLISION,
                strength=1.0 - s_distance * 100,
                s_distance=s_distance
            )
        
        return CategoricalInteraction(
            molecule1=mol1,
            molecule2=mol2,
            interaction_type=InteractionType.NONE,
            strength=0.0,
            s_distance=s_distance
        )
    
    def _find_best_harmonic(self, ratio: float, max_order: int = 10
                           ) -> Optional[Tuple[int, int, float]]:
        """Find the best harmonic match for a frequency ratio."""
        best_match = None
        best_strength = 0.0
        
        for n in range(1, max_order + 1):
            for m in range(1, max_order + 1):
                harmonic = n / m
                error = abs(ratio - harmonic) / harmonic
                
                if error < 0.05:  # Within 5%
                    strength = 1.0 / (n + m)  # Lower harmonics are stronger
                    if strength > best_strength:
                        best_strength = strength
                        best_match = (n, m, strength)
        
        return best_match
    
    def find_all_interactions(self, molecules: List[VirtualMolecule]
                             ) -> List[CategoricalInteraction]:
        """Find all pairwise interactions in a set of molecules."""
        interactions = []
        for i, mol1 in enumerate(molecules):
            for mol2 in molecules[i+1:]:
                interaction = self.check_interaction(mol1, mol2)
                if interaction.interaction_type != InteractionType.NONE:
                    interactions.append(interaction)
        
        self.interactions.extend(interactions)
        return interactions
    
    def diffusion_coefficient(self, mol_id: str) -> float:
        """
        Calculate the categorical diffusion coefficient.
        
        This measures how much a molecule spreads through S-space.
        Higher = more "thermally active" in the categorical sense.
        """
        if mol_id not in self.trajectories:
            return 0.0
        
        traj = self.trajectories[mol_id]
        if traj.length < 3:
            return 0.0
        
        # Mean squared displacement
        origin = traj.points[0]
        msd = sum(origin.distance_to(p)**2 for p in traj.points) / traj.length
        
        # Time span
        dt = traj.timestamps[-1] - traj.timestamps[0]
        if dt == 0:
            return 0.0
        
        # D = MSD / (6 * t) for 3D diffusion
        return msd / (6 * dt)
    
    def categorical_velocity(self, mol_id: str) -> SCoordinate:
        """
        Calculate the velocity vector in S-space.
        
        Returns the rate of change in each S-coordinate.
        """
        if mol_id not in self.trajectories:
            return SCoordinate(0, 0, 0)
        
        traj = self.trajectories[mol_id]
        if traj.length < 2:
            return SCoordinate(0, 0, 0)
        
        dt = traj.timestamps[-1] - traj.timestamps[-2]
        if dt == 0:
            return SCoordinate(0, 0, 0)
        
        p1 = traj.points[-2]
        p2 = traj.points[-1]
        
        return SCoordinate(
            (p2.S_k - p1.S_k) / dt,
            (p2.S_t - p1.S_t) / dt,
            (p2.S_e - p1.S_e) / dt
        )
    
    def predict_position(self, mol_id: str, 
                        time_ahead: float) -> Optional[SCoordinate]:
        """
        Predict where a molecule will be in S-space.
        
        This is categorical completion - predicting the trajectory endpoint.
        """
        if mol_id not in self.trajectories:
            return None
        
        traj = self.trajectories[mol_id]
        if traj.length < 2:
            return None
        
        velocity = self.categorical_velocity(mol_id)
        current = traj.points[-1]
        
        # Linear extrapolation in S-space
        predicted = SCoordinate(
            current.S_k + velocity.S_k * time_ahead,
            current.S_t + velocity.S_t * time_ahead,
            current.S_e + velocity.S_e * time_ahead
        )
        
        return predicted


def demonstrate_dynamics():
    """Demonstrate categorical molecular dynamics."""
    from .virtual_molecule import VirtualMolecule
    
    print("=== CATEGORICAL DYNAMICS DEMONSTRATION ===\n")
    
    dynamics = CategoricalDynamics()
    
    # Create molecules and track them
    molecules = []
    for i in range(5):
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        delta = (t2 - t1) * 1e-9
        mol = VirtualMolecule.from_hardware_timing(delta)
        molecules.append(mol)
        dynamics.track(mol)
        time.sleep(0.001)
    
    print(f"Created {len(molecules)} molecules\n")
    
    # Find interactions
    interactions = dynamics.find_all_interactions(molecules)
    print(f"Found {len(interactions)} interactions:")
    for interaction in interactions[:3]:  # Show first 3
        print(f"  {interaction.interaction_type.value}: "
              f"strength={interaction.strength:.3f}, "
              f"distance={interaction.s_distance:.4f}")
    
    # Calculate dynamics properties
    print("\n--- Trajectory Analysis ---")
    for mol in molecules[:2]:  # Show first 2
        mol_id = mol.identity
        traj = dynamics.trajectories[mol_id]
        print(f"Molecule {mol_id[:8]}...")
        print(f"  Trajectory length: {traj.length}")
        print(f"  Total S-distance: {traj.total_distance:.6f}")
    
    print("\n=== KEY INSIGHT ===")
    print("Molecules don't move through physical space.")
    print("They move through S-entropy coordinate space.")
    print("'Interactions' are harmonic coincidences, not collisions.")
    
    return dynamics


if __name__ == "__main__":
    demonstrate_dynamics()

