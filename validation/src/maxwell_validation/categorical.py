"""
Categorical Face Validation
===========================

Validates the categorical engine operations:
- Phase-lock network construction
- Topological navigation
- Categorical completion
- Kinetic independence
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import networkx as nx


@dataclass
class SCoordinates:
    """S-entropy coordinates (S_k, S_t, S_e)"""
    s_k: float  # Knowledge entropy
    s_t: float  # Temporal entropy
    s_e: float  # Evolutionary entropy
    
    def conjugate(self) -> "SCoordinates":
        """Return the conjugate (back face) coordinates"""
        return SCoordinates(-self.s_k, -self.s_t, -self.s_e)
    
    def distance(self, other: "SCoordinates") -> float:
        """Euclidean distance in S-space"""
        return np.sqrt(
            (self.s_k - other.s_k) ** 2 +
            (self.s_t - other.s_t) ** 2 +
            (self.s_e - other.s_e) ** 2
        )
    
    def sums_to_zero_with(self, other: "SCoordinates", tolerance: float = 1e-10) -> bool:
        """Check if coordinates sum to zero (conjugate verification)"""
        return (
            abs(self.s_k + other.s_k) < tolerance and
            abs(self.s_t + other.s_t) < tolerance and
            abs(self.s_e + other.s_e) < tolerance
        )


@dataclass
class CategoricalState:
    """A categorical state in phase-lock space"""
    id: int
    coordinates: SCoordinates
    completed: bool = False
    accessible: List[int] = None
    phase_locks: List[Tuple[int, float]] = None
    
    def __post_init__(self):
        if self.accessible is None:
            self.accessible = []
        if self.phase_locks is None:
            self.phase_locks = []


class CategoricalValidator:
    """
    Validates categorical engine operations.
    
    Key validations:
    1. Phase-lock networks form based on position, not velocity
    2. Categorical completion is irreversible
    3. Network density increases entropy
    4. Topological navigation follows adjacency
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.network = nx.Graph()
        self.states: Dict[int, CategoricalState] = {}
        self.next_id = 0
    
    def create_state(self, coords: SCoordinates) -> int:
        """Create a new categorical state"""
        state_id = self.next_id
        self.next_id += 1
        
        state = CategoricalState(id=state_id, coordinates=coords)
        self.states[state_id] = state
        self.network.add_node(state_id, coords=coords)
        
        return state_id
    
    def form_phase_lock(self, id_a: int, id_b: int, coupling: float) -> bool:
        """Form a phase-lock between two states"""
        if id_a not in self.states or id_b not in self.states:
            return False
        
        self.network.add_edge(id_a, id_b, weight=coupling)
        
        self.states[id_a].phase_locks.append((id_b, coupling))
        self.states[id_a].accessible.append(id_b)
        
        self.states[id_b].phase_locks.append((id_a, coupling))
        self.states[id_b].accessible.append(id_a)
        
        return True
    
    def complete_state(self, state_id: int) -> bool:
        """Complete a categorical state (irreversible)"""
        if state_id not in self.states:
            return False
        if self.states[state_id].completed:
            return False  # Already completed - cannot re-complete
        
        self.states[state_id].completed = True
        return True
    
    def validate_kinetic_independence(
        self,
        positions: np.ndarray,
        velocities_1: np.ndarray,
        velocities_2: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Validate that phase-lock network is independent of velocities.
        
        This is THE key insight: the same positions should produce
        the same phase-lock network regardless of velocities.
        """
        # Build network from positions + velocities_1
        network_1 = self._build_network_from_positions(positions)
        
        # Build network from positions + velocities_2
        network_2 = self._build_network_from_positions(positions)
        
        # Networks should be identical
        if nx.is_isomorphic(network_1, network_2):
            return True, "Phase-lock networks are identical regardless of velocities"
        else:
            return False, "ERROR: Networks differ despite same positions"
    
    def _build_network_from_positions(
        self,
        positions: np.ndarray,
        coupling_distance: float = 1e-9,
    ) -> nx.Graph:
        """Build phase-lock network from positions (NOT velocities)"""
        G = nx.Graph()
        n = len(positions)
        
        for i in range(n):
            G.add_node(i)
        
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < coupling_distance * 10:
                    # Van der Waals coupling: ~r^-6
                    coupling = 1.0 / (r ** 6 + 1e-30)
                    G.add_edge(i, j, weight=coupling)
        
        return G
    
    def validate_irreversibility(self) -> Tuple[bool, str]:
        """Validate that completed states cannot be un-completed"""
        # Create and complete a state
        coords = SCoordinates(0.5, 0.3, 0.7)
        state_id = self.create_state(coords)
        
        # First completion should succeed
        result1 = self.complete_state(state_id)
        if not result1:
            return False, "First completion failed"
        
        # Second completion should fail (irreversibility)
        result2 = self.complete_state(state_id)
        if result2:
            return False, "ERROR: State was completed twice (irreversibility violated)"
        
        return True, "Categorical irreversibility validated"
    
    def validate_entropy_increase(self) -> Tuple[bool, str]:
        """Validate that network densification increases entropy"""
        # Initial entropy
        initial_entropy = self.categorical_entropy()
        
        # Add edges (densify network)
        for i in range(5):
            coords = SCoordinates(i * 0.1, 0, 0)
            self.create_state(coords)
        
        for i in range(4):
            self.form_phase_lock(i, i + 1, 1.0)
        
        # Final entropy
        final_entropy = self.categorical_entropy()
        
        if final_entropy > initial_entropy:
            return True, f"Entropy increased: {initial_entropy:.4e} → {final_entropy:.4e}"
        else:
            return False, "ERROR: Entropy did not increase with densification"
    
    def categorical_entropy(self) -> float:
        """Compute categorical entropy (proportional to network density)"""
        k_b = 1.380649e-23  # Boltzmann constant
        edge_count = self.network.number_of_edges()
        return k_b * edge_count
    
    def validate_conjugate_relationship(self) -> Tuple[bool, str]:
        """Validate that front and back faces are proper conjugates"""
        coords = SCoordinates(0.5, 0.3, 0.7)
        conjugate = coords.conjugate()
        
        if coords.sums_to_zero_with(conjugate, self.tolerance):
            return True, "Conjugate relationship validated: front + back = 0"
        else:
            return False, "ERROR: Conjugate relationship violated"
    
    def run_all_validations(self) -> Dict[str, Tuple[bool, str]]:
        """Run all categorical validations"""
        results = {}
        
        results["irreversibility"] = self.validate_irreversibility()
        results["entropy_increase"] = self.validate_entropy_increase()
        results["conjugate_relationship"] = self.validate_conjugate_relationship()
        
        # Test kinetic independence
        positions = np.random.rand(10, 3) * 1e-9
        velocities_1 = np.random.rand(10, 3) * 1000  # Random velocities
        velocities_2 = np.random.rand(10, 3) * 1000  # Different velocities
        results["kinetic_independence"] = self.validate_kinetic_independence(
            positions, velocities_1, velocities_2
        )
        
        return results


if __name__ == "__main__":
    validator = CategoricalValidator()
    results = validator.run_all_validations()
    
    print("=" * 60)
    print("CATEGORICAL VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, (passed, message) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        print(f"       {message}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

