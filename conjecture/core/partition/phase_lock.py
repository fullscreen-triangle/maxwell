"""
Phase-Lock Structures and Coupling

Phase-locks are the fundamental mechanism by which separate oscillatory
systems become correlated. A phase-lock between systems A and B means
their phases maintain a fixed relationship.

Key insight: Phase-locks form based on POSITION (Van der Waals ~r^-6,
dipole ~r^-3), NOT velocity. This is the categorical face - independent
of kinetic observations.

Phase-lock networks define accessibility: navigation can only proceed
along phase-locked adjacencies (the "door opens" metaphor).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np

from .coordinates import SCoord


class InteractionType(Enum):
    """Types of phase-lock interactions."""
    VAN_DER_WAALS = "vdw"      # ~r^-6, universal
    DIPOLE_DIPOLE = "dipole"   # ~r^-3, polar molecules
    HYDROGEN_BOND = "hbond"    # Stronger, directional
    COVALENT = "covalent"      # Strongest, shared electrons


class MolecularType(Enum):
    """Molecular polarity affecting coupling."""
    NON_POLAR = "non_polar"
    POLAR = "polar"
    IONIC = "ionic"


@dataclass
class Coupling:
    """
    Coupling between two oscillatory systems.

    Coupling strength determines phase-lock formation probability
    and navigation accessibility.
    """
    strength: float
    interaction_type: InteractionType
    distance: float = 0.0

    @staticmethod
    def van_der_waals(r: float, c6: float = 1e-77) -> Coupling:
        """
        Van der Waals coupling: C_6 / r^6.

        Universal attractive interaction between all matter.
        C_6 coefficient ~10^-77 J·m^6 for typical molecules.
        """
        if r <= 0:
            raise ValueError("Distance must be positive")
        strength = c6 / (r ** 6)
        return Coupling(
            strength=strength,
            interaction_type=InteractionType.VAN_DER_WAALS,
            distance=r
        )

    @staticmethod
    def dipole(r: float, mu: float = 1e-30) -> Coupling:
        """
        Dipole-dipole coupling: μ² / r^3.

        For polar molecules, adds to Van der Waals.
        μ is dipole moment (~10^-30 C·m for typical polar molecules).
        """
        if r <= 0:
            raise ValueError("Distance must be positive")
        strength = (mu ** 2) / (r ** 3)
        return Coupling(
            strength=strength,
            interaction_type=InteractionType.DIPOLE_DIPOLE,
            distance=r
        )

    def __add__(self, other: Coupling) -> Coupling:
        """Combine couplings (e.g., VdW + dipole)."""
        return Coupling(
            strength=self.strength + other.strength,
            interaction_type=self.interaction_type,  # Keep primary
            distance=self.distance or other.distance
        )


@dataclass
class PhaseLockNode:
    """
    Node in the phase-lock network.

    Represents an oscillator with frequency, phase, and amplitude.
    """
    id: int
    frequency: float      # Oscillation frequency (Hz)
    phase: float = 0.0    # Current phase [0, 2π)
    amplitude: float = 1.0
    molecular_type: MolecularType = MolecularType.NON_POLAR
    s_coord: Optional[SCoord] = None

    def phase_difference(self, other: PhaseLockNode) -> float:
        """Phase difference to another node."""
        diff = (self.phase - other.phase) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        return diff

    def frequency_ratio(self, other: PhaseLockNode) -> float:
        """Frequency ratio (for resonance detection)."""
        if other.frequency == 0:
            return float('inf')
        return self.frequency / other.frequency


@dataclass
class PhaseLockEdge:
    """
    Edge in the phase-lock network.

    Represents the coupling between two oscillators.
    """
    source_id: int
    target_id: int
    coupling: Coupling
    locked: bool = False  # True if phase-lock is established
    lock_ratio: Tuple[int, int] = (1, 1)  # Frequency ratio for lock

    @property
    def coupling_strength(self) -> float:
        return self.coupling.strength


class PhaseLockNetwork:
    """
    Network of phase-locked oscillators.

    This is the categorical face of the system - the structure that
    determines accessibility and navigation possibilities, independent
    of kinetic state.
    """

    def __init__(self, coupling_threshold: float = 1e-30):
        self.nodes: Dict[int, PhaseLockNode] = {}
        self.edges: Dict[Tuple[int, int], PhaseLockEdge] = {}
        self.adjacency: Dict[int, Set[int]] = {}
        self.coupling_threshold = coupling_threshold
        self._next_id = 0

    def add_node(self, frequency: float,
                 s_coord: Optional[SCoord] = None,
                 molecular_type: MolecularType = MolecularType.NON_POLAR) -> int:
        """Add an oscillator node to the network."""
        node_id = self._next_id
        self._next_id += 1

        self.nodes[node_id] = PhaseLockNode(
            id=node_id,
            frequency=frequency,
            molecular_type=molecular_type,
            s_coord=s_coord
        )
        self.adjacency[node_id] = set()

        return node_id

    def add_edge(self, source_id: int, target_id: int, coupling: Coupling) -> bool:
        """
        Add a phase-lock edge between two nodes.

        Returns True if the coupling exceeds threshold and edge was added.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist")

        if coupling.strength < self.coupling_threshold:
            return False

        # Add bidirectional edges
        edge_forward = PhaseLockEdge(
            source_id=source_id,
            target_id=target_id,
            coupling=coupling
        )
        edge_backward = PhaseLockEdge(
            source_id=target_id,
            target_id=source_id,
            coupling=coupling
        )

        self.edges[(source_id, target_id)] = edge_forward
        self.edges[(target_id, source_id)] = edge_backward

        self.adjacency[source_id].add(target_id)
        self.adjacency[target_id].add(source_id)

        return True

    def get_neighbors(self, node_id: int) -> Set[int]:
        """Get all nodes adjacent to the given node."""
        return self.adjacency.get(node_id, set())

    def coupling_between(self, id_a: int, id_b: int) -> Optional[Coupling]:
        """Get the coupling between two nodes."""
        edge = self.edges.get((id_a, id_b))
        return edge.coupling if edge else None

    def is_accessible(self, from_id: int, to_id: int) -> bool:
        """Check if navigation from one node to another is possible."""
        return to_id in self.adjacency.get(from_id, set())

    def path_exists(self, from_id: int, to_id: int) -> bool:
        """Check if any path exists between two nodes (BFS)."""
        if from_id == to_id:
            return True

        visited = {from_id}
        queue = [from_id]

        while queue:
            current = queue.pop(0)
            for neighbor in self.adjacency.get(current, set()):
                if neighbor == to_id:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def shortest_path(self, from_id: int, to_id: int) -> Optional[List[int]]:
        """Find shortest path between two nodes (BFS)."""
        if from_id == to_id:
            return [from_id]

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self.adjacency.get(current, set()):
                if neighbor == to_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def strongest_path(self, from_id: int, to_id: int) -> Optional[List[int]]:
        """
        Find the path with maximum minimum coupling (widest path).

        This is the optimal navigation path - following strongest
        phase-lock connections.
        """
        if from_id == to_id:
            return [from_id]

        # Dijkstra variant for widest path
        max_coupling = {from_id: float('inf')}
        predecessors = {from_id: None}
        unvisited = set(self.nodes.keys())

        while unvisited:
            # Find unvisited node with maximum coupling
            current = None
            current_coupling = -1
            for node in unvisited:
                if node in max_coupling and max_coupling[node] > current_coupling:
                    current = node
                    current_coupling = max_coupling[node]

            if current is None or current == to_id:
                break

            unvisited.remove(current)

            for neighbor in self.adjacency.get(current, set()):
                if neighbor in unvisited:
                    edge = self.edges.get((current, neighbor))
                    if edge:
                        path_coupling = min(current_coupling, edge.coupling_strength)
                        if path_coupling > max_coupling.get(neighbor, -1):
                            max_coupling[neighbor] = path_coupling
                            predecessors[neighbor] = current

        # Reconstruct path
        if to_id not in predecessors:
            return None

        path = []
        current = to_id
        while current is not None:
            path.append(current)
            current = predecessors[current]

        return list(reversed(path))

    def network_density(self) -> float:
        """Compute network density (edges / possible edges)."""
        n = len(self.nodes)
        if n < 2:
            return 0.0
        possible = n * (n - 1)
        actual = len(self.edges)
        return actual / possible

    def clustering_coefficient(self, node_id: int) -> float:
        """Compute local clustering coefficient for a node."""
        neighbors = self.adjacency.get(node_id, set())
        k = len(neighbors)
        if k < 2:
            return 0.0

        # Count edges between neighbors
        neighbor_edges = 0
        neighbors_list = list(neighbors)
        for i, n1 in enumerate(neighbors_list):
            for n2 in neighbors_list[i+1:]:
                if n2 in self.adjacency.get(n1, set()):
                    neighbor_edges += 1

        possible = k * (k - 1) / 2
        return neighbor_edges / possible

    def average_clustering(self) -> float:
        """Compute average clustering coefficient."""
        if not self.nodes:
            return 0.0
        coefficients = [self.clustering_coefficient(nid) for nid in self.nodes]
        return sum(coefficients) / len(coefficients)

    @classmethod
    def from_positions(cls, positions: List[np.ndarray],
                      molecular_types: Optional[List[MolecularType]] = None,
                      base_frequency: float = 1e12,
                      coupling_distance: float = 1e-9) -> PhaseLockNetwork:
        """
        Construct network from spatial positions.

        Phase-locks form based on position (NOT velocity) - this is
        the key insight that separates categorical from kinetic.
        """
        network = cls()
        n = len(positions)

        if molecular_types is None:
            molecular_types = [MolecularType.NON_POLAR] * n

        # Create nodes
        node_ids = []
        for i, pos in enumerate(positions):
            s_coord = SCoord(
                s_k=np.clip(pos[0] / coupling_distance, 0, 1),
                s_t=np.clip(pos[1] / coupling_distance, 0, 1),
                s_e=np.clip(pos[2] / coupling_distance, 0, 1)
            )
            freq = base_frequency * (1.0 + np.linalg.norm(pos) / coupling_distance)
            node_id = network.add_node(freq, s_coord, molecular_types[i])
            node_ids.append(node_id)

        # Form phase-locks based on positions
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < coupling_distance * 10:
                    # Van der Waals (always present)
                    coupling = Coupling.van_der_waals(r)

                    # Add dipole if either is polar
                    if (molecular_types[i] == MolecularType.POLAR or
                        molecular_types[j] == MolecularType.POLAR):
                        coupling = coupling + Coupling.dipole(r)

                    network.add_edge(node_ids[i], node_ids[j], coupling)

        return network


# Demonstration
if __name__ == "__main__":
    # Create a simple network
    network = PhaseLockNetwork()

    # Add oscillators
    id1 = network.add_node(1e12, SCoord(0.0, 0.0, 0.0))
    id2 = network.add_node(1e12, SCoord(0.1, 0.0, 0.0))
    id3 = network.add_node(1e12, SCoord(0.0, 0.1, 0.0))
    id4 = network.add_node(1e12, SCoord(0.1, 0.1, 0.0))

    # Form phase-locks
    network.add_edge(id1, id2, Coupling.van_der_waals(1e-9))
    network.add_edge(id2, id4, Coupling.van_der_waals(1e-9))
    network.add_edge(id1, id3, Coupling.van_der_waals(1e-9))
    network.add_edge(id3, id4, Coupling.van_der_waals(1e-9))

    print("Phase-Lock Network Demo:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Edges: {len(network.edges)}")
    print(f"  Density: {network.network_density():.3f}")
    print(f"  Avg clustering: {network.average_clustering():.3f}")

    # Find path
    path = network.shortest_path(id1, id4)
    print(f"\n  Path from {id1} to {id4}: {path}")

    # Check accessibility
    print(f"  Direct access {id1} → {id4}: {network.is_accessible(id1, id4)}")
    print(f"  Path exists {id1} → {id4}: {network.path_exists(id1, id4)}")
