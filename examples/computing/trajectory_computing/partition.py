"""
Partition Structures and Coordinates

Partition coordinates (n, l, m, s) parameterize bounded oscillatory systems:
- n: principal partition depth (radial nesting level)
- l: angular complexity (number of angular nodes), l ∈ {0, 1, ..., n-1}
- m: orientation (spatial arrangement), m ∈ {-l, ..., +l}
- s: chirality (boundary handedness), s ∈ {-1/2, +1/2}

Capacity at depth n: 2n² distinguishable states

These coordinates are not imposed but emerge necessarily from sequential
partitioning of bounded oscillatory systems.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
from enum import Enum
import numpy as np

from .coordinates import SCoord, TritAddress


class Spin(Enum):
    """Boundary chirality (spin)."""
    DOWN = -0.5
    UP = 0.5


@dataclass(frozen=True)
class PartitionCoordinates:
    """
    Partition coordinates (n, l, m, s).

    These emerge from geometric constraints on nested oscillatory boundaries.
    The capacity formula 2n² follows immediately from the constraints.
    """
    n: int   # Principal depth, n ≥ 1
    l: int   # Angular complexity, 0 ≤ l ≤ n-1
    m: int   # Orientation, -l ≤ m ≤ +l
    s: Spin  # Chirality

    def __post_init__(self):
        """Validate coordinate constraints."""
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")
        if not 0 <= self.l <= self.n - 1:
            raise ValueError(f"l must be in [0, {self.n-1}], got {self.l}")
        if not -self.l <= self.m <= self.l:
            raise ValueError(f"m must be in [{-self.l}, {self.l}], got {self.m}")

    @staticmethod
    def capacity(n: int) -> int:
        """
        Capacity at depth n: 2n² distinguishable states.

        This is not fitted to data but derived from geometry:
        sum over l of (2l+1) states, times 2 for spin.
        """
        return 2 * n * n

    @staticmethod
    def enumerate_at_depth(n: int) -> List[PartitionCoordinates]:
        """Enumerate all 2n² states at depth n."""
        states = []
        for l in range(n):
            for m in range(-l, l + 1):
                for s in [Spin.DOWN, Spin.UP]:
                    states.append(PartitionCoordinates(n=n, l=l, m=m, s=s))
        return states

    def to_index(self) -> int:
        """
        Convert to linear index within depth n.

        Index = 2 * (sum_{l'<l} (2l'+1) + (m + l)) + (1 if s=UP else 0)
        """
        # States before this l value
        before_l = sum(2 * l_prime + 1 for l_prime in range(self.l))
        # Position within this l
        within_l = self.m + self.l
        # Spin contribution
        spin_offset = 1 if self.s == Spin.UP else 0
        return 2 * (before_l + within_l) + spin_offset

    @classmethod
    def from_index(cls, n: int, index: int) -> PartitionCoordinates:
        """Create from linear index at depth n."""
        if index < 0 or index >= cls.capacity(n):
            raise ValueError(f"Index {index} out of range for n={n}")

        spin_offset = index % 2
        s = Spin.UP if spin_offset else Spin.DOWN
        remaining = index // 2

        # Find l
        l = 0
        cumulative = 0
        while cumulative + (2 * l + 1) <= remaining:
            cumulative += 2 * l + 1
            l += 1

        # Find m
        m = remaining - cumulative - l

        return cls(n=n, l=l, m=m, s=s)

    def energy_ordering(self, alpha: float = 1.0) -> float:
        """
        Energy ordering: n + αl.

        This emerges from variational principles, with α ≈ 1.
        Reproduces the Aufbau filling principle.
        """
        return self.n + alpha * self.l

    def allowed_transitions(self) -> List[PartitionCoordinates]:
        """
        Selection rules: Δl = ±1, Δm ∈ {0, ±1}.

        These emerge from continuity requirements on oscillatory modes.
        """
        transitions = []
        for delta_l in [-1, 1]:
            new_l = self.l + delta_l
            if new_l < 0:
                continue
            for delta_m in [-1, 0, 1]:
                new_m = self.m + delta_m
                if not -new_l <= new_m <= new_l:
                    continue
                # Transitions can change n and s freely
                for new_n in [self.n - 1, self.n, self.n + 1]:
                    if new_n < 1 or new_l > new_n - 1:
                        continue
                    for new_s in [Spin.DOWN, Spin.UP]:
                        transitions.append(PartitionCoordinates(
                            n=new_n, l=new_l, m=new_m, s=new_s
                        ))
        return transitions


@dataclass
class Partition:
    """
    A partition state in S-entropy space.

    Combines partition coordinates with S-coordinates and completion status.
    """
    id: int
    coordinates: PartitionCoordinates
    s_coord: SCoord
    completed: bool = False
    accessible: Set[int] = field(default_factory=set)
    phase_locks: Dict[int, float] = field(default_factory=dict)  # id -> coupling
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.coordinates.n

    @property
    def l(self) -> int:
        return self.coordinates.l

    @property
    def m(self) -> int:
        return self.coordinates.m

    @property
    def s(self) -> Spin:
        return self.coordinates.s

    def complete(self) -> None:
        """
        Complete this partition state (irreversible).

        Once completed, a categorical state cannot be uncompleted.
        This is the source of the arrow of time.
        """
        if self.completed:
            raise RuntimeError(f"Partition {self.id} already completed - categorical completion is irreversible")
        self.completed = True

    def can_access(self, other_id: int) -> bool:
        """Check if another partition is accessible."""
        return other_id in self.accessible

    def add_phase_lock(self, other_id: int, coupling: float) -> None:
        """Add phase-lock coupling to another partition."""
        self.phase_locks[other_id] = coupling
        self.accessible.add(other_id)

    def total_coupling(self) -> float:
        """Total phase-lock coupling strength."""
        return sum(self.phase_locks.values())


class PartitionSpace:
    """
    Manager for a collection of partitions forming a phase space.

    Handles creation, navigation, and completion of partition states.
    """

    def __init__(self):
        self.partitions: Dict[int, Partition] = {}
        self._next_id: int = 0
        self._completed_count: int = 0

    def create(self, coords: PartitionCoordinates,
               s_coord: Optional[SCoord] = None) -> Partition:
        """Create a new partition state."""
        if s_coord is None:
            # Default S-coordinate from partition coordinates
            # Maps (n, l, m, s) to [0,1]^3
            s_coord = SCoord(
                s_k=coords.l / max(coords.n, 1),
                s_t=(coords.m + coords.l) / max(2 * coords.l + 1, 1),
                s_e=0.5 + coords.s.value
            )

        partition = Partition(
            id=self._next_id,
            coordinates=coords,
            s_coord=s_coord
        )
        self.partitions[self._next_id] = partition
        self._next_id += 1
        return partition

    def get(self, partition_id: int) -> Optional[Partition]:
        """Get partition by ID."""
        return self.partitions.get(partition_id)

    def complete(self, partition_id: int) -> None:
        """Complete a partition state."""
        partition = self.partitions.get(partition_id)
        if partition is None:
            raise ValueError(f"Partition {partition_id} not found")
        partition.complete()
        self._completed_count += 1

    def form_phase_lock(self, id_a: int, id_b: int, coupling: float) -> None:
        """Form bidirectional phase-lock between partitions."""
        part_a = self.partitions.get(id_a)
        part_b = self.partitions.get(id_b)
        if part_a is None or part_b is None:
            raise ValueError("Both partitions must exist")
        part_a.add_phase_lock(id_b, coupling)
        part_b.add_phase_lock(id_a, coupling)

    def accessible_from(self, partition_id: int) -> List[Partition]:
        """Get all partitions accessible from given partition."""
        partition = self.partitions.get(partition_id)
        if partition is None:
            return []
        return [self.partitions[pid] for pid in partition.accessible
                if pid in self.partitions]

    def incomplete_accessible(self, partition_id: int) -> List[Partition]:
        """Get incomplete partitions accessible from given partition."""
        return [p for p in self.accessible_from(partition_id) if not p.completed]

    @property
    def completed_count(self) -> int:
        """Number of completed partitions."""
        return self._completed_count

    @property
    def total_count(self) -> int:
        """Total number of partitions."""
        return len(self.partitions)

    def completion_ratio(self) -> float:
        """Ratio of completed to total partitions."""
        if self.total_count == 0:
            return 0.0
        return self._completed_count / self.total_count


# Demonstration
if __name__ == "__main__":
    # Verify capacity formula
    print("Capacity formula 2n² verification:")
    for n in range(1, 6):
        expected = PartitionCoordinates.capacity(n)
        actual = len(PartitionCoordinates.enumerate_at_depth(n))
        print(f"  n={n}: 2n²={expected}, enumerated={actual}, match={expected==actual}")

    # Show correspondence with electron shells
    print("\nCorrespondence with atomic electron shells:")
    shell_names = ['K', 'L', 'M', 'N', 'O']
    for n in range(1, 6):
        cap = PartitionCoordinates.capacity(n)
        print(f"  n={n} ({shell_names[n-1]} shell): capacity = {cap} electrons")

    # Selection rules
    print("\nSelection rules (Δl = ±1) from (n=3, l=1, m=0, s=↑):")
    state = PartitionCoordinates(n=3, l=1, m=0, s=Spin.UP)
    transitions = state.allowed_transitions()
    print(f"  Allowed transitions: {len(transitions)}")
    for t in transitions[:5]:
        print(f"    → (n={t.n}, l={t.l}, m={t.m}, s={'↑' if t.s==Spin.UP else '↓'})")
