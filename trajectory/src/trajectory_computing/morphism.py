"""
Morphisms and Catalysts

A morphism is a structure-preserving map between partition states.
In trajectory computing, morphisms represent valid transitions that
preserve categorical relationships.

A catalyst is an intermediate partition stage that enables transitions
that would otherwise require too many categorical steps. Catalysts
don't change the final state - they reduce the categorical distance
to make navigation feasible.

This is analogous to chemical catalysis, but at the categorical level.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Set
from enum import Enum
import numpy as np

from .coordinates import SCoord, TritAddress, Trit
from .partition import Partition, PartitionCoordinates, PartitionSpace


class MorphismType(Enum):
    """Types of categorical morphisms."""
    IDENTITY = "identity"        # No change
    REFINEMENT = "refinement"    # Deeper partition
    COARSENING = "coarsening"    # Shallower partition
    ROTATION = "rotation"        # Change orientation (m)
    TRANSITION = "transition"    # Change angular (l)
    COMPLETION = "completion"    # Mark as completed


@dataclass
class Morphism:
    """
    A structure-preserving map between partition states.

    Morphisms are the valid moves in trajectory navigation.
    They preserve categorical relationships (constraints).
    """
    source_id: int
    target_id: int
    morphism_type: MorphismType
    delta_n: int = 0     # Change in principal depth
    delta_l: int = 0     # Change in angular complexity
    delta_m: int = 0     # Change in orientation
    preserves: List[str] = field(default_factory=list)  # What's preserved

    def __post_init__(self):
        """Validate morphism constraints."""
        # Selection rules: |Δl| ≤ 1 for allowed transitions
        if abs(self.delta_l) > 1 and self.morphism_type == MorphismType.TRANSITION:
            raise ValueError(f"Selection rule violation: Δl = {self.delta_l}")

    @property
    def categorical_distance(self) -> int:
        """Number of categorical steps this morphism represents."""
        return abs(self.delta_n) + abs(self.delta_l) + abs(self.delta_m)

    def is_identity(self) -> bool:
        """Check if this is an identity morphism."""
        return self.categorical_distance == 0

    @classmethod
    def identity(cls, state_id: int) -> Morphism:
        """Create identity morphism."""
        return cls(
            source_id=state_id,
            target_id=state_id,
            morphism_type=MorphismType.IDENTITY,
            preserves=["all"]
        )

    @classmethod
    def refinement(cls, source_id: int, target_id: int) -> Morphism:
        """Create refinement morphism (increase depth)."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            morphism_type=MorphismType.REFINEMENT,
            delta_n=1,
            preserves=["ancestry"]
        )

    @classmethod
    def from_coordinates(cls, source: PartitionCoordinates,
                        target: PartitionCoordinates,
                        source_id: int = 0,
                        target_id: int = 0) -> Morphism:
        """Create morphism from coordinate change."""
        delta_n = target.n - source.n
        delta_l = target.l - source.l
        delta_m = target.m - source.m

        # Determine type
        if delta_n == 0 and delta_l == 0 and delta_m == 0:
            mtype = MorphismType.IDENTITY
        elif delta_n > 0:
            mtype = MorphismType.REFINEMENT
        elif delta_n < 0:
            mtype = MorphismType.COARSENING
        elif delta_l != 0:
            mtype = MorphismType.TRANSITION
        else:
            mtype = MorphismType.ROTATION

        return cls(
            source_id=source_id,
            target_id=target_id,
            morphism_type=mtype,
            delta_n=delta_n,
            delta_l=delta_l,
            delta_m=delta_m
        )


@dataclass
class MorphismChain:
    """
    A sequence of morphisms forming a trajectory.

    The composition of morphisms along a path.
    """
    morphisms: List[Morphism] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.morphisms)

    @property
    def source_id(self) -> Optional[int]:
        """Starting state ID."""
        return self.morphisms[0].source_id if self.morphisms else None

    @property
    def target_id(self) -> Optional[int]:
        """Ending state ID."""
        return self.morphisms[-1].target_id if self.morphisms else None

    @property
    def total_distance(self) -> int:
        """Total categorical distance."""
        return sum(m.categorical_distance for m in self.morphisms)

    def append(self, morphism: Morphism) -> None:
        """Add a morphism to the chain."""
        if self.morphisms and morphism.source_id != self.morphisms[-1].target_id:
            raise ValueError("Morphism chain discontinuity")
        self.morphisms.append(morphism)

    def is_valid_chain(self) -> bool:
        """Check if morphisms form a valid chain."""
        for i in range(len(self.morphisms) - 1):
            if self.morphisms[i].target_id != self.morphisms[i + 1].source_id:
                return False
        return True

    def compress(self) -> Morphism:
        """Compose all morphisms into a single morphism."""
        if not self.morphisms:
            raise ValueError("Cannot compress empty chain")

        total_delta_n = sum(m.delta_n for m in self.morphisms)
        total_delta_l = sum(m.delta_l for m in self.morphisms)
        total_delta_m = sum(m.delta_m for m in self.morphisms)

        return Morphism(
            source_id=self.source_id,
            target_id=self.target_id,
            morphism_type=MorphismType.TRANSITION,
            delta_n=total_delta_n,
            delta_l=total_delta_l,
            delta_m=total_delta_m
        )


@dataclass
class CatalystStage:
    """
    An intermediate partition stage in a catalyst chain.

    Represents a waypoint that reduces categorical distance.
    """
    partition: Partition
    entry_morphism: Optional[Morphism] = None
    exit_morphism: Optional[Morphism] = None
    residence_requirement: int = 1  # Minimum steps at this stage

    @property
    def coordinates(self) -> PartitionCoordinates:
        return self.partition.coordinates


@dataclass
class Catalyst:
    """
    A catalyst chain enabling difficult transitions.

    Instead of a direct transition requiring N categorical steps,
    a catalyst provides intermediate stages such that each step
    requires only 1 categorical step.

    Example: To go from (n=1, l=0) to (n=4, l=2), direct transition
    is forbidden (Δl = 2 violates selection rules). But catalyst
    chain: (1,0) → (2,1) → (3,2) → (4,2) works.
    """
    stages: List[CatalystStage] = field(default_factory=list)
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stages)

    @property
    def depth(self) -> int:
        """Number of intermediate stages."""
        return len(self.stages)

    def add_stage(self, partition: Partition) -> None:
        """Add an intermediate stage."""
        stage = CatalystStage(partition=partition)
        if self.stages:
            # Create morphisms connecting stages
            prev_stage = self.stages[-1]
            morphism = Morphism.from_coordinates(
                prev_stage.coordinates,
                partition.coordinates,
                prev_stage.partition.id,
                partition.id
            )
            prev_stage.exit_morphism = morphism
            stage.entry_morphism = morphism

        self.stages.append(stage)

    def is_valid_chain(self) -> bool:
        """
        Validate that all transitions obey selection rules.

        Each step should have |Δl| ≤ 1.
        """
        for i in range(len(self.stages) - 1):
            c1 = self.stages[i].coordinates
            c2 = self.stages[i + 1].coordinates
            if abs(c2.l - c1.l) > 1:
                return False
        return True

    def categorical_path_length(self) -> int:
        """Total categorical distance along the chain."""
        if len(self.stages) < 2:
            return 0

        total = 0
        for i in range(len(self.stages) - 1):
            m = Morphism.from_coordinates(
                self.stages[i].coordinates,
                self.stages[i + 1].coordinates
            )
            total += m.categorical_distance

        return total

    def direct_distance(self) -> int:
        """Direct categorical distance without catalyst."""
        if len(self.stages) < 2:
            return 0

        m = Morphism.from_coordinates(
            self.stages[0].coordinates,
            self.stages[-1].coordinates
        )
        return m.categorical_distance

    def catalytic_efficiency(self) -> float:
        """
        Ratio of direct distance to path length.

        > 1 means catalyst provides advantage.
        """
        path_len = self.categorical_path_length()
        if path_len == 0:
            return 1.0
        return self.direct_distance() / path_len

    @classmethod
    def build_chain(cls, start: PartitionCoordinates,
                   end: PartitionCoordinates,
                   space: PartitionSpace) -> Catalyst:
        """
        Build a catalyst chain from start to end coordinates.

        Uses greedy approach: at each step, move toward target
        while respecting selection rules (|Δl| ≤ 1).
        """
        catalyst = cls(name=f"{start} → {end}")

        current = start
        start_partition = space.create(current)
        catalyst.add_stage(start_partition)

        while (current.n != end.n or current.l != end.l or
               current.m != end.m):

            # Determine next step
            next_n = current.n
            next_l = current.l
            next_m = current.m

            # Move toward target, respecting constraints
            if current.n < end.n:
                next_n = current.n + 1
            elif current.n > end.n:
                next_n = current.n - 1

            if current.l < end.l:
                next_l = min(current.l + 1, next_n - 1)  # l < n
            elif current.l > end.l:
                next_l = max(current.l - 1, 0)

            if current.m < end.m:
                next_m = min(current.m + 1, next_l)  # |m| ≤ l
            elif current.m > end.m:
                next_m = max(current.m - 1, -next_l)

            # Create next coordinates
            try:
                next_coords = PartitionCoordinates(
                    n=next_n, l=next_l, m=next_m, s=current.s
                )
                next_partition = space.create(next_coords)
                catalyst.add_stage(next_partition)
                current = next_coords
            except ValueError:
                # Invalid coordinates, adjust
                break

        return catalyst


class MorphismCategory:
    """
    The category of morphisms over a partition space.

    Objects: Partitions
    Morphisms: Structure-preserving maps
    Composition: Sequential application
    """

    def __init__(self, space: PartitionSpace):
        self.space = space
        self.morphisms: Dict[tuple, Morphism] = {}  # (src, tgt) -> morphism

    def add_morphism(self, morphism: Morphism) -> None:
        """Register a morphism."""
        key = (morphism.source_id, morphism.target_id)
        self.morphisms[key] = morphism

    def get_morphism(self, source_id: int, target_id: int) -> Optional[Morphism]:
        """Get morphism between two objects."""
        return self.morphisms.get((source_id, target_id))

    def compose(self, m1: Morphism, m2: Morphism) -> Morphism:
        """Compose two morphisms."""
        if m1.target_id != m2.source_id:
            raise ValueError("Cannot compose: target ≠ source")

        return Morphism(
            source_id=m1.source_id,
            target_id=m2.target_id,
            morphism_type=MorphismType.TRANSITION,
            delta_n=m1.delta_n + m2.delta_n,
            delta_l=m1.delta_l + m2.delta_l,
            delta_m=m1.delta_m + m2.delta_m
        )

    def identity_for(self, obj_id: int) -> Morphism:
        """Get identity morphism for an object."""
        return Morphism.identity(obj_id)

    def all_morphisms_from(self, source_id: int) -> List[Morphism]:
        """Get all morphisms from a given source."""
        return [m for (src, _), m in self.morphisms.items() if src == source_id]

    def all_morphisms_to(self, target_id: int) -> List[Morphism]:
        """Get all morphisms to a given target."""
        return [m for (_, tgt), m in self.morphisms.items() if tgt == target_id]


# Demonstration
if __name__ == "__main__":
    from .partition import Spin

    # Create morphism from coordinate change
    c1 = PartitionCoordinates(n=2, l=1, m=0, s=Spin.UP)
    c2 = PartitionCoordinates(n=3, l=2, m=1, s=Spin.UP)

    m = Morphism.from_coordinates(c1, c2, source_id=0, target_id=1)
    print(f"Morphism: {m.morphism_type.value}")
    print(f"  Δn={m.delta_n}, Δl={m.delta_l}, Δm={m.delta_m}")
    print(f"  Categorical distance: {m.categorical_distance}")

    # Build catalyst chain
    space = PartitionSpace()
    start = PartitionCoordinates(n=1, l=0, m=0, s=Spin.UP)
    end = PartitionCoordinates(n=4, l=2, m=1, s=Spin.UP)

    print(f"\nBuilding catalyst chain from {start} to {end}:")
    catalyst = Catalyst.build_chain(start, end, space)
    print(f"  Stages: {len(catalyst)}")
    print(f"  Valid chain: {catalyst.is_valid_chain()}")
    print(f"  Path length: {catalyst.categorical_path_length()}")
    print(f"  Direct distance: {catalyst.direct_distance()}")

    for i, stage in enumerate(catalyst.stages):
        c = stage.coordinates
        print(f"    Stage {i}: (n={c.n}, l={c.l}, m={c.m})")
