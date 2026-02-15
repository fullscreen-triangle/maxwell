"""
Completion Detection and ε-Boundary

Completion is the core concept in trajectory computing:
- A completion condition specifies WHAT solutions look like
- Navigation finds trajectories TO completion
- Verification checks IF completion is satisfied

The ε-boundary is fundamental: exact closure is impossible due to
Gödelian residue (G ≡ x, the unknowable unknowable). Solutions exist
at one categorical step from closure - this is not approximation,
but the maximum possible knowledge.

Reality = ∞ - x (where x is the Gödelian residue)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Set, Union
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .coordinates import SCoord, TritAddress
from .partition import Partition, PartitionCoordinates, PartitionSpace, Spin


class CompletionStatus(Enum):
    """Status of completion check."""
    SATISFIED = "satisfied"           # Completion condition met
    UNSATISFIED = "unsatisfied"       # Not at completion
    AT_EPSILON = "at_epsilon"         # At ε-boundary (maximum knowledge)
    UNREACHABLE = "unreachable"       # Completion provably unreachable


@dataclass
class CompletionResult:
    """Result of a completion check."""
    status: CompletionStatus
    distance: float                   # Categorical distance to completion
    epsilon_distance: float           # Distance to ε-boundary
    partition: Optional[Partition] = None
    constraints_satisfied: List[str] = field(default_factory=list)
    constraints_violated: List[str] = field(default_factory=list)


class CompletionCondition(ABC):
    """
    Abstract completion condition.

    Completion conditions specify WHAT solutions look like,
    not HOW to find them. The navigator handles the "how".
    """

    @abstractmethod
    def check(self, partition: Partition) -> CompletionResult:
        """Check if partition satisfies completion condition."""
        pass

    @abstractmethod
    def epsilon_boundary(self) -> float:
        """The ε-boundary tolerance (Gödelian residue)."""
        pass

    def is_satisfied(self, partition: Partition) -> bool:
        """Simple satisfaction check."""
        result = self.check(partition)
        return result.status in [CompletionStatus.SATISFIED, CompletionStatus.AT_EPSILON]


@dataclass
class CoordinateCondition(CompletionCondition):
    """
    Completion based on partition coordinates.

    Example: Complete when reaching (n=3, l=2, m=*, s=UP)
    where * means any value.
    """
    target_n: Optional[int] = None
    target_l: Optional[int] = None
    target_m: Optional[int] = None
    target_s: Optional[Spin] = None
    epsilon: float = 0.0  # For coordinate-based, usually exact

    def check(self, partition: Partition) -> CompletionResult:
        coords = partition.coordinates
        satisfied = []
        violated = []

        distance = 0.0

        if self.target_n is not None:
            if coords.n == self.target_n:
                satisfied.append(f"n={self.target_n}")
            else:
                violated.append(f"n: expected {self.target_n}, got {coords.n}")
                distance += abs(coords.n - self.target_n)

        if self.target_l is not None:
            if coords.l == self.target_l:
                satisfied.append(f"l={self.target_l}")
            else:
                violated.append(f"l: expected {self.target_l}, got {coords.l}")
                distance += abs(coords.l - self.target_l)

        if self.target_m is not None:
            if coords.m == self.target_m:
                satisfied.append(f"m={self.target_m}")
            else:
                violated.append(f"m: expected {self.target_m}, got {coords.m}")
                distance += abs(coords.m - self.target_m)

        if self.target_s is not None:
            if coords.s == self.target_s:
                satisfied.append(f"s={self.target_s.name}")
            else:
                violated.append(f"s: expected {self.target_s.name}, got {coords.s.name}")
                distance += 1  # Spin flip is one step

        status = CompletionStatus.SATISFIED if not violated else CompletionStatus.UNSATISFIED

        return CompletionResult(
            status=status,
            distance=distance,
            epsilon_distance=max(0, distance - self.epsilon),
            partition=partition,
            constraints_satisfied=satisfied,
            constraints_violated=violated
        )

    def epsilon_boundary(self) -> float:
        return self.epsilon


@dataclass
class SCoordCondition(CompletionCondition):
    """
    Completion based on S-entropy coordinates.

    The ε-boundary is explicit here: completion when within
    ε of target S-coordinate.
    """
    target: SCoord
    epsilon: float = 0.01  # ε-boundary

    def check(self, partition: Partition) -> CompletionResult:
        distance = partition.s_coord.distance(self.target)
        epsilon_distance = max(0, distance - self.epsilon)

        if distance <= self.epsilon:
            status = CompletionStatus.AT_EPSILON
        else:
            status = CompletionStatus.UNSATISFIED

        return CompletionResult(
            status=status,
            distance=distance,
            epsilon_distance=epsilon_distance,
            partition=partition,
            constraints_satisfied=[f"distance={distance:.4f}"] if distance <= self.epsilon else [],
            constraints_violated=[f"distance={distance:.4f} > ε={self.epsilon}"] if distance > self.epsilon else []
        )

    def epsilon_boundary(self) -> float:
        return self.epsilon


@dataclass
class ConstraintCondition(CompletionCondition):
    """
    Completion based on arbitrary constraints.

    Constraints are predicates that must all be satisfied.
    This is the general form for specifying solutions.
    """
    constraints: List[Callable[[Partition], bool]] = field(default_factory=list)
    constraint_names: List[str] = field(default_factory=list)
    epsilon: float = 1.0  # Number of constraints that can be violated

    def add_constraint(self, predicate: Callable[[Partition], bool],
                      name: str = "unnamed") -> None:
        """Add a constraint."""
        self.constraints.append(predicate)
        self.constraint_names.append(name)

    def check(self, partition: Partition) -> CompletionResult:
        satisfied = []
        violated = []

        for i, (constraint, name) in enumerate(zip(self.constraints, self.constraint_names)):
            try:
                if constraint(partition):
                    satisfied.append(name)
                else:
                    violated.append(name)
            except Exception as e:
                violated.append(f"{name} (error: {e})")

        distance = len(violated)
        epsilon_distance = max(0, distance - self.epsilon)

        if distance == 0:
            status = CompletionStatus.SATISFIED
        elif distance <= self.epsilon:
            status = CompletionStatus.AT_EPSILON
        else:
            status = CompletionStatus.UNSATISFIED

        return CompletionResult(
            status=status,
            distance=distance,
            epsilon_distance=epsilon_distance,
            partition=partition,
            constraints_satisfied=satisfied,
            constraints_violated=violated
        )

    def epsilon_boundary(self) -> float:
        return self.epsilon


@dataclass
class CompositeCondition(CompletionCondition):
    """
    Composition of multiple completion conditions.

    Supports AND, OR, and NOT composition.
    """
    conditions: List[CompletionCondition] = field(default_factory=list)
    mode: str = "and"  # "and", "or"
    epsilon: float = 0.5

    def check(self, partition: Partition) -> CompletionResult:
        results = [c.check(partition) for c in self.conditions]

        if self.mode == "and":
            all_satisfied = all(
                r.status in [CompletionStatus.SATISFIED, CompletionStatus.AT_EPSILON]
                for r in results
            )
            distance = sum(r.distance for r in results)

            if all_satisfied:
                status = CompletionStatus.SATISFIED
            else:
                status = CompletionStatus.UNSATISFIED

        else:  # or
            any_satisfied = any(
                r.status in [CompletionStatus.SATISFIED, CompletionStatus.AT_EPSILON]
                for r in results
            )
            distance = min(r.distance for r in results)

            if any_satisfied:
                status = CompletionStatus.SATISFIED
            else:
                status = CompletionStatus.UNSATISFIED

        satisfied = [s for r in results for s in r.constraints_satisfied]
        violated = [v for r in results for v in r.constraints_violated]

        return CompletionResult(
            status=status,
            distance=distance,
            epsilon_distance=max(0, distance - self.epsilon),
            partition=partition,
            constraints_satisfied=satisfied,
            constraints_violated=violated
        )

    def epsilon_boundary(self) -> float:
        return self.epsilon


class CompletionDetector:
    """
    Detects completion conditions in a partition space.

    The detector is the "verification" half of "computing = verification".
    It checks which partitions satisfy given conditions.
    """

    def __init__(self, space: PartitionSpace):
        self.space = space

    def find_completions(self, condition: CompletionCondition,
                        max_results: int = 100) -> List[CompletionResult]:
        """Find all partitions satisfying the condition."""
        results = []

        for partition in self.space.partitions.values():
            result = condition.check(partition)
            if result.status in [CompletionStatus.SATISFIED, CompletionStatus.AT_EPSILON]:
                results.append(result)
                if len(results) >= max_results:
                    break

        return results

    def nearest_completion(self, partition: Partition,
                          condition: CompletionCondition) -> Optional[CompletionResult]:
        """Find the partition nearest to satisfying condition."""
        best_result = None
        best_distance = float('inf')

        for p in self.space.partitions.values():
            result = condition.check(p)
            if result.distance < best_distance:
                best_distance = result.distance
                best_result = result

        return best_result

    def completion_map(self, condition: CompletionCondition) -> Dict[int, CompletionResult]:
        """Map all partitions to their completion status."""
        return {pid: condition.check(p) for pid, p in self.space.partitions.items()}

    def epsilon_surface(self, condition: CompletionCondition) -> List[Partition]:
        """
        Find partitions on the ε-boundary.

        These are the solutions: one categorical step from closure.
        """
        results = []

        for partition in self.space.partitions.values():
            result = condition.check(partition)
            if result.status == CompletionStatus.AT_EPSILON:
                results.append(partition)

        return results


class GoedelianBoundary:
    """
    Represents the Gödelian boundary (x in ∞ - x).

    The boundary that prevents exact closure - the unknowable unknowable.
    Solutions exist at the ε-boundary, not at exact closure.
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon  # ε-boundary width

    def is_at_boundary(self, distance: float) -> bool:
        """Check if distance is at the ε-boundary."""
        return 0 < distance <= self.epsilon

    def is_beyond_boundary(self, distance: float) -> bool:
        """Check if distance exceeds the boundary."""
        return distance > self.epsilon

    def observable_reality(self, total: float = float('inf')) -> float:
        """
        Observable reality = ∞ - x.

        The Gödelian residue x prevents observing all of infinity.
        """
        return total - self.epsilon

    def maximum_knowledge(self, system_size: float) -> float:
        """
        Maximum possible knowledge about a system.

        Bounded by ε-boundary, not by computational limits.
        """
        return system_size * (1 - self.epsilon / system_size)


# Demonstration
if __name__ == "__main__":
    # Create partition space
    space = PartitionSpace()

    # Create partitions at various coordinates
    coords_list = [
        PartitionCoordinates(n=2, l=0, m=0, s=Spin.UP),
        PartitionCoordinates(n=2, l=1, m=0, s=Spin.UP),
        PartitionCoordinates(n=2, l=1, m=1, s=Spin.UP),
        PartitionCoordinates(n=3, l=2, m=0, s=Spin.UP),
        PartitionCoordinates(n=3, l=2, m=1, s=Spin.DOWN),
    ]

    for coords in coords_list:
        space.create(coords)

    # Define completion condition
    condition = CoordinateCondition(target_n=3, target_l=2)

    # Create detector
    detector = CompletionDetector(space)

    print("Completion Detection Demo:")
    print(f"  Condition: n=3, l=2")
    print(f"  Partitions in space: {len(space.partitions)}")

    # Find completions
    completions = detector.find_completions(condition)
    print(f"\n  Completions found: {len(completions)}")

    for result in completions:
        coords = result.partition.coordinates
        print(f"    (n={coords.n}, l={coords.l}, m={coords.m}, s={coords.s.name})")
        print(f"      Status: {result.status.value}")
        print(f"      Distance: {result.distance}")

    # Gödelian boundary
    goedel = GoedelianBoundary(epsilon=0.01)
    print(f"\n  Gödelian boundary (ε): {goedel.epsilon}")
    print(f"  Observable reality: ∞ - {goedel.epsilon}")
