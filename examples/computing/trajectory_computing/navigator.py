"""
Navigator: Trajectory Navigation Engine

The navigator moves through partition space toward completion conditions.
Unlike traditional algorithms that compute forward from initial conditions,
the navigator works BACKWARD from completion conditions.

Key insight: The navigator doesn't "find" solutions - it verifies that
the completion condition is reachable and traces the trajectory to it.
Computing = Verification.

Navigation strategies:
1. Greedy: Always move toward closest completion
2. Widest: Follow strongest phase-lock connections
3. Shortest: Minimize categorical distance
4. Catalyst: Use intermediate stages for difficult transitions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Set, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .coordinates import SCoord, TritAddress
from .partition import Partition, PartitionCoordinates, PartitionSpace, Spin
from .phase_lock import PhaseLockNetwork, Coupling
from .morphism import Morphism, MorphismChain, Catalyst


class NavigationStrategy(Enum):
    """Available navigation strategies."""
    GREEDY = "greedy"           # Move toward nearest completion
    WIDEST = "widest"           # Follow strongest phase-locks
    SHORTEST = "shortest"       # Minimize total categorical distance
    CATALYST = "catalyst"       # Use catalyst chains for hard transitions
    RANDOM_WALK = "random"      # Random accessible neighbor (for exploration)


@dataclass
class NavigationState:
    """Current state of navigation."""
    current_partition: Partition
    trajectory: List[int] = field(default_factory=list)
    morphism_chain: MorphismChain = field(default_factory=MorphismChain)
    steps_taken: int = 0
    backtrack_count: int = 0
    completion_distance: Optional[float] = None

    @property
    def current_id(self) -> int:
        return self.current_partition.id

    def record_step(self, new_partition: Partition, morphism: Morphism) -> None:
        """Record a navigation step."""
        self.trajectory.append(new_partition.id)
        self.morphism_chain.append(morphism)
        self.current_partition = new_partition
        self.steps_taken += 1


@dataclass
class NavigationResult:
    """Result of a navigation attempt."""
    success: bool
    final_partition: Optional[Partition]
    trajectory: List[int]
    total_steps: int
    completion_verified: bool
    categorical_distance: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompletionPredicate(ABC):
    """Abstract base for completion conditions."""

    @abstractmethod
    def is_satisfied(self, partition: Partition) -> bool:
        """Check if partition satisfies completion condition."""
        pass

    @abstractmethod
    def distance_to(self, partition: Partition) -> float:
        """Estimate categorical distance to completion."""
        pass


@dataclass
class CoordinateCompletion(CompletionPredicate):
    """Completion when reaching specific coordinates."""
    target_n: Optional[int] = None
    target_l: Optional[int] = None
    target_m: Optional[int] = None
    target_s: Optional[Spin] = None

    def is_satisfied(self, partition: Partition) -> bool:
        coords = partition.coordinates
        if self.target_n is not None and coords.n != self.target_n:
            return False
        if self.target_l is not None and coords.l != self.target_l:
            return False
        if self.target_m is not None and coords.m != self.target_m:
            return False
        if self.target_s is not None and coords.s != self.target_s:
            return False
        return True

    def distance_to(self, partition: Partition) -> float:
        coords = partition.coordinates
        distance = 0.0
        if self.target_n is not None:
            distance += abs(coords.n - self.target_n)
        if self.target_l is not None:
            distance += abs(coords.l - self.target_l)
        if self.target_m is not None:
            distance += abs(coords.m - self.target_m)
        return distance


@dataclass
class SCoordCompletion(CompletionPredicate):
    """Completion when reaching S-coordinate region."""
    target: SCoord
    epsilon: float = 0.01  # ε-boundary tolerance

    def is_satisfied(self, partition: Partition) -> bool:
        return partition.s_coord.distance(self.target) < self.epsilon

    def distance_to(self, partition: Partition) -> float:
        return partition.s_coord.distance(self.target)


@dataclass
class CustomCompletion(CompletionPredicate):
    """Custom completion predicate."""
    predicate: Callable[[Partition], bool]
    distance_fn: Callable[[Partition], float]

    def is_satisfied(self, partition: Partition) -> bool:
        return self.predicate(partition)

    def distance_to(self, partition: Partition) -> float:
        return self.distance_fn(partition)


class Navigator:
    """
    Trajectory navigation engine.

    Navigates through partition space toward completion conditions.
    The key insight: we work BACKWARD from completion, not forward
    from initial conditions.
    """

    def __init__(self, space: PartitionSpace,
                 network: Optional[PhaseLockNetwork] = None,
                 max_steps: int = 10000):
        self.space = space
        self.network = network or PhaseLockNetwork()
        self.max_steps = max_steps
        self._state: Optional[NavigationState] = None

    def navigate(self, start: Partition,
                completion: CompletionPredicate,
                strategy: NavigationStrategy = NavigationStrategy.GREEDY) -> NavigationResult:
        """
        Navigate from start toward completion condition.

        This is the core operation: trajectory completion.
        """
        self._state = NavigationState(
            current_partition=start,
            trajectory=[start.id]
        )

        while self._state.steps_taken < self.max_steps:
            current = self._state.current_partition

            # Check completion
            if completion.is_satisfied(current):
                return NavigationResult(
                    success=True,
                    final_partition=current,
                    trajectory=self._state.trajectory,
                    total_steps=self._state.steps_taken,
                    completion_verified=True,
                    categorical_distance=self._state.morphism_chain.total_distance
                )

            # Get next step based on strategy
            next_partition = self._select_next(current, completion, strategy)

            if next_partition is None:
                # Dead end - try backtracking
                next_partition = self._backtrack()
                if next_partition is None:
                    break

            # Record the step
            morphism = Morphism.from_coordinates(
                current.coordinates,
                next_partition.coordinates,
                current.id,
                next_partition.id
            )
            self._state.record_step(next_partition, morphism)
            self._state.completion_distance = completion.distance_to(next_partition)

        # Navigation failed
        return NavigationResult(
            success=False,
            final_partition=self._state.current_partition,
            trajectory=self._state.trajectory,
            total_steps=self._state.steps_taken,
            completion_verified=False,
            categorical_distance=self._state.morphism_chain.total_distance,
            metadata={"reason": "max_steps_exceeded" if self._state.steps_taken >= self.max_steps else "dead_end"}
        )

    def _select_next(self, current: Partition,
                    completion: CompletionPredicate,
                    strategy: NavigationStrategy) -> Optional[Partition]:
        """Select next partition based on strategy."""
        accessible = self._get_accessible(current)

        if not accessible:
            return None

        if strategy == NavigationStrategy.GREEDY:
            return self._greedy_select(accessible, completion)
        elif strategy == NavigationStrategy.WIDEST:
            return self._widest_select(current, accessible)
        elif strategy == NavigationStrategy.SHORTEST:
            return self._shortest_select(accessible, completion)
        elif strategy == NavigationStrategy.RANDOM_WALK:
            return self._random_select(accessible)
        else:
            return self._greedy_select(accessible, completion)

    def _get_accessible(self, partition: Partition) -> List[Partition]:
        """Get accessible partitions from current position."""
        all_accessible = {}

        # From partition's own accessibility set
        for pid in partition.accessible:
            p = self.space.get(pid)
            if p is not None:
                all_accessible[p.id] = p

        # From allowed transitions (selection rules)
        # This is the primary source - if a state exists or can be created
        for target_coords in partition.coordinates.allowed_transitions():
            existing = self._find_partition_by_coords(target_coords)
            if existing:
                all_accessible[existing.id] = existing
            else:
                # Create the partition if it doesn't exist (lazy expansion)
                new_partition = self.space.create(target_coords)
                all_accessible[new_partition.id] = new_partition

        # Filter out already visited partitions to avoid cycles
        visited = set(self._state.trajectory) if self._state else set()
        accessible_list = [p for p in all_accessible.values() if p.id not in visited]

        return accessible_list

    def _find_partition_by_coords(self, coords: PartitionCoordinates) -> Optional[Partition]:
        """Find existing partition with given coordinates."""
        for p in self.space.partitions.values():
            if (p.coordinates.n == coords.n and
                p.coordinates.l == coords.l and
                p.coordinates.m == coords.m and
                p.coordinates.s == coords.s):
                return p
        return None

    def _greedy_select(self, accessible: List[Partition],
                      completion: CompletionPredicate) -> Optional[Partition]:
        """Select partition closest to completion."""
        if not accessible:
            return None

        return min(accessible, key=lambda p: completion.distance_to(p))

    def _widest_select(self, current: Partition,
                      accessible: List[Partition]) -> Optional[Partition]:
        """Select partition with strongest phase-lock connection."""
        if not accessible:
            return None

        best = None
        best_coupling = -1.0

        for p in accessible:
            coupling = self.network.coupling_between(current.id, p.id)
            if coupling and coupling.strength > best_coupling:
                best_coupling = coupling.strength
                best = p

        return best or accessible[0]

    def _shortest_select(self, accessible: List[Partition],
                        completion: CompletionPredicate) -> Optional[Partition]:
        """Select partition minimizing total path length."""
        # For now, same as greedy - could use A* here
        return self._greedy_select(accessible, completion)

    def _random_select(self, accessible: List[Partition]) -> Optional[Partition]:
        """Random selection for exploration."""
        if not accessible:
            return None
        idx = np.random.randint(len(accessible))
        return accessible[idx]

    def _backtrack(self) -> Optional[Partition]:
        """Attempt to backtrack to try different path."""
        self._state.backtrack_count += 1

        if len(self._state.trajectory) < 2:
            return None

        # Go back one step
        self._state.trajectory.pop()
        prev_id = self._state.trajectory[-1]
        return self.space.get(prev_id)

    def verify_completion(self, partition: Partition,
                         completion: CompletionPredicate) -> bool:
        """
        Verify that a partition satisfies completion condition.

        Computing = Verification: this is the SAME operation as
        navigating to the partition.
        """
        return completion.is_satisfied(partition)

    def find_all_completions(self, completion: CompletionPredicate,
                            max_results: int = 100) -> List[Partition]:
        """Find all partitions satisfying completion condition."""
        results = []
        for partition in self.space.partitions.values():
            if completion.is_satisfied(partition):
                results.append(partition)
                if len(results) >= max_results:
                    break
        return results

    def completion_density(self, completion: CompletionPredicate) -> float:
        """Fraction of partitions satisfying completion."""
        if not self.space.partitions:
            return 0.0

        satisfied = sum(1 for p in self.space.partitions.values()
                       if completion.is_satisfied(p))
        return satisfied / len(self.space.partitions)

    def categorical_entropy(self) -> float:
        """
        Compute categorical entropy S = k_B * |E| / <E>.

        From the theory: entropy is proportional to phase-lock density.
        """
        k_b = 1.380649e-23  # Boltzmann constant
        edge_count = len(self.network.edges)
        return k_b * edge_count


class TrajectoryPlanner:
    """
    Plans trajectories using catalyst chains for difficult transitions.

    When direct navigation fails, the planner constructs intermediate
    stages that make the transition feasible.
    """

    def __init__(self, navigator: Navigator):
        self.navigator = navigator

    def plan_trajectory(self, start: PartitionCoordinates,
                       end: PartitionCoordinates) -> Catalyst:
        """
        Plan a trajectory from start to end coordinates.

        Uses catalyst chain construction to handle selection rule
        constraints.
        """
        return Catalyst.build_chain(start, end, self.navigator.space)

    def execute_plan(self, catalyst: Catalyst) -> NavigationResult:
        """Execute a planned trajectory."""
        if not catalyst.stages:
            return NavigationResult(
                success=False,
                final_partition=None,
                trajectory=[],
                total_steps=0,
                completion_verified=False,
                categorical_distance=0,
                metadata={"reason": "empty_plan"}
            )

        trajectory = []
        total_distance = 0

        for stage in catalyst.stages:
            trajectory.append(stage.partition.id)
            if stage.exit_morphism:
                total_distance += stage.exit_morphism.categorical_distance

        final = catalyst.stages[-1].partition

        return NavigationResult(
            success=True,
            final_partition=final,
            trajectory=trajectory,
            total_steps=len(catalyst.stages),
            completion_verified=True,
            categorical_distance=total_distance
        )


# Demonstration
if __name__ == "__main__":
    # Create space and navigator
    space = PartitionSpace()
    network = PhaseLockNetwork()
    navigator = Navigator(space, network)

    # Create some partitions
    coords_start = PartitionCoordinates(n=2, l=1, m=0, s=Spin.UP)
    coords_end = PartitionCoordinates(n=4, l=2, m=1, s=Spin.UP)

    start = space.create(coords_start)
    end = space.create(coords_end)

    # Create intermediate partitions for navigation path
    c1 = PartitionCoordinates(n=3, l=1, m=0, s=Spin.UP)
    c2 = PartitionCoordinates(n=3, l=2, m=0, s=Spin.UP)
    c3 = PartitionCoordinates(n=4, l=2, m=0, s=Spin.UP)

    p1 = space.create(c1)
    p2 = space.create(c2)
    p3 = space.create(c3)

    # Set up accessibility
    start.accessible = {p1.id}
    p1.accessible = {p2.id}
    p2.accessible = {p3.id}
    p3.accessible = {end.id}

    # Define completion condition
    completion = CoordinateCompletion(target_n=4, target_l=2, target_m=1)

    # Navigate
    print("Navigation Demo:")
    print(f"  Start: n={coords_start.n}, l={coords_start.l}, m={coords_start.m}")
    print(f"  Target: n={coords_end.n}, l={coords_end.l}, m={coords_end.m}")

    result = navigator.navigate(start, completion, NavigationStrategy.GREEDY)

    print(f"\n  Success: {result.success}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Trajectory: {result.trajectory}")
    print(f"  Categorical distance: {result.categorical_distance}")
