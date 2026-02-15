"""
Trajectory Computing Runtime

The runtime orchestrates trajectory completion:
1. Parse system specification
2. Build partition space and phase-lock network
3. Define completion condition
4. Navigate to completion
5. Verify solution

Key insight: The runtime doesn't "compute" solutions - it NAVIGATES to them.
Computing = Verification. The same operation that finds a solution also
verifies it.

The runtime operates on the principle that solutions must exist
(problems cannot exceed the complexity of reality), and navigation
follows Poincaré recurrence - bounded systems return arbitrarily
close to any state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
import time
import numpy as np

from .coordinates import SCoord, TritAddress, Trit
from .partition import Partition, PartitionCoordinates, PartitionSpace, Spin
from .phase_lock import PhaseLockNetwork, Coupling, MolecularType
from .morphism import Morphism, MorphismChain, Catalyst
from .navigator import (Navigator, NavigationStrategy, NavigationResult,
                        CoordinateCompletion, SCoordCompletion, TrajectoryPlanner)
from .completion import (CompletionCondition, CompletionDetector,
                        CompletionResult, CompletionStatus, GoedelianBoundary)
from .system import System, Entity, Relation, Constraint, SystemBuilder


class RuntimePhase(Enum):
    """Phases of runtime execution."""
    INITIALIZATION = "init"
    SPACE_CONSTRUCTION = "space"
    NETWORK_CONSTRUCTION = "network"
    NAVIGATION = "navigation"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RuntimeConfig:
    """Configuration for the trajectory runtime."""
    max_navigation_steps: int = 10000
    max_partitions: int = 100000
    epsilon_boundary: float = 0.01
    navigation_strategy: NavigationStrategy = NavigationStrategy.GREEDY
    enable_catalysts: bool = True
    coupling_threshold: float = 1e-30
    verbose: bool = False


@dataclass
class RuntimeMetrics:
    """Metrics collected during runtime execution."""
    phase: RuntimePhase = RuntimePhase.INITIALIZATION
    partitions_created: int = 0
    phase_locks_formed: int = 0
    navigation_steps: int = 0
    backtrack_count: int = 0
    catalyst_stages: int = 0
    completion_distance: float = float('inf')
    execution_time_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Result of trajectory computation."""
    success: bool
    solution: Optional[Partition] = None
    trajectory: List[int] = field(default_factory=list)
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    verification_result: Optional[CompletionResult] = None
    system_state: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrajectoryRuntime:
    """
    Main runtime for trajectory computing.

    The runtime orchestrates the entire process:
    1. Build the partition space from system specification
    2. Construct the phase-lock network
    3. Navigate to completion condition
    4. Verify the solution

    Computing = Verification: these are the same operation.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.space = PartitionSpace()
        self.network = PhaseLockNetwork(coupling_threshold=self.config.coupling_threshold)
        self.navigator: Optional[Navigator] = None
        self.planner: Optional[TrajectoryPlanner] = None
        self.detector: Optional[CompletionDetector] = None
        self.goedel = GoedelianBoundary(self.config.epsilon_boundary)
        self.metrics = RuntimeMetrics()
        self._system: Optional[System] = None

    def load_system(self, system: System) -> None:
        """Load a system specification."""
        self._system = system
        self.space = system.partition_space
        self._log(f"Loaded system: {system.name}")

    def initialize(self) -> None:
        """Initialize runtime components."""
        self.metrics.phase = RuntimePhase.INITIALIZATION

        self.navigator = Navigator(
            self.space,
            self.network,
            max_steps=self.config.max_navigation_steps
        )
        self.planner = TrajectoryPlanner(self.navigator)
        self.detector = CompletionDetector(self.space)

        self._log("Runtime initialized")

    def build_space(self, depth: int = 4) -> None:
        """
        Build partition space to specified depth.

        Creates 2n² partitions at each depth n from 1 to depth.
        """
        self.metrics.phase = RuntimePhase.SPACE_CONSTRUCTION
        start_time = time.time()

        for n in range(1, depth + 1):
            for coords in PartitionCoordinates.enumerate_at_depth(n):
                if len(self.space.partitions) >= self.config.max_partitions:
                    break
                self.space.create(coords)
                self.metrics.partitions_created += 1

        elapsed = (time.time() - start_time) * 1000
        self._log(f"Built partition space: {self.metrics.partitions_created} partitions in {elapsed:.1f}ms")

    def build_network(self) -> None:
        """
        Build phase-lock network from partition adjacencies.

        Phase-locks form based on categorical adjacency (position),
        NOT kinetic state (velocity).
        """
        self.metrics.phase = RuntimePhase.NETWORK_CONSTRUCTION
        start_time = time.time()

        partitions = list(self.space.partitions.values())

        for i, p1 in enumerate(partitions):
            # Add node to network
            freq = 1e12 * (p1.n + 1)  # Frequency from depth
            self.network.add_node(freq, p1.s_coord)

        # Form phase-locks based on categorical distance
        for i, p1 in enumerate(partitions):
            for p2 in partitions[i+1:]:
                # Phase-lock if adjacent in partition coordinates
                cat_dist = self._categorical_distance(p1.coordinates, p2.coordinates)

                if cat_dist <= 1:  # Adjacent partitions
                    coupling_strength = 1.0 / (cat_dist + 0.1)
                    coupling = Coupling(
                        strength=coupling_strength,
                        interaction_type=None,  # Will be VdW by default
                        distance=cat_dist
                    )

                    if self.network.add_edge(p1.id, p2.id, coupling):
                        # Also update partition accessibility
                        p1.accessible.add(p2.id)
                        p2.accessible.add(p1.id)
                        self.metrics.phase_locks_formed += 1

        elapsed = (time.time() - start_time) * 1000
        self._log(f"Built network: {self.metrics.phase_locks_formed} phase-locks in {elapsed:.1f}ms")

    def _categorical_distance(self, c1: PartitionCoordinates,
                             c2: PartitionCoordinates) -> int:
        """Compute categorical distance between coordinates."""
        return abs(c1.n - c2.n) + abs(c1.l - c2.l) + abs(c1.m - c2.m)

    def navigate(self, start: Partition, completion) -> ExecutionResult:
        """
        Navigate from start to completion.

        This is the core operation: trajectory completion.
        Completion can be any object with is_satisfied() and distance_to() methods.
        """
        self.metrics.phase = RuntimePhase.NAVIGATION
        start_time = time.time()

        # Try direct navigation first
        result = self.navigator.navigate(
            start,
            completion,
            self.config.navigation_strategy
        )

        # If direct navigation fails and catalysts enabled, try catalyst chain
        if not result.success and self.config.enable_catalysts:
            self._log("Direct navigation failed, trying catalyst chain")

            # Find target by searching space for satisfying partition
            target_partition = None
            for p in self.space.partitions.values():
                if completion.is_satisfied(p):
                    target_partition = p
                    break

            if target_partition:
                catalyst = self.planner.plan_trajectory(
                    start.coordinates,
                    target_partition.coordinates
                )
                self.metrics.catalyst_stages = len(catalyst)
                result = self.planner.execute_plan(catalyst)

        self.metrics.navigation_steps = result.total_steps
        self.metrics.completion_distance = completion.distance_to(result.final_partition) if result.final_partition else float('inf')

        elapsed = (time.time() - start_time) * 1000
        self.metrics.execution_time_ms += elapsed

        self._log(f"Navigation {'succeeded' if result.success else 'failed'} in {result.total_steps} steps")

        if result.success:
            return self._verify_and_return(result, completion)
        else:
            self.metrics.phase = RuntimePhase.FAILED
            return ExecutionResult(
                success=False,
                trajectory=result.trajectory,
                metrics=self.metrics,
                error=result.metadata.get("reason", "navigation failed")
            )

    def _verify_and_return(self, nav_result: NavigationResult,
                          completion) -> ExecutionResult:
        """
        Verify navigation result.

        Computing = Verification: verification uses the SAME operation
        as navigation (checking completion condition).
        """
        self.metrics.phase = RuntimePhase.VERIFICATION

        # Check using is_satisfied (works for both CompletionPredicate and CompletionCondition)
        is_satisfied = completion.is_satisfied(nav_result.final_partition)
        distance = completion.distance_to(nav_result.final_partition)

        if is_satisfied:
            self.metrics.phase = RuntimePhase.COMPLETED
            return ExecutionResult(
                success=True,
                solution=nav_result.final_partition,
                trajectory=nav_result.trajectory,
                metrics=self.metrics,
                verification_result=None
            )
        else:
            self.metrics.phase = RuntimePhase.FAILED
            return ExecutionResult(
                success=False,
                trajectory=nav_result.trajectory,
                metrics=self.metrics,
                verification_result=None,
                error=f"verification failed: distance={distance}"
            )

    def execute(self, system: System,
               completion: Optional[CompletionCondition] = None,
               start_coords: Optional[PartitionCoordinates] = None) -> ExecutionResult:
        """
        Full execution pipeline.

        1. Load system
        2. Initialize runtime
        3. Build space and network
        4. Navigate to completion
        5. Return result
        """
        total_start = time.time()

        try:
            # Load and initialize
            self.load_system(system)
            self.initialize()

            # Build space
            self.build_space(depth=4)
            self.build_network()

            # Determine start partition
            if start_coords:
                start = self.space.create(start_coords)
            elif self.space.partitions:
                start = list(self.space.partitions.values())[0]
            else:
                return ExecutionResult(
                    success=False,
                    metrics=self.metrics,
                    error="no partitions in space"
                )

            # Determine completion condition
            if completion is None:
                completion = system.as_completion_condition()

            # Navigate
            result = self.navigate(start, completion)

            # Add system state to result
            if result.success:
                result.system_state = {
                    eid: {pname: p.value for pname, p in e.properties.items()}
                    for eid, e in system.entities.items()
                }

            total_elapsed = (time.time() - total_start) * 1000
            result.metrics.execution_time_ms = total_elapsed

            return result

        except Exception as e:
            self.metrics.phase = RuntimePhase.FAILED
            return ExecutionResult(
                success=False,
                metrics=self.metrics,
                error=str(e)
            )

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.config.verbose:
            print(f"[{self.metrics.phase.value}] {message}")


class TrajectoryComputer:
    """
    High-level interface for trajectory computing.

    Provides a simple API for defining and solving problems.
    """

    def __init__(self, verbose: bool = False):
        self.config = RuntimeConfig(verbose=verbose)
        self.runtime = TrajectoryRuntime(self.config)

    def solve(self, system: System,
             target_coords: Optional[PartitionCoordinates] = None,
             target_scoord: Optional[SCoord] = None) -> ExecutionResult:
        """
        Solve a system specification.

        Either use system's built-in completion condition, or
        specify target coordinates/S-coordinates.
        """
        completion = None

        if target_coords:
            completion = CoordinateCompletion(
                target_n=target_coords.n,
                target_l=target_coords.l,
                target_m=target_coords.m,
                target_s=target_coords.s
            )
        elif target_scoord:
            completion = SCoordCompletion(
                target=target_scoord,
                epsilon=self.config.epsilon_boundary
            )

        return self.runtime.execute(system, completion)

    def verify(self, solution: Partition,
              system: System) -> bool:
        """
        Verify a solution satisfies system constraints.

        Computing = Verification.
        """
        completion = system.as_completion_condition()
        result = completion.check(solution)
        return result.status in [CompletionStatus.SATISFIED, CompletionStatus.AT_EPSILON]


# Demonstration
def demo_trajectory_computing():
    """
    Demonstrate trajectory computing with ball-on-ground example.
    """
    print("=" * 60)
    print("TRAJECTORY COMPUTING DEMONSTRATION")
    print("=" * 60)

    # Build system
    system = (SystemBuilder("ball_on_ground")
        .entity("ball", "object",
                position_z=10.0,
                velocity_z=-5.0)
        .entity("ground", "surface",
                height=0.0)
        .constrain("at_ground", "ball at ground level",
                  lambda e: abs(e["ball"].get_property("position_z") or 0) < 0.5)
        .constrain("at_rest", "ball at rest",
                  lambda e: abs(e["ball"].get_property("velocity_z") or 0) < 0.1)
        .build()
    )

    print(f"\nSystem: {system.name}")
    print(f"  Entities: {list(system.entities.keys())}")
    print(f"  Constraints: {list(system.constraints.keys())}")

    # Create computer
    computer = TrajectoryComputer(verbose=True)

    # Define target: at ground, at rest
    target = PartitionCoordinates(n=2, l=1, m=0, s=Spin.UP)

    print(f"\nTarget completion: (n={target.n}, l={target.l}, m={target.m})")

    # Solve
    result = computer.solve(system, target_coords=target)

    print(f"\n{'='*40}")
    print(f"RESULT: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"{'='*40}")

    if result.success:
        print(f"  Solution partition: {result.solution.id}")
        print(f"  Coordinates: (n={result.solution.n}, l={result.solution.l}, m={result.solution.m})")
        print(f"  Trajectory length: {len(result.trajectory)}")
        print(f"  Trajectory: {result.trajectory}")

    print(f"\nMetrics:")
    print(f"  Partitions created: {result.metrics.partitions_created}")
    print(f"  Phase-locks formed: {result.metrics.phase_locks_formed}")
    print(f"  Navigation steps: {result.metrics.navigation_steps}")
    print(f"  Execution time: {result.metrics.execution_time_ms:.1f}ms")

    return result


if __name__ == "__main__":
    demo_trajectory_computing()
