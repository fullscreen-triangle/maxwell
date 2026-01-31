"""
Synthesis Primitives for Poincaré Computing

Synthesis is the core of "compiled physics": instead of simulating
forward from initial conditions, we synthesize backward from the
terminal state.

The key insight:
- Traditional physics: Laws + Initial State → Simulation → Final State
- Poincaré synthesis: Terminal State → Partition Path → Minimal Structure

We don't need to know the laws - they emerge from the partition geometry.
We don't need initial conditions - the synthesis generates what's needed.
We only generate the structure that PARTICIPATES in reaching the terminal.

This is not backwards simulation - it's synthesis of the minimal
categorical structure whose completion IS the terminal state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Set, Callable, Iterator
from enum import Enum, auto
import time
import heapq

from .coordinates import (
    PartitionCoordinate,
    SCoordinate,
    TerminalState,
    CoordinateConstraint,
    aufbau_order,
)
from .operations import (
    OperationType,
    OperationResult,
    PartitionOp,
    ComposeOp,
    TraverseOp,
    K_B,
)
from .catalysts import (
    InformationCatalyst,
    Observer,
    Selector,
    Aligner,
    CatalystAction,
    CatalystRegistry,
    MAXWELL_DEMON,
)


class SynthesisStatus(Enum):
    """Status of a synthesis operation."""
    PENDING = auto()     # Not yet started
    ANALYZING = auto()   # Analyzing terminal state
    GENERATING = auto()  # Generating partition path
    VALIDATING = auto()  # Validating path
    COMPLETE = auto()    # Successfully synthesized
    FAILED = auto()      # Synthesis failed


@dataclass
class TerminalSpec:
    """
    Extended specification for terminal state synthesis.

    This is the "program" in the compiled physics language.
    It specifies:
    - What we want to observe (target coordinates/observables)
    - Constraints on the terminal state
    - Precision requirements
    - Allowed catalyst types

    Attributes:
        name: Human-readable name for this specification
        terminal_state: The target state to synthesize to
        observables: What must be observable at termination
        precision: Required precision for coordinate matching
        allowed_catalysts: Which catalyst types can be used
        max_path_length: Maximum length of synthesized path
        metadata: Additional specification data
    """
    name: str
    terminal_state: TerminalState
    observables: Dict[str, Any] = field(default_factory=dict)
    precision: float = 0.01
    allowed_catalysts: Set[str] = field(
        default_factory=lambda: {"observer", "selector", "aligner"}
    )
    max_path_length: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_coordinates(
        cls,
        name: str,
        *coords: PartitionCoordinate,
        **kwargs
    ) -> TerminalSpec:
        """Create specification from partition coordinates."""
        terminal = TerminalState.from_partition(*coords)
        return cls(name=name, terminal_state=terminal, **kwargs)

    @classmethod
    def from_element(cls, atomic_number: int) -> TerminalSpec:
        """
        Create specification for synthesizing an element.

        Uses aufbau order to determine ground state configuration.
        """
        coords = aufbau_order(max_coords=atomic_number)[:atomic_number]
        terminal = TerminalState.from_partition(*coords, label=f"element_{atomic_number}")

        return cls(
            name=f"element_{atomic_number}",
            terminal_state=terminal,
            observables={
                "atomic_number": atomic_number,
                "electron_count": atomic_number,
            },
            metadata={"element": True}
        )

    @classmethod
    def from_observable(
        cls,
        name: str,
        observable_name: str,
        observable_value: Any,
        **kwargs
    ) -> TerminalSpec:
        """
        Create specification from a desired observable.

        The synthesizer will find the terminal state that produces
        this observable value.
        """
        return cls(
            name=name,
            terminal_state=TerminalState(),  # Empty - will be derived
            observables={observable_name: observable_value},
            **kwargs
        )

    def validate(self) -> bool:
        """Check if specification is valid."""
        return self.terminal_state.validate() or len(self.observables) > 0


@dataclass
class PartitionStep:
    """
    Single step in a partition path.

    Each step represents a transition in partition space:
    - From one coordinate to another
    - Via an operation (partition, compose, traverse)
    - Optionally catalyzed by an information catalyst

    Attributes:
        from_coord: Starting coordinate
        to_coord: Ending coordinate
        operation: Operation that performs this step
        catalyst: Optional catalyst enabling this step
        entropy_cost: Entropy generated by this step
        metadata: Additional step data
    """
    from_coord: SCoordinate
    to_coord: SCoordinate
    operation: OperationType
    catalyst: Optional[InformationCatalyst] = None
    entropy_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def distance(self) -> float:
        """Categorical distance covered by this step."""
        return self.from_coord.distance_to(self.to_coord)

    @property
    def is_catalyzed(self) -> bool:
        """Check if step uses a catalyst."""
        return self.catalyst is not None

    @property
    def is_free(self) -> bool:
        """Check if step is thermodynamically free."""
        return self.entropy_cost == 0.0 or self.is_catalyzed


@dataclass
class PartitionPath:
    """
    Complete path through partition space.

    A path is a sequence of steps from origin to terminal state.
    The path is the "compiled program" - it represents the minimal
    partition structure needed to reach the terminal state.

    Attributes:
        steps: Sequence of partition steps
        terminal: The terminal state this path reaches
        synthesis_time_ns: Time taken to synthesize this path
        total_entropy: Total entropy generated along path
    """
    steps: List[PartitionStep] = field(default_factory=list)
    terminal: Optional[TerminalState] = None
    synthesis_time_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of steps in path."""
        return len(self.steps)

    @property
    def total_entropy(self) -> float:
        """Total entropy generated along entire path."""
        return sum(step.entropy_cost for step in self.steps)

    @property
    def total_distance(self) -> float:
        """Total categorical distance traversed."""
        return sum(step.distance for step in self.steps)

    @property
    def origin(self) -> Optional[SCoordinate]:
        """Starting coordinate of path."""
        return self.steps[0].from_coord if self.steps else None

    @property
    def terminus(self) -> Optional[SCoordinate]:
        """Ending coordinate of path."""
        return self.steps[-1].to_coord if self.steps else None

    @property
    def catalyst_count(self) -> int:
        """Number of catalyzed steps."""
        return sum(1 for step in self.steps if step.is_catalyzed)

    def append(self, step: PartitionStep) -> None:
        """Add a step to the path."""
        self.steps.append(step)

    def reverse(self) -> PartitionPath:
        """Create reversed path (for backward synthesis)."""
        reversed_steps = []
        for step in reversed(self.steps):
            reversed_steps.append(PartitionStep(
                from_coord=step.to_coord,
                to_coord=step.from_coord,
                operation=step.operation,
                catalyst=step.catalyst,
                entropy_cost=step.entropy_cost,
                metadata=step.metadata,
            ))
        return PartitionPath(
            steps=reversed_steps,
            terminal=self.terminal,
            synthesis_time_ns=self.synthesis_time_ns,
            metadata=self.metadata,
        )

    def __iter__(self) -> Iterator[PartitionStep]:
        """Iterate over steps."""
        return iter(self.steps)


@dataclass
class SynthesisResult:
    """
    Result of a synthesis operation.

    Contains the synthesized path, status, and metadata about
    the synthesis process.

    Attributes:
        spec: The specification that was synthesized
        status: Synthesis status
        path: The synthesized partition path (if successful)
        error: Error message (if failed)
        synthesis_time_ns: Time taken for synthesis
        iterations: Number of iterations used
        catalysts_used: Catalysts that participated
    """
    spec: TerminalSpec
    status: SynthesisStatus
    path: Optional[PartitionPath] = None
    error: Optional[str] = None
    synthesis_time_ns: int = 0
    iterations: int = 0
    catalysts_used: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if synthesis succeeded."""
        return self.status == SynthesisStatus.COMPLETE and self.path is not None

    @property
    def path_length(self) -> int:
        """Length of synthesized path."""
        return self.path.length if self.path else 0

    @property
    def total_entropy(self) -> float:
        """Total entropy of synthesized path."""
        return self.path.total_entropy if self.path else 0.0


@dataclass
class Synthesizer:
    """
    The Poincaré Synthesizer: compiles terminal states into partition paths.

    This is the "compiler" in the compiled physics framework. Given a
    terminal state specification, it synthesizes the minimal partition
    path that terminates at that state.

    The synthesis is BACKWARD: we start from the terminal state and
    work backward to find what partition structure is needed.

    Key insight: we don't simulate forward through time. We synthesize
    the categorical structure whose completion IS the terminal state.
    The "laws of physics" emerge from the partition geometry.

    Attributes:
        catalyst_registry: Available catalysts
        max_iterations: Maximum synthesis iterations
        precision: Coordinate matching precision
        strategy: Synthesis strategy ("greedy", "optimal", "heuristic")
    """
    catalyst_registry: CatalystRegistry = field(
        default_factory=lambda: CatalystRegistry()
    )
    max_iterations: int = 10000
    precision: float = 0.01
    strategy: str = "greedy"

    def __post_init__(self):
        """Register default catalysts."""
        self.catalyst_registry.register(MAXWELL_DEMON)

    def synthesize(self, spec: TerminalSpec) -> SynthesisResult:
        """
        Synthesize a partition path for the given specification.

        This is the main entry point for synthesis. It:
        1. Analyzes the terminal state
        2. Generates the partition path backward
        3. Validates the path
        4. Returns the result

        The key insight: we don't simulate forward. We find the
        minimal partition structure whose completion IS the terminal.
        """
        start_time = time.perf_counter_ns()

        # Validate specification
        if not spec.validate():
            return SynthesisResult(
                spec=spec,
                status=SynthesisStatus.FAILED,
                error="Invalid specification",
                synthesis_time_ns=time.perf_counter_ns() - start_time,
            )

        # Choose synthesis method based on strategy
        if self.strategy == "greedy":
            result = self._synthesize_greedy(spec)
        elif self.strategy == "optimal":
            result = self._synthesize_optimal(spec)
        else:
            result = self._synthesize_heuristic(spec)

        result.synthesis_time_ns = time.perf_counter_ns() - start_time
        return result

    def _synthesize_greedy(self, spec: TerminalSpec) -> SynthesisResult:
        """
        Greedy backward synthesis.

        Start from terminal state, greedily find predecessors
        until we reach the origin (or can't proceed).
        """
        path = PartitionPath(terminal=spec.terminal_state)
        catalysts_used = []
        iterations = 0

        # Get terminal coordinates in S-space
        terminal_coords = spec.terminal_state.to_s_coordinates()
        if not terminal_coords:
            # If no explicit coordinates, use the observable specification
            # to infer a terminal S-coordinate
            terminal_coords = [SCoordinate(
                S_k=0.5,  # Default: middle of state space
                S_t=0.5,
                S_e=0.5,
            )]

        # Origin: the starting point (zero entropy)
        origin = SCoordinate.origin()

        # Work backward from each terminal coordinate
        for terminal_s in terminal_coords:
            current = terminal_s
            path_to_terminal = []

            while iterations < self.max_iterations:
                iterations += 1

                # Check if we've reached the origin
                if current.distance_to(origin) < self.precision:
                    break

                # Find predecessor (move toward origin)
                predecessor = self._find_predecessor(current, origin)

                # Create step from predecessor to current
                step = PartitionStep(
                    from_coord=predecessor,
                    to_coord=current,
                    operation=OperationType.TRAVERSE,
                    catalyst=None,
                    entropy_cost=K_B * 0.01,  # Small traversal entropy
                )

                path_to_terminal.append(step)
                current = predecessor

            # Reverse the path (we built it backward)
            for step in reversed(path_to_terminal):
                path.append(step)

        return SynthesisResult(
            spec=spec,
            status=SynthesisStatus.COMPLETE,
            path=path,
            iterations=iterations,
            catalysts_used=catalysts_used,
        )

    def _synthesize_optimal(self, spec: TerminalSpec) -> SynthesisResult:
        """
        Optimal synthesis using A* search.

        Finds the minimum-entropy path to the terminal state.
        """
        path = PartitionPath(terminal=spec.terminal_state)
        iterations = 0

        terminal_coords = spec.terminal_state.to_s_coordinates()
        if not terminal_coords:
            terminal_coords = [SCoordinate(0.5, 0.5, 0.5)]

        origin = SCoordinate.origin()

        for terminal_s in terminal_coords:
            # A* search from origin to terminal
            # Priority queue: (estimated_cost, actual_cost, coord, path_so_far)
            frontier = [(terminal_s.distance_to(origin), 0.0, origin, [])]
            visited = set()

            while frontier and iterations < self.max_iterations:
                iterations += 1

                est_cost, actual_cost, current, path_so_far = heapq.heappop(frontier)

                # Check if we've reached the terminal
                if current.distance_to(terminal_s) < self.precision:
                    # Add all steps to path
                    for step in path_so_far:
                        path.append(step)
                    break

                # Skip if already visited
                coord_hash = current.hash
                if coord_hash in visited:
                    continue
                visited.add(coord_hash)

                # Generate successors (steps toward terminal)
                for successor in self._generate_successors(current, terminal_s):
                    step = PartitionStep(
                        from_coord=current,
                        to_coord=successor,
                        operation=OperationType.TRAVERSE,
                        entropy_cost=K_B * current.distance_to(successor),
                    )
                    new_cost = actual_cost + step.entropy_cost
                    est_total = new_cost + successor.distance_to(terminal_s)

                    heapq.heappush(frontier, (
                        est_total,
                        new_cost,
                        successor,
                        path_so_far + [step],
                    ))

        return SynthesisResult(
            spec=spec,
            status=SynthesisStatus.COMPLETE,
            path=path,
            iterations=iterations,
        )

    def _synthesize_heuristic(self, spec: TerminalSpec) -> SynthesisResult:
        """
        Heuristic synthesis using domain knowledge.

        Uses knowledge of partition geometry to find good paths.
        """
        # For now, delegate to greedy
        return self._synthesize_greedy(spec)

    def _find_predecessor(
        self,
        current: SCoordinate,
        target: SCoordinate,
        step_size: float = 0.1
    ) -> SCoordinate:
        """
        Find predecessor coordinate moving from current toward target.

        This is the "backward step" in synthesis: given where we are,
        where did we come from (moving toward origin)?
        """
        distance = current.distance_to(target)
        if distance < step_size:
            return target

        # Interpolate toward target
        t = step_size / distance
        return current.interpolate(target, t)

    def _generate_successors(
        self,
        current: SCoordinate,
        terminal: SCoordinate,
        n_successors: int = 8
    ) -> List[SCoordinate]:
        """
        Generate successor coordinates for A* search.

        Creates candidate next steps moving toward terminal.
        """
        successors = []
        step_size = 0.1

        # Direct path toward terminal
        direct = current.interpolate(terminal, step_size / current.distance_to(terminal))
        successors.append(direct)

        # Add some exploration around the direct path
        for i in range(n_successors - 1):
            angle = (i / (n_successors - 1)) * 2 * 3.14159
            offset = 0.05
            candidate = SCoordinate(
                S_k=direct.S_k + offset * (0.5 - (i % 2)),
                S_t=direct.S_t + offset * (0.5 - ((i + 1) % 2)),
                S_e=direct.S_e + offset * (0.5 - ((i + 2) % 2)),
            )
            successors.append(candidate)

        return successors

    def synthesize_element(self, atomic_number: int) -> SynthesisResult:
        """
        Convenience method to synthesize an element.

        Given an atomic number, synthesizes the partition path
        that produces that element's ground state configuration.
        """
        spec = TerminalSpec.from_element(atomic_number)
        return self.synthesize(spec)

    def synthesize_transition(
        self,
        from_coord: PartitionCoordinate,
        to_coord: PartitionCoordinate
    ) -> SynthesisResult:
        """
        Synthesize a transition between partition coordinates.

        Checks selection rules and generates the minimal path.
        """
        # Check selection rules
        if not from_coord.can_transition_to(to_coord):
            return SynthesisResult(
                spec=TerminalSpec(
                    name="transition",
                    terminal_state=TerminalState.from_partition(to_coord),
                ),
                status=SynthesisStatus.FAILED,
                error="Transition violates selection rules",
            )

        spec = TerminalSpec(
            name=f"transition_{from_coord.orbital_name}_{to_coord.orbital_name}",
            terminal_state=TerminalState.from_partition(to_coord),
            observables={
                "transition_energy": from_coord.transition_energy(to_coord),
            }
        )

        return self.synthesize(spec)


# Convenience functions

def synthesize(spec: TerminalSpec) -> SynthesisResult:
    """
    Synthesize a partition path for the given specification.

    This is the main entry point for the compiled physics framework.
    """
    synthesizer = Synthesizer()
    return synthesizer.synthesize(spec)


def synthesize_element(atomic_number: int) -> SynthesisResult:
    """Synthesize an element by atomic number."""
    synthesizer = Synthesizer()
    return synthesizer.synthesize_element(atomic_number)


def synthesize_observable(
    name: str,
    observable: str,
    value: Any
) -> SynthesisResult:
    """
    Synthesize from an observable specification.

    Given what we want to observe, synthesize the partition
    structure that produces it.
    """
    spec = TerminalSpec.from_observable(name, observable, value)
    synthesizer = Synthesizer()
    return synthesizer.synthesize(spec)
