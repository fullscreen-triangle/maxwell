"""
Information Catalyst Primitives for Poincaré Computing

Information catalysts enable transitions without being consumed.
They are the Maxwell Demons of the computational framework.

Key insight from the partition coordinate geometry:
- Categorical operations are ORTHOGONAL to physical operations
- The Maxwell Demon can observe and sort without thermodynamic cost
- This works because information processing happens in categorical
  space, not physical space

Catalysts don't violate the second law - they operate in a different
space entirely. The physical entropy is preserved; the categorical
operations are free.

Types of catalysts:
1. Observer: Enables measurement (couples to partition coordinates)
2. Selector: Enables choice (picks from alternatives)
3. Aligner: Enables coordination (synchronizes multiple systems)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Optional, Tuple, Dict, Any, List, Callable,
    TypeVar, Generic, Protocol, Set
)
from abc import ABC, abstractmethod
from enum import Enum, auto
import time
import hashlib

from .coordinates import (
    PartitionCoordinate,
    SCoordinate,
    TerminalState,
)
from .operations import (
    OperationType,
    OperationResult,
    PartitionOp,
    ComposeOp,
    TraverseOp,
    MeasureOp,
    K_B,
)


class CatalystState(Enum):
    """State of an information catalyst."""
    IDLE = auto()        # Ready to catalyze
    ACTIVE = auto()      # Currently catalyzing
    SATURATED = auto()   # Temporarily unable to catalyze (needs reset)
    EXHAUSTED = auto()   # Permanently unable to catalyze


class CatalystType(Enum):
    """Classification of catalyst types."""
    OBSERVER = auto()    # Enables measurement
    SELECTOR = auto()    # Enables choice
    ALIGNER = auto()     # Enables coordination
    TRANSFORMER = auto() # Enables representation change
    FILTER = auto()      # Enables constraint enforcement


T = TypeVar('T')


class Catalyzable(Protocol):
    """Protocol for things that can be catalyzed."""
    def accept_catalyst(self, catalyst: InformationCatalyst) -> OperationResult:
        """Accept catalytic action and return result."""
        ...


@dataclass
class CatalystAction:
    """
    Record of a catalyst action.

    Unlike regular operations, catalyst actions generate NO entropy
    in the categorical space. The entropy cost is zero because
    catalysis operates orthogonally to thermodynamics.

    Attributes:
        catalyst_id: Unique identifier of the catalyst
        catalyst_type: Type of catalyst
        input_state: State before catalysis
        output_state: State after catalysis
        timestamp: When action occurred
        physical_entropy: Physical entropy change (always 0)
        categorical_work: Work done in categorical space
        metadata: Additional action-specific data
    """
    catalyst_id: str
    catalyst_type: CatalystType
    input_state: Any
    output_state: Any
    timestamp: float = field(default_factory=time.time)
    physical_entropy: float = 0.0  # Always zero - orthogonal to physical
    categorical_work: float = 0.0  # Can be non-zero
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_thermodynamically_free(self) -> bool:
        """Catalysis is always thermodynamically free."""
        return self.physical_entropy == 0.0


@dataclass
class InformationCatalyst(ABC):
    """
    Base class for information catalysts.

    An information catalyst enables transitions without being consumed.
    It operates in categorical space, orthogonal to physical space,
    so it incurs no thermodynamic cost.

    This is the resolution of Maxwell's Demon paradox: the demon
    CAN sort molecules without thermodynamic cost, because observation
    and sorting are categorical operations, not physical ones.

    The "cost" that Landauer identified is only incurred when
    categorical information must be erased - but in Poincaré computing,
    we never erase, we only compose (which is reversible in the
    categorical layer).

    Attributes:
        name: Human-readable identifier
        catalyst_type: Classification of this catalyst
        state: Current operational state
        action_history: Record of all catalytic actions
        capacity: Maximum actions before saturation (None = unlimited)
    """
    name: str
    catalyst_type: CatalystType = field(default=CatalystType.OBSERVER)
    state: CatalystState = CatalystState.IDLE
    action_history: List[CatalystAction] = field(default_factory=list)
    capacity: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def catalyst_id(self) -> str:
        """Unique identifier derived from name and type."""
        data = f"{self.name}:{self.catalyst_type.name}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    @property
    def actions_performed(self) -> int:
        """Number of catalytic actions performed."""
        return len(self.action_history)

    @property
    def is_available(self) -> bool:
        """Check if catalyst can perform actions."""
        if self.state in (CatalystState.SATURATED, CatalystState.EXHAUSTED):
            return False
        if self.capacity and self.actions_performed >= self.capacity:
            return False
        return True

    def reset(self) -> None:
        """Reset catalyst to idle state (if not exhausted)."""
        if self.state != CatalystState.EXHAUSTED:
            self.state = CatalystState.IDLE

    @abstractmethod
    def catalyze(self, target: Any) -> CatalystAction:
        """
        Perform catalytic action on target.

        Must be implemented by concrete catalysts.
        """
        pass

    def _record_action(self, action: CatalystAction) -> None:
        """Record an action and update state."""
        self.action_history.append(action)

        if self.capacity and self.actions_performed >= self.capacity:
            self.state = CatalystState.SATURATED


@dataclass
class Observer(InformationCatalyst):
    """
    Observer catalyst: enables measurement without disturbance.

    The Observer couples to partition coordinates through frequency
    matching. It extracts information about the categorical state
    without affecting the physical state.

    This is the Maxwell Demon's observation mechanism. The demon
    can know which molecule is which without touching them.

    Attributes:
        frequency: Coupling frequency (Hz)
        bandwidth: Frequency range for matching
        target_coordinates: Which coordinates this observer can measure
    """
    frequency: float = 1e9  # 1 GHz default
    bandwidth: float = 1e6  # 1 MHz default
    target_coordinates: Set[str] = field(
        default_factory=lambda: {"S_k", "S_t", "S_e", "n", "l", "m", "s"}
    )

    def __post_init__(self):
        # Set the catalyst type for this subclass
        object.__setattr__(self, 'catalyst_type', CatalystType.OBSERVER)
        # Ensure action_history is initialized
        if self.action_history is None:
            object.__setattr__(self, 'action_history', [])

    def catalyze(self, target: SCoordinate | PartitionCoordinate) -> CatalystAction:
        """
        Observe the target coordinate.

        Extracts categorical information without physical disturbance.
        """
        if not self.is_available:
            raise RuntimeError(f"Observer {self.name} is not available")

        self.state = CatalystState.ACTIVE
        start_time = time.time()

        # Extract coordinate values based on target type
        if isinstance(target, SCoordinate):
            observed = {
                "S_k": target.S_k,
                "S_t": target.S_t,
                "S_e": target.S_e,
                "coordinate_type": "S",
            }
        elif isinstance(target, PartitionCoordinate):
            observed = {
                "n": target.n,
                "l": target.l,
                "m": target.m,
                "s": target.s,
                "coordinate_type": "partition",
            }
        else:
            observed = {"value": target, "coordinate_type": "unknown"}

        action = CatalystAction(
            catalyst_id=self.catalyst_id,
            catalyst_type=self.catalyst_type,
            input_state=target,
            output_state=observed,
            timestamp=start_time,
            physical_entropy=0.0,  # No physical entropy cost
            categorical_work=len(observed),  # Work in categorical space
            metadata={
                "frequency": self.frequency,
                "bandwidth": self.bandwidth,
            }
        )

        self._record_action(action)
        self.state = CatalystState.IDLE

        return action

    def can_observe(self, coord_name: str) -> bool:
        """Check if this observer can measure the given coordinate."""
        return coord_name in self.target_coordinates


@dataclass
class Selector(InformationCatalyst):
    """
    Selector catalyst: enables choice without randomness.

    The Selector picks from alternatives based on categorical criteria.
    Unlike random selection, this is deterministic but not predictable
    from the physical state alone.

    This is the Maxwell Demon's sorting mechanism. Given two molecules,
    the demon can reliably choose the faster one without coin flips.

    Attributes:
        criterion: Selection criterion function
        threshold: Threshold for binary selection
        mode: "max", "min", "threshold", "custom"
    """
    criterion: Callable[[Any], float] = field(default=lambda x: 0.0)
    threshold: float = 0.5
    mode: str = "max"

    def __post_init__(self):
        object.__setattr__(self, 'catalyst_type', CatalystType.SELECTOR)
        if self.action_history is None:
            object.__setattr__(self, 'action_history', [])

    def catalyze(self, targets: List[Any]) -> CatalystAction:
        """
        Select from the targets based on criterion.

        Returns the selected target(s) without consuming any.
        """
        if not self.is_available:
            raise RuntimeError(f"Selector {self.name} is not available")
        if not targets:
            raise ValueError("Cannot select from empty list")

        self.state = CatalystState.ACTIVE
        start_time = time.time()

        # Evaluate criterion for each target
        scores = [(t, self.criterion(t)) for t in targets]

        # Select based on mode
        if self.mode == "max":
            selected = max(scores, key=lambda x: x[1])[0]
        elif self.mode == "min":
            selected = min(scores, key=lambda x: x[1])[0]
        elif self.mode == "threshold":
            selected = [t for t, s in scores if s >= self.threshold]
        elif self.mode == "custom":
            # Custom selection via criterion
            selected = [t for t, s in scores if s > 0]
        else:
            raise ValueError(f"Unknown selection mode: {self.mode}")

        action = CatalystAction(
            catalyst_id=self.catalyst_id,
            catalyst_type=self.catalyst_type,
            input_state=targets,
            output_state=selected,
            timestamp=start_time,
            physical_entropy=0.0,  # No physical entropy cost
            categorical_work=len(targets),  # Evaluated all targets
            metadata={
                "mode": self.mode,
                "scores": scores,
                "threshold": self.threshold,
            }
        )

        self._record_action(action)
        self.state = CatalystState.IDLE

        return action


@dataclass
class Aligner(InformationCatalyst):
    """
    Aligner catalyst: enables coordination without communication.

    The Aligner synchronizes multiple systems or coordinates to
    achieve coherence. It works by establishing categorical
    relationships rather than physical connections.

    This is how entanglement works in the partition framework:
    two systems become aligned in categorical space without
    requiring signal propagation.

    Attributes:
        alignment_target: Target coordinate or state to align to
        tolerance: Alignment precision
        mode: "exact", "approximate", "asymptotic"
    """
    alignment_target: Optional[SCoordinate] = None
    tolerance: float = 0.01
    mode: str = "exact"

    def __post_init__(self):
        object.__setattr__(self, 'catalyst_type', CatalystType.ALIGNER)
        if self.action_history is None:
            object.__setattr__(self, 'action_history', [])

    def catalyze(
        self,
        targets: List[SCoordinate]
    ) -> CatalystAction:
        """
        Align the target coordinates.

        Brings all targets into categorical alignment without
        physical interaction.
        """
        if not self.is_available:
            raise RuntimeError(f"Aligner {self.name} is not available")
        if not targets:
            raise ValueError("Cannot align empty list")

        self.state = CatalystState.ACTIVE
        start_time = time.time()

        # Determine alignment point
        if self.alignment_target:
            align_to = self.alignment_target
        else:
            # Align to centroid of all targets
            align_to = SCoordinate(
                S_k=sum(t.S_k for t in targets) / len(targets),
                S_t=sum(t.S_t for t in targets) / len(targets),
                S_e=sum(t.S_e for t in targets) / len(targets),
            )

        # Compute aligned coordinates
        if self.mode == "exact":
            aligned = [align_to for _ in targets]
        elif self.mode == "approximate":
            aligned = [
                t.interpolate(align_to, 1 - self.tolerance)
                for t in targets
            ]
        elif self.mode == "asymptotic":
            # Each step gets closer but never reaches
            aligned = [
                t.interpolate(align_to, 0.9)
                for t in targets
            ]
        else:
            raise ValueError(f"Unknown alignment mode: {self.mode}")

        action = CatalystAction(
            catalyst_id=self.catalyst_id,
            catalyst_type=self.catalyst_type,
            input_state=targets,
            output_state=aligned,
            timestamp=start_time,
            physical_entropy=0.0,  # No physical entropy cost
            categorical_work=len(targets) * 3,  # 3 dimensions per target
            metadata={
                "mode": self.mode,
                "alignment_target": align_to.as_tuple(),
                "tolerance": self.tolerance,
            }
        )

        self._record_action(action)
        self.state = CatalystState.IDLE

        return action


@dataclass
class CatalystRegistry:
    """
    Registry for managing multiple catalysts.

    The registry tracks all available catalysts and their states,
    enabling coordination of complex catalytic workflows.

    Attributes:
        catalysts: Dictionary of registered catalysts by ID
        name: Registry name
    """
    name: str = "default"
    catalysts: Dict[str, InformationCatalyst] = field(default_factory=dict)

    def register(self, catalyst: InformationCatalyst) -> str:
        """Register a catalyst and return its ID."""
        self.catalysts[catalyst.catalyst_id] = catalyst
        return catalyst.catalyst_id

    def get(self, catalyst_id: str) -> Optional[InformationCatalyst]:
        """Get a catalyst by ID."""
        return self.catalysts.get(catalyst_id)

    def get_by_type(
        self,
        catalyst_type: CatalystType
    ) -> List[InformationCatalyst]:
        """Get all catalysts of a specific type."""
        return [
            c for c in self.catalysts.values()
            if c.catalyst_type == catalyst_type
        ]

    def get_available(self) -> List[InformationCatalyst]:
        """Get all available catalysts."""
        return [c for c in self.catalysts.values() if c.is_available]

    def reset_all(self) -> None:
        """Reset all catalysts to idle state."""
        for catalyst in self.catalysts.values():
            catalyst.reset()

    @property
    def total_actions(self) -> int:
        """Total actions performed by all catalysts."""
        return sum(c.actions_performed for c in self.catalysts.values())

    @property
    def total_categorical_work(self) -> float:
        """Total categorical work performed by all catalysts."""
        return sum(
            action.categorical_work
            for c in self.catalysts.values()
            for action in c.action_history
        )


# Convenience factory functions

def create_observer(
    name: str,
    frequency: float = 1e9,
    target_coords: Optional[Set[str]] = None
) -> Observer:
    """Create an Observer catalyst."""
    return Observer(
        name=name,
        frequency=frequency,
        target_coordinates=target_coords or {"S_k", "S_t", "S_e", "n", "l", "m", "s"},
    )


def create_selector(
    name: str,
    criterion: Callable[[Any], float],
    mode: str = "max"
) -> Selector:
    """Create a Selector catalyst."""
    return Selector(
        name=name,
        criterion=criterion,
        mode=mode,
    )


def create_aligner(
    name: str,
    target: Optional[SCoordinate] = None,
    mode: str = "exact"
) -> Aligner:
    """Create an Aligner catalyst."""
    return Aligner(
        name=name,
        alignment_target=target,
        mode=mode,
    )


# Standard catalysts (pre-defined)

# The Maxwell Demon: observes and sorts without entropy cost
MAXWELL_DEMON = Observer(
    name="maxwell_demon",
    frequency=1e12,  # THz - very fast observation
    bandwidth=1e9,   # GHz bandwidth
    target_coordinates={"S_k", "S_t", "S_e", "n", "l", "m", "s"},
    capacity=None,   # Unlimited observations
)

# Energy selector: picks lower energy states
ENERGY_SELECTOR = Selector(
    name="energy_selector",
    criterion=lambda c: -c.energy if hasattr(c, 'energy') else 0,
    mode="max",  # Max of negative energy = min energy
)

# Ground state aligner: aligns to ground state
GROUND_ALIGNER = Aligner(
    name="ground_aligner",
    alignment_target=SCoordinate(S_k=0.0, S_t=0.0, S_e=0.0),
    mode="asymptotic",  # Approach but never reach
)
