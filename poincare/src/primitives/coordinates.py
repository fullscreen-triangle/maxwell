"""
Coordinate Primitives for Poincaré Computing

Coordinates are the fundamental addressing system in partition space.
They specify positions in categorical state space, not physical space.

Three coordinate systems, all equivalent (Triple Equivalence):
1. PartitionCoordinate (n, l, m, s) - discrete partition addressing
2. SCoordinate (S_k, S_t, S_e) - continuous categorical state
3. TerminalState - specification of desired outcome

The key insight: coordinates ARE the physics. Laws emerge from
the geometric constraints on coordinate values and transitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Set, FrozenSet
from enum import Enum, auto
import hashlib
import time


class CoordinateType(Enum):
    """Classification of coordinate systems."""
    PARTITION = auto()    # Discrete (n, l, m, s)
    CATEGORICAL = auto()  # Continuous (S_k, S_t, S_e)
    TERMINAL = auto()     # Specification of outcome


@dataclass(frozen=True)
class PartitionCoordinate:
    """
    Discrete coordinate in partition space.

    These are the "quantum numbers" that emerge from bounded phase space
    geometry. The constraints are geometric necessities, not postulates:

    - n >= 1 (at least one partition exists)
    - 0 <= l < n (complexity cannot exceed depth)
    - -l <= m <= l (orientation within complexity)
    - s in {-1/2, +1/2} (binary chirality)

    Capacity at depth n: 2n² states (derived, not assumed)

    Attributes:
        n: Partition depth (principal quantum number)
        l: Angular complexity (azimuthal quantum number)
        m: Orientation (magnetic quantum number)
        s: Chirality (spin quantum number)
    """
    n: int
    l: int
    m: int
    s: float

    def __post_init__(self):
        """Validate geometric constraints."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1 (at least one partition), got {self.n}")
        if not (0 <= self.l < self.n):
            raise ValueError(f"l must be in [0, n-1] = [0, {self.n-1}], got {self.l}")
        if not (-self.l <= self.m <= self.l):
            raise ValueError(f"m must be in [-l, l] = [{-self.l}, {self.l}], got {self.m}")
        if self.s not in (-0.5, 0.5):
            raise ValueError(f"s must be +/- 0.5 (binary chirality), got {self.s}")

    @property
    def shell_capacity(self) -> int:
        """Number of states at this depth level: 2n²."""
        return 2 * self.n ** 2

    @property
    def energy(self) -> float:
        """
        Energy ordering: E ~ -1/(n + αl)².

        For hydrogen-like systems, α ≈ 0 gives E_n = -13.6 eV / n².
        The aufbau order uses α ≈ 1 (Madelung rule).
        """
        E_0 = 13.6  # eV, Rydberg energy
        alpha = 0.0  # Pure radial ordering
        return -E_0 / (self.n + alpha * self.l) ** 2

    @property
    def orbital_name(self) -> str:
        """Spectroscopic notation: 1s, 2p, 3d, etc."""
        orbital_letters = "spdfghiklmnoqrtuvwxyz"
        return f"{self.n}{orbital_letters[self.l]}"

    def as_tuple(self) -> Tuple[int, int, int, float]:
        """Return as (n, l, m, s) tuple."""
        return (self.n, self.l, self.m, self.s)

    def can_transition_to(self, other: PartitionCoordinate) -> bool:
        """
        Check if transition to other coordinate is allowed.

        Selection rules from boundary continuity:
        - Δl = ±1 (angular complexity changes by 1)
        - Δm ∈ {0, ±1} (orientation changes by at most 1)
        - Δs = 0 (chirality conserved in EM transitions)
        """
        delta_l = abs(other.l - self.l)
        delta_m = abs(other.m - self.m)
        delta_s = abs(other.s - self.s)

        return delta_l == 1 and delta_m <= 1 and delta_s == 0

    def transition_energy(self, other: PartitionCoordinate) -> float:
        """Energy difference for transition to other state."""
        return other.energy - self.energy


@dataclass(frozen=True)
class SCoordinate:
    """
    Continuous coordinate in categorical S-space.

    The triple S-coordinates represent uncertainty/entropy in three
    orthogonal dimensions:

    - S_k: Knowledge entropy (uncertainty in state identification)
    - S_t: Temporal entropy (uncertainty in timing)
    - S_e: Evolution entropy (uncertainty in trajectory)

    All values bounded to [0, 1]. The unit cube [0,1]³ is the
    categorical state space where all computation occurs.

    Attributes:
        S_k: Knowledge entropy [0, 1]
        S_t: Temporal entropy [0, 1]
        S_e: Evolution entropy [0, 1]
    """
    S_k: float
    S_t: float
    S_e: float

    def __post_init__(self):
        """Clamp values to valid range."""
        object.__setattr__(self, 'S_k', max(0.0, min(1.0, self.S_k)))
        object.__setattr__(self, 'S_t', max(0.0, min(1.0, self.S_t)))
        object.__setattr__(self, 'S_e', max(0.0, min(1.0, self.S_e)))

    @classmethod
    def origin(cls) -> SCoordinate:
        """The origin: zero entropy in all dimensions."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def maximum(cls) -> SCoordinate:
        """Maximum entropy state."""
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def from_partition(cls, coord: PartitionCoordinate) -> SCoordinate:
        """
        Map partition coordinate to S-space.

        The mapping preserves geometric structure:
        - Higher n → higher S_k (more states to distinguish)
        - Higher l → higher S_t (more angular complexity)
        - m/l ratio → S_e (orientation relative to complexity)
        """
        # Normalize to [0, 1]
        n_max = 7  # Practical limit for most systems
        S_k = min(1.0, (coord.n - 1) / (n_max - 1)) if n_max > 1 else 0.0
        S_t = coord.l / coord.n if coord.n > 0 else 0.0
        S_e = (coord.m + coord.l) / (2 * coord.l + 1) if coord.l > 0 else 0.5

        return cls(S_k, S_t, S_e)

    def distance_to(self, other: SCoordinate) -> float:
        """
        Categorical distance (not physical distance).

        This is the fundamental metric in S-space. Euclidean for now,
        but could be generalized to other metrics.
        """
        return (
            (self.S_k - other.S_k) ** 2 +
            (self.S_t - other.S_t) ** 2 +
            (self.S_e - other.S_e) ** 2
        ) ** 0.5

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return as (S_k, S_t, S_e) tuple."""
        return (self.S_k, self.S_t, self.S_e)

    @property
    def total_entropy(self) -> float:
        """Sum of all entropy components."""
        return self.S_k + self.S_t + self.S_e

    @property
    def hash(self) -> str:
        """Unique identifier for this coordinate."""
        data = f"{self.S_k:.10f},{self.S_t:.10f},{self.S_e:.10f}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def interpolate(self, other: SCoordinate, t: float) -> SCoordinate:
        """Linear interpolation toward other coordinate."""
        t = max(0.0, min(1.0, t))
        return SCoordinate(
            self.S_k + t * (other.S_k - self.S_k),
            self.S_t + t * (other.S_t - self.S_t),
            self.S_e + t * (other.S_e - self.S_e),
        )


@dataclass
class CoordinateConstraint:
    """
    Constraint on coordinate values.

    Constraints define the geometry of partition space. They are not
    arbitrary restrictions but geometric necessities arising from
    bounded phase space.

    Attributes:
        name: Human-readable constraint name
        predicate: Function that checks if coordinate satisfies constraint
        message: Error message if constraint violated
    """
    name: str
    predicate: callable
    message: str

    def check(self, coord: PartitionCoordinate | SCoordinate) -> bool:
        """Check if coordinate satisfies this constraint."""
        return self.predicate(coord)

    def validate(self, coord: PartitionCoordinate | SCoordinate) -> None:
        """Raise ValueError if constraint violated."""
        if not self.check(coord):
            raise ValueError(f"Constraint '{self.name}' violated: {self.message}")


# Standard geometric constraints
PARTITION_CONSTRAINTS = [
    CoordinateConstraint(
        "depth_positive",
        lambda c: c.n >= 1,
        "Partition depth n must be >= 1"
    ),
    CoordinateConstraint(
        "complexity_bounded",
        lambda c: 0 <= c.l < c.n,
        "Angular complexity l must be in [0, n-1]"
    ),
    CoordinateConstraint(
        "orientation_bounded",
        lambda c: -c.l <= c.m <= c.l,
        "Orientation m must be in [-l, l]"
    ),
    CoordinateConstraint(
        "chirality_binary",
        lambda c: c.s in (-0.5, 0.5),
        "Chirality s must be +/- 0.5"
    ),
]


@dataclass
class TerminalState:
    """
    Specification of a desired terminal state.

    This is the "program" in Poincaré computing. Instead of specifying
    initial conditions and laws, we specify what we want to observe
    at the end. The synthesizer then generates the minimal partition
    path that terminates at this state.

    Attributes:
        coordinates: Set of partition coordinates that must be occupied
        s_region: Region in S-space where terminal state must fall
        constraints: Additional constraints the terminal state must satisfy
        metadata: Additional properties (observables, labels, etc.)
    """
    coordinates: FrozenSet[PartitionCoordinate] = field(default_factory=frozenset)
    s_region: Optional[Tuple[SCoordinate, SCoordinate]] = None  # (min, max) bounds
    constraints: List[CoordinateConstraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_partition(cls, *coords: PartitionCoordinate, **metadata) -> TerminalState:
        """Create terminal state from partition coordinates."""
        return cls(
            coordinates=frozenset(coords),
            metadata=metadata
        )

    @classmethod
    def from_s_region(
        cls,
        min_coord: SCoordinate,
        max_coord: SCoordinate,
        **metadata
    ) -> TerminalState:
        """Create terminal state from S-space region."""
        return cls(
            s_region=(min_coord, max_coord),
            metadata=metadata
        )

    @classmethod
    def ground_state(cls, n: int = 1) -> TerminalState:
        """
        Create a ground state terminal specification.

        The ground state is the lowest energy configuration:
        n=1, l=0, m=0, s=±1/2
        """
        coords = [
            PartitionCoordinate(n=1, l=0, m=0, s=0.5),
            PartitionCoordinate(n=1, l=0, m=0, s=-0.5),
        ][:n]  # Take n electrons
        return cls.from_partition(*coords, label="ground_state")

    def contains(self, coord: PartitionCoordinate | SCoordinate) -> bool:
        """Check if a coordinate is within this terminal state."""
        if isinstance(coord, PartitionCoordinate):
            return coord in self.coordinates
        elif isinstance(coord, SCoordinate) and self.s_region:
            min_s, max_s = self.s_region
            return (
                min_s.S_k <= coord.S_k <= max_s.S_k and
                min_s.S_t <= coord.S_t <= max_s.S_t and
                min_s.S_e <= coord.S_e <= max_s.S_e
            )
        return False

    def validate(self) -> bool:
        """Check if terminal state specification is valid."""
        # Check all coordinates satisfy geometric constraints
        for coord in self.coordinates:
            for constraint in PARTITION_CONSTRAINTS:
                if not constraint.check(coord):
                    return False

        # Check Pauli exclusion (no duplicate coordinates)
        if len(self.coordinates) != len(set(self.coordinates)):
            return False

        # Check custom constraints
        for constraint in self.constraints:
            for coord in self.coordinates:
                if not constraint.check(coord):
                    return False

        return True

    @property
    def electron_count(self) -> int:
        """Number of electrons (occupied partition coordinates)."""
        return len(self.coordinates)

    @property
    def total_energy(self) -> float:
        """Sum of energies of all occupied coordinates."""
        return sum(c.energy for c in self.coordinates)

    def to_s_coordinates(self) -> List[SCoordinate]:
        """Convert all partition coordinates to S-space."""
        return [SCoordinate.from_partition(c) for c in self.coordinates]


# Utility functions

def shell_capacity(n: int) -> int:
    """
    Number of states at partition depth n.

    This is the fundamental capacity theorem: C(n) = 2n².
    Derived from counting (n, l, m, s) combinations with geometric constraints.
    """
    return 2 * n ** 2


def total_capacity(n_max: int) -> int:
    """Total states up to and including depth n_max."""
    return sum(shell_capacity(n) for n in range(1, n_max + 1))


def aufbau_order(max_coords: int = 20) -> List[PartitionCoordinate]:
    """
    Generate partition coordinates in energy-ordered (aufbau) sequence.

    The aufbau principle emerges from energy minimization:
    sort by (n + l, n), then fill each (n, l, m, s) combination.
    """
    coords = []
    n_max = 8  # Sufficient for most needs

    # Generate all valid (n, l) pairs
    nl_pairs = []
    for n in range(1, n_max + 1):
        for l in range(n):
            nl_pairs.append((n + l, n, l))  # (sort_key, n, l)

    # Sort by Madelung rule: (n+l, n)
    nl_pairs.sort()

    # Fill each orbital
    for _, n, l in nl_pairs:
        for m in range(-l, l + 1):
            for s in (0.5, -0.5):
                coords.append(PartitionCoordinate(n=n, l=l, m=m, s=s))
                if len(coords) >= max_coords:
                    return coords

    return coords
