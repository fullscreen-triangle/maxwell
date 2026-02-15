"""
S-Entropy Coordinates and Ternary Addressing

The S-entropy coordinate space S = [0,1]^3 with coordinates:
- S_k: knowledge entropy
- S_t: temporal entropy
- S_e: evolution entropy

Ternary representation maps trits to dimensions:
- 0 → refinement along S_k
- 1 → refinement along S_t
- 2 → refinement along S_e

A trit sequence encodes BOTH position (which cell) AND trajectory (how to get there).
This is not two representations - it IS the same mathematical object.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Iterator, Optional
import numpy as np


class Axis(IntEnum):
    """S-entropy axes corresponding to trit values."""
    K = 0  # Knowledge entropy
    T = 1  # Temporal entropy
    E = 2  # Evolution entropy


class Trit(IntEnum):
    """Ternary digit with values {0, 1, 2} mapping to axes."""
    ZERO = 0   # Refine along S_k
    ONE = 1    # Refine along S_t
    TWO = 2    # Refine along S_e

    @property
    def axis(self) -> Axis:
        """The axis this trit refines."""
        return Axis(self.value)

    @classmethod
    def from_axis(cls, axis: Axis) -> Trit:
        """Create trit from axis."""
        return cls(axis.value)


@dataclass
class SCoord:
    """
    S-Entropy Coordinate in [0,1]^3.

    Represents a point in the bounded S-entropy phase space.
    """
    s_k: float  # Knowledge entropy [0,1]
    s_t: float  # Temporal entropy [0,1]
    s_e: float  # Evolution entropy [0,1]

    def __post_init__(self):
        """Validate coordinates are in [0,1]."""
        for name, val in [('s_k', self.s_k), ('s_t', self.s_t), ('s_e', self.s_e)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {val}")

    def __getitem__(self, axis: Axis) -> float:
        """Get coordinate by axis."""
        if axis == Axis.K:
            return self.s_k
        elif axis == Axis.T:
            return self.s_t
        else:
            return self.s_e

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.s_k, self.s_t, self.s_e])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> SCoord:
        """Create from numpy array."""
        return cls(s_k=arr[0], s_t=arr[1], s_e=arr[2])

    @classmethod
    def origin(cls) -> SCoord:
        """Origin of S-space."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def center(cls) -> SCoord:
        """Center of S-space."""
        return cls(0.5, 0.5, 0.5)

    def distance(self, other: SCoord) -> float:
        """Euclidean distance to another point."""
        return np.linalg.norm(self.to_array() - other.to_array())

    def __add__(self, other: SCoord) -> SCoord:
        """Add coordinates (clamped to [0,1])."""
        return SCoord(
            s_k=np.clip(self.s_k + other.s_k, 0.0, 1.0),
            s_t=np.clip(self.s_t + other.s_t, 0.0, 1.0),
            s_e=np.clip(self.s_e + other.s_e, 0.0, 1.0)
        )

    def __sub__(self, other: SCoord) -> SCoord:
        """Subtract coordinates (clamped to [0,1])."""
        return SCoord(
            s_k=np.clip(self.s_k - other.s_k, 0.0, 1.0),
            s_t=np.clip(self.s_t - other.s_t, 0.0, 1.0),
            s_e=np.clip(self.s_e - other.s_e, 0.0, 1.0)
        )

    def scale(self, factor: float) -> SCoord:
        """Scale coordinates (clamped to [0,1])."""
        return SCoord(
            s_k=np.clip(self.s_k * factor, 0.0, 1.0),
            s_t=np.clip(self.s_t * factor, 0.0, 1.0),
            s_e=np.clip(self.s_e * factor, 0.0, 1.0)
        )


@dataclass
class TritAddress:
    """
    Ternary address encoding both position and trajectory.

    A k-trit string addresses one of 3^k cells in S-space.
    The sequence of trits specifies:
    1. The cell (position)
    2. The refinement path (trajectory)

    These are THE SAME - not two views, but one object.
    """
    trits: List[Trit] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Number of trits (partition depth)."""
        return len(self.trits)

    @property
    def cell_count(self) -> int:
        """Number of cells at this depth: 3^k."""
        return 3 ** self.depth

    def to_scoord(self) -> SCoord:
        """
        Convert to S-coordinate.

        Each trit refines the coordinate along its axis by factor 3.
        The sequence determines the final position.
        """
        coords = [0.0, 0.0, 0.0]  # [s_k, s_t, s_e]
        scales = [1.0, 1.0, 1.0]   # Current scale for each axis

        for trit in self.trits:
            axis = trit.axis.value
            scales[axis] /= 3.0
            # Add (trit_value * scale) to determine position within subdivision
            # trit=0: lower third, trit=1: middle third, trit=2: upper third
            # But for our mapping, trit value IS the axis, so we use different logic:
            # Each trit refines along ONE axis, adding 1/3^depth to that coordinate
            coords[axis] += scales[axis]

        return SCoord(s_k=coords[0], s_t=coords[1], s_e=coords[2])

    @classmethod
    def from_scoord(cls, coord: SCoord, depth: int) -> TritAddress:
        """
        Convert S-coordinate to trit address at given depth.

        This is approximate for finite depth - the continuous coordinate
        maps to the cell containing it.
        """
        trits = []
        remaining = [coord.s_k, coord.s_t, coord.s_e]

        for _ in range(depth):
            # Find which axis has the largest remaining value
            # This determines the next trit
            max_axis = int(np.argmax(remaining))
            trits.append(Trit(max_axis))
            remaining[max_axis] -= 1.0 / (3 ** (len(trits)))
            remaining = [max(0.0, r) for r in remaining]

        return cls(trits=trits)

    def refine(self, trit: Trit) -> TritAddress:
        """
        Refine by appending a trit.

        This extends the trajectory AND specifies a more precise position.
        """
        return TritAddress(trits=self.trits + [trit])

    def parent(self) -> Optional[TritAddress]:
        """Get parent cell (remove last trit)."""
        if self.depth == 0:
            return None
        return TritAddress(trits=self.trits[:-1])

    def children(self) -> List[TritAddress]:
        """Get all 3 child cells."""
        return [self.refine(Trit(i)) for i in range(3)]

    def __str__(self) -> str:
        """String representation: 0t followed by trits."""
        return "0t" + "".join(str(t.value) for t in self.trits)

    @classmethod
    def from_string(cls, s: str) -> TritAddress:
        """Parse from string like '0t012021'."""
        if s.startswith("0t"):
            s = s[2:]
        trits = [Trit(int(c)) for c in s]
        return cls(trits=trits)

    def as_trajectory(self) -> List[SCoord]:
        """
        Return the trajectory encoded by this address.

        Each prefix of the trit sequence gives an intermediate position.
        """
        trajectory = [SCoord.origin()]
        for i in range(1, self.depth + 1):
            partial = TritAddress(trits=self.trits[:i])
            trajectory.append(partial.to_scoord())
        return trajectory


@dataclass
class Tryte:
    """
    Ternary byte: 6 trits encoding 3^6 = 729 cells.

    Compare to binary byte: 8 bits encoding 2^8 = 256 values.
    Tryte is more information-dense for 3D navigation.
    """
    trits: Tuple[Trit, Trit, Trit, Trit, Trit, Trit]

    @classmethod
    def from_address(cls, addr: TritAddress) -> Tryte:
        """Extract first 6 trits as tryte."""
        if addr.depth < 6:
            padded = addr.trits + [Trit.ZERO] * (6 - addr.depth)
        else:
            padded = addr.trits[:6]
        return cls(trits=tuple(padded))

    def to_int(self) -> int:
        """Convert to integer [0, 728]."""
        result = 0
        for i, t in enumerate(self.trits):
            result += t.value * (3 ** (5 - i))
        return result

    @classmethod
    def from_int(cls, n: int) -> Tryte:
        """Create from integer [0, 728]."""
        if not 0 <= n <= 728:
            raise ValueError(f"Tryte integer must be in [0, 728], got {n}")
        trits = []
        for _ in range(6):
            trits.append(Trit(n % 3))
            n //= 3
        return cls(trits=tuple(reversed(trits)))


def categorical_distance(addr1: TritAddress, addr2: TritAddress) -> int:
    """
    Categorical distance between two addresses.

    This is the minimum number of categorical completions (trit changes)
    to transform addr1 into addr2.
    """
    # Pad to same length
    max_depth = max(addr1.depth, addr2.depth)
    trits1 = addr1.trits + [Trit.ZERO] * (max_depth - addr1.depth)
    trits2 = addr2.trits + [Trit.ZERO] * (max_depth - addr2.depth)

    # Count differing trits
    return sum(1 for t1, t2 in zip(trits1, trits2) if t1 != t2)


# Demonstration of trajectory-position identity
if __name__ == "__main__":
    # Create an address
    addr = TritAddress.from_string("0t012102")

    print(f"Address: {addr}")
    print(f"Depth: {addr.depth}")
    print(f"Cell count at depth: {addr.cell_count}")

    # The address IS both position and trajectory
    position = addr.to_scoord()
    trajectory = addr.as_trajectory()

    print(f"\nPosition (S-coordinate): ({position.s_k:.4f}, {position.s_t:.4f}, {position.s_e:.4f})")
    print(f"\nTrajectory (sequence of positions):")
    for i, point in enumerate(trajectory):
        print(f"  Step {i}: ({point.s_k:.4f}, {point.s_t:.4f}, {point.s_e:.4f})")

    print("\nThe position and trajectory are THE SAME OBJECT - not two views.")
