"""
Poincaré Computing Primitives

Fundamental building blocks for the compiled physics language.
These primitives enable backward synthesis from terminal states
rather than forward simulation from initial conditions.

Core Concepts:
- Coordinates: Addressing in partition and categorical space
- Operations: Partition, compose, traverse (irreversible)
- Catalysts: Enable transitions without consumption
- Synthesis: Terminal state → minimal partition path
"""

from .coordinates import (
    PartitionCoordinate,
    SCoordinate,
    TerminalState,
    CoordinateConstraint,
)

from .operations import (
    PartitionOp,
    ComposeOp,
    TraverseOp,
    OperationResult,
)

from .catalysts import (
    InformationCatalyst,
    Observer,
    Selector,
    Aligner,
    CatalystRegistry,
)

from .synthesis import (
    TerminalSpec,
    PartitionPath,
    SynthesisResult,
    Synthesizer,
)

__all__ = [
    # Coordinates
    "PartitionCoordinate",
    "SCoordinate",
    "TerminalState",
    "CoordinateConstraint",
    # Operations
    "PartitionOp",
    "ComposeOp",
    "TraverseOp",
    "OperationResult",
    # Catalysts
    "InformationCatalyst",
    "Observer",
    "Selector",
    "Aligner",
    "CatalystRegistry",
    # Synthesis
    "TerminalSpec",
    "PartitionPath",
    "SynthesisResult",
    "Synthesizer",
]
