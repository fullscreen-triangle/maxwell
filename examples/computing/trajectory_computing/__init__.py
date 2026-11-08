"""
Trajectory Computing: A Post-Explanatory Programming Paradigm

Computation as trajectory completion in bounded S-entropy phase space.

Core Principles:
1. Triple Equivalence: oscillation ≡ category ≡ partition
2. Trajectory-Position Identity: the address IS the path
3. ε-Boundary Completion: solutions at one categorical step from closure
4. Computing = Verification: same operation, same ε-boundary

Usage:
    from trajectory_computing import (
        TrajectoryRuntime, TrajectoryComputer,
        System, SystemBuilder,
        PartitionCoordinates, SCoord
    )

    # Define system
    system = (SystemBuilder("my_system")
        .entity("ball", "object", position=10.0, velocity=-5.0)
        .entity("ground", "surface", height=0.0)
        .constrain("at_ground", "ball at ground",
                  lambda e: e["ball"].get_property("position") < 0.1)
        .build()
    )

    # Solve (navigate to completion)
    computer = TrajectoryComputer()
    result = computer.solve(system)

    if result.success:
        print(f"Solution found: {result.solution}")
"""

# Core coordinates
from .coordinates import (
    SCoord,
    TritAddress,
    Trit,
    Tryte,
    Axis,
    categorical_distance,
)

# Partition structures
from .partition import (
    Partition,
    PartitionCoordinates,
    PartitionSpace,
    Spin,
)

# Phase-lock network
from .phase_lock import (
    PhaseLockNetwork,
    PhaseLockNode,
    PhaseLockEdge,
    Coupling,
    InteractionType,
    MolecularType,
)

# Morphisms and catalysts
from .morphism import (
    Morphism,
    MorphismChain,
    MorphismType,
    MorphismCategory,
    Catalyst,
    CatalystStage,
)

# Navigation
from .navigator import (
    Navigator,
    NavigationStrategy,
    NavigationState,
    NavigationResult,
    CompletionPredicate,
    CoordinateCompletion,
    SCoordCompletion,
    CustomCompletion,
    TrajectoryPlanner,
)

# Completion detection
from .completion import (
    CompletionCondition,
    CompletionDetector,
    CompletionResult,
    CompletionStatus,
    CoordinateCondition,
    SCoordCondition,
    ConstraintCondition,
    CompositeCondition,
    GoedelianBoundary,
)

# System specification
from .system import (
    System,
    SystemBuilder,
    Entity,
    Property,
    PropertyType,
    Relation,
    RelationType,
    Constraint,
)

# Runtime
from .runtime import (
    TrajectoryRuntime,
    TrajectoryComputer,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimePhase,
    ExecutionResult,
)

__version__ = "0.1.0"
__all__ = [
    # Coordinates
    "SCoord",
    "TritAddress",
    "Trit",
    "Tryte",
    "Axis",
    "categorical_distance",

    # Partitions
    "Partition",
    "PartitionCoordinates",
    "PartitionSpace",
    "Spin",

    # Phase-locks
    "PhaseLockNetwork",
    "PhaseLockNode",
    "PhaseLockEdge",
    "Coupling",
    "InteractionType",
    "MolecularType",

    # Morphisms
    "Morphism",
    "MorphismChain",
    "MorphismType",
    "MorphismCategory",
    "Catalyst",
    "CatalystStage",

    # Navigation
    "Navigator",
    "NavigationStrategy",
    "NavigationState",
    "NavigationResult",
    "CompletionPredicate",
    "CoordinateCompletion",
    "SCoordCompletion",
    "CustomCompletion",
    "TrajectoryPlanner",

    # Completion
    "CompletionCondition",
    "CompletionDetector",
    "CompletionResult",
    "CompletionStatus",
    "CoordinateCondition",
    "SCoordCondition",
    "ConstraintCondition",
    "CompositeCondition",
    "GoedelianBoundary",

    # System
    "System",
    "SystemBuilder",
    "Entity",
    "Property",
    "PropertyType",
    "Relation",
    "RelationType",
    "Constraint",

    # Runtime
    "TrajectoryRuntime",
    "TrajectoryComputer",
    "RuntimeConfig",
    "RuntimeMetrics",
    "RuntimePhase",
    "ExecutionResult",
]
