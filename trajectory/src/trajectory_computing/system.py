"""
System Specification: Entities, Relations, and Constraints

A system in trajectory computing is specified declaratively:
- Entities: The objects that exist
- Relations: How entities connect
- Constraints: What must be true of the solution

The system specification defines the COMPLETION CONDITION.
The runtime navigates TO that condition.

Example (ball on ground):
  entity ball: position, velocity
  entity ground: surface
  relation contact: ball.position.z = ground.surface.z
  constraint gravity: ball.velocity = -9.8 * t
  complete when: ball at ground with velocity = 0

Traditional approach: Simulate forward from initial conditions
Trajectory approach: Navigate backward from completion condition
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .coordinates import SCoord, TritAddress
from .partition import Partition, PartitionCoordinates, PartitionSpace, Spin
from .completion import CompletionCondition, ConstraintCondition, CompletionResult


class PropertyType(Enum):
    """Types of entity properties."""
    SCALAR = "scalar"       # Single value
    VECTOR = "vector"       # [x, y, z]
    TENSOR = "tensor"       # Matrix
    CATEGORICAL = "cat"     # Discrete category
    BOOLEAN = "bool"        # True/False


@dataclass
class Property:
    """A property of an entity."""
    name: str
    property_type: PropertyType
    value: Any = None
    units: str = ""
    bounds: Optional[Tuple[float, float]] = None

    def validate(self) -> bool:
        """Validate property value."""
        if self.bounds and self.property_type == PropertyType.SCALAR:
            return self.bounds[0] <= self.value <= self.bounds[1]
        return True


@dataclass
class Entity:
    """
    An entity in the system.

    Entities are the objects that exist - they have properties
    and can participate in relations.
    """
    id: str
    entity_type: str
    properties: Dict[str, Property] = field(default_factory=dict)
    partition: Optional[Partition] = None  # Location in partition space
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_property(self, name: str, ptype: PropertyType,
                    value: Any = None, units: str = "",
                    bounds: Optional[Tuple[float, float]] = None) -> Property:
        """Add a property to this entity."""
        prop = Property(name, ptype, value, units, bounds)
        self.properties[name] = prop
        return prop

    def get_property(self, name: str) -> Optional[Any]:
        """Get property value."""
        prop = self.properties.get(name)
        return prop.value if prop else None

    def set_property(self, name: str, value: Any) -> None:
        """Set property value."""
        if name in self.properties:
            self.properties[name].value = value
        else:
            raise KeyError(f"Property {name} not found")

    def to_scoord(self) -> SCoord:
        """Convert entity state to S-coordinates."""
        # Map properties to [0,1]^3
        # This is domain-specific - here we use a simple default
        props = list(self.properties.values())
        values = []

        for prop in props[:3]:  # Use first 3 properties
            if prop.property_type == PropertyType.SCALAR:
                if prop.bounds:
                    normalized = (prop.value - prop.bounds[0]) / (prop.bounds[1] - prop.bounds[0])
                else:
                    normalized = 0.5  # Default if no bounds
                values.append(np.clip(normalized, 0, 1))
            else:
                values.append(0.5)

        # Pad to 3 dimensions
        while len(values) < 3:
            values.append(0.5)

        return SCoord(s_k=values[0], s_t=values[1], s_e=values[2])


class RelationType(Enum):
    """Types of relations between entities."""
    EQUALITY = "eq"         # a = b
    INEQUALITY = "neq"      # a ≠ b
    LESS_THAN = "lt"        # a < b
    GREATER_THAN = "gt"     # a > b
    CONTACT = "contact"     # Physical contact
    COUPLING = "coupling"   # Phase-lock coupling
    CAUSATION = "cause"     # a causes b
    MEMBERSHIP = "member"   # a ∈ b


@dataclass
class Relation:
    """
    A relation between entities.

    Relations define how entities connect and constrain each other.
    """
    id: str
    relation_type: RelationType
    source_id: str
    target_id: str
    source_property: Optional[str] = None
    target_property: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def as_predicate(self, entities: Dict[str, Entity]) -> Callable[[], bool]:
        """Convert relation to a predicate function."""
        source = entities.get(self.source_id)
        target = entities.get(self.target_id)

        if source is None or target is None:
            return lambda: False

        def check() -> bool:
            s_val = source.get_property(self.source_property) if self.source_property else None
            t_val = target.get_property(self.target_property) if self.target_property else None

            if self.relation_type == RelationType.EQUALITY:
                return s_val == t_val
            elif self.relation_type == RelationType.INEQUALITY:
                return s_val != t_val
            elif self.relation_type == RelationType.LESS_THAN:
                return s_val < t_val
            elif self.relation_type == RelationType.GREATER_THAN:
                return s_val > t_val
            elif self.relation_type == RelationType.CONTACT:
                # Contact means within epsilon distance
                epsilon = self.parameters.get("epsilon", 0.01)
                return abs(s_val - t_val) < epsilon
            else:
                return True

        return check


@dataclass
class Constraint:
    """
    A constraint on the system.

    Constraints define what must be true - they are part of
    the completion condition.
    """
    id: str
    name: str
    predicate: Callable[[Dict[str, Entity]], bool]
    description: str = ""
    weight: float = 1.0  # Importance for optimization

    def is_satisfied(self, entities: Dict[str, Entity]) -> bool:
        """Check if constraint is satisfied."""
        try:
            return self.predicate(entities)
        except Exception:
            return False


class System:
    """
    A system specification in trajectory computing.

    The system defines:
    - What exists (entities)
    - How things connect (relations)
    - What must be true (constraints)

    The system specification IS the completion condition.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.partition_space: PartitionSpace = PartitionSpace()

    # Entity management
    def add_entity(self, entity_id: str, entity_type: str,
                  **properties: Any) -> Entity:
        """Add an entity to the system."""
        entity = Entity(id=entity_id, entity_type=entity_type)

        for name, value in properties.items():
            if isinstance(value, tuple) and len(value) == 2:
                # (value, type)
                entity.add_property(name, value[1], value[0])
            else:
                # Infer type
                if isinstance(value, bool):
                    entity.add_property(name, PropertyType.BOOLEAN, value)
                elif isinstance(value, (int, float)):
                    entity.add_property(name, PropertyType.SCALAR, value)
                elif isinstance(value, (list, np.ndarray)):
                    entity.add_property(name, PropertyType.VECTOR, value)
                else:
                    entity.add_property(name, PropertyType.CATEGORICAL, value)

        self.entities[entity_id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    # Relation management
    def add_relation(self, relation_id: str, rtype: RelationType,
                    source: str, target: str,
                    source_prop: Optional[str] = None,
                    target_prop: Optional[str] = None,
                    **params: Any) -> Relation:
        """Add a relation between entities."""
        relation = Relation(
            id=relation_id,
            relation_type=rtype,
            source_id=source,
            target_id=target,
            source_property=source_prop,
            target_property=target_prop,
            parameters=params
        )
        self.relations[relation_id] = relation
        return relation

    # Constraint management
    def add_constraint(self, constraint_id: str, name: str,
                      predicate: Callable[[Dict[str, Entity]], bool],
                      description: str = "",
                      weight: float = 1.0) -> Constraint:
        """Add a constraint to the system."""
        constraint = Constraint(
            id=constraint_id,
            name=name,
            predicate=predicate,
            description=description,
            weight=weight
        )
        self.constraints[constraint_id] = constraint
        return constraint

    # Completion condition
    def as_completion_condition(self) -> CompletionCondition:
        """Convert system specification to completion condition."""
        condition = ConstraintCondition()

        # Add relation-based constraints
        for rel in self.relations.values():
            predicate = rel.as_predicate(self.entities)
            condition.add_constraint(
                lambda p, pred=predicate: pred(),
                f"relation_{rel.id}"
            )

        # Add explicit constraints
        for const in self.constraints.values():
            condition.add_constraint(
                lambda p, c=const: c.is_satisfied(self.entities),
                const.name
            )

        return condition

    def check_completion(self) -> Dict[str, bool]:
        """Check which constraints are satisfied."""
        results = {}

        for rel_id, rel in self.relations.items():
            predicate = rel.as_predicate(self.entities)
            results[f"relation_{rel_id}"] = predicate()

        for const_id, const in self.constraints.items():
            results[const.name] = const.is_satisfied(self.entities)

        return results

    def completion_distance(self) -> int:
        """Number of unsatisfied constraints."""
        results = self.check_completion()
        return sum(1 for satisfied in results.values() if not satisfied)

    def is_complete(self) -> bool:
        """Check if all constraints are satisfied."""
        return self.completion_distance() == 0

    # State management
    def state_vector(self) -> np.ndarray:
        """Get system state as vector."""
        values = []
        for entity in self.entities.values():
            for prop in entity.properties.values():
                if prop.property_type == PropertyType.SCALAR:
                    values.append(float(prop.value or 0))
                elif prop.property_type == PropertyType.VECTOR:
                    values.extend(prop.value or [0, 0, 0])
        return np.array(values)

    def to_scoord(self) -> SCoord:
        """Convert system state to S-coordinates."""
        state = self.state_vector()
        if len(state) < 3:
            state = np.pad(state, (0, 3 - len(state)))

        # Normalize to [0, 1]
        state_min = state.min()
        state_max = state.max()
        if state_max > state_min:
            normalized = (state[:3] - state_min) / (state_max - state_min)
        else:
            normalized = np.array([0.5, 0.5, 0.5])

        return SCoord(
            s_k=np.clip(normalized[0], 0, 1),
            s_t=np.clip(normalized[1], 0, 1),
            s_e=np.clip(normalized[2], 0, 1)
        )


class SystemBuilder:
    """
    Fluent builder for system specifications.

    Provides a clean API for defining systems.
    """

    def __init__(self, name: str):
        self.system = System(name)

    def entity(self, entity_id: str, entity_type: str,
              **properties: Any) -> SystemBuilder:
        """Add an entity."""
        self.system.add_entity(entity_id, entity_type, **properties)
        return self

    def relate(self, relation_id: str, rtype: RelationType,
              source: str, target: str,
              source_prop: Optional[str] = None,
              target_prop: Optional[str] = None,
              **params: Any) -> SystemBuilder:
        """Add a relation."""
        self.system.add_relation(
            relation_id, rtype, source, target,
            source_prop, target_prop, **params
        )
        return self

    def constrain(self, constraint_id: str, name: str,
                 predicate: Callable[[Dict[str, Entity]], bool],
                 description: str = "") -> SystemBuilder:
        """Add a constraint."""
        self.system.add_constraint(constraint_id, name, predicate, description)
        return self

    def build(self) -> System:
        """Build the system."""
        return self.system


# Demonstration: Ball on Ground
def demo_ball_on_ground():
    """
    Example: Ball on ground problem.

    Traditional: Start with ball at height h, simulate gravity
    Trajectory: Specify completion (ball at ground, velocity = 0),
                navigate to that state
    """
    # Build system specification
    system = (SystemBuilder("ball_on_ground")
        .entity("ball", "object",
                position_z=10.0,  # Initial: 10m high
                velocity_z=-5.0)  # Initial: falling
        .entity("ground", "surface",
                height=0.0)
        .relate("contact", RelationType.CONTACT,
                "ball", "ground",
                source_prop="position_z",
                target_prop="height",
                epsilon=0.01)
        .constrain("at_rest", "ball at rest",
                  lambda e: abs(e["ball"].get_property("velocity_z") or 0) < 0.1,
                  "Ball velocity near zero")
        .build()
    )

    print("Ball on Ground System:")
    print(f"  Entities: {list(system.entities.keys())}")
    print(f"  Relations: {list(system.relations.keys())}")
    print(f"  Constraints: {list(system.constraints.keys())}")

    # Check initial state
    print(f"\nInitial State:")
    print(f"  Ball position: {system.entities['ball'].get_property('position_z')}")
    print(f"  Ball velocity: {system.entities['ball'].get_property('velocity_z')}")
    print(f"  Completion status: {system.check_completion()}")
    print(f"  Distance to completion: {system.completion_distance()}")

    # Set final state (what we're navigating TO)
    system.entities["ball"].set_property("position_z", 0.0)
    system.entities["ball"].set_property("velocity_z", 0.0)

    print(f"\nFinal State (completion):")
    print(f"  Ball position: {system.entities['ball'].get_property('position_z')}")
    print(f"  Ball velocity: {system.entities['ball'].get_property('velocity_z')}")
    print(f"  Completion status: {system.check_completion()}")
    print(f"  Is complete: {system.is_complete()}")

    return system


if __name__ == "__main__":
    demo_ball_on_ground()
