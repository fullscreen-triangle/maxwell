"""
Problem Translator

Converts problems from human/classical descriptions into categorical form.

The translation process:
1. Parse problem description → extract entities, relations, constraints
2. Build categorical structure → S-entropy manifold representation
3. Define initial position → starting point in hierarchy
4. Define completion criteria → what "solved" looks like

This is like a compiler, but instead of generating instructions,
it generates categorical structures that the processor can navigate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
import numpy as np
import time
import re

# Import memory and processor components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from categorical_memory.s_entropy_address import SCoordinate, SEntropyAddress
from categorical_memory.categorical_hierarchy import CategoricalHierarchy


class ProblemType(Enum):
    """Types of problems the categorical computer can solve."""
    OPTIMIZATION = "optimization"      # Find best solution
    SEARCH = "search"                  # Find specific element
    PATTERN_MATCH = "pattern_match"    # Find matching patterns
    CONSTRAINT = "constraint"          # Satisfy constraints
    BIOLOGICAL = "biological"          # Biological system analysis
    CLASSIFICATION = "classification"  # Categorize entities
    PREDICTION = "prediction"          # Predict outcomes (via completion)
    TRANSFORMATION = "transformation"  # Transform structures


@dataclass
class CategoricalEntity:
    """
    An entity in the categorical problem space.
    
    Entities are the "objects" of the problem - things that have
    categorical identity and relationships.
    """
    name: str
    category: str  # What type of entity
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # S-entropy representation
    coordinate: Optional[SCoordinate] = None
    signature: Optional[np.ndarray] = None
    
    def to_coordinate(self) -> SCoordinate:
        """Convert entity properties to S-coordinate."""
        if self.coordinate:
            return self.coordinate
        
        # Hash properties to generate coordinate
        prop_values = list(self.properties.values())
        
        # S_k: Based on number/variability of properties
        S_k = len(prop_values) * 0.1
        
        # S_t: Based on mean of numeric properties
        numerics = [v for v in prop_values if isinstance(v, (int, float))]
        S_t = np.mean(numerics) if numerics else 0.0
        
        # S_e: Based on entropy of property distribution
        S_e = len(set(str(v) for v in prop_values)) * 0.1
        
        self.coordinate = SCoordinate(S_k=S_k, S_t=float(S_t), S_e=S_e)
        return self.coordinate


@dataclass
class CategoricalRelation:
    """
    A relation between entities in categorical space.
    
    Relations define how entities connect and constrain each other.
    """
    source: str  # Source entity name
    target: str  # Target entity name
    relation_type: str  # Type of relation
    strength: float = 1.0  # Relation strength
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoricalConstraint:
    """
    A constraint that must be satisfied by the solution.
    
    Constraints define what "completion" means - the categorical
    endpoint must satisfy all constraints.
    """
    name: str
    constraint_type: str  # "equality", "inequality", "membership", "structure"
    expression: str  # Constraint expression
    entities: List[str]  # Entities involved
    
    # Compiled form
    _evaluator: Optional[Callable] = field(default=None, repr=False)
    
    def compile(self, context: Dict[str, Any]):
        """Compile constraint into evaluator function."""
        # Create a safe evaluation context
        safe_context = {
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'np': np,
        }
        safe_context.update(context)
        
        def evaluator(state: Dict[str, Any]) -> Tuple[bool, float]:
            """Evaluate constraint, return (satisfied, violation_degree)."""
            try:
                full_context = {**safe_context, **state}
                
                if self.constraint_type == "equality":
                    # expression should be "left == right"
                    parts = self.expression.split("==")
                    if len(parts) == 2:
                        left = eval(parts[0].strip(), {"__builtins__": {}}, full_context)
                        right = eval(parts[1].strip(), {"__builtins__": {}}, full_context)
                        violation = abs(float(left) - float(right))
                        return violation < 1e-6, violation
                        
                elif self.constraint_type == "inequality":
                    # expression should be "left < right" or "left > right"
                    if "<" in self.expression:
                        parts = self.expression.split("<")
                        left = eval(parts[0].strip(), {"__builtins__": {}}, full_context)
                        right = eval(parts[1].strip(), {"__builtins__": {}}, full_context)
                        violation = max(0, float(left) - float(right))
                        return float(left) < float(right), violation
                    elif ">" in self.expression:
                        parts = self.expression.split(">")
                        left = eval(parts[0].strip(), {"__builtins__": {}}, full_context)
                        right = eval(parts[1].strip(), {"__builtins__": {}}, full_context)
                        violation = max(0, float(right) - float(left))
                        return float(left) > float(right), violation
                        
                elif self.constraint_type == "membership":
                    # expression should be "element in collection"
                    result = eval(self.expression, {"__builtins__": {}}, full_context)
                    return bool(result), 0.0 if result else 1.0
                    
                elif self.constraint_type == "structure":
                    # Custom structural constraint
                    result = eval(self.expression, {"__builtins__": {}}, full_context)
                    return bool(result), 0.0 if result else 1.0
                    
                return False, float('inf')
                
            except Exception as e:
                return False, float('inf')
        
        self._evaluator = evaluator
        
    def evaluate(self, state: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate constraint on current state."""
        if self._evaluator:
            return self._evaluator(state)
        return False, float('inf')


@dataclass
class CategoricalProblem:
    """
    A complete problem in categorical form.
    
    This is what the processor operates on - a structured representation
    of entities, relations, constraints, and completion criteria.
    """
    name: str
    problem_type: ProblemType
    
    # Problem structure
    entities: Dict[str, CategoricalEntity] = field(default_factory=dict)
    relations: List[CategoricalRelation] = field(default_factory=list)
    constraints: List[CategoricalConstraint] = field(default_factory=list)
    
    # Initial state
    initial_state: Dict[str, Any] = field(default_factory=dict)
    
    # Completion criteria
    objective: Optional[str] = None  # For optimization
    target: Optional[str] = None     # For search
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    source_description: str = ""
    
    def add_entity(self, name: str, category: str, **properties) -> CategoricalEntity:
        """Add an entity to the problem."""
        entity = CategoricalEntity(name=name, category=category, properties=properties)
        self.entities[name] = entity
        return entity
    
    def add_relation(self, source: str, target: str, relation_type: str, **properties):
        """Add a relation between entities."""
        relation = CategoricalRelation(
            source=source, 
            target=target, 
            relation_type=relation_type,
            properties=properties
        )
        self.relations.append(relation)
        
    def add_constraint(self, name: str, constraint_type: str, expression: str, entities: List[str]):
        """Add a constraint."""
        constraint = CategoricalConstraint(
            name=name,
            constraint_type=constraint_type,
            expression=expression,
            entities=entities
        )
        self.constraints.append(constraint)
        
    def compile(self):
        """Compile all constraints for evaluation."""
        context = {name: entity.properties for name, entity in self.entities.items()}
        for constraint in self.constraints:
            constraint.compile(context)
    
    def to_s_entropy_manifold(self) -> Dict[str, SCoordinate]:
        """
        Convert problem to S-entropy manifold representation.
        
        Each entity becomes a point in S-entropy space.
        Relations become distances/connections between points.
        Constraints become regions to navigate toward/avoid.
        """
        manifold = {}
        
        for name, entity in self.entities.items():
            manifold[name] = entity.to_coordinate()
            
        return manifold
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix of entity relations."""
        n = len(self.entities)
        names = list(self.entities.keys())
        name_to_idx = {name: i for i, name in enumerate(names)}
        
        matrix = np.zeros((n, n))
        
        for rel in self.relations:
            if rel.source in name_to_idx and rel.target in name_to_idx:
                i, j = name_to_idx[rel.source], name_to_idx[rel.target]
                matrix[i, j] = rel.strength
                matrix[j, i] = rel.strength  # Symmetric
                
        return matrix


@dataclass
class CategoricalSolution:
    """
    A solution to a categorical problem.
    
    The solution is not just an answer - it's a complete trajectory
    through S-entropy space that shows HOW the categorical completion
    was reached.
    """
    problem_name: str
    solved: bool
    
    # The solution value(s)
    result: Any = None
    
    # Trajectory through S-entropy space
    trajectory: List[SCoordinate] = field(default_factory=list)
    
    # Constraint satisfaction
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    total_violation: float = 0.0
    
    # Performance metrics
    navigation_steps: int = 0
    completion_time: float = 0.0
    
    # Categorical analysis
    completion_point: Optional[SCoordinate] = None
    categorical_distance: float = 0.0
    
    def summary(self) -> str:
        """Get solution summary."""
        status = "SOLVED" if self.solved else "UNSOLVED"
        satisfied = sum(self.constraints_satisfied.values())
        total = len(self.constraints_satisfied)
        
        return (
            f"{status}: {self.problem_name}\n"
            f"  Result: {self.result}\n"
            f"  Constraints: {satisfied}/{total} satisfied\n"
            f"  Steps: {self.navigation_steps}\n"
            f"  Time: {self.completion_time:.4f}s\n"
        )


class ProblemTranslator:
    """
    Translates problem descriptions into categorical form.
    
    This is the "compiler" that converts human-readable problems
    into structures the categorical processor can navigate.
    """
    
    # Pattern recognizers for different problem types
    OPTIMIZATION_PATTERNS = [
        r'minimize', r'maximize', r'optimize', r'best', r'optimal',
        r'find.*minimum', r'find.*maximum', r'smallest', r'largest'
    ]
    SEARCH_PATTERNS = [
        r'find', r'search', r'locate', r'where', r'which',
        r'identify', r'discover'
    ]
    CONSTRAINT_PATTERNS = [
        r'such that', r'subject to', r'must', r'require', r'constraint',
        r'satisf', r'condition'
    ]
    BIOLOGICAL_PATTERNS = [
        r'protein', r'molecule', r'cell', r'gene', r'enzyme',
        r'metabol', r'pathway', r'receptor', r'binding', r'drug'
    ]
    
    def __init__(self):
        """Initialize the translator."""
        self._compiled_patterns = {
            ProblemType.OPTIMIZATION: [re.compile(p, re.I) for p in self.OPTIMIZATION_PATTERNS],
            ProblemType.SEARCH: [re.compile(p, re.I) for p in self.SEARCH_PATTERNS],
            ProblemType.CONSTRAINT: [re.compile(p, re.I) for p in self.CONSTRAINT_PATTERNS],
            ProblemType.BIOLOGICAL: [re.compile(p, re.I) for p in self.BIOLOGICAL_PATTERNS],
        }
    
    def detect_problem_type(self, description: str) -> ProblemType:
        """Detect the type of problem from description."""
        scores = {ptype: 0 for ptype in ProblemType}
        
        for ptype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(description):
                    scores[ptype] += 1
        
        # Return highest scoring type, default to SEARCH
        max_score = max(scores.values())
        if max_score == 0:
            return ProblemType.SEARCH
            
        for ptype, score in scores.items():
            if score == max_score:
                return ptype
                
        return ProblemType.SEARCH
    
    def translate(
        self, 
        description: str,
        entities: Optional[Dict[str, Dict]] = None,
        constraints: Optional[List[str]] = None,
        objective: Optional[str] = None,
    ) -> CategoricalProblem:
        """
        Translate a problem description into categorical form.
        
        Args:
            description: Natural language problem description
            entities: Optional dict of entity definitions
            constraints: Optional list of constraint expressions
            objective: Optional objective function (for optimization)
            
        Returns:
            A CategoricalProblem ready for the processor
        """
        # Detect problem type
        problem_type = self.detect_problem_type(description)
        
        # Create problem
        problem = CategoricalProblem(
            name=self._generate_name(description),
            problem_type=problem_type,
            source_description=description,
        )
        
        # Add entities
        if entities:
            for name, props in entities.items():
                category = props.pop('category', 'default')
                problem.add_entity(name, category, **props)
        else:
            # Extract entities from description
            self._extract_entities_from_description(problem, description)
        
        # Add constraints
        if constraints:
            for i, expr in enumerate(constraints):
                ctype = self._infer_constraint_type(expr)
                problem.add_constraint(
                    name=f"constraint_{i}",
                    constraint_type=ctype,
                    expression=expr,
                    entities=list(problem.entities.keys())
                )
        
        # Set objective
        if objective:
            problem.objective = objective
        
        # Compile constraints
        problem.compile()
        
        return problem
    
    def _generate_name(self, description: str) -> str:
        """Generate a problem name from description."""
        words = description.split()[:5]
        return "_".join(w.lower() for w in words if w.isalnum())
    
    def _extract_entities_from_description(self, problem: CategoricalProblem, description: str):
        """Extract entities from natural language description."""
        # Simple extraction: look for nouns and quantities
        # In a real system, this would use NLP
        
        # Find numbers with labels
        number_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(\w+)')
        for match in number_pattern.finditer(description):
            value, label = match.groups()
            problem.add_entity(
                name=label,
                category='quantity',
                value=float(value)
            )
    
    def _infer_constraint_type(self, expression: str) -> str:
        """Infer constraint type from expression."""
        if '==' in expression:
            return 'equality'
        elif '<' in expression or '>' in expression:
            return 'inequality'
        elif ' in ' in expression:
            return 'membership'
        else:
            return 'structure'
    
    def translate_optimization(
        self,
        objective: str,
        variables: Dict[str, Tuple[float, float]],  # name -> (min, max)
        constraints: Optional[List[str]] = None,
        minimize: bool = True,
    ) -> CategoricalProblem:
        """
        Translate an optimization problem.
        
        Args:
            objective: Objective function expression
            variables: Variable bounds {name: (min, max)}
            constraints: Constraint expressions
            minimize: Whether to minimize (True) or maximize (False)
            
        Returns:
            CategoricalProblem for optimization
        """
        problem = CategoricalProblem(
            name=f"optimize_{objective[:20]}",
            problem_type=ProblemType.OPTIMIZATION,
            objective=objective,
        )
        
        # Variables become entities
        for name, (lo, hi) in variables.items():
            problem.add_entity(
                name=name,
                category='variable',
                lower_bound=lo,
                upper_bound=hi,
                current_value=(lo + hi) / 2  # Start in middle
            )
        
        # Add constraints
        if constraints:
            for i, expr in enumerate(constraints):
                ctype = self._infer_constraint_type(expr)
                problem.add_constraint(
                    name=f"c{i}",
                    constraint_type=ctype,
                    expression=expr,
                    entities=list(variables.keys())
                )
        
        problem.initial_state = {
            'minimize': minimize,
            'variables': {n: (lo + hi) / 2 for n, (lo, hi) in variables.items()}
        }
        
        problem.compile()
        return problem
    
    def translate_search(
        self,
        search_space: List[Any],
        target_condition: str,
        space_name: str = "items",
    ) -> CategoricalProblem:
        """
        Translate a search problem.
        
        Args:
            search_space: Items to search through
            target_condition: Condition the target must satisfy
            space_name: Name for the search space
            
        Returns:
            CategoricalProblem for search
        """
        problem = CategoricalProblem(
            name=f"search_{space_name}",
            problem_type=ProblemType.SEARCH,
            target=target_condition,
        )
        
        # Search space becomes entities
        for i, item in enumerate(search_space):
            if isinstance(item, dict):
                problem.add_entity(f"{space_name}_{i}", 'item', **item)
            else:
                problem.add_entity(f"{space_name}_{i}", 'item', value=item)
        
        # Target becomes constraint
        problem.add_constraint(
            name="target",
            constraint_type="structure",
            expression=target_condition,
            entities=list(problem.entities.keys())
        )
        
        problem.compile()
        return problem
    
    def translate_pattern_match(
        self,
        pattern: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> CategoricalProblem:
        """
        Translate a pattern matching problem.
        
        Args:
            pattern: Pattern to match (as property dict)
            candidates: Candidate items to match against
            
        Returns:
            CategoricalProblem for pattern matching
        """
        problem = CategoricalProblem(
            name="pattern_match",
            problem_type=ProblemType.PATTERN_MATCH,
        )
        
        # Pattern becomes reference entity
        pattern_copy = pattern.copy()
        pattern_copy.pop('name', None)  # Remove 'name' to avoid conflict
        problem.add_entity("pattern", "reference", **pattern_copy)
        
        # Candidates become entities to match
        for i, cand in enumerate(candidates):
            cand_copy = cand.copy()
            # Remove 'name' if present to avoid conflict with entity name parameter
            cand_copy.pop('name', None)
            problem.add_entity(f"candidate_{i}", "candidate", **cand_copy)
            
            # Add similarity relation
            problem.add_relation(
                source="pattern",
                target=f"candidate_{i}",
                relation_type="similarity"
            )
        
        problem.compile()
        return problem
    
    def translate_biological(
        self,
        molecules: List[Dict[str, Any]],
        interactions: List[Tuple[str, str, str]],  # (mol1, mol2, type)
        target_property: Optional[str] = None,
    ) -> CategoricalProblem:
        """
        Translate a biological problem.
        
        Args:
            molecules: Molecule definitions
            interactions: Known interactions (mol1, mol2, type)
            target_property: Property to optimize/find
            
        Returns:
            CategoricalProblem for biological analysis
        """
        problem = CategoricalProblem(
            name="biological_system",
            problem_type=ProblemType.BIOLOGICAL,
            objective=target_property,
        )
        
        # Molecules become entities
        for mol in molecules:
            mol_copy = mol.copy()
            name = mol_copy.pop('name', f"mol_{len(problem.entities)}")
            mol_copy.pop('category', None)  # Remove to avoid conflict
            problem.add_entity(name, "molecule", **mol_copy)
        
        # Interactions become relations
        for mol1, mol2, itype in interactions:
            problem.add_relation(mol1, mol2, itype)
        
        problem.compile()
        return problem

