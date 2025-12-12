"""
Problem Type Helpers

Convenience classes for creating specific types of categorical problems.
These provide structured interfaces for common problem types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np

from .problem_translator import (
    ProblemTranslator, 
    CategoricalProblem, 
    ProblemType,
    CategoricalEntity,
)


class OptimizationProblem:
    """
    Helper for creating optimization problems.
    
    Example:
        problem = OptimizationProblem.create(
            objective="x**2 + y**2",
            variables={'x': (-5, 5), 'y': (-5, 5)},
            constraints=["x + y > 0"],
            minimize=True
        )
    """
    
    @staticmethod
    def create(
        objective: str,
        variables: Dict[str, Tuple[float, float]],
        constraints: Optional[List[str]] = None,
        minimize: bool = True,
        name: Optional[str] = None,
    ) -> CategoricalProblem:
        """Create an optimization problem."""
        translator = ProblemTranslator()
        problem = translator.translate_optimization(
            objective=objective,
            variables=variables,
            constraints=constraints,
            minimize=minimize
        )
        if name:
            problem.name = name
        return problem
    
    @staticmethod
    def quadratic(
        A: np.ndarray,  # Quadratic term
        b: np.ndarray,  # Linear term
        bounds: Optional[Tuple[float, float]] = None,
    ) -> CategoricalProblem:
        """
        Create a quadratic optimization problem: min x'Ax + b'x
        """
        n = len(b)
        variables = {f"x{i}": bounds or (-10, 10) for i in range(n)}
        
        # Build objective string
        terms = []
        for i in range(n):
            for j in range(n):
                if abs(A[i, j]) > 1e-10:
                    terms.append(f"{A[i,j]}*x{i}*x{j}")
        for i in range(n):
            if abs(b[i]) > 1e-10:
                terms.append(f"{b[i]}*x{i}")
        
        objective = " + ".join(terms) if terms else "0"
        
        return OptimizationProblem.create(
            objective=objective,
            variables=variables,
            minimize=True,
            name="quadratic_optimization"
        )


class SearchProblem:
    """
    Helper for creating search problems.
    
    Example:
        problem = SearchProblem.create(
            items=[1, 4, 9, 16, 25],
            condition="item == 16"
        )
    """
    
    @staticmethod
    def create(
        items: List[Any],
        condition: str,
        name: Optional[str] = None,
    ) -> CategoricalProblem:
        """Create a search problem."""
        translator = ProblemTranslator()
        problem = translator.translate_search(
            search_space=items,
            target_condition=condition,
        )
        if name:
            problem.name = name
        return problem
    
    @staticmethod
    def find_in_sorted(
        items: List[float],
        target: float,
    ) -> CategoricalProblem:
        """Search for target in sorted list."""
        return SearchProblem.create(
            items=items,
            condition=f"abs(item - {target}) < 0.001",
            name="sorted_search"
        )
    
    @staticmethod
    def find_by_property(
        items: List[Dict[str, Any]],
        property_name: str,
        property_value: Any,
    ) -> CategoricalProblem:
        """Search for item with specific property value."""
        condition = f"item.get('{property_name}') == {repr(property_value)}"
        return SearchProblem.create(
            items=items,
            condition=condition,
            name=f"search_by_{property_name}"
        )


class PatternMatchProblem:
    """
    Helper for creating pattern matching problems.
    
    Example:
        problem = PatternMatchProblem.create(
            pattern={'type': 'protein', 'active': True},
            candidates=[
                {'type': 'protein', 'active': True, 'name': 'kinase'},
                {'type': 'lipid', 'active': False, 'name': 'cholesterol'},
            ]
        )
    """
    
    @staticmethod
    def create(
        pattern: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        name: Optional[str] = None,
    ) -> CategoricalProblem:
        """Create a pattern matching problem."""
        translator = ProblemTranslator()
        problem = translator.translate_pattern_match(
            pattern=pattern,
            candidates=candidates
        )
        if name:
            problem.name = name
        return problem
    
    @staticmethod
    def fuzzy_match(
        template: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        required_keys: Optional[List[str]] = None,
    ) -> CategoricalProblem:
        """
        Fuzzy pattern matching where only required_keys must match exactly.
        """
        if required_keys:
            pattern = {k: v for k, v in template.items() if k in required_keys}
        else:
            pattern = template
        return PatternMatchProblem.create(pattern, candidates, "fuzzy_match")


class ConstraintProblem:
    """
    Helper for creating constraint satisfaction problems.
    
    Example:
        problem = ConstraintProblem.create(
            variables={'x': (0, 10), 'y': (0, 10), 'z': (0, 10)},
            constraints=[
                "x + y + z == 15",
                "x < y",
                "y < z",
            ]
        )
    """
    
    @staticmethod
    def create(
        variables: Dict[str, Tuple[float, float]],
        constraints: List[str],
        name: Optional[str] = None,
    ) -> CategoricalProblem:
        """Create a constraint satisfaction problem."""
        problem = CategoricalProblem(
            name=name or "constraint_satisfaction",
            problem_type=ProblemType.CONSTRAINT,
        )
        
        # Add variables as entities
        for var_name, (lo, hi) in variables.items():
            problem.add_entity(
                name=var_name,
                category='variable',
                lower_bound=lo,
                upper_bound=hi,
                current_value=(lo + hi) / 2
            )
        
        # Add constraints
        for i, expr in enumerate(constraints):
            # Infer type
            if '==' in expr:
                ctype = 'equality'
            elif '<' in expr or '>' in expr:
                ctype = 'inequality'
            else:
                ctype = 'structure'
            
            problem.add_constraint(
                name=f"c{i}",
                constraint_type=ctype,
                expression=expr,
                entities=list(variables.keys())
            )
        
        problem.compile()
        return problem
    
    @staticmethod
    def n_queens(n: int) -> CategoricalProblem:
        """
        Create an N-Queens constraint problem.
        Variables represent queen positions (row for each column).
        """
        variables = {f"q{i}": (0, n-1) for i in range(n)}
        
        constraints = []
        # No two queens in same row
        for i in range(n):
            for j in range(i+1, n):
                constraints.append(f"q{i} != q{j}")
        
        # No two queens on same diagonal
        for i in range(n):
            for j in range(i+1, n):
                constraints.append(f"abs(q{i} - q{j}) != {j-i}")
        
        return ConstraintProblem.create(variables, constraints, f"{n}_queens")


class BiologicalProblem:
    """
    Helper for creating biological system problems.
    
    These problems model molecular systems, interactions, and pathways.
    
    Example:
        problem = BiologicalProblem.create_binding(
            ligand={'name': 'drug_x', 'mass': 500, 'charge': -1},
            receptor={'name': 'target_receptor', 'binding_site': 'alpha'},
        )
    """
    
    @staticmethod
    def create(
        molecules: List[Dict[str, Any]],
        interactions: List[Tuple[str, str, str]],
        target_property: Optional[str] = None,
        name: Optional[str] = None,
    ) -> CategoricalProblem:
        """Create a biological problem."""
        translator = ProblemTranslator()
        problem = translator.translate_biological(
            molecules=molecules,
            interactions=interactions,
            target_property=target_property
        )
        if name:
            problem.name = name
        return problem
    
    @staticmethod
    def create_binding(
        ligand: Dict[str, Any],
        receptor: Dict[str, Any],
        binding_affinity_constraint: Optional[str] = None,
    ) -> CategoricalProblem:
        """
        Create a ligand-receptor binding problem.
        """
        ligand_copy = ligand.copy()
        receptor_copy = receptor.copy()
        
        ligand_name = ligand_copy.pop('name', 'ligand')
        receptor_name = receptor_copy.pop('name', 'receptor')
        
        molecules = [
            {'name': ligand_name, 'category': 'ligand', **ligand_copy},
            {'name': receptor_name, 'category': 'receptor', **receptor_copy},
        ]
        
        interactions = [
            (ligand_name, receptor_name, 'binding'),
        ]
        
        problem = BiologicalProblem.create(
            molecules=molecules,
            interactions=interactions,
            target_property='binding_affinity',
            name='ligand_receptor_binding'
        )
        
        if binding_affinity_constraint:
            problem.add_constraint(
                name="affinity",
                constraint_type="inequality",
                expression=binding_affinity_constraint,
                entities=[ligand_name, receptor_name]
            )
            problem.compile()
        
        return problem
    
    @staticmethod
    def create_pathway(
        enzymes: List[Dict[str, Any]],
        substrates: List[Dict[str, Any]],
        reactions: List[Tuple[str, str, str, str]],  # (enzyme, substrate, product, rate)
    ) -> CategoricalProblem:
        """
        Create a metabolic pathway problem.
        
        Args:
            enzymes: Enzyme definitions
            substrates: Substrate/metabolite definitions
            reactions: (enzyme, substrate, product, rate_constant) tuples
        """
        molecules = []
        interactions = []
        
        # Add enzymes
        for enzyme in enzymes:
            enzyme_copy = enzyme.copy()
            enzyme_copy['category'] = 'enzyme'
            molecules.append(enzyme_copy)
        
        # Add substrates
        for substrate in substrates:
            sub_copy = substrate.copy()
            sub_copy['category'] = 'substrate'
            molecules.append(sub_copy)
        
        # Add reactions as interactions
        for enzyme, substrate, product, rate in reactions:
            interactions.append((enzyme, substrate, f'catalyzes_{rate}'))
            interactions.append((substrate, product, f'converts_{rate}'))
        
        return BiologicalProblem.create(
            molecules=molecules,
            interactions=interactions,
            target_property='pathway_flux',
            name='metabolic_pathway'
        )
    
    @staticmethod
    def create_protein_folding(
        sequence: str,
        known_structures: Optional[List[str]] = None,
    ) -> CategoricalProblem:
        """
        Create a protein folding problem (simplified categorical model).
        
        This represents folding as navigating through conformational S-entropy space.
        """
        problem = CategoricalProblem(
            name="protein_folding",
            problem_type=ProblemType.BIOLOGICAL,
        )
        
        # Each amino acid is an entity
        for i, aa in enumerate(sequence):
            problem.add_entity(
                name=f"aa_{i}",
                category='amino_acid',
                residue=aa,
                position=i,
                phi=0.0,  # Backbone angles to optimize
                psi=0.0,
            )
        
        # Add backbone connectivity
        for i in range(len(sequence) - 1):
            problem.add_relation(
                source=f"aa_{i}",
                target=f"aa_{i+1}",
                relation_type='backbone'
            )
        
        # Add constraints for valid angles
        for i in range(len(sequence)):
            problem.add_constraint(
                name=f"phi_{i}_range",
                constraint_type='inequality',
                expression=f"-180 < phi_{i} < 180",
                entities=[f"aa_{i}"]
            )
            problem.add_constraint(
                name=f"psi_{i}_range",
                constraint_type='inequality',
                expression=f"-180 < psi_{i} < 180",
                entities=[f"aa_{i}"]
            )
        
        problem.objective = "minimize_energy"
        problem.initial_state = {
            'sequence': sequence,
            'known_structures': known_structures or [],
        }
        
        problem.compile()
        return problem


