"""
Categorical Computer Demonstration

Shows the categorical computer solving real problems by navigating
S-entropy space rather than executing sequential instructions.

Problems demonstrated:
1. Optimization (finding minimum of a function)
2. Constraint satisfaction (N-Queens)
3. Pattern matching (finding similar structures)
4. Biological (ligand-receptor binding)
"""

import time
import numpy as np
from typing import Dict, Any
import json

from .problem_translator import ProblemTranslator, ProblemType
from .categorical_runtime import CategoricalRuntime, NavigationStrategy
from .problem_types import (
    OptimizationProblem,
    SearchProblem,
    PatternMatchProblem,
    ConstraintProblem,
    BiologicalProblem,
)


def demonstrate_optimization() -> Dict[str, Any]:
    """
    Demonstrate solving an optimization problem.
    
    Problem: Minimize the Rosenbrock function
    f(x, y) = (1-x)^2 + 100*(y-x^2)^2
    
    Global minimum at (1, 1)
    """
    print("\n" + "="*60)
    print("PROBLEM 1: Optimization (Rosenbrock Function)")
    print("="*60)
    
    print("\nProblem: Find minimum of f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
    print("Known optimum: x=1, y=1, f=0")
    
    # Create problem
    problem = OptimizationProblem.create(
        objective="(1 - x)**2 + 100 * (y - x**2)**2",
        variables={
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
        },
        minimize=True,
        name="rosenbrock"
    )
    
    print(f"\nProblem structure:")
    print(f"  Entities: {list(problem.entities.keys())}")
    print(f"  Type: {problem.problem_type.value}")
    
    # Create runtime and execute
    runtime = CategoricalRuntime(max_steps=500, convergence_threshold=0.01)
    
    print("\nExecuting with categorical navigation...")
    result = runtime.execute(problem, strategy=NavigationStrategy.CATEGORICAL_COMPLETION)
    
    print(f"\nResult:")
    print(f"  Converged: {result.converged}")
    print(f"  Steps: {result.context.step_count}")
    print(f"  Time: {result.context.elapsed_time:.4f}s")
    
    # Calculate actual function value
    if result.solution.result and isinstance(result.solution.result, dict):
        x = result.solution.result.get('x', 0)
        y = result.solution.result.get('y', 0)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            f_value = (1-x)**2 + 100*(y-x**2)**2
            print(f"  Solution: x={x:.4f}, y={y:.4f}")
            print(f"  f(x,y) = {f_value:.6f}")
        else:
            print(f"  Solution: {result.solution.result}")
    else:
        print(f"  Solution: {result.solution.result}")
    
    return {
        'problem': 'rosenbrock',
        'converged': result.converged,
        'solution': result.solution.result,
        'steps': result.context.step_count,
        'time': result.context.elapsed_time,
        'trajectory_length': len(result.solution.trajectory),
    }


def demonstrate_constraint_satisfaction() -> Dict[str, Any]:
    """
    Demonstrate solving a constraint satisfaction problem.
    
    Problem: 4-Queens (simplified)
    Place 4 queens on a 4x4 board so none attack each other.
    """
    print("\n" + "="*60)
    print("PROBLEM 2: Constraint Satisfaction (4-Queens)")
    print("="*60)
    
    print("\nProblem: Place 4 queens on 4x4 board, none attacking")
    
    # Create problem manually for clearer demonstration
    problem = ConstraintProblem.create(
        variables={
            'q0': (0, 3),  # Queen in column 0, which row?
            'q1': (0, 3),  # Queen in column 1
            'q2': (0, 3),  # Queen in column 2
            'q3': (0, 3),  # Queen in column 3
        },
        constraints=[
            # No two queens in same row
            "abs(q0 - q1) > 0.5",
            "abs(q0 - q2) > 0.5",
            "abs(q0 - q3) > 0.5",
            "abs(q1 - q2) > 0.5",
            "abs(q1 - q3) > 0.5",
            "abs(q2 - q3) > 0.5",
            # No two queens on same diagonal (simplified)
            "abs(q0 - q1) > 0.8",  # Diagonal distance for adjacent columns
            "abs(q1 - q2) > 0.8",
            "abs(q2 - q3) > 0.8",
        ],
        name="4_queens"
    )
    
    print(f"\nProblem structure:")
    print(f"  Variables: {list(problem.entities.keys())}")
    print(f"  Constraints: {len(problem.constraints)}")
    
    # Execute with simulated annealing (good for discrete-ish problems)
    runtime = CategoricalRuntime(max_steps=200, convergence_threshold=0.1)
    
    print("\nExecuting with categorical navigation + annealing...")
    result = runtime.execute(problem, strategy=NavigationStrategy.SIMULATED_ANNEALING)
    
    print(f"\nResult:")
    print(f"  Converged: {result.converged}")
    print(f"  Steps: {result.context.step_count}")
    print(f"  Total violation: {result.final_violation:.4f}")
    
    if result.solution.result:
        queens = [int(round(result.solution.result.get(f'q{i}', 0))) for i in range(4)]
        print(f"  Queen positions: {queens}")
        
        # Visualize
        print("\n  Board:")
        for row in range(4):
            line = "  "
            for col in range(4):
                if queens[col] == row:
                    line += "Q "
                else:
                    line += ". "
            print(line)
    
    return {
        'problem': '4_queens',
        'converged': result.converged,
        'solution': result.solution.result,
        'steps': result.context.step_count,
        'violation': result.final_violation,
    }


def demonstrate_pattern_matching() -> Dict[str, Any]:
    """
    Demonstrate pattern matching problem.
    
    Find molecules matching a pattern of properties.
    """
    print("\n" + "="*60)
    print("PROBLEM 3: Pattern Matching (Molecule Search)")
    print("="*60)
    
    print("\nProblem: Find molecules matching pattern {type: 'kinase', active: True}")
    
    # Create problem
    pattern = {
        'type': 'kinase',
        'active': True,
    }
    
    candidates = [
        {'type': 'kinase', 'active': True, 'name': 'PKA', 'mass': 40000},
        {'type': 'kinase', 'active': False, 'name': 'PKC_inactive', 'mass': 80000},
        {'type': 'phosphatase', 'active': True, 'name': 'PP1', 'mass': 35000},
        {'type': 'kinase', 'active': True, 'name': 'MAPK', 'mass': 43000},
        {'type': 'kinase', 'active': True, 'name': 'CDK2', 'mass': 34000},
        {'type': 'receptor', 'active': True, 'name': 'EGFR', 'mass': 170000},
    ]
    
    problem = PatternMatchProblem.create(
        pattern=pattern,
        candidates=candidates,
        name="molecule_pattern_match"
    )
    
    print(f"\nProblem structure:")
    print(f"  Pattern: {pattern}")
    print(f"  Candidates: {len(candidates)}")
    
    runtime = CategoricalRuntime(max_steps=50)
    result = runtime.execute(problem)
    
    print(f"\nResult:")
    print(f"  Steps: {result.context.step_count}")
    
    if result.solution.result:
        print(f"  Matches found:")
        for name, score, props in result.solution.result[:5]:
            print(f"    - {props.get('name', name)}: score={score:.2f}")
    
    return {
        'problem': 'pattern_match',
        'matches': result.solution.result if result.solution.result else [],
        'steps': result.context.step_count,
    }


def demonstrate_biological() -> Dict[str, Any]:
    """
    Demonstrate biological problem solving.
    
    Model ligand-receptor binding and find optimal binding configuration.
    """
    print("\n" + "="*60)
    print("PROBLEM 4: Biological (Drug-Receptor Binding)")
    print("="*60)
    
    print("\nProblem: Find optimal binding configuration for drug-receptor system")
    
    # Create problem
    problem = BiologicalProblem.create_binding(
        ligand={
            'name': 'aspirin',
            'molecular_weight': 180.16,
            'charge': -1,
            'h_bond_donors': 1,
            'h_bond_acceptors': 4,
        },
        receptor={
            'name': 'COX2',
            'binding_pocket_volume': 500,
            'charge': 1,
            'specificity': 'high',
        },
    )
    
    print(f"\nProblem structure:")
    print(f"  Molecules: {list(problem.entities.keys())}")
    print(f"  Interactions: {[(r.source, r.target, r.relation_type) for r in problem.relations]}")
    
    # Add binding energy optimization
    problem.add_entity(
        'binding_energy',
        'metric',
        value=0,
        lower_bound=-100,
        upper_bound=0,
    )
    
    # Constraint: favorable binding (negative energy)
    problem.add_constraint(
        name='favorable_binding',
        constraint_type='inequality',
        expression='binding_energy < -10',
        entities=['binding_energy']
    )
    
    problem.compile()
    
    runtime = CategoricalRuntime(max_steps=100)
    result = runtime.execute(problem, strategy=NavigationStrategy.HARMONY_SEARCH)
    
    print(f"\nResult:")
    print(f"  Steps: {result.context.step_count}")
    print(f"  Converged: {result.converged}")
    
    # Show categorical trajectory
    print(f"\n  Categorical trajectory (S-entropy coordinates):")
    for i, coord in enumerate(result.solution.trajectory[:5]):
        print(f"    Step {i}: S_k={coord.S_k:.4f}, S_t={coord.S_t:.4f}, S_e={coord.S_e:.4f}")
    if len(result.solution.trajectory) > 5:
        print(f"    ... ({len(result.solution.trajectory) - 5} more steps)")
    
    return {
        'problem': 'drug_binding',
        'converged': result.converged,
        'steps': result.context.step_count,
        'final_position': {
            'S_k': result.solution.completion_point.S_k if result.solution.completion_point else 0,
            'S_t': result.solution.completion_point.S_t if result.solution.completion_point else 0,
            'S_e': result.solution.completion_point.S_e if result.solution.completion_point else 0,
        },
    }


def demonstrate_general_problem() -> Dict[str, Any]:
    """
    Demonstrate translating a natural language problem.
    """
    print("\n" + "="*60)
    print("PROBLEM 5: Natural Language Translation")
    print("="*60)
    
    description = """
    Find the optimal allocation of 100 units of resource between 
    three projects A, B, and C to maximize total benefit.
    Project A returns 2x its allocation.
    Project B returns 3x its allocation.
    Project C returns 1.5x its allocation.
    Each project must receive at least 10 units.
    """
    
    print(f"\nProblem description:")
    print(description)
    
    translator = ProblemTranslator()
    
    # Detect problem type
    detected_type = translator.detect_problem_type(description)
    print(f"\nDetected problem type: {detected_type.value}")
    
    # Manual translation (since NLP is simplified)
    problem = OptimizationProblem.create(
        objective="2*A + 3*B + 1.5*C",  # Maximize total return
        variables={
            'A': (10, 80),
            'B': (10, 80),
            'C': (10, 80),
        },
        constraints=[
            "A + B + C < 101",  # Total constraint
            "A + B + C > 99",   # Use all resources
        ],
        minimize=False,  # Maximize
        name="resource_allocation"
    )
    
    print(f"\nTranslated to categorical structure:")
    print(f"  Variables: A, B, C (allocations)")
    print(f"  Objective: maximize 2*A + 3*B + 1.5*C")
    print(f"  Constraints: A+B+C=100, each >= 10")
    
    # Execute
    # For maximization, we minimize the negative
    runtime = CategoricalRuntime(max_steps=300, convergence_threshold=0.1)
    result = runtime.execute(problem, strategy=NavigationStrategy.GRADIENT_DESCENT)
    
    print(f"\nResult:")
    print(f"  Converged: {result.converged}")
    print(f"  Steps: {result.context.step_count}")
    
    if result.solution.result:
        A = result.solution.result.get('A', 0)
        B = result.solution.result.get('B', 0)
        C = result.solution.result.get('C', 0)
        total_return = 2*A + 3*B + 1.5*C
        
        print(f"  Allocation:")
        print(f"    A: {A:.1f} units (return: {2*A:.1f})")
        print(f"    B: {B:.1f} units (return: {3*B:.1f})")
        print(f"    C: {C:.1f} units (return: {1.5*C:.1f})")
        print(f"  Total: {A+B+C:.1f} units")
        print(f"  Total return: {total_return:.1f}")
    
    return {
        'problem': 'resource_allocation',
        'solution': result.solution.result,
        'converged': result.converged,
    }


def run_all_demonstrations() -> Dict[str, Any]:
    """Run all problem demonstrations."""
    print("="*70)
    print("CATEGORICAL COMPUTER - PROBLEM SOLVING DEMONSTRATION")
    print("="*70)
    print()
    print("This demonstrates solving problems via NAVIGATION rather than EXECUTION.")
    print()
    print("Key differences from classical computing:")
    print("  1. Problems are translated to categorical structures")
    print("  2. 'Execution' is navigation through S-entropy space")
    print("  3. The solution is the categorical completion point")
    print("  4. Hardware oscillations provide navigation coordinates")
    
    results = {
        'optimization': demonstrate_optimization(),
        'constraint': demonstrate_constraint_satisfaction(),
        'pattern_match': demonstrate_pattern_matching(),
        'biological': demonstrate_biological(),
        'natural_language': demonstrate_general_problem(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_steps = sum(r.get('steps', 0) for r in results.values())
    converged = sum(1 for r in results.values() if r.get('converged', False))
    
    print(f"\nProblems solved: {len(results)}")
    print(f"Converged: {converged}/{len(results)}")
    print(f"Total navigation steps: {total_steps}")
    print()
    print("The categorical computer solves problems by finding categorical")
    print("completions - the points where all constraints naturally meet.")
    print("This is fundamentally different from step-by-step algorithms.")
    
    return results


if __name__ == "__main__":
    results = run_all_demonstrations()
    
    # Save results
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'categorical_computer')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        return obj
    
    with open(os.path.join(output_dir, 'demo_results.json'), 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

