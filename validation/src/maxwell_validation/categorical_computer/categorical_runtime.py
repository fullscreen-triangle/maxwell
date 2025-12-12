"""
Categorical Runtime

The execution engine that runs categorical problems on the processor.

Unlike classical runtime which executes instructions sequentially,
this runtime:
1. Initializes position in S-entropy space
2. Navigates toward categorical completion
3. Checks constraints at each step
4. Returns when completion is reached (or timeout)

The "execution" is actually NAVIGATION through the hierarchy.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum

# Import processor and memory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from categorical_memory.s_entropy_address import SCoordinate, SEntropyAddress
from categorical_memory.hardware_oscillator import HardwareOscillatorCapture
from categorical_memory.precision_calculator import PrecisionByDifferenceCalculator
from categorical_memory.categorical_hierarchy import CategoricalHierarchy
from categorical_memory.memory_controller import CategoricalMemoryController

from .problem_translator import CategoricalProblem, CategoricalSolution, ProblemType


class NavigationStrategy(Enum):
    """Strategies for navigating S-entropy space."""
    GRADIENT_DESCENT = "gradient"  # Follow constraint gradient
    CATEGORICAL_COMPLETION = "completion"  # Navigate to completion point
    SIMULATED_ANNEALING = "annealing"  # Random + acceptance
    HARMONY_SEARCH = "harmony"  # Based on harmonic coincidences


@dataclass
class ExecutionContext:
    """
    Context for categorical execution.
    
    Contains the current state of the computation including
    position in S-entropy space, constraint satisfaction, and
    trajectory history.
    """
    problem: CategoricalProblem
    
    # Current position in S-entropy space
    current_position: SCoordinate = field(default_factory=lambda: SCoordinate(0, 0, 0))
    
    # Current variable/entity values
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Trajectory
    trajectory: List[SCoordinate] = field(default_factory=list)
    precision_history: List[float] = field(default_factory=list)
    
    # Constraint tracking
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    
    # Performance
    step_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def total_violation(self) -> float:
        return sum(self.constraint_violations.values())
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def record_step(self, position: SCoordinate, precision: float):
        """Record a navigation step."""
        self.trajectory.append(position)
        self.precision_history.append(precision)
        self.current_position = position
        self.step_count += 1


@dataclass
class RuntimeResult:
    """Complete result from runtime execution."""
    solution: CategoricalSolution
    context: ExecutionContext
    
    # Execution metadata
    strategy_used: NavigationStrategy = NavigationStrategy.CATEGORICAL_COMPLETION
    converged: bool = False
    timeout: bool = False
    
    # Analysis
    convergence_rate: float = 0.0
    final_violation: float = 0.0


class CategoricalRuntime:
    """
    The main runtime engine for categorical computation.
    
    This ties together:
    - Hardware oscillator capture (real timing data)
    - Precision-by-difference calculation (S-entropy coordinates)
    - Categorical hierarchy navigation
    - Memory management (S-RAM)
    - Constraint evaluation
    
    Execution is navigation through S-entropy space until
    categorical completion is reached.
    """
    
    def __init__(
        self,
        max_steps: int = 1000,
        convergence_threshold: float = 1e-6,
        timeout: float = 60.0,
    ):
        """
        Initialize the categorical runtime.
        
        Args:
            max_steps: Maximum navigation steps
            convergence_threshold: Threshold for completion
            timeout: Maximum execution time (seconds)
        """
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.timeout = timeout
        
        # Core components
        self.oscillator = HardwareOscillatorCapture()
        self.precision_calc = PrecisionByDifferenceCalculator(self.oscillator)
        self.memory = CategoricalMemoryController()
        self.hierarchy = CategoricalHierarchy()
        
        # Calibrate
        self.oscillator.calibrate(duration=0.1)
        
        # Execution history
        self.execution_history: List[RuntimeResult] = []
        
    def execute(
        self,
        problem: CategoricalProblem,
        strategy: NavigationStrategy = NavigationStrategy.CATEGORICAL_COMPLETION,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> RuntimeResult:
        """
        Execute a categorical problem.
        
        Args:
            problem: The problem to solve
            strategy: Navigation strategy to use
            initial_state: Optional initial variable values
            
        Returns:
            RuntimeResult with solution and metadata
        """
        # Initialize context
        context = ExecutionContext(problem=problem)
        
        # Set initial state
        if initial_state:
            context.state = initial_state.copy()
        elif problem.initial_state:
            context.state = problem.initial_state.copy()
            # Flatten 'variables' if present
            if 'variables' in context.state and isinstance(context.state['variables'], dict):
                variables = context.state.pop('variables')
                context.state.update(variables)
        else:
            context.state = self._default_initial_state(problem)
        
        # Get initial position
        context.current_position = self._state_to_coordinate(context.state)
        context.trajectory.append(context.current_position)
        
        # Create address for this execution
        exec_key = f"exec_{problem.name}_{time.time()}"
        address = self.precision_calc.create_address(exec_key)
        
        # Store problem in memory
        self.memory.store(exec_key, problem)
        
        # Execute navigation
        converged = False
        timeout_reached = False
        
        while context.step_count < self.max_steps:
            # Check timeout
            if context.elapsed_time > self.timeout:
                timeout_reached = True
                break
            
            # Evaluate constraints
            self._evaluate_constraints(context)
            
            # Check convergence
            if context.total_violation < self.convergence_threshold:
                converged = True
                break
            
            # Navigate one step
            if strategy == NavigationStrategy.GRADIENT_DESCENT:
                self._step_gradient(context)
            elif strategy == NavigationStrategy.CATEGORICAL_COMPLETION:
                self._step_completion(context, address)
            elif strategy == NavigationStrategy.SIMULATED_ANNEALING:
                self._step_annealing(context)
            elif strategy == NavigationStrategy.HARMONY_SEARCH:
                self._step_harmony(context)
            
            # Record precision
            precision = self.precision_calc.calculate_precision_difference()
            context.record_step(context.current_position, precision)
            self.precision_calc.update_address(exec_key)
        
        # Build solution
        solution = self._build_solution(context, converged)
        
        # Build result
        result = RuntimeResult(
            solution=solution,
            context=context,
            strategy_used=strategy,
            converged=converged,
            timeout=timeout_reached,
            final_violation=context.total_violation,
        )
        
        # Calculate convergence rate
        if len(context.trajectory) > 1:
            violations = [sum(context.constraint_violations.values())]
            # Approximate from trajectory
            result.convergence_rate = (violations[0] - context.total_violation) / len(context.trajectory)
        
        self.execution_history.append(result)
        return result
    
    def _default_initial_state(self, problem: CategoricalProblem) -> Dict[str, Any]:
        """Generate default initial state from problem."""
        state = {}
        
        for name, entity in problem.entities.items():
            # For variables with bounds, start in middle
            if 'lower_bound' in entity.properties and 'upper_bound' in entity.properties:
                lo = entity.properties['lower_bound']
                hi = entity.properties['upper_bound']
                state[name] = (lo + hi) / 2
            elif 'value' in entity.properties:
                state[name] = entity.properties['value']
            else:
                state[name] = 0.0
                
        return state
    
    def _state_to_coordinate(self, state: Dict[str, Any]) -> SCoordinate:
        """Convert current state to S-coordinate."""
        values = []
        for v in state.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, bool):
                values.append(1.0 if v else 0.0)
                
        if not values:
            return SCoordinate(0, 0, 0)
        
        # S_k: variance (rate of change across variables)
        S_k = float(np.std(values)) if len(values) > 1 else 0.0
        
        # S_t: mean (central tendency)
        S_t = float(np.mean(values))
        
        # S_e: entropy
        if len(values) > 1:
            hist, _ = np.histogram(values, bins=min(10, len(values)))
            hist = hist / (hist.sum() + 1e-10)
            S_e = float(-np.sum(hist * np.log(hist + 1e-10)))
        else:
            S_e = 0.0
            
        return SCoordinate(S_k=S_k, S_t=S_t, S_e=S_e)
    
    def _evaluate_constraints(self, context: ExecutionContext):
        """Evaluate all constraints on current state."""
        context.constraint_violations.clear()
        
        for constraint in context.problem.constraints:
            satisfied, violation = constraint.evaluate(context.state)
            context.constraint_violations[constraint.name] = violation
    
    def _step_gradient(self, context: ExecutionContext):
        """Take a gradient descent step toward constraint satisfaction."""
        step_size = 0.1
        
        for name in context.state:
            if isinstance(context.state[name], (int, float)):
                # Numerical gradient
                original = context.state[name]
                
                # Forward difference
                context.state[name] = original + 0.01
                self._evaluate_constraints(context)
                forward_violation = context.total_violation
                
                # Backward difference
                context.state[name] = original - 0.01
                self._evaluate_constraints(context)
                backward_violation = context.total_violation
                
                # Gradient
                gradient = (forward_violation - backward_violation) / 0.02
                
                # Update
                context.state[name] = original - step_size * gradient
                
                # Respect bounds if present
                entity = context.problem.entities.get(name)
                if entity and 'lower_bound' in entity.properties:
                    context.state[name] = max(
                        entity.properties['lower_bound'],
                        context.state[name]
                    )
                if entity and 'upper_bound' in entity.properties:
                    context.state[name] = min(
                        entity.properties['upper_bound'],
                        context.state[name]
                    )
        
        context.current_position = self._state_to_coordinate(context.state)
    
    def _step_completion(self, context: ExecutionContext, address: SEntropyAddress):
        """Take a step toward categorical completion point."""
        # Predict completion point
        completion = address.predict_completion()
        
        if not completion:
            # Fallback to gradient
            self._step_gradient(context)
            return
        
        # Navigate toward completion
        current = context.current_position
        
        # Direction vector in S-space
        direction = np.array([
            completion.S_k - current.S_k,
            completion.S_t - current.S_t,
            completion.S_e - current.S_e,
        ])
        
        distance = np.linalg.norm(direction)
        if distance < 1e-10:
            return
        
        # Normalize and scale
        step_size = min(0.1, distance)
        direction = direction / distance * step_size
        
        # Update S-coordinate
        new_position = SCoordinate(
            S_k=current.S_k + direction[0],
            S_t=current.S_t + direction[1],
            S_e=current.S_e + direction[2],
        )
        
        # Map back to state space
        # This is the inverse mapping - from S-coordinate to variable values
        self._coordinate_to_state(new_position, context)
        context.current_position = new_position
    
    def _coordinate_to_state(self, coord: SCoordinate, context: ExecutionContext):
        """Map S-coordinate back to state space."""
        # Use the coordinate components to adjust variables
        # S_t (thermal/mean) directly influences variable magnitudes
        # S_k (kinetic/variance) influences spread
        # S_e (entropy) influences randomness
        
        var_names = list(context.state.keys())
        n_vars = len(var_names)
        
        if n_vars == 0:
            return
        
        # Base value from S_t
        base_value = coord.S_t
        
        # Spread from S_k
        spread = coord.S_k * 2
        
        for i, name in enumerate(var_names):
            if isinstance(context.state[name], (int, float)):
                # Distribute around mean with spread
                offset = (i - n_vars / 2) * spread / max(1, n_vars)
                context.state[name] = base_value + offset
                
                # Respect bounds
                entity = context.problem.entities.get(name)
                if entity:
                    if 'lower_bound' in entity.properties:
                        context.state[name] = max(
                            entity.properties['lower_bound'],
                            context.state[name]
                        )
                    if 'upper_bound' in entity.properties:
                        context.state[name] = min(
                            entity.properties['upper_bound'],
                            context.state[name]
                        )
    
    def _step_annealing(self, context: ExecutionContext):
        """Simulated annealing step."""
        temperature = 1.0 / (1 + context.step_count * 0.01)
        
        for name in context.state:
            if isinstance(context.state[name], (int, float)):
                original = context.state[name]
                original_violation = context.total_violation
                
                # Random perturbation
                perturbation = np.random.normal(0, temperature)
                context.state[name] = original + perturbation
                
                # Respect bounds
                entity = context.problem.entities.get(name)
                if entity:
                    if 'lower_bound' in entity.properties:
                        context.state[name] = max(
                            entity.properties['lower_bound'],
                            context.state[name]
                        )
                    if 'upper_bound' in entity.properties:
                        context.state[name] = min(
                            entity.properties['upper_bound'],
                            context.state[name]
                        )
                
                # Evaluate
                self._evaluate_constraints(context)
                new_violation = context.total_violation
                
                # Accept or reject
                if new_violation > original_violation:
                    # Accept with probability based on temperature
                    accept_prob = np.exp(-(new_violation - original_violation) / (temperature + 1e-10))
                    if np.random.random() > accept_prob:
                        context.state[name] = original
        
        context.current_position = self._state_to_coordinate(context.state)
    
    def _step_harmony(self, context: ExecutionContext):
        """Harmony search step - uses harmonic relationships."""
        # Get harmonic signature from hardware
        signature = self.oscillator.get_precision_signature(n_samples=5)
        
        # Use signature to guide search
        for i, name in enumerate(context.state):
            if isinstance(context.state[name], (int, float)):
                sig_idx = i % len(signature)
                
                # Harmonic adjustment
                harmonic_factor = 1.0 + signature[sig_idx] * 10
                
                # Apply to bounds-constrained search
                entity = context.problem.entities.get(name)
                if entity:
                    lo = entity.properties.get('lower_bound', -1)
                    hi = entity.properties.get('upper_bound', 1)
                    
                    # Harmonic position in range
                    position = (np.sin(harmonic_factor * np.pi) + 1) / 2
                    context.state[name] = lo + position * (hi - lo)
        
        context.current_position = self._state_to_coordinate(context.state)
    
    def _build_solution(
        self, 
        context: ExecutionContext, 
        converged: bool
    ) -> CategoricalSolution:
        """Build solution from execution context."""
        # Determine result based on problem type
        result = None
        
        if context.problem.problem_type == ProblemType.OPTIMIZATION:
            # Result is the optimized variable values
            # Include 'variables' from initial state if present
            if 'variables' in context.state and isinstance(context.state['variables'], dict):
                result = context.state['variables'].copy()
            else:
                result = {k: v for k, v in context.state.items() 
                         if isinstance(v, (int, float))}
            
        elif context.problem.problem_type == ProblemType.SEARCH:
            # Result is the found item (if any)
            for name, entity in context.problem.entities.items():
                # Check if this entity satisfies target
                target_constraint = next(
                    (c for c in context.problem.constraints if c.name == "target"),
                    None
                )
                if target_constraint:
                    satisfied, _ = target_constraint.evaluate({**context.state, 'item': entity.properties})
                    if satisfied:
                        result = entity.properties
                        break
                        
        elif context.problem.problem_type == ProblemType.PATTERN_MATCH:
            # Result is the matching candidates
            matches = []
            pattern_entity = context.problem.entities.get('pattern')
            if pattern_entity:
                for name, entity in context.problem.entities.items():
                    if entity.category == 'candidate':
                        # Compare to pattern
                        match_score = self._compute_match_score(
                            pattern_entity.properties,
                            entity.properties
                        )
                        if match_score > 0.5:
                            matches.append((name, match_score, entity.properties))
            result = sorted(matches, key=lambda x: -x[1])
            
        else:
            # Generic result is final state
            result = context.state.copy()
        
        # Build constraint satisfaction dict
        constraints_satisfied = {
            name: (violation < self.convergence_threshold)
            for name, violation in context.constraint_violations.items()
        }
        
        return CategoricalSolution(
            problem_name=context.problem.name,
            solved=converged,
            result=result,
            trajectory=context.trajectory.copy(),
            constraints_satisfied=constraints_satisfied,
            total_violation=context.total_violation,
            navigation_steps=context.step_count,
            completion_time=context.elapsed_time,
            completion_point=context.current_position,
        )
    
    def _compute_match_score(
        self, 
        pattern: Dict[str, Any], 
        candidate: Dict[str, Any]
    ) -> float:
        """Compute similarity between pattern and candidate."""
        if not pattern or not candidate:
            return 0.0
        
        matches = 0
        total = 0
        
        for key, value in pattern.items():
            total += 1
            if key in candidate:
                if candidate[key] == value:
                    matches += 1
                elif isinstance(value, (int, float)) and isinstance(candidate[key], (int, float)):
                    # Numerical similarity
                    diff = abs(float(value) - float(candidate[key]))
                    matches += max(0, 1 - diff / (abs(float(value)) + 1))
        
        return matches / total if total > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        if not self.execution_history:
            return {'executions': 0}
        
        solved = sum(1 for r in self.execution_history if r.solution.solved)
        total_steps = sum(r.context.step_count for r in self.execution_history)
        total_time = sum(r.context.elapsed_time for r in self.execution_history)
        
        return {
            'executions': len(self.execution_history),
            'solved': solved,
            'solve_rate': solved / len(self.execution_history),
            'total_steps': total_steps,
            'avg_steps': total_steps / len(self.execution_history),
            'total_time': total_time,
            'avg_time': total_time / len(self.execution_history),
            'memory_stats': self.memory.get_statistics(),
            'precision_stats': self.precision_calc.get_statistics(),
        }

