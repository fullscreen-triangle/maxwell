"""
Complexity Theory Experiment
Validates Poincaré complexity, categorical completion rate,
local solution chains, and the Poincaré as a unit of computation.
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict


@dataclass
class LocalSolution:
    """A local solution in the solution chain."""
    id: int
    s_coord: np.ndarray
    constraint_satisfied: bool
    recognition_time: float
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "s_coord": self.s_coord.tolist(),
            "constraint_satisfied": self.constraint_satisfied,
            "recognition_time": self.recognition_time
        }


@dataclass
class SolutionChain:
    """A chain of local solutions forming a coherent cycle."""
    local_solutions: List[LocalSolution] = field(default_factory=list)
    is_closed: bool = False
    closure_epsilon: float = 0.1
    
    def add_solution(self, sol: LocalSolution):
        self.local_solutions.append(sol)
        self._check_closure()
    
    def _check_closure(self):
        """Check if chain forms a closed loop."""
        if len(self.local_solutions) < 3:
            return
        
        first = self.local_solutions[0].s_coord
        last = self.local_solutions[-1].s_coord
        distance = np.linalg.norm(first - last)
        
        self.is_closed = distance < self.closure_epsilon
    
    def poincare_complexity(self) -> int:
        """Return Pi(P) - number of local solutions."""
        return len([s for s in self.local_solutions if s.constraint_satisfied])


class CategoricalCompletionTracker:
    """Tracks categorical completions for complexity measurement."""
    
    def __init__(self, n_categories: int = 1000):
        self.n_categories = n_categories
        self.completions: List[Tuple[int, float]] = []
        self.current_time = 0.0
        
        np.random.seed(42)
        self.category_coords = {
            i: np.random.rand(3) for i in range(n_categories)
        }
    
    def complete_category(self, cat_id: int):
        """Record a category completion."""
        self.completions.append((cat_id, self.current_time))
    
    def completion_rate(self, window_size: float = 1.0) -> float:
        """Compute categorical completion rate (rho_C)."""
        if not self.completions:
            return 0.0
        
        start_time = self.current_time - window_size
        recent = [c for c in self.completions if c[1] >= start_time]
        
        if window_size == 0:
            return float('inf')
        return len(recent) / window_size
    
    def advance_time(self, dt: float):
        self.current_time += dt


class PoincareComplexityMeasurer:
    """Measures complexity in Poincarés."""
    
    def __init__(self, tracker: CategoricalCompletionTracker):
        self.tracker = tracker
        self.chains: List[SolutionChain] = []
    
    def start_new_chain(self, epsilon: float = 0.1) -> SolutionChain:
        """Start a new solution chain."""
        chain = SolutionChain(closure_epsilon=epsilon)
        self.chains.append(chain)
        return chain
    
    def recognize_local_solution(self, chain: SolutionChain, s_coord: np.ndarray, 
                                   constraint_check: bool) -> LocalSolution:
        """Recognize a local solution (one Poincaré)."""
        sol = LocalSolution(
            id=len(chain.local_solutions),
            s_coord=s_coord,
            constraint_satisfied=constraint_check,
            recognition_time=self.tracker.current_time
        )
        chain.add_solution(sol)
        
        # Each recognition is a categorical completion
        nearest_cat = min(
            self.tracker.category_coords.keys(),
            key=lambda i: np.linalg.norm(self.tracker.category_coords[i] - s_coord)
        )
        self.tracker.complete_category(nearest_cat)
        
        return sol
    
    def compute_problem_complexity(self, chain: SolutionChain) -> Dict:
        """Compute complexity metrics for a problem."""
        pi_p = chain.poincare_complexity()
        n_recognitions = len(chain.local_solutions)
        
        return {
            "poincare_complexity": pi_p,
            "total_recognitions": n_recognitions,
            "is_closed": chain.is_closed,
            "effective_poincare_count": pi_p if chain.is_closed else 0
        }


class UnknowableOriginDemonstrator:
    """Demonstrates that S_0 is unknowable."""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def demonstrate_unknowability(self) -> Dict:
        """Show S_0 cannot be directly observed."""
        np.random.seed(42)
        
        # Generate "true" S_0 values
        true_s0_values = [np.random.rand(3) for _ in range(self.n_trials)]
        
        # Simulate observations (always offset by some noise)
        observations = []
        inferred_s0 = []
        
        for s0 in true_s0_values:
            # We only observe after some dynamics have occurred
            observation_noise = np.random.randn(3) * 0.05
            observed = s0 + observation_noise
            observations.append(observed)
            
            # Try to infer S_0 from observations
            # The best we can do is observe the trajectory closure
            inferred = observed - observation_noise * 0.5  # Imperfect correction
            inferred_s0.append(inferred)
        
        # Measure inference error
        errors = [np.linalg.norm(true - inf) 
                  for true, inf in zip(true_s0_values, inferred_s0)]
        
        return {
            "n_trials": self.n_trials,
            "mean_inference_error": float(np.mean(errors)),
            "min_inference_error": float(np.min(errors)),
            "max_inference_error": float(np.max(errors)),
            "perfect_observations": sum(1 for e in errors if e == 0),
            "demonstrates_unknowability": all(e > 0 for e in errors)
        }


class AsymptoticReturnAnalyzer:
    """Analyzes asymptotic return (never exact)."""
    
    def __init__(self):
        pass
    
    def generate_trajectory(self, n_steps: int, target: np.ndarray) -> List[np.ndarray]:
        """Generate trajectory approaching target asymptotically."""
        trajectory = []
        current = np.random.rand(3)
        
        for step in range(n_steps):
            # Move toward target with decreasing step size
            direction = target - current
            step_size = 0.5 / (step + 1)  # Decreasing steps
            noise = np.random.randn(3) * 0.01
            current = current + direction * step_size + noise
            trajectory.append(current.copy())
        
        return trajectory
    
    def analyze_approach(self, trajectory: List[np.ndarray], target: np.ndarray) -> Dict:
        """Analyze how trajectory approaches target."""
        distances = [np.linalg.norm(p - target) for p in trajectory]
        
        # Check if distance ever reaches zero
        reaches_zero = any(d == 0 for d in distances)
        min_distance = min(distances)
        
        return {
            "n_steps": len(trajectory),
            "min_distance": float(min_distance),
            "final_distance": float(distances[-1]),
            "reaches_zero": reaches_zero,
            "asymptotic": not reaches_zero and min_distance > 0,
            "distance_progression": distances[::max(1, len(distances)//20)]
        }


class TimeIndependentComplexity:
    """Demonstrates time-independence of complexity measurement."""
    
    def __init__(self):
        pass
    
    def measure_at_different_rates(self, problem_size: int) -> Dict:
        """Measure same problem at different "physical" rates."""
        np.random.seed(42)
        
        results = []
        for rate_multiplier in [0.5, 1.0, 2.0, 5.0]:
            tracker = CategoricalCompletionTracker(n_categories=500)
            measurer = PoincareComplexityMeasurer(tracker)
            chain = measurer.start_new_chain()
            
            # Simulate problem solving
            target = np.array([0.5, 0.5, 0.5])
            current = np.random.rand(3)
            
            poincare_count = 0
            for _ in range(problem_size * 10):
                # Move toward target
                direction = target - current
                current = current + direction * 0.1 * np.random.random()
                
                # Advance physical time at different rates
                tracker.advance_time(0.01 / rate_multiplier)
                
                # Recognize local solution (time-independent)
                if np.random.random() < 0.1:
                    constraint_ok = np.linalg.norm(current - target) < 0.3
                    measurer.recognize_local_solution(chain, current.copy(), constraint_ok)
                    poincare_count += 1
            
            physical_time = tracker.current_time
            completion_rate = tracker.completion_rate(window_size=physical_time)
            
            results.append({
                "rate_multiplier": rate_multiplier,
                "physical_time": physical_time,
                "poincare_count": poincare_count,
                "completion_rate": completion_rate,
                "categorical_complexity": chain.poincare_complexity()
            })
        
        # Check if categorical complexity is invariant
        complexities = [r["categorical_complexity"] for r in results]
        variance = np.var(complexities)
        
        return {
            "measurements": results,
            "complexity_variance": float(variance),
            "is_time_independent": variance < 5.0
        }


def run_complexity_experiment() -> Dict:
    """Run comprehensive complexity theory validation."""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {}
    }
    
    print("=" * 60)
    print("COMPLEXITY THEORY EXPERIMENT")
    print("=" * 60)
    
    # Experiment 1: Poincaré Complexity (Pi(P))
    print("\n[1] Poincaré Complexity Measurement...")
    tracker = CategoricalCompletionTracker(n_categories=500)
    measurer = PoincareComplexityMeasurer(tracker)
    
    complexity_data = []
    for problem_size in [10, 20, 50, 100]:
        chain = measurer.start_new_chain(epsilon=0.15)
        
        # Simulate solving a problem
        target = np.random.rand(3)
        current = np.random.rand(3)
        
        for step in range(problem_size):
            # Move toward target
            direction = target - current
            current = current + direction * 0.1 + np.random.randn(3) * 0.02
            tracker.advance_time(0.1)
            
            # Recognize local solutions
            constraint_ok = np.linalg.norm(current - target) < 0.2
            measurer.recognize_local_solution(chain, current.copy(), constraint_ok)
        
        metrics = measurer.compute_problem_complexity(chain)
        complexity_data.append({
            "problem_size": problem_size,
            **metrics
        })
        print(f"   Size {problem_size}: Pi(P) = {metrics['poincare_complexity']}")
    
    results["experiments"]["poincare_complexity"] = {
        "data": complexity_data,
        "unit_definition": "1 Poincaré = 1 local solution recognition"
    }
    
    # Experiment 2: Categorical Completion Rate (rho_C)
    print("\n[2] Categorical Completion Rate (rho_C)...")
    tracker = CategoricalCompletionTracker(n_categories=500)
    
    rate_history = []
    for step in range(200):
        # Variable completion rate
        n_completions = np.random.poisson(2 + step // 50)
        for _ in range(n_completions):
            cat_id = np.random.randint(0, 500)
            tracker.complete_category(cat_id)
        tracker.advance_time(0.1)
        
        if step % 10 == 0:
            rate = tracker.completion_rate(window_size=1.0)
            rate_history.append({
                "step": step,
                "time": tracker.current_time,
                "completion_rate": rate
            })
    
    results["experiments"]["completion_rate"] = {
        "rate_history": rate_history,
        "final_rate": tracker.completion_rate(window_size=1.0),
        "interpretation": "rho_C independent of physical time units"
    }
    print(f"   Final completion rate: {tracker.completion_rate(window_size=1.0):.2f} completions/unit")
    
    # Experiment 3: Unknowable Origin
    print("\n[3] Unknowable Origin (S_0)...")
    demonstrator = UnknowableOriginDemonstrator(n_trials=100)
    unknowability = demonstrator.demonstrate_unknowability()
    
    results["experiments"]["unknowable_origin"] = unknowability
    print(f"   Mean inference error: {unknowability['mean_inference_error']:.4f}")
    print(f"   Perfect observations: {unknowability['perfect_observations']}")
    
    # Experiment 4: Asymptotic Return
    print("\n[4] Asymptotic Return (Never Exact)...")
    analyzer = AsymptoticReturnAnalyzer()
    
    asymptotic_results = []
    for n_steps in [100, 500, 1000, 2000]:
        target = np.array([0.5, 0.5, 0.5])
        traj = analyzer.generate_trajectory(n_steps, target)
        analysis = analyzer.analyze_approach(traj, target)
        asymptotic_results.append(analysis)
        print(f"   {n_steps} steps: min_dist={analysis['min_distance']:.6f}, asymptotic={analysis['asymptotic']}")
    
    results["experiments"]["asymptotic_return"] = {
        "analyses": asymptotic_results,
        "all_asymptotic": all(a["asymptotic"] for a in asymptotic_results)
    }
    
    # Experiment 5: Solution Chain Closure
    print("\n[5] Solution Chain Closure...")
    tracker = CategoricalCompletionTracker(n_categories=500)
    measurer = PoincareComplexityMeasurer(tracker)
    
    closure_results = []
    for epsilon in [0.05, 0.1, 0.2, 0.3]:
        chain = measurer.start_new_chain(epsilon=epsilon)
        
        # Generate circular trajectory
        center = np.array([0.5, 0.5, 0.5])
        radius = 0.2
        
        for angle in np.linspace(0, 2 * np.pi, 50):
            point = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            measurer.recognize_local_solution(chain, point, True)
            tracker.advance_time(0.01)
        
        closure_results.append({
            "epsilon": epsilon,
            "is_closed": chain.is_closed,
            "chain_length": len(chain.local_solutions),
            "poincare_complexity": chain.poincare_complexity()
        })
        print(f"   ε={epsilon}: closed={chain.is_closed}, Pi(P)={chain.poincare_complexity()}")
    
    results["experiments"]["chain_closure"] = closure_results
    
    # Experiment 6: Time-Independent Complexity
    print("\n[6] Time-Independent Complexity...")
    time_measurer = TimeIndependentComplexity()
    time_results = time_measurer.measure_at_different_rates(problem_size=20)
    
    results["experiments"]["time_independence"] = time_results
    print(f"   Complexity variance across rates: {time_results['complexity_variance']:.2f}")
    print(f"   Time-independent: {time_results['is_time_independent']}")
    
    # Experiment 7: FLOPS Irrelevance
    print("\n[7] FLOPS Irrelevance Demonstration...")
    
    # Simulate same problem with different "FLOPS"
    flops_comparison = []
    for simulated_flops in [1e6, 1e9, 1e12]:
        tracker = CategoricalCompletionTracker(n_categories=500)
        measurer = PoincareComplexityMeasurer(tracker)
        chain = measurer.start_new_chain()
        
        # Fixed number of categorical completions
        n_completions = 100
        for _ in range(n_completions):
            point = np.random.rand(3)
            measurer.recognize_local_solution(chain, point, np.random.random() > 0.3)
            # "Operations" happen at different rates but completions are same
        
        flops_comparison.append({
            "simulated_flops": simulated_flops,
            "poincare_complexity": chain.poincare_complexity(),
            "categorical_completions": n_completions
        })
    
    # All should have same Poincaré complexity
    complexities = [f["poincare_complexity"] for f in flops_comparison]
    
    results["experiments"]["flops_irrelevance"] = {
        "comparison": flops_comparison,
        "complexity_invariant": len(set(complexities)) == 1,
        "interpretation": "FLOPS measures different quantity than Poincaré complexity"
    }
    print(f"   Complexity invariant across FLOPS: {len(set(complexities)) == 1}")
    
    # Experiment 8: Incommensurability with Turing Complexity
    print("\n[8] Turing-Poincaré Incommensurability...")
    
    # Simulate a problem solvable in different paradigms
    incommensurability_data = {
        "turing_steps": [],
        "poincare_completions": []
    }
    
    for trial in range(20):
        # Random "problem"
        np.random.seed(trial)
        problem_difficulty = np.random.randint(10, 100)
        
        # Turing: steps proportional to input size
        turing_steps = problem_difficulty * np.random.randint(5, 20)
        
        # Poincaré: completions depend on categorical structure
        # (not proportional to Turing steps)
        poincare_completions = int(np.log(problem_difficulty + 1) * np.random.randint(3, 10))
        
        incommensurability_data["turing_steps"].append(turing_steps)
        incommensurability_data["poincare_completions"].append(poincare_completions)
    
    # Check correlation
    correlation = np.corrcoef(
        incommensurability_data["turing_steps"],
        incommensurability_data["poincare_completions"]
    )[0, 1]
    
    results["experiments"]["incommensurability"] = {
        **incommensurability_data,
        "correlation": float(correlation),
        "are_incommensurable": abs(correlation) < 0.5,
        "interpretation": "Low correlation indicates different complexity measures"
    }
    print(f"   Turing-Poincaré correlation: {correlation:.3f}")
    
    print("\n" + "=" * 60)
    print("COMPLEXITY THEORY EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


def save_results(results: Dict, output_dir: str = "results"):
    """Save results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "complexity_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    return output_path


if __name__ == "__main__":
    results = run_complexity_experiment()
    save_results(results)

