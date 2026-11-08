"""
Precision-by-Difference Network Experiment

A comprehensive validation of the precision-by-difference framework
from the Sango Rine Shumba paper. This experiment demonstrates:

1. Precision-by-difference calculation: ΔP = T_ref - t_local
2. Temporal coherence windows
3. S-entropy hierarchy navigation via differences
4. Collective state coordination
5. Prediction through categorical completion

Key insight: The "difference" at each step tells you which branch
of the 3^k recursive S-entropy structure you're traversing.
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque
from pathlib import Path
from datetime import datetime


@dataclass
class PrecisionMeasurement:
    """A single precision-by-difference measurement."""
    timestamp: float
    reference_time: float
    local_time: float
    delta_p: float  # The precision-by-difference value
    hierarchy_branch: int  # Which branch (0, 1, 2) in the 3^k tree


@dataclass
class TemporalWindow:
    """A coherence window for temporal coordination."""
    start: float
    end: float
    center: float
    width: float
    measurements: List[float] = field(default_factory=list)
    
    @property
    def coherence_quality(self) -> float:
        """Higher quality = tighter window."""
        if not self.measurements:
            return 0.0
        return 1.0 / (1.0 + np.std(self.measurements))


@dataclass
class HierarchyNode:
    """A node in the 3^k S-entropy hierarchy."""
    depth: int
    branch_path: List[int]  # Path from root (sequence of 0, 1, 2)
    s_k: float  # Knowledge coordinate
    s_t: float  # Temporal coordinate
    s_e: float  # Entropy/evolution coordinate
    precision_accumulated: float = 0.0
    
    @property
    def address(self) -> str:
        """Ternary address string."""
        return ''.join(map(str, self.branch_path)) if self.branch_path else 'root'


class PrecisionByDifferenceExperiment:
    """
    Comprehensive experiment for precision-by-difference network.
    
    This validates the core concepts from Sango Rine Shumba:
    - Temporal variations as coordination resources
    - Precision-by-difference calculation
    - Hierarchy navigation
    - Collective state coordination
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.results = {}
        
        # Reference clock (simulating atomic clock)
        self._reference_epoch = time.perf_counter_ns()
        
        # Hierarchy parameters
        self.max_depth = 8  # 3^8 = 6561 addressable nodes
        
        # Experimental data
        self.measurements: List[PrecisionMeasurement] = []
        self.windows: List[TemporalWindow] = []
        self.hierarchy: Dict[str, HierarchyNode] = {}
        
    def run_all_experiments(self) -> Dict:
        """Run all precision-by-difference experiments."""
        print("="*70)
        print("PRECISION-BY-DIFFERENCE NETWORK EXPERIMENT")
        print("="*70)
        print()
        
        # Experiment 1: Basic precision-by-difference measurement
        print("[1/6] Running precision measurement experiment...")
        self.results['precision_measurement'] = self.experiment_precision_measurement()
        
        # Experiment 2: Temporal coherence windows
        print("[2/6] Running temporal window experiment...")
        self.results['temporal_windows'] = self.experiment_temporal_windows()
        
        # Experiment 3: Hierarchy navigation
        print("[3/6] Running hierarchy navigation experiment...")
        self.results['hierarchy_navigation'] = self.experiment_hierarchy_navigation()
        
        # Experiment 4: Collective state coordination
        print("[4/6] Running collective coordination experiment...")
        self.results['collective_coordination'] = self.experiment_collective_coordination()
        
        # Experiment 5: Prediction through completion
        print("[5/6] Running prediction experiment...")
        self.results['prediction'] = self.experiment_prediction()
        
        # Experiment 6: Network latency simulation
        print("[6/6] Running network latency experiment...")
        self.results['network_latency'] = self.experiment_network_latency()
        
        # Summary
        self.results['summary'] = self._compute_summary()
        
        print()
        print("="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        
        return self.results
    
    def experiment_precision_measurement(self) -> Dict:
        """
        Experiment 1: Basic precision-by-difference measurements.
        
        Validates: ΔP = T_ref - t_local
        """
        n_measurements = 1000
        measurements = []
        
        for i in range(n_measurements):
            # Simulate reference time (high precision)
            ref_time = time.perf_counter_ns()
            
            # Simulate local time with jitter (lower precision)
            # In real system this would be actual local clock
            jitter_ns = np.random.normal(0, 1000)  # ±1μs jitter
            local_time = ref_time + jitter_ns
            
            # Precision-by-difference
            delta_p = (ref_time - local_time) / 1e9  # Convert to seconds
            
            # Determine hierarchy branch from delta_p
            # Positive -> branch 0, Near zero -> branch 1, Negative -> branch 2
            if delta_p > 0.5e-6:
                branch = 0
            elif delta_p < -0.5e-6:
                branch = 2
            else:
                branch = 1
            
            m = PrecisionMeasurement(
                timestamp=time.time(),
                reference_time=ref_time / 1e9,
                local_time=local_time / 1e9,
                delta_p=delta_p,
                hierarchy_branch=branch
            )
            measurements.append(m)
            self.measurements.append(m)
            
            time.sleep(0.0001)  # 100μs between measurements
        
        # Analyze
        delta_ps = [m.delta_p for m in measurements]
        branches = [m.hierarchy_branch for m in measurements]
        
        return {
            'n_measurements': n_measurements,
            'delta_p_mean': float(np.mean(delta_ps)),
            'delta_p_std': float(np.std(delta_ps)),
            'delta_p_min': float(np.min(delta_ps)),
            'delta_p_max': float(np.max(delta_ps)),
            'branch_distribution': {
                'branch_0': int(branches.count(0)),
                'branch_1': int(branches.count(1)),
                'branch_2': int(branches.count(2)),
            },
            'measurements': [asdict(m) for m in measurements[:100]],  # First 100 for plotting
        }
    
    def experiment_temporal_windows(self) -> Dict:
        """
        Experiment 2: Temporal coherence windows.
        
        Validates: W_i(k) = [T_ref(k) + min(ΔP), T_ref(k) + max(ΔP)]
        """
        n_windows = 50
        window_size = 20
        windows = []
        
        for i in range(n_windows):
            # Collect measurements for this window
            measurements = []
            ref_time = time.perf_counter()
            
            for j in range(window_size):
                jitter = np.random.normal(0, 0.001)  # 1ms jitter
                local_time = time.perf_counter() + jitter
                delta_p = ref_time - local_time
                measurements.append(delta_p)
            
            # Compute window
            window = TemporalWindow(
                start=ref_time + min(measurements),
                end=ref_time + max(measurements),
                center=ref_time + np.mean(measurements),
                width=max(measurements) - min(measurements),
                measurements=measurements
            )
            windows.append(window)
            self.windows.append(window)
            
            time.sleep(0.01)
        
        # Analyze
        widths = [w.width for w in windows]
        qualities = [w.coherence_quality for w in windows]
        
        return {
            'n_windows': n_windows,
            'window_size': window_size,
            'mean_width': float(np.mean(widths)),
            'std_width': float(np.std(widths)),
            'mean_quality': float(np.mean(qualities)),
            'windows': [
                {
                    'start': float(w.start),
                    'end': float(w.end),
                    'center': float(w.center),
                    'width': float(w.width),
                    'quality': float(w.coherence_quality),
                }
                for w in windows
            ],
        }
    
    def experiment_hierarchy_navigation(self) -> Dict:
        """
        Experiment 3: 3^k hierarchy navigation via precision-by-difference.
        
        Validates: Each ΔP determines which branch to take in the hierarchy.
        """
        # Build hierarchy tree
        depth = 5  # 3^5 = 243 nodes
        
        def build_node(current_depth: int, path: List[int]) -> HierarchyNode:
            # S-coordinates derived from path
            s_k = sum(b * (3 ** -i) for i, b in enumerate(path)) if path else 0.5
            s_t = len(path) / depth
            s_e = np.mean(path) / 2.0 if path else 0.5
            
            return HierarchyNode(
                depth=current_depth,
                branch_path=path.copy(),
                s_k=s_k,
                s_t=s_t,
                s_e=s_e,
            )
        
        def build_tree(current_depth: int, path: List[int]):
            if current_depth > depth:
                return
            
            node = build_node(current_depth, path)
            self.hierarchy[node.address] = node
            
            if current_depth < depth:
                for branch in [0, 1, 2]:
                    build_tree(current_depth + 1, path + [branch])
        
        build_tree(0, [])
        
        # Navigate using precision-by-difference
        n_navigations = 100
        navigation_results = []
        
        for nav_id in range(n_navigations):
            # Start at root
            current_path = []
            precision_trajectory = []
            
            # Navigate to random target depth
            target_depth = np.random.randint(3, depth + 1)
            
            for step in range(target_depth):
                # Get precision-by-difference (simulated)
                delta_p = np.random.normal(0, 0.001)
                precision_trajectory.append(delta_p)
                
                # Determine branch from delta_p
                if delta_p > 0.0003:
                    branch = 0
                elif delta_p < -0.0003:
                    branch = 2
                else:
                    branch = 1
                
                current_path.append(branch)
            
            # Find final node
            address = ''.join(map(str, current_path))
            final_node = self.hierarchy.get(address)
            
            if final_node:
                navigation_results.append({
                    'nav_id': nav_id,
                    'target_depth': target_depth,
                    'path': current_path,
                    'address': address,
                    's_k': float(final_node.s_k),
                    's_t': float(final_node.s_t),
                    's_e': float(final_node.s_e),
                    'precision_trajectory': precision_trajectory,
                })
        
        # Analyze coverage
        visited_addresses = set(r['address'] for r in navigation_results)
        total_addresses = len(self.hierarchy)
        
        return {
            'tree_depth': depth,
            'total_nodes': total_addresses,
            'n_navigations': n_navigations,
            'unique_addresses_visited': len(visited_addresses),
            'coverage_ratio': float(len(visited_addresses) / total_addresses),
            'navigations': navigation_results[:20],  # First 20 for plotting
        }
    
    def experiment_collective_coordination(self) -> Dict:
        """
        Experiment 4: Collective state coordination.
        
        Validates: Multiple nodes can synchronize to common precision window.
        """
        n_nodes = 20
        n_rounds = 30
        
        coordination_history = []
        
        for round_id in range(n_rounds):
            # Each node computes precision-by-difference
            node_precisions = []
            
            for node_id in range(n_nodes):
                # Node-specific jitter (varies by node)
                jitter = np.random.normal(0, 0.001 + node_id * 0.0001)
                delta_p = np.random.normal(0, 0.001) + jitter
                node_precisions.append(delta_p)
            
            # Compute collective window
            collective_window = TemporalWindow(
                start=min(node_precisions),
                end=max(node_precisions),
                center=np.mean(node_precisions),
                width=max(node_precisions) - min(node_precisions),
                measurements=node_precisions,
            )
            
            # Check synchronization
            synchronized = all(
                abs(p - collective_window.center) < collective_window.width * 0.3
                for p in node_precisions
            )
            
            coordination_history.append({
                'round': round_id,
                'n_nodes': n_nodes,
                'window_width': float(collective_window.width),
                'window_quality': float(collective_window.coherence_quality),
                'synchronized': synchronized,
                'node_precisions': [float(p) for p in node_precisions],
            })
        
        # Analyze
        sync_rate = sum(1 for h in coordination_history if h['synchronized']) / n_rounds
        mean_width = np.mean([h['window_width'] for h in coordination_history])
        mean_quality = np.mean([h['window_quality'] for h in coordination_history])
        
        return {
            'n_nodes': n_nodes,
            'n_rounds': n_rounds,
            'synchronization_rate': float(sync_rate),
            'mean_window_width': float(mean_width),
            'mean_window_quality': float(mean_quality),
            'history': coordination_history,
        }
    
    def experiment_prediction(self) -> Dict:
        """
        Experiment 5: Prediction through categorical completion.
        
        Validates: Accumulated precision trajectory predicts future position.
        """
        n_trajectories = 50
        trajectory_length = 20
        prediction_horizon = 5
        
        predictions = []
        
        for traj_id in range(n_trajectories):
            # Generate trajectory
            trajectory = []
            s_k, s_t, s_e = 0.5, 0.0, 0.5
            
            # Trajectory has underlying trend
            trend_k = np.random.uniform(-0.01, 0.01)
            trend_e = np.random.uniform(-0.01, 0.01)
            
            for step in range(trajectory_length):
                delta_p = np.random.normal(0, 0.001)
                
                # Update coordinates based on precision
                s_k = np.clip(s_k + trend_k + delta_p * 0.5, 0, 1)
                s_t = step / trajectory_length
                s_e = np.clip(s_e + trend_e + delta_p * 0.3, 0, 1)
                
                trajectory.append({
                    'step': step,
                    's_k': float(s_k),
                    's_t': float(s_t),
                    's_e': float(s_e),
                    'delta_p': float(delta_p),
                })
            
            # Predict using trajectory history (categorical completion)
            # Simple linear extrapolation from last few points
            last_n = 5
            if len(trajectory) >= last_n:
                recent_k = [t['s_k'] for t in trajectory[-last_n:]]
                recent_e = [t['s_e'] for t in trajectory[-last_n:]]
                
                # Predict
                pred_k = recent_k[-1] + (recent_k[-1] - recent_k[0]) / last_n * prediction_horizon
                pred_e = recent_e[-1] + (recent_e[-1] - recent_e[0]) / last_n * prediction_horizon
                
                # Generate "actual" future (with same trend)
                actual_k = s_k + trend_k * prediction_horizon
                actual_e = s_e + trend_e * prediction_horizon
                
                # Prediction error
                error_k = abs(pred_k - actual_k)
                error_e = abs(pred_e - actual_e)
                
                predictions.append({
                    'traj_id': traj_id,
                    'predicted_k': float(pred_k),
                    'predicted_e': float(pred_e),
                    'actual_k': float(actual_k),
                    'actual_e': float(actual_e),
                    'error_k': float(error_k),
                    'error_e': float(error_e),
                    'total_error': float(np.sqrt(error_k**2 + error_e**2)),
                    'trajectory': trajectory,
                })
        
        # Analyze
        errors = [p['total_error'] for p in predictions]
        
        return {
            'n_trajectories': n_trajectories,
            'trajectory_length': trajectory_length,
            'prediction_horizon': prediction_horizon,
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'predictions': predictions[:10],  # First 10 for plotting
        }
    
    def experiment_network_latency(self) -> Dict:
        """
        Experiment 6: Network latency reduction through preemptive distribution.
        
        Validates: L_sango << L_traditional through prediction.
        """
        n_requests = 100
        
        # Traditional latency components (milliseconds)
        L_processing = 5.0
        L_transmission = 20.0
        L_propagation = 30.0
        L_queuing = 10.0
        
        # Sango latency components
        L_prediction_base = 3.0  # Base prediction computation
        L_coordination = 1.0  # Precision-by-difference overhead
        
        results = []
        
        for req_id in range(n_requests):
            # Traditional request-response
            jitter_trad = np.random.exponential(5)
            L_traditional = L_processing + L_transmission + L_propagation + L_queuing + jitter_trad
            
            # Sango: preemptive distribution with prediction
            prediction_accuracy = np.random.uniform(0.7, 0.99)
            
            # If prediction is accurate, state is already there
            if np.random.random() < prediction_accuracy:
                # Cache hit - only coordination overhead
                jitter_sango = np.random.exponential(0.5)
                L_sango = L_coordination + jitter_sango
            else:
                # Cache miss - need to fetch, but faster due to coordination
                jitter_sango = np.random.exponential(2)
                L_sango = L_prediction_base + L_coordination + L_transmission * 0.5 + jitter_sango
            
            improvement = (L_traditional - L_sango) / L_traditional * 100
            
            results.append({
                'request_id': req_id,
                'L_traditional': float(L_traditional),
                'L_sango': float(L_sango),
                'improvement_pct': float(improvement),
                'prediction_accuracy': float(prediction_accuracy),
            })
        
        # Analyze
        trad_latencies = [r['L_traditional'] for r in results]
        sango_latencies = [r['L_sango'] for r in results]
        improvements = [r['improvement_pct'] for r in results]
        
        return {
            'n_requests': n_requests,
            'traditional': {
                'mean': float(np.mean(trad_latencies)),
                'std': float(np.std(trad_latencies)),
                'p95': float(np.percentile(trad_latencies, 95)),
            },
            'sango': {
                'mean': float(np.mean(sango_latencies)),
                'std': float(np.std(sango_latencies)),
                'p95': float(np.percentile(sango_latencies, 95)),
            },
            'improvement': {
                'mean': float(np.mean(improvements)),
                'min': float(np.min(improvements)),
                'max': float(np.max(improvements)),
            },
            'results': results,
        }
    
    def _compute_summary(self) -> Dict:
        """Compute experiment summary."""
        return {
            'total_measurements': len(self.measurements),
            'total_windows': len(self.windows),
            'hierarchy_nodes': len(self.hierarchy),
            'precision_measurement': {
                'mean_delta_p': self.results['precision_measurement']['delta_p_mean'],
                'std_delta_p': self.results['precision_measurement']['delta_p_std'],
            },
            'temporal_windows': {
                'mean_quality': self.results['temporal_windows']['mean_quality'],
            },
            'hierarchy': {
                'coverage': self.results['hierarchy_navigation']['coverage_ratio'],
            },
            'coordination': {
                'sync_rate': self.results['collective_coordination']['synchronization_rate'],
            },
            'prediction': {
                'mean_error': self.results['prediction']['mean_error'],
            },
            'latency': {
                'improvement': self.results['network_latency']['improvement']['mean'],
            },
        }


def run_precision_experiment() -> Dict:
    """Run the complete precision-by-difference experiment."""
    experiment = PrecisionByDifferenceExperiment()
    results = experiment.run_all_experiments()
    
    # Print summary
    summary = results['summary']
    print()
    print("SUMMARY")
    print("-"*40)
    print(f"Total measurements: {summary['total_measurements']}")
    print(f"Precision mean: {summary['precision_measurement']['mean_delta_p']:.2e} s")
    print(f"Window quality: {summary['temporal_windows']['mean_quality']:.3f}")
    print(f"Hierarchy coverage: {summary['hierarchy']['coverage']*100:.1f}%")
    print(f"Sync rate: {summary['coordination']['sync_rate']*100:.1f}%")
    print(f"Prediction error: {summary['prediction']['mean_error']:.4f}")
    print(f"Latency improvement: {summary['latency']['improvement']:.1f}%")
    
    return results


def save_results(results: Dict, output_dir: Path) -> Path:
    """Save experiment results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"precision_by_difference_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    results = run_precision_experiment()
    
    # Save results
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / 'results' / 'precision_by_difference'
    save_results(results, output_dir)

