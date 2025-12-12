"""
Categorical Memory System Demonstration

This script demonstrates the S-RAM (categorical memory) system in action,
showing how precision-by-difference values navigate the S-entropy hierarchy
to achieve intelligent memory management.

Key demonstrations:
1. Hardware oscillator capture (real timing data)
2. Precision-by-difference calculation
3. S-entropy address creation and navigation
4. Categorical hierarchy traversal
5. Memory tier management via categorical completion
"""

import time
import numpy as np
from typing import Dict, Any, List
import json

from .hardware_oscillator import HardwareOscillatorCapture
from .precision_calculator import PrecisionByDifferenceCalculator
from .s_entropy_address import SEntropyAddress, SCoordinate
from .categorical_hierarchy import CategoricalHierarchy
from .memory_controller import CategoricalMemoryController, MemoryTier


def demonstrate_hardware_oscillators() -> Dict[str, Any]:
    """
    Demonstrate real hardware oscillator capture.
    
    These are actual timing variations from the computer's hardware,
    not simulations.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Hardware Oscillator Capture")
    print("="*60)
    
    oscillator = HardwareOscillatorCapture(sample_rate=100)
    
    # Calibrate
    print("\nCalibrating oscillators (measuring reference values)...")
    calibration = oscillator.calibrate(duration=0.5)
    print(f"Calibration complete: {calibration}")
    
    # Capture samples
    print("\nCapturing multi-source samples...")
    all_samples = []
    for i in range(10):
        samples = oscillator.capture_multi_source()
        all_samples.extend(samples)
        time.sleep(0.01)
    
    # Analyze
    precision_diffs = [s.precision_diff for s in all_samples]
    
    results = {
        'n_samples': len(all_samples),
        'sources': list(set(s.source for s in all_samples)),
        'precision_diff_mean': float(np.mean(precision_diffs)),
        'precision_diff_std': float(np.std(precision_diffs)),
        'precision_diff_range': [float(min(precision_diffs)), float(max(precision_diffs))],
        'calibration': calibration,
    }
    
    print(f"\nCaptured {results['n_samples']} samples from {len(results['sources'])} sources")
    print(f"Precision-by-difference statistics:")
    print(f"  Mean: {results['precision_diff_mean']:.2e}")
    print(f"  Std:  {results['precision_diff_std']:.2e}")
    print(f"  Range: [{results['precision_diff_range'][0]:.2e}, {results['precision_diff_range'][1]:.2e}]")
    
    # Convert to S-coordinate
    signature = oscillator.get_precision_signature(n_samples=10)
    s_coord = oscillator.signature_to_scoordinate(signature)
    
    results['s_coordinate'] = {
        'S_k': s_coord.S_k,
        'S_t': s_coord.S_t, 
        'S_e': s_coord.S_e,
    }
    
    print(f"\nConverted to S-coordinate:")
    print(f"  S_k (kinetic):  {s_coord.S_k:.6f}")
    print(f"  S_t (thermal):  {s_coord.S_t:.6f}")
    print(f"  S_e (entropic): {s_coord.S_e:.6f}")
    
    return results


def demonstrate_precision_navigation() -> Dict[str, Any]:
    """
    Demonstrate precision-by-difference navigation through S-entropy space.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Precision-by-Difference Navigation")
    print("="*60)
    
    calculator = PrecisionByDifferenceCalculator()
    
    # Create addresses and build trajectories
    print("\nCreating addresses and building trajectories...")
    
    addresses = {}
    for name in ['alpha', 'beta', 'gamma']:
        addr = calculator.create_address(name)
        
        # Build trajectory with multiple updates
        for _ in range(20):
            calculator.update_address(name)
            time.sleep(0.005)  # Small delay to accumulate differences
        
        addresses[name] = addr
    
    # Analyze trajectories
    results = {
        'addresses': {},
        'navigation_paths': {},
    }
    
    for name, addr in addresses.items():
        depth, branches = calculator.compute_hierarchy_position(name)
        completion = calculator.predict_optimal_location(name)
        
        results['addresses'][name] = {
            'trajectory_hash': addr.trajectory_hash,
            'trajectory_length': len(addr.trajectory),
            'hierarchy_depth': depth,
            'branch_path': branches[:10],  # First 10 branches
            'completion_predicted': {
                'S_k': completion.S_k,
                'S_t': completion.S_t,
                'S_e': completion.S_e,
            } if completion else None,
        }
        
        print(f"\nAddress '{name}':")
        print(f"  Hash: {addr.trajectory_hash}")
        print(f"  Depth: {depth}")
        print(f"  Branches: {branches[:5]}...")
    
    # Navigate between addresses
    for from_name in ['alpha', 'beta']:
        for to_name in ['beta', 'gamma']:
            if from_name != to_name:
                path, cost = calculator.navigate_between(from_name, to_name)
                results['navigation_paths'][f"{from_name}->{to_name}"] = {
                    'path_length': len(path),
                    'cost': cost,
                    'path_preview': path[:5],
                }
                print(f"\nNavigation {from_name} -> {to_name}:")
                print(f"  Path length: {len(path)}")
                print(f"  Cost: {cost:.4f}")
    
    results['calculator_stats'] = calculator.get_statistics()
    
    return results


def demonstrate_hierarchy_storage() -> Dict[str, Any]:
    """
    Demonstrate the 3^k categorical hierarchy for data storage.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Categorical Hierarchy Storage")
    print("="*60)
    
    hierarchy = CategoricalHierarchy[str]()
    calculator = PrecisionByDifferenceCalculator()
    
    # Store some data with different access patterns
    print("\nStoring data with categorical addressing...")
    
    stored = []
    for i in range(20):
        key = f"data_{i}"
        data = f"Content for item {i}"
        
        # Create address through precision calculations
        addr = calculator.create_address(key)
        for _ in range(5 + i):  # Different trajectory lengths
            calculator.update_address(key)
            time.sleep(0.002)
        
        # Store in hierarchy
        node = hierarchy.store(key, data, addr)
        stored.append({
            'key': key,
            'depth': node.depth,
            'branch_index': node.branch_index,
            'path': node.path_from_root[:5],
        })
        
        if i < 5:
            print(f"  Stored '{key}' at depth {node.depth}, path: {node.path_from_root[:5]}")
    
    # Retrieve and analyze
    print("\nRetrieving and analyzing...")
    
    results = {
        'stored_items': stored,
        'branch_stats': hierarchy.get_branch_statistics(),
        'retrievals': [],
    }
    
    # Test retrieval
    for i in [0, 5, 10, 15]:
        key = f"data_{i}"
        data, node = hierarchy.retrieve_by_key(key)
        results['retrievals'].append({
            'key': key,
            'found': data is not None,
            'node_depth': node.depth if node else None,
        })
    
    # Find nearest neighbors
    print("\nFinding nearest neighbors in categorical space...")
    test_addr = calculator.get_address("data_10")
    if test_addr:
        neighbors = hierarchy.find_nearest(test_addr, max_distance=3)
        results['nearest_neighbors'] = [
            {'data': d, 'distance': dist}
            for d, dist, _ in neighbors[:5]
        ]
        print(f"  Found {len(neighbors)} neighbors within distance 3")
    
    # Compression
    print("\nCompressing hierarchy (categorical filtering)...")
    compression = hierarchy.compress_hierarchy()
    results['compression'] = compression
    print(f"  Removed {compression['nodes_removed']} unused nodes")
    print(f"  Compression ratio: {compression['compression_ratio']:.2f}x")
    
    return results


def demonstrate_memory_controller() -> Dict[str, Any]:
    """
    Demonstrate the full categorical memory controller (S-RAM).
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Categorical Memory Controller (S-RAM)")
    print("="*60)
    
    controller = CategoricalMemoryController[str]()
    
    # Store test data
    print("\nStoring test data across categorical tiers...")
    
    for i in range(30):
        key = f"item_{i}"
        data = f"This is the content for item number {i}" * 10
        entry = controller.store(key, data)
        
        if i < 5:
            print(f"  '{key}' stored in tier: {entry.tier.name}")
    
    # Show initial tier distribution
    stats = controller.get_statistics()
    print(f"\nInitial tier distribution:")
    for tier, count in stats['tier_sizes'].items():
        print(f"  {tier}: {count} items")
    
    # Simulate access pattern
    print("\nSimulating access pattern (categorical navigation)...")
    
    import random
    
    # Access some items repeatedly (they should get promoted)
    hot_items = ['item_5', 'item_10', 'item_15']
    for _ in range(10):
        for key in hot_items:
            controller.retrieve(key)
            time.sleep(0.005)
    
    # Access others once (should stay cold)
    cold_items = ['item_25', 'item_26', 'item_27']
    for key in cold_items:
        controller.retrieve(key)
    
    # Show updated statistics
    stats = controller.get_statistics()
    print(f"\nAfter access pattern:")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Promotions: {stats['promotions']}")
    print(f"  Demotions: {stats['demotions']}")
    
    # Prefetch based on categorical completion
    print("\nPrefetching based on categorical completion...")
    prefetched = controller.prefetch(n=5)
    print(f"  Prefetched: {prefetched}")
    
    # Show current position
    position = controller.get_categorical_position()
    print(f"\nCurrent categorical position:")
    if position['coordinate']:
        print(f"  S_k: {position['coordinate']['S_k']:.6f}")
        print(f"  S_t: {position['coordinate']['S_t']:.6f}")
        print(f"  S_e: {position['coordinate']['S_e']:.6f}")
    print(f"  Depth: {position['depth']}")
    print(f"  Trajectory hash: {position['trajectory_hash']}")
    
    results = {
        'final_statistics': stats,
        'current_position': position,
        'hot_items': hot_items,
        'cold_items': cold_items,
        'prefetched': prefetched,
    }
    
    return results


def demonstrate_harmonic_coincidences() -> Dict[str, Any]:
    """
    Demonstrate harmonic coincidence detection between hardware and molecular frequencies.
    
    This is how the virtual spectrometer maps hardware oscillations to molecular snapshots.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 5: Harmonic Coincidence Detection")
    print("="*60)
    
    oscillator = HardwareOscillatorCapture()
    
    # Search for coincidences with typical molecular frequencies
    molecular_freqs = {
        'C-H_stretch': 9.0e13,  # ~3000 cm^-1
        'C=O_stretch': 5.1e13,  # ~1700 cm^-1
        'O-H_bend': 4.5e13,     # ~1500 cm^-1
        'protein_vibration': 1e12,  # THz regime
        'membrane_fluctuation': 1e6,  # MHz regime
    }
    
    results = {
        'molecular_frequencies': molecular_freqs,
        'coincidences_found': {},
    }
    
    for mol_name, freq in molecular_freqs.items():
        print(f"\nSearching coincidences for {mol_name} ({freq:.2e} Hz)...")
        
        coincidences = oscillator.get_harmonic_coincidences(freq, tolerance=0.001)
        
        if coincidences:
            # Best coincidence
            best = min(coincidences, key=lambda c: c['error'])
            results['coincidences_found'][mol_name] = {
                'best_source': best['source'],
                'harmonic_ratio': f"{best['harmonic_n']}:{best['harmonic_m']}",
                'error': best['error'],
                'total_found': len(coincidences),
            }
            print(f"  Found {len(coincidences)} coincidences")
            print(f"  Best: {best['source']} at {best['harmonic_n']}:{best['harmonic_m']} (error: {best['error']:.4f})")
        else:
            print(f"  No coincidences found within tolerance")
    
    return results


def run_all_demonstrations() -> Dict[str, Any]:
    """
    Run all demonstrations and compile results.
    """
    print("\n" + "="*70)
    print("CATEGORICAL MEMORY SYSTEM (S-RAM) - COMPLETE DEMONSTRATION")
    print("="*70)
    print("\nThis demonstrates the S-entropy based memory system where:")
    print("  - History IS the address (precision-by-difference trajectory)")
    print("  - Navigation is categorical completion, not prediction")
    print("  - The 3^k hierarchy provides recursive self-similar structure")
    print("  - Hardware oscillations map to molecular frequencies")
    
    all_results = {}
    
    # Run each demonstration
    all_results['hardware_oscillators'] = demonstrate_hardware_oscillators()
    all_results['precision_navigation'] = demonstrate_precision_navigation()
    all_results['hierarchy_storage'] = demonstrate_hierarchy_storage()
    all_results['memory_controller'] = demonstrate_memory_controller()
    all_results['harmonic_coincidences'] = demonstrate_harmonic_coincidences()
    
    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\nKey Results:")
    print(f"  - Captured {all_results['hardware_oscillators']['n_samples']} real hardware samples")
    print(f"  - Created {len(all_results['precision_navigation']['addresses'])} navigable addresses")
    print(f"  - Stored {len(all_results['hierarchy_storage']['stored_items'])} items in hierarchy")
    print(f"  - Memory controller hit rate: {all_results['memory_controller']['final_statistics']['hit_rate']:.2%}")
    print(f"  - Found harmonic coincidences for {len(all_results['harmonic_coincidences']['coincidences_found'])} molecular types")
    
    return all_results


if __name__ == "__main__":
    results = run_all_demonstrations()
    
    # Save results
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'categorical_memory')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'demonstration_results.json')
    
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


