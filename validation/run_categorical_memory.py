#!/usr/bin/env python
"""
Run the Categorical Memory (S-RAM) System Demonstration

This script demonstrates the virtual RAM system based on:
- S-entropy hierarchy navigation
- Precision-by-difference addressing
- Categorical completion for memory management
- Hardware oscillator mapping to molecular frequencies

The system enhances existing memory by using categorical principles
to determine which data should be readily available (hot) vs archived (cold).
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maxwell_validation.categorical_memory.demo import run_all_demonstrations


def convert_for_json(obj):
    """Convert numpy types and other non-serializable objects for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def save_results(results):
    """Save results to JSON file."""
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'categorical_memory')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'demonstration_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("="*70)
    print("S-RAM: CATEGORICAL MEMORY SYSTEM")
    print("="*70)
    print()
    print("This system implements virtual RAM based on S-entropy navigation.")
    print()
    print("Core Principles:")
    print("  1. History IS the address - access patterns encode location")
    print("  2. Precision-by-difference provides navigation coordinates")
    print("  3. Categorical completion determines optimal data placement")
    print("  4. 3^k hierarchy gives recursive self-similar structure")
    print("  5. Hardware oscillations map to molecular frequencies")
    print()
    print("This doesn't replace existing memory - it enhances it by")
    print("providing intelligent caching based on categorical analysis.")
    print()
    
    # Run demonstrations
    results = run_all_demonstrations()
    
    # Save results
    save_results(results)
    
    # Print key insights
    print("\n" + "="*70)
    print("INSIGHTS FROM DEMONSTRATION")
    print("="*70)
    
    print("\n1. PRECISION-BY-DIFFERENCE AS ADDRESS:")
    print("   The difference between reference and local time is NOT an error.")
    print("   It IS the address in S-entropy space. Accumulated differences")
    print("   form a trajectory that uniquely identifies a position in the")
    print("   recursive 3^k hierarchy.")
    
    print("\n2. CATEGORICAL NAVIGATION vs PREDICTION:")
    print("   Traditional caching uses prediction (LRU, LFU, etc.).")
    print("   Categorical memory uses NAVIGATION - the endpoint is already")
    print("   encoded in the trajectory. We extract it, not predict it.")
    
    print("\n3. THE 3^k BRANCHING STRUCTURE:")
    print("   Each BMD operation decomposes into 3 sub-operations.")
    print("   This creates a tree where each node has 3 children.")
    print("   At depth d, there are 3^d possible positions.")
    print("   This is the S-entropy address space.")
    
    print("\n4. HARMONIC COINCIDENCE MAPPING:")
    print("   Hardware oscillators (CPU, memory, I/O) have characteristic")
    print("   frequencies. When these form harmonic relationships with")
    print("   molecular frequencies, we can perform categorical measurement")
    print("   (snapshot access) with zero backaction.")
    
    print("\n5. MEMORY AS MAXWELL DEMON:")
    print("   The memory controller is a Maxwell Demon navigating the")
    print("   hierarchy. 'Hot' data is near the current position in")
    print("   S-entropy space. 'Cold' data is categorically distant.")
    print("   Promotion/demotion follows categorical completion paths.")
    
    return results


if __name__ == "__main__":
    main()
