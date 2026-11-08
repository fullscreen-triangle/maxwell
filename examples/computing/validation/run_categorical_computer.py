#!/usr/bin/env python
"""
Run the Categorical Computer Demonstration

This demonstrates solving real problems using categorical navigation
rather than sequential algorithm execution.

The categorical computer:
1. Translates problems into categorical structures
2. Navigates S-entropy space toward completion
3. Returns the categorical completion point as the solution

This is fundamentally different from classical computing:
- Classical: Execute instructions step by step
- Categorical: Navigate to where constraints naturally meet
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maxwell_validation.categorical_computer.demo import run_all_demonstrations


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
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'categorical_computer')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'demo_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("="*70)
    print("CATEGORICAL COMPUTER")
    print("="*70)
    print()
    print("A new computing paradigm based on S-entropy navigation.")
    print()
    print("Components:")
    print("  - Processor: Categorical completion (oscillator-based)")
    print("  - Memory: S-RAM (precision-by-difference addressing)")
    print("  - Translator: Problem -> Categorical Structure -> Solution")
    print()
    print("The key insight: Problems are not 'solved' by algorithms.")
    print("They are 'navigated' - we find the point in categorical space")
    print("where all constraints naturally meet (categorical completion).")
    print()
    
    # Run demonstrations
    results = run_all_demonstrations()
    
    # Save results
    save_results(results)
    
    # Final analysis
    print("\n" + "="*70)
    print("ARCHITECTURAL INSIGHTS")
    print("="*70)
    
    print("""
1. PROBLEM TRANSLATION (not compilation)
   Classical: Source code -> Machine instructions -> Execute
   Categorical: Problem description -> S-entropy manifold -> Navigate
   
   The translator identifies entities, relations, and constraints,
   then maps them to positions in S-entropy space.

2. NAVIGATION (not execution)
   Classical: CPU executes one instruction at a time
   Categorical: Navigate S-entropy space toward completion
   
   Each 'step' is guided by precision-by-difference values from
   real hardware oscillations. The trajectory IS the computation.

3. CATEGORICAL COMPLETION (not algorithm termination)
   Classical: Algorithm finishes when it reaches 'return'
   Categorical: Navigation stops at completion point
   
   The completion point is where constraints naturally meet.
   It's not computed - it's FOUND through navigation.

4. HARDWARE GROUNDING
   The S-entropy coordinates come from REAL hardware timing:
   - CPU cycle variations
   - Memory access latencies
   - I/O timing jitter
   
   This grounds abstract categorical navigation in physical reality.

5. UNIVERSAL PROBLEM INTERFACE
   Any problem that can be expressed as:
   - Entities (things)
   - Relations (connections)
   - Constraints (requirements)
   
   Can be solved by finding the categorical completion.
   This includes optimization, search, constraint satisfaction,
   pattern matching, and biological system analysis.
""")
    
    return results


if __name__ == "__main__":
    main()
