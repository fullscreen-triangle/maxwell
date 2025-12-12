"""
Run Precision-by-Difference Experiment and Generate Panel

This script:
1. Runs the precision-by-difference network experiment
2. Saves the results 
3. Generates the visualization panel
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from maxwell_validation.precision_by_difference_experiment import (
    run_precision_experiment,
    save_results
)


def main():
    print("="*70)
    print("PRECISION-BY-DIFFERENCE NETWORK VALIDATION")
    print("="*70)
    print()
    
    # Run experiment
    print("Running experiment...")
    results = run_precision_experiment()
    
    # Save results
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results' / 'precision_by_difference'
    save_results(results, output_dir)
    
    print()
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print()
    print("Now run 'python generate_all_panels.py' to generate visualizations.")
    

if __name__ == "__main__":
    main()

