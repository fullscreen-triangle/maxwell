"""
Run All Theory Experiments
Master script for running all theoretical framework validation experiments
and generating publication-quality panel charts.
"""

import sys
import os
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_all_experiments():
    """Run all theory experiments and save results."""
    print("=" * 70)
    print("POINCARÉ COMPUTING - THEORETICAL FRAMEWORK VALIDATION")
    print("=" * 70)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment 1: System Topology
    print("\n" + "=" * 50)
    print("[1/5] SYSTEM TOPOLOGY EXPERIMENT")
    print("=" * 50)
    try:
        from system_topology_experiment import run_system_topology_experiment, save_results
        topology_results = run_system_topology_experiment()
        save_results(topology_results, str(results_dir))
        results["system_topology"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {e}")
        results["system_topology"] = f"FAILED: {e}"
    
    # Experiment 2: Exhaustive Computing
    print("\n" + "=" * 50)
    print("[2/5] EXHAUSTIVE COMPUTING EXPERIMENT")
    print("=" * 50)
    try:
        from exhaustive_computing_experiment import run_exhaustive_computing_experiment, save_results
        exhaustive_results = run_exhaustive_computing_experiment()
        save_results(exhaustive_results, str(results_dir))
        results["exhaustive_computing"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {e}")
        results["exhaustive_computing"] = f"FAILED: {e}"
    
    # Experiment 3: Categorical Compiler
    print("\n" + "=" * 50)
    print("[3/5] CATEGORICAL COMPILER EXPERIMENT")
    print("=" * 50)
    try:
        from categorical_compiler_experiment import run_categorical_compiler_experiment, save_results
        compiler_results = run_categorical_compiler_experiment()
        save_results(compiler_results, str(results_dir))
        results["categorical_compiler"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {e}")
        results["categorical_compiler"] = f"FAILED: {e}"
    
    # Experiment 4: Complexity Theory
    print("\n" + "=" * 50)
    print("[4/5] COMPLEXITY THEORY EXPERIMENT")
    print("=" * 50)
    try:
        from complexity_experiment import run_complexity_experiment, save_results
        complexity_results = run_complexity_experiment()
        save_results(complexity_results, str(results_dir))
        results["complexity"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {e}")
        results["complexity"] = f"FAILED: {e}"
    
    # Experiment 5: St-Stellas Thermodynamics
    print("\n" + "=" * 50)
    print("[5/5] ST-STELLAS THERMODYNAMICS EXPERIMENT")
    print("=" * 50)
    try:
        from st_stellas_experiment import run_st_stellas_experiment, save_results
        stellas_results = run_st_stellas_experiment()
        save_results(stellas_results, str(results_dir))
        results["st_stellas"] = "SUCCESS"
    except Exception as e:
        print(f"ERROR: {e}")
        results["st_stellas"] = f"FAILED: {e}"
    
    # Save overall results
    overall_path = results_dir / "experiment_summary.json"
    with open(overall_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "all_success": all(v == "SUCCESS" for v in results.values())
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for exp, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  [{status_symbol}] {exp}: {status}")
    
    return results


def generate_all_panels():
    """Generate all panel charts."""
    print("\n" + "=" * 70)
    print("GENERATING PANEL CHARTS")
    print("=" * 70)
    
    # Create figures directory
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from generate_theory_panels import main as generate_panels
        generate_panels()
        return True
    except Exception as e:
        print(f"ERROR generating panels: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete validation pipeline."""
    start_time = time.time()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run experiments
    results = run_all_experiments()
    
    # Generate panels
    panels_success = generate_all_panels()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Experiments: {sum(1 for v in results.values() if v == 'SUCCESS')}/{len(results)} successful")
    print(f"Panels: {'Generated' if panels_success else 'FAILED'}")
    print("\nResults saved to: poincare/results/")
    print("Figures saved to: poincare/figures/")


if __name__ == "__main__":
    main()

