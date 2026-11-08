#!/usr/bin/env python3
"""
Maxwell's Demon Resolution: Complete Validation Suite
======================================================

This script runs the complete validation pipeline:
1. All 7 dissolution experiments
2. Generates publication-quality panel figures (4 charts each)
3. Stores all results persistently
4. Generates summary reports

Usage:
    python run_validation.py

Output:
    results/
    ├── experiments/       # JSON files for each experiment
    ├── data/             # CSV files with raw data
    ├── figures/          # Generated visualizations
    ├── publication/      # Publication-ready panel figures
    └── summary/          # Validation summaries
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maxwell_validation import (
    MaxwellDemonExperiments,
    ExperimentConfig,
    generate_all_panel_figures,
    DissolutionValidator,
    validate_quantum_gates,
    validate_integrated_circuit,
    get_results_manager,
    SemiconductorValidationExperiments,
    IntegratedCircuitValidationExperiments,
    SemiconductorExperimentConfig,
)


def run_complete_validation(output_dir: str = "results"):
    """
    Run the complete Maxwell's Demon validation suite.

    This produces:
    - 7 experiments with persistent results
    - 7 panel figures (4 charts each) for publication
    - Quantum gates validation
    - Integrated circuit validation
    - Summary report
    """
    start_time = time.time()

    print("=" * 80)
    print("MAXWELL'S DEMON RESOLUTION: COMPLETE VALIDATION SUITE")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {output_dir}\n")

    # Initialize results manager
    results_manager = get_results_manager(output_dir)

    # =========================================================================
    # PHASE 1: Run Dissolution Experiments
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DISSOLUTION EXPERIMENTS")
    print("=" * 80)

    config = ExperimentConfig(
        n_molecules=100,
        box_size=10.0,
        seed=42,
        output_dir=output_dir
    )

    experiments = MaxwellDemonExperiments(config)
    experiment_results = experiments.run_all_experiments()

    # =========================================================================
    # PHASE 2: Run Dissolution Validator (Statistical)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: STATISTICAL DISSOLUTION VALIDATION")
    print("=" * 80)

    validator = DissolutionValidator()
    dissolution_results = validator.run_all_validations()
    validator.print_summary(dissolution_results)

    # =========================================================================
    # PHASE 3: Validate Quantum Gates (SSRN 5680582)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: QUANTUM GATES VALIDATION (SSRN 5680582)")
    print("=" * 80)

    quantum_results = validate_quantum_gates()
    print(f"\nQuantum gates all passed: {quantum_results['all_passed']}")
    print(f"Clock frequency: {quantum_results.get('clock_frequency', 758)} Hz")
    print(f"Coherence time: {quantum_results.get('coherence_time', 0.010)*1000:.1f} ms")

    # =========================================================================
    # PHASE 4: Validate Integrated Circuit (SSRN 5680570) - Quick Check
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: INTEGRATED CIRCUIT QUICK VALIDATION (SSRN 5680570)")
    print("=" * 80)

    ic_results = validate_integrated_circuit()
    print(f"\nIntegrated circuit all passed: {ic_results['all_passed']}")

    # Print component validations
    for component, passed in ic_results.get('validations', {}).items():
        status = "✓" if passed else "✗"
        print(f"  {status} {component}")

    # =========================================================================
    # PHASE 5: Comprehensive Semiconductor Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: COMPREHENSIVE SEMICONDUCTOR VALIDATION")
    print("=" * 80)

    semi_config = SemiconductorExperimentConfig(output_dir=output_dir)
    semi_experiments = SemiconductorValidationExperiments(semi_config)
    semi_results = semi_experiments.run_all_experiments()

    # =========================================================================
    # PHASE 6: Comprehensive Integrated Circuit Validation (7 Components)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: 7-COMPONENT INTEGRATED CIRCUIT VALIDATION")
    print("=" * 80)

    ic_experiments = IntegratedCircuitValidationExperiments(semi_config)
    ic_full_results = ic_experiments.run_all_experiments()

    # =========================================================================
    # PHASE 7: Generate Publication Figures
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: GENERATING PUBLICATION FIGURES")
    print("=" * 80)

    publication_dir = str(Path(output_dir) / "publication")
    figure_paths = generate_all_panel_figures(publication_dir)

    print(f"\nGenerated {len(figure_paths)} panel figures")
    for arg, path in figure_paths.items():
        print(f"  {arg}: {path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    all_experiments_passed = experiment_results['all_validated']
    all_dissolution_passed = all(r.validated for r in dissolution_results.values())
    all_quantum_passed = quantum_results['all_passed']
    all_ic_passed = ic_results['all_passed']
    all_semi_passed = semi_results['all_validated']
    all_ic_full_passed = ic_full_results['all_validated']

    overall_passed = (all_experiments_passed and all_dissolution_passed and
                      all_quantum_passed and all_ic_passed and
                      all_semi_passed and all_ic_full_passed)

    print(f"\nExperiments (7 dissolution): {'PASSED' if all_experiments_passed else 'FAILED'}")
    print(f"Statistical dissolution: {'PASSED' if all_dissolution_passed else 'FAILED'}")
    print(f"Quantum gates (SSRN 5680582): {'PASSED' if all_quantum_passed else 'FAILED'}")
    print(f"Integrated circuit quick (SSRN 5680570): {'PASSED' if all_ic_passed else 'FAILED'}")
    print(f"Semiconductor comprehensive: {'PASSED' if all_semi_passed else 'FAILED'}")
    print(f"IC 7-component comprehensive: {'PASSED' if all_ic_full_passed else 'FAILED'}")

    print(f"\n{'='*80}")
    if overall_passed:
        print("OVERALL: ALL VALIDATIONS PASSED")
        print("\nCONCLUSION: Maxwell's Demon does not exist.")
        print("The apparent 'demon' is the projection of hidden categorical dynamics")
        print("onto the observable kinetic face of information.")
    else:
        print("OVERALL: SOME VALIDATIONS FAILED")
        print("Check individual results for details.")
    print(f"{'='*80}")

    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - experiments/: Individual experiment JSON files")
    print(f"  - data/: Raw CSV data files")
    print(f"  - figures/: Visualization images")
    print(f"  - publication/: Publication-ready panel figures (4 charts each)")
    print(f"  - summary/: Validation summary reports")

    return {
        'experiments': experiment_results,
        'dissolution': dissolution_results,
        'quantum_gates': quantum_results,
        'integrated_circuit': ic_results,
        'semiconductor': semi_results,
        'ic_full': ic_full_results,
        'figures': figure_paths,
        'overall_passed': overall_passed,
        'runtime': elapsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Maxwell's Demon Resolution Validation Suite"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results (default: results)"
    )

    args = parser.parse_args()

    results = run_complete_validation(args.output)

    # Exit with appropriate code
    sys.exit(0 if results['overall_passed'] else 1)

