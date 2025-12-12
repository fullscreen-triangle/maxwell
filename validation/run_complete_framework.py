#!/usr/bin/env python
"""
Complete Unified Framework Validation and Visualization
=========================================================

Runs the complete validation suite for:
1. Maxwell's Demon 7-fold Dissolution
2. Semiconductor and Integrated Circuit Validation
3. Oscillator-Processor Duality
4. Categorical Measurement (Trans-Planckian)
5. Unified Theoretical Framework

Generates publication-quality figures for all components.
"""

import sys
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maxwell_validation import (
    # Dissolution experiments
    MaxwellDemonExperiments,
    ExperimentConfig,
    DissolutionValidator,
    
    # Quantum gates and IC
    validate_quantum_gates,
    validate_integrated_circuit,
    
    # Semiconductor experiments
    SemiconductorValidationExperiments,
    IntegratedCircuitValidationExperiments,
    SemiconductorExperimentConfig,
    
    # New unified framework
    validate_oscillator_processor_duality,
    validate_categorical_measurement,
    validate_unified_framework,
    
    # Visualizations
    generate_all_dissolution_figures,
    generate_all_semiconductor_figures,
    generate_all_unified_figures,
    
    # Results management
    get_results_manager,
)


def run_complete_validation(output_dir: str = "results"):
    """Run the complete unified framework validation."""
    print("=" * 80)
    print("UNIFIED CATEGORICAL MECHANICS FRAMEWORK")
    print("Complete Validation and Visualization Suite")
    print("=" * 80)
    
    start_time = time.time()
    output_path = Path(output_dir)
    results_manager = get_results_manager(output_dir)
    
    all_results = {
        'run_id': results_manager.run_id,
        'validations': {},
        'figures': {},
    }
    
    # =========================================================================
    # PHASE 1: Oscillator-Processor Duality
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: OSCILLATOR-PROCESSOR DUALITY")
    print("=" * 80)
    
    duality_results = validate_oscillator_processor_duality()
    all_results['validations']['oscillator_processor'] = {
        'passed': duality_results['all_verified'],
        'summary': {
            'processors_created': duality_results['foundry_stats']['total_processors'],
            'zero_computation': duality_results['zero_computation']['zero_computation_verified'],
            'entropy_equivalence': duality_results['entropy_equivalence']['equivalence_verified'],
        }
    }
    
    # =========================================================================
    # PHASE 2: Categorical Measurement (Trans-Planckian)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: CATEGORICAL MEASUREMENT (TRANS-PLANCKIAN)")
    print("=" * 80)
    
    categorical_results = validate_categorical_measurement()
    all_results['validations']['categorical_measurement'] = {
        'passed': categorical_results['validation']['all_verified'],
        'summary': {
            'temporal_precision': categorical_results['results']['temporal_precision'],
            'orders_below_planck': categorical_results['results']['orders_below_planck'],
            'effective_frequency': categorical_results['results']['effective_frequency'],
        }
    }
    
    # =========================================================================
    # PHASE 3: Unified Framework
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: UNIFIED THEORETICAL FRAMEWORK")
    print("=" * 80)
    
    unified_results = validate_unified_framework()
    all_results['validations']['unified_framework'] = {
        'passed': unified_results['all_verified'],
        'connections': len(unified_results['connections']),
    }
    
    # =========================================================================
    # PHASE 4: Maxwell Demon Dissolution
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: MAXWELL DEMON 7-FOLD DISSOLUTION")
    print("=" * 80)
    
    config = ExperimentConfig(output_dir=output_dir)
    experiments = MaxwellDemonExperiments(config)
    dissolution_results = experiments.run_all_experiments()
    
    all_results['validations']['maxwell_dissolution'] = {
        'passed': dissolution_results['all_validated'],
        'experiments': list(dissolution_results['experiments'].keys()),
    }
    
    # =========================================================================
    # PHASE 5: Semiconductor Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: SEMICONDUCTOR VALIDATION")
    print("=" * 80)
    
    semi_config = SemiconductorExperimentConfig(output_dir=output_dir)
    semi_experiments = SemiconductorValidationExperiments(semi_config)
    semi_results = semi_experiments.run_all_experiments()
    
    all_results['validations']['semiconductor'] = {
        'passed': semi_results['all_validated'],
        'experiments': list(semi_results['experiments'].keys()),
    }
    
    # =========================================================================
    # PHASE 6: Integrated Circuit Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: 7-COMPONENT INTEGRATED CIRCUIT")
    print("=" * 80)
    
    ic_experiments = IntegratedCircuitValidationExperiments(semi_config)
    ic_results = ic_experiments.run_all_experiments()
    
    all_results['validations']['integrated_circuit'] = {
        'passed': ic_results['all_validated'],
        'experiments': list(ic_results['experiments'].keys()),
    }
    
    # =========================================================================
    # PHASE 7: Generate All Figures
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: GENERATING PUBLICATION FIGURES")
    print("=" * 80)
    
    # Dissolution figures
    print("\n[7.1] Dissolution argument figures...")
    dissolution_figs = generate_all_dissolution_figures(
        str(output_path / "figures" / "dissolution")
    )
    all_results['figures']['dissolution'] = dissolution_figs
    
    # Semiconductor figures
    print("\n[7.2] Semiconductor/IC figures...")
    semi_figs = generate_all_semiconductor_figures(
        str(output_path / "figures" / "semiconductor")
    )
    all_results['figures']['semiconductor'] = semi_figs
    
    # Unified framework figures
    print("\n[7.3] Unified framework figures...")
    unified_figs = generate_all_unified_figures(
        str(output_path / "figures" / "unified")
    )
    all_results['figures']['unified'] = unified_figs
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    # Calculate overall pass status
    validations = all_results['validations']
    all_passed = all(v.get('passed', False) for v in validations.values())
    
    print("\n" + "=" * 80)
    print("COMPLETE FRAMEWORK VALIDATION SUMMARY")
    print("=" * 80)
    
    for name, result in validations.items():
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        print(f"  {status}: {name}")
    
    total_figures = sum(len(figs) for figs in all_results['figures'].values())
    
    print(f"\nFigures generated: {total_figures}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nOverall: {'ALL VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    print("=" * 80)
    
    # Save summary
    summary_path = output_path / "validation_summary.json"
    with open(summary_path, 'w') as f:
        # Create serializable summary
        summary = {
            'run_id': all_results['run_id'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'all_passed': all_passed,
            'validations': {
                k: {'passed': v.get('passed', False)} 
                for k, v in validations.items()
            },
            'figure_counts': {
                k: len(v) for k, v in all_results['figures'].items()
            },
        }
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    all_results['all_passed'] = all_passed
    all_results['elapsed'] = elapsed
    
    return all_results


if __name__ == "__main__":
    results = run_complete_validation()

