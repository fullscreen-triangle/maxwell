#!/usr/bin/env python
"""
Run Processor Benchmark and Biological Quantum Gates Validation

This script:
1. Benchmarks Categorical Processor vs Classical ALU on real tasks
2. Validates biological quantum gates (X, Y, Z, H, CNOT, Phase, Measure)
3. Tests biological transistor circuits (NAND, Inverter, ALU)
4. Saves all results

Based on:
- "On the Thermodynamic Consequences of an Oscillatory Reality" (Sachikonye, 2025)
  - 758 Hz biological clock
  - < 100 μs gate times
  - > 85% fidelity
  - Landauer-optimal efficiency
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maxwell_validation.processor_benchmark import run_processor_benchmark
from maxwell_validation.biological_quantum_gates import run_all_validations


def main():
    print("="*70)
    print("CATEGORICAL PROCESSOR AND BIOLOGICAL GATES VALIDATION")
    print("="*70)
    print()
    print("This validation suite demonstrates:")
    print()
    print("1. CATEGORICAL PROCESSOR vs CLASSICAL ALU")
    print("   - O(1) categorical completion vs O(n), O(n²), O(n³) classical")
    print("   - Speedup comparison on real tasks")
    print("   - Energy efficiency (Landauer limit)")
    print()
    print("2. BIOLOGICAL QUANTUM GATES")
    print("   - X, Y, Z, Hadamard, Phase, T gates")
    print("   - CNOT two-qubit gate")
    print("   - Bell state creation")
    print("   - Measurement (state collapse)")
    print()
    print("3. BIOLOGICAL TRANSISTOR CIRCUITS")
    print("   - Phase-lock gated transistors")
    print("   - CMOS-like inverter")
    print("   - Universal NAND gate")
    print("   - Full ALU (add, subtract, multiply)")
    print()
    print("="*70)
    print()
    
    # Run processor benchmark
    print("\n" + "="*70)
    print("PART 1: PROCESSOR BENCHMARK")
    print("="*70)
    run_processor_benchmark()
    
    # Run biological gates validation
    print("\n" + "="*70)
    print("PART 2: BIOLOGICAL QUANTUM GATES")
    print("="*70)
    run_all_validations()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/processor_benchmark/")
    print("  - results/biological_gates/")


if __name__ == "__main__":
    main()

