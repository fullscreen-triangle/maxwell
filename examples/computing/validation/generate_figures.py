#!/usr/bin/env python
"""
Master Figure Generation Script
================================

Generates all publication-quality visualizations for:
1. Maxwell's Demon 7-fold dissolution arguments
2. Semiconductor validation experiments
3. Integrated circuit 7-component architecture
4. Unified theoretical framework (oscillator-processor duality)
5. Data-driven figures from experimental results

Output: High-resolution PNGs/PDFs in results/figures/ and results/publication/
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maxwell_validation.visualize_dissolution import generate_all_dissolution_figures
from maxwell_validation.visualize_semiconductor import generate_all_semiconductor_figures
from maxwell_validation.visualize_unified_framework import generate_all_unified_figures
from maxwell_validation.publication_figures import generate_publication_figures
from maxwell_validation.maxwell_images import generate_maxwell_figures
from maxwell_validation.processor_images import generate_processor_figures


def main():
    """Generate all publication figures"""
    print("=" * 80)
    print("MAXWELL VALIDATION FRAMEWORK: COMPLETE FIGURE GENERATION")
    print("=" * 80)
    print()
    
    start_time = time.time()
    output_base = Path("results/figures")
    
    all_figures = {}
    
    # =========================================================================
    # SECTION 1: Maxwell's Demon Dissolution (Schematic)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: MAXWELL'S DEMON DISSOLUTION (Schematic Panels)")
    print("=" * 80)
    
    try:
        dissolution_figures = generate_all_dissolution_figures(
            str(output_base / "dissolution")
        )
        all_figures.update({f"dissolution_{k}": v for k, v in dissolution_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate dissolution figures: {e}")
    
    # =========================================================================
    # SECTION 2: Semiconductor & Integrated Circuits (Schematic)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: SEMICONDUCTOR & INTEGRATED CIRCUITS (Schematic)")
    print("=" * 80)
    
    try:
        semiconductor_figures = generate_all_semiconductor_figures(
            str(output_base / "semiconductor")
        )
        all_figures.update({f"semiconductor_{k}": v for k, v in semiconductor_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate semiconductor figures: {e}")
    
    # =========================================================================
    # SECTION 3: Unified Theoretical Framework
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: UNIFIED FRAMEWORK (Oscillator-Processor Duality)")
    print("=" * 80)
    
    try:
        unified_figures = generate_all_unified_figures(
            str(output_base / "unified")
        )
        all_figures.update({f"unified_{k}": v for k, v in unified_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate unified figures: {e}")
    
    # =========================================================================
    # SECTION 4: Publication Panel Figures (7 Arguments)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: PUBLICATION PANELS (7-Fold Dissolution)")
    print("=" * 80)
    
    try:
        publication_figures = generate_publication_figures()
        all_figures.update({f"publication_{k}": v for k, v in publication_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate publication figures: {e}")
    
    # =========================================================================
    # SECTION 5: Data-Driven Maxwell Figures
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: MAXWELL DEMON DATA FIGURES")
    print("=" * 80)
    
    try:
        maxwell_data_figures = generate_maxwell_figures()
        if maxwell_data_figures:
            all_figures.update({f"maxwell_data_{k}": v for k, v in maxwell_data_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate Maxwell data figures: {e}")
    
    # =========================================================================
    # SECTION 6: Data-Driven Processor Figures
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: PROCESSOR DATA FIGURES")
    print("=" * 80)
    
    try:
        processor_data_figures = generate_processor_figures()
        if processor_data_figures:
            all_figures.update({f"processor_data_{k}": v for k, v in processor_data_figures.items()})
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate processor data figures: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal figures generated: {len(all_figures)}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nOutput directories:")
    print(f"  • Schematics: {output_base.absolute()}")
    print(f"  • Publication: {(output_base.parent / 'publication').absolute()}")
    
    print("\nFigures by category:")
    
    categories = {
        'dissolution': 'Dissolution arguments (schematic)',
        'semiconductor': 'Semiconductor/IC (schematic)',
        'unified': 'Unified framework',
        'publication': 'Publication panels',
        'maxwell_data': 'Maxwell data figures',
        'processor_data': 'Processor data figures',
    }
    
    for prefix, label in categories.items():
        count = sum(1 for k in all_figures if k.startswith(prefix))
        if count > 0:
            print(f"  • {label}: {count}")
    
    print("\n" + "=" * 80)
    print("ALL FIGURES READY FOR PUBLICATION")
    print("=" * 80)
    
    return all_figures


if __name__ == "__main__":
    figures = main()
