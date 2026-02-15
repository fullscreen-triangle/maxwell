"""
Generate all figures for Dodecapartite Virtual Microscopy paper.

This script generates 6 main figures with multiple panels each:
- Figure 1: Quintupartite/Dodecapartite Constraint Architecture
- Figure 2: Oxygen Triangulation (Metabolic GPS)
- Figure 3: Multi-Modal Constraint Integration
- Figure 4: Experimental Validation
- Figure 5: Biological Applications
- Figure 6: Computational Implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns
from pathlib import Path
import sys

# Import figure generation functions
from figure_01_constraint_architecture import generate_figure_1
from figure_02_oxygen_triangulation import generate_figure_2
from figure_03_constraint_integration import generate_figure_3
from figure_04_experimental_validation import generate_figure_4
from figure_05_biological_applications import generate_figure_5
from figure_06_computational_implementation import generate_figure_6

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path(__file__).parent
DPI = 300  # High resolution for publication

def main():
    """Generate all figures."""
    print("Generating figures for Dodecapartite Virtual Microscopy paper...")
    
    print("\n[1/6] Generating Figure 1: Constraint Architecture...")
    generate_figure_1(OUTPUT_DIR, DPI)
    
    print("[2/6] Generating Figure 2: Oxygen Triangulation...")
    generate_figure_2(OUTPUT_DIR, DPI)
    
    print("[3/6] Generating Figure 3: Multi-Modal Constraint Integration...")
    generate_figure_3(OUTPUT_DIR, DPI)
    
    print("[4/6] Generating Figure 4: Experimental Validation...")
    generate_figure_4(OUTPUT_DIR, DPI)
    
    print("[5/6] Generating Figure 5: Biological Applications...")
    generate_figure_5(OUTPUT_DIR, DPI)
    
    print("[6/6] Generating Figure 6: Computational Implementation...")
    generate_figure_6(OUTPUT_DIR, DPI)
    
    print("\n[OK] All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
