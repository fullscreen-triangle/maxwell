"""
Master script to generate all shared panels for the three papers:
1. Panel 1: Triple Equivalence (Oscillation ≡ Category ≡ Partition)
2. Panel 2: Entropy Derivation (S = k_B M ln n)
3. Panel 3: Categorical Enthalpy (H = U + Σn_a Φ_a → U + PV)

These panels are shared across:
- Categorical Fluid Dynamics
- Categorical Current Flow  
- Resolution of Loschmidt's Paradox
"""

import subprocess
import sys
import os

def run_panel_script(script_name):
    """Run a panel generation script."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'='*60}")
    print(f"Generating: {script_name}")
    print('='*60)
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"SUCCESS: {script_name}")
    else:
        print(f"ERROR in {script_name}:")
        print(result.stderr)
        return False
    return True

if __name__ == "__main__":
    scripts = [
        'generate_panel1_triple_equivalence.py',
        'generate_panel2_entropy_derivation.py',
        'generate_panel3_categorical_enthalpy.py',
    ]
    
    all_success = True
    for script in scripts:
        success = run_panel_script(script)
        all_success = all_success and success
    
    print("\n" + "="*60)
    if all_success:
        print("ALL PANELS GENERATED SUCCESSFULLY!")
        print("\nOutput files in: poincare/docs/shared-figures/figures/")
        print("  - panel1_triple_equivalence.png/.pdf")
        print("  - panel2_entropy_derivation.png/.pdf")
        print("  - panel3_categorical_enthalpy.png/.pdf")
    else:
        print("SOME PANELS FAILED - Check errors above")
    print("="*60)

