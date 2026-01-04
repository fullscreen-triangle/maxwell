"""
Quick test script to verify validation experiments can run.
"""

import sys
import os

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
poincare_src = os.path.join(base_path, 'poincare', 'src')
if poincare_src not in sys.path:
    sys.path.insert(0, poincare_src)

print("Testing imports...")

try:
    from virtual_molecule import SCoordinate, VirtualMolecule
    print("✓ virtual_molecule imported")
except ImportError as e:
    print(f"✗ virtual_molecule import failed: {e}")
    sys.exit(1)

try:
    from virtual_chamber import VirtualChamber
    print("✓ virtual_chamber imported")
except ImportError as e:
    print(f"✗ virtual_chamber import failed: {e}")
    sys.exit(1)

try:
    from virtual_aperture import CategoricalAperture
    print("✓ virtual_aperture imported")
except ImportError as e:
    print(f"✗ virtual_aperture import failed: {e}")
    sys.exit(1)

print("\nTesting basic functionality...")

# Test SCoordinate
coord1 = SCoordinate(0.5, 0.5, 0.5)
coord2 = SCoordinate(0.6, 0.6, 0.6)
distance = coord1.distance_to(coord2)
print(f"✓ SCoordinate distance: {distance:.4f}")

# Test VirtualChamber
chamber = VirtualChamber()
chamber.populate(10)
print(f"✓ VirtualChamber populated: {len(list(chamber.gas))} molecules")

# Test CategoricalAperture
aperture = CategoricalAperture(center=coord1, radius=0.3)
passed = aperture.filter(list(chamber.gas))
print(f"✓ CategoricalAperture filtered: {len(passed)}/{len(list(chamber.gas))} passed")

print("\n✓ All basic tests passed!")
print("\nYou can now run: python validation_experiments.py")

