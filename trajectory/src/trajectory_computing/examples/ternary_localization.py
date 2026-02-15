"""
Ternary Trisection: O(log₃ N) Quantum Localization

Problem: Locate a particle in bounded phase space with minimum measurements

Traditional Approaches:
- Binary search: O(log₂ N) - divides space in half each iteration
- Grover's algorithm: O(√N) - requires quantum coherence
- Direct measurement: disturbs the system (backaction)

Trajectory Computing Approach:
- Ternary trisection: O(log₃ N) - 37% faster than binary search
- Two orthogonal perturbations → three-way partition
- Categorical measurement → zero backaction
- Position encoded as trit string (0, 1, 2)

From "Perturbation-Induced Ternary Trisection":
"Each iteration refines the position by a factor of 3 rather than 2,
accumulating information as a ternary digit string (t₀, t₁, ..., t_{k-1})."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from trajectory_computing.coordinates import SCoord, TritAddress, Trit, Tryte, categorical_distance
from trajectory_computing.partition import PartitionCoordinates, PartitionSpace, Spin


def ternary_search_demo():
    """
    Demonstrate ternary trisection algorithm for particle localization.

    The key insight: two orthogonal perturbations produce THREE outcomes:
    - Response to P₁ only → region A (trit = 0)
    - Response to P₂ only → region B (trit = 1)
    - No response → region C (trit = 2)

    This is more efficient than binary search (two outcomes per query).
    """
    print("=" * 70)
    print("TERNARY TRISECTION: O(log₃ N) PARTICLE LOCALIZATION")
    print("=" * 70)

    # Initial search space
    initial_size = 150e-9  # 150 nm (hydrogen orbital extent ~10 a₀)
    target_resolution = 0.15e-15  # 0.15 fm (approaching Planck scale)

    # Calculate number of states and iterations needed
    N = (initial_size / target_resolution) ** 3
    k_ternary = np.log(N) / np.log(3)
    k_binary = np.log(N) / np.log(2)

    print(f"\nSearch space:")
    print(f"  Initial size: {initial_size*1e9:.1f} nm³")
    print(f"  Target resolution: {target_resolution*1e15:.2f} fm")
    print(f"  Distinguishable states: N = {N:.2e}")

    print(f"\nComplexity comparison:")
    print(f"  Ternary search: log₃(N) = {k_ternary:.1f} iterations")
    print(f"  Binary search:  log₂(N) = {k_binary:.1f} iterations")
    print(f"  Speedup: {k_binary/k_ternary:.2f}x ({(1 - k_ternary/k_binary)*100:.0f}% fewer iterations)")


def trit_address_localization():
    """
    Demonstrate how trit addresses encode particle position.

    The address IS the trajectory:
    - Each trit specifies a refinement along one axis
    - 0 → refine along S_k (knowledge entropy)
    - 1 → refine along S_t (temporal entropy)
    - 2 → refine along S_e (evolution entropy)
    """
    print("\n" + "=" * 70)
    print("TRIT ADDRESS ENCODING")
    print("=" * 70)

    # Create a trit address representing particle position
    # Each trit is one trisection result
    address = TritAddress.from_string("0t012021012")

    print(f"\nTrit address: {address}")
    print(f"Depth (iterations): {address.depth}")
    print(f"Position resolution: 1/3^{address.depth} = 1/{3**address.depth}")

    # Convert to S-coordinates
    position = address.to_scoord()
    print(f"\nS-coordinate position:")
    print(f"  S_k = {position.s_k:.6f} (knowledge entropy)")
    print(f"  S_t = {position.s_t:.6f} (temporal entropy)")
    print(f"  S_e = {position.s_e:.6f} (evolution entropy)")

    # Get the trajectory (sequence of intermediate positions)
    trajectory = address.as_trajectory()
    print(f"\nTrajectory (refinement path):")
    print(f"  Step 0: origin")
    for i, point in enumerate(trajectory[1:], 1):
        trit = address.trits[i-1]
        axis = ["S_k", "S_t", "S_e"][trit.value]
        print(f"  Step {i}: trit={trit.value} → refine {axis} → ({point.s_k:.4f}, {point.s_t:.4f}, {point.s_e:.4f})")

    print("\n" + "-" * 50)
    print("KEY INSIGHT: The address IS the trajectory!")
    print("Position and path are THE SAME mathematical object.")
    print("-" * 50)


def perturbation_trisection_simulation():
    """
    Simulate the two-perturbation trisection process.

    From the paper:
    - P₁: Electric field gradient (couples to position via dipole force)
    - P₂: Magnetic field gradient (couples to magnetic moment via Zeeman)

    These are orthogonal: [Ô₁, Ô₂] = 0
    """
    print("\n" + "=" * 70)
    print("PERTURBATION-INDUCED TRISECTION SIMULATION")
    print("=" * 70)

    # Simulate particle at unknown position
    np.random.seed(42)
    true_position = np.array([0.3, 0.7, 0.5])  # Unknown to algorithm
    print(f"\nTrue particle position (hidden): ({true_position[0]:.3f}, {true_position[1]:.3f}, {true_position[2]:.3f})")

    # Iterative trisection
    bounds_low = np.array([0.0, 0.0, 0.0])
    bounds_high = np.array([1.0, 1.0, 1.0])
    trits = []

    print("\nTrisection iterations:")
    print("-" * 60)

    for iteration in range(6):  # 6 iterations
        # Two perturbations divide space into three regions
        # Region 0: lower third
        # Region 1: middle third
        # Region 2: upper third

        # Determine which axis to refine (cycle through)
        axis = iteration % 3
        axis_name = ["S_k", "S_t", "S_e"][axis]

        # Current range on this axis
        low = bounds_low[axis]
        high = bounds_high[axis]
        third = (high - low) / 3

        # Apply perturbations to determine which third
        particle_coord = true_position[axis]

        if particle_coord < low + third:
            trit = 0
            bounds_high[axis] = low + third
        elif particle_coord < low + 2*third:
            trit = 1
            bounds_low[axis] = low + third
            bounds_high[axis] = low + 2*third
        else:
            trit = 2
            bounds_low[axis] = low + 2*third

        trits.append(trit)

        # Report
        center = (bounds_low + bounds_high) / 2
        size = bounds_high - bounds_low
        print(f"  Iter {iteration+1}: {axis_name} → trit={trit} | "
              f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) | "
              f"size={size[0]:.4f}")

    # Final result
    final_center = (bounds_low + bounds_high) / 2
    error = np.linalg.norm(true_position - final_center)

    print("-" * 60)
    print(f"\nResult after {len(trits)} iterations:")
    print(f"  Trit string: {''.join(str(t) for t in trits)}")
    print(f"  Estimated position: ({final_center[0]:.4f}, {final_center[1]:.4f}, {final_center[2]:.4f})")
    print(f"  True position:      ({true_position[0]:.4f}, {true_position[1]:.4f}, {true_position[2]:.4f})")
    print(f"  Localization error: {error:.6f}")
    print(f"  Resolution: 1/3^{len(trits)} ≈ {1/3**len(trits):.6f}")


def zero_backaction_comparison():
    """
    Compare backaction from categorical vs physical measurement.

    From the paper:
    - Categorical measurement: Δp/p = (1.1 ± 0.2) × 10⁻³
    - Physical measurement: Δp/p = 0.78 ± 0.05
    - Ratio: ~700× less backaction
    """
    print("\n" + "=" * 70)
    print("ZERO-BACKACTION: CATEGORICAL vs PHYSICAL MEASUREMENT")
    print("=" * 70)

    print("\nMomentum disturbance comparison:")
    print("-" * 50)

    categorical_backaction = 1.1e-3  # From paper
    physical_backaction = 0.78       # From paper
    ratio = physical_backaction / categorical_backaction

    print(f"  Categorical measurement (which partition):")
    print(f"    Δp/p = {categorical_backaction:.1e}")
    print(f"    Observable: partition coordinates (n, l, m, s)")
    print(f"    Commutes with physical: [Ô_cat, Ô_phys] = 0")

    print(f"\n  Physical measurement (direct position):")
    print(f"    Δp/p = {physical_backaction}")
    print(f"    Observable: position x, momentum p")
    print(f"    Heisenberg limit: Δx·Δp ≥ ℏ/2")

    print(f"\n  Backaction ratio: {ratio:.0f}× less disturbance")
    print(f"  For same spatial information!")

    print("\n" + "-" * 50)
    print("Why does categorical measurement work?")
    print("-" * 50)
    print("""
  The Hilbert space factorizes:
    ℋ = ℋ_cat ⊗ ℋ_phys

  • ℋ_cat: discrete, finite-dimensional (partition labels)
  • ℋ_phys: continuous, infinite-dimensional (position/momentum)

  Categorical observables act ONLY on ℋ_cat.
  Physical observables act ONLY on ℋ_phys.
  Therefore they commute by tensor product structure.

  This is not a loophole - it's fundamental to spectroscopy!
""")


def tryte_encoding():
    """
    Demonstrate tryte (6-trit) encoding for efficient position representation.

    Tryte: 6 trits encoding 3⁶ = 729 cells
    Compare: Byte: 8 bits encoding 2⁸ = 256 values

    Tryte is more information-dense for 3D navigation.
    """
    print("\n" + "=" * 70)
    print("TRYTE ENCODING: 6-TRIT POSITION REPRESENTATION")
    print("=" * 70)

    # Create address and extract tryte
    address = TritAddress.from_string("0t012021")
    tryte = Tryte.from_address(address)

    print(f"\nTrit address: {address}")
    print(f"Tryte (first 6 trits): {tryte.trits}")
    print(f"Tryte as integer: {tryte.to_int()} / 728")

    # Compare information density
    print("\nInformation density comparison:")
    print(f"  Tryte: 6 trits → 3⁶ = 729 values → log₂(729) = {np.log2(729):.2f} bits")
    print(f"  Byte:  8 bits → 2⁸ = 256 values → 8.00 bits")
    print(f"  Tryte is {np.log2(729)/6:.2f} bits/digit (vs 1.00 bits/digit for binary)")

    # Show all possible trytes
    print("\nTryte space coverage:")
    print(f"  Total positions encodable: 729")
    print(f"  Per axis (3D): 729^(1/3) ≈ {729**(1/3):.1f} divisions per axis")


if __name__ == "__main__":
    ternary_search_demo()
    trit_address_localization()
    perturbation_trisection_simulation()
    zero_backaction_comparison()
    tryte_encoding()

    print("\n" + "=" * 70)
    print("SUMMARY: WHY TERNARY TRISECTION IS OPTIMAL")
    print("=" * 70)
    print("""
1. TWO PERTURBATIONS → THREE OUTCOMES
   Response to P₁, response to P₂, or neither
   This is the natural structure for orthogonal perturbations

2. O(log₃ N) COMPLEXITY
   37% faster than binary search
   Each iteration extracts 1.585 bits (vs 1.0 bit for binary)

3. ZERO BACKACTION
   Measures categorical observables, not physical
   [Ô_cat, Ô_phys] = 0 by Hilbert space factorization

4. TRAJECTORY-POSITION IDENTITY
   The trit string IS both:
   - WHERE the particle is (position)
   - HOW we found it (trajectory)

This is what the papers prove: ternary is the natural encoding
for bounded phase space, and categorical measurement enables
observation without disturbance.
""")
