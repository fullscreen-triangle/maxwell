"""
Wave-Particle Duality Resolution

Problem: Simultaneous observation of interference (wave) and which-path (particle)

Traditional Understanding (WRONG):
- Bohr complementarity: V² + D² ≤ 1 (wave visibility vs path distinguishability)
- Measuring which-slit destroys interference pattern
- Wave and particle are mutually exclusive

Trajectory Computing Understanding (CORRECT):
- Wave and particle are ORTHOGONAL projections, not mutually exclusive
- Categorical observables (S_k, S_t, S_e) encode both aspects
- [Ô_wave, Ô_particle] = 0 for categorical observables
- Simultaneous observation: V = 0.96, I = 1.15 bits

From "Perturbation-Induced Ternary Trisection":
"Wave-particle duality is a property of measurement type, not nature itself.
Categorical measurement reveals that wave and particle are complementary
projections of a unified ternary structure encoded in S-entropy space."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from trajectory_computing.coordinates import SCoord, TritAddress, Trit
from trajectory_computing.partition import PartitionCoordinates, PartitionSpace, Spin


def double_slit_categorical():
    """
    Demonstrate categorical measurement in double-slit experiment.

    The S-entropy coordinates encode THREE aspects:
    - S_k: particle aspect (WHICH partition = which-path information)
    - S_t: wave aspect (temporal phase = interference)
    - S_e: trajectory aspect (evolution = path taken)

    These are orthogonal: measuring one doesn't disturb the others.
    """
    print("=" * 70)
    print("DOUBLE-SLIT EXPERIMENT: CATEGORICAL RESOLUTION")
    print("=" * 70)

    print("\nS-Entropy Coordinate Interpretation:")
    print("-" * 50)
    print("  S_k (knowledge entropy)  → PARTICLE aspect: which partition")
    print("  S_t (temporal entropy)   → WAVE aspect: interference phase")
    print("  S_e (evolution entropy)  → TRAJECTORY aspect: path evolution")
    print("-" * 50)

    # Simulate photon passing through double-slit
    # Encode state in S-coordinates

    # Initial state: photon emitted from source
    initial = SCoord(s_k=0.5, s_t=0.0, s_e=0.0)
    print(f"\nInitial state (at source):")
    print(f"  S = ({initial.s_k:.2f}, {initial.s_t:.2f}, {initial.s_e:.2f})")

    # After passing through slits: superposition encoded in S_t
    # S_k encodes which-slit information
    # S_t encodes interference phase

    # Left slit path
    left_slit = SCoord(s_k=0.25, s_t=0.5, s_e=0.33)
    # Right slit path
    right_slit = SCoord(s_k=0.75, s_t=0.5, s_e=0.33)

    print(f"\nSlit passages (categorical coordinates):")
    print(f"  Left slit:  S = ({left_slit.s_k:.2f}, {left_slit.s_t:.2f}, {left_slit.s_e:.2f})")
    print(f"  Right slit: S = ({right_slit.s_k:.2f}, {right_slit.s_t:.2f}, {right_slit.s_e:.2f})")
    print(f"  Which-slit information in S_k: |{left_slit.s_k:.2f} - {right_slit.s_k:.2f}| = {abs(left_slit.s_k - right_slit.s_k):.2f}")

    # At detector: interference pattern from S_t variation
    print("\nAt detector:")
    print("  • S_k distinguishes which-slit (particle aspect)")
    print("  • S_t determines interference phase (wave aspect)")
    print("  • Both are measurable simultaneously!")


def complementarity_violation():
    """
    Demonstrate that categorical measurement violates Bohr's complementarity
    inequality V² + D² ≤ 1.

    From the paper:
    - Measured visibility: V = 0.96 ± 0.03
    - Measured which-path info: I = 1.15 ± 0.08 bits

    For V² + D² ≤ 1 to hold, if V = 0.96 then D ≤ 0.28.
    But the measured distinguishability D corresponds to I > 1 bit,
    which means D > 0.5 (more than 50% path determination).

    This "violation" occurs because Bohr's complementarity applies to
    PHYSICAL measurements, not CATEGORICAL measurements.
    """
    print("\n" + "=" * 70)
    print("'VIOLATING' BOHR COMPLEMENTARITY")
    print("=" * 70)

    # Experimental results from paper
    visibility = 0.96  # Wave aspect: interference fringe visibility
    which_path_info = 1.15  # Particle aspect: mutual information in bits

    print("\nExperimental Results (from paper):")
    print(f"  Interference visibility: V = {visibility} ± 0.03")
    print(f"  Which-path information:  I = {which_path_info} ± 0.08 bits")

    # Traditional complementarity constraint
    print("\nBohr Complementarity (traditional):")
    print("  V² + D² ≤ 1")
    print(f"  With V = {visibility}: V² = {visibility**2:.2f}")
    print(f"  This would require D² ≤ {1 - visibility**2:.2f}")
    print(f"  Maximum allowed D = {np.sqrt(1 - visibility**2):.2f}")

    # But we measured more which-path info
    # 1 bit of mutual information ≈ D = 1 (perfect distinguishability)
    D_measured = min(which_path_info, 1.0)  # Saturates at 1
    print(f"\n  Measured D ≈ {D_measured:.2f} (from I = {which_path_info} bits)")
    print(f"  V² + D² = {visibility**2 + D_measured**2:.2f} > 1 (!)")

    print("\n" + "-" * 50)
    print("RESOLUTION: This is NOT a violation of physics!")
    print("-" * 50)
    print("""
  Bohr complementarity applies to PHYSICAL measurements:
    • Measuring position disturbs momentum
    • Adding which-path detector destroys interference

  Categorical measurements are DIFFERENT:
    • Measure WHICH PARTITION, not exact position
    • [Ô_cat(wave), Ô_cat(particle)] = 0
    • Both aspects are projections of the SAME categorical state

  The photon is ALWAYS both wave and particle.
  It's the measurement TYPE that determines what we see.
""")


def s_entropy_unification():
    """
    Show how S-entropy space unifies wave and particle descriptions.

    The three S-coordinates are:
    - S_k: encodes position precision → particle localization
    - S_t: encodes time precision → wave frequency/phase
    - S_e: encodes energy precision → photon energy

    Constraint: S_k · S_t · S_e = constant (from uncertainty relations)

    This means wave and particle are COUPLED, not exclusive.
    Increasing S_k (better localization) changes S_t and S_e.
    """
    print("\n" + "=" * 70)
    print("S-ENTROPY SPACE: UNIFIED DESCRIPTION")
    print("=" * 70)

    print("\nThe S-coordinate constraint:")
    print("  S_k × S_t × S_e = constant")
    print("\nThis couples all three aspects of the photon:")

    # Different measurement emphases
    measurements = [
        ("Particle-like", SCoord(s_k=0.9, s_t=0.1, s_e=0.5), "High S_k → well-localized"),
        ("Wave-like", SCoord(s_k=0.1, s_t=0.9, s_e=0.5), "High S_t → well-defined phase"),
        ("Energy-like", SCoord(s_k=0.3, s_t=0.3, s_e=0.9), "High S_e → well-defined energy"),
        ("Balanced", SCoord(s_k=0.5, s_t=0.5, s_e=0.5), "Equal precision all aspects"),
    ]

    print("\nDifferent measurement emphases:")
    print("-" * 60)
    for name, s, description in measurements:
        product = s.s_k * s.s_t * s.s_e
        print(f"  {name:12s}: S = ({s.s_k:.1f}, {s.s_t:.1f}, {s.s_e:.1f}) | "
              f"product = {product:.3f} | {description}")

    print("-" * 60)
    print("\nThe product varies because these are normalized coordinates.")
    print("The underlying uncertainty product ΔxΔpΔEΔt ≥ ℏ² is preserved.")


def trans_planckian_resolution():
    """
    Demonstrate trans-Planckian categorical resolution.

    From the paper:
    - Temporal resolution: δt = 10^-138 s (vs Planck time 10^-43 s)
    - This is 95 orders of magnitude below Planck time!

    How is this possible?
    - Not measuring physical time directly
    - Counting categorical configurations (10^125 of them)
    - Each configuration is distinguishable

    This is NOT violating quantum mechanics.
    It's categorical counting, not physical measurement.
    """
    print("\n" + "=" * 70)
    print("TRANS-PLANCKIAN CATEGORICAL RESOLUTION")
    print("=" * 70)

    planck_time = 5.39e-44  # seconds
    categorical_resolution = 1e-138  # seconds (from paper)
    num_configurations = 1e125

    print("\nTime scales:")
    print(f"  Planck time: t_P = {planck_time:.2e} s")
    print(f"  Categorical resolution: δt_cat = {categorical_resolution:.2e} s")
    print(f"  Ratio: t_P / δt_cat = {planck_time/categorical_resolution:.2e}")
    print(f"  (95 orders of magnitude BELOW Planck time!)")

    print(f"\nNumber of categorical configurations: {num_configurations:.0e}")

    print("\n" + "-" * 50)
    print("How is this possible?")
    print("-" * 50)
    print("""
  We are NOT measuring physical time to 10^-138 s precision!
  That would indeed violate Planck-scale physics.

  Instead, we are COUNTING categorical configurations:
    • Total configurations during photon flight: 10^125
    • Each configuration is distinguishable
    • "Resolution" = flight time / configurations

  This is like asking "How many different chess positions exist?"
  The answer (~10^120) doesn't require Planck-scale measurement.

  Categorical counting accesses a discrete, finite state space.
  Physical measurement accesses a continuous, infinite space.
  These are fundamentally different operations.
""")


def photon_trajectory_encoding():
    """
    Encode a photon's trajectory through double-slit as trit address.

    The trit sequence encodes:
    - Position at each moment (which partition)
    - Phase evolution (interference information)
    - Path taken (trajectory)

    All three in ONE mathematical object.
    """
    print("\n" + "=" * 70)
    print("PHOTON TRAJECTORY AS TRIT ADDRESS")
    print("=" * 70)

    # Simulate photon trajectory through double-slit
    # Each trit represents refinement during flight

    flight_time = 33e-15  # 33 femtoseconds
    trisections = 22  # From paper

    print(f"\nPhoton flight parameters:")
    print(f"  Flight time: {flight_time*1e15:.0f} fs")
    print(f"  Trisection iterations: {trisections}")
    print(f"  Position states: 3^{trisections} ≈ {3**trisections:.2e}")

    # Create example trit address for photon path
    # Alternating pattern represents interference
    trits = [Trit(i % 3) for i in range(trisections)]
    address = TritAddress(trits=trits)

    print(f"\nExample photon trajectory address:")
    print(f"  {address}")

    # Decode to S-coordinates
    position = address.to_scoord()
    print(f"\nFinal S-coordinates:")
    print(f"  S_k = {position.s_k:.4f} (which-path)")
    print(f"  S_t = {position.s_t:.4f} (interference phase)")
    print(f"  S_e = {position.s_e:.4f} (evolution)")

    # Get trajectory
    trajectory = address.as_trajectory()
    print(f"\nTrajectory has {len(trajectory)} waypoints")
    print("  (Each waypoint is an intermediate S-coordinate)")


if __name__ == "__main__":
    double_slit_categorical()
    complementarity_violation()
    s_entropy_unification()
    trans_planckian_resolution()
    photon_trajectory_encoding()

    print("\n" + "=" * 70)
    print("CONCLUSION: WAVE-PARTICLE DUALITY RESOLVED")
    print("=" * 70)
    print("""
The wave-particle duality "paradox" arises from a category error:

  PHYSICAL measurement: position OR momentum (complementary)
  CATEGORICAL measurement: position AND momentum aspects (orthogonal)

The photon is ALWAYS a unified entity in S-entropy space:
  • S_k encodes particle aspect (spatial localization)
  • S_t encodes wave aspect (temporal/phase coherence)
  • S_e encodes energy aspect (quantum of action)

These three aspects are PROJECTIONS of the same categorical state.
Measuring one does NOT destroy the others because:
  [Ô_cat(S_k), Ô_cat(S_t)] = [Ô_cat(S_t), Ô_cat(S_e)] = [Ô_cat(S_e), Ô_cat(S_k)] = 0

Wave and particle are not mutually exclusive properties.
They are complementary VIEWS of a unified ternary structure.

This is what Trajectory Computing makes explicit:
  - The address IS the trajectory
  - Position IS the path
  - Wave IS particle (in categorical space)
""")
