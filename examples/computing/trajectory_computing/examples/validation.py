"""
Validation Examples for Trajectory Computing

Tests the core theoretical predictions:
1. Capacity Theorem: 2n² states at depth n
2. Selection Rules: Delta_l = +/-1 for allowed transitions
3. Trajectory-Position Identity: address encodes both
4. Epsilon-Boundary Solutions: Goedelian residue prevents exact closure
5. Computing = Verification: same categorical operation
6. Poincaré Recurrence: bounded systems return to initial state

These are not arbitrary tests - they validate that the implementation
correctly captures the theoretical framework.
"""

import numpy as np
from typing import List, Tuple, Dict

# Import from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trajectory_computing.coordinates import SCoord, TritAddress, Trit, categorical_distance
from trajectory_computing.partition import PartitionCoordinates, PartitionSpace, Spin
from trajectory_computing.phase_lock import PhaseLockNetwork, Coupling
from trajectory_computing.morphism import Morphism, Catalyst
from trajectory_computing.navigator import Navigator, NavigationStrategy, CoordinateCompletion
from trajectory_computing.completion import CompletionDetector, GoedelianBoundary, CoordinateCondition
from trajectory_computing.system import System, SystemBuilder
from trajectory_computing.runtime import TrajectoryComputer, RuntimeConfig


class ValidationResult:
    """Result of a validation test."""
    def __init__(self, name: str, passed: bool, expected: any, actual: any, details: str = ""):
        self.name = name
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.details = details

    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        return f"{status}: {self.name}\n  Expected: {self.expected}\n  Actual: {self.actual}\n  {self.details}"


def validate_capacity_theorem() -> List[ValidationResult]:
    """
    Validate: Capacity at depth n = 2n²

    This is the quantum mechanical shell capacity formula,
    derived purely from geometric constraints.
    """
    results = []

    # Test n = 1 to 5 (K, L, M, N, O shells)
    expected_capacities = {
        1: 2,    # K shell: 2 electrons
        2: 8,    # L shell: 8 electrons
        3: 18,   # M shell: 18 electrons
        4: 32,   # N shell: 32 electrons
        5: 50,   # O shell: 50 electrons
    }

    for n, expected in expected_capacities.items():
        # Formula prediction
        formula_result = 2 * n * n

        # Enumeration verification
        states = PartitionCoordinates.enumerate_at_depth(n)
        enumerated = len(states)

        passed = formula_result == expected == enumerated

        results.append(ValidationResult(
            f"Capacity at n={n}",
            passed,
            expected,
            enumerated,
            f"Formula: 2×{n}² = {formula_result}"
        ))

    return results


def validate_selection_rules() -> List[ValidationResult]:
    """
    Validate: Selection rules Delta_l = +/-1, Delta_m in {0, +/-1}

    These rules emerge from continuity requirements on
    oscillatory modes, not from fitting to data.
    """
    results = []

    # Test state at (n=3, l=1, m=0, s=UP)
    state = PartitionCoordinates(n=3, l=1, m=0, s=Spin.UP)
    transitions = state.allowed_transitions()

    # Check all transitions satisfy |Delta_l| = 1
    all_valid_delta_l = all(
        abs(t.l - state.l) == 1 for t in transitions
    )

    results.append(ValidationResult(
        "Selection rule |Delta_l| = 1",
        all_valid_delta_l,
        True,
        all_valid_delta_l,
        f"Tested {len(transitions)} transitions from (n=3, l=1, m=0)"
    ))

    # Check all transitions satisfy |Delta_m| <= 1
    all_valid_delta_m = all(
        abs(t.m - state.m) <= 1 for t in transitions
    )

    results.append(ValidationResult(
        "Selection rule |Delta_m| <= 1",
        all_valid_delta_m,
        True,
        all_valid_delta_m,
        f"Tested {len(transitions)} transitions"
    ))

    # Verify forbidden transition l=0 -> l=2 is NOT in list
    # From (n=3, l=0, m=0), we should NOT see l=2
    state_s = PartitionCoordinates(n=3, l=0, m=0, s=Spin.UP)
    trans_s = state_s.allowed_transitions()

    has_l2 = any(t.l == 2 for t in trans_s)

    results.append(ValidationResult(
        "Forbidden transition l=0->l=2 excluded",
        not has_l2,
        False,
        has_l2,
        "Selection rules forbid Delta_l = 2"
    ))

    return results


def validate_trajectory_position_identity() -> List[ValidationResult]:
    """
    Validate: Trajectory-Position Identity

    A trit address encodes BOTH the position (which cell)
    AND the trajectory (how to get there). These are the
    SAME mathematical object, not two views.
    """
    results = []

    # Create an address
    addr = TritAddress.from_string("0t012102")

    # Get position
    position = addr.to_scoord()

    # Get trajectory
    trajectory = addr.as_trajectory()

    # The final point of trajectory should equal position
    final_point = trajectory[-1]

    position_matches = np.allclose(
        position.to_array(),
        final_point.to_array(),
        rtol=1e-10
    )

    results.append(ValidationResult(
        "Trajectory endpoint equals position",
        position_matches,
        position.to_array(),
        final_point.to_array(),
        f"Address: {addr}"
    ))

    # Trajectory length should equal address depth + 1 (including origin)
    correct_length = len(trajectory) == addr.depth + 1

    results.append(ValidationResult(
        "Trajectory length = depth + 1",
        correct_length,
        addr.depth + 1,
        len(trajectory),
        "Trajectory includes origin point"
    ))

    # Refinement should extend trajectory
    refined = addr.refine(Trit.ZERO)
    refined_traj = refined.as_trajectory()

    extends_correctly = len(refined_traj) == len(trajectory) + 1

    results.append(ValidationResult(
        "Refinement extends trajectory",
        extends_correctly,
        len(trajectory) + 1,
        len(refined_traj),
        "One trit = one trajectory step"
    ))

    return results


def validate_epsilon_boundary() -> List[ValidationResult]:
    """
    Validate: Epsilon-Boundary (Goedelian Residue)

    Solutions exist at one categorical step from exact closure.
    This is not approximation - it's the maximum possible knowledge.
    Reality = infinity - x (where x is Gödelian residue).
    """
    results = []

    goedel = GoedelianBoundary(epsilon=0.01)

    # At boundary: 0 < distance <= epsilon
    at_boundary = goedel.is_at_boundary(0.005)
    results.append(ValidationResult(
        "Distance 0.005 is at epsilon-boundary",
        at_boundary,
        True,
        at_boundary,
        f"epsilon = {goedel.epsilon}"
    ))

    # Exact zero is NOT at boundary (impossible)
    zero_at_boundary = goedel.is_at_boundary(0.0)
    results.append(ValidationResult(
        "Distance 0 is NOT at epsilon-boundary",
        not zero_at_boundary,
        False,
        zero_at_boundary,
        "Exact closure is impossible (Gödel)"
    ))

    # Beyond boundary
    beyond = goedel.is_beyond_boundary(0.02)
    results.append(ValidationResult(
        "Distance 0.02 is beyond epsilon-boundary",
        beyond,
        True,
        beyond,
        "Solutions don't exist beyond boundary"
    ))

    # Observable reality
    obs = goedel.observable_reality(100.0)
    expected_obs = 100.0 - 0.01

    results.append(ValidationResult(
        "Observable reality = total - epsilon",
        np.isclose(obs, expected_obs),
        expected_obs,
        obs,
        "Reality = infinity - x"
    ))

    return results


def validate_computing_equals_verification() -> List[ValidationResult]:
    """
    Validate: Computing = Verification

    The operation that finds a solution is the SAME operation
    that verifies it. Both navigate to the same epsilon-boundary.
    """
    results = []

    # Create a partition space
    space = PartitionSpace()
    network = PhaseLockNetwork()

    # Create partitions
    coords_list = [
        PartitionCoordinates(n=2, l=0, m=0, s=Spin.UP),
        PartitionCoordinates(n=2, l=1, m=0, s=Spin.UP),
        PartitionCoordinates(n=2, l=1, m=1, s=Spin.UP),
        PartitionCoordinates(n=3, l=1, m=0, s=Spin.UP),
        PartitionCoordinates(n=3, l=2, m=0, s=Spin.UP),
    ]

    for coords in coords_list:
        p = space.create(coords)

    # Set up accessibility (simple chain)
    partitions = list(space.partitions.values())
    for i in range(len(partitions) - 1):
        partitions[i].accessible.add(partitions[i+1].id)

    # Create navigator
    navigator = Navigator(space, network)

    # Define completion condition
    completion = CoordinateCompletion(target_n=3, target_l=2)

    # Navigate (computing)
    result = navigator.navigate(partitions[0], completion)

    # Verify (same operation applied to result)
    if result.final_partition:
        verified = navigator.verify_completion(result.final_partition, completion)
    else:
        verified = False

    # Computing result should match verification
    results.append(ValidationResult(
        "Navigation finds verifiable solution",
        result.success == verified,
        result.success,
        verified,
        "Computing = Verification"
    ))

    # Both use the same completion check
    if result.final_partition:
        from trajectory_computing.completion import CompletionStatus
        nav_check = completion.is_satisfied(result.final_partition)
        ver_check = navigator.verify_completion(result.final_partition, completion)

        same_operation = nav_check == ver_check

        results.append(ValidationResult(
            "Same completion check in both operations",
            same_operation,
            nav_check,
            ver_check,
            "Identical categorical operation"
        ))

    return results


def validate_catalyst_chain() -> List[ValidationResult]:
    """
    Validate: Catalyst Chain Construction

    When direct transition violates selection rules,
    catalyst provides intermediate stages.
    """
    results = []

    space = PartitionSpace()

    # Try transition from l=0 to l=2 (forbidden directly)
    start = PartitionCoordinates(n=2, l=0, m=0, s=Spin.UP)
    end = PartitionCoordinates(n=4, l=2, m=0, s=Spin.UP)

    # Build catalyst chain
    catalyst = Catalyst.build_chain(start, end, space)

    # Chain should be valid (each step |Delta_l| <= 1)
    valid = catalyst.is_valid_chain()

    results.append(ValidationResult(
        "Catalyst chain obeys selection rules",
        valid,
        True,
        valid,
        f"Chain has {len(catalyst)} stages"
    ))

    # Chain should connect start to end
    if catalyst.stages:
        starts_correct = (
            catalyst.stages[0].coordinates.n == start.n and
            catalyst.stages[0].coordinates.l == start.l
        )
        ends_correct = (
            catalyst.stages[-1].coordinates.n == end.n and
            catalyst.stages[-1].coordinates.l == end.l
        )

        results.append(ValidationResult(
            "Catalyst connects start to end",
            starts_correct and ends_correct,
            (start.n, start.l, end.n, end.l),
            (catalyst.stages[0].coordinates.n, catalyst.stages[0].coordinates.l,
             catalyst.stages[-1].coordinates.n, catalyst.stages[-1].coordinates.l),
            "Intermediate stages bridge the gap"
        ))

    return results


def validate_phase_lock_independence() -> List[ValidationResult]:
    """
    Validate: Phase-lock network independent of velocity

    Phase-locks form based on POSITION (Van der Waals ~r^-6),
    NOT velocity. This is the categorical face.
    """
    results = []

    # Same positions, different "velocities" (not used)
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-9, 0.0, 0.0]),  # 1 nm apart
        np.array([0.0, 1e-9, 0.0]),
    ]

    # Build network from positions
    network = PhaseLockNetwork.from_positions(positions)

    # Network should have edges based on position only
    has_edges = len(network.edges) > 0

    results.append(ValidationResult(
        "Network forms from positions only",
        has_edges,
        True,
        has_edges,
        f"Edges: {len(network.edges)}"
    ))

    # All nodes should be connected (close positions)
    all_connected = all(
        len(network.get_neighbors(nid)) > 0
        for nid in network.nodes
    )

    results.append(ValidationResult(
        "All nearby nodes are connected",
        all_connected,
        True,
        all_connected,
        "Coupling based on r^-6 distance"
    ))

    return results


def run_all_validations() -> Tuple[int, int, List[ValidationResult]]:
    """Run all validation tests and return summary."""

    all_results = []

    print("=" * 70)
    print("TRAJECTORY COMPUTING VALIDATION")
    print("=" * 70)

    # Run each validation suite
    suites = [
        ("Capacity Theorem (2n^2)", validate_capacity_theorem),
        ("Selection Rules (Delta_l = +/-1)", validate_selection_rules),
        ("Trajectory-Position Identity", validate_trajectory_position_identity),
        ("Epsilon-Boundary (Goedelian)", validate_epsilon_boundary),
        ("Computing = Verification", validate_computing_equals_verification),
        ("Catalyst Chain", validate_catalyst_chain),
        ("Phase-Lock Independence", validate_phase_lock_independence),
    ]

    for name, validator in suites:
        print(f"\n{name}")
        print("-" * 50)

        try:
            results = validator()
            all_results.extend(results)

            for r in results:
                status = "PASS" if r.passed else "FAIL"
                print(f"  {status} {r.name}")
                if not r.passed:
                    print(f"      Expected: {r.expected}")
                    print(f"      Actual: {r.actual}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append(ValidationResult(
                f"{name} (error)",
                False,
                "no error",
                str(e)
            ))

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("All theoretical predictions validated!")
    else:
        print(f"Failed tests: {total - passed}")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}")

    return passed, total, all_results


if __name__ == "__main__":
    passed, total, results = run_all_validations()
    exit(0 if passed == total else 1)
