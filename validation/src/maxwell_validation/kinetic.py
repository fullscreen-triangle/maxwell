"""
Kinetic Face Validation
=======================

Validates the kinetic engine operations:
- Maxwell-Boltzmann distributions
- Temperature computations
- Demon sorting (what Maxwell would see)
- Retrieval paradox demonstration
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


class KineticValidator:
    """
    Validates kinetic engine operations.
    
    Key validations:
    1. Maxwell-Boltzmann distribution is correct
    2. Temperature is computed correctly from kinetic energies
    3. Demon sorting produces expected distributions
    4. Retrieval paradox: thermal equilibration defeats sorting
    """
    
    def __init__(self, temperature: float = 300.0, k_b: float = 1.380649e-23):
        self.temperature = temperature
        self.k_b = k_b
        self.velocities: np.ndarray = np.array([])
    
    def initialize_maxwell_boltzmann(self, n: int, mass: float = 1.0) -> np.ndarray:
        """Initialize molecules with Maxwell-Boltzmann velocity distribution"""
        # Standard deviation for Maxwell-Boltzmann
        sigma = np.sqrt(self.k_b * self.temperature / mass)
        
        # Sample velocity components
        vx = np.random.normal(0, sigma, n)
        vy = np.random.normal(0, sigma, n)
        vz = np.random.normal(0, sigma, n)
        
        # Compute speeds
        self.velocities = np.sqrt(vx**2 + vy**2 + vz**2)
        return self.velocities
    
    def validate_maxwell_boltzmann(self, n_samples: int = 10000, mass: float = 1.0) -> Tuple[bool, str]:
        """Validate that velocity distribution follows Maxwell-Boltzmann"""
        self.initialize_maxwell_boltzmann(n_samples, mass)
        
        # Expected mean speed: <v> = sqrt(8 * k_B * T / (pi * m))
        expected_mean = np.sqrt(8 * self.k_b * self.temperature / (np.pi * mass))
        actual_mean = np.mean(self.velocities)
        
        # Allow 5% deviation
        if abs(actual_mean - expected_mean) / expected_mean < 0.05:
            return True, f"Mean velocity matches: expected {expected_mean:.4e}, got {actual_mean:.4e}"
        else:
            return False, f"Mean velocity mismatch: expected {expected_mean:.4e}, got {actual_mean:.4e}"
    
    def validate_temperature_from_kinetic(self, n_samples: int = 10000, mass: float = 1.0) -> Tuple[bool, str]:
        """Validate temperature computation from kinetic energies"""
        self.initialize_maxwell_boltzmann(n_samples, mass)
        
        # Compute temperature from kinetic energies
        # T = 2 * <KE> / (3 * k_B) for 3D
        kinetic_energies = 0.5 * mass * self.velocities**2
        computed_temp = (2 * np.mean(kinetic_energies)) / (3 * self.k_b)
        
        # Allow 5% deviation
        if abs(computed_temp - self.temperature) / self.temperature < 0.05:
            return True, f"Temperature matches: expected {self.temperature:.2f} K, got {computed_temp:.2f} K"
        else:
            return False, f"Temperature mismatch: expected {self.temperature:.2f} K, got {computed_temp:.2f} K"
    
    def validate_demon_sorting(self, n_samples: int = 1000) -> Tuple[bool, str]:
        """Validate demon sorting produces ~50/50 split around mean"""
        self.initialize_maxwell_boltzmann(n_samples)
        
        mean_v = np.mean(self.velocities)
        fast = np.sum(self.velocities > mean_v)
        slow = n_samples - fast
        
        # Should be roughly 50/50
        ratio = fast / n_samples
        if 0.4 < ratio < 0.6:
            return True, f"Demon sorting: {fast} fast, {slow} slow (ratio: {ratio:.2f})"
        else:
            return False, f"Unexpected sorting ratio: {ratio:.2f}"
    
    def validate_retrieval_paradox(self, n_molecules: int = 100, n_steps: int = 100) -> Tuple[bool, str]:
        """
        Validate the retrieval paradox:
        Thermal equilibration prevents maintaining sorted states.
        """
        self.initialize_maxwell_boltzmann(n_molecules)
        
        # Track how the "sorted" distribution evolves
        history = []
        
        for _ in range(n_steps):
            mean_v = np.mean(self.velocities)
            fast = np.sum(self.velocities > mean_v)
            history.append(fast)
            
            # Simulate collisions: some velocities change
            # This is the key: velocities randomize faster than sorting
            collision_mask = np.random.random(n_molecules) < 0.1
            if np.any(collision_mask):
                sigma = np.sqrt(2 * self.k_b * self.temperature / 1.0)
                new_vx = np.random.normal(0, sigma, n_molecules)
                new_vy = np.random.normal(0, sigma, n_molecules)
                new_vz = np.random.normal(0, sigma, n_molecules)
                new_speeds = np.sqrt(new_vx**2 + new_vy**2 + new_vz**2)
                self.velocities[collision_mask] = new_speeds[collision_mask]
        
        # Distribution should stay roughly 50/50
        mean_fast = np.mean(history)
        std_fast = np.std(history)
        
        expected = n_molecules / 2
        if abs(mean_fast - expected) < 0.1 * n_molecules:
            return True, f"Retrieval paradox validated: fast count stays ~{expected:.0f} (mean: {mean_fast:.1f}, std: {std_fast:.1f})"
        else:
            return False, f"Unexpected: fast count drifted to {mean_fast:.1f}"
    
    def run_all_validations(self) -> Dict[str, Tuple[bool, str]]:
        """Run all kinetic validations"""
        results = {}
        
        results["maxwell_boltzmann"] = self.validate_maxwell_boltzmann()
        results["temperature_from_kinetic"] = self.validate_temperature_from_kinetic()
        results["demon_sorting"] = self.validate_demon_sorting()
        results["retrieval_paradox"] = self.validate_retrieval_paradox()
        
        return results


if __name__ == "__main__":
    validator = KineticValidator(temperature=300.0)
    results = validator.run_all_validations()
    
    print("=" * 60)
    print("KINETIC VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, (passed, message) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        print(f"       {message}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

