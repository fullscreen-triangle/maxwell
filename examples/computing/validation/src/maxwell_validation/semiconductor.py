"""
Biological Semiconductor Module
===============================

Implements the biological oscillatory semiconductor framework:
- Oscillatory holes as P-type carriers
- Molecular carriers as N-type
- P-N junction formation and rectification
- Therapeutic conductivity

Based on "Biological Oscillatory Semiconductors: Quantum Field Therapeutics"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from .types import (
    OscillatoryHole, MolecularCarrier, OscillatorySignature,
    PNJunction, CarrierType, ProcessorConfig
)


@dataclass
class SemiconductorSubstrate:
    """
    A biological semiconductor substrate supporting therapeutic current flow.
    
    Implements the fundamental equations from the semiconductor theory:
    - Hole mobility: μ_h = q_h τ_h / m_h*
    - Drift velocity: v_d = μ_h E_therapeutic
    - Conductivity: σ = n μ_n e + p μ_p e
    """
    holes: List[OscillatoryHole] = field(default_factory=list)
    carriers: List[MolecularCarrier] = field(default_factory=list)
    temperature: float = 300.0  # K
    config: ProcessorConfig = field(default_factory=ProcessorConfig)
    
    # Physical constants
    K_B: float = 1.380649e-23  # Boltzmann constant
    E_CHARGE: float = 1.602e-19  # Elementary charge
    
    def add_hole(self, missing_signature: OscillatorySignature, 
                 position: np.ndarray = None) -> OscillatoryHole:
        """Create and add an oscillatory hole"""
        hole = OscillatoryHole(
            id=len(self.holes),
            missing_signature=missing_signature,
            position=position if position is not None else np.random.rand(3) * 1e-6
        )
        self.holes.append(hole)
        return hole
    
    def add_carrier(self, signature: OscillatorySignature,
                   molecular_mass: float, concentration: float,
                   position: np.ndarray = None) -> MolecularCarrier:
        """Create and add a molecular carrier"""
        carrier = MolecularCarrier(
            id=len(self.carriers),
            signature=signature,
            molecular_mass=molecular_mass,
            concentration=concentration,
            position=position if position is not None else np.random.rand(3) * 1e-6
        )
        self.carriers.append(carrier)
        return carrier
    
    @property
    def hole_concentration(self) -> float:
        """Total hole concentration (cm^-3)"""
        if not self.holes:
            return 0.0
        return sum(h.concentration for h in self.holes) / len(self.holes)
    
    @property
    def carrier_concentration(self) -> float:
        """Total carrier concentration (cm^-3)"""
        if not self.carriers:
            return 0.0
        # Convert molar to cm^-3
        return sum(c.concentration * 6.022e23 / 1e3 for c in self.carriers)
    
    @property
    def is_p_type(self) -> bool:
        """Check if substrate is P-type (hole dominated)"""
        return self.hole_concentration > self.carrier_concentration
    
    @property
    def is_n_type(self) -> bool:
        """Check if substrate is N-type (carrier dominated)"""
        return self.carrier_concentration > self.hole_concentration
    
    def therapeutic_conductivity(self) -> float:
        """
        Compute total therapeutic conductivity.
        
        σ = n μ_n e + p μ_p e
        """
        p = self.hole_concentration
        n = self.carrier_concentration
        mu_p = self.config.hole_mobility
        mu_n = self.config.carrier_mobility
        e = self.E_CHARGE
        
        return n * mu_n * e + p * mu_p * e
    
    def hole_current_density(self, therapeutic_field: float) -> float:
        """
        Compute hole current density.
        
        J_h = q_h p_h μ_h E - q_h D_h ∇p_h
        
        (Simplified: drift only, no diffusion gradient)
        """
        p = self.hole_concentration
        mu_h = self.config.hole_mobility
        q_h = 1.0  # Normalized therapeutic charge
        
        return q_h * p * mu_h * therapeutic_field
    
    def find_recombination_pairs(self, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """
        Find carrier-hole pairs that can recombine.
        
        Recombination occurs when a carrier's signature can fill a hole.
        """
        pairs = []
        for hole in self.holes:
            for carrier in self.carriers:
                if carrier.can_fill(hole, threshold):
                    pairs.append((carrier.id, hole.id))
        return pairs
    
    def recombination_step(self) -> int:
        """
        Perform one recombination step.
        
        Carriers fill holes where signatures match.
        Returns number of recombinations.
        """
        pairs = self.find_recombination_pairs()
        recombined = 0
        
        filled_holes = set()
        used_carriers = set()
        
        for carrier_id, hole_id in pairs:
            if hole_id not in filled_holes and carrier_id not in used_carriers:
                filled_holes.add(hole_id)
                used_carriers.add(carrier_id)
                recombined += 1
        
        # Remove filled holes and used carriers
        self.holes = [h for h in self.holes if h.id not in filled_holes]
        self.carriers = [c for c in self.carriers if c.id not in used_carriers]
        
        return recombined
    
    def thermal_generation(self, energy_gap: float = 0.1) -> int:
        """
        Generate new holes through thermal excitation.
        
        G_thermal = A T^(3/2) exp(-E_gap / k_B T)
        """
        a = 1e10  # Generation coefficient
        rate = a * (self.temperature ** 1.5) * np.exp(-energy_gap / (self.K_B * self.temperature))
        
        # Probabilistic generation
        n_generated = np.random.poisson(rate * 1e-20)  # Scale factor
        
        for _ in range(n_generated):
            # Generate random missing signature
            sig = OscillatorySignature(
                amplitude=np.random.uniform(0.5, 1.5),
                frequency=np.random.uniform(1e12, 1e13),
                phase=np.random.uniform(0, 2 * np.pi)
            )
            self.add_hole(sig)
        
        return n_generated


@dataclass
class BiologicalPNJunction:
    """
    A biological P-N junction for therapeutic rectification.
    
    Enables directional therapeutic flow and signal processing.
    """
    p_substrate: SemiconductorSubstrate
    n_substrate: SemiconductorSubstrate
    junction_position: float = 0.0  # x = 0 is the junction
    
    @property
    def built_in_potential(self) -> float:
        """
        Compute built-in therapeutic potential V_bi.
        
        V_bi = (k_B T / q) ln(N_A N_D / n_i²)
        """
        k_b = 1.380649e-23
        e = 1.602e-19
        T = (self.p_substrate.temperature + self.n_substrate.temperature) / 2
        
        N_A = max(self.p_substrate.hole_concentration, 1.0)
        N_D = max(self.n_substrate.carrier_concentration, 1.0)
        n_i = np.sqrt(N_A * N_D)
        
        return (k_b * T / e) * np.log(N_A * N_D / n_i**2)
    
    @property
    def depletion_width(self) -> float:
        """
        Compute depletion region width W.
        
        W = sqrt(2ε/q * (N_A + N_D)/(N_A N_D) * V_bi)
        """
        epsilon = 8.85e-12 * 80  # Biological permittivity (water-like)
        e = 1.602e-19
        
        N_A = max(self.p_substrate.hole_concentration, 1.0)
        N_D = max(self.n_substrate.carrier_concentration, 1.0)
        V_bi = self.built_in_potential
        
        return np.sqrt(2 * epsilon / e * ((N_A + N_D) / (N_A * N_D)) * abs(V_bi))
    
    def current(self, applied_voltage: float) -> float:
        """
        Compute therapeutic current (diode equation).
        
        I = I_0 [exp(qV/k_B T) - 1]
        """
        k_b = 1.380649e-23
        e = 1.602e-19
        T = (self.p_substrate.temperature + self.n_substrate.temperature) / 2
        
        I_0 = 1e-12  # Reverse saturation current (A)
        
        return I_0 * (np.exp(e * applied_voltage / (k_b * T)) - 1)
    
    def rectification_ratio(self, voltage: float = 0.1) -> float:
        """Compute rectification ratio at given voltage"""
        I_forward = self.current(voltage)
        I_reverse = abs(self.current(-voltage))
        return I_forward / max(I_reverse, 1e-20)
    
    def is_forward_biased(self, applied_voltage: float) -> bool:
        """Check if junction is forward biased"""
        return applied_voltage > 0
    
    def therapeutic_flow_direction(self, applied_voltage: float) -> str:
        """Determine therapeutic flow direction"""
        if self.is_forward_biased(applied_voltage):
            return "P → N (therapeutic enhancement)"
        else:
            return "Blocked (therapeutic blocking)"


class SemiconductorNetwork:
    """
    A network of biological semiconductor substrates and junctions.
    
    Models the complete therapeutic circuit.
    """
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self.substrates: Dict[str, SemiconductorSubstrate] = {}
        self.junctions: List[BiologicalPNJunction] = []
    
    def add_substrate(self, name: str, substrate: SemiconductorSubstrate):
        """Add a substrate to the network"""
        self.substrates[name] = substrate
    
    def create_p_substrate(self, name: str, n_holes: int = 5) -> SemiconductorSubstrate:
        """Create a P-type substrate with holes"""
        substrate = SemiconductorSubstrate(config=self.config)
        
        for _ in range(n_holes):
            sig = OscillatorySignature(
                amplitude=np.random.uniform(0.8, 1.2),
                frequency=np.random.uniform(1e12, 1e13),
                phase=np.random.uniform(0, 2 * np.pi)
            )
            substrate.add_hole(sig)
        
        self.substrates[name] = substrate
        return substrate
    
    def create_n_substrate(self, name: str, n_carriers: int = 3,
                          molecular_mass: float = 300.0,
                          concentration: float = 1e-6) -> SemiconductorSubstrate:
        """Create an N-type substrate with carriers"""
        substrate = SemiconductorSubstrate(config=self.config)
        
        for _ in range(n_carriers):
            sig = OscillatorySignature(
                amplitude=np.random.uniform(0.8, 1.2),
                frequency=np.random.uniform(1e12, 1e13),
                phase=np.random.uniform(0, 2 * np.pi)
            )
            substrate.add_carrier(sig, molecular_mass, concentration)
        
        self.substrates[name] = substrate
        return substrate
    
    def create_junction(self, p_name: str, n_name: str) -> BiologicalPNJunction:
        """Create a P-N junction between two substrates"""
        junction = BiologicalPNJunction(
            p_substrate=self.substrates[p_name],
            n_substrate=self.substrates[n_name]
        )
        self.junctions.append(junction)
        return junction
    
    def total_conductivity(self) -> float:
        """Compute total network conductivity"""
        return sum(s.therapeutic_conductivity() for s in self.substrates.values())
    
    def simulate_step(self) -> Dict[str, int]:
        """
        Simulate one step of semiconductor dynamics.
        
        Returns counts of events.
        """
        events = {
            "recombinations": 0,
            "thermal_generations": 0
        }
        
        for substrate in self.substrates.values():
            events["recombinations"] += substrate.recombination_step()
            events["thermal_generations"] += substrate.thermal_generation()
        
        return events
    
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        total_holes = sum(len(s.holes) for s in self.substrates.values())
        total_carriers = sum(len(s.carriers) for s in self.substrates.values())
        
        return {
            "n_substrates": len(self.substrates),
            "n_junctions": len(self.junctions),
            "total_holes": total_holes,
            "total_carriers": total_carriers,
            "total_conductivity": self.total_conductivity(),
            "is_p_dominated": total_holes > total_carriers,
            "rectification_ratios": [j.rectification_ratio() for j in self.junctions]
        }


def validate_semiconductor_model() -> Dict:
    """
    Validate the semiconductor model against theoretical predictions.
    """
    results = {}
    
    # Test 1: P-N Junction formation
    network = SemiconductorNetwork()
    p_sub = network.create_p_substrate("p_region", n_holes=5)
    n_sub = network.create_n_substrate("n_region", n_carriers=3)
    junction = network.create_junction("p_region", "n_region")
    
    results["junction_built_in_potential"] = junction.built_in_potential
    results["junction_depletion_width"] = junction.depletion_width
    results["rectification_ratio"] = junction.rectification_ratio(0.1)
    
    # Test 2: Conductivity measurements
    results["p_conductivity"] = p_sub.therapeutic_conductivity()
    results["n_conductivity"] = n_sub.therapeutic_conductivity()
    results["p_is_p_type"] = p_sub.is_p_type
    results["n_is_n_type"] = n_sub.is_n_type
    
    # Test 3: Rectification behavior
    voltages = np.linspace(-0.1, 0.1, 21)
    currents = [junction.current(v) for v in voltages]
    results["iv_curve"] = list(zip(voltages.tolist(), currents))
    
    # Test 4: Hole mobility
    hole = OscillatoryHole(
        id=0,
        missing_signature=OscillatorySignature(1.0, 1e12, 0.0),
        mobility=0.0123
    )
    results["hole_drift_velocity"] = hole.drift_velocity(1e6)  # 1 V/cm field
    results["hole_diffusion_coeff"] = hole.diffusion_coefficient(300.0)
    
    # Validate against expected values
    results["validations"] = {
        "rectification_valid": results["rectification_ratio"] > 10,
        "p_type_valid": results["p_is_p_type"],
        "n_type_valid": results["n_is_n_type"],
        "mobility_valid": 0.001 < results["hole_drift_velocity"] < 100
    }
    
    return results


if __name__ == "__main__":
    results = validate_semiconductor_model()
    print("Semiconductor Model Validation:")
    print("=" * 50)
    for key, value in results.items():
        if key != "iv_curve":
            print(f"{key}: {value}")
    print("\nAll validations passed:", all(results["validations"].values()))

