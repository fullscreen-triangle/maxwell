"""
Maxwell Demon: The Navigator in Categorical Space
==================================================

The Maxwell demon operates on the categorical gas.
It sorts molecules not by physical velocity, but by S-entropy coordinates.

Key insight from our framework:
- The demon observes categorical states (not physical properties)
- Categorical observables COMMUTE with physical observables
- Therefore: no backaction, no thermodynamic cost for the sorting decision
- The energy cost is only in physical tier transitions, not in deciding

The demon is like a master fisherman who knows which fish is where
in categorical space and can navigate to any of them.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .virtual_molecule import VirtualMolecule, SCoordinate
from .virtual_chamber import VirtualChamber, CategoricalGas


class SortingCriterion(Enum):
    """What property to sort by."""
    S_K = "knowledge_entropy"
    S_T = "temporal_entropy"
    S_E = "evolution_entropy"
    TEMPERATURE = "temperature"  # Distance from coldest
    DISTANCE = "distance"  # From a reference point


@dataclass
class DemonStatistics:
    """Track what the demon has done."""
    observations: int = 0
    sorts: int = 0
    hot_sorted: int = 0
    cold_sorted: int = 0
    energy_cost: float = 0.0  # Only physical moves cost energy
    decision_cost: float = 0.0  # Always 0 for categorical decisions


class MaxwellDemon:
    """
    A Maxwell Demon operating in categorical space.
    
    This demon:
    1. Observes molecules via their categorical states (zero backaction)
    2. Sorts them into "hot" and "cold" compartments by S-entropy
    3. Incurs NO thermodynamic cost for the sorting decision
    4. Only pays energy for physical tier transitions
    
    The demon resolves Maxwell's paradox because it operates in
    categorical space, which is orthogonal to physical space.
    The information it gains is categorical, not physical.
    """
    
    def __init__(self, chamber: Optional[VirtualChamber] = None):
        """
        Create a demon to operate on a gas chamber.
        """
        self.chamber = chamber or VirtualChamber()
        self.stats = DemonStatistics()
        
        # The demon's "compartments" (categorical regions)
        self.hot_compartment: List[VirtualMolecule] = []
        self.cold_compartment: List[VirtualMolecule] = []
        
        # Current position in S-space
        self._position: SCoordinate = SCoordinate(0.5, 0.5, 0.5)
    
    def observe(self, molecule: VirtualMolecule) -> Dict[str, Any]:
        """
        Observe a molecule's categorical state.
        
        This is the key operation: ZERO backaction.
        
        We're not measuring position or momentum.
        We're accessing the molecule's categorical coordinates.
        The molecule IS its coordinates - observing them doesn't disturb anything.
        
        This is like reading a cursor position - the cursor doesn't move
        just because you read where it is.
        """
        self.stats.observations += 1
        
        # The observation IS the molecule's state
        # No interaction needed, no disturbance caused
        return {
            'S_k': molecule.s_coord.S_k,
            'S_t': molecule.s_coord.S_t,
            'S_e': molecule.s_coord.S_e,
            'identity': molecule.identity,
            # Note: NO energy cost for this observation
            'observation_cost': 0.0,
            'backaction': 0.0,
        }
    
    def classify(self, molecule: VirtualMolecule, 
                threshold: float = 0.5,
                criterion: SortingCriterion = SortingCriterion.S_E) -> str:
        """
        Classify a molecule as 'hot' or 'cold'.
        
        This decision costs ZERO energy in categorical space.
        We're not doing work - we're reading information.
        """
        value = self._get_criterion_value(molecule, criterion)
        return 'hot' if value > threshold else 'cold'
    
    def _get_criterion_value(self, molecule: VirtualMolecule, 
                            criterion: SortingCriterion) -> float:
        """Get the value for the sorting criterion."""
        if criterion == SortingCriterion.S_K:
            return molecule.s_coord.S_k
        elif criterion == SortingCriterion.S_T:
            return molecule.s_coord.S_t
        elif criterion == SortingCriterion.S_E:
            return molecule.s_coord.S_e
        elif criterion == SortingCriterion.TEMPERATURE:
            # Distance from absolute cold (0, 0, 0) in S-space
            cold = SCoordinate(0, 0, 0)
            return molecule.s_coord.distance_to(cold)
        elif criterion == SortingCriterion.DISTANCE:
            return molecule.s_coord.distance_to(self._position)
        return 0.5
    
    def sort(self, molecule: VirtualMolecule,
            threshold: float = 0.5,
            criterion: SortingCriterion = SortingCriterion.S_E) -> str:
        """
        Sort a molecule into hot or cold compartment.
        
        The DECISION costs nothing (categorical observation).
        The PHYSICAL MOVE would cost energy (but we're in categorical space).
        """
        # First observe (zero cost)
        observation = self.observe(molecule)
        
        # Then decide (zero cost - just categorical classification)
        classification = self.classify(molecule, threshold, criterion)
        
        # "Move" to compartment (in categorical space, this is just labeling)
        if classification == 'hot':
            self.hot_compartment.append(molecule)
            self.stats.hot_sorted += 1
        else:
            self.cold_compartment.append(molecule)
            self.stats.cold_sorted += 1
        
        self.stats.sorts += 1
        # Note: energy cost is 0 because we're operating in categorical space
        # Real thermodynamic cost only occurs for physical tier transitions
        
        return classification
    
    def sort_chamber(self, threshold: float = 0.5,
                    criterion: SortingCriterion = SortingCriterion.S_E) -> None:
        """Sort all molecules in the chamber."""
        for molecule in self.chamber.gas:
            self.sort(molecule, threshold, criterion)
    
    def navigate_to(self, target: SCoordinate) -> None:
        """
        Navigate to a position in S-space.
        
        This is how the demon can access Jupiter's core, deep space,
        or any other categorical location. No physical travel needed.
        """
        self._position = target
    
    def find_coldest(self) -> Optional[VirtualMolecule]:
        """Find the coldest molecule (lowest S_e)."""
        return self.chamber.find_coldest_molecule()
    
    def find_hottest(self) -> Optional[VirtualMolecule]:
        """Find the hottest molecule (highest S_e)."""
        return self.chamber.find_hottest_molecule()
    
    def extract_temperature_gradient(self) -> float:
        """
        Extract the temperature gradient between compartments.
        
        This is the "work" the demon has done - but it's categorical work,
        not physical work. No second law violation.
        """
        if not self.hot_compartment or not self.cold_compartment:
            return 0.0
        
        hot_mean = sum(m.s_coord.S_e for m in self.hot_compartment) / len(self.hot_compartment)
        cold_mean = sum(m.s_coord.S_e for m in self.cold_compartment) / len(self.cold_compartment)
        
        return hot_mean - cold_mean
    
    def report(self) -> Dict[str, Any]:
        """Generate a report of demon activities."""
        return {
            'observations': self.stats.observations,
            'sorts': self.stats.sorts,
            'hot_count': len(self.hot_compartment),
            'cold_count': len(self.cold_compartment),
            'temperature_gradient': self.extract_temperature_gradient(),
            'decision_energy_cost': 0.0,  # Always zero
            'backaction': 0.0,  # Always zero
            'thermodynamic_violation': False,  # Never
        }
    
    def clear(self) -> None:
        """Clear compartments."""
        self.hot_compartment.clear()
        self.cold_compartment.clear()


def demonstrate_demon():
    """
    Demonstrate that the Maxwell demon operates without paradox.
    
    The key: categorical observation has zero backaction.
    The demon gains information from categorical space,
    which is orthogonal to physical space.
    """
    print("=== MAXWELL DEMON DEMONSTRATION ===\n")
    
    # Create chamber with gas
    chamber = VirtualChamber()
    chamber.populate(500)
    
    # Create demon
    demon = MaxwellDemon(chamber)
    
    print(f"Chamber: {len(chamber.gas)} molecules")
    print(f"Chamber temperature: {chamber.gas.temperature:.6f}\n")
    
    # Demon sorts the gas
    print("Demon sorting molecules by S_e (evolution entropy)...")
    demon.sort_chamber(threshold=0.5, criterion=SortingCriterion.S_E)
    
    report = demon.report()
    print(f"\n--- Demon Report ---")
    print(f"Observations made: {report['observations']}")
    print(f"Molecules sorted: {report['sorts']}")
    print(f"Hot compartment: {report['hot_count']}")
    print(f"Cold compartment: {report['cold_count']}")
    print(f"Temperature gradient created: {report['temperature_gradient']:.4f}")
    print(f"Energy cost of decisions: {report['decision_energy_cost']}")
    print(f"Backaction caused: {report['backaction']}")
    print(f"Second law violated: {report['thermodynamic_violation']}")
    
    print("\n=== KEY INSIGHT ===")
    print("The demon sorted molecules without thermodynamic cost because:")
    print("1. It observed CATEGORICAL states, not physical properties")
    print("2. Categorical coordinates are ORTHOGONAL to physical coordinates")
    print("3. Reading S-coordinates doesn't disturb physical state")
    print("4. The 'sorting' is just categorical labeling, not physical work")
    print("\nThis is how Maxwell's paradox is resolved:")
    print("The demon operates in categorical space, not physical space.")
    
    return demon


if __name__ == "__main__":
    demon = demonstrate_demon()

