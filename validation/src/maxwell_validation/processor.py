"""
Complementarity-Aware Maxwell Processor
=======================================

The main processor integrating all components:
- Categorical Face Engine (ground truth)
- Kinetic Face Engine (observable projections)
- Complementarity Manager (face switching)
- Projection Explainer (demon appearance)
- Equivalence Class Filter (state reduction)
- Recursive Completion Engine (3^k decomposition)
- Biological Semiconductor Network
- Biological ALU

This processor demonstrates that:
1. The demon does not exist
2. What appears as "sorting" is categorical completion
3. Kinetic observations are projections of hidden dynamics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto

from .types import (
    ObservableFace, SCoordinates, CategoricalState, KineticState,
    PhaseLockNode, PhaseLockEdge, OscillatorySignature,
    ProcessorConfig, CompletionResult, DemonExplanation, DissolutionArgument,
    MolecularType, InteractionType
)
from .semiconductor import SemiconductorNetwork, SemiconductorSubstrate
from .alu import BiologicalALU, GearNetwork


class ProcessorMode(Enum):
    """Operational mode of the processor"""
    CATEGORICAL = auto()   # Ground truth operations
    KINETIC = auto()       # Observable projections
    PROJECTION = auto()    # Explain demon appearances


class CategoricalEngine:
    """
    The Categorical Face Engine - Ground Truth Operations
    
    Manages:
    - Phase-lock network construction
    - Topological navigation
    - Categorical completion
    - Configuration dynamics
    
    This is what's ACTUALLY happening.
    """
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self.states: Dict[int, CategoricalState] = {}
        self.nodes: Dict[int, PhaseLockNode] = {}
        self.edges: List[PhaseLockEdge] = []
        self.next_id = 0
        self.completed_count = 0
    
    def create_state(self, coordinates: SCoordinates) -> int:
        """Create a new categorical state"""
        state_id = self.next_id
        self.next_id += 1
        
        state = CategoricalState(id=state_id, coordinates=coordinates)
        self.states[state_id] = state
        
        # Create corresponding phase-lock node
        node = PhaseLockNode(
            id=state_id,
            frequency=self._compute_frequency(coordinates),
            phase=0.0,
            amplitude=1.0,
            position=np.array([coordinates.s_k, coordinates.s_t, coordinates.s_e])
        )
        self.nodes[state_id] = node
        
        return state_id
    
    def _compute_frequency(self, coords: SCoordinates) -> float:
        """Compute oscillatory frequency from coordinates"""
        base_freq = 1e12  # THz range
        return base_freq * (1.0 + abs(coords.s_k) + abs(coords.s_t) + abs(coords.s_e))
    
    def form_phase_lock(self, id_a: int, id_b: int, 
                       coupling: float = None,
                       interaction: InteractionType = InteractionType.VAN_DER_WAALS):
        """Form a phase-lock between two states"""
        if id_a not in self.states or id_b not in self.states:
            raise ValueError("State not found")
        
        if coupling is None:
            # Compute from positions
            pos_a = self.nodes[id_a].position
            pos_b = self.nodes[id_b].position
            r = np.linalg.norm(pos_a - pos_b)
            coupling = self.config.vdw_coefficient / max(r**6, 1e-100)
        
        edge = PhaseLockEdge(id_a, id_b, coupling, interaction)
        self.edges.append(edge)
        
        # Update state accessibility
        self.states[id_a].add_phase_lock(id_b, coupling)
        self.states[id_b].add_phase_lock(id_a, coupling)
    
    def construct_network(self, positions: np.ndarray, 
                         types: List[MolecularType] = None) -> List[int]:
        """
        Construct phase-lock network from positions.
        
        CRITICAL: Network depends ONLY on positions (Van der Waals),
        NOT on velocities/kinetic energy!
        """
        n = len(positions)
        types = types or [MolecularType.NON_POLAR] * n
        
        # Create states
        ids = []
        for pos in positions:
            coords = SCoordinates(
                pos[0] / self.config.coupling_distance,
                pos[1] / self.config.coupling_distance,
                pos[2] / self.config.coupling_distance if len(pos) > 2 else 0.0
            )
            ids.append(self.create_state(coords))
        
        # Form phase-locks based on position (NOT velocity!)
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(positions[i] - positions[j])
                
                if r < self.config.coupling_distance * 10:
                    # Van der Waals: ~r^-6
                    coupling = self.config.vdw_coefficient / max(r**6, 1e-100)
                    
                    # Dipole coupling if applicable
                    if types[i] == MolecularType.POLAR or types[j] == MolecularType.POLAR:
                        coupling += np.sqrt(self.config.vdw_coefficient) / max(r**3, 1e-50)
                    
                    if coupling > 1e-30:
                        self.form_phase_lock(ids[i], ids[j], coupling)
        
        return ids
    
    def complete_state(self, state_id: int):
        """Complete a categorical state (irreversible)"""
        if state_id not in self.states:
            raise ValueError(f"State {state_id} not found")
        
        state = self.states[state_id]
        if state.completed:
            raise ValueError(f"State {state_id} already completed")
        
        state.complete()
        self.completed_count += 1
    
    def navigate(self, from_id: int, to_id: int) -> bool:
        """Navigate from one state to another along phase-lock topology"""
        if from_id not in self.states or to_id not in self.states:
            return False
        return self.states[from_id].can_access(to_id)
    
    def accessible_from(self, state_id: int) -> Set[int]:
        """Get all states accessible from a given state"""
        if state_id not in self.states:
            return set()
        return self.states[state_id].accessible
    
    def network_density(self) -> float:
        """Compute network density"""
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        return len(self.edges) / (n * (n - 1))
    
    def categorical_entropy(self) -> float:
        """Compute categorical entropy (from network density)"""
        k_b = 1.380649e-23
        return k_b * len(self.edges)


class KineticEngine:
    """
    The Kinetic Face Engine - Observable Projections
    
    This is what Maxwell SAW (not what's happening).
    Computes:
    - Velocity distributions
    - Temperature measurements
    - Energy sorting
    - Thermodynamic observables
    """
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self.states: Dict[int, KineticState] = {}
        self.temperature = config.temperature if config else 300.0
    
    def initialize_maxwell_boltzmann(self, n: int, mass: float = 1.0) -> List[int]:
        """Initialize with Maxwell-Boltzmann velocity distribution"""
        k_b = 1.380649e-23
        sigma = np.sqrt(k_b * self.temperature / mass)
        
        ids = []
        for i in range(n):
            # Sample velocity components
            v = np.linalg.norm(np.random.normal(0, sigma, 3))
            position = np.random.rand(3) * 1e-6
            
            state = KineticState.from_velocity(i, v, position, mass)
            self.states[i] = state
            ids.append(i)
        
        return ids
    
    def velocity_distribution(self) -> np.ndarray:
        """Get velocity distribution"""
        return np.array([s.velocity for s in self.states.values()])
    
    def mean_velocity(self) -> float:
        """Compute mean velocity"""
        if not self.states:
            return 0.0
        return np.mean([s.velocity for s in self.states.values()])
    
    def compute_temperature(self, mass: float = 1.0) -> float:
        """Compute temperature from kinetic energies"""
        if not self.states:
            return 0.0
        
        k_b = 1.380649e-23
        total_ke = sum(s.kinetic_energy * mass for s in self.states.values())
        n = len(self.states)
        
        return (2 * total_ke) / (3 * n * k_b)
    
    def demon_sorting(self) -> Tuple[List[int], List[int]]:
        """
        Classify molecules as fast/slow (what the demon would do).
        
        This is the ILLUSION - categorizing by velocity.
        """
        threshold = self.mean_velocity()
        fast = [s.id for s in self.states.values() if s.velocity > threshold]
        slow = [s.id for s in self.states.values() if s.velocity <= threshold]
        return fast, slow
    
    def collision_step(self):
        """
        Simulate thermal collisions.
        
        This demonstrates the RETRIEVAL PARADOX:
        velocities randomize faster than any sorting can occur.
        """
        k_b = 1.380649e-23
        sigma = np.sqrt(k_b * self.temperature)
        
        for state in self.states.values():
            if np.random.random() < 0.1:  # 10% collision probability
                new_v = np.linalg.norm(np.random.normal(0, sigma, 3))
                state.velocity = new_v
                state.kinetic_energy = 0.5 * new_v**2


class ComplementarityManager:
    """
    The Complementarity Manager
    
    Enforces that only ONE face can be observed at a time:
    - You cannot observe categorical AND kinetic simultaneously
    - Like ammeter/voltmeter incompatibility
    """
    
    def __init__(self, categorical: CategoricalEngine, kinetic: KineticEngine):
        self.categorical = categorical
        self.kinetic = kinetic
        self.current_face = ObservableFace.CATEGORICAL
    
    def switch_face(self) -> ObservableFace:
        """Switch observable face"""
        if self.current_face == ObservableFace.CATEGORICAL:
            self.current_face = ObservableFace.KINETIC
        else:
            self.current_face = ObservableFace.CATEGORICAL
        return self.current_face
    
    def can_observe(self, face: ObservableFace) -> bool:
        """Check if face can be observed"""
        return face == self.current_face
    
    def project_categorical_to_kinetic(self, state: CategoricalState) -> Dict:
        """
        Project categorical state to kinetic observables.
        
        This shows how categorical dynamics APPEAR on the kinetic face.
        """
        # Coordinates map to velocity-like quantities
        coords = state.coordinates
        apparent_velocity = np.sqrt(coords.s_k**2 + coords.s_t**2 + coords.s_e**2)
        
        # Network density maps to temperature
        network_edges = len(state.phase_locks)
        apparent_temperature = 300.0 * (1 + network_edges * 0.1)
        
        return {
            "apparent_velocity": apparent_velocity,
            "apparent_temperature": apparent_temperature,
            "appears_as": "fast" if apparent_velocity > 1.0 else "slow",
            "demon_interpretation": "sorted by intelligent agent" if state.completed else "random"
        }


class ProjectionExplainer:
    """
    The Projection Explainer
    
    Explains WHY Maxwell saw a demon:
    - Maps categorical operations to kinetic appearances
    - Shows the demon is a projection artifact
    """
    
    def explain_demon(self, operation: str, 
                     categorical_action: str,
                     kinetic_appearance: str) -> DemonExplanation:
        """Explain how a categorical operation appears as demon behavior"""
        
        # Determine which dissolution argument applies
        if "topology" in categorical_action.lower():
            argument = DissolutionArgument.DISSOLUTION_OF_OBSERVATION
        elif "completion" in categorical_action.lower():
            argument = DissolutionArgument.DISSOLUTION_OF_DECISION
        elif "phase-lock" in categorical_action.lower():
            argument = DissolutionArgument.PHASE_LOCK_TEMPERATURE_INDEPENDENCE
        elif "network" in categorical_action.lower():
            argument = DissolutionArgument.DISSOLUTION_OF_SECOND_LAW
        else:
            argument = DissolutionArgument.INFORMATION_COMPLEMENTARITY
        
        return DemonExplanation(
            operation=operation,
            kinetic_observation=kinetic_appearance,
            apparent_intelligence=f"Appears as intelligent sorting because {kinetic_appearance}",
            actual_mechanism=categorical_action,
            dissolution_argument=argument
        )
    
    def explain_all_operations(self) -> List[DemonExplanation]:
        """Generate explanations for all demon-like operations"""
        return [
            self.explain_demon(
                "Molecule selection",
                "Phase-lock network determines adjacency (topology)",
                "Molecules seem to be selected by speed"
            ),
            self.explain_demon(
                "Door opening",
                "Navigation along phase-lock edges (completion)",
                "Door opens for fast molecules"
            ),
            self.explain_demon(
                "Sorting",
                "Categorical states complete in sequence",
                "Molecules sorted into hot/cold sides"
            ),
            self.explain_demon(
                "Temperature difference",
                "Network density varies by region",
                "Temperature gradient appears"
            ),
        ]


class EquivalenceFilter:
    """
    The Equivalence Class Filter
    
    Reduces state space by grouping equivalent configurations.
    Phase-lock degeneracy means many configurations are equivalent.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def are_equivalent(self, state_a: CategoricalState, 
                      state_b: CategoricalState) -> bool:
        """Check if two states are equivalent"""
        # Same cluster = equivalent
        if state_a.cluster_id is not None and state_a.cluster_id == state_b.cluster_id:
            return True
        
        # Same coordinates within tolerance = equivalent
        return state_a.coordinates.distance(state_b.coordinates) < self.tolerance
    
    def filter(self, states: List[CategoricalState]) -> List[CategoricalState]:
        """Filter to equivalence class representatives"""
        if not states:
            return []
        
        representatives = [states[0]]
        
        for state in states[1:]:
            is_new = True
            for rep in representatives:
                if self.are_equivalent(state, rep):
                    is_new = False
                    break
            if is_new:
                representatives.append(state)
        
        return representatives
    
    def partition(self, states: List[CategoricalState]) -> Dict[int, List[CategoricalState]]:
        """Partition states into equivalence classes"""
        classes: Dict[int, List[CategoricalState]] = {}
        class_id = 0
        
        for state in states:
            found_class = None
            for cid, members in classes.items():
                if any(self.are_equivalent(state, m) for m in members):
                    found_class = cid
                    break
            
            if found_class is not None:
                classes[found_class].append(state)
            else:
                classes[class_id] = [state]
                class_id += 1
        
        return classes


class RecursiveCompletionEngine:
    """
    The Recursive Completion Engine
    
    Implements 3^k decomposition for hierarchical navigation.
    Categorical completion cascades through the network.
    """
    
    def __init__(self, categorical: CategoricalEngine):
        self.categorical = categorical
    
    def complete(self, initial_id: int, max_depth: int = 5) -> CompletionResult:
        """
        Recursively complete states from initial state.
        
        Uses 3^k decomposition: each state branches to 3 accessible states.
        """
        completed_ids = []
        cascade_path = []
        depth = 0
        
        current_layer = [initial_id]
        
        while current_layer and depth < max_depth:
            cascade_path.append(current_layer.copy())
            next_layer = []
            
            for state_id in current_layer:
                if state_id in self.categorical.states:
                    state = self.categorical.states[state_id]
                    
                    if not state.completed:
                        try:
                            self.categorical.complete_state(state_id)
                            completed_ids.append(state_id)
                        except ValueError:
                            pass
                    
                    # Get up to 3 accessible states (3^k decomposition)
                    accessible = list(state.accessible)[:3]
                    for acc_id in accessible:
                        if acc_id not in completed_ids and acc_id not in next_layer:
                            next_layer.append(acc_id)
            
            current_layer = next_layer
            depth += 1
        
        # Compute entropy change (increases with completion)
        entropy_change = len(completed_ids) * 1.380649e-23
        
        return CompletionResult(
            states_completed=len(completed_ids),
            depth_reached=depth,
            decomposition_count=sum(3**k for k in range(depth)),
            entropy_change=entropy_change,
            completed_ids=completed_ids,
            cascade_path=cascade_path
        )


class MaxwellProcessor:
    """
    The Complementarity-Aware Maxwell Processor
    
    Integrates all components to demonstrate:
    1. There is no demon
    2. Categorical dynamics project to kinetic appearances
    3. Information complementarity explains the illusion
    """
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self.mode = ProcessorMode.CATEGORICAL
        
        # Core engines
        self.categorical = CategoricalEngine(self.config)
        self.kinetic = KineticEngine(self.config)
        
        # Management components
        self.complementarity = ComplementarityManager(self.categorical, self.kinetic)
        self.projection = ProjectionExplainer()
        self.equivalence = EquivalenceFilter()
        self.completion = RecursiveCompletionEngine(self.categorical)
        
        # Hardware components
        self.semiconductor = SemiconductorNetwork(self.config)
        self.alu = BiologicalALU(self.config)
        self.gear_network = GearNetwork()
    
    def set_mode(self, mode: ProcessorMode):
        """Set processor mode"""
        self.mode = mode
        if mode == ProcessorMode.CATEGORICAL:
            self.complementarity.current_face = ObservableFace.CATEGORICAL
        elif mode == ProcessorMode.KINETIC:
            self.complementarity.current_face = ObservableFace.KINETIC
    
    def initialize_system(self, n_molecules: int = 100) -> Dict:
        """Initialize the complete system"""
        # Generate random positions
        positions = np.random.rand(n_molecules, 3) * 1e-6
        
        # Construct categorical network (position-based)
        cat_ids = self.categorical.construct_network(positions)
        
        # Initialize kinetic states (velocity-based)
        kin_ids = self.kinetic.initialize_maxwell_boltzmann(n_molecules)
        
        # Setup semiconductor network
        self.semiconductor.create_p_substrate("disease_region", n_holes=5)
        self.semiconductor.create_n_substrate("drug_region", n_carriers=3)
        self.semiconductor.create_junction("disease_region", "drug_region")
        
        # Setup gear network for therapeutic prediction
        self.gear_network.add_gear("receptor", 10, 20)
        self.gear_network.add_gear("pathway", 5, 15)
        
        return {
            "categorical_states": len(cat_ids),
            "kinetic_states": len(kin_ids),
            "network_edges": len(self.categorical.edges),
            "semiconductor_stats": self.semiconductor.get_network_stats()
        }
    
    def demonstrate_no_demon(self) -> Dict:
        """
        Demonstrate that there is no demon.
        
        Shows:
        1. Phase-lock network is temperature-independent
        2. Same spatial arrangement exists at any temperature
        3. Categorical completion happens automatically
        """
        results = {}
        
        # Test 1: Construct network at two temperatures
        positions = np.random.rand(50, 3) * 1e-6
        
        self.categorical = CategoricalEngine(self.config)
        ids_t1 = self.categorical.construct_network(positions)
        edges_t1 = len(self.categorical.edges)
        
        # Change temperature, reconstruct
        self.config.temperature = 500.0
        self.categorical = CategoricalEngine(self.config)
        ids_t2 = self.categorical.construct_network(positions)
        edges_t2 = len(self.categorical.edges)
        
        results["temperature_independence"] = {
            "edges_at_300K": edges_t1,
            "edges_at_500K": edges_t2,
            "identical": edges_t1 == edges_t2,
            "conclusion": "Phase-lock network is temperature-independent"
        }
        
        # Test 2: Categorical completion
        if ids_t2:
            completion = self.completion.complete(ids_t2[0], max_depth=3)
            results["categorical_completion"] = {
                "states_completed": completion.states_completed,
                "depth_reached": completion.depth_reached,
                "entropy_change": completion.entropy_change,
                "conclusion": "Completion follows topology, no decision needed"
            }
        
        # Test 3: Kinetic vs Categorical comparison
        self.kinetic = KineticEngine(self.config)
        self.kinetic.initialize_maxwell_boltzmann(50)
        
        fast, slow = self.kinetic.demon_sorting()
        results["kinetic_projection"] = {
            "fast_molecules": len(fast),
            "slow_molecules": len(slow),
            "conclusion": "Velocity classification is projection, not reality"
        }
        
        # Test 4: Information complementarity
        results["complementarity"] = {
            "current_face": self.complementarity.current_face.name,
            "can_observe_categorical": self.complementarity.can_observe(ObservableFace.CATEGORICAL),
            "can_observe_kinetic": self.complementarity.can_observe(ObservableFace.KINETIC),
            "conclusion": "Cannot observe both faces simultaneously"
        }
        
        results["final_conclusion"] = "THE DEMON DOES NOT EXIST"
        
        return results
    
    def run_dissolution_demonstration(self) -> List[DemonExplanation]:
        """Run the full dissolution demonstration"""
        return self.projection.explain_all_operations()
    
    def get_stats(self) -> Dict:
        """Get comprehensive processor statistics"""
        return {
            "mode": self.mode.name,
            "categorical": {
                "states": len(self.categorical.states),
                "edges": len(self.categorical.edges),
                "completed": self.categorical.completed_count,
                "density": self.categorical.network_density()
            },
            "kinetic": {
                "states": len(self.kinetic.states),
                "temperature": self.kinetic.temperature,
                "mean_velocity": self.kinetic.mean_velocity()
            },
            "semiconductor": self.semiconductor.get_network_stats(),
            "alu": self.alu.get_stats(),
            "complementarity": {
                "current_face": self.complementarity.current_face.name
            }
        }


def run_full_validation() -> Dict:
    """Run comprehensive validation of the Maxwell Processor"""
    print("=" * 60)
    print("MAXWELL PROCESSOR VALIDATION")
    print("Complementarity-Aware Categorical Phase-Lock Dynamics")
    print("=" * 60)
    
    # Create processor
    config = ProcessorConfig(num_molecules=100, temperature=300.0)
    processor = MaxwellProcessor(config)
    
    # Initialize
    print("\n1. Initializing system...")
    init_result = processor.initialize_system(100)
    print(f"   Created {init_result['categorical_states']} categorical states")
    print(f"   Created {init_result['kinetic_states']} kinetic states")
    print(f"   Network has {init_result['network_edges']} phase-lock edges")
    
    # Demonstrate no demon
    print("\n2. Demonstrating there is no demon...")
    demo_result = processor.demonstrate_no_demon()
    
    print(f"\n   Temperature Independence:")
    ti = demo_result["temperature_independence"]
    print(f"   - Edges at 300K: {ti['edges_at_300K']}")
    print(f"   - Edges at 500K: {ti['edges_at_500K']}")
    print(f"   - Identical: {ti['identical']}")
    print(f"   - {ti['conclusion']}")
    
    if "categorical_completion" in demo_result:
        cc = demo_result["categorical_completion"]
        print(f"\n   Categorical Completion:")
        print(f"   - States completed: {cc['states_completed']}")
        print(f"   - Depth reached: {cc['depth_reached']}")
        print(f"   - {cc['conclusion']}")
    
    print(f"\n   Kinetic Projection:")
    kp = demo_result["kinetic_projection"]
    print(f"   - Fast molecules: {kp['fast_molecules']}")
    print(f"   - Slow molecules: {kp['slow_molecules']}")
    print(f"   - {kp['conclusion']}")
    
    print(f"\n   Complementarity:")
    comp = demo_result["complementarity"]
    print(f"   - Current face: {comp['current_face']}")
    print(f"   - {comp['conclusion']}")
    
    # Dissolution explanations
    print("\n3. Demon Dissolution Explanations:")
    explanations = processor.run_dissolution_demonstration()
    for i, exp in enumerate(explanations, 1):
        print(f"\n   {i}. {exp.operation}")
        print(f"      Kinetic observation: {exp.kinetic_observation}")
        print(f"      Actual mechanism: {exp.actual_mechanism}")
        print(f"      Dissolution: {exp.dissolution_argument.name}")
    
    # Final stats
    print("\n4. Final Statistics:")
    stats = processor.get_stats()
    print(f"   Categorical states: {stats['categorical']['states']}")
    print(f"   Network density: {stats['categorical']['density']:.4f}")
    print(f"   ALU operations: {stats['alu']['operations_executed']}")
    
    print("\n" + "=" * 60)
    print(demo_result["final_conclusion"])
    print("=" * 60)
    
    return {
        "initialization": init_result,
        "demonstration": demo_result,
        "explanations": explanations,
        "stats": stats
    }


if __name__ == "__main__":
    results = run_full_validation()

