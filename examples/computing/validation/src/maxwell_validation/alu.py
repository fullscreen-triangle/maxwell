"""
Biological ALU Engine
=====================

Implements the Biological Semiconductor Junction Oscillatory Integrated Logic Circuits:
- BMD Transistors (information switches)
- Tri-dimensional logic gates (AND, OR, XOR in S-space)
- Gear ratio interconnects
- S-dictionary memory
- Virtual processor ALU operations

Based on the biological computer architecture from the theoretical framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum, auto
from .types import SCoordinates, OscillatorySignature, ProcessorConfig


class GateType(Enum):
    """Types of tri-dimensional logic gates"""
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    NAND = auto()
    NOR = auto()
    HADAMARD = auto()  # Quantum-inspired superposition
    PHASE = auto()      # Phase rotation


class ALUOperation(Enum):
    """ALU operations"""
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    PHASE_SHIFT = auto()
    FREQUENCY_MODULATE = auto()
    RESONANCE_MATCH = auto()
    ENTROPY_COMPUTE = auto()


@dataclass
class BMDTransistor:
    """
    A Biological Maxwell Demon Transistor.
    
    Acts as an information switch that amplifies or gates signals
    based on pattern recognition (not voltage).
    
    The BMD transistor operates through:
    1. Pattern recognition at the gate (input filtering)
    2. Information catalysis (amplification)
    3. Therapeutic channeling (output direction)
    """
    id: int
    gate_pattern: OscillatorySignature  # Pattern that activates the gate
    gain: float = 1000.0                 # Information amplification factor
    threshold: float = 0.5               # Activation threshold
    state: bool = False                  # Current switch state
    
    # Catalytic efficiency (bits/molecule)
    catalytic_efficiency: float = 3000.0
    
    def match_score(self, input_signal: OscillatorySignature) -> float:
        """Compute pattern match score with gate"""
        return input_signal.overlap_integral(self.gate_pattern)
    
    def activate(self, input_signal: OscillatorySignature) -> bool:
        """Attempt to activate the transistor"""
        score = self.match_score(input_signal)
        self.state = score > self.threshold
        return self.state
    
    def amplify(self, input_value: float) -> float:
        """Amplify input if transistor is active"""
        if self.state:
            return input_value * self.gain
        return 0.0
    
    def information_catalysis(self, input_bits: float) -> float:
        """
        Apply information catalysis.
        
        Returns enhanced information processing capacity.
        """
        if self.state:
            return input_bits * (1 + np.log2(self.catalytic_efficiency))
        return input_bits


@dataclass
class TriDimensionalGate:
    """
    A tri-dimensional logic gate operating in S-space.
    
    Unlike classical gates that operate on bits,
    these gates operate on S-coordinates (S_k, S_t, S_e).
    """
    gate_type: GateType
    id: int = 0
    
    def __call__(self, *inputs: SCoordinates) -> SCoordinates:
        """Apply the gate operation"""
        if self.gate_type == GateType.AND:
            return self._and_gate(inputs)
        elif self.gate_type == GateType.OR:
            return self._or_gate(inputs)
        elif self.gate_type == GateType.XOR:
            return self._xor_gate(inputs)
        elif self.gate_type == GateType.NOT:
            return self._not_gate(inputs[0])
        elif self.gate_type == GateType.HADAMARD:
            return self._hadamard_gate(inputs[0])
        elif self.gate_type == GateType.PHASE:
            return self._phase_gate(inputs[0], np.pi/4)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")
    
    def _and_gate(self, inputs: Tuple[SCoordinates, ...]) -> SCoordinates:
        """
        AND gate: minimum of each coordinate.
        Categorical intersection.
        """
        if len(inputs) < 2:
            return inputs[0] if inputs else SCoordinates.origin()
        
        s_k = min(inp.s_k for inp in inputs)
        s_t = min(inp.s_t for inp in inputs)
        s_e = min(inp.s_e for inp in inputs)
        return SCoordinates(s_k, s_t, s_e)
    
    def _or_gate(self, inputs: Tuple[SCoordinates, ...]) -> SCoordinates:
        """
        OR gate: maximum of each coordinate.
        Categorical union.
        """
        if len(inputs) < 2:
            return inputs[0] if inputs else SCoordinates.origin()
        
        s_k = max(inp.s_k for inp in inputs)
        s_t = max(inp.s_t for inp in inputs)
        s_e = max(inp.s_e for inp in inputs)
        return SCoordinates(s_k, s_t, s_e)
    
    def _xor_gate(self, inputs: Tuple[SCoordinates, ...]) -> SCoordinates:
        """
        XOR gate: symmetric difference.
        Categorical exclusive operation.
        """
        if len(inputs) < 2:
            return inputs[0] if inputs else SCoordinates.origin()
        
        # XOR as difference of magnitudes
        a, b = inputs[0], inputs[1]
        s_k = abs(a.s_k - b.s_k)
        s_t = abs(a.s_t - b.s_t)
        s_e = abs(a.s_e - b.s_e)
        return SCoordinates(s_k, s_t, s_e)
    
    def _not_gate(self, input_coord: SCoordinates) -> SCoordinates:
        """
        NOT gate: conjugate coordinates.
        Categorical inversion.
        """
        return input_coord.conjugate()
    
    def _hadamard_gate(self, input_coord: SCoordinates) -> SCoordinates:
        """
        Hadamard-like gate: creates superposition.
        Rotates into equal superposition of all basis states.
        """
        norm = np.sqrt(input_coord.s_k**2 + input_coord.s_t**2 + input_coord.s_e**2)
        if norm < 1e-10:
            return SCoordinates(1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))
        
        # Rotate to equal superposition
        factor = 1 / np.sqrt(3)
        return SCoordinates(
            factor * (input_coord.s_k + input_coord.s_t + input_coord.s_e) / norm,
            factor * (input_coord.s_k - input_coord.s_t + input_coord.s_e) / norm,
            factor * (input_coord.s_k + input_coord.s_t - input_coord.s_e) / norm
        )
    
    def _phase_gate(self, input_coord: SCoordinates, angle: float) -> SCoordinates:
        """
        Phase gate: rotates in S_k-S_t plane.
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        new_sk = cos_a * input_coord.s_k - sin_a * input_coord.s_t
        new_st = sin_a * input_coord.s_k + cos_a * input_coord.s_t
        return SCoordinates(new_sk, new_st, input_coord.s_e)


@dataclass
class GearRatio:
    """
    A gear ratio for frequency transformation between oscillatory components.
    
    ω_output = G × ω_input
    
    Enables instant therapeutic prediction through predictable
    frequency transformations.
    """
    input_teeth: int
    output_teeth: int
    
    @property
    def ratio(self) -> float:
        """Compute gear ratio"""
        return self.output_teeth / self.input_teeth
    
    def transform_frequency(self, input_freq: float) -> float:
        """Transform input frequency to output frequency"""
        return input_freq * self.ratio
    
    def transform_signature(self, sig: OscillatorySignature) -> OscillatorySignature:
        """Transform an oscillatory signature through the gear"""
        return OscillatorySignature(
            amplitude=sig.amplitude,
            frequency=self.transform_frequency(sig.frequency),
            phase=sig.phase  # Phase preserved
        )


@dataclass
class SDictionaryEntry:
    """An entry in the S-dictionary memory"""
    key: SCoordinates
    value: any
    timestamp: float = 0.0
    access_count: int = 0


class SDictionaryMemory:
    """
    S-Dictionary Memory for biological computing.
    
    Stores information indexed by S-coordinates rather than
    conventional memory addresses.
    """
    
    def __init__(self, capacity: int = 1000, distance_threshold: float = 0.1):
        self.capacity = capacity
        self.distance_threshold = distance_threshold
        self.entries: List[SDictionaryEntry] = []
        self.time = 0.0
    
    def store(self, key: SCoordinates, value: any):
        """Store a value at S-coordinates"""
        # Check for existing entry nearby
        for entry in self.entries:
            if key.distance(entry.key) < self.distance_threshold:
                entry.value = value
                entry.timestamp = self.time
                return
        
        # Create new entry
        if len(self.entries) >= self.capacity:
            # Remove least accessed
            self.entries.sort(key=lambda e: e.access_count)
            self.entries.pop(0)
        
        self.entries.append(SDictionaryEntry(key, value, self.time))
        self.time += 1
    
    def retrieve(self, key: SCoordinates) -> Optional[any]:
        """Retrieve value at S-coordinates"""
        best_match = None
        best_distance = float('inf')
        
        for entry in self.entries:
            dist = key.distance(entry.key)
            if dist < best_distance:
                best_distance = dist
                best_match = entry
        
        if best_match and best_distance < self.distance_threshold:
            best_match.access_count += 1
            return best_match.value
        return None
    
    def fuzzy_retrieve(self, key: SCoordinates, n: int = 5) -> List[Tuple[SCoordinates, any, float]]:
        """Retrieve n nearest entries with distances"""
        distances = [(e.key, e.value, key.distance(e.key)) for e in self.entries]
        distances.sort(key=lambda x: x[2])
        return distances[:n]


class BiologicalALU:
    """
    A Biological Arithmetic Logic Unit.
    
    Operates on oscillatory signatures and S-coordinates
    rather than binary numbers.
    """
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        
        # Initialize components
        self.transistors: List[BMDTransistor] = []
        self.gates: Dict[str, TriDimensionalGate] = {}
        self.gears: Dict[str, GearRatio] = {}
        self.memory = SDictionaryMemory()
        
        # Initialize standard gates
        self._init_standard_gates()
        
        # Accumulator register (S-coordinates)
        self.accumulator = SCoordinates.origin()
        
        # Operation count
        self.op_count = 0
    
    def _init_standard_gates(self):
        """Initialize standard gate set"""
        self.gates["AND"] = TriDimensionalGate(GateType.AND)
        self.gates["OR"] = TriDimensionalGate(GateType.OR)
        self.gates["XOR"] = TriDimensionalGate(GateType.XOR)
        self.gates["NOT"] = TriDimensionalGate(GateType.NOT)
        self.gates["H"] = TriDimensionalGate(GateType.HADAMARD)
        self.gates["P"] = TriDimensionalGate(GateType.PHASE)
    
    def add_transistor(self, gate_pattern: OscillatorySignature, 
                      gain: float = 1000.0) -> BMDTransistor:
        """Add a BMD transistor to the ALU"""
        transistor = BMDTransistor(
            id=len(self.transistors),
            gate_pattern=gate_pattern,
            gain=gain
        )
        self.transistors.append(transistor)
        return transistor
    
    def add_gear(self, name: str, input_teeth: int, output_teeth: int) -> GearRatio:
        """Add a gear ratio to the ALU"""
        gear = GearRatio(input_teeth, output_teeth)
        self.gears[name] = gear
        return gear
    
    def execute(self, op: ALUOperation, 
                operand_a: SCoordinates,
                operand_b: Optional[SCoordinates] = None) -> SCoordinates:
        """Execute an ALU operation"""
        self.op_count += 1
        
        if op == ALUOperation.ADD:
            result = SCoordinates(
                operand_a.s_k + (operand_b.s_k if operand_b else 0),
                operand_a.s_t + (operand_b.s_t if operand_b else 0),
                operand_a.s_e + (operand_b.s_e if operand_b else 0)
            )
        
        elif op == ALUOperation.SUBTRACT:
            result = SCoordinates(
                operand_a.s_k - (operand_b.s_k if operand_b else 0),
                operand_a.s_t - (operand_b.s_t if operand_b else 0),
                operand_a.s_e - (operand_b.s_e if operand_b else 0)
            )
        
        elif op == ALUOperation.MULTIPLY:
            # Hadamard product in S-space
            if operand_b:
                result = SCoordinates(
                    operand_a.s_k * operand_b.s_k,
                    operand_a.s_t * operand_b.s_t,
                    operand_a.s_e * operand_b.s_e
                )
            else:
                result = operand_a
        
        elif op == ALUOperation.ENTROPY_COMPUTE:
            # Compute entropy from coordinates
            arr = operand_a.to_array()
            arr = np.abs(arr) + 1e-10  # Ensure positive
            arr = arr / arr.sum()  # Normalize
            entropy = -np.sum(arr * np.log2(arr))
            result = SCoordinates(entropy, entropy, entropy)
        
        elif op == ALUOperation.PHASE_SHIFT:
            # Apply phase gate
            result = self.gates["P"](operand_a)
        
        else:
            result = operand_a
        
        self.accumulator = result
        return result
    
    def apply_gate(self, gate_name: str, *inputs: SCoordinates) -> SCoordinates:
        """Apply a named gate to inputs"""
        if gate_name not in self.gates:
            raise ValueError(f"Unknown gate: {gate_name}")
        return self.gates[gate_name](*inputs)
    
    def transform_frequency(self, gear_name: str, 
                           signature: OscillatorySignature) -> OscillatorySignature:
        """Transform a signature through a gear"""
        if gear_name not in self.gears:
            raise ValueError(f"Unknown gear: {gear_name}")
        return self.gears[gear_name].transform_signature(signature)
    
    def store(self, key: SCoordinates, value: any):
        """Store value in S-dictionary memory"""
        self.memory.store(key, value)
    
    def load(self, key: SCoordinates) -> any:
        """Load value from S-dictionary memory"""
        return self.memory.retrieve(key)
    
    def get_stats(self) -> Dict:
        """Get ALU statistics"""
        return {
            "operations_executed": self.op_count,
            "transistors": len(self.transistors),
            "gates": list(self.gates.keys()),
            "gears": list(self.gears.keys()),
            "memory_entries": len(self.memory.entries),
            "accumulator": self.accumulator
        }


class GearNetwork:
    """
    A network of interconnected gears for frequency transformation.
    
    Implements: ω_therapeutic = G_pathway × ω_drug
    """
    
    def __init__(self):
        self.gears: Dict[str, GearRatio] = {}
        self.connections: Dict[str, List[str]] = {}  # gear -> connected gears
    
    def add_gear(self, name: str, input_teeth: int, output_teeth: int):
        """Add a gear to the network"""
        self.gears[name] = GearRatio(input_teeth, output_teeth)
        self.connections[name] = []
    
    def connect(self, gear_a: str, gear_b: str):
        """Connect two gears (output of a feeds input of b)"""
        if gear_a in self.connections:
            self.connections[gear_a].append(gear_b)
    
    def compute_pathway_ratio(self, path: List[str]) -> float:
        """Compute total gear ratio through a pathway"""
        total_ratio = 1.0
        for gear_name in path:
            if gear_name in self.gears:
                total_ratio *= self.gears[gear_name].ratio
        return total_ratio
    
    def transform_through_pathway(self, signature: OscillatorySignature,
                                  path: List[str]) -> OscillatorySignature:
        """Transform a signature through a gear pathway"""
        current = signature
        for gear_name in path:
            if gear_name in self.gears:
                current = self.gears[gear_name].transform_signature(current)
        return current
    
    def therapeutic_frequency(self, drug_signature: OscillatorySignature,
                             pathway: List[str]) -> float:
        """
        Compute therapeutic frequency from drug frequency.
        
        ω_therapeutic = G_pathway × ω_drug
        """
        G = self.compute_pathway_ratio(pathway)
        return G * drug_signature.frequency


def validate_alu() -> Dict:
    """Validate the biological ALU"""
    results = {}
    
    # Create ALU
    alu = BiologicalALU()
    
    # Test 1: Basic S-coordinate operations
    a = SCoordinates(1.0, 2.0, 3.0)
    b = SCoordinates(0.5, 1.0, 1.5)
    
    add_result = alu.execute(ALUOperation.ADD, a, b)
    results["add"] = (add_result.s_k, add_result.s_t, add_result.s_e)
    
    sub_result = alu.execute(ALUOperation.SUBTRACT, a, b)
    results["subtract"] = (sub_result.s_k, sub_result.s_t, sub_result.s_e)
    
    # Test 2: Logic gates
    and_result = alu.apply_gate("AND", a, b)
    results["AND_gate"] = (and_result.s_k, and_result.s_t, and_result.s_e)
    
    or_result = alu.apply_gate("OR", a, b)
    results["OR_gate"] = (or_result.s_k, or_result.s_t, or_result.s_e)
    
    not_result = alu.apply_gate("NOT", a)
    results["NOT_gate"] = (not_result.s_k, not_result.s_t, not_result.s_e)
    
    # Test 3: BMD Transistor
    gate_pattern = OscillatorySignature(1.0, 1e12, 0.0)
    transistor = alu.add_transistor(gate_pattern)
    
    matching_signal = OscillatorySignature(1.0, 1e12, 0.1)
    non_matching = OscillatorySignature(1.0, 1e13, 0.0)
    
    results["transistor_match"] = transistor.activate(matching_signal)
    results["transistor_amplification"] = transistor.amplify(1.0)
    
    transistor.activate(non_matching)
    results["transistor_no_match"] = not transistor.state
    
    # Test 4: Gear network
    gear_net = GearNetwork()
    gear_net.add_gear("g1", 10, 20)  # 2:1 ratio
    gear_net.add_gear("g2", 5, 15)   # 3:1 ratio
    gear_net.connect("g1", "g2")
    
    drug_sig = OscillatorySignature(1.0, 1e9, 0.0)
    pathway = ["g1", "g2"]
    
    results["pathway_ratio"] = gear_net.compute_pathway_ratio(pathway)
    results["therapeutic_freq"] = gear_net.therapeutic_frequency(drug_sig, pathway)
    
    # Test 5: S-dictionary memory
    alu.store(a, "state_A")
    alu.store(b, "state_B")
    
    results["memory_retrieve_a"] = alu.load(a)
    results["memory_retrieve_b"] = alu.load(b)
    
    # Fuzzy retrieval
    near_a = SCoordinates(1.01, 2.01, 3.01)
    results["fuzzy_retrieve"] = alu.load(near_a)
    
    # Validations
    results["validations"] = {
        "add_valid": add_result.s_k == 1.5 and add_result.s_t == 3.0,
        "transistor_valid": results["transistor_match"] == True,
        "gear_ratio_valid": abs(results["pathway_ratio"] - 6.0) < 0.01,
        "memory_valid": results["memory_retrieve_a"] == "state_A"
    }
    
    results["stats"] = alu.get_stats()
    
    return results


if __name__ == "__main__":
    results = validate_alu()
    print("Biological ALU Validation:")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("\nAll validations passed:", all(results["validations"].values()))

