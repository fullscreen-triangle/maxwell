"""
Comprehensive Tests for Maxwell Processor
==========================================

Tests all components and validates the seven-fold dissolution
of Maxwell's Demon.
"""

import pytest
import numpy as np
from maxwell_validation import (
    # Types
    SCoordinates,
    OscillatorySignature,
    OscillatoryHole,
    MolecularCarrier,
    CategoricalState,
    KineticState,
    ProcessorConfig,
    DissolutionArgument,
    ObservableFace,
    
    # Semiconductor
    SemiconductorSubstrate,
    BiologicalPNJunction,
    SemiconductorNetwork,
    
    # ALU
    BiologicalALU,
    BMDTransistor,
    TriDimensionalGate,
    GearNetwork,
    
    # Processor
    MaxwellProcessor,
    CategoricalEngine,
    KineticEngine,
    ComplementarityManager,
    
    # Dissolution
    DissolutionValidator,
)


class TestSCoordinates:
    """Test S-entropy coordinates"""
    
    def test_creation(self):
        coords = SCoordinates(1.0, 2.0, 3.0)
        assert coords.s_k == 1.0
        assert coords.s_t == 2.0
        assert coords.s_e == 3.0
    
    def test_conjugate(self):
        coords = SCoordinates(1.0, 2.0, 3.0)
        conj = coords.conjugate()
        assert conj.s_k == -1.0
        assert conj.s_t == -2.0
        assert conj.s_e == -3.0
    
    def test_conjugate_sum_to_zero(self):
        coords = SCoordinates(1.0, 2.0, 3.0)
        conj = coords.conjugate()
        assert coords.sums_to_zero_with(conj)
    
    def test_distance(self):
        a = SCoordinates(0.0, 0.0, 0.0)
        b = SCoordinates(1.0, 0.0, 0.0)
        assert abs(a.distance(b) - 1.0) < 1e-10
    
    def test_origin(self):
        origin = SCoordinates.origin()
        assert origin.s_k == 0.0
        assert origin.s_t == 0.0
        assert origin.s_e == 0.0


class TestOscillatorySignature:
    """Test oscillatory signatures"""
    
    def test_resonance(self):
        sig1 = OscillatorySignature(1.0, 1e12, 0.0)
        sig2 = OscillatorySignature(1.0, 1.05e12, 0.0)
        assert sig1.resonates_with(sig2, bandwidth=0.1)
    
    def test_no_resonance(self):
        sig1 = OscillatorySignature(1.0, 1e12, 0.0)
        sig2 = OscillatorySignature(1.0, 2e12, 0.0)
        assert not sig1.resonates_with(sig2, bandwidth=0.1)
    
    def test_overlap_integral(self):
        sig1 = OscillatorySignature(1.0, 1e12, 0.0)
        sig2 = OscillatorySignature(1.0, 1e12, 0.0)
        overlap = sig1.overlap_integral(sig2)
        assert overlap > 0.9  # Should be nearly 1 for identical signatures


class TestOscillatoryHole:
    """Test oscillatory holes (P-type carriers)"""
    
    def test_creation(self):
        sig = OscillatorySignature(1.0, 1e12, 0.0)
        hole = OscillatoryHole(id=0, missing_signature=sig)
        assert hole.mobility == 0.0123
    
    def test_drift_velocity(self):
        sig = OscillatorySignature(1.0, 1e12, 0.0)
        hole = OscillatoryHole(id=0, missing_signature=sig, mobility=0.01)
        v = hole.drift_velocity(1e6)
        assert v == 1e4


class TestSemiconductor:
    """Test biological semiconductor model"""
    
    def test_p_type_substrate(self):
        substrate = SemiconductorSubstrate()
        sig = OscillatorySignature(1.0, 1e12, 0.0)
        substrate.add_hole(sig)
        assert substrate.is_p_type
    
    def test_n_type_substrate(self):
        substrate = SemiconductorSubstrate()
        sig = OscillatorySignature(1.0, 1e12, 0.0)
        substrate.add_carrier(sig, molecular_mass=300.0, concentration=1e-3)
        # With high concentration, should be N-type
        assert len(substrate.carriers) == 1
    
    def test_junction_rectification(self):
        network = SemiconductorNetwork()
        network.create_p_substrate("p", n_holes=5)
        network.create_n_substrate("n", n_carriers=3)
        junction = network.create_junction("p", "n")
        
        ratio = junction.rectification_ratio(0.1)
        assert ratio > 1.0  # Forward should conduct more than reverse
    
    def test_junction_current(self):
        network = SemiconductorNetwork()
        network.create_p_substrate("p", n_holes=5)
        network.create_n_substrate("n", n_carriers=3)
        junction = network.create_junction("p", "n")
        
        i_forward = junction.current(0.1)
        i_reverse = junction.current(-0.1)
        assert i_forward > i_reverse


class TestBiologicalALU:
    """Test biological ALU"""
    
    def test_addition(self):
        alu = BiologicalALU()
        from maxwell_validation.alu import ALUOperation
        
        a = SCoordinates(1.0, 2.0, 3.0)
        b = SCoordinates(0.5, 1.0, 1.5)
        result = alu.execute(ALUOperation.ADD, a, b)
        
        assert result.s_k == 1.5
        assert result.s_t == 3.0
        assert result.s_e == 4.5
    
    def test_and_gate(self):
        alu = BiologicalALU()
        a = SCoordinates(1.0, 2.0, 3.0)
        b = SCoordinates(0.5, 3.0, 1.5)
        result = alu.apply_gate("AND", a, b)
        
        assert result.s_k == 0.5  # min
        assert result.s_t == 2.0  # min
        assert result.s_e == 1.5  # min
    
    def test_or_gate(self):
        alu = BiologicalALU()
        a = SCoordinates(1.0, 2.0, 3.0)
        b = SCoordinates(0.5, 3.0, 1.5)
        result = alu.apply_gate("OR", a, b)
        
        assert result.s_k == 1.0  # max
        assert result.s_t == 3.0  # max
        assert result.s_e == 3.0  # max
    
    def test_not_gate(self):
        alu = BiologicalALU()
        a = SCoordinates(1.0, 2.0, 3.0)
        result = alu.apply_gate("NOT", a)
        
        assert result.s_k == -1.0
        assert result.s_t == -2.0
        assert result.s_e == -3.0
    
    def test_bmd_transistor(self):
        alu = BiologicalALU()
        gate_pattern = OscillatorySignature(1.0, 1e12, 0.0)
        transistor = alu.add_transistor(gate_pattern)
        
        matching = OscillatorySignature(1.0, 1e12, 0.1)
        assert transistor.activate(matching) == True
        assert transistor.amplify(1.0) == 1000.0
    
    def test_gear_network(self):
        net = GearNetwork()
        net.add_gear("g1", 10, 20)
        net.add_gear("g2", 5, 15)
        
        ratio = net.compute_pathway_ratio(["g1", "g2"])
        assert abs(ratio - 6.0) < 0.01  # 2 * 3 = 6
    
    def test_s_dictionary_memory(self):
        alu = BiologicalALU()
        key = SCoordinates(1.0, 2.0, 3.0)
        alu.store(key, "test_value")
        
        retrieved = alu.load(key)
        assert retrieved == "test_value"


class TestCategoricalEngine:
    """Test categorical face engine"""
    
    def test_state_creation(self):
        engine = CategoricalEngine()
        coords = SCoordinates(0.5, 0.3, 0.7)
        state_id = engine.create_state(coords)
        
        assert state_id in engine.states
        assert engine.states[state_id].coordinates.s_k == 0.5
    
    def test_phase_lock_formation(self):
        engine = CategoricalEngine()
        id1 = engine.create_state(SCoordinates(0.0, 0.0, 0.0))
        id2 = engine.create_state(SCoordinates(0.1, 0.0, 0.0))
        
        engine.form_phase_lock(id1, id2, 1.0)
        
        assert engine.states[id1].can_access(id2)
        assert engine.states[id2].can_access(id1)
    
    def test_completion(self):
        engine = CategoricalEngine()
        state_id = engine.create_state(SCoordinates(0.0, 0.0, 0.0))
        
        engine.complete_state(state_id)
        assert engine.states[state_id].completed
        
        # Cannot complete again
        with pytest.raises(ValueError):
            engine.complete_state(state_id)
    
    def test_network_temperature_independence(self):
        """
        CRITICAL TEST: Network depends on position, NOT temperature.
        This is key to dissolving the demon.
        """
        positions = np.random.rand(20, 3) * 1e-6
        
        # Build at 300K
        config1 = ProcessorConfig(temperature=300.0)
        engine1 = CategoricalEngine(config1)
        engine1.construct_network(positions)
        edges1 = len(engine1.edges)
        
        # Build at 500K - SAME positions
        config2 = ProcessorConfig(temperature=500.0)
        engine2 = CategoricalEngine(config2)
        engine2.construct_network(positions)
        edges2 = len(engine2.edges)
        
        # Should be identical!
        assert edges1 == edges2


class TestKineticEngine:
    """Test kinetic face engine"""
    
    def test_maxwell_boltzmann(self):
        engine = KineticEngine()
        ids = engine.initialize_maxwell_boltzmann(100)
        
        assert len(ids) == 100
        assert engine.mean_velocity() > 0
    
    def test_demon_sorting(self):
        engine = KineticEngine()
        engine.initialize_maxwell_boltzmann(100)
        
        fast, slow = engine.demon_sorting()
        assert len(fast) + len(slow) == 100
    
    def test_retrieval_paradox(self):
        """
        Test that thermal equilibration prevents sustained sorting.
        """
        engine = KineticEngine()
        engine.initialize_maxwell_boltzmann(100)
        
        initial_fast, _ = engine.demon_sorting()
        initial_ratio = len(initial_fast) / 100
        
        # Run collision steps
        for _ in range(100):
            engine.collision_step()
        
        final_fast, _ = engine.demon_sorting()
        final_ratio = len(final_fast) / 100
        
        # Should stay around 0.5 (can't maintain sorting)
        assert 0.3 < final_ratio < 0.7


class TestComplementarity:
    """Test complementarity enforcement"""
    
    def test_face_switching(self):
        cat = CategoricalEngine()
        kin = KineticEngine()
        manager = ComplementarityManager(cat, kin)
        
        assert manager.current_face == ObservableFace.CATEGORICAL
        
        manager.switch_face()
        assert manager.current_face == ObservableFace.KINETIC
    
    def test_cannot_observe_both(self):
        cat = CategoricalEngine()
        kin = KineticEngine()
        manager = ComplementarityManager(cat, kin)
        
        # Initially on categorical
        assert manager.can_observe(ObservableFace.CATEGORICAL)
        assert not manager.can_observe(ObservableFace.KINETIC)
        
        manager.switch_face()
        assert not manager.can_observe(ObservableFace.CATEGORICAL)
        assert manager.can_observe(ObservableFace.KINETIC)


class TestMaxwellProcessor:
    """Test the full Maxwell Processor"""
    
    def test_initialization(self):
        processor = MaxwellProcessor()
        result = processor.initialize_system(50)
        
        assert result["categorical_states"] == 50
        assert result["kinetic_states"] == 50
    
    def test_no_demon_demonstration(self):
        processor = MaxwellProcessor()
        processor.initialize_system(50)
        
        result = processor.demonstrate_no_demon()
        
        assert result["temperature_independence"]["identical"]
        assert result["final_conclusion"] == "THE DEMON DOES NOT EXIST"


class TestDissolutionArguments:
    """Test all seven dissolution arguments"""
    
    @pytest.fixture
    def validator(self):
        return DissolutionValidator()
    
    def test_temporal_triviality(self, validator):
        result = validator.validate_temporal_triviality()
        assert result.validated
        assert result.argument == DissolutionArgument.TEMPORAL_TRIVIALITY
    
    def test_phase_lock_temperature_independence(self, validator):
        result = validator.validate_phase_lock_temperature_independence()
        assert result.validated
        assert result.argument == DissolutionArgument.PHASE_LOCK_TEMPERATURE_INDEPENDENCE
    
    def test_retrieval_paradox(self, validator):
        result = validator.validate_retrieval_paradox()
        assert result.validated
        assert result.argument == DissolutionArgument.RETRIEVAL_PARADOX
    
    def test_dissolution_of_observation(self, validator):
        result = validator.validate_dissolution_of_observation()
        assert result.validated
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_OBSERVATION
    
    def test_dissolution_of_decision(self, validator):
        result = validator.validate_dissolution_of_decision()
        assert result.validated
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_DECISION
    
    def test_dissolution_of_second_law(self, validator):
        result = validator.validate_dissolution_of_second_law()
        assert result.validated
        assert result.argument == DissolutionArgument.DISSOLUTION_OF_SECOND_LAW
    
    def test_information_complementarity(self, validator):
        result = validator.validate_information_complementarity()
        assert result.validated
        assert result.argument == DissolutionArgument.INFORMATION_COMPLEMENTARITY
    
    def test_all_seven_arguments(self, validator):
        """Validate all seven dissolution arguments"""
        results = validator.run_all_validations()
        
        assert len(results) == 7
        assert all(r.validated for r in results.values())


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from initialization to dissolution"""
        # Create processor
        config = ProcessorConfig(num_molecules=50, temperature=300.0)
        processor = MaxwellProcessor(config)
        
        # Initialize
        init = processor.initialize_system(50)
        assert init["categorical_states"] == 50
        
        # Switch modes
        processor.set_mode(ProcessorMode.CATEGORICAL)
        assert processor.mode == ProcessorMode.CATEGORICAL
        
        processor.set_mode(ProcessorMode.KINETIC)
        assert processor.mode == ProcessorMode.KINETIC
        
        # Get stats
        stats = processor.get_stats()
        assert "categorical" in stats
        assert "kinetic" in stats
        assert "semiconductor" in stats
        assert "alu" in stats
    
    def test_semiconductor_alu_integration(self):
        """Test semiconductor and ALU working together"""
        processor = MaxwellProcessor()
        processor.initialize_system(30)
        
        # ALU should be initialized
        assert processor.alu is not None
        
        # Semiconductor network should be initialized
        stats = processor.semiconductor.get_network_stats()
        assert stats["n_substrates"] > 0
    
    def test_phase_lock_network_validation(self):
        """Validate phase-lock network properties"""
        processor = MaxwellProcessor()
        processor.initialize_system(50)
        
        # Network should exist
        assert len(processor.categorical.edges) > 0
        
        # Density should be reasonable
        density = processor.categorical.network_density()
        assert 0 < density <= 1


# Import ProcessorMode for tests
from maxwell_validation.processor import ProcessorMode


class TestQuantumGates:
    """
    Test quantum logic gates in biological membranes.
    
    Based on: Sachikonye (2025) - SSRN 5680582
    "Derivation of Quantum Logic Gates in Biological Membranes"
    
    Key specifications:
    - 758 Hz clock frequency
    - 10 ms coherence times
    - <100 μs gate operation times
    - >85% gate fidelity
    """
    
    def test_oscillatory_qubit_creation(self):
        from maxwell_validation import OscillatoryQubit
        
        q0 = OscillatoryQubit.zero()
        assert abs(q0.probability_0 - 1.0) < 0.01
        
        q1 = OscillatoryQubit.one()
        assert abs(q1.probability_1 - 1.0) < 0.01
        
        qplus = OscillatoryQubit.plus()
        assert abs(qplus.probability_0 - 0.5) < 0.01
    
    def test_clock_frequency(self):
        from maxwell_validation import OscillatoryQubit
        
        qubit = OscillatoryQubit.zero()
        # Paper specifies 758 Hz
        assert qubit.frequency == 758.0
    
    def test_coherence_time(self):
        from maxwell_validation import OscillatoryQubit
        
        qubit = OscillatoryQubit.zero()
        # Paper specifies 10 ms coherence
        assert qubit.coherence_time == 0.010
        
        # Initially coherent
        assert qubit.is_coherent
        
        # After too much time, loses coherence
        qubit.time_since_refresh = 0.015
        assert not qubit.is_coherent
    
    def test_gate_operation_time(self):
        from maxwell_validation import QuantumGate, QuantumGateType
        
        gate = QuantumGate(QuantumGateType.HADAMARD)
        # Paper specifies <100 μs
        assert gate.operation_time < 100e-6
    
    def test_gate_fidelity(self):
        from maxwell_validation import QuantumGate, QuantumGateType
        
        gate = QuantumGate(QuantumGateType.HADAMARD)
        # Paper specifies >85%
        assert gate.fidelity >= 0.85
    
    def test_x_gate(self):
        from maxwell_validation import OscillatoryQubit, QuantumGate, QuantumGateType
        
        qubit = OscillatoryQubit.zero()
        x_gate = QuantumGate(QuantumGateType.X)
        x_gate.apply(qubit)
        
        # X gate flips |0⟩ to |1⟩
        assert qubit.probability_1 > 0.8
    
    def test_hadamard_gate(self):
        from maxwell_validation import OscillatoryQubit, QuantumGate, QuantumGateType
        
        qubit = OscillatoryQubit.zero()
        h_gate = QuantumGate(QuantumGateType.HADAMARD)
        h_gate.apply(qubit)
        
        # H gate creates superposition
        assert 0.4 < qubit.probability_0 < 0.6
        assert 0.4 < qubit.probability_1 < 0.6
    
    def test_cnot_gate(self):
        from maxwell_validation import OscillatoryQubit, CNOTGate
        
        control = OscillatoryQubit.one()
        target = OscillatoryQubit.zero()
        
        cnot = CNOTGate()
        cnot.apply(control, target)
        
        # CNOT with control=|1⟩ should flip target
        assert target.probability_1 > 0.5
    
    def test_biological_quantum_processor(self):
        from maxwell_validation import BiologicalQuantumProcessor
        
        processor = BiologicalQuantumProcessor()
        
        # Check specifications
        assert processor.clock_frequency == 758.0  # Hz
        assert processor.coherence_time == 0.010   # 10 ms
        assert processor.temperature == 310.0      # Physiological
        
        # Add qubits
        q0 = processor.add_qubit('0')
        q1 = processor.add_qubit('0')
        
        # Apply gates
        processor.apply_gate('H', q0)
        processor.apply_cnot(q0, q1)
        
        # Should create entanglement (Bell state)
        results = processor.measure_all()
        # In Bell state, both qubits should match
        assert results[0] == results[1]
    
    def test_second_law_compliance(self):
        from maxwell_validation import BiologicalQuantumProcessor
        
        processor = BiologicalQuantumProcessor()
        q = processor.add_qubit('0')
        
        for _ in range(10):
            processor.apply_gate('H', q)
            processor.apply_gate('T', q)
        
        # Paper claims 100% second law compliance
        assert processor.second_law_compliance > 0.99
    
    def test_landauer_efficiency(self):
        from maxwell_validation import BiologicalQuantumProcessor
        
        processor = BiologicalQuantumProcessor()
        q = processor.add_qubit('0')
        
        processor.apply_gate('X', q)
        processor.apply_gate('H', q)
        
        # Should be Landauer-optimal
        efficiency = processor.landauer_efficiency
        assert efficiency > 0  # Some efficiency
    
    def test_full_validation(self):
        from maxwell_validation import validate_quantum_gates
        
        results = validate_quantum_gates()
        assert results["all_passed"]


class TestIntegratedCircuit:
    """
    Test biological semiconductor junction oscillatory integrated logic circuits.
    
    Based on: Sachikonye (2025) - SSRN 5680570
    
    7 Components:
    1. BMD Transistors (42.1 on/off ratio, <1 μs switching)
    2. Tri-dimensional Logic Gates (AND/OR/XOR parallel)
    3. Gear Ratio Interconnects (23,500× speedup)
    4. S-Dictionary Memory (10^10 states, O(1) retrieval)
    5. Virtual Processor ALU (47 BMDs, <100 ns)
    6. Seven-Channel I/O (>10^12 bits/s)
    7. Consciousness Interface (242% fire-circle enhancement)
    """
    
    def test_bmd_transistor_specifications(self):
        from maxwell_validation import BMDTransistor
        from maxwell_validation.types import OscillatorySignature
        
        sig = OscillatorySignature(1.0, 1e12, 0.0)
        transistor = BMDTransistor(id=0, gate_signature=sig)
        
        # Paper specifies 42.1 on/off ratio
        assert abs(transistor.on_off_ratio - 42.1) < 0.1
        
        # Paper specifies <1 μs switching time
        assert transistor.switching_time < 1e-6
        
        # Probability enhancement 10^12
        assert transistor.probability_enhancement > 1e11
    
    def test_tri_dimensional_logic_gates(self):
        from maxwell_validation import TriDimensionalLogicGate, LogicDimension
        from maxwell_validation.types import SCoordinates
        
        gate = TriDimensionalLogicGate(id=0)
        a = SCoordinates(1.0, 0.5, 0.8)
        b = SCoordinates(0.3, 0.7, 0.4)
        
        results = gate.compute_all(a, b)
        
        # AND (minimum)
        and_result = results[LogicDimension.KNOWLEDGE]
        assert and_result.s_k == min(a.s_k, b.s_k)
        assert and_result.s_t == min(a.s_t, b.s_t)
        
        # OR (maximum)
        or_result = results[LogicDimension.TIME]
        assert or_result.s_k == max(a.s_k, b.s_k)
        assert or_result.s_t == max(a.s_t, b.s_t)
        
        # XOR (difference)
        xor_result = results[LogicDimension.ENTROPY]
        assert abs(xor_result.s_k - abs(a.s_k - b.s_k)) < 0.01
    
    def test_gear_ratio_interconnects(self):
        from maxwell_validation import GearRatioInterconnect
        
        interconnect = GearRatioInterconnect(id=0)
        
        # Paper specifies 23,500× speedup
        assert interconnect.speedup_factor == 23500
    
    def test_s_dictionary_memory(self):
        from maxwell_validation import SDictionaryMemory
        from maxwell_validation.types import SCoordinates
        
        memory = SDictionaryMemory()
        
        # Store and retrieve
        key = SCoordinates(0.5, 0.5, 0.5)
        memory.store(key, "test_value")
        
        retrieved = memory.retrieve(key)
        assert retrieved == "test_value"
        
        # Paper specifies 10^10 addressable states
        assert memory.addressable_states >= 1e10
    
    def test_virtual_processor_alu(self):
        from maxwell_validation import VirtualProcessorALU, ALUOperation
        from maxwell_validation.types import SCoordinates
        
        alu = VirtualProcessorALU()
        
        # Paper specifies 47 BMDs
        assert alu.bmd_count == 47
        
        # Paper specifies <100 ns operation
        assert alu.operation_time < 100e-9
        
        # Test operations
        a = SCoordinates(1.0, 2.0, 3.0)
        b = SCoordinates(0.5, 1.0, 1.5)
        
        result = alu.execute(ALUOperation.ADD, a, b)
        assert result.s_k == 1.5
        assert result.s_t == 3.0
    
    def test_seven_channel_io(self):
        from maxwell_validation import SevenChannelIO, IOChannel
        
        io = SevenChannelIO()
        
        # Paper specifies 7 channels
        assert len(io.channels) == 7
        
        # Paper specifies >10^12 bits/s bandwidth
        assert io.aggregate_bandwidth >= 1e12
        
        # All channels exist
        for channel in IOChannel:
            assert channel in io.channels
    
    def test_consciousness_interface(self):
        from maxwell_validation import ConsciousnessInterface
        from maxwell_validation.types import SCoordinates
        
        interface = ConsciousnessInterface()
        
        # Paper specifies 242% fire-circle enhancement
        assert abs(interface.fire_circle_enhancement - 2.42) < 0.01
        
        # Paper specifies 78% pathway efficiency
        assert abs(interface.pathway_efficiency - 0.78) < 0.01
        
        # Paper specifies 12 clinical coordinates
        assert interface.clinical_coordinates == 12
        
        # Test pathway navigation
        start = SCoordinates(0, 0, 0)
        target = SCoordinates(1, 1, 1)
        path, efficiency = interface.navigate_pathway(start, target)
        
        assert len(path) > 0
        assert efficiency > 0
    
    def test_complete_integrated_circuit(self):
        from maxwell_validation import BiologicalIntegratedCircuit
        from maxwell_validation.types import SCoordinates
        
        circuit = BiologicalIntegratedCircuit()
        circuit.build_standard_circuit()
        
        stats = circuit.get_stats()
        
        # Check all components present
        assert stats["transistors"] == 47
        assert stats["logic_gates"] == 10
        assert stats["interconnects"] == 100
        
        # Trans-Planckian precision
        assert circuit.timing_precision < 1e-40
        
        # ENAQT enhancement
        assert circuit.enaqt_enhancement == 0.24
    
    def test_circuit_pathway_duality(self):
        from maxwell_validation import BiologicalIntegratedCircuit
        from maxwell_validation.types import SCoordinates
        
        circuit = BiologicalIntegratedCircuit()
        
        # Paper specifies ||S_circuit - S_pathway|| < 0.1
        circuit_s = SCoordinates(0.5, 0.5, 0.5)
        pathway_s = SCoordinates(0.52, 0.48, 0.51)
        
        is_dual, distance = circuit.verify_circuit_pathway_duality(circuit_s, pathway_s)
        
        assert is_dual
        assert distance < 0.1
    
    def test_full_ic_validation(self):
        from maxwell_validation import validate_integrated_circuit
        
        results = validate_integrated_circuit()
        assert results["all_passed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
