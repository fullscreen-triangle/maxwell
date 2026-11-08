"""
Maxwell Validation Package
==========================

A comprehensive validation framework for the Maxwell Processor,
demonstrating the seven-fold dissolution of Maxwell's Demon.

Components:
- types: Core data structures (S-coordinates, states, etc.)
- semiconductor: Biological oscillatory semiconductor model
- alu: Biological ALU with tri-dimensional logic gates
- processor: Main complementarity-aware processor
- dissolution: Seven dissolution argument validators
- categorical: Categorical engine validation
- kinetic: Kinetic engine validation
- complementarity: Complementarity validation

The central thesis:
    THERE IS NO DEMON.
    What appears as intelligent sorting is categorical completion
    through phase-lock network topology.
"""

from .types import (
    ObservableFace,
    SCoordinates,
    OscillatorySignature,
    OscillatoryHole,
    MolecularCarrier,
    CategoricalState,
    KineticState,
    PNJunction,
    ProcessorConfig,
    CompletionResult,
    DemonExplanation,
    DissolutionArgument,
)

from .semiconductor import (
    SemiconductorSubstrate,
    BiologicalPNJunction,
    SemiconductorNetwork,
    validate_semiconductor_model,
)

from .alu import (
    BiologicalALU,
    BMDTransistor,
    TriDimensionalGate,
    GearRatio,
    GearNetwork,
    SDictionaryMemory,
    validate_alu,
)

from .processor import (
    MaxwellProcessor,
    CategoricalEngine,
    KineticEngine,
    ComplementarityManager,
    ProjectionExplainer,
    EquivalenceFilter,
    RecursiveCompletionEngine,
    ProcessorMode,
    run_full_validation,
)

from .dissolution import (
    DissolutionValidator,
    DissolutionResult,
)

from .quantum_gates import (
    OscillatoryQubit,
    QuantumGate,
    QuantumGateType,
    CNOTGate,
    BiologicalQuantumProcessor,
    validate_quantum_gates,
)

from .integrated_circuit import (
    TriDimensionalLogicGate,
    LogicDimension,
    GearRatioInterconnect,
    VirtualProcessorALU,
    ALUOperation,
    SevenChannelIO,
    IOChannel,
    ConsciousnessInterface,
    BiologicalIntegratedCircuit,
    validate_integrated_circuit,
)

from .results_manager import (
    ResultsManager,
    ExperimentResult,
    get_results_manager,
)

from .experiments import (
    MaxwellDemonExperiments,
    ExperimentConfig,
)

from .publication_figures import (
    generate_all_panel_figures,
    generate_publication_figures,
    create_panel_arg1_temporal_triviality,
    create_panel_arg2_temperature_independence,
    create_panel_arg3_retrieval_paradox,
    create_panel_arg4_dissolution_observation,
    create_panel_arg5_dissolution_decision,
    create_panel_arg6_dissolution_second_law,
    create_panel_arg7_information_complementarity,
)

from .maxwell_images import (
    MaxwellDemonFigureGenerator,
    generate_maxwell_figures,
)

from .processor_images import (
    ProcessorFigureGenerator,
    generate_processor_figures,
)

from .utils import (
    generate_configuration,
    assign_velocities,
    build_phase_lock_network,
    calculate_network_properties,
    calculate_kinetic_energy,
    calculate_temperature,
    calculate_entropy,
    identify_clusters,
    calculate_three_distances,
)

from .semiconductor_experiments import (
    SemiconductorValidationExperiments,
    IntegratedCircuitValidationExperiments,
    SemiconductorExperimentConfig,
    run_semiconductor_and_circuit_validation,
)

from .visualize_dissolution import (
    visualize_arg1_temporal_triviality,
    visualize_arg2_temperature_independence,
    visualize_arg3_retrieval_paradox,
    visualize_arg4_dissolution_observation,
    visualize_arg5_dissolution_decision,
    visualize_arg6_dissolution_second_law,
    visualize_arg7_information_complementarity,
    generate_all_dissolution_figures,
)

from .visualize_semiconductor import (
    visualize_pn_junction,
    visualize_hole_dynamics,
    visualize_recombination,
    visualize_bmd_transistor,
    visualize_logic_gates,
    visualize_complete_ic,
    generate_all_semiconductor_figures,
)

from .oscillator_processor_duality import (
    OscillatorProcessorDuality,
    VirtualFoundry,
    EntropyEndpointNavigator,
    OscillationState,
    ProcessorType,
    validate_oscillator_processor_duality,
)

from .categorical_measurement import (
    TransPlanckianMeasurement,
    HarmonicCoincidenceNetwork,
    MaxwellDemonDecomposition,
    CategoricalMeasurementOperator,
    validate_categorical_measurement,
)

from .unified_framework import (
    UnifiedTheoreticalFramework,
    TheoreticalDomain,
    validate_unified_framework,
)

from .visualize_unified_framework import (
    visualize_oscillator_processor_duality,
    visualize_trans_planckian,
    visualize_information_complementarity,
    visualize_theoretical_web,
    generate_all_unified_figures,
)

__version__ = "0.1.0"
__author__ = "Kundai Farai Sachikonye"

__all__ = [
    # Types
    "ObservableFace",
    "SCoordinates",
    "OscillatorySignature",
    "OscillatoryHole",
    "MolecularCarrier",
    "CategoricalState",
    "KineticState",
    "PNJunction",
    "ProcessorConfig",
    "CompletionResult",
    "DemonExplanation",
    "DissolutionArgument",

    # Semiconductor
    "SemiconductorSubstrate",
    "BiologicalPNJunction",
    "SemiconductorNetwork",
    "validate_semiconductor_model",

    # ALU
    "BiologicalALU",
    "BMDTransistor",
    "TriDimensionalGate",
    "GearRatio",
    "GearNetwork",
    "SDictionaryMemory",
    "validate_alu",

    # Processor
    "MaxwellProcessor",
    "CategoricalEngine",
    "KineticEngine",
    "ComplementarityManager",
    "ProjectionExplainer",
    "EquivalenceFilter",
    "RecursiveCompletionEngine",
    "ProcessorMode",
    "run_full_validation",

    # Dissolution
    "DissolutionValidator",
    "DissolutionResult",

    # Quantum Gates (from SSRN 5680582)
    "OscillatoryQubit",
    "QuantumGate",
    "QuantumGateType",
    "CNOTGate",
    "BiologicalQuantumProcessor",
    "validate_quantum_gates",

    # Integrated Circuit (from SSRN 5680570)
    "TriDimensionalLogicGate",
    "LogicDimension",
    "GearRatioInterconnect",
    "VirtualProcessorALU",
    "ALUOperation",
    "SevenChannelIO",
    "IOChannel",
    "ConsciousnessInterface",
    "BiologicalIntegratedCircuit",
    "validate_integrated_circuit",

    # Results Management
    "ResultsManager",
    "ExperimentResult",
    "get_results_manager",

    # Experiments
    "MaxwellDemonExperiments",
    "ExperimentConfig",

    # Publication Figures
    "generate_all_panel_figures",
    "generate_publication_figures",
    "create_panel_arg1_temporal_triviality",
    "create_panel_arg2_temperature_independence",
    "create_panel_arg3_retrieval_paradox",
    "create_panel_arg4_dissolution_observation",
    "create_panel_arg5_dissolution_decision",
    "create_panel_arg6_dissolution_second_law",
    "create_panel_arg7_information_complementarity",
    
    # Maxwell Images (from data)
    "MaxwellDemonFigureGenerator",
    "generate_maxwell_figures",
    
    # Processor Images (from data)
    "ProcessorFigureGenerator",
    "generate_processor_figures",

    # Utilities
    "generate_configuration",
    "assign_velocities",
    "build_phase_lock_network",
    "calculate_network_properties",
    "calculate_kinetic_energy",
    "calculate_temperature",
    "calculate_entropy",
    "identify_clusters",
    "calculate_three_distances",

    # Semiconductor & IC Experiments
    "SemiconductorValidationExperiments",
    "IntegratedCircuitValidationExperiments",
    "SemiconductorExperimentConfig",
    "run_semiconductor_and_circuit_validation",
    
    # Dissolution Visualizations
    "visualize_arg1_temporal_triviality",
    "visualize_arg2_temperature_independence",
    "visualize_arg3_retrieval_paradox",
    "visualize_arg4_dissolution_observation",
    "visualize_arg5_dissolution_decision",
    "visualize_arg6_dissolution_second_law",
    "visualize_arg7_information_complementarity",
    "generate_all_dissolution_figures",
    
    # Semiconductor/IC Visualizations
    "visualize_pn_junction",
    "visualize_hole_dynamics",
    "visualize_recombination",
    "visualize_bmd_transistor",
    "visualize_logic_gates",
    "visualize_complete_ic",
    "generate_all_semiconductor_figures",
    
    # Oscillator-Processor Duality
    "OscillatorProcessorDuality",
    "VirtualFoundry",
    "EntropyEndpointNavigator",
    "OscillationState",
    "ProcessorType",
    "validate_oscillator_processor_duality",
    
    # Categorical Measurement (Trans-Planckian)
    "TransPlanckianMeasurement",
    "HarmonicCoincidenceNetwork",
    "MaxwellDemonDecomposition",
    "CategoricalMeasurementOperator",
    "validate_categorical_measurement",
    
    # Unified Framework
    "UnifiedTheoreticalFramework",
    "TheoreticalDomain",
    "validate_unified_framework",
    
    # Unified Visualizations
    "visualize_oscillator_processor_duality",
    "visualize_trans_planckian",
    "visualize_information_complementarity",
    "visualize_theoretical_web",
    "generate_all_unified_figures",
]
