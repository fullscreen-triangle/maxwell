//! Core types for the Maxwell Processor
//!
//! This module defines the fundamental data structures representing:
//! - Categorical states (S_k, S_t, S_e coordinates)
//! - Phase-lock networks
//! - Kinetic observables
//! - Complementarity structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The observable face of information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservableFace {
    /// Categorical face: phase-lock networks, topological navigation
    Categorical,
    /// Kinetic face: velocities, temperatures, thermodynamic observables
    Kinetic,
}

impl std::fmt::Display for ObservableFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObservableFace::Categorical => write!(f, "CATEGORICAL"),
            ObservableFace::Kinetic => write!(f, "KINETIC"),
        }
    }
}

/// S-entropy coordinates (S_k, S_t, S_e)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SCoordinates {
    /// Knowledge entropy (information dimension)
    pub s_k: f64,
    /// Temporal entropy (time dimension)
    pub s_t: f64,
    /// Evolutionary entropy (entropy dimension)
    pub s_e: f64,
}

impl SCoordinates {
    pub fn new(s_k: f64, s_t: f64, s_e: f64) -> Self {
        Self { s_k, s_t, s_e }
    }
    
    /// Compute the conjugate (back face) coordinates
    pub fn conjugate(&self) -> Self {
        Self {
            s_k: -self.s_k,
            s_t: -self.s_t,
            s_e: -self.s_e,
        }
    }
    
    /// Phase conjugate (inverts knowledge only)
    pub fn phase_conjugate(&self) -> Self {
        Self {
            s_k: -self.s_k,
            s_t: self.s_t,
            s_e: self.s_e,
        }
    }
    
    /// Euclidean distance in S-space
    pub fn distance(&self, other: &Self) -> f64 {
        let dk = self.s_k - other.s_k;
        let dt = self.s_t - other.s_t;
        let de = self.s_e - other.s_e;
        (dk * dk + dt * dt + de * de).sqrt()
    }
    
    /// Check if coordinates sum to zero (conjugate verification)
    pub fn sums_to_zero_with(&self, other: &Self, tolerance: f64) -> bool {
        let sum_k = (self.s_k + other.s_k).abs();
        let sum_t = (self.s_t + other.s_t).abs();
        let sum_e = (self.s_e + other.s_e).abs();
        sum_k < tolerance && sum_t < tolerance && sum_e < tolerance
    }
}

impl Default for SCoordinates {
    fn default() -> Self {
        Self { s_k: 0.0, s_t: 0.0, s_e: 0.0 }
    }
}

/// A categorical state in phase-lock space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalState {
    /// Unique identifier
    pub id: u64,
    /// S-entropy coordinates
    pub coordinates: SCoordinates,
    /// Phase-lock cluster membership
    pub cluster_id: Option<u64>,
    /// Completion status
    pub completed: bool,
    /// Accessible states (by ID)
    pub accessible: Vec<u64>,
    /// Phase-lock edges (node_id, coupling_strength)
    pub phase_locks: Vec<(u64, f64)>,
}

impl CategoricalState {
    pub fn new(id: u64, coordinates: SCoordinates) -> Self {
        Self {
            id,
            coordinates,
            cluster_id: None,
            completed: false,
            accessible: Vec::new(),
            phase_locks: Vec::new(),
        }
    }
    
    /// Mark this state as completed (irreversible)
    pub fn complete(&mut self) {
        self.completed = true;
    }
    
    /// Check if another state is accessible from this one
    pub fn can_access(&self, other_id: u64) -> bool {
        self.accessible.contains(&other_id)
    }
    
    /// Add a phase-lock edge
    pub fn add_phase_lock(&mut self, other_id: u64, coupling: f64) {
        self.phase_locks.push((other_id, coupling));
        if !self.accessible.contains(&other_id) {
            self.accessible.push(other_id);
        }
    }
}

/// A categorical operation (what actually happens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CategoricalOperation {
    /// Complete a categorical state
    Complete { state_id: u64 },
    /// Navigate along phase-lock topology
    Navigate { from: u64, to: u64 },
    /// Form a new phase-lock edge
    FormPhaseLock { node_a: u64, node_b: u64, coupling: f64 },
    /// Densify network (add edges)
    Densify { edges: Vec<(u64, u64, f64)> },
    /// Filter to equivalence class
    FilterEquivalence { representative_id: u64 },
}

/// Kinetic state (what Maxwell observed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticState {
    /// Molecular velocity (m/s)
    pub velocity: f64,
    /// Kinetic energy (J)
    pub kinetic_energy: f64,
    /// Position (for reference, not for phase-lock)
    pub position: [f64; 3],
    /// Apparent temperature contribution (K)
    pub temperature_contribution: f64,
}

impl KineticState {
    pub fn new(velocity: f64, position: [f64; 3]) -> Self {
        let mass = 1.0; // normalized mass
        let kinetic_energy = 0.5 * mass * velocity * velocity;
        Self {
            velocity,
            kinetic_energy,
            position,
            temperature_contribution: kinetic_energy, // proportional
        }
    }
    
    /// Classify as "fast" or "slow" relative to threshold
    pub fn classify(&self, threshold: f64) -> MoleculeClassification {
        if self.velocity > threshold {
            MoleculeClassification::Fast
        } else {
            MoleculeClassification::Slow
        }
    }
}

/// Maxwell's classification (what the demon would use)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoleculeClassification {
    Fast,
    Slow,
}

/// Projection of categorical state to kinetic face
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticProjection {
    /// The categorical state being projected
    pub source_state_id: u64,
    /// Apparent velocity distribution
    pub velocity_distribution: Vec<f64>,
    /// Apparent temperature
    pub apparent_temperature: f64,
    /// Apparent sorting (what Maxwell would see)
    pub apparent_sorting: Vec<MoleculeClassification>,
    /// Explanation of why this appears as "demon" behavior
    pub demon_appearance: String,
}

/// Explanation of why a categorical operation appears as demon behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemonExplanation {
    /// The categorical operation
    pub operation: String,
    /// What Maxwell would observe on the kinetic face
    pub kinetic_observation: String,
    /// Why it appears as intelligent sorting
    pub apparent_intelligence: String,
    /// The actual mechanism (no demon)
    pub actual_mechanism: String,
    /// Which of the seven dissolution arguments applies
    pub dissolution_argument: DissolutionArgument,
}

/// The seven arguments dissolving the demon
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DissolutionArgument {
    TemporalTriviality,
    PhaseLockTemperatureIndependence,
    RetrievalParadox,
    DissolutionOfObservation,
    DissolutionOfDecision,
    DissolutionOfSecondLaw,
    InformationComplementarity,
}

impl std::fmt::Display for DissolutionArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TemporalTriviality => write!(f, "Temporal Triviality: Fluctuations produce same result"),
            Self::PhaseLockTemperatureIndependence => write!(f, "Phase-Lock Temperature Independence: Same arrangement at any T"),
            Self::RetrievalParadox => write!(f, "Retrieval Paradox: Cannot outpace thermal equilibration"),
            Self::DissolutionOfObservation => write!(f, "Dissolution of Observation: Topology doesn't need measurement"),
            Self::DissolutionOfDecision => write!(f, "Dissolution of Decision: Pathways are automatic"),
            Self::DissolutionOfSecondLaw => write!(f, "Dissolution of Second Law: Categorical entropy increases"),
            Self::InformationComplementarity => write!(f, "Information Complementarity: Demon is projection of hidden face"),
        }
    }
}

/// Result of recursive completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    /// Number of states completed
    pub states_completed: usize,
    /// Depth reached
    pub depth_reached: usize,
    /// 3^k decomposition count
    pub decomposition_count: usize,
    /// Total entropy change
    pub entropy_change: f64,
    /// Completed state IDs
    pub completed_ids: Vec<u64>,
    /// Cascade path
    pub cascade_path: Vec<Vec<u64>>,
}

/// Phase-lock network node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseLockNode {
    pub id: u64,
    pub frequency: f64,      // Oscillatory frequency
    pub phase: f64,          // Current phase
    pub amplitude: f64,      // Oscillation amplitude
    pub molecular_type: MolecularType,
}

/// Type of molecule (affects phase-lock formation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MolecularType {
    Polar,
    NonPolar,
    Dipolar,
}

/// Phase-lock edge (coupling between nodes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseLockEdge {
    pub source: u64,
    pub target: u64,
    pub coupling_strength: f64,  // Van der Waals or dipole coupling
    pub interaction_type: InteractionType,
}

/// Type of intermolecular interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    VanDerWaals,      // ~r^-6 dependence
    DipoleDipole,     // ~r^-3 dependence
    InducedDipole,    // ~r^-4 dependence
    Vibrational,      // Frequency coupling
}

/// Configuration for the processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Number of molecules to simulate
    pub num_molecules: usize,
    /// Temperature (K)
    pub temperature: f64,
    /// Maximum phase-lock coupling distance
    pub coupling_distance: f64,
    /// Van der Waals coefficient
    pub vdw_coefficient: f64,
    /// Collision frequency (Hz)
    pub collision_frequency: f64,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            num_molecules: 1000,
            temperature: 300.0,      // Room temperature
            coupling_distance: 1e-9, // ~1 nm
            vdw_coefficient: 1e-77,  // J·m^6
            collision_frequency: 1e10, // 10 GHz typical for gases
            tolerance: 1e-10,
        }
    }
}

/// Statistics about the current state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    /// Current observable face
    pub observable_face: ObservableFace,
    /// Number of categorical states
    pub num_categorical_states: usize,
    /// Number of completed states
    pub num_completed_states: usize,
    /// Number of phase-lock edges
    pub num_phase_lock_edges: usize,
    /// Network density
    pub network_density: f64,
    /// Total categorical entropy
    pub categorical_entropy: f64,
    /// Total spatial entropy (if kinetic face observed)
    pub spatial_entropy: Option<f64>,
}

