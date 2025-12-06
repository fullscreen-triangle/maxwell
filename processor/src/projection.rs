//! Projection Explainer
//!
//! This module explains WHY Maxwell saw a demon:
//! - Maps categorical dynamics to kinetic observables
//! - Shows how categorical completion "projects" as sorting
//! - Generates the "demon appearance" from hidden face dynamics
//! - Reveals the complementarity that caused the paradox

use crate::types::*;

/// The Projection Explainer
///
/// Explains how categorical operations appear when projected
/// onto the kinetic face. This is why Maxwell saw a demon:
/// he was observing the kinetic face while categorical dynamics
/// occurred on the hidden face.
#[derive(Debug, Clone)]
pub struct ProjectionExplainer {
    /// Verbosity level for explanations
    verbose: bool,
}

impl ProjectionExplainer {
    pub fn new() -> Self {
        Self { verbose: true }
    }
    
    pub fn with_verbosity(verbose: bool) -> Self {
        Self { verbose }
    }
    
    // ========================================================================
    // CATEGORICAL → KINETIC MAPPING
    // ========================================================================
    
    /// Project a categorical state to kinetic observables
    pub fn project(&self, state: &CategoricalState) -> KineticProjection {
        // Key insight: the same categorical state can have ANY velocity distribution
        // We generate a "representative" kinetic projection, but it's not unique
        
        let coords = &state.coordinates;
        
        // Generate pseudo-random but deterministic velocity distribution
        // based on categorical coordinates (for consistency)
        let seed = state.id as f64;
        let n_molecules = 100;
        let mut velocities = Vec::with_capacity(n_molecules);
        
        for i in 0..n_molecules {
            // Pseudo-random velocity (deterministic from seed)
            let v = ((seed + i as f64).sin().abs() * 1000.0) + 100.0;
            velocities.push(v);
        }
        
        // Compute apparent temperature
        let mean_v_sq: f64 = velocities.iter().map(|v| v * v).sum::<f64>() / n_molecules as f64;
        let apparent_temperature = mean_v_sq / 3.0; // Simplified
        
        // Classify molecules (what demon would see)
        let mean_v = velocities.iter().sum::<f64>() / n_molecules as f64;
        let apparent_sorting: Vec<MoleculeClassification> = velocities.iter()
            .map(|&v| if v > mean_v { MoleculeClassification::Fast } else { MoleculeClassification::Slow })
            .collect();
        
        let demon_appearance = self.explain_projection_as_demon(state);
        
        KineticProjection {
            source_state_id: state.id,
            velocity_distribution: velocities,
            apparent_temperature,
            apparent_sorting,
            demon_appearance,
        }
    }
    
    /// Explain why a projection looks like demon behavior
    fn explain_projection_as_demon(&self, state: &CategoricalState) -> String {
        format!(
            "Categorical state {} with coordinates ({:.2}, {:.2}, {:.2}) projects to \
             a velocity distribution that APPEARS sorted. The 'sorting' is an artifact \
             of observing the kinetic face: the categorical structure (phase-lock \
             network) already distinguishes molecules, but this distinction is \
             invisible on the kinetic face. Maxwell interpreted this hidden structure \
             as requiring an intelligent sorting agent.",
            state.id,
            state.coordinates.s_k,
            state.coordinates.s_t,
            state.coordinates.s_e
        )
    }
    
    // ========================================================================
    // "DEMON" APPEARANCE GENERATOR
    // ========================================================================
    
    /// Explain how a categorical operation appears as demon behavior
    pub fn explain_demon(&self, operation: &CategoricalOperation) -> DemonExplanation {
        match operation {
            CategoricalOperation::Complete { state_id } => {
                DemonExplanation {
                    operation: format!("Complete state {}", state_id),
                    kinetic_observation: "A molecule appears to 'know' which side to go to".to_string(),
                    apparent_intelligence: "The molecule seems to make a 'decision' about its destination".to_string(),
                    actual_mechanism: "Categorical completion follows phase-lock network topology. \
                        The 'decision' is following adjacency, not intelligence.".to_string(),
                    dissolution_argument: DissolutionArgument::DissolutionOfDecision,
                }
            }
            CategoricalOperation::Navigate { from, to } => {
                DemonExplanation {
                    operation: format!("Navigate {} → {}", from, to),
                    kinetic_observation: "Molecules appear to move in a coordinated, non-random way".to_string(),
                    apparent_intelligence: "Something seems to be 'guiding' the molecules".to_string(),
                    actual_mechanism: "Navigation follows phase-lock adjacency. The coordination \
                        is topological, not externally imposed.".to_string(),
                    dissolution_argument: DissolutionArgument::DissolutionOfObservation,
                }
            }
            CategoricalOperation::FormPhaseLock { node_a, node_b, coupling } => {
                DemonExplanation {
                    operation: format!("Form phase-lock between {} and {} (coupling: {:.2e})", 
                        node_a, node_b, coupling),
                    kinetic_observation: "Two molecules become correlated without apparent cause".to_string(),
                    apparent_intelligence: "Some force appears to be creating order".to_string(),
                    actual_mechanism: "Phase-locks form through Van der Waals and dipole forces, \
                        which depend on position, not velocity. No 'ordering agent' needed.".to_string(),
                    dissolution_argument: DissolutionArgument::PhaseLockTemperatureIndependence,
                }
            }
            CategoricalOperation::Densify { edges } => {
                DemonExplanation {
                    operation: format!("Densify network: {} edges", edges.len()),
                    kinetic_observation: "System becomes more 'ordered' over time".to_string(),
                    apparent_intelligence: "Entropy appears to decrease, suggesting external intervention".to_string(),
                    actual_mechanism: "Network densification INCREASES categorical entropy, \
                        even if spatial entropy appears to decrease. Total entropy increases.".to_string(),
                    dissolution_argument: DissolutionArgument::DissolutionOfSecondLaw,
                }
            }
            CategoricalOperation::FilterEquivalence { representative_id } => {
                DemonExplanation {
                    operation: format!("Filter to equivalence class (rep: {})", representative_id),
                    kinetic_observation: "Many configurations collapse to one 'selected' state".to_string(),
                    apparent_intelligence: "Something is 'choosing' the representative".to_string(),
                    actual_mechanism: "Equivalence class filtering is a compression operation, \
                        not selection. The representative is determined by topology, not choice.".to_string(),
                    dissolution_argument: DissolutionArgument::InformationComplementarity,
                }
            }
        }
    }
    
    // ========================================================================
    // APPARENT SORTING EXPLANATION
    // ========================================================================
    
    /// Explain why molecules appear sorted by temperature
    pub fn explain_apparent_sorting(&self, fast_count: usize, slow_count: usize) -> String {
        format!(
            "Apparent sorting: {} fast, {} slow molecules.\n\n\
             WHY THIS IS NOT DEMON SORTING:\n\
             1. Phase-lock clusters correlate with molecular properties\n\
             2. Molecular properties correlate with kinetic behavior\n\
             3. Therefore, phase-lock clusters correlate with kinetic behavior\n\
             4. BUT: the correlation is NOT causal - the same categorical \
                structure exists at any temperature\n\n\
             The 'sorted' appearance is the projection of categorical \
             structure onto the kinetic face, not evidence of sorting.",
            fast_count, slow_count
        )
    }
    
    // ========================================================================
    // SHADOW DYNAMICS VISUALIZATION
    // ========================================================================
    
    /// Generate a textual visualization of how hidden dynamics appear
    pub fn visualize_shadow_dynamics(&self, categorical_op: &CategoricalOperation) -> String {
        let explanation = self.explain_demon(categorical_op);
        
        format!(
            "╔══════════════════════════════════════════════════════════════════╗\n\
             ║                    SHADOW DYNAMICS VISUALIZATION                  ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ CATEGORICAL FACE (Hidden)         │ KINETIC FACE (Observed)      ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ Operation:                         │ Observation:                 ║\n\
             ║ {:<35} │ {:<28} ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ WHAT'S ACTUALLY HAPPENING:                                       ║\n\
             ║ {:<66} ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ WHY MAXWELL SAW A DEMON:                                         ║\n\
             ║ {:<66} ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ DISSOLUTION ARGUMENT:                                            ║\n\
             ║ {:<66} ║\n\
             ╚══════════════════════════════════════════════════════════════════╝",
            truncate(&explanation.operation, 35),
            truncate(&explanation.kinetic_observation, 28),
            truncate(&explanation.actual_mechanism, 66),
            truncate(&explanation.apparent_intelligence, 66),
            truncate(&explanation.dissolution_argument.to_string(), 66)
        )
    }
    
    // ========================================================================
    // THE SEVEN-FOLD DISSOLUTION
    // ========================================================================
    
    /// Get the applicable dissolution argument for a situation
    pub fn applicable_dissolution(&self, operation: &CategoricalOperation) -> Vec<DissolutionArgument> {
        match operation {
            CategoricalOperation::Complete { .. } => vec![
                DissolutionArgument::DissolutionOfDecision,
                DissolutionArgument::TemporalTriviality,
            ],
            CategoricalOperation::Navigate { .. } => vec![
                DissolutionArgument::DissolutionOfObservation,
                DissolutionArgument::InformationComplementarity,
            ],
            CategoricalOperation::FormPhaseLock { .. } => vec![
                DissolutionArgument::PhaseLockTemperatureIndependence,
            ],
            CategoricalOperation::Densify { .. } => vec![
                DissolutionArgument::DissolutionOfSecondLaw,
                DissolutionArgument::RetrievalParadox,
            ],
            CategoricalOperation::FilterEquivalence { .. } => vec![
                DissolutionArgument::InformationComplementarity,
            ],
        }
    }
    
    /// Explain the information complementarity principle
    pub fn explain_complementarity(&self) -> String {
        "INFORMATION COMPLEMENTARITY PRINCIPLE\n\n\
         Information has two conjugate faces that cannot be simultaneously observed:\n\n\
         • CATEGORICAL FACE: Phase-lock networks, topological navigation, \n\
           categorical completion, configuration dynamics\n\
           → This is what's actually happening\n\n\
         • KINETIC FACE: Velocity distributions, temperature measurements, \n\
           energy sorting, thermodynamic observables\n\
           → This is what Maxwell observed\n\n\
         These faces are like ammeter/voltmeter measurements:\n\
         - An ammeter measures current directly, calculates voltage\n\
         - A voltmeter measures voltage directly, calculates current\n\
         - They cannot both measure at the same point simultaneously\n\n\
         Similarly:\n\
         - Observing the kinetic face: categorical dynamics appear as 'demon' behavior\n\
         - Observing the categorical face: no demon needed, just topology\n\n\
         THE DEMON IS THE PROJECTION OF HIDDEN CATEGORICAL DYNAMICS \n\
         ONTO THE OBSERVABLE KINETIC FACE.".to_string()
    }
}

impl Default for ProjectionExplainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncate a string to a maximum length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_projection() {
        let explainer = ProjectionExplainer::new();
        let state = CategoricalState::new(1, SCoordinates::new(0.5, 0.3, 0.7));
        
        let projection = explainer.project(&state);
        
        assert_eq!(projection.source_state_id, 1);
        assert!(!projection.velocity_distribution.is_empty());
        assert!(!projection.demon_appearance.is_empty());
    }
    
    #[test]
    fn test_demon_explanation() {
        let explainer = ProjectionExplainer::new();
        let op = CategoricalOperation::Complete { state_id: 42 };
        
        let explanation = explainer.explain_demon(&op);
        
        assert!(!explanation.operation.is_empty());
        assert!(!explanation.kinetic_observation.is_empty());
        assert!(!explanation.actual_mechanism.is_empty());
    }
}

