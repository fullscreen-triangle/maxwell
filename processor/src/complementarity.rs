//! Complementarity Manager
//!
//! This module enforces the fundamental complementarity constraint:
//! only one face of information can be observed at a time.
//!
//! Analogous to ammeter/voltmeter measurement incompatibility:
//! - You can measure current (ammeter) OR voltage (voltmeter)
//! - You cannot measure both simultaneously at the same point
//! - Same for categorical vs kinetic faces

use crate::types::*;
use crate::error::{ProcessorError, ProcessorResult};
use crate::categorical::CategoricalEngine;
use crate::kinetic::KineticEngine;
use std::sync::Arc;
use parking_lot::RwLock;

/// The Complementarity Manager
///
/// Enforces that only one face can be observed at a time.
/// Provides translation between faces via derivation (not simultaneous observation).
#[derive(Debug)]
pub struct ComplementarityManager {
    /// Current observable face
    current_face: RwLock<ObservableFace>,
    
    /// Reference to categorical engine
    categorical: Arc<CategoricalEngine>,
    
    /// Reference to kinetic engine
    kinetic: Arc<KineticEngine>,
    
    /// Face switch count (for debugging/analysis)
    switch_count: RwLock<u64>,
}

impl ComplementarityManager {
    /// Create a new complementarity manager
    pub fn new(categorical: Arc<CategoricalEngine>, kinetic: Arc<KineticEngine>) -> Self {
        Self {
            current_face: RwLock::new(ObservableFace::Categorical),
            categorical,
            kinetic,
            switch_count: RwLock::new(0),
        }
    }
    
    // ========================================================================
    // FACE SWITCHING
    // ========================================================================
    
    /// Get the current observable face
    pub fn current_face(&self) -> ObservableFace {
        *self.current_face.read()
    }
    
    /// Switch to the other face
    /// This is analogous to switching from ammeter to voltmeter
    pub fn switch_face(&self) -> ObservableFace {
        let mut face = self.current_face.write();
        *face = match *face {
            ObservableFace::Categorical => ObservableFace::Kinetic,
            ObservableFace::Kinetic => ObservableFace::Categorical,
        };
        *self.switch_count.write() += 1;
        *face
    }
    
    /// Set a specific face
    pub fn set_face(&self, face: ObservableFace) {
        let mut current = self.current_face.write();
        if *current != face {
            *self.switch_count.write() += 1;
        }
        *current = face;
    }
    
    /// Get switch count
    pub fn switch_count(&self) -> u64 {
        *self.switch_count.read()
    }
    
    // ========================================================================
    // INCOMPATIBILITY ENFORCEMENT
    // ========================================================================
    
    /// Check if a face is currently observable
    pub fn is_observable(&self, face: ObservableFace) -> bool {
        *self.current_face.read() == face
    }
    
    /// Attempt to observe a face, fail if wrong face
    pub fn observe(&self, face: ObservableFace) -> ProcessorResult<()> {
        let current = *self.current_face.read();
        if current != face {
            return Err(ProcessorError::WrongFace {
                expected: face,
                actual: current,
            });
        }
        Ok(())
    }
    
    /// Attempt to observe both faces simultaneously (always fails)
    pub fn observe_both(&self) -> ProcessorResult<()> {
        Err(ProcessorError::ComplementarityViolation)
    }
    
    // ========================================================================
    // PROJECTION CALCULATION
    // ========================================================================
    
    /// Derive kinetic observables from categorical state
    /// This is like calculating V = IR when measuring I with ammeter
    pub fn derive_kinetic(&self, state: &CategoricalState) -> KineticState {
        // Categorical state doesn't determine velocity, but we can
        // generate a consistent kinetic representation
        
        // The key insight: this is a DERIVED value, not a measurement
        // The kinetic state is computed from the categorical coordinates
        // but doesn't represent "what's actually there"
        
        let coords = &state.coordinates;
        
        // Generate representative kinetic state
        // (This is fundamentally arbitrary - same categorical state
        //  can correspond to any temperature, per our theorem)
        let position = [coords.s_k, coords.s_t, coords.s_e];
        let velocity = 0.0; // No velocity information in categorical state!
        
        KineticState::new(velocity, position)
    }
    
    /// Derive categorical structure from kinetic observations
    /// This is like calculating I = V/R when measuring V with voltmeter
    pub fn derive_categorical(&self, state: &KineticState) -> CategoricalState {
        // The kinetic state doesn't determine categorical structure
        // Phase-lock networks are independent of velocity!
        
        // Generate representative categorical state
        let coords = SCoordinates::new(
            state.position[0],
            state.position[1],
            state.position[2],
        );
        
        CategoricalState::new(0, coords)
    }
    
    // ========================================================================
    // TRANSLATION BETWEEN FACES
    // ========================================================================
    
    /// Get what's observable on the current face
    pub fn observe_current(&self) -> FaceObservation {
        match *self.current_face.read() {
            ObservableFace::Categorical => {
                let states = self.categorical.all_states();
                let (nodes, edges, density) = self.categorical.network_stats();
                let entropy = self.categorical.categorical_entropy();
                
                FaceObservation::Categorical {
                    states,
                    network_nodes: nodes,
                    network_edges: edges,
                    network_density: density,
                    categorical_entropy: entropy,
                }
            }
            ObservableFace::Kinetic => {
                let velocities = self.kinetic.velocity_distribution();
                let temperature = self.kinetic.get_temperature();
                let mean_v = self.kinetic.mean_velocity();
                let (fast, slow) = self.kinetic.demon_sorting();
                
                FaceObservation::Kinetic {
                    velocities,
                    temperature,
                    mean_velocity: mean_v,
                    fast_count: fast.len(),
                    slow_count: slow.len(),
                }
            }
        }
    }
    
    /// Get what's hidden on the current face (must be derived, not observed)
    pub fn derive_hidden(&self) -> FaceDerivation {
        match *self.current_face.read() {
            ObservableFace::Categorical => {
                // Kinetic face is hidden, derive it
                FaceDerivation::Kinetic {
                    derived_temperature: self.kinetic.get_temperature(),
                    derived_from: "categorical structure".to_string(),
                    derivation_method: "T = emergent from phase-lock clusters".to_string(),
                    warning: "Derived, not measured. Same categorical state exists at any T.".to_string(),
                }
            }
            ObservableFace::Kinetic => {
                // Categorical face is hidden, derive it
                let (nodes, edges, density) = self.categorical.network_stats();
                FaceDerivation::Categorical {
                    derived_network_density: density,
                    derived_edges: edges,
                    derived_from: "kinetic observations".to_string(),
                    derivation_method: "Phase-lock topology from position correlations".to_string(),
                    warning: "Derived, not measured. Kinetic energy doesn't determine topology.".to_string(),
                }
            }
        }
    }
}

/// What's observable on a face
#[derive(Debug, Clone)]
pub enum FaceObservation {
    Categorical {
        states: Vec<CategoricalState>,
        network_nodes: usize,
        network_edges: usize,
        network_density: f64,
        categorical_entropy: f64,
    },
    Kinetic {
        velocities: Vec<f64>,
        temperature: f64,
        mean_velocity: f64,
        fast_count: usize,
        slow_count: usize,
    },
}

/// What's derived (not directly observed) on the hidden face
#[derive(Debug, Clone)]
pub enum FaceDerivation {
    Categorical {
        derived_network_density: f64,
        derived_edges: usize,
        derived_from: String,
        derivation_method: String,
        warning: String,
    },
    Kinetic {
        derived_temperature: f64,
        derived_from: String,
        derivation_method: String,
        warning: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_face_switching() {
        let cat = Arc::new(CategoricalEngine::new());
        let kin = Arc::new(KineticEngine::new());
        let manager = ComplementarityManager::new(cat, kin);
        
        assert_eq!(manager.current_face(), ObservableFace::Categorical);
        
        manager.switch_face();
        assert_eq!(manager.current_face(), ObservableFace::Kinetic);
        
        manager.switch_face();
        assert_eq!(manager.current_face(), ObservableFace::Categorical);
        
        assert_eq!(manager.switch_count(), 2);
    }
    
    #[test]
    fn test_complementarity_violation() {
        let cat = Arc::new(CategoricalEngine::new());
        let kin = Arc::new(KineticEngine::new());
        let manager = ComplementarityManager::new(cat, kin);
        
        // Cannot observe both faces
        assert!(manager.observe_both().is_err());
    }
    
    #[test]
    fn test_wrong_face_observation() {
        let cat = Arc::new(CategoricalEngine::new());
        let kin = Arc::new(KineticEngine::new());
        let manager = ComplementarityManager::new(cat, kin);
        
        // Default is categorical
        assert!(manager.observe(ObservableFace::Categorical).is_ok());
        assert!(manager.observe(ObservableFace::Kinetic).is_err());
    }
}

