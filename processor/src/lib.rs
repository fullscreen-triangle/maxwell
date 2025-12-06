//! # Maxwell Processor
//!
//! A Complementarity-Aware Processor for Categorical Phase-Lock Dynamics.
//!
//! This processor implements the theoretical framework from "Resolution of Maxwell's Demon
//! Through Categorical Phase-Lock Topology" by operating on both faces of information:
//!
//! - **Categorical Face**: Phase-lock networks, topological navigation, categorical completion
//! - **Kinetic Face**: Velocity distributions, temperature, thermodynamic observables
//!
//! The processor enforces complementarity: only one face can be observed at a time,
//! exactly as ammeter/voltmeter measurements in electrical circuits.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              COMPLEMENTARITY-AWARE PROCESSOR                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐           ┌─────────────────┐              │
//! │  │  CATEGORICAL    │◄─────────►│    KINETIC      │              │
//! │  │  FACE ENGINE    │           │  FACE ENGINE    │              │
//! │  │                 │           │                 │              │
//! │  │ • Phase-lock    │           │ • Velocities    │              │
//! │  │ • Topology      │           │ • Temperature   │              │
//! │  │ • Completion    │           │ • Energy        │              │
//! │  └────────┬────────┘           └────────┬────────┘              │
//! │           │                              │                       │
//! │           └──────────┬───────────────────┘                       │
//! │                      ▼                                           │
//! │           ┌─────────────────────┐                               │
//! │           │  COMPLEMENTARITY    │                               │
//! │           │     MANAGER         │                               │
//! │           │                     │                               │
//! │           │ • Face switching    │                               │
//! │           │ • Incompatibility   │                               │
//! │           │ • Projection        │                               │
//! │           └──────────┬──────────┘                               │
//! │                      ▼                                           │
//! │           ┌─────────────────────┐                               │
//! │           │    PROJECTION       │                               │
//! │           │    EXPLAINER        │                               │
//! │           │                     │                               │
//! │           │ • Cat → Kinetic     │                               │
//! │           │ • "Demon" generator │                               │
//! │           └─────────────────────┘                               │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod categorical;
pub mod kinetic;
pub mod complementarity;
pub mod projection;
pub mod equivalence;
pub mod completion;
pub mod types;
pub mod error;

// Re-exports
pub use categorical::CategoricalEngine;
pub use kinetic::KineticEngine;
pub use complementarity::ComplementarityManager;
pub use projection::ProjectionExplainer;
pub use equivalence::EquivalenceFilter;
pub use completion::RecursiveCompletionEngine;
pub use types::*;
pub use error::ProcessorError;

use std::sync::Arc;
use parking_lot::RwLock;

/// The main processor that coordinates all engines
#[derive(Debug)]
pub struct MaxwellProcessor {
    /// Current observable face
    observable_face: RwLock<ObservableFace>,
    
    /// Categorical face engine (ground truth)
    categorical: Arc<CategoricalEngine>,
    
    /// Kinetic face engine (observable projections)
    kinetic: Arc<KineticEngine>,
    
    /// Complementarity manager
    complementarity: ComplementarityManager,
    
    /// Projection explainer
    projection: ProjectionExplainer,
    
    /// Equivalence class filter
    equivalence: EquivalenceFilter,
    
    /// Recursive completion engine
    completion: RecursiveCompletionEngine,
}

impl MaxwellProcessor {
    /// Create a new processor with default configuration
    pub fn new() -> Self {
        let categorical = Arc::new(CategoricalEngine::new());
        let kinetic = Arc::new(KineticEngine::new());
        
        Self {
            observable_face: RwLock::new(ObservableFace::Categorical),
            categorical: categorical.clone(),
            kinetic: kinetic.clone(),
            complementarity: ComplementarityManager::new(categorical.clone(), kinetic.clone()),
            projection: ProjectionExplainer::new(),
            equivalence: EquivalenceFilter::new(),
            completion: RecursiveCompletionEngine::new(),
        }
    }
    
    /// Get the current observable face
    pub fn observable_face(&self) -> ObservableFace {
        *self.observable_face.read()
    }
    
    /// Switch the observable face (complementarity operation)
    pub fn switch_face(&self) -> Result<ObservableFace, ProcessorError> {
        let mut face = self.observable_face.write();
        *face = match *face {
            ObservableFace::Categorical => ObservableFace::Kinetic,
            ObservableFace::Kinetic => ObservableFace::Categorical,
        };
        Ok(*face)
    }
    
    /// Execute in categorical mode (ground truth operations)
    pub fn execute_categorical<F, R>(&self, f: F) -> Result<R, ProcessorError>
    where
        F: FnOnce(&CategoricalEngine) -> R,
    {
        let face = self.observable_face.read();
        if *face != ObservableFace::Categorical {
            return Err(ProcessorError::WrongFace {
                expected: ObservableFace::Categorical,
                actual: *face,
            });
        }
        Ok(f(&self.categorical))
    }
    
    /// Execute in kinetic mode (observable projections)
    pub fn execute_kinetic<F, R>(&self, f: F) -> Result<R, ProcessorError>
    where
        F: FnOnce(&KineticEngine) -> R,
    {
        let face = self.observable_face.read();
        if *face != ObservableFace::Kinetic {
            return Err(ProcessorError::WrongFace {
                expected: ObservableFace::Kinetic,
                actual: *face,
            });
        }
        Ok(f(&self.kinetic))
    }
    
    /// Project categorical state to kinetic face (shows why Maxwell saw a demon)
    pub fn project_to_kinetic(&self, state: &CategoricalState) -> KineticProjection {
        self.projection.project(state)
    }
    
    /// Explain how a categorical operation appears as "demon" behavior
    pub fn explain_demon_appearance(&self, operation: &CategoricalOperation) -> DemonExplanation {
        self.projection.explain_demon(operation)
    }
    
    /// Filter to equivalence class representative
    pub fn filter_equivalence(&self, states: &[CategoricalState]) -> Vec<CategoricalState> {
        self.equivalence.filter(states)
    }
    
    /// Execute recursive 3^k completion
    pub fn complete_recursively(&self, initial: &CategoricalState, depth: usize) -> CompletionResult {
        self.completion.complete(initial, depth)
    }
    
    /// Get the complementarity manager for advanced operations
    pub fn complementarity(&self) -> &ComplementarityManager {
        &self.complementarity
    }
}

impl Default for MaxwellProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_face_switching() {
        let processor = MaxwellProcessor::new();
        assert_eq!(processor.observable_face(), ObservableFace::Categorical);
        
        processor.switch_face().unwrap();
        assert_eq!(processor.observable_face(), ObservableFace::Kinetic);
        
        processor.switch_face().unwrap();
        assert_eq!(processor.observable_face(), ObservableFace::Categorical);
    }
    
    #[test]
    fn test_complementarity_enforcement() {
        let processor = MaxwellProcessor::new();
        
        // Should succeed on categorical face
        let result = processor.execute_categorical(|_| 42);
        assert!(result.is_ok());
        
        // Should fail on kinetic face when in categorical mode
        let result = processor.execute_kinetic(|_| 42);
        assert!(result.is_err());
    }
}

