//! Error types for the Maxwell Processor

use crate::types::ObservableFace;
use thiserror::Error;

/// Errors that can occur during processor operations
#[derive(Error, Debug)]
pub enum ProcessorError {
    /// Attempted to observe wrong face
    #[error("Cannot observe {expected} face: currently on {actual} face. Switch faces first.")]
    WrongFace {
        expected: ObservableFace,
        actual: ObservableFace,
    },
    
    /// Attempted to observe both faces simultaneously (complementarity violation)
    #[error("Complementarity violation: cannot observe both faces simultaneously")]
    ComplementarityViolation,
    
    /// State not found
    #[error("Categorical state {0} not found")]
    StateNotFound(u64),
    
    /// State already completed (irreversibility)
    #[error("State {0} already completed: categorical irreversibility prevents re-occupation")]
    StateAlreadyCompleted(u64),
    
    /// Invalid state transition
    #[error("Cannot transition from state {from} to state {to}: not accessible")]
    InvalidTransition { from: u64, to: u64 },
    
    /// Phase-lock network error
    #[error("Phase-lock network error: {0}")]
    PhaseLockError(String),
    
    /// Completion engine error
    #[error("Completion error: {0}")]
    CompletionError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    /// Equivalence class error
    #[error("Equivalence class error: {0}")]
    EquivalenceError(String),
}

/// Result type for processor operations
pub type ProcessorResult<T> = Result<T, ProcessorError>;

