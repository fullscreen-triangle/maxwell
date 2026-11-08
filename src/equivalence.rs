//! Equivalence Class Filter
//!
//! This module implements categorical operations on equivalence classes:
//! - Configuration equivalence
//! - Phase-lock degeneracy
//! - State space reduction
//! - Representative selection
//!
//! Multiple categorical states can be equivalent (same physical configuration),
//! and this module filters to canonical representatives.

use crate::types::*;
use std::collections::HashMap;

/// The Equivalence Class Filter
///
/// Filters categorical states to equivalence class representatives.
/// This implements the S-Dictionary memory structure where multiple
/// states map to the same categorical position.
#[derive(Debug, Clone)]
pub struct EquivalenceFilter {
    /// Tolerance for considering states equivalent
    tolerance: f64,
    
    /// Cached equivalence classes (representative_id -> member_ids)
    classes: HashMap<u64, Vec<u64>>,
}

impl EquivalenceFilter {
    pub fn new() -> Self {
        Self {
            tolerance: 0.1,
            classes: HashMap::new(),
        }
    }
    
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            classes: HashMap::new(),
        }
    }
    
    // ========================================================================
    // CONFIGURATION EQUIVALENCE
    // ========================================================================
    
    /// Check if two states are equivalent
    pub fn are_equivalent(&self, a: &CategoricalState, b: &CategoricalState) -> bool {
        a.coordinates.distance(&b.coordinates) < self.tolerance
    }
    
    /// Find the equivalence class for a state
    pub fn find_class<'a>(&self, state: &CategoricalState, states: &'a [CategoricalState]) -> Vec<&'a CategoricalState> {
        states.iter()
            .filter(|s| self.are_equivalent(state, s))
            .collect()
    }
    
    // ========================================================================
    // PHASE-LOCK DEGENERACY
    // ========================================================================
    
    /// Check if two states have degenerate phase-lock structure
    pub fn are_phase_lock_degenerate(&self, a: &CategoricalState, b: &CategoricalState) -> bool {
        // Same phase-lock count and similar coupling strengths
        if a.phase_locks.len() != b.phase_locks.len() {
            return false;
        }
        
        // Check if coupling strength distributions are similar
        let mut a_couplings: Vec<f64> = a.phase_locks.iter().map(|(_, c)| *c).collect();
        let mut b_couplings: Vec<f64> = b.phase_locks.iter().map(|(_, c)| *c).collect();
        
        a_couplings.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b_couplings.sort_by(|x, y| x.partial_cmp(y).unwrap());
        
        a_couplings.iter().zip(b_couplings.iter())
            .all(|(a, b)| (a - b).abs() < self.tolerance)
    }
    
    // ========================================================================
    // STATE SPACE REDUCTION
    // ========================================================================
    
    /// Filter states to equivalence class representatives
    pub fn filter(&self, states: &[CategoricalState]) -> Vec<CategoricalState> {
        let mut representatives = Vec::new();
        let mut seen = Vec::new();
        
        for state in states {
            // Check if this state is equivalent to any seen representative
            let is_duplicate = seen.iter().any(|rep: &CategoricalState| {
                self.are_equivalent(state, rep)
            });
            
            if !is_duplicate {
                representatives.push(state.clone());
                seen.push(state.clone());
            }
        }
        
        representatives
    }
    
    /// Compute state space reduction ratio
    pub fn reduction_ratio(&self, original_count: usize, reduced_count: usize) -> f64 {
        if original_count == 0 {
            0.0
        } else {
            1.0 - (reduced_count as f64 / original_count as f64)
        }
    }
    
    // ========================================================================
    // REPRESENTATIVE SELECTION
    // ========================================================================
    
    /// Select the canonical representative from an equivalence class
    /// Uses minimum categorical entropy as selection criterion
    pub fn select_representative(&self, class: &[CategoricalState]) -> Option<CategoricalState> {
        class.iter()
            .min_by(|a, b| {
                let entropy_a = self.categorical_entropy(a);
                let entropy_b = self.categorical_entropy(b);
                entropy_a.partial_cmp(&entropy_b).unwrap()
            })
            .cloned()
    }
    
    /// Compute categorical entropy for a state
    fn categorical_entropy(&self, state: &CategoricalState) -> f64 {
        // Entropy from phase-lock structure
        let phase_lock_entropy = if state.phase_locks.is_empty() {
            0.0
        } else {
            let total_coupling: f64 = state.phase_locks.iter().map(|(_, c)| c).sum();
            state.phase_locks.iter()
                .map(|(_, c)| {
                    let p = c / total_coupling;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum()
        };
        
        // Entropy from coordinate magnitude
        let coord_entropy = (state.coordinates.s_k.powi(2) 
            + state.coordinates.s_t.powi(2) 
            + state.coordinates.s_e.powi(2)).sqrt();
        
        phase_lock_entropy + coord_entropy
    }
    
    // ========================================================================
    // EQUIVALENCE CLASS OPERATIONS
    // ========================================================================
    
    /// Build all equivalence classes from a set of states
    pub fn build_classes(&mut self, states: &[CategoricalState]) -> HashMap<u64, Vec<u64>> {
        self.classes.clear();
        let mut assigned = vec![false; states.len()];
        
        for i in 0..states.len() {
            if assigned[i] {
                continue;
            }
            
            // This state becomes the representative
            let rep_id = states[i].id;
            let mut class_members = vec![rep_id];
            assigned[i] = true;
            
            // Find all equivalent states
            for j in (i + 1)..states.len() {
                if !assigned[j] && self.are_equivalent(&states[i], &states[j]) {
                    class_members.push(states[j].id);
                    assigned[j] = true;
                }
            }
            
            self.classes.insert(rep_id, class_members);
        }
        
        self.classes.clone()
    }
    
    /// Get the representative for a state
    pub fn get_representative(&self, state_id: u64) -> Option<u64> {
        for (rep, members) in &self.classes {
            if members.contains(&state_id) {
                return Some(*rep);
            }
        }
        None
    }
    
    /// Get all members of a class
    pub fn get_class_members(&self, representative_id: u64) -> Option<&Vec<u64>> {
        self.classes.get(&representative_id)
    }
    
    /// Get number of equivalence classes
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
    
    /// Get statistics about the equivalence classes
    pub fn class_statistics(&self) -> EquivalenceStats {
        let sizes: Vec<usize> = self.classes.values().map(|v| v.len()).collect();
        
        let total_states: usize = sizes.iter().sum();
        let num_classes = sizes.len();
        let avg_size = if num_classes > 0 { 
            total_states as f64 / num_classes as f64 
        } else { 
            0.0 
        };
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let min_size = sizes.iter().min().copied().unwrap_or(0);
        
        EquivalenceStats {
            total_states,
            num_classes,
            average_class_size: avg_size,
            max_class_size: max_size,
            min_class_size: min_size,
            reduction_ratio: if total_states > 0 {
                1.0 - (num_classes as f64 / total_states as f64)
            } else {
                0.0
            },
        }
    }
}

impl Default for EquivalenceFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about equivalence classes
#[derive(Debug, Clone)]
pub struct EquivalenceStats {
    pub total_states: usize,
    pub num_classes: usize,
    pub average_class_size: f64,
    pub max_class_size: usize,
    pub min_class_size: usize,
    pub reduction_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_equivalence_detection() {
        let filter = EquivalenceFilter::with_tolerance(0.1);
        
        let state1 = CategoricalState::new(1, SCoordinates::new(0.0, 0.0, 0.0));
        let state2 = CategoricalState::new(2, SCoordinates::new(0.05, 0.0, 0.0));
        let state3 = CategoricalState::new(3, SCoordinates::new(1.0, 0.0, 0.0));
        
        assert!(filter.are_equivalent(&state1, &state2));
        assert!(!filter.are_equivalent(&state1, &state3));
    }
    
    #[test]
    fn test_filtering() {
        let filter = EquivalenceFilter::with_tolerance(0.1);
        
        let states = vec![
            CategoricalState::new(1, SCoordinates::new(0.0, 0.0, 0.0)),
            CategoricalState::new(2, SCoordinates::new(0.05, 0.0, 0.0)), // Equivalent to 1
            CategoricalState::new(3, SCoordinates::new(1.0, 0.0, 0.0)),
            CategoricalState::new(4, SCoordinates::new(1.05, 0.0, 0.0)), // Equivalent to 3
        ];
        
        let representatives = filter.filter(&states);
        
        // Should have 2 representatives (one for each equivalence class)
        assert_eq!(representatives.len(), 2);
    }
    
    #[test]
    fn test_class_building() {
        let mut filter = EquivalenceFilter::with_tolerance(0.1);
        
        let states = vec![
            CategoricalState::new(1, SCoordinates::new(0.0, 0.0, 0.0)),
            CategoricalState::new(2, SCoordinates::new(0.05, 0.0, 0.0)),
            CategoricalState::new(3, SCoordinates::new(1.0, 0.0, 0.0)),
        ];
        
        let classes = filter.build_classes(&states);
        
        assert_eq!(classes.len(), 2);
        
        let stats = filter.class_statistics();
        assert_eq!(stats.total_states, 3);
        assert_eq!(stats.num_classes, 2);
    }
}

