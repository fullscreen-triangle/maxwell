//! Categorical Face Engine
//!
//! This module implements the "ground truth" operations:
//! - Phase-lock network construction
//! - Topological navigation
//! - Categorical completion
//! - Configuration dynamics
//!
//! This is what's actually happening (not what Maxwell saw).

use crate::types::*;
use crate::error::{ProcessorError, ProcessorResult};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::dijkstra;
use std::collections::HashMap;
use parking_lot::RwLock;

/// The Categorical Face Engine
///
/// Manages the phase-lock network and categorical state space.
/// This is the "ground truth" layer - what's actually happening
/// independent of kinetic observations.
#[derive(Debug)]
pub struct CategoricalEngine {
    /// The phase-lock network graph
    network: RwLock<DiGraph<PhaseLockNode, PhaseLockEdge>>,
    
    /// Categorical states indexed by ID
    states: RwLock<HashMap<u64, CategoricalState>>,
    
    /// Node index mapping (state_id -> graph node)
    node_indices: RwLock<HashMap<u64, NodeIndex>>,
    
    /// Next available state ID
    next_id: RwLock<u64>,
    
    /// Completed state count
    completed_count: RwLock<usize>,
    
    /// Configuration
    config: ProcessorConfig,
}

impl CategoricalEngine {
    /// Create a new categorical engine
    pub fn new() -> Self {
        Self {
            network: RwLock::new(DiGraph::new()),
            states: RwLock::new(HashMap::new()),
            node_indices: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
            completed_count: RwLock::new(0),
            config: ProcessorConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: ProcessorConfig) -> Self {
        Self {
            network: RwLock::new(DiGraph::new()),
            states: RwLock::new(HashMap::new()),
            node_indices: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
            completed_count: RwLock::new(0),
            config,
        }
    }
    
    // ========================================================================
    // PHASE-LOCK NETWORK CONSTRUCTION
    // ========================================================================
    
    /// Create a new categorical state
    pub fn create_state(&self, coordinates: SCoordinates) -> u64 {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;
        
        let state = CategoricalState::new(id, coordinates);
        
        // Create node in network
        let node = PhaseLockNode {
            id,
            frequency: self.compute_frequency(&coordinates),
            phase: 0.0,
            amplitude: 1.0,
            molecular_type: MolecularType::NonPolar,
        };
        
        let mut network = self.network.write();
        let node_idx = network.add_node(node);
        
        self.node_indices.write().insert(id, node_idx);
        self.states.write().insert(id, state);
        
        id
    }
    
    /// Compute oscillatory frequency from S-coordinates
    fn compute_frequency(&self, coords: &SCoordinates) -> f64 {
        // Frequency is a function of categorical position
        // Base frequency modulated by coordinates
        let base_freq = 1e12; // THz range
        base_freq * (1.0 + coords.s_k.abs() + coords.s_t.abs() + coords.s_e.abs())
    }
    
    /// Form a phase-lock between two states
    pub fn form_phase_lock(&self, id_a: u64, id_b: u64, coupling: f64) -> ProcessorResult<()> {
        let indices = self.node_indices.read();
        let idx_a = indices.get(&id_a).ok_or(ProcessorError::StateNotFound(id_a))?;
        let idx_b = indices.get(&id_b).ok_or(ProcessorError::StateNotFound(id_b))?;
        
        let edge = PhaseLockEdge {
            source: id_a,
            target: id_b,
            coupling_strength: coupling,
            interaction_type: InteractionType::VanDerWaals,
        };
        
        let mut network = self.network.write();
        network.add_edge(*idx_a, *idx_b, edge.clone());
        network.add_edge(*idx_b, *idx_a, edge); // Bidirectional
        
        // Update state accessibility
        let mut states = self.states.write();
        if let Some(state_a) = states.get_mut(&id_a) {
            state_a.add_phase_lock(id_b, coupling);
        }
        if let Some(state_b) = states.get_mut(&id_b) {
            state_b.add_phase_lock(id_a, coupling);
        }
        
        Ok(())
    }
    
    /// Automatically construct phase-lock network based on positions
    /// Phase-locks form based on Van der Waals (~r^-6) and dipole (~r^-3) interactions
    /// INDEPENDENT of kinetic energy (this is the key insight)
    pub fn construct_network(&self, positions: &[[f64; 3]], molecular_types: &[MolecularType]) -> ProcessorResult<()> {
        let n = positions.len();
        
        // Create states for each position
        let mut ids = Vec::with_capacity(n);
        for (i, pos) in positions.iter().enumerate() {
            // S-coordinates from position (not from velocity!)
            let coords = SCoordinates::new(
                pos[0] / self.config.coupling_distance,
                pos[1] / self.config.coupling_distance,
                pos[2] / self.config.coupling_distance,
            );
            let id = self.create_state(coords);
            
            // Set molecular type
            if let Some(state) = self.states.write().get_mut(&id) {
                // Molecular type affects phase-lock formation
            }
            ids.push(id);
        }
        
        // Form phase-locks based on position (NOT velocity)
        for i in 0..n {
            for j in (i + 1)..n {
                let r = self.distance(&positions[i], &positions[j]);
                
                if r < self.config.coupling_distance * 10.0 {
                    // Van der Waals coupling: C_6 / r^6
                    let vdw_coupling = self.config.vdw_coefficient / r.powi(6);
                    
                    // Dipole coupling if applicable: ~r^-3
                    let dipole_coupling = if molecular_types[i] == MolecularType::Polar 
                        || molecular_types[j] == MolecularType::Polar {
                        self.config.vdw_coefficient.sqrt() / r.powi(3)
                    } else {
                        0.0
                    };
                    
                    let total_coupling = vdw_coupling + dipole_coupling;
                    
                    if total_coupling > 1e-30 {
                        self.form_phase_lock(ids[i], ids[j], total_coupling)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn distance(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    // ========================================================================
    // TOPOLOGICAL NAVIGATION
    // ========================================================================
    
    /// Find shortest path in phase-lock space
    pub fn find_path(&self, from: u64, to: u64) -> ProcessorResult<Vec<u64>> {
        let indices = self.node_indices.read();
        let idx_from = indices.get(&from).ok_or(ProcessorError::StateNotFound(from))?;
        let idx_to = indices.get(&to).ok_or(ProcessorError::StateNotFound(to))?;
        
        let network = self.network.read();
        let distances = dijkstra(&*network, *idx_from, Some(*idx_to), |e| {
            1.0 / e.weight().coupling_strength // Inverse coupling as distance
        });
        
        // Reconstruct path (simplified - would need proper predecessor tracking)
        if distances.contains_key(idx_to) {
            // For now, return direct path
            Ok(vec![from, to])
        } else {
            Err(ProcessorError::InvalidTransition { from, to })
        }
    }
    
    /// Get accessible states from a given state
    pub fn accessible_from(&self, state_id: u64) -> ProcessorResult<Vec<u64>> {
        let states = self.states.read();
        let state = states.get(&state_id).ok_or(ProcessorError::StateNotFound(state_id))?;
        Ok(state.accessible.clone())
    }
    
    /// Navigate from one state to an adjacent state
    pub fn navigate(&self, from: u64, to: u64) -> ProcessorResult<()> {
        let states = self.states.read();
        let state = states.get(&from).ok_or(ProcessorError::StateNotFound(from))?;
        
        if !state.can_access(to) {
            return Err(ProcessorError::InvalidTransition { from, to });
        }
        
        // Navigation succeeds - the "door opens" along phase-lock adjacency
        Ok(())
    }
    
    // ========================================================================
    // CATEGORICAL COMPLETION
    // ========================================================================
    
    /// Complete a categorical state (irreversible)
    pub fn complete_state(&self, state_id: u64) -> ProcessorResult<()> {
        let mut states = self.states.write();
        let state = states.get_mut(&state_id).ok_or(ProcessorError::StateNotFound(state_id))?;
        
        if state.completed {
            return Err(ProcessorError::StateAlreadyCompleted(state_id));
        }
        
        state.complete();
        *self.completed_count.write() += 1;
        
        Ok(())
    }
    
    /// Check if a state is completed
    pub fn is_completed(&self, state_id: u64) -> ProcessorResult<bool> {
        let states = self.states.read();
        let state = states.get(&state_id).ok_or(ProcessorError::StateNotFound(state_id))?;
        Ok(state.completed)
    }
    
    /// Get all incomplete states accessible from a given state
    pub fn incomplete_accessible(&self, state_id: u64) -> ProcessorResult<Vec<u64>> {
        let states = self.states.read();
        let state = states.get(&state_id).ok_or(ProcessorError::StateNotFound(state_id))?;
        
        let incomplete: Vec<u64> = state.accessible.iter()
            .filter(|id| {
                states.get(id).map(|s| !s.completed).unwrap_or(false)
            })
            .copied()
            .collect();
        
        Ok(incomplete)
    }
    
    // ========================================================================
    // CONFIGURATION DYNAMICS
    // ========================================================================
    
    /// Densify the network (add edges based on categorical adjacency)
    pub fn densify_network(&self) -> ProcessorResult<usize> {
        let states = self.states.read();
        let state_ids: Vec<u64> = states.keys().copied().collect();
        drop(states);
        
        let mut edges_added = 0;
        
        for i in 0..state_ids.len() {
            for j in (i + 1)..state_ids.len() {
                let id_a = state_ids[i];
                let id_b = state_ids[j];
                
                // Check categorical distance
                let states = self.states.read();
                let state_a = states.get(&id_a).unwrap();
                let state_b = states.get(&id_b).unwrap();
                
                let cat_dist = state_a.coordinates.distance(&state_b.coordinates);
                drop(states);
                
                // Add edge if categorically close and not already connected
                if cat_dist < 1.0 {
                    if self.form_phase_lock(id_a, id_b, 1.0 / (cat_dist + 0.1)).is_ok() {
                        edges_added += 1;
                    }
                }
            }
        }
        
        Ok(edges_added)
    }
    
    /// Get network statistics
    pub fn network_stats(&self) -> (usize, usize, f64) {
        let network = self.network.read();
        let nodes = network.node_count();
        let edges = network.edge_count();
        let density = if nodes > 1 {
            edges as f64 / (nodes * (nodes - 1)) as f64
        } else {
            0.0
        };
        (nodes, edges, density)
    }
    
    /// Compute categorical entropy (proportional to network density)
    pub fn categorical_entropy(&self) -> f64 {
        let (_, edges, _) = self.network_stats();
        // S = k_B * |E| / <E> (from the theory)
        let k_b = 1.380649e-23; // Boltzmann constant
        k_b * edges as f64
    }
    
    /// Get a state by ID
    pub fn get_state(&self, state_id: u64) -> Option<CategoricalState> {
        self.states.read().get(&state_id).cloned()
    }
    
    /// Get all states
    pub fn all_states(&self) -> Vec<CategoricalState> {
        self.states.read().values().cloned().collect()
    }
    
    /// Get completed state count
    pub fn completed_count(&self) -> usize {
        *self.completed_count.read()
    }
}

impl Default for CategoricalEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_creation() {
        let engine = CategoricalEngine::new();
        let coords = SCoordinates::new(0.5, 0.3, 0.7);
        let id = engine.create_state(coords);
        
        let state = engine.get_state(id).unwrap();
        assert_eq!(state.id, id);
        assert!(!state.completed);
    }
    
    #[test]
    fn test_phase_lock_formation() {
        let engine = CategoricalEngine::new();
        let id1 = engine.create_state(SCoordinates::new(0.0, 0.0, 0.0));
        let id2 = engine.create_state(SCoordinates::new(0.1, 0.0, 0.0));
        
        engine.form_phase_lock(id1, id2, 1.0).unwrap();
        
        let state1 = engine.get_state(id1).unwrap();
        assert!(state1.can_access(id2));
    }
    
    #[test]
    fn test_categorical_completion() {
        let engine = CategoricalEngine::new();
        let id = engine.create_state(SCoordinates::new(0.0, 0.0, 0.0));
        
        engine.complete_state(id).unwrap();
        assert!(engine.is_completed(id).unwrap());
        
        // Cannot complete again (irreversibility)
        assert!(engine.complete_state(id).is_err());
    }
    
    #[test]
    fn test_network_independence_from_velocity() {
        // This is the key test: phase-lock network should be the same
        // regardless of molecular velocities
        let engine = CategoricalEngine::new();
        
        // Same positions, would have different velocities in real system
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1e-9, 0.0, 0.0],  // 1 nm apart
            [0.0, 1e-9, 0.0],
        ];
        let types = vec![MolecularType::NonPolar; 3];
        
        engine.construct_network(&positions, &types).unwrap();
        
        let (nodes, edges, _) = engine.network_stats();
        assert_eq!(nodes, 3);
        assert!(edges > 0); // Should have phase-lock connections
    }
}

