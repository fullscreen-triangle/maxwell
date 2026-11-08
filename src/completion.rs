//! Recursive Completion Engine
//!
//! This module implements the 3^k decomposition for categorical completion:
//! - 3^k hierarchical decomposition
//! - Cascade propagation
//! - Accessibility pathways
//! - Self-propagating BMD cascades

use crate::types::*;
use crate::error::{ProcessorError, ProcessorResult};
use std::collections::{HashMap, VecDeque};

/// The Recursive Completion Engine
///
/// Implements the 3^k decomposition where each BMD operation
/// spawns sub-BMDs, creating exponential parallel processing.
#[derive(Debug, Clone)]
pub struct RecursiveCompletionEngine {
    /// Maximum recursion depth
    max_depth: usize,
    
    /// Completion history
    history: Vec<CompletionStep>,
    
    /// Statistics
    stats: CompletionStats,
}

/// A single step in the completion cascade
#[derive(Debug, Clone)]
pub struct CompletionStep {
    pub depth: usize,
    pub state_id: u64,
    pub spawned_children: Vec<u64>,
    pub entropy_change: f64,
}

/// Statistics about completion operations
#[derive(Debug, Clone, Default)]
pub struct CompletionStats {
    pub total_completions: usize,
    pub max_depth_reached: usize,
    pub total_cascade_size: usize,
    pub total_entropy_change: f64,
}

impl RecursiveCompletionEngine {
    pub fn new() -> Self {
        Self {
            max_depth: 10,
            history: Vec::new(),
            stats: CompletionStats::default(),
        }
    }
    
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            max_depth,
            history: Vec::new(),
            stats: CompletionStats::default(),
        }
    }
    
    // ========================================================================
    // 3^k DECOMPOSITION
    // ========================================================================
    
    /// Perform recursive completion with 3^k decomposition
    pub fn complete(&self, initial: &CategoricalState, depth: usize) -> CompletionResult {
        let mut completed_ids = Vec::new();
        let mut cascade_path = Vec::new();
        let mut total_entropy_change = 0.0;
        let mut queue = VecDeque::new();
        let actual_depth = depth.min(self.max_depth);
        
        // Start with initial state
        queue.push_back((initial.clone(), 0));
        
        while let Some((state, current_depth)) = queue.pop_front() {
            if current_depth > actual_depth || state.completed {
                continue;
            }
            
            // Complete this state
            completed_ids.push(state.id);
            
            // Track cascade path
            if cascade_path.len() <= current_depth {
                cascade_path.push(Vec::new());
            }
            cascade_path[current_depth].push(state.id);
            
            // Entropy increases with network density
            let entropy_contribution = self.compute_entropy_contribution(&state);
            total_entropy_change += entropy_contribution;
            
            // 3^k decomposition: each state spawns 3 sub-states
            if current_depth < actual_depth {
                let children = self.spawn_children(&state, current_depth);
                for child in children {
                    queue.push_back((child, current_depth + 1));
                }
            }
        }
        
        // Calculate decomposition count: 3^k summation
        let decomposition_count: usize = (0..=actual_depth)
            .map(|k| 3_usize.pow(k as u32))
            .sum();
        
        CompletionResult {
            states_completed: completed_ids.len(),
            depth_reached: actual_depth,
            decomposition_count,
            entropy_change: total_entropy_change,
            completed_ids,
            cascade_path,
        }
    }
    
    /// Spawn 3 child states (tri-dimensional decomposition)
    fn spawn_children(&self, parent: &CategoricalState, depth: usize) -> Vec<CategoricalState> {
        let coords = &parent.coordinates;
        let scale = 0.5_f64.powi(depth as i32 + 1);
        
        // Three children in each S-coordinate direction
        vec![
            // Knowledge dimension child
            CategoricalState::new(
                parent.id * 3 + 1,
                SCoordinates::new(
                    coords.s_k + scale,
                    coords.s_t,
                    coords.s_e,
                ),
            ),
            // Time dimension child
            CategoricalState::new(
                parent.id * 3 + 2,
                SCoordinates::new(
                    coords.s_k,
                    coords.s_t + scale,
                    coords.s_e,
                ),
            ),
            // Entropy dimension child
            CategoricalState::new(
                parent.id * 3 + 3,
                SCoordinates::new(
                    coords.s_k,
                    coords.s_t,
                    coords.s_e + scale,
                ),
            ),
        ]
    }
    
    /// Compute entropy contribution from completing a state
    fn compute_entropy_contribution(&self, state: &CategoricalState) -> f64 {
        // Entropy proportional to phase-lock network density
        let k_b = 1.380649e-23;
        let edge_count = state.phase_locks.len() as f64;
        k_b * (edge_count + 1.0).ln()
    }
    
    // ========================================================================
    // CASCADE PROPAGATION
    // ========================================================================
    
    /// Propagate completion through accessible states
    pub fn propagate_cascade(&self, start: &CategoricalState, accessible: &[CategoricalState]) -> Vec<u64> {
        let mut completed = Vec::new();
        let mut to_complete = VecDeque::new();
        let mut visited = HashMap::new();
        
        to_complete.push_back(start.id);
        visited.insert(start.id, true);
        
        while let Some(current_id) = to_complete.pop_front() {
            completed.push(current_id);
            
            // Find accessible states from current
            for state in accessible {
                if !visited.contains_key(&state.id) && state.accessible.contains(&current_id) {
                    visited.insert(state.id, true);
                    to_complete.push_back(state.id);
                }
            }
        }
        
        completed
    }
    
    /// Compute cascade size at each depth
    pub fn cascade_sizes(&self, max_depth: usize) -> Vec<usize> {
        (0..=max_depth)
            .map(|k| 3_usize.pow(k as u32))
            .collect()
    }
    
    /// Total cascade size for depth k: (3^(k+1) - 1) / 2
    pub fn total_cascade_size(&self, depth: usize) -> usize {
        (3_usize.pow(depth as u32 + 1) - 1) / 2
    }
    
    // ========================================================================
    // ACCESSIBILITY PATHWAYS
    // ========================================================================
    
    /// Find all pathways from start to goal
    pub fn find_pathways(
        &self,
        start_id: u64,
        goal_id: u64,
        states: &HashMap<u64, CategoricalState>,
        max_path_length: usize,
    ) -> Vec<Vec<u64>> {
        let mut paths = Vec::new();
        let mut current_path = vec![start_id];
        
        self.dfs_paths(
            start_id,
            goal_id,
            states,
            &mut current_path,
            &mut paths,
            max_path_length,
        );
        
        paths
    }
    
    fn dfs_paths(
        &self,
        current: u64,
        goal: u64,
        states: &HashMap<u64, CategoricalState>,
        path: &mut Vec<u64>,
        paths: &mut Vec<Vec<u64>>,
        max_length: usize,
    ) {
        if current == goal {
            paths.push(path.clone());
            return;
        }
        
        if path.len() >= max_length {
            return;
        }
        
        if let Some(state) = states.get(&current) {
            for &next_id in &state.accessible {
                if !path.contains(&next_id) {
                    path.push(next_id);
                    self.dfs_paths(next_id, goal, states, path, paths, max_length);
                    path.pop();
                }
            }
        }
    }
    
    /// Compute shortest path length
    pub fn shortest_path_length(
        &self,
        start_id: u64,
        goal_id: u64,
        states: &HashMap<u64, CategoricalState>,
    ) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut visited = HashMap::new();
        
        queue.push_back((start_id, 0));
        visited.insert(start_id, true);
        
        while let Some((current, dist)) = queue.pop_front() {
            if current == goal_id {
                return Some(dist);
            }
            
            if let Some(state) = states.get(&current) {
                for &next_id in &state.accessible {
                    if !visited.contains_key(&next_id) {
                        visited.insert(next_id, true);
                        queue.push_back((next_id, dist + 1));
                    }
                }
            }
        }
        
        None
    }
    
    // ========================================================================
    // STATISTICS
    // ========================================================================
    
    /// Get completion statistics
    pub fn get_stats(&self) -> &CompletionStats {
        &self.stats
    }
    
    /// Get completion history
    pub fn get_history(&self) -> &[CompletionStep] {
        &self.history
    }
    
    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.stats = CompletionStats::default();
    }
}

impl Default for RecursiveCompletionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_3k_decomposition() {
        let engine = RecursiveCompletionEngine::new();
        let initial = CategoricalState::new(0, SCoordinates::new(0.0, 0.0, 0.0));
        
        let result = engine.complete(&initial, 2);
        
        // Depth 0: 1, Depth 1: 3, Depth 2: 9 = 13 total
        assert_eq!(result.decomposition_count, 13);
        assert_eq!(result.depth_reached, 2);
    }
    
    #[test]
    fn test_cascade_sizes() {
        let engine = RecursiveCompletionEngine::new();
        let sizes = engine.cascade_sizes(3);
        
        assert_eq!(sizes, vec![1, 3, 9, 27]);
    }
    
    #[test]
    fn test_total_cascade_size() {
        let engine = RecursiveCompletionEngine::new();
        
        assert_eq!(engine.total_cascade_size(0), 1);
        assert_eq!(engine.total_cascade_size(1), 4);  // 1 + 3
        assert_eq!(engine.total_cascade_size(2), 13); // 1 + 3 + 9
    }
    
    #[test]
    fn test_entropy_increase() {
        let engine = RecursiveCompletionEngine::new();
        let initial = CategoricalState::new(0, SCoordinates::new(0.0, 0.0, 0.0));
        
        let result = engine.complete(&initial, 2);
        
        // Entropy should increase (second law)
        assert!(result.entropy_change > 0.0);
    }
}

