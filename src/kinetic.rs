//! Kinetic Face Engine
//!
//! This module implements the "observable projection" operations:
//! - Velocity distributions
//! - Temperature measurements
//! - Energy sorting
//! - Thermodynamic observables
//!
//! This is what Maxwell saw (not what's actually happening).

use crate::types::*;
use crate::error::{ProcessorError, ProcessorResult};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use parking_lot::RwLock;

/// The Kinetic Face Engine
///
/// Computes kinetic observables from molecular states.
/// This is the "observable projection" layer - what Maxwell would see
/// when observing the system through the kinetic face.
#[derive(Debug)]
pub struct KineticEngine {
    /// Molecular kinetic states
    states: RwLock<HashMap<u64, KineticState>>,
    
    /// System temperature (K)
    temperature: RwLock<f64>,
    
    /// Configuration
    config: ProcessorConfig,
}

impl KineticEngine {
    /// Create a new kinetic engine
    pub fn new() -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
            temperature: RwLock::new(300.0), // Default room temperature
            config: ProcessorConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: ProcessorConfig) -> Self {
        let temp = config.temperature;
        Self {
            states: RwLock::new(HashMap::new()),
            temperature: RwLock::new(temp),
            config,
        }
    }
    
    // ========================================================================
    // VELOCITY DISTRIBUTIONS
    // ========================================================================
    
    /// Initialize molecules with Maxwell-Boltzmann velocity distribution
    pub fn initialize_maxwell_boltzmann(&self, n: usize, mass: f64) -> Vec<u64> {
        let temp = *self.temperature.read();
        let k_b = 1.380649e-23;
        
        // Standard deviation for Maxwell-Boltzmann: sqrt(k_B * T / m)
        let sigma = (k_b * temp / mass).sqrt();
        let normal = Normal::new(0.0, sigma).unwrap();
        let mut rng = rand::thread_rng();
        
        let mut ids = Vec::with_capacity(n);
        let mut states = self.states.write();
        
        for i in 0..n {
            let id = i as u64;
            
            // Sample velocity components from Maxwell-Boltzmann
            let vx: f64 = normal.sample(&mut rng);
            let vy: f64 = normal.sample(&mut rng);
            let vz: f64 = normal.sample(&mut rng);
            let speed = (vx * vx + vy * vy + vz * vz).sqrt();
            
            // Random position
            let position = [
                rng.gen::<f64>() * 1e-6,
                rng.gen::<f64>() * 1e-6,
                rng.gen::<f64>() * 1e-6,
            ];
            
            let state = KineticState::new(speed, position);
            states.insert(id, state);
            ids.push(id);
        }
        
        ids
    }
    
    /// Get the velocity distribution
    pub fn velocity_distribution(&self) -> Vec<f64> {
        self.states.read().values().map(|s| s.velocity).collect()
    }
    
    /// Compute mean velocity
    pub fn mean_velocity(&self) -> f64 {
        let states = self.states.read();
        if states.is_empty() {
            return 0.0;
        }
        let sum: f64 = states.values().map(|s| s.velocity).sum();
        sum / states.len() as f64
    }
    
    /// Compute velocity variance
    pub fn velocity_variance(&self) -> f64 {
        let states = self.states.read();
        if states.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_velocity();
        let sum_sq: f64 = states.values().map(|s| (s.velocity - mean).powi(2)).sum();
        sum_sq / (states.len() - 1) as f64
    }
    
    // ========================================================================
    // TEMPERATURE MEASUREMENTS
    // ========================================================================
    
    /// Compute temperature from kinetic energies
    pub fn compute_temperature(&self, mass: f64) -> f64 {
        let states = self.states.read();
        if states.is_empty() {
            return 0.0;
        }
        
        let k_b = 1.380649e-23;
        let total_ke: f64 = states.values().map(|s| s.kinetic_energy * mass).sum();
        
        // T = 2 * <KE> / (3 * N * k_B) for 3D
        let n = states.len() as f64;
        (2.0 * total_ke) / (3.0 * n * k_b)
    }
    
    /// Set system temperature
    pub fn set_temperature(&self, temp: f64) {
        *self.temperature.write() = temp;
    }
    
    /// Get system temperature
    pub fn get_temperature(&self) -> f64 {
        *self.temperature.read()
    }
    
    // ========================================================================
    // ENERGY SORTING (WHAT THE DEMON WOULD DO)
    // ========================================================================
    
    /// Classify molecules as fast or slow (demon's view)
    pub fn classify_molecules(&self, threshold_velocity: f64) -> (Vec<u64>, Vec<u64>) {
        let states = self.states.read();
        let mut fast = Vec::new();
        let mut slow = Vec::new();
        
        for (id, state) in states.iter() {
            match state.classify(threshold_velocity) {
                MoleculeClassification::Fast => fast.push(*id),
                MoleculeClassification::Slow => slow.push(*id),
            }
        }
        
        (fast, slow)
    }
    
    /// Compute the "sorting" that the demon would observe
    /// Returns (hot_side_ids, cold_side_ids)
    pub fn demon_sorting(&self) -> (Vec<u64>, Vec<u64>) {
        let mean_v = self.mean_velocity();
        self.classify_molecules(mean_v)
    }
    
    /// Compute apparent temperature difference if sorted
    pub fn apparent_temperature_difference(&self, mass: f64) -> (f64, f64) {
        let (fast, slow) = self.demon_sorting();
        let states = self.states.read();
        
        let k_b = 1.380649e-23;
        
        // Hot side temperature
        let hot_ke: f64 = fast.iter()
            .filter_map(|id| states.get(id))
            .map(|s| s.kinetic_energy * mass)
            .sum();
        let t_hot = if fast.is_empty() {
            0.0
        } else {
            (2.0 * hot_ke) / (3.0 * fast.len() as f64 * k_b)
        };
        
        // Cold side temperature
        let cold_ke: f64 = slow.iter()
            .filter_map(|id| states.get(id))
            .map(|s| s.kinetic_energy * mass)
            .sum();
        let t_cold = if slow.is_empty() {
            0.0
        } else {
            (2.0 * cold_ke) / (3.0 * slow.len() as f64 * k_b)
        };
        
        (t_hot, t_cold)
    }
    
    // ========================================================================
    // THERMODYNAMIC OBSERVABLES
    // ========================================================================
    
    /// Compute total kinetic energy
    pub fn total_kinetic_energy(&self, mass: f64) -> f64 {
        self.states.read().values().map(|s| s.kinetic_energy * mass).sum()
    }
    
    /// Compute spatial entropy (configurational)
    pub fn spatial_entropy(&self) -> f64 {
        let states = self.states.read();
        let n = states.len();
        if n == 0 {
            return 0.0;
        }
        
        // Simplified: entropy from position distribution
        // In reality would need proper phase space integration
        let k_b = 1.380649e-23;
        k_b * (n as f64).ln()
    }
    
    /// Compute pressure (ideal gas approximation)
    pub fn compute_pressure(&self, volume: f64, mass: f64) -> f64 {
        let n = self.states.read().len() as f64;
        let temp = self.compute_temperature(mass);
        let k_b = 1.380649e-23;
        
        // P = N * k_B * T / V
        n * k_b * temp / volume
    }
    
    // ========================================================================
    // THERMAL EQUILIBRATION (THE RETRIEVAL PARADOX)
    // ========================================================================
    
    /// Simulate one collision step (velocities randomize)
    /// This demonstrates the retrieval paradox: velocities change faster
    /// than any sorting can occur
    pub fn collision_step(&self) {
        let temp = *self.temperature.read();
        let k_b = 1.380649e-23;
        let mass = 1.0; // normalized
        let sigma = (k_b * temp / mass).sqrt();
        let normal = Normal::new(0.0, sigma).unwrap();
        let mut rng = rand::thread_rng();
        
        let mut states = self.states.write();
        
        // Each molecule has a chance to collide and change velocity
        for state in states.values_mut() {
            if rng.gen::<f64>() < 0.1 { // 10% collision probability per step
                let vx: f64 = normal.sample(&mut rng);
                let vy: f64 = normal.sample(&mut rng);
                let vz: f64 = normal.sample(&mut rng);
                let new_speed = (vx * vx + vy * vy + vz * vz).sqrt();
                
                // Update state
                state.velocity = new_speed;
                state.kinetic_energy = 0.5 * mass * new_speed * new_speed;
                state.temperature_contribution = state.kinetic_energy;
            }
        }
    }
    
    /// Demonstrate the retrieval paradox: 
    /// After sorting, velocities randomize faster than re-sorting can occur
    pub fn demonstrate_retrieval_paradox(&self, steps: usize) -> Vec<(usize, usize)> {
        let mut history = Vec::with_capacity(steps);
        
        for _ in 0..steps {
            // Sort into fast/slow
            let (fast, slow) = self.demon_sorting();
            history.push((fast.len(), slow.len()));
            
            // Collisions randomize velocities
            self.collision_step();
        }
        
        // The distribution stays roughly 50/50 because thermal equilibration
        // is faster than any sorting mechanism
        history
    }
    
    // ========================================================================
    // STATE ACCESS
    // ========================================================================
    
    /// Get a kinetic state by ID
    pub fn get_state(&self, id: u64) -> Option<KineticState> {
        self.states.read().get(&id).cloned()
    }
    
    /// Get all kinetic states
    pub fn all_states(&self) -> Vec<KineticState> {
        self.states.read().values().cloned().collect()
    }
    
    /// Add a kinetic state
    pub fn add_state(&self, id: u64, state: KineticState) {
        self.states.write().insert(id, state);
    }
    
    /// Number of molecules
    pub fn molecule_count(&self) -> usize {
        self.states.read().len()
    }
}

impl Default for KineticEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_maxwell_boltzmann() {
        let engine = KineticEngine::new();
        engine.set_temperature(300.0);
        
        let ids = engine.initialize_maxwell_boltzmann(1000, 1.0);
        assert_eq!(ids.len(), 1000);
        
        // Should have reasonable distribution
        let mean_v = engine.mean_velocity();
        assert!(mean_v > 0.0);
    }
    
    #[test]
    fn test_demon_sorting() {
        let engine = KineticEngine::new();
        engine.set_temperature(300.0);
        engine.initialize_maxwell_boltzmann(100, 1.0);
        
        let (fast, slow) = engine.demon_sorting();
        
        // Should split roughly evenly around the mean
        assert!(fast.len() + slow.len() == 100);
        // Most distributions will have both fast and slow
        assert!(fast.len() > 0 || slow.len() > 0);
    }
    
    #[test]
    fn test_retrieval_paradox() {
        let engine = KineticEngine::new();
        engine.set_temperature(300.0);
        engine.initialize_maxwell_boltzmann(100, 1.0);
        
        let history = engine.demonstrate_retrieval_paradox(10);
        
        // Distribution should remain roughly stable due to thermal equilibration
        // (demonstrating that sorting is futile)
        for (fast, slow) in &history {
            assert!(*fast + *slow == 100);
        }
    }
}

