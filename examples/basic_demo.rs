//! Basic demonstration of the Maxwell Processor
//!
//! Run with: `cargo run --example basic_demo`

use maxwell_processor::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           MAXWELL PROCESSOR - BASIC DEMONSTRATION                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Create the processor
    let processor = MaxwellProcessor::new();
    
    // =========================================================================
    // Part 1: Categorical Face (Ground Truth)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 1: CATEGORICAL FACE (What's Actually Happening)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\nCurrent face: {}", processor.observable_face());
    
    processor.execute_categorical(|engine| {
        // Create categorical states
        println!("\nCreating categorical states...");
        
        let state_ids: Vec<u64> = (0..5)
            .map(|i| {
                let coords = SCoordinates::new(
                    i as f64 * 0.2,
                    (i as f64 * 0.3).sin(),
                    (i as f64 * 0.4).cos(),
                );
                engine.create_state(coords)
            })
            .collect();
        
        println!("  Created {} states", state_ids.len());
        
        // Form phase-locks (based on position, NOT velocity!)
        println!("\nForming phase-lock network...");
        for i in 0..4 {
            let coupling = 1.0 / ((i + 1) as f64);
            engine.form_phase_lock(state_ids[i], state_ids[i + 1], coupling)
                .expect("Failed to form phase-lock");
        }
        
        let (nodes, edges, density) = engine.network_stats();
        println!("  Network: {} nodes, {} edges", nodes, edges);
        println!("  Density: {:.4}", density);
        println!("  Categorical entropy: {:.4e} J/K", engine.categorical_entropy());
        
        // Complete a state (irreversible!)
        println!("\nCompleting state 0...");
        engine.complete_state(0).expect("Failed to complete state");
        println!("  State 0 completed: {}", engine.is_completed(0).unwrap());
        
        // Try to complete again (should fail)
        match engine.complete_state(0) {
            Ok(_) => println!("  ERROR: Should not be able to complete twice!"),
            Err(e) => println!("  Correctly rejected: {}", e),
        }
    }).expect("Categorical operation failed");
    
    // =========================================================================
    // Part 2: Switch to Kinetic Face
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 2: SWITCHING FACES (Complementarity in Action)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\nSwitching from {} to kinetic face...", processor.observable_face());
    processor.switch_face().expect("Failed to switch face");
    println!("Now on: {}", processor.observable_face());
    
    // Try to access categorical face (should fail!)
    match processor.execute_categorical(|_| ()) {
        Ok(_) => println!("ERROR: Should not access categorical from kinetic face!"),
        Err(e) => println!("Correctly blocked: {}", e),
    }
    
    // =========================================================================
    // Part 3: Kinetic Face (Maxwell's View)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 3: KINETIC FACE (What Maxwell Observed)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    processor.execute_kinetic(|engine| {
        engine.set_temperature(300.0);
        
        println!("\nInitializing Maxwell-Boltzmann distribution...");
        let ids = engine.initialize_maxwell_boltzmann(100, 1.0);
        println!("  {} molecules initialized at {} K", ids.len(), engine.get_temperature());
        println!("  Mean velocity: {:.2} m/s", engine.mean_velocity());
        
        // What the demon would "see"
        let (fast, slow) = engine.demon_sorting();
        println!("\n'Demon' sorting (what Maxwell would observe):");
        println!("  Fast molecules: {}", fast.len());
        println!("  Slow molecules: {}", slow.len());
        
        // Demonstrate retrieval paradox
        println!("\nDemonstrating retrieval paradox (5 steps)...");
        let history = engine.demonstrate_retrieval_paradox(5);
        for (i, (f, s)) in history.iter().enumerate() {
            println!("  Step {}: {} fast, {} slow", i, f, s);
        }
        println!("  → Distribution stays ~50/50 due to thermal equilibration");
    }).expect("Kinetic operation failed");
    
    // =========================================================================
    // Part 4: Projection (Why Maxwell Saw a Demon)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 4: PROJECTION (Why Maxwell Saw a Demon)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let state = CategoricalState::new(42, SCoordinates::new(0.5, 0.3, 0.7));
    let projection = processor.project_to_kinetic(&state);
    
    println!("\nProjecting categorical state to kinetic face:");
    println!("  Source state ID: {}", projection.source_state_id);
    println!("  Apparent temperature: {:.2}", projection.apparent_temperature);
    
    let fast = projection.apparent_sorting.iter()
        .filter(|&&c| c == MoleculeClassification::Fast)
        .count();
    println!("  Apparent sorting: {} fast, {} slow", 
        fast, projection.apparent_sorting.len() - fast);
    
    println!("\nWhy this looks like 'demon' behavior:");
    println!("  {}", projection.demon_appearance);
    
    // =========================================================================
    // Conclusion
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CONCLUSION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("The 'demon' is NOT an agent. It is the projection of hidden");
    println!("categorical dynamics onto the observable kinetic face.");
    println!();
    println!("There is no demon. There is only the phase-lock network.");
    println!();
}

