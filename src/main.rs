//! Maxwell Processor CLI
//!
//! Command-line interface for the Complementarity-Aware Processor.

use maxwell_processor::*;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "maxwell")]
#[command(author = "Kundai Farai Sachikonye")]
#[command(version = "0.1.0")]
#[command(about = "Complementarity-Aware Processor for Categorical Phase-Lock Dynamics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Verbosity level
    #[arg(short, long, default_value = "info")]
    verbosity: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a demonstration of the processor
    Demo {
        /// Number of molecules to simulate
        #[arg(short, long, default_value = "100")]
        molecules: usize,
        
        /// Temperature in Kelvin
        #[arg(short, long, default_value = "300.0")]
        temperature: f64,
    },
    
    /// Show face complementarity
    Complementarity,
    
    /// Demonstrate the seven-fold dissolution of Maxwell's Demon
    Dissolution,
    
    /// Run the retrieval paradox demonstration
    RetrievalParadox {
        /// Number of steps
        #[arg(short, long, default_value = "100")]
        steps: usize,
    },
    
    /// Show projection of categorical to kinetic face
    Projection,
    
    /// Run 3^k recursive completion
    Complete {
        /// Recursion depth
        #[arg(short, long, default_value = "3")]
        depth: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    // Set up logging
    let level = match cli.verbosity.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");
    
    match cli.command {
        Commands::Demo { molecules, temperature } => {
            run_demo(molecules, temperature);
        }
        Commands::Complementarity => {
            show_complementarity();
        }
        Commands::Dissolution => {
            show_dissolution();
        }
        Commands::RetrievalParadox { steps } => {
            run_retrieval_paradox(steps);
        }
        Commands::Projection => {
            show_projection();
        }
        Commands::Complete { depth } => {
            run_completion(depth);
        }
    }
}

fn run_demo(molecules: usize, temperature: f64) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           MAXWELL PROCESSOR DEMONSTRATION                        ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Complementarity-Aware Processor for Categorical Phase-Lock       ║");
    println!("║ Dynamics. Implements the seven-fold dissolution of Maxwell's     ║");
    println!("║ Demon through categorical phase-lock topology.                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    info!("Creating processor with {} molecules at {} K", molecules, temperature);
    
    let processor = MaxwellProcessor::new();
    
    println!("Current observable face: {}", processor.observable_face());
    println!();
    
    // Create some categorical states
    println!("Creating categorical states...");
    processor.execute_categorical(|engine| {
        for i in 0..10 {
            let coords = SCoordinates::new(
                i as f64 * 0.1,
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.1).cos(),
            );
            engine.create_state(coords);
        }
        
        let (nodes, edges, density) = engine.network_stats();
        println!("  Created {} nodes, {} edges, density: {:.4}", nodes, edges, density);
    }).unwrap();
    
    // Form some phase-locks
    println!("\nForming phase-lock network...");
    processor.execute_categorical(|engine| {
        for i in 0..9 {
            let _ = engine.form_phase_lock(i, i + 1, 1.0 / ((i + 1) as f64));
        }
        
        let (nodes, edges, density) = engine.network_stats();
        println!("  Network: {} nodes, {} edges, density: {:.4}", nodes, edges, density);
        println!("  Categorical entropy: {:.4e} J/K", engine.categorical_entropy());
    }).unwrap();
    
    // Switch to kinetic face
    println!("\nSwitching to kinetic face...");
    processor.switch_face().unwrap();
    println!("Current observable face: {}", processor.observable_face());
    
    // Initialize kinetic states
    processor.execute_kinetic(|engine| {
        engine.set_temperature(temperature);
        let ids = engine.initialize_maxwell_boltzmann(molecules, 1.0);
        println!("  Initialized {} molecules with Maxwell-Boltzmann distribution", ids.len());
        println!("  Mean velocity: {:.2} m/s", engine.mean_velocity());
        println!("  Temperature: {:.2} K", engine.get_temperature());
        
        let (fast, slow) = engine.demon_sorting();
        println!("\n  'Demon' sorting (what Maxwell would see):");
        println!("    Fast molecules: {}", fast.len());
        println!("    Slow molecules: {}", slow.len());
    }).unwrap();
    
    println!("\n✓ Demo complete");
}

fn show_complementarity() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              INFORMATION COMPLEMENTARITY                          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  Information has TWO CONJUGATE FACES that cannot be observed     ║");
    println!("║  simultaneously, like ammeter/voltmeter in electrical circuits:  ║");
    println!("║                                                                   ║");
    println!("║  ┌─────────────────────┐     ┌─────────────────────┐             ║");
    println!("║  │   CATEGORICAL FACE  │ ≢   │    KINETIC FACE     │             ║");
    println!("║  │                     │     │                     │             ║");
    println!("║  │ • Phase-lock network│     │ • Velocity distrib. │             ║");
    println!("║  │ • Topological nav.  │     │ • Temperature       │             ║");
    println!("║  │ • Cat. completion   │     │ • Energy sorting    │             ║");
    println!("║  │ • Config. dynamics  │     │ • Thermo observables│             ║");
    println!("║  │                     │     │                     │             ║");
    println!("║  │ (Ground Truth)      │     │ (Maxwell's View)    │             ║");
    println!("║  └─────────────────────┘     └─────────────────────┘             ║");
    println!("║                                                                   ║");
    println!("║  CANNOT OBSERVE BOTH SIMULTANEOUSLY                               ║");
    println!("║                                                                   ║");
    println!("║  When observing the KINETIC face, categorical dynamics appear    ║");
    println!("║  as 'demon' behavior. The demon IS the projection of hidden      ║");
    println!("║  categorical dynamics onto the observable kinetic face.          ║");
    println!("║                                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn show_dissolution() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         THE SEVEN-FOLD DISSOLUTION OF MAXWELL'S DEMON            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  1. TEMPORAL TRIVIALITY                                          ║");
    println!("║     Fluctuations produce the same configurations naturally.      ║");
    println!("║     The demon is redundant.                                      ║");
    println!("║                                                                   ║");
    println!("║  2. PHASE-LOCK TEMPERATURE INDEPENDENCE                          ║");
    println!("║     Same spatial arrangement exists at any temperature.          ║");
    println!("║     The demon sorts the wrong property.                          ║");
    println!("║                                                                   ║");
    println!("║  3. RETRIEVAL PARADOX                                            ║");
    println!("║     Cannot outpace thermal equilibration (~10³³ ops/s needed).   ║");
    println!("║     The demon is self-defeating.                                 ║");
    println!("║                                                                   ║");
    println!("║  4. DISSOLUTION OF OBSERVATION                                   ║");
    println!("║     Topology determines accessibility without measurement.       ║");
    println!("║     The demon needs no information.                              ║");
    println!("║                                                                   ║");
    println!("║  5. DISSOLUTION OF DECISION                                      ║");
    println!("║     Categorical pathways follow topology automatically.          ║");
    println!("║     The demon makes no decisions.                                ║");
    println!("║                                                                   ║");
    println!("║  6. DISSOLUTION OF SECOND LAW                                    ║");
    println!("║     Categorical completion increases entropy.                    ║");
    println!("║     The demon violates nothing.                                  ║");
    println!("║                                                                   ║");
    println!("║  7. INFORMATION COMPLEMENTARITY                                  ║");
    println!("║     The demon is the projection of hidden categorical dynamics   ║");
    println!("║     onto the observable kinetic face.                            ║");
    println!("║     The demon is a projection artifact.                          ║");
    println!("║                                                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║                    THERE IS NO DEMON.                             ║");
    println!("║                                                                   ║");
    println!("║  There is only the phase-lock network, completing categorical    ║");
    println!("║  states according to topology, indifferent to velocities.        ║");
    println!("║                                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn run_retrieval_paradox(steps: usize) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              THE RETRIEVAL PARADOX DEMONSTRATION                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    let kinetic = KineticEngine::new();
    kinetic.set_temperature(300.0);
    kinetic.initialize_maxwell_boltzmann(100, 1.0);
    
    println!("Demonstrating that velocity-based sorting is self-defeating:");
    println!("Thermal equilibration randomizes velocities faster than sorting.\n");
    
    let history = kinetic.demonstrate_retrieval_paradox(steps);
    
    println!("Step  | Fast | Slow | Note");
    println!("------|------|------|------");
    
    for (i, (fast, slow)) in history.iter().enumerate() {
        let note = if *fast > 55 || *slow > 55 {
            "← deviation"
        } else {
            ""
        };
        println!("{:5} | {:4} | {:4} | {}", i, fast, slow, note);
    }
    
    println!("\nThe distribution stays ~50/50 because thermal equilibration");
    println!("(~10¹⁰ collisions/s) randomizes velocities faster than any");
    println!("sorting mechanism could maintain a sorted state.");
    println!("\nThe demon CANNOT outpace thermal relaxation.");
}

fn show_projection() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          CATEGORICAL → KINETIC PROJECTION                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    let explainer = ProjectionExplainer::new();
    
    // Create a sample categorical state
    let state = CategoricalState::new(42, SCoordinates::new(0.5, 0.3, 0.7));
    
    // Project it
    let projection = explainer.project(&state);
    
    println!("Categorical State:");
    println!("  ID: {}", state.id);
    println!("  Coordinates: S_k={:.2}, S_t={:.2}, S_e={:.2}", 
        state.coordinates.s_k, state.coordinates.s_t, state.coordinates.s_e);
    
    println!("\nKinetic Projection:");
    println!("  Apparent temperature: {:.2}", projection.apparent_temperature);
    println!("  Velocity distribution: {} samples", projection.velocity_distribution.len());
    
    let fast_count = projection.apparent_sorting.iter()
        .filter(|&&c| c == MoleculeClassification::Fast).count();
    let slow_count = projection.apparent_sorting.len() - fast_count;
    println!("  Apparent 'sorting': {} fast, {} slow", fast_count, slow_count);
    
    println!("\nWhy this appears as 'demon' behavior:");
    println!("{}", projection.demon_appearance);
    
    // Show complementarity explanation
    println!("\n{}", "=".repeat(70));
    println!("{}", explainer.explain_complementarity());
}

fn run_completion(depth: usize) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              3^k RECURSIVE COMPLETION                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    let engine = RecursiveCompletionEngine::with_max_depth(depth);
    let initial = CategoricalState::new(0, SCoordinates::new(0.0, 0.0, 0.0));
    
    println!("Initial state: ID={}, coords=({:.2}, {:.2}, {:.2})",
        initial.id, initial.coordinates.s_k, initial.coordinates.s_t, initial.coordinates.s_e);
    println!("Recursion depth: {}", depth);
    println!();
    
    let result = engine.complete(&initial, depth);
    
    println!("Results:");
    println!("  States completed: {}", result.states_completed);
    println!("  Depth reached: {}", result.depth_reached);
    println!("  3^k decomposition count: {}", result.decomposition_count);
    println!("  Total entropy change: {:.4e} J/K", result.entropy_change);
    
    println!("\nCascade path by depth:");
    for (d, ids) in result.cascade_path.iter().enumerate() {
        println!("  Depth {}: {} states", d, ids.len());
    }
    
    println!("\nCascade size formula: (3^(k+1) - 1) / 2");
    let sizes = engine.cascade_sizes(depth);
    println!("Sizes per depth: {:?}", sizes);
    println!("Total: {}", engine.total_cascade_size(depth));
}

