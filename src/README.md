# Maxwell Processor

**Complementarity-Aware Processor for Categorical Phase-Lock Dynamics**

A computational implementation of the theoretical framework from "Resolution of Maxwell's Demon Through Categorical Phase-Lock Topology".

## Overview

This processor implements the seven-fold dissolution of Maxwell's Demon by operating on **both faces of information**:

- **Categorical Face**: Phase-lock networks, topological navigation, categorical completion
- **Kinetic Face**: Velocity distributions, temperature, thermodynamic observables

The key insight: **information has two conjugate faces that cannot be simultaneously observed**, like ammeter/voltmeter measurements in electrical circuits. The "demon" that Maxwell saw was the projection of hidden categorical dynamics onto the observable kinetic face.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              COMPLEMENTARITY-AWARE PROCESSOR                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐           ┌─────────────────┐              │
│  │  CATEGORICAL    │◄─────────►│    KINETIC      │              │
│  │  FACE ENGINE    │           │  FACE ENGINE    │              │
│  │                 │           │                 │              │
│  │ • Phase-lock    │           │ • Velocities    │              │
│  │ • Topology      │           │ • Temperature   │              │
│  │ • Completion    │           │ • Energy        │              │
│  └────────┬────────┘           └────────┬────────┘              │
│           │                              │                       │
│           └──────────┬───────────────────┘                       │
│                      ▼                                           │
│           ┌─────────────────────┐                               │
│           │  COMPLEMENTARITY    │                               │
│           │     MANAGER         │                               │
│           │                     │                               │
│           │ • Face switching    │                               │
│           │ • Incompatibility   │                               │
│           │ • Projection        │                               │
│           └──────────┬──────────┘                               │
│                      ▼                                           │
│           ┌─────────────────────┐                               │
│           │    PROJECTION       │                               │
│           │    EXPLAINER        │                               │
│           │                     │                               │
│           │ • Cat → Kinetic     │                               │
│           │ • "Demon" generator │                               │
│           └─────────────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `categorical` | Phase-lock network construction, topological navigation, categorical completion |
| `kinetic` | Maxwell-Boltzmann distributions, temperature, demon sorting |
| `complementarity` | Face switching, incompatibility enforcement |
| `projection` | Categorical → Kinetic mapping, demon explanation |
| `equivalence` | Configuration equivalence, state space reduction |
| `completion` | 3^k recursive completion, cascade propagation |

## The Seven-Fold Dissolution

This processor validates all seven arguments dissolving Maxwell's Demon:

1. **Temporal Triviality**: Fluctuations produce the same configurations naturally
2. **Phase-Lock Temperature Independence**: Same arrangement exists at any temperature
3. **Retrieval Paradox**: Cannot outpace thermal equilibration (~10³³ ops/s needed)
4. **Dissolution of Observation**: Topology determines accessibility without measurement
5. **Dissolution of Decision**: Categorical pathways follow topology automatically
6. **Dissolution of Second Law**: Categorical completion increases entropy
7. **Information Complementarity**: Demon is projection of hidden categorical dynamics

## Usage

### CLI

```bash
# Run demonstration
cargo run -- demo --molecules 1000 --temperature 300

# Show complementarity
cargo run -- complementarity

# Demonstrate the seven-fold dissolution
cargo run -- dissolution

# Run retrieval paradox demonstration
cargo run -- retrieval-paradox --steps 100

# Show categorical → kinetic projection
cargo run -- projection

# Run 3^k recursive completion
cargo run -- complete --depth 5
```

### Library

```rust
use maxwell_processor::*;

// Create processor
let processor = MaxwellProcessor::new();

// Execute on categorical face (ground truth)
processor.execute_categorical(|engine| {
    let id = engine.create_state(SCoordinates::new(0.5, 0.3, 0.7));
    engine.complete_state(id).unwrap();
}).unwrap();

// Switch to kinetic face
processor.switch_face().unwrap();

// Execute on kinetic face (Maxwell's view)
processor.execute_kinetic(|engine| {
    engine.initialize_maxwell_boltzmann(1000, 1.0);
    let (fast, slow) = engine.demon_sorting();
    println!("'Demon' sorting: {} fast, {} slow", fast.len(), slow.len());
}).unwrap();

// Explain why this appears as demon behavior
let state = CategoricalState::new(0, SCoordinates::new(0.5, 0.3, 0.7));
let projection = processor.project_to_kinetic(&state);
println!("{}", projection.demon_appearance);
```

## Building

```bash
cd processor
cargo build --release
```

## Testing

```bash
cargo test
```

## Benchmarks

```bash
cargo bench
```

## Python Validation

See `../validation/` for the Python validation suite that independently validates:
- Categorical operations
- Kinetic distributions
- Complementarity constraints
- All seven dissolution arguments

## License

MIT

## Author

Kundai Farai Sachikonye
Technical University of Munich

