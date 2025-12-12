# Maxwell


<div align="center">
  <img src="assets/img/Alexander_VI.png" alt="Borgia Logo" width="200"/>
</div>


**Resolution of Maxwell's Demon Through Categorical Phase-Lock Topology**

This repository contains:

1. **Theoretical Framework** (`docs/`): The complete mathematical resolution of Maxwell's Demon paradox through categorical phase-lock networks
2. **Processor Implementation** (`processor/`): A Rust implementation of the Complementarity-Aware Processor
3. **Validation Suite** (`validation/`): Python package for validating the theoretical framework

## The Resolution

Maxwell's Demon paradox has puzzled physicists for over 150 years. We provide a complete resolution through **seven independent arguments**:

| # | Argument | Dissolution |
|---|----------|-------------|
| 1 | **Temporal Triviality** | Fluctuations produce same configurations naturally |
| 2 | **Phase-Lock Temperature Independence** | Same arrangement exists at any temperature |
| 3 | **Retrieval Paradox** | Cannot outpace thermal equilibration |
| 4 | **Dissolution of Observation** | Topology determines accessibility without measurement |
| 5 | **Dissolution of Decision** | Categorical pathways follow topology automatically |
| 6 | **Dissolution of Second Law** | Categorical completion increases entropy |
| 7 | **Information Complementarity** | Demon is projection of hidden categorical dynamics |

## The Key Insight

**Information has two conjugate faces that cannot be simultaneously observed:**

```
┌─────────────────────────┐          ┌─────────────────────────┐
│    CATEGORICAL FACE     │    ≢     │     KINETIC FACE        │
│                         │          │                         │
│  • Phase-lock networks  │          │  • Velocity distrib.    │
│  • Topological nav.     │          │  • Temperature          │
│  • Categorical compl.   │          │  • Energy sorting       │
│  • Config. dynamics     │          │  • Thermo observables   │
│                         │          │                         │
│  (Ground Truth)         │          │  (Maxwell's View)       │
└─────────────────────────┘          └─────────────────────────┘
           
         CANNOT OBSERVE BOTH SIMULTANEOUSLY
         (like ammeter/voltmeter in circuits)
```

Maxwell observed the **kinetic face**. The "demon" he saw was the projection of hidden **categorical dynamics** onto his observable face. The demon is not an agent—it is a **projection artifact**.

## Project Structure

```
maxwell/
├── docs/
│   ├── foundation/          # Theoretical papers
│   │   ├── gibbs-paradox/   # Categorical resolution of Gibbs' paradox
│   │   ├── observation-boundary/  # N_max and the observation boundary
│   │   ├── categorical-completion/  # Oscillations=Categories duality
│   │   ├── time-emergence/  # Time from categorical completion
│   │   └── pixel-maxwell-demon/  # Information complementarity
│   └── resolution/          # Maxwell Demon resolution paper
│       ├── resolution-of-maxwell-demons.tex
│       └── sections/
│           ├── phase-lock-networks.tex
│           ├── categorical-completion.tex
│           ├── categorical-selection.tex
│           ├── temperature-emergence.tex
│           ├── entropy-mechanism.tex
│           └── demon-desolution.tex
├── processor/               # Rust implementation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Main processor
│       ├── categorical.rs   # Categorical face engine
│       ├── kinetic.rs       # Kinetic face engine
│       ├── complementarity.rs  # Face switching
│       ├── projection.rs    # Demon appearance explanation
│       ├── equivalence.rs   # Equivalence class filter
│       ├── completion.rs    # 3^k recursive completion
│       └── main.rs          # CLI
└── validation/              # Python validation
    ├── pyproject.toml
    └── src/maxwell_validation/
        ├── categorical.py   # Categorical validation
        ├── kinetic.py       # Kinetic validation
        ├── complementarity.py  # Complementarity tests
        └── dissolution.py   # Seven-fold dissolution
```

## Quick Start

### Run the Processor

```bash
cd processor
cargo run -- dissolution  # Show the seven-fold dissolution
cargo run -- demo         # Run a demonstration
```

### Run Python Validation

```bash
cd validation
pip install -e .
python -m maxwell_validation.dissolution  # Validate all seven arguments
```

## There Is No Demon

The demon dissolves the moment you observe the categorical face:

- On the **kinetic face**: molecules appear sorted, requiring an agent
- On the **categorical face**: phase-lock topology navigates automatically

**There is no demon. There is only the phase-lock network, completing categorical states according to topology, indifferent to the velocities that Maxwell's thought experiment privileged but that physics does not.**

## Author

Kundai Farai Sachikonye  
Technical University of Munich  
kundai.sachikonye@wzw.tum.de

## License

MIT

