# Maxwell


<div align="center">
  <img src="docs/assets/mbende.jpg" alt="Logo" width="300"/>
</div>


**Categorical Computing Framework: S-Entropy Navigation, Virtual Gas Ensembles, and Phase-Lock Topology**

This repository contains the theoretical foundation and experimental validation for a computing framework based on categorical completion rather than sequential instruction execution. The framework comprises categorical processing, categorical memory, semantic processing through molecular encoding, and a resolution of Maxwell's Demon paradox.

## Framework Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Categorical Thermodynamics** | Monograph and gas-law papers | `docs/thermodynamics/` |
| **Categorical Processor** | Oscillator-based computation through phase-lock networks | `docs/computing/processor/`, `src/` |
| **Categorical Memory** | S-entropy coordinate addressing with precision-by-difference navigation | `docs/computing/categorical-memory/` |
| **Categorical Computing** | Unified computation through molecular structure prediction | `docs/computing/categorical-computing/` |
| **Maxwell Demon Resolution** | Dissolution through categorical phase-lock topology | `docs/thermodynamics/maxwell-demon/` |
| **Poincaré Computing** | Unified computation where recurrence = solution | `examples/thermodynamics/poincare/` |

## Core Concepts

### S-Entropy Coordinates

The framework operates in a three-dimensional coordinate space:

- **S_k** (Knowledge Entropy): Uncertainty in state identification
- **S_t** (Temporal Entropy): Uncertainty in timing
- **S_e** (Evolution Entropy): Uncertainty in trajectory

These coordinates are derived from hardware timing measurements, not simulated or approximated.

### Virtual Gas Ensemble

Hardware oscillations (CPU cycles, memory timing, I/O jitter) create virtual molecules positioned in S-entropy space. The computer is not a device that processes a gas—the computer *is* the gas. Key properties:

- **Temperature**: Timing jitter variance
- **Pressure**: Molecular creation rate (sampling rate)
- **Volume**: S-space region occupied

### Spectrometer-Molecule Identity

The measurement apparatus and measured entity are the same categorical state. The "fishing tackle" defines what "fish" can be caught—there is no separation between observer and observed in categorical measurement.

### Spatial Distance Irrelevance

Categorical navigation does not involve spatial propagation. Jupiter's core and a local molecule are equally accessible—both are S-coordinate positions reached through categorical navigation.

### Poincaré Computing

Named for Henri Poincaré's recurrence theorem: in a bounded phase space, a dynamical system returns to its initial state. In this framework, **recurrence is solution**—computation completes when the categorical state returns to its origin.

The trajectory through S-entropy space IS the answer. A single categorical state is simultaneously:
- **Memory**: S-coordinates are addresses
- **Processor**: Oscillation frequency is processing rate  
- **Semantics**: Harmonic structure is meaning

There is no separation between these subsystems. The categorical state doesn't "call" memory or "invoke" processing—it IS all three, viewed from different perspectives.

## Repository Structure

```
maxwell/
├── docs/                        # Papers (LaTeX + PDF), figures, notes — by topic
│   ├── thermodynamics/          # Monograph, gas laws, gas/flux dynamics, Maxwell demon
│   ├── computing/               # Processor, memory, categorical/trajectory computing
│   ├── measurement/             # Hardware timing, zero-backaction, trisection
│   ├── biochemistry/            # Catalysis, intracellular trajectories, microscopy
│   ├── physics/                 # Light derivation
│   ├── assets/
│   └── DUPLICATES.md            # Byte-identical copies removed in the 2026-11 cleanup
├── examples/                    # Python code, mirroring the docs topics
│   ├── thermodynamics/poincare/ # Virtual Gas Ensemble implementation
│   ├── computing/validation/    # Experimental validation (maxwell_validation)
│   ├── <topic>/<paper>/         # Figure scripts for the paper in docs/<topic>/<paper>/
│   └── basic_demo.rs            # Rust example (cargo run --example basic_demo)
├── src/                         # Rust processor (maxwell-processor crate)
└── web/                         # Next.js / three.js visualisation app (maupertuis)
```

Figure scripts were moved out of their paper folders without changing their paths, so
a script in `examples/<topic>/<paper>/` that writes `figures/...` relative to itself
needs pointing at `docs/<topic>/<paper>/` before it is re-run.

## Running the Virtual Gas Ensemble

The `examples/thermodynamics/poincare/` package implements the virtual gas ensemble from hardware timing measurements.

```bash
cd examples/thermodynamics/poincare

# Run all experiments
python run_all_experiments.py

# Run individual experiments
python src/run_virtual_molecule.py
python src/run_virtual_chamber.py
python src/run_maxwell_demon.py
python src/run_thermodynamics.py

# Generate panel visualizations
python src/generate_panels.py
```

Results are saved to `examples/thermodynamics/poincare/results/` as JSON files.

## Running Validation

```bash
cd examples/computing/validation
pip install -e .

# Run complete framework validation
python run_complete_framework.py

# Generate publication figures
python generate_figures.py
python generate_all_panels.py
```

## Maxwell Demon Resolution

The resolution demonstrates that Maxwell's Demon paradox dissolves through seven independent arguments:

1. **Temporal Triviality**: Fluctuations produce the same configurations naturally
2. **Phase-Lock Temperature Independence**: The same arrangement exists at any temperature
3. **Retrieval Paradox**: Velocity-based sorting cannot outpace thermal equilibration
4. **Phase-Lock Kinetic Independence**: Network topology does not depend on molecular velocities
5. **Categorical-Physical Distance Inequivalence**: Categorical adjacency does not correspond to spatial proximity
6. **Temperature Emergence**: Temperature emerges from phase-lock cluster statistics
7. **Information Complementarity**: The demon is the projection of hidden categorical dynamics onto the observable kinetic face

The demon is not an agent—it is a projection artifact arising from observing only one face of a two-faced information structure.

## Theoretical Papers

| Paper | Location | Description |
|-------|----------|-------------|
| Categorical Thermodynamics (monograph) | `docs/thermodynamics/monograph/monogram.tex` | Fifteen papers assembled into one theory; plan in `monogram.md` |
| Categorical Processing Unit | `docs/computing/processor/categorical-processing-unit.tex` | Oscillator-processor duality and biological semiconductor computation |
| Categorical Memory | `docs/computing/categorical-memory/molecular-dynamics-categorical-memory.tex` | S-entropy addressing and precision-by-difference navigation |
| Categorical Computing | `docs/computing/categorical-computing/maxwell-categorical-computing.tex` | Semantic processing through molecular structure prediction |
| Maxwell Demon Paradox | `docs/thermodynamics/maxwell-demon/maxwell-demon-paradox.tex` | Dissolution through phase-lock topology |

Each paper includes a section on virtual gas ensembles establishing the categorical foundation.

## Experimental Results

The validation suite produces:

- Hardware timing measurements from CPU, memory, and I/O oscillators
- S-entropy coordinate distributions
- Precision-by-difference trajectories
- Maxwell demon sorting with zero backaction
- Thermodynamic properties (temperature, pressure, entropy) from hardware jitter
- Publication-quality panel figures

Key validation results:
- 100% cache hit rate with categorical prefetching
- 96.1% latency reduction through precision-by-difference navigation
- Zero backaction in categorical measurement (all 200+ observations at exactly 0)
- Maxwell-Boltzmann distribution emerges from hardware timing jitter

## Author

Kundai Farai Sachikonye  
Technical University of Munich  
kundai.sachikonye@wzw.tum.de

## License

MIT
