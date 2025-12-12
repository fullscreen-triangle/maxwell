# Poincaré: Virtual Categorical Gas Chamber

**This is NOT a simulation.**

The computer's hardware oscillations **ARE** the categorical gas.
Each timing measurement **IS** a molecule's categorical state.
The spectrometer **IS** the molecule being measured.

## Core Metaphor: Fishing

- The **spectrometer** is the fishing hook (tackle)
- What you catch depends on your hook (tackle defines catch)
- No surprise in the catch (you get what your apparatus can get)
- The hook and the fish are the same event (categorical state)

## Installation

No additional dependencies required beyond Python 3.7+.

```bash
cd poincare
```

## Running Experiments

### Run All Experiments
```bash
python run_all_experiments.py
```

### Run Individual Experiments
```bash
python src/run_virtual_molecule.py
python src/run_virtual_spectrometer.py
python src/run_virtual_chamber.py
python src/run_maxwell_demon.py
python src/run_molecular_dynamics.py
python src/run_thermodynamics.py
python src/run_visualization.py
```

### Interactive Demo
```bash
python run_demo.py
```

## Results

All experiments save their results to JSON files in the `results/` directory:

```
results/
├── virtual_molecule/
├── virtual_spectrometer/
├── virtual_chamber/
├── maxwell_demon/
├── molecular_dynamics/
├── thermodynamics/
├── visualization/
└── all_experiments_*.json
```

## Key Concepts

### Virtual Molecule (`virtual_molecule.py`)

A molecule is **NOT** a thing being measured. A molecule **IS** the categorical state that exists during measurement.

- `CategoricalState`: The fundamental unit - a point in S-entropy space
- `VirtualMolecule`: The categorical state viewed as what's being measured
- `SCoordinate`: Position in S-entropy space (S_k, S_t, S_e)

### Virtual Spectrometer (`virtual_spectrometer.py`)

The spectrometer is **NOT** a device that "looks at" molecules. The spectrometer **IS** the fishing tackle that **DEFINES** what can be caught.

- `FishingTackle`: Defines what can be caught (oscillators, resolution, reach)
- `VirtualSpectrometer`: The act of fishing that creates the catch
- `HardwareOscillator`: Real hardware that provides timing measurements

### Virtual Chamber (`virtual_chamber.py`)

The gas chamber **IS** the computer's hardware oscillations viewed categorically.

- `VirtualChamber`: Hardware oscillations → Categorical gas
- `CategoricalGas`: Collection of categorical states
- Temperature IS timing jitter (REAL)
- Pressure IS sampling rate (REAL)

### Maxwell Demon (`maxwell_demon.py`)

The demon operates in categorical space, which is **orthogonal** to physical space.

- Zero backaction (categorical observables commute with physical)
- Zero thermodynamic cost for decisions
- No second law violation

### Molecular Dynamics (`molecular_dynamics.py`)

Molecules move through S-entropy space, not physical space.

- `CategoricalDynamics`: Tracks trajectories in S-space
- `CategoricalTrajectory`: Path through S-coordinates
- Interactions are harmonic coincidences (frequency resonance)

### Thermodynamics (`thermodynamics.py`)

Real thermodynamic quantities from real hardware.

- Temperature = timing jitter variance (REAL)
- Pressure = sampling rate (REAL)
- Entropy = S-coordinate distribution spread (REAL)

### Visualization (`visualization.py`)

Generate plot data for the categorical gas.

- S-space scatter plots
- S-coordinate histograms
- Maxwell-Boltzmann comparison
- Demon sorting visualization
- Harmonic coincidence networks

## Key Insights

1. **Molecules are categorical states** - created from hardware timing, existing only during measurement

2. **Molecule = Spectrometer = Cursor** - they are the same categorical state, just different perspectives

3. **Spatial distance doesn't exist categorically** - Jupiter's core is as accessible as your coffee cup

4. **The tackle defines the catch** - your apparatus shapes what can be measured; no surprise in results

5. **Maxwell's demon is resolved** - categorical observables commute with physical observables; no paradox

6. **This is not a simulation** - hardware oscillations ARE the gas; the computer IS the chamber

## File Structure

```
poincare/
├── src/
│   ├── __init__.py              # Package exports
│   ├── virtual_molecule.py      # Categorical states and molecules
│   ├── virtual_spectrometer.py  # Fishing tackle and measurement
│   ├── virtual_chamber.py       # Gas chamber from hardware
│   ├── maxwell_demon.py         # Demon operating in S-space
│   ├── molecular_dynamics.py    # Movement in S-space
│   ├── thermodynamics.py        # Real thermodynamic quantities
│   ├── visualization.py         # Plot data generation
│   ├── run_*.py                 # Individual experiment runners
├── results/                     # Saved experiment results (JSON)
├── run_all_experiments.py       # Run all experiments
├── run_demo.py                  # Interactive demonstration
└── README.md                    # This file
```

