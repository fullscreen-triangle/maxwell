# Validation Experiments for Categorical Completion Mechanics

This directory contains validation experiments for the paper "Categorical Completion Mechanics: Path Independence and Sufficiency in Bounded Information Systems."

## Overview

The validation experiments test the key theorems and principles from the paper:

1. **Path Independence Theorem**: Infinite input configurations → finite output states
2. **Sufficiency Principle**: Functional adequacy over representational accuracy
3. **Oscillatory Aperture Properties**: Phase-lock degeneracy, temperature independence
4. **Consensus Calibration**: Truth as equilibrium in multi-agent systems
5. **Categorical Completion**: Irreversibility and temporal direction

## Running the Experiments

### Prerequisites

The experiments use the existing virtual gas ensemble, virtual aperture, and categorical instrument frameworks from `poincare/src/`.

### Basic Usage

```bash
# From the project root
cd ramanujin/publication/perception-propagation
python validation_experiments.py
```

This will run all validation experiments and save results to `validation_results.json`.

### Individual Experiments

You can also run specific validators:

```python
from validation_experiments import (
    PathIndependenceValidator,
    SufficiencyPrincipleValidator,
    OscillatoryApertureValidator,
    ConsensusCalibrationValidator,
    CategoricalCompletionValidator
)

# Path Independence
path_validator = PathIndependenceValidator()
result = path_validator.infinite_input_substitutability_test(n_inputs=1000)
print(f"Success: {result.success}")

# Sufficiency Principle
sufficiency_validator = SufficiencyPrincipleValidator()
result = sufficiency_validator.sufficiency_vs_accuracy_test()
print(f"Success: {result.success}")

# Oscillatory Apertures
aperture_validator = OscillatoryApertureValidator()
result = aperture_validator.temperature_independence_test()
print(f"Success: {result.success}")

# Consensus Calibration
consensus_validator = ConsensusCalibrationValidator()
result = consensus_validator.multi_agent_consensus_test(n_agents=10)
print(f"Success: {result.success}")

# Categorical Completion
completion_validator = CategoricalCompletionValidator()
result = completion_validator.categorical_irreversibility_test()
print(f"Success: {result.success}")
```

## Experiment Details

### 1. Path Independence Experiments

**Infinite Input Substitutability Test**
- Generates 1000+ different input configurations
- Tests that all produce identical output state
- Validates: `unique_outputs == 1` and `output_convergence > 0.99`

**Output Indeterminacy Test**
- Given output state O, attempts reverse mapping
- Validates: Finds multiple inputs producing same output (infinite in limit)

**Zoo Scenario Simulation**
- Three agents with different inputs (visual, learned, social)
- Validates: All converge to same output despite different paths

### 2. Sufficiency Principle Experiments

**Sufficiency vs Accuracy Test**
- Tests scenarios where accurate representation is impossible
- Validates: System achieves sufficiency (`> 0.9`) even when accuracy is low (`< 0.1`)

**Potential Field Gradient Test**
- Tracks state transitions following potential field gradients
- Validates: Transitions decrease potential (`mean_decrease > 0`) and follow gradients (`success_rate > 0.8`)

### 3. Oscillatory Aperture Experiments

**Phase-Lock Degeneracy Test**
- Tests that ~10^6 different weak force configurations produce same oscillatory signature
- Validates: Equivalence class size `> 100,000` (approaching 10^6)

**Temperature Independence Test**
- Tests aperture selectivity at different temperatures
- Validates: Selectivity variance `< 0.01` (temperature-independent)

**Cascade Amplification Test**
- Tests exponential selectivity amplification: `S_total = s^n`
- Validates: Measured selectivity matches theoretical prediction (ratio ~1.0)

### 4. Consensus Calibration Experiments

**Multi-Agent Consensus Test**
- Tests consensus formation across agent network
- Validates: Final consensus `> 0.95` and consensus increases over time (`> 0.3`)

### 5. Categorical Completion Experiments

**Categorical Irreversibility Test**
- Tests that completed states cannot be re-occupied
- Validates: Reoccupation rate `< 0.01` (essentially zero)

## Expected Results

All experiments should pass with the following criteria:

- **Path Independence**: 100% output convergence, infinite input variety
- **Sufficiency**: High sufficiency (>0.9) even with low accuracy (<0.1)
- **Oscillatory Apertures**: Temperature-independent, exponential cascade amplification
- **Consensus**: High final consensus (>0.95) with increasing alignment over time
- **Categorical Completion**: Zero reoccupation of completed states

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "category": [
    {
      "experiment_name": "Experiment Name",
      "success": true,
      "validation_status": "PASS",
      "measurements": {
        "key": "value"
      },
      "predictions": {
        "expected": "value"
      }
    }
  ]
}
```

## Troubleshooting

If imports fail, ensure:
1. You're running from the project root or have `poincare/src` in your Python path
2. The virtual frameworks are properly installed
3. All dependencies (numpy, etc.) are available

The script includes fallback stubs for testing, but full functionality requires the actual frameworks.

