# Maxwell Validation

**Python validation suite for the Complementarity-Aware Processor**

This package provides independent validation of the theoretical framework resolving Maxwell's Demon through categorical phase-lock topology.

## Installation

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Usage

### Run All Validations

```bash
python -m maxwell_validation.dissolution
```

### Individual Validators

```python
from maxwell_validation import (
    CategoricalValidator,
    KineticValidator,
    ComplementarityValidator,
    DissolutionValidator,
)

# Validate categorical operations
cat = CategoricalValidator()
results = cat.run_all_validations()

# Validate kinetic operations
kin = KineticValidator(temperature=300.0)
results = kin.run_all_validations()

# Validate complementarity
comp = ComplementarityValidator()
results = comp.run_all_validations()

# Validate all seven dissolution arguments
diss = DissolutionValidator()
results = diss.run_all_validations()
diss.print_summary(results)
```

## Validation Modules

### `categorical.py`

Validates the categorical engine:
- Phase-lock network construction (position-based, NOT velocity-based)
- Categorical completion irreversibility
- Network density increases entropy
- Conjugate relationship between faces

### `kinetic.py`

Validates the kinetic engine:
- Maxwell-Boltzmann velocity distribution
- Temperature computation from kinetic energies
- Demon sorting (what Maxwell would see)
- Retrieval paradox demonstration

### `complementarity.py`

Validates complementarity constraints:
- Cannot observe both faces simultaneously
- Face switching works correctly
- Derivation vs measurement distinction
- Ammeter/voltmeter analogy

### `dissolution.py`

Validates all seven dissolution arguments:

1. **Temporal Triviality**: Fluctuations produce same configurations
2. **Phase-Lock Temperature Independence**: Same arrangement at any T
3. **Retrieval Paradox**: Cannot outpace thermal equilibration
4. **Dissolution of Observation**: Topology determines accessibility
5. **Dissolution of Decision**: Pathways follow automatically
6. **Dissolution of Second Law**: Categorical entropy increases
7. **Information Complementarity**: Demon is projection artifact

## Testing

```bash
pytest -v
```

## Output Example

```
======================================================================
THE SEVEN-FOLD DISSOLUTION OF MAXWELL'S DEMON
======================================================================

✓ Argument 1: TEMPORAL_TRIVIALITY
  Temporal triviality validated
  Sorted configurations occur naturally through fluctuations.

✓ Argument 2: PHASE_LOCK_TEMPERATURE_INDEPENDENCE
  Phase-lock temperature independence validated
  Same positions produce same phase-lock network regardless of temperature.

✓ Argument 3: RETRIEVAL_PARADOX
  Retrieval paradox validated
  Sorting ratio stays ~0.50. Cannot outpace equilibration.

✓ Argument 4: DISSOLUTION_OF_OBSERVATION
  Dissolution of observation validated
  Network determined purely by topology, not velocity measurement.

✓ Argument 5: DISSOLUTION_OF_DECISION
  Dissolution of decision validated
  Path followed automatically from topology, no decisions made.

✓ Argument 6: DISSOLUTION_OF_SECOND_LAW
  Dissolution of second law validated
  Entropy increased. Second law upheld.

✓ Argument 7: INFORMATION_COMPLEMENTARITY
  Information complementarity validated
  The demon is the projection of hidden categorical dynamics.

======================================================================
ALL SEVEN ARGUMENTS VALIDATED
THERE IS NO DEMON.
======================================================================
```

## License

MIT

