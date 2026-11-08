"""
Semantic Processor

A language model based on S-entropy navigation and molecular structure prediction.

Key differences from traditional LLMs:
- No training required (molecular structure encodes relationships)
- No stored weights (harmonic coincidence networks)
- O(log n) complexity (categorical addressing)
- Hardware-grounded (atmospheric computing)

Two approaches:
1. Semantic Maxwell Demon: Navigation through S-entropy space
2. Molecular Semantics: Understanding as molecular structure prediction

Theoretical foundations:
- docs/semantic-maxwell-demon/semantic-maxwell-demons.tex
- docs/foundation/molecular-structure-prediction/molecular-structure-prediction.tex
"""

from .semantic_encoder import (
    SemanticEncoder,
    SemanticCoordinate,
    SemanticTrajectory,
)
from .semantic_gravity import (
    SemanticGravityField,
    SemanticAttractor,
)
from .empty_dictionary import (
    EmptyDictionarySynthesis,
    SemanticInterpretation,
)
from .semantic_language_model import (
    CategoricalLanguageModel,
    SemanticResponse,
)
from .molecular_semantics import (
    VirtualMolecule,
    HarmonicCoincidence,
    HarmonicCoincidenceNetwork,
    AtmosphericSemanticMemory,
    MolecularSemanticProcessor,
)

__all__ = [
    # Semantic Encoder
    'SemanticEncoder',
    'SemanticCoordinate',
    'SemanticTrajectory',
    # Semantic Gravity
    'SemanticGravityField',
    'SemanticAttractor',
    # Empty Dictionary
    'EmptyDictionarySynthesis',
    'SemanticInterpretation',
    # Categorical Language Model
    'CategoricalLanguageModel',
    'SemanticResponse',
    # Molecular Semantics
    'VirtualMolecule',
    'HarmonicCoincidence',
    'HarmonicCoincidenceNetwork',
    'AtmosphericSemanticMemory',
    'MolecularSemanticProcessor',
]

