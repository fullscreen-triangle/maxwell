"""
Categorical Computer

A complete computing system based on S-entropy navigation and categorical completion.

Components:
- Processor: Categorical completion processor (oscillator-based)
- Memory: S-RAM (precision-by-difference addressing)
- Translator: Problem → Categorical Structure → Solution

The key difference from classical computing:
- Classical: Specify HOW to solve (algorithm)
- Categorical: Specify WHAT solution looks like (structure)

The system navigates S-entropy space to find categorical completions
that satisfy the problem constraints.
"""

from .problem_translator import (
    ProblemTranslator,
    CategoricalProblem,
    CategoricalSolution,
    ProblemType,
)
from .categorical_runtime import (
    CategoricalRuntime,
    ExecutionContext,
    RuntimeResult,
)
from .problem_types import (
    OptimizationProblem,
    SearchProblem,
    PatternMatchProblem,
    ConstraintProblem,
    BiologicalProblem,
)

__all__ = [
    'ProblemTranslator',
    'CategoricalProblem',
    'CategoricalSolution',
    'ProblemType',
    'CategoricalRuntime',
    'ExecutionContext',
    'RuntimeResult',
    'OptimizationProblem',
    'SearchProblem',
    'PatternMatchProblem',
    'ConstraintProblem',
    'BiologicalProblem',
]


