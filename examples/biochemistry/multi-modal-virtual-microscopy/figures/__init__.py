"""
Figure generation module for Dodecapartite Virtual Microscopy paper.
"""

from .figure_01_constraint_architecture import generate_figure_1
from .figure_02_oxygen_triangulation import generate_figure_2
from .figure_03_constraint_integration import generate_figure_3
from .figure_04_experimental_validation import generate_figure_4
from .figure_05_biological_applications import generate_figure_5
from .figure_06_computational_implementation import generate_figure_6

__all__ = [
    'generate_figure_1',
    'generate_figure_2',
    'generate_figure_3',
    'generate_figure_4',
    'generate_figure_5',
    'generate_figure_6',
]
