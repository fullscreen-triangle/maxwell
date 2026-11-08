"""
Semantic Gravity Field

From the paper:
"We establish that semantic spaces exhibit thermodynamic structure with
well-defined potential energy functions, gravity fields, and equilibrium dynamics."

The gravity field guides navigation through semantic space:
- Semantic attractors pull toward coherent meanings
- Repulsors push away from contradictions/incoherence
- The field gradient determines navigation direction

This replaces the need for trained attention mechanisms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import time

from .semantic_encoder import SemanticCoordinate, SemanticTrajectory


@dataclass
class SemanticAttractor:
    """
    A point of semantic coherence that "attracts" nearby meanings.
    
    Attractors represent stable semantic concepts:
    - Questions attract toward answers
    - Subjects attract predicates
    - Premises attract conclusions
    """
    position: SemanticCoordinate
    strength: float  # How strongly it attracts
    category: str  # What type of attractor
    
    # Attractor shape
    radius: float = 1.0  # Range of influence
    falloff: str = "inverse_square"  # How strength decays with distance
    
    # Metadata
    label: str = ""
    created_at: float = field(default_factory=time.time)
    
    def potential_at(self, point: SemanticCoordinate) -> float:
        """
        Calculate gravitational potential at a point.
        
        Negative potential = attraction toward this attractor.
        """
        distance = self.position.distance_to(point)
        
        if self.falloff == "inverse_square":
            # Standard gravity: V = -GM/r
            return -self.strength / (distance + 0.1)
        
        elif self.falloff == "gaussian":
            # Soft falloff: V = -A * exp(-r²/2σ²)
            return -self.strength * np.exp(-distance**2 / (2 * self.radius**2))
        
        elif self.falloff == "linear":
            # Linear within radius, zero outside
            if distance < self.radius:
                return -self.strength * (1 - distance / self.radius)
            return 0.0
        
        else:
            return -self.strength / (distance + 0.1)
    
    def gradient_at(self, point: SemanticCoordinate) -> np.ndarray:
        """
        Calculate gradient of potential at a point.
        
        Gradient points toward lower potential (toward attractor).
        """
        pos_vec = self.position.to_vector()
        point_vec = point.to_vector()
        
        direction = pos_vec - point_vec
        distance = np.linalg.norm(direction) + 1e-10
        direction = direction / distance  # Normalize
        
        if self.falloff == "inverse_square":
            magnitude = self.strength / (distance**2 + 0.1)
        elif self.falloff == "gaussian":
            magnitude = self.strength * distance / self.radius**2 * np.exp(-distance**2 / (2 * self.radius**2))
        elif self.falloff == "linear":
            magnitude = self.strength / self.radius if distance < self.radius else 0.0
        else:
            magnitude = self.strength / (distance**2 + 0.1)
        
        return direction * magnitude


class SemanticGravityField:
    """
    The complete gravity field over semantic space.
    
    From the paper:
    "Semantic gravity field construction defining potential energy landscapes"
    
    The field is the sum of all attractors and repulsors.
    Navigation follows the gradient toward minimum potential.
    """
    
    def __init__(self):
        """Initialize the gravity field."""
        self.attractors: List[SemanticAttractor] = []
        self.repulsors: List[SemanticAttractor] = []  # Negative strength attractors
        
        # Dynamic attractors created during navigation
        self._dynamic_attractors: List[SemanticAttractor] = []
        
        # Field statistics
        self._total_potential_calls = 0
        self._total_gradient_calls = 0
    
    def add_attractor(
        self,
        position: SemanticCoordinate,
        strength: float = 1.0,
        category: str = "general",
        radius: float = 1.0,
        label: str = "",
    ):
        """Add an attractor to the field."""
        attractor = SemanticAttractor(
            position=position,
            strength=abs(strength),
            category=category,
            radius=radius,
            label=label,
        )
        self.attractors.append(attractor)
    
    def add_repulsor(
        self,
        position: SemanticCoordinate,
        strength: float = 1.0,
        category: str = "incoherence",
        radius: float = 1.0,
        label: str = "",
    ):
        """Add a repulsor to the field."""
        repulsor = SemanticAttractor(
            position=position,
            strength=-abs(strength),  # Negative for repulsion
            category=category,
            radius=radius,
            label=label,
        )
        self.repulsors.append(repulsor)
    
    def add_dynamic_attractor(self, attractor: SemanticAttractor):
        """Add a temporary attractor that can be cleared."""
        self._dynamic_attractors.append(attractor)
    
    def clear_dynamic_attractors(self):
        """Clear all dynamic attractors."""
        self._dynamic_attractors.clear()
    
    def potential_at(self, point: SemanticCoordinate) -> float:
        """
        Calculate total gravitational potential at a point.
        
        Lower potential = more semantically coherent.
        """
        self._total_potential_calls += 1
        
        total = 0.0
        
        for attractor in self.attractors:
            total += attractor.potential_at(point)
        
        for repulsor in self.repulsors:
            total += repulsor.potential_at(point)
        
        for dynamic in self._dynamic_attractors:
            total += dynamic.potential_at(point)
        
        return total
    
    def gradient_at(self, point: SemanticCoordinate) -> np.ndarray:
        """
        Calculate total gradient at a point.
        
        Gradient points toward lower potential (more coherent meaning).
        """
        self._total_gradient_calls += 1
        
        total = np.zeros(6)
        
        for attractor in self.attractors:
            total += attractor.gradient_at(point)
        
        for repulsor in self.repulsors:
            total -= repulsor.gradient_at(point)  # Reverse for repulsion
        
        for dynamic in self._dynamic_attractors:
            total += dynamic.gradient_at(point)
        
        return total
    
    def navigate_step(
        self,
        current: SemanticCoordinate,
        step_size: float = 0.1,
        noise_scale: float = 0.01,
    ) -> SemanticCoordinate:
        """
        Take one navigation step following the gradient.
        
        Includes small noise for exploration (stochastic sampling).
        """
        gradient = self.gradient_at(current)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-10:
            gradient = gradient / grad_norm
        
        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, size=6)
        
        # Step in gradient direction
        current_vec = current.to_vector()
        new_vec = current_vec + step_size * gradient + noise
        
        return SemanticCoordinate.from_vector(
            new_vec,
            source=current.source_text,
            layer=current.layer
        )
    
    def navigate_to_minimum(
        self,
        start: SemanticCoordinate,
        max_steps: int = 100,
        tolerance: float = 1e-4,
        step_size: float = 0.1,
    ) -> Tuple[SemanticCoordinate, List[SemanticCoordinate]]:
        """
        Navigate from start to local potential minimum.
        
        This finds the "categorical completion" - the stable semantic endpoint.
        
        Returns:
            Tuple of (final_position, trajectory)
        """
        trajectory = [start]
        current = start
        
        for _ in range(max_steps):
            # Take step
            next_pos = self.navigate_step(current, step_size, noise_scale=0.001)
            trajectory.append(next_pos)
            
            # Check convergence
            if current.distance_to(next_pos) < tolerance:
                break
            
            current = next_pos
        
        return current, trajectory
    
    def create_question_attractor(self, question_coord: SemanticCoordinate) -> SemanticAttractor:
        """
        Create an attractor that pulls toward answers to a question.
        
        Questions create gravitational wells that answers fall into.
        """
        # Question attractor is offset in certain dimensions
        # (answers tend to be more "concrete" than questions)
        pos_vec = question_coord.to_vector()
        
        # Shift toward higher semantic density, lower entropy
        pos_vec[4] *= 1.5  # S_sem: answers have more semantic content
        pos_vec[2] *= 0.7  # S_e: answers have less ambiguity
        
        answer_pos = SemanticCoordinate.from_vector(pos_vec, "answer_attractor")
        
        return SemanticAttractor(
            position=answer_pos,
            strength=2.0,
            category="question_answer",
            radius=2.0,
            falloff="gaussian",
            label=f"answer_to:{question_coord.source_text[:20]}"
        )
    
    def create_context_field(self, context_trajectory: SemanticTrajectory):
        """
        Create attractors from a context trajectory.
        
        Each point in the context creates an attractor,
        biasing navigation toward contextually relevant meanings.
        """
        self.clear_dynamic_attractors()
        
        # Create attractor at each context point with decaying strength
        n = len(context_trajectory.coordinates)
        
        for i, coord in enumerate(context_trajectory.coordinates):
            # More recent context is stronger
            recency = (i + 1) / n
            strength = 0.5 * recency
            
            attractor = SemanticAttractor(
                position=coord,
                strength=strength,
                category="context",
                radius=1.5,
                falloff="gaussian",
                label=f"context_{i}"
            )
            
            self.add_dynamic_attractor(attractor)
    
    def find_semantic_basin(
        self,
        center: SemanticCoordinate,
        n_samples: int = 20,
        radius: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Explore the semantic basin around a point.
        
        Returns information about the local potential landscape.
        """
        # Sample points around center
        center_vec = center.to_vector()
        samples = []
        potentials = []
        
        for _ in range(n_samples):
            # Random direction
            direction = np.random.randn(6)
            direction = direction / np.linalg.norm(direction)
            
            # Random distance
            distance = np.random.uniform(0, radius)
            
            sample_vec = center_vec + direction * distance
            sample = SemanticCoordinate.from_vector(sample_vec)
            
            samples.append(sample)
            potentials.append(self.potential_at(sample))
        
        # Analyze basin
        center_potential = self.potential_at(center)
        min_potential = min(potentials)
        max_potential = max(potentials)
        
        # Find minimum location
        min_idx = np.argmin(potentials)
        basin_minimum = samples[min_idx]
        
        return {
            'center': center,
            'center_potential': center_potential,
            'min_potential': min_potential,
            'max_potential': max_potential,
            'basin_depth': max_potential - min_potential,
            'basin_minimum': basin_minimum,
            'is_local_minimum': center_potential <= min_potential + 1e-6,
            'gradient_magnitude': np.linalg.norm(self.gradient_at(center)),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get field statistics."""
        return {
            'n_attractors': len(self.attractors),
            'n_repulsors': len(self.repulsors),
            'n_dynamic': len(self._dynamic_attractors),
            'potential_calls': self._total_potential_calls,
            'gradient_calls': self._total_gradient_calls,
        }

