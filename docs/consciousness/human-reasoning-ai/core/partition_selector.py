"""
Partition Selector: Navigate S-Entropy Space and Select Partitions

This module implements the core human-reasoning mechanism:
- Navigate to solutions via S-entropy descent (not search)
- Select partitions based on S-entropy compatibility (not optimization)

Key insight: O(1) navigation replaces O(N!) search
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Tuple
from .substrate_harvester import SEntropyCoordinates


@dataclass
class Partition:
    """A possible state/decision/outcome"""
    id: str
    content: Any
    s_entropy: float = 0.0  # S-entropy cost of this partition
    metadata: dict = field(default_factory=dict)


@dataclass
class NavigationResult:
    """Result of S-entropy navigation"""
    selected_partition: Partition
    s_coordinates: SEntropyCoordinates
    alternatives: List[Partition]
    confidence: float
    path_length: int  # Steps taken in navigation


class PartitionSelector:
    """
    Navigates S-entropy space and selects partitions.
    
    Instead of searching all possibilities:
        Solution = argmin over all states (O(N!))
    
    We navigate:
        Solution = descend(gradient(S)) (O(1))
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 convergence_threshold: float = 0.01,
                 max_steps: int = 100,
                 noise_scale: float = 0.05):
        """
        Initialize the partition selector.
        
        Args:
            learning_rate: Step size for gradient descent
            convergence_threshold: Stop when gradient magnitude below this
            max_steps: Maximum navigation steps
            noise_scale: Scale of non-deterministic perturbation
        """
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        self.noise_scale = noise_scale
    
    def compute_s_entropy(self, 
                         partition: Partition,
                         context: SEntropyCoordinates) -> float:
        """
        Compute S-entropy cost of a partition in current context.
        
        Args:
            partition: The partition to evaluate
            context: Current S-entropy coordinates
            
        Returns:
            S-entropy cost (lower = better fit)
        """
        # Base entropy from partition itself
        base_entropy = partition.s_entropy
        
        # Context-dependent adjustment
        # Partitions that align with current trajectory have lower entropy
        context_magnitude = context.magnitude()
        
        # Simple model: entropy increases with context magnitude
        # (more uncertainty = higher cost for any choice)
        adjusted = base_entropy * (1 + context_magnitude)
        
        return adjusted
    
    def compute_gradient(self,
                        current_coords: SEntropyCoordinates,
                        partitions: List[Partition]) -> np.ndarray:
        """
        Compute gradient of S-entropy field.
        
        Args:
            current_coords: Current position in S-space
            partitions: Available partitions to consider
            
        Returns:
            Gradient vector pointing toward increasing entropy
        """
        # Numerical gradient estimation
        epsilon = 0.01
        gradient = np.zeros(3)
        
        base_entropy = self._field_entropy(current_coords, partitions)
        
        # Partial derivatives
        for i, attr in enumerate(['S_k', 'S_t', 'S_e']):
            perturbed = SEntropyCoordinates(
                S_k=current_coords.S_k + (epsilon if i == 0 else 0),
                S_t=current_coords.S_t + (epsilon if i == 1 else 0),
                S_e=current_coords.S_e + (epsilon if i == 2 else 0)
            )
            perturbed_entropy = self._field_entropy(perturbed, partitions)
            gradient[i] = (perturbed_entropy - base_entropy) / epsilon
        
        return gradient
    
    def _field_entropy(self, 
                      coords: SEntropyCoordinates,
                      partitions: List[Partition]) -> float:
        """
        Compute total entropy field at a point.
        
        The field is shaped by all available partitions.
        Minimum is at the best-fit partition.
        """
        if not partitions:
            return coords.magnitude()
        
        # Field is superposition of partition contributions
        total = 0.0
        for p in partitions:
            # Each partition creates a "well" in the entropy field
            distance = abs(p.s_entropy - coords.magnitude())
            total += distance
        
        return total / len(partitions)
    
    def navigate(self,
                start_coords: SEntropyCoordinates,
                partitions: List[Partition]) -> Tuple[SEntropyCoordinates, int]:
        """
        Navigate through S-entropy space toward minimum.
        
        This is the O(1) operation that replaces O(N!) search.
        
        Args:
            start_coords: Starting position in S-space
            partitions: Available partitions (define the field)
            
        Returns:
            Tuple of (final coordinates, steps taken)
        """
        current = np.array([start_coords.S_k, start_coords.S_t, start_coords.S_e])
        
        for step in range(self.max_steps):
            # Current coordinates as object
            current_coords = SEntropyCoordinates(
                S_k=current[0],
                S_t=current[1],
                S_e=current[2]
            )
            
            # Compute gradient
            gradient = self.compute_gradient(current_coords, partitions)
            
            # Check convergence
            if np.linalg.norm(gradient) < self.convergence_threshold:
                return current_coords, step
            
            # Descend (move opposite to gradient)
            step_vector = -self.learning_rate * gradient
            
            # Add non-deterministic perturbation (human-like)
            noise = np.random.normal(0, self.noise_scale, 3)
            step_vector += noise
            
            # Update position
            current = current + step_vector
            
            # Clamp to valid range
            current = np.clip(current, 0, 1)
        
        final_coords = SEntropyCoordinates(
            S_k=current[0],
            S_t=current[1],
            S_e=current[2]
        )
        return final_coords, self.max_steps
    
    def select(self,
              coords: SEntropyCoordinates,
              partitions: List[Partition]) -> NavigationResult:
        """
        Select the best-fit partition at current coordinates.
        
        Args:
            coords: Current S-entropy coordinates
            partitions: Available partitions
            
        Returns:
            NavigationResult with selected partition and metadata
        """
        if not partitions:
            raise ValueError("No partitions available for selection")
        
        # Navigate to minimum
        final_coords, steps = self.navigate(coords, partitions)
        
        # Score each partition at final position
        scores = []
        for p in partitions:
            entropy = self.compute_s_entropy(p, final_coords)
            scores.append((p, entropy))
        
        # Sort by entropy (lower = better)
        scores.sort(key=lambda x: x[1])
        
        # Select best
        selected = scores[0][0]
        
        # Confidence from entropy difference
        if len(scores) > 1:
            gap = scores[1][1] - scores[0][1]
            confidence = min(1.0, gap * 10)  # Scale gap to confidence
        else:
            confidence = 1.0
        
        # Alternatives
        alternatives = [p for p, _ in scores[1:4]]  # Top 3 alternatives
        
        return NavigationResult(
            selected_partition=selected,
            s_coordinates=final_coords,
            alternatives=alternatives,
            confidence=confidence,
            path_length=steps
        )
    
    def navigate_and_select(self,
                           start_coords: SEntropyCoordinates,
                           partitions: List[Partition]) -> NavigationResult:
        """
        Combined navigation and selection in one call.
        
        Args:
            start_coords: Initial S-entropy coordinates
            partitions: Available partitions
            
        Returns:
            NavigationResult
        """
        return self.select(start_coords, partitions)


# Test if run directly
if __name__ == "__main__":
    from substrate_harvester import SubstrateHarvester
    
    # Create some test partitions
    partitions = [
        Partition("A", "Option A", s_entropy=0.2),
        Partition("B", "Option B", s_entropy=0.5),
        Partition("C", "Option C", s_entropy=0.8),
    ]
    
    # Get starting coordinates
    harvester = SubstrateHarvester()
    start = harvester.extract("What should I choose?")
    
    # Navigate and select
    selector = PartitionSelector()
    result = selector.navigate_and_select(start, partitions)
    
    print(f"Navigation Result:")
    print(f"  Selected: {result.selected_partition.id} ({result.selected_partition.content})")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Steps taken: {result.path_length}")
    print(f"  Final S-coords: ({result.s_coordinates.S_k:.3f}, {result.s_coordinates.S_t:.3f}, {result.s_coordinates.S_e:.3f})")

