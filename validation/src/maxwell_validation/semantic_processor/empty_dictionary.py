"""
Empty Dictionary Synthesis

The core insight: MEANING CAN BE GENERATED WITHOUT STORED KNOWLEDGE.

From the paper:
"The empty dictionary synthesis architecture proves that semantic understanding
can be generated in real-time through Bayesian inference on coordinate samples
without requiring stored semantic knowledge, pre-trained models, or domain-specific
databases."

This is the fundamental difference from LLMs:
- LLM: Look up learned patterns in weights
- Categorical: Navigate S-entropy space to find meaning

The dictionary is "empty" because we don't store semantic mappings.
Meaning emerges from the navigation process itself.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time

from .semantic_encoder import SemanticEncoder, SemanticCoordinate, SemanticTrajectory
from .semantic_gravity import SemanticGravityField, SemanticAttractor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from categorical_memory.s_entropy_address import SEntropyAddress
from categorical_memory.precision_calculator import PrecisionByDifferenceCalculator
from categorical_memory.hardware_oscillator import HardwareOscillatorCapture


@dataclass
class SemanticInterpretation:
    """
    An interpretation synthesized from empty dictionary processing.
    
    This is what we produce instead of "generated text":
    - Position in S-entropy space (the meaning)
    - Confidence based on navigation convergence
    - Trajectory showing how meaning was derived
    """
    # The synthesized meaning
    semantic_position: SemanticCoordinate
    
    # Confidence metrics
    confidence: float  # 0-1, based on convergence quality
    coherence: float   # Semantic coherence of final position
    
    # Navigation trajectory
    trajectory: List[SemanticCoordinate] = field(default_factory=list)
    
    # Analysis
    semantic_category: str = ""
    interpretation_type: str = ""  # "answer", "completion", "continuation", etc.
    
    # Metadata
    input_text: str = ""
    synthesis_time: float = 0.0
    navigation_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': {
                'S_k': self.semantic_position.S_k,
                'S_t': self.semantic_position.S_t,
                'S_e': self.semantic_position.S_e,
                'S_syn': self.semantic_position.S_syn,
                'S_sem': self.semantic_position.S_sem,
                'S_prag': self.semantic_position.S_prag,
            },
            'confidence': self.confidence,
            'coherence': self.coherence,
            'category': self.semantic_category,
            'type': self.interpretation_type,
            'input': self.input_text,
            'time': self.synthesis_time,
            'steps': self.navigation_steps,
        }


class EmptyDictionarySynthesis:
    """
    Empty Dictionary Synthesis Engine.
    
    Generates semantic interpretations through navigation rather than retrieval.
    
    Process:
    1. Encode input text to S-entropy coordinates
    2. Create gravity field from context and constraints
    3. Navigate to semantic minimum (completion point)
    4. Extract interpretation from final position
    
    No stored semantic knowledge. No trained weights.
    Just navigation through coordinate space.
    """
    
    def __init__(
        self,
        oscillator: Optional[HardwareOscillatorCapture] = None,
    ):
        """Initialize the synthesis engine."""
        self.oscillator = oscillator or HardwareOscillatorCapture()
        self.encoder = SemanticEncoder(self.oscillator)
        self.gravity = SemanticGravityField()
        self.precision_calc = PrecisionByDifferenceCalculator(self.oscillator)
        
        # Calibrate hardware
        self.oscillator.calibrate(duration=0.1)
        
        # Synthesis history
        self.synthesis_history: List[SemanticInterpretation] = []
    
    def synthesize(
        self,
        input_text: str,
        context: Optional[str] = None,
        interpretation_type: str = "completion",
        max_steps: int = 50,
    ) -> SemanticInterpretation:
        """
        Synthesize a semantic interpretation from input.
        
        Args:
            input_text: The input to interpret
            context: Optional context to bias interpretation
            interpretation_type: What kind of interpretation to seek
            max_steps: Maximum navigation steps
            
        Returns:
            SemanticInterpretation with synthesized meaning
        """
        start_time = time.time()
        
        # Step 1: Encode input
        encoded = self.encoder.encode(input_text)
        
        # Get starting position based on encoding type
        if encoded['type'] == 'word':
            start_position = encoded['coordinate']
        elif encoded['type'] == 'phrase':
            start_position = encoded['coordinate']
        elif encoded['type'] == 'sentence':
            traj = encoded['trajectory']
            start_position = traj.endpoint or traj.centroid
        elif encoded['type'] == 'paragraph':
            # Use centroid of all trajectories
            all_coords = []
            for t in encoded['trajectory_bundle']:
                all_coords.extend(t.coordinates)
            vectors = np.array([c.to_vector() for c in all_coords])
            mean_vec = np.mean(vectors, axis=0)
            start_position = SemanticCoordinate.from_vector(mean_vec, input_text)
        elif encoded['type'] == 'document':
            start_position = encoded['manifold']['centroid']
        else:
            start_position = SemanticCoordinate(0, 0, 0, source_text=input_text)
        
        # Step 2: Create gravity field
        self.gravity.clear_dynamic_attractors()
        
        # Add context attractors if provided
        if context:
            context_encoded = self.encoder.encode(context)
            if context_encoded['type'] == 'sentence':
                self.gravity.create_context_field(context_encoded['trajectory'])
            elif context_encoded['type'] == 'paragraph':
                for traj in context_encoded['trajectory_bundle']:
                    self.gravity.create_context_field(traj)
        
        # Add interpretation-type-specific attractors
        if interpretation_type == "answer":
            # Create answer attractor (more concrete, less ambiguous)
            answer_attractor = self.gravity.create_question_attractor(start_position)
            self.gravity.add_dynamic_attractor(answer_attractor)
            
        elif interpretation_type == "continuation":
            # Create continuation attractor (extends trajectory)
            if encoded['type'] == 'sentence':
                completion = encoded['trajectory'].predict_completion()
                if completion:
                    attractor = SemanticAttractor(
                        position=completion,
                        strength=1.5,
                        category="continuation",
                        radius=2.0,
                        falloff="gaussian"
                    )
                    self.gravity.add_dynamic_attractor(attractor)
        
        elif interpretation_type == "summary":
            # Create summary attractor (higher semantic density, lower entropy)
            summary_vec = start_position.to_vector()
            summary_vec[4] *= 2.0  # Higher semantic density
            summary_vec[2] *= 0.5  # Lower entropy
            summary_vec[1] *= 0.7  # Narrower interpretation
            
            summary_pos = SemanticCoordinate.from_vector(summary_vec, "summary")
            attractor = SemanticAttractor(
                position=summary_pos,
                strength=2.0,
                category="summary",
                radius=1.5,
                falloff="gaussian"
            )
            self.gravity.add_dynamic_attractor(attractor)
        
        # Step 3: Navigate to semantic minimum
        final_position, trajectory = self.gravity.navigate_to_minimum(
            start=start_position,
            max_steps=max_steps,
            step_size=0.1,
            tolerance=1e-4,
        )
        
        # Step 4: Compute confidence and coherence
        # Confidence: based on convergence (smaller final gradient = higher confidence)
        final_gradient = np.linalg.norm(self.gravity.gradient_at(final_position))
        confidence = 1.0 / (1.0 + final_gradient)
        
        # Coherence: based on potential depth (deeper = more coherent)
        basin = self.gravity.find_semantic_basin(final_position, n_samples=10)
        coherence = 1.0 / (1.0 + abs(basin['center_potential']))
        
        # Step 5: Determine semantic category
        semantic_category = self._categorize_position(final_position)
        
        # Build interpretation
        interpretation = SemanticInterpretation(
            semantic_position=final_position,
            confidence=confidence,
            coherence=coherence,
            trajectory=trajectory,
            semantic_category=semantic_category,
            interpretation_type=interpretation_type,
            input_text=input_text,
            synthesis_time=time.time() - start_time,
            navigation_steps=len(trajectory),
        )
        
        self.synthesis_history.append(interpretation)
        return interpretation
    
    def synthesize_response(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> SemanticInterpretation:
        """
        Synthesize a response to a query.
        
        This is the main interface for question-answering.
        """
        # Determine if this is a question
        is_question = query.strip().endswith('?')
        
        if is_question:
            return self.synthesize(query, context, interpretation_type="answer")
        else:
            return self.synthesize(query, context, interpretation_type="continuation")
    
    def synthesize_summary(
        self,
        text: str,
    ) -> SemanticInterpretation:
        """Synthesize a summary of text."""
        return self.synthesize(text, interpretation_type="summary")
    
    def _categorize_position(self, position: SemanticCoordinate) -> str:
        """Categorize a semantic position."""
        vec = position.to_vector()
        
        # Simple categorization based on dominant dimensions
        categories = {
            'abstract': (vec[2] > 1.0 and vec[4] < 0.5),  # High entropy, low semantic
            'concrete': (vec[2] < 0.5 and vec[4] > 1.0),  # Low entropy, high semantic
            'dynamic': (vec[0] > 1.0),  # High kinetic
            'stable': (vec[0] < 0.5),   # Low kinetic
            'broad': (vec[1] > 1.0),    # High thermal (broad interpretation)
            'narrow': (vec[1] < 0.5),   # Low thermal (specific)
        }
        
        active_categories = [cat for cat, condition in categories.items() if condition]
        
        if not active_categories:
            return "neutral"
        
        return "_".join(active_categories)
    
    def compare_interpretations(
        self,
        interp1: SemanticInterpretation,
        interp2: SemanticInterpretation,
    ) -> Dict[str, Any]:
        """Compare two interpretations."""
        distance = interp1.semantic_position.distance_to(interp2.semantic_position)
        
        vec1 = interp1.semantic_position.to_vector()
        vec2 = interp2.semantic_position.to_vector()
        
        # Cosine similarity
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        
        return {
            'distance': distance,
            'similarity': (cos_sim + 1) / 2,  # Normalize to 0-1
            'category_match': interp1.semantic_category == interp2.semantic_category,
            'confidence_diff': abs(interp1.confidence - interp2.confidence),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        if not self.synthesis_history:
            return {'total_syntheses': 0}
        
        times = [i.synthesis_time for i in self.synthesis_history]
        confidences = [i.confidence for i in self.synthesis_history]
        steps = [i.navigation_steps for i in self.synthesis_history]
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'avg_time': np.mean(times),
            'avg_confidence': np.mean(confidences),
            'avg_steps': np.mean(steps),
            'gravity_stats': self.gravity.get_statistics(),
        }

