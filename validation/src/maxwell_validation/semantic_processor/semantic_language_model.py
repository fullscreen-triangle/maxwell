"""
Categorical Language Model

A complete language model based on S-entropy navigation.

This is NOT a traditional LLM:
- No training on massive datasets
- No billions of parameters
- No attention mechanisms
- No next-token prediction

Instead:
- Text → S-entropy coordinates
- Navigation through semantic space
- Categorical completion finds meaning
- Hardware oscillations ground the process

This is what the paper calls "Empty Dictionary Synthesis" - 
generating semantic understanding without stored knowledge.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterator
import time

from .semantic_encoder import SemanticEncoder, SemanticCoordinate, SemanticTrajectory
from .semantic_gravity import SemanticGravityField, SemanticAttractor
from .empty_dictionary import EmptyDictionarySynthesis, SemanticInterpretation

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from categorical_memory.hardware_oscillator import HardwareOscillatorCapture
from categorical_memory.memory_controller import CategoricalMemoryController


@dataclass
class SemanticResponse:
    """
    A response from the categorical language model.
    
    Contains the semantic interpretation plus metadata
    about how it was generated.
    """
    # The interpretation
    interpretation: SemanticInterpretation
    
    # Response characteristics
    semantic_vector: np.ndarray
    confidence: float
    coherence: float
    
    # Generation metadata
    input_query: str
    context_used: Optional[str] = None
    generation_time: float = 0.0
    navigation_steps: int = 0
    
    # Semantic analysis
    response_category: str = ""
    semantic_distance_from_query: float = 0.0
    
    def to_readable(self) -> str:
        """
        Convert semantic position to a readable description.
        
        Since we don't generate text, we describe the meaning space.
        """
        pos = self.interpretation.semantic_position
        
        lines = [
            f"Semantic Response Analysis",
            f"=" * 40,
            f"Query: {self.input_query[:50]}...",
            f"",
            f"Meaning Space Position:",
            f"  S_k (kinetic/dynamic):     {pos.S_k:.4f}",
            f"  S_t (thermal/breadth):     {pos.S_t:.4f}",
            f"  S_e (entropy/ambiguity):   {pos.S_e:.4f}",
            f"  S_syn (syntactic):         {pos.S_syn:.4f}",
            f"  S_sem (semantic density):  {pos.S_sem:.4f}",
            f"  S_prag (pragmatic):        {pos.S_prag:.4f}",
            f"",
            f"Interpretation:",
            f"  Category: {self.interpretation.semantic_category}",
            f"  Type: {self.interpretation.interpretation_type}",
            f"  Confidence: {self.confidence:.2%}",
            f"  Coherence: {self.coherence:.2%}",
            f"",
            f"Navigation:",
            f"  Steps: {self.navigation_steps}",
            f"  Time: {self.generation_time:.4f}s",
            f"  Distance from query: {self.semantic_distance_from_query:.4f}",
        ]
        
        return "\n".join(lines)


class CategoricalLanguageModel:
    """
    The complete Categorical Language Model.
    
    Architecture:
    1. Semantic Encoder: Text → S-entropy coordinates
    2. Gravity Field: Creates semantic landscape
    3. Empty Dictionary: Synthesizes meaning through navigation
    4. Memory Controller: Manages context through S-RAM
    
    This replaces:
    - Tokenizer → Semantic Encoder
    - Embedding layer → S-entropy mapping
    - Transformer layers → Gravity navigation
    - Output layer → Categorical completion
    - Softmax → Confidence from convergence
    """
    
    def __init__(
        self,
        max_context_length: int = 1000,
        max_navigation_steps: int = 100,
    ):
        """
        Initialize the categorical language model.
        
        Args:
            max_context_length: Maximum context to maintain
            max_navigation_steps: Maximum steps for navigation
        """
        self.max_context_length = max_context_length
        self.max_navigation_steps = max_navigation_steps
        
        # Core components
        self.oscillator = HardwareOscillatorCapture()
        self.encoder = SemanticEncoder(self.oscillator)
        self.synthesis = EmptyDictionarySynthesis(self.oscillator)
        self.memory = CategoricalMemoryController()
        
        # Conversation context
        self._context_buffer: List[Tuple[str, SemanticInterpretation]] = []
        
        # Calibrate hardware
        self.oscillator.calibrate(duration=0.1)
        
        # Statistics
        self._total_responses = 0
        self._total_time = 0.0
    
    def _build_context_string(self) -> str:
        """Build context string from buffer."""
        if not self._context_buffer:
            return ""
        
        # Take recent context up to limit
        context_parts = []
        total_len = 0
        
        for text, _ in reversed(self._context_buffer):
            if total_len + len(text) > self.max_context_length:
                break
            context_parts.append(text)
            total_len += len(text)
        
        return " ".join(reversed(context_parts))
    
    def process(
        self,
        query: str,
        use_context: bool = True,
    ) -> SemanticResponse:
        """
        Process a query and generate a semantic response.
        
        This is the main interface - equivalent to LLM's generate().
        
        Args:
            query: The input query
            use_context: Whether to use conversation context
            
        Returns:
            SemanticResponse with the interpreted meaning
        """
        start_time = time.time()
        
        # Build context
        context = self._build_context_string() if use_context else None
        
        # Encode query
        query_encoded = self.encoder.encode(query)
        
        # Get query position
        if query_encoded['type'] == 'sentence':
            query_position = query_encoded['trajectory'].endpoint
        elif query_encoded['type'] in ['word', 'phrase']:
            query_position = query_encoded['coordinate']
        else:
            # For longer texts, use centroid
            if query_encoded['type'] == 'document':
                query_position = query_encoded['manifold']['centroid']
            else:
                query_position = SemanticCoordinate(0, 0, 0, source_text=query)
        
        # Synthesize response
        interpretation = self.synthesis.synthesize_response(
            query=query,
            context=context,
        )
        
        # Calculate distance from query
        semantic_distance = query_position.distance_to(interpretation.semantic_position)
        
        # Store in memory
        self.memory.store(f"query_{self._total_responses}", query)
        self.memory.store(f"response_{self._total_responses}", interpretation)
        
        # Add to context
        self._context_buffer.append((query, interpretation))
        if len(self._context_buffer) > 10:  # Keep last 10 exchanges
            self._context_buffer.pop(0)
        
        # Build response
        generation_time = time.time() - start_time
        
        response = SemanticResponse(
            interpretation=interpretation,
            semantic_vector=interpretation.semantic_position.to_vector(),
            confidence=interpretation.confidence,
            coherence=interpretation.coherence,
            input_query=query,
            context_used=context,
            generation_time=generation_time,
            navigation_steps=interpretation.navigation_steps,
            response_category=interpretation.semantic_category,
            semantic_distance_from_query=semantic_distance,
        )
        
        self._total_responses += 1
        self._total_time += generation_time
        
        return response
    
    def answer(self, question: str) -> SemanticResponse:
        """Answer a question."""
        return self.process(question)
    
    def continue_text(self, text: str) -> SemanticResponse:
        """Generate a continuation of text."""
        interpretation = self.synthesis.synthesize(
            text,
            interpretation_type="continuation"
        )
        
        return SemanticResponse(
            interpretation=interpretation,
            semantic_vector=interpretation.semantic_position.to_vector(),
            confidence=interpretation.confidence,
            coherence=interpretation.coherence,
            input_query=text,
            generation_time=interpretation.synthesis_time,
            navigation_steps=interpretation.navigation_steps,
            response_category=interpretation.semantic_category,
            semantic_distance_from_query=0.0,
        )
    
    def summarize(self, text: str) -> SemanticResponse:
        """Summarize text."""
        interpretation = self.synthesis.synthesize_summary(text)
        
        return SemanticResponse(
            interpretation=interpretation,
            semantic_vector=interpretation.semantic_position.to_vector(),
            confidence=interpretation.confidence,
            coherence=interpretation.coherence,
            input_query=text,
            generation_time=interpretation.synthesis_time,
            navigation_steps=interpretation.navigation_steps,
            response_category=interpretation.semantic_category,
            semantic_distance_from_query=0.0,
        )
    
    def compare(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts semantically.
        
        Returns semantic similarity and analysis.
        """
        # Encode both
        encoded1 = self.encoder.encode(text1)
        encoded2 = self.encoder.encode(text2)
        
        # Get positions
        def get_position(encoded):
            if encoded['type'] == 'sentence':
                return encoded['trajectory'].centroid
            elif encoded['type'] in ['word', 'phrase']:
                return encoded['coordinate']
            elif encoded['type'] == 'document':
                return encoded['manifold']['centroid']
            else:
                return SemanticCoordinate(0, 0, 0)
        
        pos1 = get_position(encoded1)
        pos2 = get_position(encoded2)
        
        # Calculate metrics
        distance = pos1.distance_to(pos2)
        
        vec1 = pos1.to_vector()
        vec2 = pos2.to_vector()
        
        # Cosine similarity
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        
        # Per-dimension comparison
        dim_names = ['S_k', 'S_t', 'S_e', 'S_syn', 'S_sem', 'S_prag']
        dim_diffs = {name: float(abs(vec1[i] - vec2[i])) for i, name in enumerate(dim_names)}
        
        return {
            'text1': text1[:50] + '...' if len(text1) > 50 else text1,
            'text2': text2[:50] + '...' if len(text2) > 50 else text2,
            'semantic_distance': distance,
            'similarity': (cos_sim + 1) / 2,  # Normalize to 0-1
            'dimension_differences': dim_diffs,
            'position1': {n: float(vec1[i]) for i, n in enumerate(dim_names)},
            'position2': {n: float(vec2[i]) for i, n in enumerate(dim_names)},
        }
    
    def reset_context(self):
        """Clear conversation context."""
        self._context_buffer.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'total_responses': self._total_responses,
            'total_time': self._total_time,
            'avg_time_per_response': self._total_time / max(1, self._total_responses),
            'context_buffer_size': len(self._context_buffer),
            'synthesis_stats': self.synthesis.get_statistics(),
            'memory_stats': self.memory.get_statistics(),
        }
    
    def explain_difference(self, query: str) -> str:
        """
        Explain how this differs from traditional LLMs.
        """
        response = self.process(query)
        
        explanation = f"""
CATEGORICAL LANGUAGE MODEL vs TRADITIONAL LLM
==============================================

Your query: "{query[:50]}..."

TRADITIONAL LLM WOULD:
1. Tokenize: Split into subword tokens
2. Embed: Map tokens to 1024+ dimensional vectors
3. Attention: O(n²) self-attention across all tokens
4. Transform: Pass through 12-96 transformer layers
5. Predict: Softmax over 50,000+ vocabulary tokens
6. Sample: Pick next token, repeat

CATEGORICAL MODEL DOES:
1. Encode: Map to 6D S-entropy coordinates
   - S_k = {response.semantic_vector[0]:.4f} (semantic velocity)
   - S_t = {response.semantic_vector[1]:.4f} (interpretation breadth)
   - S_e = {response.semantic_vector[2]:.4f} (ambiguity level)
   - S_syn = {response.semantic_vector[3]:.4f} (syntactic structure)
   - S_sem = {response.semantic_vector[4]:.4f} (semantic density)
   - S_prag = {response.semantic_vector[5]:.4f} (pragmatic weight)

2. Navigate: Follow gravity gradient in {response.navigation_steps} steps
   - No attention mechanism (O(log n) not O(n²))
   - Hardware oscillations provide navigation signal

3. Complete: Find categorical completion point
   - Confidence: {response.confidence:.2%}
   - Coherence: {response.coherence:.2%}

KEY DIFFERENCES:
- Parameters: 0 trained weights vs billions
- Training data: None required vs petabytes
- Computation: Navigation vs matrix multiplication
- Grounding: Real hardware timing vs abstract computation
- Interpretability: Direct S-entropy position vs opaque weights

The meaning is not "retrieved" from stored patterns.
It is NAVIGATED TO through coordinate space.
"""
        return explanation


def demonstrate_model():
    """Demonstrate the categorical language model."""
    print("="*70)
    print("CATEGORICAL LANGUAGE MODEL DEMONSTRATION")
    print("="*70)
    print()
    print("This is NOT a traditional LLM. Key differences:")
    print("  - No training required (empty dictionary)")
    print("  - No stored weights (navigation, not retrieval)")
    print("  - O(log n) complexity (not O(n²) attention)")
    print("  - 6D semantic space (not 1024+ embeddings)")
    print()
    
    model = CategoricalLanguageModel()
    
    # Test queries
    queries = [
        "What is the meaning of life?",
        "How do computers work?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    
    print("Processing queries...\n")
    
    for query in queries:
        print(f"Query: {query}")
        response = model.process(query)
        print(f"  Category: {response.response_category}")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Coherence: {response.coherence:.2%}")
        print(f"  Steps: {response.navigation_steps}")
        print(f"  Time: {response.generation_time:.4f}s")
        print()
    
    # Show comparison
    print("\nSemantic Comparison:")
    comparison = model.compare(
        "The cat sat on the mat.",
        "A feline rested upon the rug."
    )
    print(f"  Similarity: {comparison['similarity']:.2%}")
    print(f"  Distance: {comparison['semantic_distance']:.4f}")
    
    # Show explanation
    print("\n" + model.explain_difference("What is consciousness?"))
    
    return model


if __name__ == "__main__":
    demonstrate_model()

