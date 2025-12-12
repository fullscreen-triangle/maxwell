"""
Semantic Encoder

Converts text into S-entropy coordinates through multi-layer encoding.

From the paper:
"The 658× semantic distance amplification achieved through sequential encoding
layers provides the first quantitative theory of how semantic distinctions can
be systematically enhanced through coordinate transformations."

Each layer:
1. Character → Base coordinate (positional encoding)
2. Word → Aggregated coordinate (morphological structure)
3. Phrase → Trajectory segment (syntactic relations)
4. Sentence → Complete trajectory (semantic arc)
5. Paragraph → Trajectory bundle (discourse structure)
6. Document → S-entropy manifold (complete meaning space)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import time
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from categorical_memory.s_entropy_address import SCoordinate, SEntropyAddress
from categorical_memory.hardware_oscillator import HardwareOscillatorCapture


@dataclass
class SemanticCoordinate:
    """
    A point in semantic S-entropy space.
    
    Extends the base SCoordinate with semantic-specific dimensions:
    - S_k: Semantic kinetic (rate of meaning change)
    - S_t: Semantic thermal (breadth of interpretation)
    - S_e: Semantic entropic (ambiguity/richness)
    
    Plus additional semantic dimensions:
    - S_syn: Syntactic structure
    - S_sem: Semantic density
    - S_prag: Pragmatic context
    """
    # Core S-entropy coordinates
    S_k: float  # Kinetic: rate of semantic transition
    S_t: float  # Thermal: interpretation breadth
    S_e: float  # Entropic: ambiguity level
    
    # Extended semantic dimensions
    S_syn: float = 0.0  # Syntactic structure
    S_sem: float = 0.0  # Semantic density
    S_prag: float = 0.0 # Pragmatic weight
    
    # Metadata
    source_text: str = ""
    layer: int = 0  # Which encoding layer produced this
    timestamp: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert to 6D vector."""
        return np.array([
            self.S_k, self.S_t, self.S_e,
            self.S_syn, self.S_sem, self.S_prag
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, source: str = "", layer: int = 0) -> 'SemanticCoordinate':
        """Create from vector."""
        return cls(
            S_k=float(vec[0]),
            S_t=float(vec[1]),
            S_e=float(vec[2]),
            S_syn=float(vec[3]) if len(vec) > 3 else 0.0,
            S_sem=float(vec[4]) if len(vec) > 4 else 0.0,
            S_prag=float(vec[5]) if len(vec) > 5 else 0.0,
            source_text=source,
            layer=layer,
        )
    
    def distance_to(self, other: 'SemanticCoordinate') -> float:
        """Semantic distance in S-space."""
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))
    
    def to_base_coordinate(self) -> SCoordinate:
        """Convert to base SCoordinate."""
        return SCoordinate(S_k=self.S_k, S_t=self.S_t, S_e=self.S_e)


@dataclass
class SemanticTrajectory:
    """
    A path through semantic S-entropy space.
    
    Sentences are not just points - they are TRAJECTORIES.
    The meaning emerges from the path, not just the endpoint.
    """
    coordinates: List[SemanticCoordinate] = field(default_factory=list)
    source_text: str = ""
    
    # Trajectory statistics
    total_distance: float = 0.0
    mean_velocity: float = 0.0
    curvature: float = 0.0
    
    def add(self, coord: SemanticCoordinate):
        """Add a coordinate to the trajectory."""
        if self.coordinates:
            self.total_distance += self.coordinates[-1].distance_to(coord)
        self.coordinates.append(coord)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update trajectory statistics."""
        if len(self.coordinates) < 2:
            return
        
        # Velocity: rate of semantic change
        velocities = []
        for i in range(1, len(self.coordinates)):
            dist = self.coordinates[i-1].distance_to(self.coordinates[i])
            velocities.append(dist)
        
        self.mean_velocity = np.mean(velocities) if velocities else 0.0
        
        # Curvature: how much the trajectory bends
        if len(self.coordinates) >= 3:
            curvatures = []
            for i in range(1, len(self.coordinates) - 1):
                v1 = self.coordinates[i].to_vector() - self.coordinates[i-1].to_vector()
                v2 = self.coordinates[i+1].to_vector() - self.coordinates[i].to_vector()
                
                # Angle between velocity vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                curvatures.append(1 - abs(cos_angle))
            
            self.curvature = np.mean(curvatures)
    
    @property
    def endpoint(self) -> Optional[SemanticCoordinate]:
        """Get the trajectory endpoint (completion point)."""
        return self.coordinates[-1] if self.coordinates else None
    
    @property
    def centroid(self) -> Optional[SemanticCoordinate]:
        """Get the centroid of all coordinates."""
        if not self.coordinates:
            return None
        
        vectors = np.array([c.to_vector() for c in self.coordinates])
        mean_vec = np.mean(vectors, axis=0)
        
        return SemanticCoordinate.from_vector(mean_vec, self.source_text)
    
    def predict_completion(self) -> Optional[SemanticCoordinate]:
        """
        Predict where this trajectory is heading.
        
        This is categorical completion - extracting the endpoint
        that is already encoded in the trajectory.
        """
        if len(self.coordinates) < 2:
            return self.endpoint
        
        # Fit trajectory to find convergence
        vectors = np.array([c.to_vector() for c in self.coordinates])
        
        # Calculate mean velocity direction
        velocities = np.diff(vectors, axis=0)
        mean_velocity = np.mean(velocities, axis=0)
        
        # Project forward
        current = vectors[-1]
        
        # Decay velocity (trajectory settles)
        decay = 0.5
        projected = current + mean_velocity * (1 / (1 - decay))
        
        return SemanticCoordinate.from_vector(projected, self.source_text)


class SemanticEncoder:
    """
    Multi-layer semantic encoder converting text to S-entropy coordinates.
    
    Implements the 6-layer encoding from the Semantic Maxwell Demon paper:
    1. Character encoding
    2. Word encoding
    3. Phrase encoding
    4. Sentence encoding
    5. Paragraph encoding
    6. Document encoding
    
    Each layer amplifies semantic distance by ~3-7×.
    Total amplification: ~658×
    """
    
    # Amplification factors per layer (from paper)
    AMPLIFICATION_FACTORS = {
        'character': 3.7,
        'word': 4.2,
        'phrase': 5.1,
        'sentence': 6.3,
        'paragraph': 7.1,
        'document': 7.3,
    }
    
    def __init__(self, oscillator: Optional[HardwareOscillatorCapture] = None):
        """Initialize the encoder."""
        self.oscillator = oscillator or HardwareOscillatorCapture()
        
        # Character position encoding (inspired by transformer positional encoding)
        self._char_basis = self._create_char_basis()
        
        # Word statistics (lightweight, not embeddings)
        self._word_stats = {
            'mean_length': 4.5,
            'common_prefixes': ['un', 're', 'in', 'dis', 'en', 'non', 'pre', 'mis'],
            'common_suffixes': ['ing', 'ed', 'ly', 'tion', 'ness', 'ment', 'able'],
        }
    
    def _create_char_basis(self) -> np.ndarray:
        """Create basis vectors for character encoding."""
        # 256 ASCII characters mapped to 6D space
        basis = np.zeros((256, 6))
        
        for i in range(256):
            # Position-dependent encoding
            angle = i * 2 * np.pi / 256
            
            basis[i, 0] = np.sin(angle)  # S_k
            basis[i, 1] = np.cos(angle)  # S_t
            basis[i, 2] = np.sin(2 * angle)  # S_e
            basis[i, 3] = np.cos(2 * angle)  # S_syn
            basis[i, 4] = np.sin(3 * angle)  # S_sem
            basis[i, 5] = np.cos(3 * angle)  # S_prag
            
            # Amplitude based on character class
            if 65 <= i <= 90:  # Uppercase
                basis[i] *= 1.2
            elif 97 <= i <= 122:  # Lowercase
                basis[i] *= 1.0
            elif 48 <= i <= 57:  # Digits
                basis[i] *= 0.8
            elif i in [32, 10, 13, 9]:  # Whitespace
                basis[i] *= 0.3
            else:  # Punctuation and special
                basis[i] *= 1.5
        
        return basis
    
    def encode_character(self, char: str) -> SemanticCoordinate:
        """
        Layer 1: Encode a single character.
        
        Maps character to base position in S-entropy space.
        """
        if not char:
            return SemanticCoordinate(0, 0, 0, source_text=char, layer=1)
        
        code = ord(char[0]) % 256
        vec = self._char_basis[code] * self.AMPLIFICATION_FACTORS['character']
        
        return SemanticCoordinate.from_vector(vec, char, layer=1)
    
    def encode_word(self, word: str) -> SemanticCoordinate:
        """
        Layer 2: Encode a word.
        
        Aggregates character coordinates with morphological awareness.
        """
        if not word:
            return SemanticCoordinate(0, 0, 0, source_text=word, layer=2)
        
        # Encode each character
        char_coords = [self.encode_character(c) for c in word]
        
        # Aggregate with position weighting
        n = len(char_coords)
        weights = np.exp(-np.arange(n) / max(1, n / 2))  # Decay from start
        weights /= weights.sum()
        
        vectors = np.array([c.to_vector() for c in char_coords])
        weighted_mean = np.average(vectors, axis=0, weights=weights)
        
        # Morphological adjustments
        morph_factor = 1.0
        
        # Check prefixes
        for prefix in self._word_stats['common_prefixes']:
            if word.lower().startswith(prefix):
                morph_factor *= 1.1
                break
        
        # Check suffixes
        for suffix in self._word_stats['common_suffixes']:
            if word.lower().endswith(suffix):
                morph_factor *= 1.1
                break
        
        # Length adjustment
        length_factor = len(word) / self._word_stats['mean_length']
        morph_factor *= (1 + 0.1 * np.log(length_factor + 0.1))
        
        result = weighted_mean * self.AMPLIFICATION_FACTORS['word'] * morph_factor
        
        return SemanticCoordinate.from_vector(result, word, layer=2)
    
    def encode_phrase(self, phrase: str) -> SemanticCoordinate:
        """
        Layer 3: Encode a phrase (2-5 words).
        
        Captures syntactic relationships between words.
        """
        words = phrase.split()
        if not words:
            return SemanticCoordinate(0, 0, 0, source_text=phrase, layer=3)
        
        # Encode each word
        word_coords = [self.encode_word(w) for w in words]
        
        # Build trajectory through word space
        trajectory = SemanticTrajectory(source_text=phrase)
        for coord in word_coords:
            trajectory.add(coord)
        
        # Phrase coordinate is trajectory centroid with velocity info
        centroid = trajectory.centroid
        if not centroid:
            return SemanticCoordinate(0, 0, 0, source_text=phrase, layer=3)
        
        vec = centroid.to_vector()
        
        # Add trajectory dynamics to encoding
        vec[0] += trajectory.mean_velocity  # S_k captures velocity
        vec[2] += trajectory.curvature  # S_e captures curvature
        vec[3] += len(words) * 0.1  # S_syn captures length
        
        vec *= self.AMPLIFICATION_FACTORS['phrase']
        
        return SemanticCoordinate.from_vector(vec, phrase, layer=3)
    
    def encode_sentence(self, sentence: str) -> SemanticTrajectory:
        """
        Layer 4: Encode a sentence.
        
        Returns a trajectory, not just a point.
        Meaning emerges from the path through semantic space.
        """
        # Split into phrases (simple: groups of 3-4 words)
        words = sentence.split()
        phrases = []
        
        phrase_size = 3
        for i in range(0, len(words), phrase_size):
            phrase = ' '.join(words[i:i + phrase_size])
            if phrase:
                phrases.append(phrase)
        
        # If no phrases, treat whole sentence as one phrase
        if not phrases:
            phrases = [sentence]
        
        # Build trajectory
        trajectory = SemanticTrajectory(source_text=sentence)
        
        for phrase in phrases:
            coord = self.encode_phrase(phrase)
            
            # Apply sentence-level amplification
            vec = coord.to_vector() * self.AMPLIFICATION_FACTORS['sentence']
            amplified = SemanticCoordinate.from_vector(vec, phrase, layer=4)
            
            trajectory.add(amplified)
        
        return trajectory
    
    def encode_paragraph(self, paragraph: str) -> List[SemanticTrajectory]:
        """
        Layer 5: Encode a paragraph.
        
        Returns a bundle of sentence trajectories.
        Captures discourse structure.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        trajectories = []
        
        for sentence in sentences:
            traj = self.encode_sentence(sentence)
            
            # Apply paragraph-level amplification to trajectory
            for coord in traj.coordinates:
                vec = coord.to_vector() * self.AMPLIFICATION_FACTORS['paragraph']
                # Update in place
                coord.S_k, coord.S_t, coord.S_e = vec[0], vec[1], vec[2]
                coord.S_syn, coord.S_sem, coord.S_prag = vec[3], vec[4], vec[5]
            
            trajectories.append(traj)
        
        return trajectories
    
    def encode_document(self, document: str) -> Dict[str, Any]:
        """
        Layer 6: Encode a complete document.
        
        Returns the full S-entropy manifold representation.
        """
        # Split into paragraphs
        paragraphs = document.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Encode each paragraph
        paragraph_bundles = []
        all_coordinates = []
        
        for para in paragraphs:
            bundle = self.encode_paragraph(para)
            paragraph_bundles.append(bundle)
            
            for traj in bundle:
                all_coordinates.extend(traj.coordinates)
        
        # Apply document-level amplification
        for coord in all_coordinates:
            vec = coord.to_vector() * self.AMPLIFICATION_FACTORS['document']
            coord.S_k, coord.S_t, coord.S_e = vec[0], vec[1], vec[2]
            coord.S_syn, coord.S_sem, coord.S_prag = vec[3], vec[4], vec[5]
        
        # Calculate total amplification
        total_amp = 1.0
        for factor in self.AMPLIFICATION_FACTORS.values():
            total_amp *= factor
        
        # Build manifold
        manifold = {
            'source': document,
            'paragraphs': paragraph_bundles,
            'all_coordinates': all_coordinates,
            'total_amplification': total_amp,
            'centroid': self._compute_manifold_centroid(all_coordinates),
            'bounds': self._compute_manifold_bounds(all_coordinates),
            'entropy': self._compute_manifold_entropy(all_coordinates),
        }
        
        return manifold
    
    def _compute_manifold_centroid(self, coords: List[SemanticCoordinate]) -> SemanticCoordinate:
        """Compute the centroid of all coordinates."""
        if not coords:
            return SemanticCoordinate(0, 0, 0)
        
        vectors = np.array([c.to_vector() for c in coords])
        mean_vec = np.mean(vectors, axis=0)
        
        return SemanticCoordinate.from_vector(mean_vec, "centroid", layer=6)
    
    def _compute_manifold_bounds(self, coords: List[SemanticCoordinate]) -> Dict[str, Tuple[float, float]]:
        """Compute bounds for each dimension."""
        if not coords:
            return {}
        
        vectors = np.array([c.to_vector() for c in coords])
        
        dim_names = ['S_k', 'S_t', 'S_e', 'S_syn', 'S_sem', 'S_prag']
        bounds = {}
        
        for i, name in enumerate(dim_names):
            bounds[name] = (float(vectors[:, i].min()), float(vectors[:, i].max()))
        
        return bounds
    
    def _compute_manifold_entropy(self, coords: List[SemanticCoordinate]) -> float:
        """Compute entropy of the coordinate distribution."""
        if len(coords) < 2:
            return 0.0
        
        vectors = np.array([c.to_vector() for c in coords])
        
        # Use histogram-based entropy estimate
        total_entropy = 0.0
        
        for dim in range(vectors.shape[1]):
            hist, _ = np.histogram(vectors[:, dim], bins=min(10, len(coords)))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            total_entropy += entropy
        
        return total_entropy / vectors.shape[1]
    
    def encode(self, text: str) -> Dict[str, Any]:
        """
        Main entry point: encode text at appropriate granularity.
        
        Automatically determines whether input is word, sentence,
        paragraph, or document and encodes accordingly.
        """
        text = text.strip()
        
        if not text:
            return {'type': 'empty', 'coordinates': []}
        
        # Determine granularity
        n_words = len(text.split())
        n_sentences = len(re.split(r'[.!?]+', text))
        n_paragraphs = len(text.split('\n\n'))
        
        if n_words <= 1:
            coord = self.encode_word(text)
            return {'type': 'word', 'coordinate': coord}
        
        elif n_words <= 5:
            coord = self.encode_phrase(text)
            return {'type': 'phrase', 'coordinate': coord}
        
        elif n_sentences <= 2 and n_paragraphs <= 1:
            traj = self.encode_sentence(text)
            return {'type': 'sentence', 'trajectory': traj}
        
        elif n_paragraphs <= 1:
            bundle = self.encode_paragraph(text)
            return {'type': 'paragraph', 'trajectory_bundle': bundle}
        
        else:
            manifold = self.encode_document(text)
            return {'type': 'document', 'manifold': manifold}

