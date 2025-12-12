#!/usr/bin/env python
"""
Run the Molecular Semantics Demonstration

This demonstrates semantic understanding through molecular structure prediction:
- Words encoded as virtual molecules with vibrational frequencies
- Meaning derived from harmonic coincidence networks
- No training required - structure encodes relationships
- Atmospheric memory for zero-cost storage
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maxwell_validation.semantic_processor.molecular_semantics import (
    MolecularSemanticProcessor,
    VirtualMolecule,
    HarmonicCoincidenceNetwork,
    AtmosphericSemanticMemory,
)


def convert_for_json(obj):
    """Convert numpy types and other non-serializable objects for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'S_k'):
        return {
            'S_k': float(obj.S_k),
            'S_t': float(obj.S_t),
            'S_e': float(obj.S_e)
        }
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def save_results(results, output_dir):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'molecular_semantics_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def demonstrate():
    print("="*70)
    print("MOLECULAR SEMANTICS DEMONSTRATION")
    print("="*70)

    # Create the processor
    processor = MolecularSemanticProcessor()
    results = {}

    # Example 1: Word Encoding as Virtual Molecules
    print("\n" + "="*60)
    print("EXAMPLE 1: Word Encoding as Virtual Molecules")
    print("="*60)
    
    text = "The cat sat on the mat"
    words = text.lower().split()
    print(f"\nText: '{text}'")
    print(f"Words: {len(words)}")
    
    print("\nMolecular encoding:")
    molecules = []
    for word in words:
        mol = VirtualMolecule(word=word)
        molecules.append({
            'word': word,
            'fundamental_frequency': mol.fundamental_frequency,
            's_coordinate': {
                'S_k': mol.S_k,
                'S_t': mol.S_t,
                'S_e': mol.S_e
            },
            'mass': mol.mass
        })
        print(f"  {word:8s}: freq={mol.fundamental_frequency:.2e} Hz, "
              f"S=({mol.S_k:.3f}, {mol.S_t:.3f}, {mol.S_e:.3f})")
    
    results['encoding'] = {
        'text': text,
        'words': words,
        'molecules': molecules
    }

    # Example 2: Understanding with Harmonic Networks
    print("\n" + "="*60)
    print("EXAMPLE 2: Understanding Text via Molecular Analysis")
    print("="*60)
    
    understanding = processor.understand(text)
    print(f"\nText understood: '{text}'")
    print(f"  Words processed: {understanding['words_processed']}")
    print(f"  Network nodes: {understanding['molecular_network'].get('total_molecules', 0)}")
    print(f"  Network edges: {understanding['molecular_network'].get('total_coincidences', 0)}")
    print(f"  Processing time: {understanding['processing_time']*1000:.2f} ms")
    
    results['understanding'] = {
        'text': text,
        'words_processed': understanding['words_processed'],
        'network_stats': understanding['molecular_network'],
        'processing_time': understanding['processing_time']
    }

    # Example 3: Semantic Similarity via Molecular Comparison
    print("\n" + "="*60)
    print("EXAMPLE 3: Semantic Similarity via Molecular Comparison")
    print("="*60)
    
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast red wolf leaps above the tired hound"
    text3 = "The cat is sleeping"

    comparison12 = processor.compare(text1, text2)
    print(f"\nComparing:")
    print(f"  Text 1: '{text1}'")
    print(f"  Text 2: '{text2}'")
    print(f"  Similarity: {comparison12.get('similarity', 0):.2%}")
    print(f"  Frequency overlap: {comparison12.get('frequency_overlap', 0):.2f}")
    print(f"  S-distance: {comparison12.get('s_distance', 0):.4f}")

    comparison13 = processor.compare(text1, text3)
    print(f"\nComparing:")
    print(f"  Text 1: '{text1}'")
    print(f"  Text 3: '{text3}'")
    print(f"  Similarity: {comparison13.get('similarity', 0):.2%}")
    print(f"  S-distance: {comparison13.get('s_distance', 0):.4f}")

    results['similarity'] = {
        'comparison_1': {
            'text1': text1, 
            'text2': text2, 
            'similarity': comparison12.get('similarity', 0),
            's_distance': comparison12.get('s_distance', 0)
        },
        'comparison_2': {
            'text1': text1, 
            'text2': text3, 
            'similarity': comparison13.get('similarity', 0),
            's_distance': comparison13.get('s_distance', 0)
        }
    }

    # Example 4: Atmospheric Memory Storage
    print("\n" + "="*60)
    print("EXAMPLE 4: Atmospheric Memory Storage")
    print("="*60)
    
    memory = AtmosphericSemanticMemory(volume_cm3=10.0)
    
    # Store some sentences
    sentences = [
        "The quick brown fox",
        "A lazy dog sleeps",
        "Science explores nature"
    ]
    
    for sentence in sentences:
        addresses = memory.store_sentence(sentence)
        print(f"\n  Stored: '{sentence}'")
        print(f"  Addresses: {len(addresses)} S-coordinate positions")
    
    stats = memory.get_statistics()
    print(f"\nAtmospheric Memory Statistics:")
    print(f"  Volume: {stats['volume_cm3']} cm³")
    print(f"  Words stored: {stats['words_stored']}")
    print(f"  Addresses used: {stats['addresses_used']}")
    print(f"  Total addresses available: {stats['n_addresses']:.2e}")
    
    results['atmospheric_memory'] = {
        'sentences_stored': sentences,
        'statistics': stats
    }

    # Architecture Explanation
    print("\n" + "="*60)
    print("MOLECULAR SEMANTICS ARCHITECTURE")
    print("="*60)
    
    print("""
TRADITIONAL LLM:
  Text -> Tokens -> Embeddings -> Attention -> Output
  - Requires training on billions of tokens
  - Stores billions of parameters
  - O(n²) attention complexity

MOLECULAR SEMANTICS:
  Text -> Virtual Molecules -> Harmonic Coincidence -> Understanding
  - No training required (like molecular spectroscopy)
  - Zero storage cost (atmospheric memory)
  - O(log n) complexity (categorical addressing)

THE MAPPING:
  Word         <-> Virtual Molecule (has frequencies, S-coordinates)
  Sentence     <-> Molecular Ensemble (collective dynamics)
  Meaning      <-> Vibrational Mode (extractable from structure)
  Context      <-> Known Modes (enable prediction)
  Understanding <-> Harmonic Prediction

KEY INSIGHT:
  Just as unknown molecular vibrational modes can be predicted
  from known modes using harmonic coincidence networks,
  unknown semantic meaning can be predicted from known context.

  The 'training' that LLMs need is replaced by the harmonic
  structure of the molecular encoding itself.

ZERO COST:
  - No training (structure encodes relationships)
  - No parameters (frequencies are computed, not stored)
  - No energy for storage (thermally driven atmospheric memory)
""")

    return results


def main():
    results = demonstrate()
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results', 'molecular_semantics')
    save_results(results, output_dir)
    
    return results


if __name__ == "__main__":
    main()
