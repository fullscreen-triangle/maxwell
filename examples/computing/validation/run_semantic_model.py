#!/usr/bin/env python
"""
Run the Categorical Language Model

This demonstrates a language model based on S-entropy navigation
rather than neural network training.

Key insight: Language understanding doesn't require:
- Training on billions of tokens
- Storing billions of parameters
- O(n²) attention mechanisms

Instead:
- Navigate S-entropy space
- Use hardware oscillations for grounding
- Find meaning through categorical completion
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maxwell_validation.semantic_processor.semantic_language_model import (
    CategoricalLanguageModel,
    demonstrate_model,
)


def main():
    print("="*70)
    print("CATEGORICAL LANGUAGE MODEL")
    print("Based on S-Entropy Navigation")
    print("="*70)
    print()
    print("This model does NOT use:")
    print("  x  Transformer architecture")
    print("  x  Attention mechanisms")
    print("  x  Pre-training on massive data")
    print("  x  Billions of parameters")
    print("  x  Next-token prediction")
    print()
    print("This model DOES use:")
    print("  ✓  6-dimensional S-entropy space")
    print("  ✓  Semantic gravity navigation")
    print("  ✓  Hardware oscillation grounding")
    print("  ✓  Categorical completion")
    print("  ✓  Empty dictionary synthesis")
    print()
    
    # Run demonstration
    model = demonstrate_model()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter queries to see semantic analysis.")
    print("Type 'quit' to exit, 'compare' to compare two texts.")
    print("Type 'explain' to see how this differs from LLMs.")
    print()
    
    while True:
        try:
            user_input = input("\nQuery> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'explain':
                print(model.explain_difference("How does this work?"))
                continue
            
            if user_input.lower() == 'compare':
                text1 = input("First text: ").strip()
                text2 = input("Second text: ").strip()
                
                comparison = model.compare(text1, text2)
                print(f"\nSemantic Comparison:")
                print(f"  Similarity: {comparison['similarity']:.2%}")
                print(f"  Distance: {comparison['semantic_distance']:.4f}")
                print(f"  Dimension differences:")
                for dim, diff in comparison['dimension_differences'].items():
                    print(f"    {dim}: {diff:.4f}")
                continue
            
            # Process query
            response = model.process(user_input)
            print(response.to_readable())
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Final statistics
    print("\n" + "="*70)
    print("SESSION STATISTICS")
    print("="*70)
    stats = model.get_statistics()
    print(f"Total responses: {stats['total_responses']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Average time per response: {stats['avg_time_per_response']:.4f}s")
    
    return model


if __name__ == "__main__":
    main()

