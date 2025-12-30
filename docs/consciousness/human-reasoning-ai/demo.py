"""
Human-Reasoning AI Demo

Demonstrates the complete pipeline with simulated conversation.
Shows how the system:
- Interprets functionally (not exactly)
- Navigates using S-entropy (not searching)
- Completes gaps (works with incomplete information)
- Tracks user state (makes it personal)
- Anticipates needs (offers before asked)
"""

import sys
sys.path.insert(0, '.')

from core.pipeline import HumanReasoningPipeline, PipelineInput


def run_demo():
    """Run the human-reasoning AI demo"""
    
    print("=" * 70)
    print("HUMAN-REASONING AI: Consciousness-Patterned Intelligence Demo")
    print("=" * 70)
    print()
    print("This demo shows AI that reasons like humans:")
    print("  • Navigates to solutions (O(1)) instead of computing (O(N!))")
    print("  • Works with incomplete information via categorical completion")
    print("  • Interprets meaning functionally, not exactly")
    print("  • Anticipates needs before they're expressed")
    print("  • Makes conversation personal through user modeling")
    print()
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = HumanReasoningPipeline()
    
    # Simulated conversation showing different aspects
    conversation = [
        {
            "message": "Can you help me with this thing?",
            "explanation": "Notice: Vague input with implicit reference ('this thing')"
        },
        {
            "message": "I've been trying for an hour and nothing works",
            "explanation": "Notice: Frustration detected, time interpreted functionally"
        },
        {
            "message": "Maybe I should try something else...",
            "explanation": "Notice: Hedging indicates uncertainty, needs anticipation"
        },
        {
            "message": "What if we approached it differently?",
            "explanation": "Notice: Exploratory mode, curious state detected"
        },
        {
            "message": "Got it! That's almost what I needed",
            "explanation": "Notice: 'almost' interpreted functionally, not as 'failed'"
        },
    ]
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n{'─' * 70}")
        print(f"TURN {i}")
        print(f"{'─' * 70}")
        print(f"\n📝 {turn['explanation']}")
        print(f"\n👤 USER: {turn['message']}")
        
        # Process through pipeline
        input_data = PipelineInput(
            message=turn['message'],
            user_id="demo_user",
            conversation_id="demo_conversation"
        )
        
        response = pipeline.process(input_data)
        
        print(f"\n🤖 AI RESPONSE:")
        print(f"   {response.content}")
        
        print(f"\n📊 PIPELINE ANALYSIS:")
        print(f"   ├─ Interpretation: {response.interpreted_meaning}")
        print(f"   ├─ Gaps completed: {response.completed_gaps or 'None'}")
        print(f"   ├─ User state: {response.user_model_summary['emotional_state']}")
        print(f"   │   └─ Frustration: {response.user_model_summary['frustration_level']:.1%}")
        print(f"   ├─ S-Entropy coords: (S_k={response.s_entropy_coords.S_k:.3f}, S_t={response.s_entropy_coords.S_t:.3f}, S_e={response.s_entropy_coords.S_e:.3f})")
        print(f"   ├─ Flow guidance: {response.flow_guidance_applied}")
        print(f"   └─ Response confidence: {response.confidence:.1%}")
        
        if response.anticipated_needs_addressed:
            print(f"\n   🔮 PROACTIVELY ADDRESSING:")
            for need in response.anticipated_needs_addressed:
                print(f"      • {need}")
    
    print(f"\n{'=' * 70}")
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key Observations:")
    print("  1. Vague inputs were COMPLETED, not rejected")
    print("  2. Time expressions ('an hour') were FUNCTIONAL, not literal")
    print("  3. Frustration was TRACKED and ADDRESSED proactively")
    print("  4. 'Almost' was interpreted as SUCCESS, not failure")
    print("  5. User model EVOLVED through the conversation")
    print()
    print("This is human-like reasoning: approximate, contextual, personal.")


def run_functional_interpretation_demo():
    """Demo specifically for functional interpretation"""
    
    print("\n" + "=" * 70)
    print("FUNCTIONAL INTERPRETATION DEMO")
    print("=" * 70)
    
    from core.functional_interpreter import FunctionalInterpreter
    
    interpreter = FunctionalInterpreter()
    
    print("\n📐 TIME INTERPRETATION (exact → functional)")
    print("-" * 50)
    
    time_phrases = [
        ("Give me a minute", None),
        ("Give me a minute", "urgent"),
        ("Give me a minute", "casual"),
        ("I'll be there soon", None),
        ("Call me later", None),
    ]
    
    for phrase, context in time_phrases:
        result = interpreter.interpret(phrase, context)
        cat = result.category
        ctx_str = f" [context: {context}]" if context else ""
        print(f"  '{phrase}'{ctx_str}")
        print(f"    → Category: {cat.name}")
        print(f"    → Prototype: {cat.prototype}s")
        print(f"    → Acceptable: {cat.acceptable_range[0]:.0f}s - {cat.acceptable_range[1]:.0f}s")
        print()
    
    print("\n✅ COMPLETION EVALUATION (exact → good enough)")
    print("-" * 50)
    
    test_cases = [
        (0.85, "done", "Task is 85% complete"),
        (0.92, "done", "Task is 92% complete"),
        (0.78, "done", "Task is 78% complete"),
        (0.50, "done", "Task is 50% complete"),
    ]
    
    for progress, standard, description in test_cases:
        complete, status = interpreter.evaluate_completion(progress, standard)
        result = "✓ COMPLETE" if complete else "○ IN PROGRESS"
        print(f"  {description}")
        print(f"    → {result}: {status}")
        print()
    
    print("\n🎯 'GOOD ENOUGH' EVALUATION")
    print("-" * 50)
    
    good_enough_tests = [
        (45, "wait a minute", "45 seconds for 'a minute'"),
        (120, "wait a minute", "120 seconds for 'a minute'"),
        (300, "wait a minute", "300 seconds for 'a minute'"),
    ]
    
    for value, goal, description in good_enough_tests:
        ok, reason = interpreter.is_good_enough(value, goal)
        result = "✓ ACCEPTABLE" if ok else "✗ NOT ACCEPTABLE"
        print(f"  {description}")
        print(f"    → {result}: {reason}")
        print()


def run_categorical_completion_demo():
    """Demo specifically for categorical completion"""
    
    print("\n" + "=" * 70)
    print("CATEGORICAL COMPLETION DEMO")
    print("=" * 70)
    
    from core.categorical_completer import CategoricalCompleter
    
    completer = CategoricalCompleter()
    
    print("\n🔍 DETECTING AND FILLING GAPS")
    print("-" * 50)
    
    test_inputs = [
        "Can you help with it?",
        "I need to fix this...",
        "Send the message",
    ]
    
    for text in test_inputs:
        result = completer.complete(text, context={'topic': 'coding'})
        
        print(f"\n  Input: '{text}'")
        print(f"  Gaps found: {len(result.gaps_found)}")
        for gap in result.gaps_found:
            print(f"    • {gap.category} (importance: {gap.importance:.1f})")
        print(f"  Completed: '{result.completed}'")
    
    print("\n📋 STRUCTURED DATA COMPLETION")
    print("-" * 50)
    
    incomplete_data = {
        'action': 'send',
        'target': None,
        'message': 'hello',
        'priority': None
    }
    
    print(f"\n  Input: {incomplete_data}")
    result = completer.complete(incomplete_data)
    print(f"  Completed: {result.completed}")
    print(f"  Strategies used: {list(result.strategies_used.values())}")


if __name__ == "__main__":
    run_demo()
    run_functional_interpretation_demo()
    run_categorical_completion_demo()

