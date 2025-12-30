"""
Categorical Completer: Fill Gaps in Information

This module implements categorical completion:
- Detect what's missing from input
- Fill gaps using context, history, and structure
- Enable operation on incomplete information (like humans do)

Key insight: Humans work with 30% explicit, 70% completed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re


class CompletionStrategy(Enum):
    """Strategies for filling gaps"""
    CONTEXTUAL = "contextual"      # Infer from surrounding context
    HISTORICAL = "historical"       # Use patterns from history
    STRUCTURAL = "structural"       # Infer from problem structure
    TRAJECTORY = "trajectory"       # Beginning + End -> middle
    PROTOTYPE = "prototype"         # Use category prototype


@dataclass
class Gap:
    """A detected gap in information"""
    category: str           # What type of information is missing
    location: str           # Where in the structure
    importance: float       # How critical is this gap (0-1)
    possible_fills: List[Any] = field(default_factory=list)


@dataclass 
class CompletionResult:
    """Result of categorical completion"""
    original: Any
    completed: Any
    gaps_found: List[Gap]
    strategies_used: Dict[str, CompletionStrategy]
    confidence_map: Dict[str, float]


class CategoricalCompleter:
    """
    Fills gaps in information through categorical inference.
    
    Operates like human cognition:
    - Sees partial pattern -> completes the whole
    - Sees problem structure -> infers solution structure
    - Works with incomplete data -> still produces output
    """
    
    def __init__(self):
        """Initialize the completer with default prototypes"""
        # Category prototypes (default values)
        self.prototypes = {
            'time': 'now',
            'place': 'here',
            'person': 'someone',
            'thing': 'something',
            'action': 'does',
            'quantity': 'some',
            'reason': 'because',
            'manner': 'somehow',
        }
        
        # Structural patterns for inference
        self.structural_patterns = {
            'question': ['subject', 'verb', 'object'],
            'request': ['action', 'target', 'parameters'],
            'statement': ['subject', 'predicate'],
            'command': ['verb', 'object'],
        }
        
        # Historical completions (learned patterns)
        self.history: List[Dict] = []
    
    def detect_gaps(self, 
                   input_data: Any,
                   expected_structure: Optional[List[str]] = None) -> List[Gap]:
        """
        Detect gaps (missing information) in input.
        
        Args:
            input_data: The input to analyze
            expected_structure: Optional expected categories
            
        Returns:
            List of detected gaps
        """
        gaps = []
        
        if isinstance(input_data, str):
            gaps.extend(self._detect_text_gaps(input_data))
        elif isinstance(input_data, dict):
            gaps.extend(self._detect_dict_gaps(input_data, expected_structure))
        
        return gaps
    
    def _detect_text_gaps(self, text: str) -> List[Gap]:
        """Detect gaps in text input"""
        gaps = []
        
        # Short text = missing context
        if len(text) < 20:
            gaps.append(Gap(
                category='context',
                location='global',
                importance=0.7,
                possible_fills=['(needs more context)']
            ))
        
        # Check for implicit references
        implicit_patterns = [
            (r'\b(it|this|that)\b', 'reference', 0.6),
            (r'\b(here|there)\b', 'place', 0.5),
            (r'\b(now|then|soon)\b', 'time', 0.4),
            (r'\b(they|them|someone)\b', 'person', 0.5),
        ]
        
        for pattern, category, importance in implicit_patterns:
            if re.search(pattern, text.lower()):
                gaps.append(Gap(
                    category=category,
                    location='implicit_reference',
                    importance=importance
                ))
        
        # Check for ellipsis or trailing thoughts
        if text.rstrip().endswith('...') or text.rstrip().endswith('—'):
            gaps.append(Gap(
                category='continuation',
                location='end',
                importance=0.8
            ))
        
        return gaps
    
    def _detect_dict_gaps(self, 
                         data: Dict,
                         expected: Optional[List[str]] = None) -> List[Gap]:
        """Detect gaps in dictionary/structured input"""
        gaps = []
        
        # Check for None values
        for key, value in data.items():
            if value is None:
                gaps.append(Gap(
                    category=key,
                    location=f'data.{key}',
                    importance=0.9
                ))
        
        # Check for expected but missing keys
        if expected:
            for key in expected:
                if key not in data:
                    gaps.append(Gap(
                        category=key,
                        location=f'data.{key}',
                        importance=0.8
                    ))
        
        return gaps
    
    def fill_gap(self,
                gap: Gap,
                context: Dict,
                strategy: Optional[CompletionStrategy] = None) -> Tuple[Any, CompletionStrategy, float]:
        """
        Fill a single gap using appropriate strategy.
        
        Args:
            gap: The gap to fill
            context: Available context for inference
            strategy: Optional forced strategy
            
        Returns:
            Tuple of (filled value, strategy used, confidence)
        """
        if strategy is None:
            strategy = self._select_strategy(gap, context)
        
        if strategy == CompletionStrategy.CONTEXTUAL:
            return self._fill_from_context(gap, context)
        elif strategy == CompletionStrategy.HISTORICAL:
            return self._fill_from_history(gap, context)
        elif strategy == CompletionStrategy.STRUCTURAL:
            return self._fill_from_structure(gap, context)
        elif strategy == CompletionStrategy.TRAJECTORY:
            return self._fill_from_trajectory(gap, context)
        else:  # PROTOTYPE
            return self._fill_from_prototype(gap)
    
    def _select_strategy(self, gap: Gap, context: Dict) -> CompletionStrategy:
        """Select best strategy for filling a gap"""
        # If we have strong context, use it
        if gap.category in context and context.get(gap.category):
            return CompletionStrategy.CONTEXTUAL
        
        # If we have relevant history, use it
        if self._has_relevant_history(gap):
            return CompletionStrategy.HISTORICAL
        
        # If structure suggests completion, use it
        if gap.location.startswith('data.') or gap.category in self.structural_patterns:
            return CompletionStrategy.STRUCTURAL
        
        # Default to prototype
        return CompletionStrategy.PROTOTYPE
    
    def _fill_from_context(self, gap: Gap, context: Dict) -> Tuple[Any, CompletionStrategy, float]:
        """Fill gap from surrounding context"""
        if gap.category in context:
            return context[gap.category], CompletionStrategy.CONTEXTUAL, 0.9
        
        # Try to infer from related context
        related = self._find_related_context(gap.category, context)
        if related:
            return related, CompletionStrategy.CONTEXTUAL, 0.7
        
        return None, CompletionStrategy.CONTEXTUAL, 0.0
    
    def _fill_from_history(self, gap: Gap, context: Dict) -> Tuple[Any, CompletionStrategy, float]:
        """Fill gap from historical patterns"""
        for entry in reversed(self.history):  # Most recent first
            if entry.get('category') == gap.category:
                return entry.get('value'), CompletionStrategy.HISTORICAL, 0.8
        
        return None, CompletionStrategy.HISTORICAL, 0.0
    
    def _fill_from_structure(self, gap: Gap, context: Dict) -> Tuple[Any, CompletionStrategy, float]:
        """Fill gap from structural inference"""
        # If we know the structure type, infer missing parts
        structure_type = context.get('structure_type', 'statement')
        
        if structure_type in self.structural_patterns:
            pattern = self.structural_patterns[structure_type]
            if gap.category in pattern:
                # Infer from position in structure
                idx = pattern.index(gap.category)
                # Use prototype for that position
                return f'[{gap.category}]', CompletionStrategy.STRUCTURAL, 0.6
        
        return None, CompletionStrategy.STRUCTURAL, 0.0
    
    def _fill_from_trajectory(self, gap: Gap, context: Dict) -> Tuple[Any, CompletionStrategy, float]:
        """Fill gap from beginning + end -> middle inference"""
        start = context.get('start')
        end = context.get('end')
        
        if start and end:
            # Infer middle from endpoints
            middle = f'[{start} -> ... -> {end}]'
            return middle, CompletionStrategy.TRAJECTORY, 0.7
        
        return None, CompletionStrategy.TRAJECTORY, 0.0
    
    def _fill_from_prototype(self, gap: Gap) -> Tuple[Any, CompletionStrategy, float]:
        """Fill gap with category prototype"""
        if gap.category in self.prototypes:
            return self.prototypes[gap.category], CompletionStrategy.PROTOTYPE, 0.5
        
        return f'[{gap.category}?]', CompletionStrategy.PROTOTYPE, 0.3
    
    def _has_relevant_history(self, gap: Gap) -> bool:
        """Check if history has relevant entries"""
        return any(e.get('category') == gap.category for e in self.history)
    
    def _find_related_context(self, category: str, context: Dict) -> Optional[Any]:
        """Find related information in context"""
        # Simple relatedness: check for substring matches
        for key, value in context.items():
            if category in key or key in category:
                return value
        return None
    
    def complete(self,
                input_data: Any,
                context: Optional[Dict] = None,
                user_model: Optional[Dict] = None) -> CompletionResult:
        """
        Complete all gaps in input data.
        
        Args:
            input_data: The incomplete input
            context: Available context
            user_model: Optional user model for personalized completion
            
        Returns:
            CompletionResult with completed data
        """
        context = context or {}
        
        # Detect gaps
        gaps = self.detect_gaps(input_data)
        
        # Fill each gap
        strategies_used = {}
        confidence_map = {}
        completions = {}
        
        for gap in gaps:
            value, strategy, confidence = self.fill_gap(gap, context)
            if value is not None:
                completions[gap.location] = value
                strategies_used[gap.location] = strategy
                confidence_map[gap.location] = confidence
                
                # Record in history
                self.history.append({
                    'category': gap.category,
                    'value': value,
                    'confidence': confidence
                })
        
        # Build completed output
        if isinstance(input_data, str):
            completed = self._apply_text_completions(input_data, completions, gaps)
        elif isinstance(input_data, dict):
            completed = self._apply_dict_completions(input_data, completions)
        else:
            completed = input_data
        
        return CompletionResult(
            original=input_data,
            completed=completed,
            gaps_found=gaps,
            strategies_used=strategies_used,
            confidence_map=confidence_map
        )
    
    def _apply_text_completions(self,
                               text: str,
                               completions: Dict,
                               gaps: List[Gap]) -> str:
        """Apply completions to text"""
        result = text
        
        # Add context annotation if needed
        annotations = []
        for gap in gaps:
            if gap.location in completions:
                annotations.append(f"[{gap.category}: {completions[gap.location]}]")
        
        if annotations:
            result = result + " " + " ".join(annotations)
        
        return result
    
    def _apply_dict_completions(self,
                               data: Dict,
                               completions: Dict) -> Dict:
        """Apply completions to dictionary"""
        result = data.copy()
        
        for location, value in completions.items():
            # Parse location like "data.key"
            if location.startswith('data.'):
                key = location[5:]
                result[key] = value
        
        return result


# Test if run directly
if __name__ == "__main__":
    completer = CategoricalCompleter()
    
    # Test with incomplete text
    text = "Can you help with it..."
    result = completer.complete(text, context={'topic': 'coding'})
    
    print("Text Completion:")
    print(f"  Original: {result.original}")
    print(f"  Completed: {result.completed}")
    print(f"  Gaps found: {len(result.gaps_found)}")
    for gap in result.gaps_found:
        print(f"    - {gap.category} at {gap.location} (importance: {gap.importance})")
    
    # Test with incomplete dict
    data = {'action': 'send', 'target': None, 'message': 'hello'}
    result = completer.complete(data, expected_structure=['action', 'target', 'message'])
    
    print("\nDict Completion:")
    print(f"  Original: {result.original}")
    print(f"  Completed: {result.completed}")

