"""
Functional Interpreter: From Exact Values to Functional Categories

This module replaces exact-value processing with functional categories:
- "1 minute" = category of acceptable durations, not 60000ms
- "Done" = functionally complete, not all-criteria-satisfied
- Evaluation checks function, not accuracy

Key insight: Humans don't enforce precision, they accept "good enough"
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
import re


@dataclass
class FunctionalCategory:
    """A category defined by function, not value"""
    name: str
    prototype: Any  # Central/typical value
    acceptable_range: Tuple[Any, Any]  # (min, max) or semantic bounds
    context_modifiers: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    functional_test: Optional[Callable[[Any], bool]] = None
    
    def contains(self, value: Any) -> bool:
        """Check if value is in this category"""
        if self.functional_test:
            return self.functional_test(value)
        
        try:
            min_val, max_val = self.acceptable_range
            return min_val <= value <= max_val
        except TypeError:
            # Non-comparable types: check equality with prototype
            return value == self.prototype
    
    def apply_context(self, context: str) -> 'FunctionalCategory':
        """Return category with context-adjusted range"""
        if context in self.context_modifiers:
            min_mult, max_mult = self.context_modifiers[context]
            old_min, old_max = self.acceptable_range
            try:
                new_range = (old_min * min_mult, old_max * max_mult)
                return FunctionalCategory(
                    name=self.name,
                    prototype=self.prototype,
                    acceptable_range=new_range,
                    context_modifiers=self.context_modifiers,
                    functional_test=self.functional_test
                )
            except TypeError:
                pass
        return self


@dataclass
class InterpretationResult:
    """Result of functional interpretation"""
    original: str
    category: FunctionalCategory
    literal_value: Any
    confidence: float
    context_applied: Optional[str] = None


class FunctionalInterpreter:
    """
    Converts exact statements to functional categories.
    
    Instead of parsing "1 minute" as 60000ms, interprets it as
    a functional category: "short duration, acceptable range depends on context"
    """
    
    def __init__(self):
        """Initialize with default categories"""
        self.categories = self._build_default_categories()
        self.patterns = self._build_patterns()
    
    def _build_default_categories(self) -> Dict[str, FunctionalCategory]:
        """Build default functional categories"""
        return {
            # Time categories
            'moment': FunctionalCategory(
                name='moment',
                prototype=1.0,  # seconds
                acceptable_range=(0.1, 5.0),
                context_modifiers={
                    'casual': (0.5, 3.0),
                    'urgent': (0.1, 0.5),
                    'relaxed': (1.0, 10.0)
                }
            ),
            'minute': FunctionalCategory(
                name='minute',
                prototype=60.0,
                acceptable_range=(30.0, 180.0),  # 30s to 3min all "a minute"
                context_modifiers={
                    'casual': (0.5, 5.0),
                    'meeting': (0.8, 1.2),
                    'cooking': (0.9, 1.1)
                }
            ),
            'soon': FunctionalCategory(
                name='soon',
                prototype=300.0,  # 5 minutes
                acceptable_range=(60.0, 3600.0),  # 1min to 1hr
                context_modifiers={
                    'urgent': (0.1, 0.3),
                    'casual': (1.0, 4.0)
                }
            ),
            'later': FunctionalCategory(
                name='later',
                prototype=3600.0,  # 1 hour
                acceptable_range=(1800.0, 86400.0),  # 30min to 1 day
            ),
            
            # Quantity categories
            'few': FunctionalCategory(
                name='few',
                prototype=3,
                acceptable_range=(2, 5)
            ),
            'some': FunctionalCategory(
                name='some',
                prototype=5,
                acceptable_range=(3, 10)
            ),
            'many': FunctionalCategory(
                name='many',
                prototype=20,
                acceptable_range=(10, 100)
            ),
            'lot': FunctionalCategory(
                name='lot',
                prototype=50,
                acceptable_range=(20, 500)
            ),
            
            # Completion categories
            'done': FunctionalCategory(
                name='done',
                prototype=1.0,  # 100%
                acceptable_range=(0.85, 1.0),  # 85%+ is "done"
                functional_test=lambda x: x >= 0.85
            ),
            'almost': FunctionalCategory(
                name='almost',
                prototype=0.9,
                acceptable_range=(0.7, 0.95)
            ),
            'started': FunctionalCategory(
                name='started',
                prototype=0.1,
                acceptable_range=(0.01, 0.3)
            ),
            
            # Quality categories
            'good': FunctionalCategory(
                name='good',
                prototype=0.8,
                acceptable_range=(0.6, 0.95),
                functional_test=lambda x: x >= 0.6
            ),
            'okay': FunctionalCategory(
                name='okay',
                prototype=0.5,
                acceptable_range=(0.4, 0.7)
            ),
            'bad': FunctionalCategory(
                name='bad',
                prototype=0.2,
                acceptable_range=(0.0, 0.4)
            ),
        }
    
    def _build_patterns(self) -> List[Tuple[str, str]]:
        """Build regex patterns for category detection"""
        return [
            # Time patterns
            (r'\b(\d+)\s*minutes?\b', 'minute'),
            (r'\b(\d+)\s*seconds?\b', 'moment'),
            (r'\b(a|one)\s*moment\b', 'moment'),
            (r'\b(a|one)\s*minute\b', 'minute'),
            (r'\bsoon\b', 'soon'),
            (r'\blater\b', 'later'),
            (r'\bshortly\b', 'soon'),
            
            # Quantity patterns
            (r'\b(a\s*)?few\b', 'few'),
            (r'\bsome\b', 'some'),
            (r'\bmany\b', 'many'),
            (r'\b(a\s*)?lot\b', 'lot'),
            (r'\bseveral\b', 'some'),
            
            # Completion patterns
            (r'\bdone\b', 'done'),
            (r'\bfinished\b', 'done'),
            (r'\bcompleted?\b', 'done'),
            (r'\balmost\b', 'almost'),
            (r'\bnearly\b', 'almost'),
            (r'\bstarted\b', 'started'),
            (r'\bbegun?\b', 'started'),
            
            # Quality patterns
            (r'\bgood\b', 'good'),
            (r'\bgreat\b', 'good'),
            (r'\bokay\b', 'okay'),
            (r'\bok\b', 'okay'),
            (r'\bfine\b', 'okay'),
            (r'\bbad\b', 'bad'),
            (r'\bpoor\b', 'bad'),
        ]
    
    def identify_category(self, text: str) -> Optional[Tuple[str, Any]]:
        """
        Identify functional category in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (category_name, extracted_value) or None
        """
        text_lower = text.lower()
        
        for pattern, category_name in self.patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Extract literal value if present
                groups = match.groups()
                literal = groups[0] if groups and groups[0] else None
                
                # Convert numeric literals
                if literal and literal.isdigit():
                    literal = int(literal)
                
                return (category_name, literal)
        
        return None
    
    def interpret(self,
                 statement: str,
                 context: Optional[str] = None) -> InterpretationResult:
        """
        Interpret statement as functional category.
        
        Args:
            statement: The statement to interpret
            context: Optional context (casual, urgent, etc.)
            
        Returns:
            InterpretationResult with functional category
        """
        # Identify category
        result = self.identify_category(statement)
        
        if result:
            category_name, literal_value = result
            category = self.categories.get(category_name)
            
            if category:
                # Apply context if provided
                if context:
                    category = category.apply_context(context)
                
                return InterpretationResult(
                    original=statement,
                    category=category,
                    literal_value=literal_value,
                    confidence=0.9,
                    context_applied=context
                )
        
        # No category found: create generic
        generic = FunctionalCategory(
            name='unspecified',
            prototype=statement,
            acceptable_range=(statement, statement)
        )
        
        return InterpretationResult(
            original=statement,
            category=generic,
            literal_value=statement,
            confidence=0.3,
            context_applied=context
        )
    
    def is_good_enough(self,
                      result: Any,
                      goal: str,
                      context: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if result is "good enough" for the goal.
        
        This is the key function: instead of checking exact match,
        check functional acceptability.
        
        Args:
            result: The result to evaluate
            goal: The goal statement (will be interpreted)
            context: Optional context
            
        Returns:
            Tuple of (is_acceptable, reason)
        """
        # Interpret the goal
        interpretation = self.interpret(goal, context)
        category = interpretation.category
        
        # Check if result falls in acceptable range
        if category.contains(result):
            return True, f"Result {result} is within '{category.name}' range"
        
        # Check if close enough to proceed
        try:
            prototype = category.prototype
            min_val, max_val = category.acceptable_range
            
            # Calculate how far outside the range
            if result < min_val:
                gap = (min_val - result) / min_val
            elif result > max_val:
                gap = (result - max_val) / max_val
            else:
                gap = 0
            
            # Allow 20% overage as "close enough"
            if gap < 0.2:
                return True, f"Result {result} is close enough to '{category.name}'"
            
            return False, f"Result {result} is outside '{category.name}' range"
            
        except (TypeError, ValueError):
            # Can't do numeric comparison
            return result == category.prototype, "Exact match check"
    
    def evaluate_completion(self,
                           progress: float,
                           standard: str = "done") -> Tuple[bool, str]:
        """
        Evaluate if progress level counts as completion.
        
        Args:
            progress: Progress value (0-1)
            standard: Completion standard to use
            
        Returns:
            Tuple of (is_complete, status)
        """
        category = self.categories.get(standard, self.categories['done'])
        
        if category.contains(progress):
            return True, f"Completed ({progress*100:.0f}% meets '{standard}' standard)"
        
        # Check if almost there
        almost = self.categories.get('almost')
        if almost and almost.contains(progress):
            return False, f"Almost done ({progress*100:.0f}%)"
        
        started = self.categories.get('started')
        if started and started.contains(progress):
            return False, f"Just started ({progress*100:.0f}%)"
        
        return False, f"In progress ({progress*100:.0f}%)"
    
    def normalize_quantity(self,
                          quantity: Union[int, str],
                          context: Optional[str] = None) -> FunctionalCategory:
        """
        Normalize a quantity to a functional category.
        
        Args:
            quantity: Number or quantity word
            context: Optional context
            
        Returns:
            Appropriate FunctionalCategory
        """
        if isinstance(quantity, str):
            result = self.identify_category(quantity)
            if result:
                category_name, _ = result
                return self.categories.get(category_name, self.categories['some'])
        
        # Numeric: map to appropriate category
        if isinstance(quantity, (int, float)):
            if quantity <= 2:
                return self.categories['few']
            elif quantity <= 5:
                return self.categories['few']
            elif quantity <= 10:
                return self.categories['some']
            elif quantity <= 50:
                return self.categories['many']
            else:
                return self.categories['lot']
        
        return self.categories['some']


# Test if run directly
if __name__ == "__main__":
    interpreter = FunctionalInterpreter()
    
    # Test time interpretation
    print("Time interpretations:")
    for phrase in ["wait a minute", "just a moment", "I'll be there soon", "call me later"]:
        result = interpreter.interpret(phrase)
        cat = result.category
        print(f"  '{phrase}' -> {cat.name}: prototype={cat.prototype}, range={cat.acceptable_range}")
    
    # Test with context
    print("\nContext effects on 'a minute':")
    for ctx in ['casual', 'meeting', None]:
        result = interpreter.interpret("wait a minute", context=ctx)
        cat = result.category
        print(f"  context='{ctx}': range={cat.acceptable_range}")
    
    # Test good enough evaluation
    print("\nGood enough evaluations:")
    test_cases = [
        (45, "wait a minute"),
        (120, "wait a minute"),
        (0.87, "done"),
        (0.75, "done"),
    ]
    for result, goal in test_cases:
        ok, reason = interpreter.is_good_enough(result, goal)
        print(f"  {result} for '{goal}': {ok} - {reason}")
    
    # Test completion evaluation
    print("\nCompletion evaluations:")
    for progress in [0.1, 0.5, 0.75, 0.88, 0.95]:
        complete, status = interpreter.evaluate_completion(progress)
        print(f"  {progress*100:.0f}%: complete={complete}, {status}")

