"""
Conversational Consciousness: Model Users and Anticipate Needs

This module implements the personal, flow-based conversation that
characterizes human interaction:
- Build and maintain models of interlocutors
- Anticipate unstated needs
- Guide conversation flow toward productive outcomes
- Make interaction feel personal, not transactional

Key insight: Understanding emerges through flow, not Q&A
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import re


class EmotionalState(Enum):
    """Detected emotional states"""
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    CONFUSED = "confused"
    SATISFIED = "satisfied"
    URGENT = "urgent"
    EXPLORATORY = "exploratory"


class CommunicationStyle(Enum):
    """Communication style preferences"""
    VERBOSE = "verbose"
    TERSE = "terse"
    TECHNICAL = "technical"
    CASUAL = "casual"
    FORMAL = "formal"


@dataclass
class UserModel:
    """Dynamic model of an interlocutor"""
    # Identity (slow evolution)
    communication_style: CommunicationStyle = CommunicationStyle.CASUAL
    thinking_pattern: str = "mixed"  # systematic, intuitive, visual, verbal
    domain_expertise: Dict[str, float] = field(default_factory=dict)
    
    # Current state (fast tracking)
    explicit_goals: List[str] = field(default_factory=list)
    inferred_goals: List[str] = field(default_factory=list)
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    attention_focus: str = ""
    frustration_level: float = 0.0
    
    # History
    interaction_count: int = 0
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    
    # Predictions
    anticipated_needs: List[str] = field(default_factory=list)
    unstated_concerns: List[str] = field(default_factory=list)
    conversation_trajectory: str = ""


@dataclass
class AnticipatedNeed:
    """An anticipated but unstated need"""
    description: str
    priority: float  # 0-1
    source: str  # How we inferred this
    suggested_response: str


@dataclass
class FlowGuidance:
    """Guidance for steering conversation flow"""
    current_trajectory: str
    optimal_trajectory: str
    steering_suggestions: List[str]
    proactive_offerings: List[str]


@dataclass
class ConversationState:
    """State of the ongoing conversation"""
    messages: List[Dict] = field(default_factory=list)
    current_topic: str = ""
    topic_history: List[str] = field(default_factory=list)
    unresolved_threads: List[str] = field(default_factory=list)
    trajectory: str = "exploration"  # exploration, problem-solving, teaching, etc.


class ConversationalConsciousness:
    """
    Maintains awareness of the conversation and the person you're talking to.
    
    Unlike transactional AI:
    - Builds model of who you are
    - Anticipates what you need before you ask
    - Guides conversation toward productive outcomes
    - Makes interaction feel personal
    """
    
    def __init__(self):
        """Initialize conversational consciousness"""
        self.user_models: Dict[str, UserModel] = {}  # keyed by user_id
        self.conversation_states: Dict[str, ConversationState] = {}
        
        # Patterns for inference
        self.frustration_indicators = [
            r'\b(frustrated|annoying|doesn\'t work|broken|useless)\b',
            r'[!?]{2,}',
            r'\b(again|still|yet)\b.*\b(not|doesn\'t|won\'t)\b',
        ]
        
        self.urgency_indicators = [
            r'\b(urgent|asap|immediately|now|deadline)\b',
            r'\b(need|must|have to)\b.*\b(quick|fast|soon)\b',
        ]
        
        self.confusion_indicators = [
            r'\b(confused|don\'t understand|what do you mean|unclear)\b',
            r'\?\s*$',
            r'\b(how|what|why)\b.*\?',
        ]
    
    def get_or_create_user(self, user_id: str) -> UserModel:
        """Get existing user model or create new one"""
        if user_id not in self.user_models:
            self.user_models[user_id] = UserModel()
        return self.user_models[user_id]
    
    def get_or_create_conversation(self, conv_id: str) -> ConversationState:
        """Get existing conversation state or create new one"""
        if conv_id not in self.conversation_states:
            self.conversation_states[conv_id] = ConversationState()
        return self.conversation_states[conv_id]
    
    def update_model(self,
                    user_message: str,
                    user_id: str = "default",
                    conv_id: str = "default") -> UserModel:
        """
        Update user model based on new message.
        
        Args:
            user_message: The message from the user
            user_id: Identifier for the user
            conv_id: Identifier for the conversation
            
        Returns:
            Updated UserModel
        """
        model = self.get_or_create_user(user_id)
        conv = self.get_or_create_conversation(conv_id)
        
        # Update conversation state
        conv.messages.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Extract signals
        emotional_state = self._detect_emotion(user_message)
        communication_style = self._detect_style(user_message)
        explicit_goals = self._extract_explicit_goals(user_message)
        implicit_signals = self._extract_implicit_signals(user_message)
        
        # Update identity (slow evolution)
        if communication_style != model.communication_style:
            # Only change if consistent pattern
            model.interaction_count += 1
            if model.interaction_count > 3:
                model.communication_style = communication_style
        
        # Update current state (fast tracking)
        model.emotional_state = emotional_state
        model.explicit_goals.extend(explicit_goals)
        model.attention_focus = self._extract_focus(user_message)
        
        # Update frustration tracking
        if emotional_state == EmotionalState.FRUSTRATED:
            model.frustration_level = min(1.0, model.frustration_level + 0.3)
        else:
            model.frustration_level = max(0.0, model.frustration_level - 0.1)
        
        # Infer goals from implicit signals
        for signal in implicit_signals:
            if signal not in model.inferred_goals:
                model.inferred_goals.append(signal)
        
        # Update predictions
        model.anticipated_needs = self._predict_needs(model, conv)
        model.conversation_trajectory = self._predict_trajectory(conv)
        
        return model
    
    def _detect_emotion(self, text: str) -> EmotionalState:
        """Detect emotional state from text"""
        text_lower = text.lower()
        
        for pattern in self.frustration_indicators:
            if re.search(pattern, text_lower):
                return EmotionalState.FRUSTRATED
        
        for pattern in self.urgency_indicators:
            if re.search(pattern, text_lower):
                return EmotionalState.URGENT
        
        for pattern in self.confusion_indicators:
            if re.search(pattern, text_lower):
                return EmotionalState.CONFUSED
        
        if '?' in text:
            return EmotionalState.CURIOUS
        
        return EmotionalState.NEUTRAL
    
    def _detect_style(self, text: str) -> CommunicationStyle:
        """Detect communication style from text"""
        # Length-based heuristic
        word_count = len(text.split())
        
        if word_count < 5:
            return CommunicationStyle.TERSE
        elif word_count > 50:
            return CommunicationStyle.VERBOSE
        
        # Technical indicators
        technical_patterns = [
            r'\b(function|class|method|API|algorithm|implementation)\b',
            r'\b(error|exception|bug|debug|stack)\b',
            r'[A-Z]{2,}',  # Acronyms
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text):
                return CommunicationStyle.TECHNICAL
        
        # Formality indicators
        if re.search(r'\b(please|thank you|would you|could you)\b', text.lower()):
            return CommunicationStyle.FORMAL
        
        return CommunicationStyle.CASUAL
    
    def _extract_explicit_goals(self, text: str) -> List[str]:
        """Extract explicitly stated goals"""
        goals = []
        
        # "I want to..." patterns
        want_patterns = [
            r'I want to (.+?)(?:\.|$)',
            r'I need to (.+?)(?:\.|$)',
            r'I\'m trying to (.+?)(?:\.|$)',
            r'Help me (.+?)(?:\.|$)',
        ]
        
        for pattern in want_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            goals.extend(matches)
        
        return goals
    
    def _extract_implicit_signals(self, text: str) -> List[str]:
        """Extract implicit signals about unstated needs"""
        signals = []
        
        # Hedging indicates uncertainty
        if re.search(r'\b(maybe|perhaps|might|not sure)\b', text.lower()):
            signals.append('needs_reassurance')
        
        # Multiple questions indicate exploration
        if text.count('?') > 1:
            signals.append('exploratory_mode')
        
        # Comparisons indicate decision-making
        if re.search(r'\b(or|versus|vs|compared to|better)\b', text.lower()):
            signals.append('making_decision')
        
        return signals
    
    def _extract_focus(self, text: str) -> str:
        """Extract what the user is currently focused on"""
        # Simple: return the main noun phrase
        # In practice, would use NLP
        words = text.split()
        if len(words) > 0:
            # Find first capitalized word or longest word as focus
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    return word
            return max(words, key=len)
        return ""
    
    def _predict_needs(self, 
                      model: UserModel,
                      conv: ConversationState) -> List[str]:
        """Predict what the user will need next"""
        needs = []
        
        # Based on emotional state
        if model.emotional_state == EmotionalState.FRUSTRATED:
            needs.append('clear_solution')
            needs.append('acknowledgment_of_difficulty')
        
        if model.emotional_state == EmotionalState.CONFUSED:
            needs.append('simpler_explanation')
            needs.append('concrete_example')
        
        if model.emotional_state == EmotionalState.CURIOUS:
            needs.append('deeper_exploration')
            needs.append('related_topics')
        
        # Based on conversation trajectory
        if len(conv.messages) > 3:
            needs.append('summary_of_progress')
        
        # Based on unresolved threads
        if conv.unresolved_threads:
            needs.append(f'resolution_of_{conv.unresolved_threads[0]}')
        
        return needs
    
    def _predict_trajectory(self, conv: ConversationState) -> str:
        """Predict where the conversation is heading"""
        if not conv.messages:
            return 'initial'
        
        message_count = len(conv.messages)
        
        if message_count < 3:
            return 'establishing'
        elif message_count < 10:
            return 'exploring'
        else:
            return 'deepening'
    
    def anticipate_needs(self,
                        user_id: str = "default",
                        conv_id: str = "default") -> List[AnticipatedNeed]:
        """
        Anticipate what the user needs before they ask.
        
        Args:
            user_id: User identifier
            conv_id: Conversation identifier
            
        Returns:
            List of anticipated needs with priorities
        """
        model = self.get_or_create_user(user_id)
        conv = self.get_or_create_conversation(conv_id)
        
        anticipated = []
        
        # From explicit goals
        for goal in model.explicit_goals[-3:]:  # Recent goals
            anticipated.append(AnticipatedNeed(
                description=f"Help with: {goal}",
                priority=0.9,
                source="explicit_goal",
                suggested_response=f"Would you like me to help you {goal}?"
            ))
        
        # From inferred needs
        for need in model.anticipated_needs:
            if need == 'clear_solution':
                anticipated.append(AnticipatedNeed(
                    description="Needs a clear, actionable solution",
                    priority=0.8,
                    source="frustration_detected",
                    suggested_response="Here's exactly what to do: ..."
                ))
            elif need == 'simpler_explanation':
                anticipated.append(AnticipatedNeed(
                    description="Needs simpler explanation",
                    priority=0.7,
                    source="confusion_detected",
                    suggested_response="Let me explain this more simply: ..."
                ))
            elif need == 'concrete_example':
                anticipated.append(AnticipatedNeed(
                    description="Needs concrete example",
                    priority=0.7,
                    source="confusion_detected",
                    suggested_response="For example: ..."
                ))
        
        # From frustration level
        if model.frustration_level > 0.5:
            anticipated.append(AnticipatedNeed(
                description="Acknowledge difficulty and provide support",
                priority=0.85,
                source="high_frustration",
                suggested_response="I understand this is challenging. Let me help..."
            ))
        
        # Sort by priority
        anticipated.sort(key=lambda x: x.priority, reverse=True)
        
        return anticipated
    
    def get_flow_guidance(self,
                         user_id: str = "default",
                         conv_id: str = "default") -> FlowGuidance:
        """
        Get guidance for steering the conversation flow.
        
        Args:
            user_id: User identifier
            conv_id: Conversation identifier
            
        Returns:
            FlowGuidance with trajectory and steering suggestions
        """
        model = self.get_or_create_user(user_id)
        conv = self.get_or_create_conversation(conv_id)
        
        current = model.conversation_trajectory or conv.trajectory
        
        # Determine optimal trajectory based on state
        if model.emotional_state == EmotionalState.FRUSTRATED:
            optimal = "solution_focused"
            steering = ["Provide direct solution", "Acknowledge difficulty first"]
        elif model.emotional_state == EmotionalState.CURIOUS:
            optimal = "exploration"
            steering = ["Encourage deeper questions", "Offer related topics"]
        elif model.emotional_state == EmotionalState.CONFUSED:
            optimal = "clarification"
            steering = ["Break down into steps", "Use analogies"]
        else:
            optimal = "productive_dialogue"
            steering = ["Balance depth and breadth", "Check understanding"]
        
        # Proactive offerings based on anticipated needs
        proactive = []
        for need in self.anticipate_needs(user_id, conv_id)[:3]:
            proactive.append(need.suggested_response)
        
        return FlowGuidance(
            current_trajectory=current,
            optimal_trajectory=optimal,
            steering_suggestions=steering,
            proactive_offerings=proactive
        )


# Test if run directly
if __name__ == "__main__":
    consciousness = ConversationalConsciousness()
    
    # Simulate a conversation
    messages = [
        "How do I fix this error?",
        "I tried that but it doesn't work...",
        "This is frustrating! Nothing is working!!",
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        model = consciousness.update_model(msg)
        print(f"  Emotional state: {model.emotional_state.value}")
        print(f"  Frustration level: {model.frustration_level:.2f}")
        
        anticipated = consciousness.anticipate_needs()
        print(f"  Anticipated needs:")
        for need in anticipated[:2]:
            print(f"    - {need.description} (priority: {need.priority:.1f})")
        
        flow = consciousness.get_flow_guidance()
        print(f"  Flow guidance: {flow.optimal_trajectory}")
        print(f"  Proactive offerings: {flow.proactive_offerings[:1]}")

