"""
Human-Reasoning AI Pipeline: Integrated Processing

This module integrates all components into a unified processing pipeline
that implements human-like reasoning patterns:

1. Receive input
2. Interpret functionally (not exactly)
3. Extract S-entropy coordinates
4. Update user model
5. Anticipate needs
6. Navigate to solution
7. Complete gaps
8. Generate response

The result is AI that thinks in human patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .substrate_harvester import SubstrateHarvester, SEntropyCoordinates
from .partition_selector import PartitionSelector, Partition, NavigationResult
from .categorical_completer import CategoricalCompleter, CompletionResult
from .conversational_consciousness import (
    ConversationalConsciousness, 
    UserModel, 
    AnticipatedNeed,
    FlowGuidance
)
from .functional_interpreter import FunctionalInterpreter, InterpretationResult


@dataclass
class PipelineInput:
    """Input to the pipeline"""
    message: str
    user_id: str = "default"
    conversation_id: str = "default"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResponse:
    """Response from the pipeline"""
    # Core response
    content: str
    
    # What was understood
    interpreted_meaning: str
    completed_gaps: List[str]
    
    # Proactive elements
    anticipated_needs_addressed: List[str]
    flow_guidance_applied: str
    
    # Metadata
    s_entropy_coords: SEntropyCoordinates
    user_model_summary: Dict[str, Any]
    confidence: float
    
    # Debug info
    processing_stages: Dict[str, Any] = field(default_factory=dict)


class HumanReasoningPipeline:
    """
    Integrated pipeline for human-like reasoning.
    
    Processes input through all stages:
    - Functional interpretation
    - S-entropy extraction
    - User modeling
    - Need anticipation
    - Solution navigation
    - Gap completion
    - Response generation
    """
    
    def __init__(self,
                 response_generator: Optional[callable] = None):
        """
        Initialize the pipeline with all components.
        
        Args:
            response_generator: Optional function to generate final response
                               Default uses simple template-based generation
        """
        self.substrate_harvester = SubstrateHarvester()
        self.partition_selector = PartitionSelector()
        self.categorical_completer = CategoricalCompleter()
        self.conversational_consciousness = ConversationalConsciousness()
        self.functional_interpreter = FunctionalInterpreter()
        
        self.response_generator = response_generator or self._default_response_generator
        
        # Available response partitions (templates)
        self.response_partitions = self._build_response_partitions()
    
    def _build_response_partitions(self) -> List[Partition]:
        """Build available response patterns"""
        return [
            Partition(
                id="direct_answer",
                content={"type": "direct", "style": "concise"},
                s_entropy=0.2
            ),
            Partition(
                id="exploratory",
                content={"type": "exploratory", "style": "questioning"},
                s_entropy=0.5
            ),
            Partition(
                id="supportive",
                content={"type": "supportive", "style": "empathetic"},
                s_entropy=0.3
            ),
            Partition(
                id="educational",
                content={"type": "educational", "style": "explanatory"},
                s_entropy=0.4
            ),
            Partition(
                id="clarifying",
                content={"type": "clarifying", "style": "questioning"},
                s_entropy=0.6
            ),
        ]
    
    def process(self, input_data: PipelineInput) -> PipelineResponse:
        """
        Process input through the full pipeline.
        
        Args:
            input_data: PipelineInput with message and context
            
        Returns:
            PipelineResponse with response and metadata
        """
        stages = {}
        
        # Stage 1: Functional Interpretation
        interpretation = self.functional_interpreter.interpret(
            input_data.message,
            context=input_data.context.get('context_type')
        )
        stages['interpretation'] = {
            'category': interpretation.category.name,
            'confidence': interpretation.confidence
        }
        
        # Stage 2: S-Entropy Extraction
        s_coords = self.substrate_harvester.extract(
            input_data.message,
            input_data.context
        )
        stages['s_entropy'] = {
            'S_k': s_coords.S_k,
            'S_t': s_coords.S_t,
            'S_e': s_coords.S_e,
            'magnitude': s_coords.magnitude()
        }
        
        # Stage 3: Update User Model
        user_model = self.conversational_consciousness.update_model(
            input_data.message,
            input_data.user_id,
            input_data.conversation_id
        )
        stages['user_model'] = {
            'emotional_state': user_model.emotional_state.value,
            'frustration': user_model.frustration_level,
            'style': user_model.communication_style.value
        }
        
        # Stage 4: Anticipate Needs
        anticipated = self.conversational_consciousness.anticipate_needs(
            input_data.user_id,
            input_data.conversation_id
        )
        stages['anticipated'] = [
            {'need': n.description, 'priority': n.priority}
            for n in anticipated[:3]
        ]
        
        # Stage 5: Get Flow Guidance
        flow = self.conversational_consciousness.get_flow_guidance(
            input_data.user_id,
            input_data.conversation_id
        )
        stages['flow'] = {
            'current': flow.current_trajectory,
            'optimal': flow.optimal_trajectory
        }
        
        # Stage 6: Navigate to Response Partition
        # Adjust partitions based on user state
        adjusted_partitions = self._adjust_partitions_for_user(
            self.response_partitions,
            user_model,
            anticipated
        )
        
        nav_result = self.partition_selector.navigate_and_select(
            s_coords,
            adjusted_partitions
        )
        stages['navigation'] = {
            'selected': nav_result.selected_partition.id,
            'confidence': nav_result.confidence,
            'steps': nav_result.path_length
        }
        
        # Stage 7: Complete Gaps
        completion = self.categorical_completer.complete(
            input_data.message,
            context=input_data.context,
            user_model={'style': user_model.communication_style.value}
        )
        stages['completion'] = {
            'gaps_found': len(completion.gaps_found),
            'strategies': list(completion.strategies_used.values())
        }
        
        # Stage 8: Generate Response
        response_content = self.response_generator(
            nav_result.selected_partition,
            completion,
            anticipated,
            flow,
            user_model,
            input_data
        )
        
        # Build full response
        return PipelineResponse(
            content=response_content,
            interpreted_meaning=f"Understood as '{interpretation.category.name}' with {interpretation.confidence:.0%} confidence",
            completed_gaps=[g.category for g in completion.gaps_found],
            anticipated_needs_addressed=[n.description for n in anticipated[:2]],
            flow_guidance_applied=flow.optimal_trajectory,
            s_entropy_coords=s_coords,
            user_model_summary={
                'emotional_state': user_model.emotional_state.value,
                'frustration_level': user_model.frustration_level,
                'communication_style': user_model.communication_style.value
            },
            confidence=nav_result.confidence,
            processing_stages=stages
        )
    
    def _adjust_partitions_for_user(self,
                                   partitions: List[Partition],
                                   user_model: UserModel,
                                   anticipated: List[AnticipatedNeed]) -> List[Partition]:
        """Adjust partition S-entropy based on user state"""
        adjusted = []
        
        for p in partitions:
            new_entropy = p.s_entropy
            
            # Frustrated users need supportive, direct responses
            if user_model.frustration_level > 0.5:
                if p.id == 'supportive':
                    new_entropy *= 0.5  # More likely
                elif p.id == 'exploratory':
                    new_entropy *= 2.0  # Less likely
            
            # Confused users need clarifying responses
            if user_model.emotional_state.value == 'confused':
                if p.id in ['clarifying', 'educational']:
                    new_entropy *= 0.6
            
            # Curious users benefit from exploration
            if user_model.emotional_state.value == 'curious':
                if p.id == 'exploratory':
                    new_entropy *= 0.5
            
            adjusted.append(Partition(
                id=p.id,
                content=p.content,
                s_entropy=new_entropy
            ))
        
        return adjusted
    
    def _default_response_generator(self,
                                   partition: Partition,
                                   completion: CompletionResult,
                                   anticipated: List[AnticipatedNeed],
                                   flow: FlowGuidance,
                                   user_model: UserModel,
                                   input_data: PipelineInput) -> str:
        """
        Default response generator using templates.
        
        In practice, this would integrate with an LLM.
        """
        response_type = partition.content.get('type', 'direct')
        
        # Build response based on type
        if response_type == 'direct':
            response = f"[Direct response to: {input_data.message}]"
        elif response_type == 'supportive':
            response = f"I understand this might be challenging. [Supportive response to: {input_data.message}]"
        elif response_type == 'exploratory':
            response = f"That's an interesting direction. [Exploratory response to: {input_data.message}]"
        elif response_type == 'educational':
            response = f"Let me explain this step by step. [Educational response to: {input_data.message}]"
        elif response_type == 'clarifying':
            response = f"To make sure I understand correctly: [Clarifying response to: {input_data.message}]"
        else:
            response = f"[Response to: {input_data.message}]"
        
        # Add proactive elements based on anticipated needs
        if anticipated and anticipated[0].priority > 0.7:
            response += f"\n\n[Proactively addressing: {anticipated[0].description}]"
        
        # Add completion annotations
        if completion.gaps_found:
            gaps_str = ", ".join(g.category for g in completion.gaps_found)
            response += f"\n\n[Completed gaps: {gaps_str}]"
        
        return response


# Test if run directly  
if __name__ == "__main__":
    pipeline = HumanReasoningPipeline()
    
    # Test conversation
    messages = [
        "How do I fix this error?",
        "I tried that but it doesn't work...",
        "This is frustrating! Nothing is working!!",
        "Okay, let me try something different",
    ]
    
    print("=" * 60)
    print("Human-Reasoning AI Pipeline Demo")
    print("=" * 60)
    
    for msg in messages:
        print(f"\n{'='*60}")
        print(f"USER: {msg}")
        print("-" * 60)
        
        input_data = PipelineInput(message=msg)
        response = pipeline.process(input_data)
        
        print(f"\nAI RESPONSE:")
        print(response.content)
        
        print(f"\n[METADATA]")
        print(f"  Interpreted: {response.interpreted_meaning}")
        print(f"  Gaps filled: {response.completed_gaps}")
        print(f"  User state: {response.user_model_summary}")
        print(f"  Flow guidance: {response.flow_guidance_applied}")
        print(f"  S-entropy: ({response.s_entropy_coords.S_k:.3f}, {response.s_entropy_coords.S_t:.3f}, {response.s_entropy_coords.S_e:.3f})")
        print(f"  Confidence: {response.confidence:.2f}")

