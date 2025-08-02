"""
Memory management policies for Risk agents - smart trimming and summarization rules.
"""
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
import logging


class RiskMemoryPolicy:
    """Defines memory management policies specific to Risk gameplay."""
    
    def __init__(self, max_tokens: int = 8000, keep_recent_exchanges: int = 3):
        """
        Initialize memory policy.
        
        Args:
            max_tokens: Maximum tokens to keep in conversation
            keep_recent_exchanges: Number of recent Q&A exchanges to always keep
        """
        self.max_tokens = max_tokens
        self.keep_recent_exchanges = keep_recent_exchanges
        self.logger = logging.getLogger("memory_policy")
    
    def should_trim_memory(self, messages: List[BaseMessage]) -> bool:
        """Check if memory should be trimmed based on token count."""
        if not messages:
            return False
        
        token_count = count_tokens_approximately(messages)
        should_trim = token_count > self.max_tokens
        
        if should_trim:
            self.logger.info(f"Memory trimming needed: {token_count} tokens > {self.max_tokens} limit")
        
        return should_trim
    
    def trim_conversation(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim conversation while preserving important Risk game context.
        
        Args:
            messages: Full conversation history
            
        Returns:
            Trimmed conversation with essential context preserved
        """
        if not messages:
            return messages
        
        # Always preserve system message if present
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Use LangChain's trim_messages with Risk-specific settings
        try:
            trimmed_messages = trim_messages(
                non_system_messages,
                strategy="last",  # Keep recent messages
                token_counter=count_tokens_approximately,
                max_tokens=self.max_tokens - (count_tokens_approximately(system_messages) if system_messages else 0),
                start_on="human",  # Conversations should start with human messages
                end_on=("human", "ai"),  # Can end on either
                include_system=False,  # We handle system messages separately
                allow_partial=False
            )
            
            # Combine system messages with trimmed conversation
            result = system_messages + trimmed_messages
            
            # Log the trimming result
            original_tokens = count_tokens_approximately(messages)
            trimmed_tokens = count_tokens_approximately(result)
            self.logger.info(f"Trimmed conversation: {original_tokens} → {trimmed_tokens} tokens")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error trimming conversation: {e}")
            # Fallback: keep system + last N messages
            recent_count = min(10, len(non_system_messages))
            fallback = system_messages + non_system_messages[-recent_count:]
            self.logger.info(f"Using fallback trimming: keeping last {recent_count} messages")
            return fallback
    
    def identify_important_messages(self, messages: List[BaseMessage]) -> List[int]:
        """
        Identify indices of messages that should be preserved during trimming.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of indices of important messages
        """
        important_indices = []
        
        for i, message in enumerate(messages):
            # Always keep system messages
            if isinstance(message, SystemMessage):
                important_indices.append(i)
                continue
            
            # Check message content for importance indicators
            content = str(message.content).lower()
            
            # Game state transitions
            if any(phrase in content for phrase in [
                "phase transition", "game start", "turn update", "game over",
                "victory", "eliminated", "continent control"
            ]):
                important_indices.append(i)
                continue
            
            # Strategic decisions
            if any(phrase in content for phrase in [
                "strategy update", "alliance", "betrayal", "diplomatic",
                "attack plan", "defense strategy"
            ]):
                important_indices.append(i)
                continue
            
            # Critical game events
            if any(phrase in content for phrase in [
                "conquered", "eliminated", "card trade", "reinforcement bonus",
                "continent bonus", "major victory"
            ]):
                important_indices.append(i)
                continue
        
        return important_indices
    
    def create_conversation_summary(self, trimmed_messages: List[BaseMessage]) -> str:
        """
        Create a summary of trimmed conversation content.
        
        Args:
            trimmed_messages: Messages that were removed from conversation
            
        Returns:
            Summary string to preserve context
        """
        if not trimmed_messages:
            return ""
        
        # Extract key information from trimmed messages
        game_events = []
        strategic_decisions = []
        diplomatic_events = []
        
        for message in trimmed_messages:
            content = str(message.content).lower()
            
            # Categorize message content
            if any(term in content for term in ["attack", "conquer", "territory", "battle"]):
                game_events.append(f"Military action: {content[:100]}...")
            elif any(term in content for term in ["strategy", "plan", "objective"]):
                strategic_decisions.append(f"Strategy: {content[:100]}...")
            elif any(term in content for term in ["alliance", "negotiate", "message", "diplomatic"]):
                diplomatic_events.append(f"Diplomacy: {content[:100]}...")
        
        # Build summary
        summary_parts = []
        
        if game_events:
            summary_parts.append(f"Recent military actions ({len(game_events)}): " + "; ".join(game_events[:3]))
        
        if strategic_decisions:
            summary_parts.append(f"Strategic decisions ({len(strategic_decisions)}): " + "; ".join(strategic_decisions[:2]))
        
        if diplomatic_events:
            summary_parts.append(f"Diplomatic events ({len(diplomatic_events)}): " + "; ".join(diplomatic_events[:2]))
        
        if summary_parts:
            return "[PREVIOUS CONVERSATION SUMMARY]\n" + "\n".join(summary_parts) + "\n[END SUMMARY]\n"
        else:
            return "[PREVIOUS CONVERSATION SUMMARY: General Risk gameplay discussion]\n"


class RiskConversationManager:
    """Manages conversation flow and context for Risk agents."""
    
    def __init__(self, memory_policy: RiskMemoryPolicy):
        self.memory_policy = memory_policy
        self.logger = logging.getLogger("conversation_manager")
    
    def should_add_context_separator(self, messages: List[BaseMessage]) -> bool:
        """Check if we should add a context separator to mark new turns."""
        if not messages:
            return True
        
        # Add separator if last message was from AI (agent's response)
        return len(messages) > 0 and isinstance(messages[-1], AIMessage)
    
    def format_turn_transition_message(self, turn_number: int, phase: str) -> str:
        """Format a message to mark turn transitions."""
        return f"--- TURN {turn_number} | PHASE: {phase.upper()} ---"
    
    def add_turn_context_message(
        self, 
        messages: List[BaseMessage], 
        turn_number: int, 
        phase: str, 
        context_update: str
    ) -> List[BaseMessage]:
        """
        Add a context update message to the conversation.
        
        Args:
            messages: Current conversation history
            turn_number: Current turn number
            phase: Current game phase  
            context_update: Context update from GameNarrator
            
        Returns:
            Updated conversation with context message
        """
        # Create the context message
        separator = self.format_turn_transition_message(turn_number, phase)
        full_context = f"{separator}\n\n{context_update}"
        
        context_message = HumanMessage(content=full_context)
        
        # Add to conversation
        updated_messages = messages + [context_message]
        
        # Apply memory management if needed
        if self.memory_policy.should_trim_memory(updated_messages):
            updated_messages = self.memory_policy.trim_conversation(updated_messages)
        
        return updated_messages
    
    def extract_last_agent_action(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the last action taken by the agent from conversation."""
        # Look for the most recent AI message that mentions tool usage
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = str(message.content)
                # Look for common action patterns
                if any(action in content.lower() for action in [
                    "place_armies", "attack_territory", "fortify_position", 
                    "trade_cards", "send_message", "end_turn"
                ]):
                    return content
        
        return None
