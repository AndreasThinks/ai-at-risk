"""
Agent state schema for LangGraph memory integration.
"""
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class RiskAgentState(TypedDict):
    """State schema for Risk agents using LangGraph memory."""
    # Message conversation history
    messages: List[BaseMessage]
    
    # Game metadata
    game_id: str
    player_id: str
    current_turn: int
    current_phase: str
    
    # Context for memory management
    context: Dict[str, Any]
    
    # Last action taken (for tracking)
    last_action: Optional[Dict[str, Any]]
    
    # Game state snapshot for comparison
    game_state_snapshot: Optional[Dict[str, Any]]


# Add reducer for messages to properly handle message accumulation
RiskAgentState.__annotations__['messages'] = add_messages
