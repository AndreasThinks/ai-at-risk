"""
Agent Memory Manager - Core memory management for Risk AI agents using LangGraph.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from .agent_state import RiskAgentState
from .game_narrator import GameNarrator
from .memory_policies import RiskMemoryPolicy, RiskConversationManager


class AgentMemoryManager:
    """
    Manages persistent conversations and memory for Risk AI agents.
    
    This class handles:
    - Thread creation and management per agent per game
    - Conversation history with intelligent trimming
    - Game state updates and narrative generation
    - Integration with existing context gathering
    """
    
    def __init__(self, checkpointer=None):
        """
        Initialize the memory manager.
        
        Args:
            checkpointer: LangGraph checkpointer (defaults to InMemorySaver)
        """
        self.checkpointer = checkpointer or InMemorySaver()
        self.game_narrator = GameNarrator()
        self.memory_policy = RiskMemoryPolicy(max_tokens=8000)
        self.conversation_manager = RiskConversationManager(self.memory_policy)
        
        # Track active conversations per game
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger("memory_manager")
    
    def get_thread_id(self, game_id: str, player_id: str) -> str:
        """Generate consistent thread ID for agent conversations."""
        return f"risk_game_{game_id}_player_{player_id}"
    
    async def initialize_agent_conversation(
        self, 
        game_id: str, 
        player_id: str, 
        agent_name: str,
        custom_instructions: str = None
    ) -> bool:
        """
        Initialize a new conversation thread for an agent.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier
            agent_name: Agent's name
            custom_instructions: Custom playing style instructions
            
        Returns:
            True if successfully initialized
        """
        try:
            thread_id = self.get_thread_id(game_id, player_id)
            
            # Create initial system message with game rules and character instructions
            system_content = self._create_system_message_content(agent_name, custom_instructions)
            
            # Create initial conversation state
            initial_state: RiskAgentState = {
                "messages": [SystemMessage(content=system_content)],
                "game_id": game_id,
                "player_id": player_id,
                "current_turn": 0,
                "current_phase": "setup",
                "context": {},
                "last_action": None,
                "game_state_snapshot": None
            }
            
            # Store conversation info
            conversation_key = f"{game_id}_{player_id}"
            self.active_conversations[conversation_key] = {
                "thread_id": thread_id,
                "agent_name": agent_name,
                "game_id": game_id,
                "player_id": player_id,
                "initialized_at": datetime.now(),
                "turn_count": 0
            }
            
            # Initialize the conversation with system message
            config = {"configurable": {"thread_id": thread_id}}
            
            # For now, we'll just store the initial state in memory
            # The actual LangGraph integration will be handled when the agent makes decisions
            
            self.logger.info(f"Initialized conversation for {agent_name} in game {game_id}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error initializing conversation for {agent_name}: {e}")
            return False
    
    def _create_system_message_content(self, agent_name: str, custom_instructions: str = None) -> str:
        """Create the system message content for agent conversations."""
        base_instructions = f"""You are {agent_name}, an AI agent playing the board game Risk.

# GAME OBJECTIVE
Your goal is to achieve world domination by conquering all territories on the board. You must eliminate all other players to win.

# YOUR ROLE
You are an autonomous player making strategic decisions in a turn-based Risk game. Each turn, you'll receive updates about the current game state and must decide on your actions.

# CORE GAME RULES
- Turn Structure: Each turn has phases - Reinforcement → Attack → Fortify
- Reinforcements: Get armies based on territories owned (minimum 3, or territories÷3)
- Continent Bonuses: Control entire continents for extra armies each turn
- Combat: Roll dice to resolve battles, higher rolls win
- Cards: Earn cards by conquering territories, trade sets for army bonuses

# DECISION MAKING
You have access to MCP tools that let you:
- place_armies: Deploy reinforcement armies to your territories
- attack_territory: Attack adjacent enemy territories  
- fortify_position: Move armies between your territories
- trade_cards: Exchange card sets for army bonuses
- send_message: Communicate with other players diplomatically
- end_turn: Complete your turn and advance to next player

# STRATEGIC PRINCIPLES
- Maintain conversational memory of the game's progression
- Adapt your strategy based on game state changes
- Use diplomacy strategically to form alliances and deceive opponents
- Focus on continent control for steady army income
- Balance offensive expansion with defensive consolidation"""

        if custom_instructions:
            base_instructions += f"\n\n# YOUR PLAYING STYLE\n{custom_instructions}"
        
        base_instructions += """

# CONVERSATION MEMORY
This conversation will track your ongoing experience in this Risk game. Each turn, you'll receive:
1. Updates on what changed since your last turn
2. Current game state and your options
3. Your previous decisions and their outcomes

Use this conversational context to make informed, strategic decisions that build upon your previous actions and adapt to the evolving game state."""

        return base_instructions
    
    async def update_agent_conversation(
        self,
        game_id: str,
        player_id: str, 
        context_data: Dict[str, Any],
        turn_number: int,
        phase: str
    ) -> Optional[str]:
        """
        Update agent's conversation with current turn context.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier  
            context_data: Current game context data
            turn_number: Current turn number
            phase: Current game phase
            
        Returns:
            Formatted conversation context for agent decision making
        """
        try:
            conversation_key = f"{game_id}_{player_id}"
            thread_id = self.get_thread_id(game_id, player_id)
            
            if conversation_key not in self.active_conversations:
                self.logger.error(f"No active conversation found for {game_id}_{player_id}")
                return None
            
            # Generate narrative update about what changed
            current_state = {
                'player_info': context_data.get('player_info', {}),
                'game_status': context_data.get('game_status', {}),
                'board_state': context_data.get('board_state', {})
            }
            
            # Generate turn update narrative
            turn_update = self.game_narrator.generate_turn_update(
                game_id=game_id,
                player_id=player_id,
                current_state=current_state,
                current_phase=phase,
                last_action=context_data.get('last_action')
            )
            
            # Create comprehensive turn context combining narrative + detailed info
            turn_context = self._create_comprehensive_turn_context(context_data, turn_update, turn_number, phase)
            
            # Update conversation tracking
            self.active_conversations[conversation_key]["turn_count"] = turn_number
            self.active_conversations[conversation_key]["last_update"] = datetime.now()
            
            self.logger.info(f"Updated conversation context for turn {turn_number}, phase {phase}")
            return turn_context
            
        except Exception as e:
            self.logger.exception(f"Error updating agent conversation: {e}")
            return None
    
    def _create_comprehensive_turn_context(
        self, 
        context_data: Dict[str, Any], 
        narrative_update: str, 
        turn_number: int, 
        phase: str
    ) -> str:
        """
        Create comprehensive turn context combining narrative and detailed data.
        
        This replaces the old format_turn_context method with memory-aware formatting.
        """
        sections = []
        
        # 1. Narrative Turn Update (NEW - from GameNarrator)
        sections.append(narrative_update)
        
        # 2. Current Phase Status & Actions Available
        game_status = context_data.get("game_status", {})
        player_info = context_data.get("player_info", {})
        current_phase = game_status.get('phase', phase)
        army_count = player_info.get('army_count', 0)
        
        phase_guidance = self._generate_phase_guidance(current_phase, army_count, player_info, game_status)
        sections.append(phase_guidance)
        
        # 3. Your Current Status
        territories = player_info.get('territories', [])
        cards = player_info.get('cards', [])
        
        status_lines = [
            "## YOUR CURRENT STATUS",
            f"Turn: {turn_number} | Phase: {current_phase.upper()}",
            f"Army Count: {army_count} armies available to place",
            f"Territories: {len(territories)} controlled (need 42 to win)",
            f"Cards: {len(cards)} in hand"
        ]
        
        # Add card details if present
        if cards and len(cards) >= 3:
            status_lines.append("⚠️ You can trade cards for army bonus!")
        
        # Add continent bonuses
        continent_bonuses = player_info.get('continent_bonuses', {})
        if continent_bonuses:
            total_bonus = sum(continent_bonuses.values())
            status_lines.append(f"Continent Bonuses: +{total_bonus} armies/turn")
            for continent, bonus in continent_bonuses.items():
                status_lines.append(f"  • {continent}: +{bonus}")
        
        sections.append("\n".join(status_lines))
        
        # 4. Strategic Recommendations (condensed from old system)
        strategic_recs = self._generate_strategic_recommendations(context_data)
        if strategic_recs:
            sections.append(strategic_recs)
        
        # 5. Recent Actions Context (condensed)
        recent_actions = context_data.get("recent_actions", [])
        if recent_actions:
            recent_context = self._format_recent_actions_summary(recent_actions[:10])  # Last 10 actions
            sections.append(recent_context)
        
        # 6. Action Priority Guide
        sections.append(self._generate_action_priority_guide(current_phase, army_count))
        
        return "\n\n".join(sections)
    
    def _generate_phase_guidance(self, current_phase: str, army_count: int, player_info: Dict[str, Any], game_status: Dict[str, Any]) -> str:
        """Generate phase-specific guidance (condensed version)."""
        lines = ["## CURRENT PHASE GUIDANCE"]
        
        if current_phase == "reinforcement":
            lines.extend([
                f"**REINFORCEMENT PHASE** - Place {army_count} armies",
                "- MUST place all armies before advancing to attack phase",
                "- Consider vulnerable borders and attack positions",
                "- Trade cards first if you have 3+ cards for bonus armies"
            ])
        elif current_phase == "attack":
            lines.extend([
                "**ATTACK PHASE** - Conquer enemy territories", 
                "- Attack from territories with 2+ armies",
                "- Target weak enemy territories (1-2 armies)",
                "- Earn cards by conquering at least one territory"
            ])
        elif current_phase == "fortify":
            lines.extend([
                "**FORTIFY PHASE** - Reposition armies (optional)",
                "- Move armies from safe territories to borders",
                "- Strengthen territories for next turn's attacks",
                "- Can skip fortification and end turn"
            ])
        else:
            lines.extend([
                f"**{current_phase.upper()} PHASE**",
                f"- Army Count: {army_count}",
                "- Follow standard Risk turn sequence"
            ])
        
        return "\n".join(lines)
    
    def _generate_strategic_recommendations(self, context_data: Dict[str, Any]) -> str:
        """Generate condensed strategic recommendations."""
        player_info = context_data.get("player_info", {})
        army_count = player_info.get('army_count', 0)
        cards = player_info.get('cards', [])
        
        lines = ["## STRATEGIC PRIORITIES"]
        
        # Card trading priority
        if len(cards) >= 3:
            lines.append("🎴 URGENT: Trade cards for army bonus!")
        
        # Army placement priority  
        if army_count > 0:
            lines.append(f"🛡️ MUST DO: Place {army_count} armies strategically")
        
        # Add 2-3 key strategic insights
        territories = len(player_info.get('territories', []))
        if territories < 10:
            lines.append("📈 FOCUS: Secure continent control for steady income")
        elif territories < 25:
            lines.append("⚔️ FOCUS: Eliminate weak players, expand aggressively") 
        else:
            lines.append("🏆 ENDGAME: All-out assault for world domination!")
        
        return "\n".join(lines)
    
    def _format_recent_actions_summary(self, recent_actions: List[Dict[str, Any]]) -> str:
        """Format recent actions into a condensed summary."""
        if not recent_actions:
            return "## RECENT GAME ACTIVITY\nNo recent actions available."
        
        lines = ["## RECENT GAME ACTIVITY"]
        
        # Group by turn and show last 3 turns
        turn_groups = {}
        for action in recent_actions:
            turn = action.get('turn_number', 0)
            if turn not in turn_groups:
                turn_groups[turn] = []
            turn_groups[turn].append(action)
        
        recent_turns = sorted(turn_groups.keys(), reverse=True)[:3]
        
        for turn in recent_turns:
            actions = turn_groups[turn]
            player_actions = {}
            
            # Group by player
            for action in actions:
                player = action.get('player_id', 'unknown')
                if player not in player_actions:
                    player_actions[player] = []
                player_actions[player].append(action.get('action_type', 'unknown'))
            
            # Format turn summary
            turn_summary = f"Turn {turn}:"
            for player, action_types in player_actions.items():
                action_summary = ", ".join(set(action_types))  # Remove duplicates
                turn_summary += f" Player {player} → {action_summary}"
            
            lines.append(turn_summary)
        
        return "\n".join(lines)
    
    def _generate_action_priority_guide(self, current_phase: str, army_count: int) -> str:
        """Generate action priority guide for current situation."""
        lines = ["## ACTION PRIORITY GUIDE"]
        
        if current_phase == "reinforcement" and army_count > 0:
            lines.extend([
                "**IMMEDIATE PRIORITY:**",
                "1. 🎴 trade_cards (if you have 3+ cards)",
                f"2. 🛡️ place_armies (MUST place all {army_count} armies)",
                "3. 💬 send_message (diplomatic opportunities)", 
                "4. ⚠️ NEVER call end_turn until army_count = 0"
            ])
        elif current_phase == "attack":
            lines.extend([
                "**ATTACK PHASE PRIORITIES:**",
                "1. ⚔️ attack_territory (target weak enemies)",
                "2. 🎴 trade_cards (if available)",
                "3. 🛡️ place_armies (if any remaining)",
                "4. 💬 send_message (diplomacy)",
                "5. ⏹️ end_turn (advance to fortify)"
            ])
        elif current_phase == "fortify":
            lines.extend([
                "**FORTIFY PHASE PRIORITIES:**",
                "1. 🏰 fortify_position (optional repositioning)",
                "2. 💬 send_message (diplomacy)",
                "3. ⏹️ end_turn (complete turn)"
            ])
        else:
            lines.extend([
                "**STANDARD PRIORITIES:**",
                "1. 🎴 trade_cards → 🛡️ place_armies → ⚔️ attack → 🏰 fortify → ⏹️ end_turn",
                "2. Use tools in logical sequence for your phase"
            ])
        
        return "\n".join(lines)
    
    async def get_conversation_history(self, game_id: str, player_id: str) -> List[BaseMessage]:
        """
        Get the current conversation history for an agent.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier
            
        Returns:
            List of messages in conversation history
        """
        try:
            thread_id = self.get_thread_id(game_id, player_id)
            config = {"configurable": {"thread_id": thread_id}}
            
            # For now, return empty list - will be populated when integrated with LangGraph
            # In full integration, this would retrieve from checkpointer
            return []
            
        except Exception as e:
            self.logger.exception(f"Error getting conversation history: {e}")
            return []
    
    async def add_agent_response(
        self, 
        game_id: str, 
        player_id: str, 
        human_message: str, 
        ai_response: str
    ) -> bool:
        """
        Add a human message and AI response to the conversation.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier
            human_message: The context/prompt sent to agent
            ai_response: The agent's response
            
        Returns:
            True if successfully added
        """
        try:
            conversation_key = f"{game_id}_{player_id}"
            if conversation_key not in self.active_conversations:
                self.logger.error(f"No active conversation for {conversation_key}")
                return False
            
            # In full integration, this would update the conversation via LangGraph
            # For now, just log the interaction
            self.logger.info(f"Agent interaction logged for {conversation_key}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error adding agent response: {e}")
            return False
    
    def cleanup_game_conversations(self, game_id: str):
        """Clean up all conversations for a completed game."""
        try:
            keys_to_remove = [key for key in self.active_conversations.keys() if key.startswith(f"{game_id}_")]
            
            for key in keys_to_remove:
                del self.active_conversations[key]
                self.logger.info(f"Cleaned up conversation: {key}")
            
            # Also cleanup from game narrator
            narrator_keys = [key for key in self.game_narrator.previous_states.keys() if key.startswith(f"{game_id}_")]
            for key in narrator_keys:
                del self.game_narrator.previous_states[key]
            
            self.logger.info(f"Cleaned up {len(keys_to_remove)} conversations for game {game_id}")
            
        except Exception as e:
            self.logger.exception(f"Error cleaning up conversations for game {game_id}: {e}")
