# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import time
import httpx
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

# Handle imports for both Docker and local environments
try:
    # Try Docker-style import first (when running from /app directory)
    from persistence.action_tracker import action_tracker
except ImportError:
    # Fall back to local-style import (when running from project root)
    from src.persistence.action_tracker import action_tracker

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Import the context summarizer
try:
    from .context_summarizer import context_summarizer
except ImportError:
    context_summarizer = None

def create_llm(model_type: str = "ollama", **kwargs):
    """
    Factory function to create different types of LLMs.
    
    Args:
        model_type: Type of model ("ollama" or "openai")
        **kwargs: Additional arguments for the specific LLM
        
    Returns:
        LLM instance
    """
    if model_type.lower() == "openai":
        # OpenAI-compatible model configuration
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        
        if not api_key:
            raise ValueError("API key is required for OpenAI-compatible models")
        if not base_url:
            raise ValueError("Base URL is required for OpenAI-compatible models")
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name
        )
    
    elif model_type.lower() == "ollama":
        # Ollama model configuration (default)
        model_name = kwargs.get("model_name", "qwen3:14b")
        num_ctx = kwargs.get("num_ctx", 10240)
        
        return ChatOllama(
            model=model_name,
            num_ctx=num_ctx
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'ollama', 'openai'")

class RiskAgent:
    """
    An agent that can play the game of Risk autonomously using MCP tools.
    Each agent checks periodically if it's their turn and takes actions.
    """
    
    def __init__(self, client, agent_name: str, llm=None, llm_config=None, custom_instructions: str = None):
        """
        Initialize a Risk agent.
        
        Args:
            client: MCP client with access to Risk tools
            agent_name: Name of the agent (e.g., "Alice", "Bob")
            llm: Optional pre-configured LLM instance to use
            llm_config: Optional dict with LLM configuration:
                {
                    "model_type": "ollama" or "openai",
                    "api_key": "your-api-key" (for OpenAI-compatible),
                    "base_url": "https://api.provider.com/v1" (for OpenAI-compatible),
                    "model_name": "model-name",
                    "temperature": 0.7,
                    "max_tokens": 4000 (for OpenAI-compatible),
                    "num_ctx": 10240 (for Ollama)
                }
            custom_instructions: Optional custom playing style instructions
        """
        self.client = client
        self.name = agent_name
        self.game_id = None
        self.player_id = None
        self.is_running = False
        self.agent = None
        
        # Store custom instructions for this agent
        self.custom_instructions = custom_instructions or "Play strategically, balance offense and defense, and adapt to the game situation"
        
        # Circuit breaker to prevent strategy update loops
        self.consecutive_strategy_updates = 0
        self.max_consecutive_strategy_updates = 2
        self.last_action_was_strategy_update = False
        
        # Set up logging
        self.logger = logging.getLogger(f"risk_agent.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Try to create file handler with fallback to console-only
        try:
            # Try multiple log directories in order of preference
            log_paths = [
                f"/app/data/logs/risk_agent_{self.name}.log",  # Preferred: data/logs directory
                f"/tmp/risk_agent_{self.name}.log",            # Fallback: tmp directory
                f"./data/logs/risk_agent_{self.name}.log",     # Local development
                f"./risk_agent_{self.name}.log"                # Last resort: current directory
            ]
            
            file_handler_created = False
            for log_path in log_paths:
                try:
                    # Create directory if it doesn't exist
                    import os
                    log_dir = os.path.dirname(log_path)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
                    
                    # Try to create file handler
                    handler = logging.FileHandler(log_path)
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
                    self.logger.info(f"Agent {self.name} logging to file: {log_path}")
                    file_handler_created = True
                    break
                except (PermissionError, OSError) as e:
                    continue  # Try next path
            
            if not file_handler_created:
                self.logger.warning(f"Agent {self.name} could not create file handler, using console logging only")
                
        except Exception as e:
            self.logger.warning(f"Agent {self.name} error setting up file logging: {e}, using console logging only")
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize LLM
        if llm:
            # Use provided LLM instance
            self.llm = llm
            self.logger.info(f"Using provided LLM instance: {type(llm).__name__}")
        elif llm_config:
            # Create LLM from configuration
            try:
                self.llm = create_llm(**llm_config)
                model_type = llm_config.get("model_type", "unknown")
                model_name = llm_config.get("model_name", "unknown")
                self.logger.info(f"Created {model_type} LLM with model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to create LLM from config: {e}")
                self.logger.info("Falling back to default Ollama configuration")
                self.llm = ChatOllama(model="qwen3:14b", num_ctx=10240)
        else:
            # Use default Ollama configuration
            self.llm = ChatOllama(model="qwen3:14b", num_ctx=10240)
            self.logger.info("Using default Ollama LLM: qwen3:14b")
    
    async def initialize(self):
        """Initialize the agent with tools from the MCP server"""
        self.logger.info(f"Initializing agent {self.name}")
        
        # Get tools from MCP client - they're already LangChain-compatible
        tools = await self.client.get_tools()
        self.logger.info(f"Loaded {len(tools)} tools from MCP server")
        
        # Create the React agent - it will handle tool calling automatically
        self.agent = create_react_agent(self.llm, tools)
        self.logger.info(f"Agent {self.name} initialized successfully")
    
    async def join_game(self, game_id: str) -> bool:
        """
        Join a Risk game.
        Note: This is now handled by the GameRunner, this method now
        just stores the game ID and returns success.
        
        Args:
            game_id: ID of the game to join
            
        Returns:
            bool: True always (since GameRunner handles the actual joining)
        """
        self.logger.info(f"Game ID set to {game_id}")
        self.game_id = game_id
        return True
    
    async def gather_turn_context(self) -> Dict[str, Any]:
        """
        Gather rich context about the current game state for the agent's decision making.
        
        Returns:
            Dict[str, Any]: A dictionary containing all context information
        """
        if not self.game_id or not self.player_id:
            self.logger.warning("Can't gather context - not in a game")
            return {}
            
        context = {}
        
        try:
            async with httpx.AsyncClient() as client:
                # 1. Get current game status and player info
                self.logger.info("Gathering game status and player info")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/status", 
                    params={"player_id": self.player_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    context["game_status"] = data.get("game_status", {})
                    context["player_info"] = data.get("player_info", {})
                else:
                    self.logger.error(f"Failed to get game status: HTTP {response.status_code}")
                
                # 2. Get board state
                self.logger.info("Gathering board state")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/board", 
                    params={"player_id": self.player_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Ensure board state includes players list for opponent analysis
                    context["board_state"] = data
                    
                    # Also get player list for opponent analysis
                    game_status = context.get("game_status", {})
                    if "players" in game_status:
                        context["board_state"]["players"] = game_status["players"]
                else:
                    self.logger.error(f"Failed to get board state: HTTP {response.status_code}")
                
                # 3. Get current turn info
                self.logger.info("Gathering current turn info")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/current-turn"
                )
                
                if response.status_code == 200:
                    context["current_turn"] = response.json()
                else:
                    self.logger.warning(f"Failed to get current turn info: HTTP {response.status_code}")
                
                # 4. Get actions taken in the current turn
                current_turn = context.get("current_turn", {}).get("current_turn_number", 1)
                self.logger.info(f"Gathering actions for current turn {current_turn}")
                
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/turns/{current_turn}"
                )
                
                if response.status_code == 200:
                    context["current_turn_actions"] = response.json()
                elif response.status_code == 404:
                    # Turn not found in database yet - this is normal for new games/turns
                    self.logger.info(f"Turn {current_turn} not yet recorded in database (this is normal for new turns)")
                    context["current_turn_actions"] = {"actions": []}
                else:
                    self.logger.warning(f"Failed to get current turn actions: HTTP {response.status_code}")
                    context["current_turn_actions"] = {"actions": []}
                
                # 5. Get recent game history (last 5 turns)
                self.logger.info("Gathering recent game history")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/actions",
                    params={"limit": int(os.getenv('AGENT_ACTION_LIMIT', '75'))}  # Get enough actions to cover ~7-8 turns
                )
                
                if response.status_code == 200:
                    data = response.json()
                    context["recent_actions"] = data.get("actions", [])
                else:
                    self.logger.error(f"Failed to get recent actions: HTTP {response.status_code}")
                
                # 6. Get player's current strategies
                self.logger.info("Gathering player strategies")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/player/{self.player_id}/strategies"
                )
                
                if response.status_code == 200:
                    context["strategies"] = response.json()
                else:
                    self.logger.warning(f"Failed to get player strategies: HTTP {response.status_code}")
                
                # 7. Get recent messages
                self.logger.info("Gathering recent messages")
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/messages",
                    params={"player_id": self.player_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    context["messages"] = data.get("messages", [])
                else:
                    self.logger.error(f"Failed to get messages: HTTP {response.status_code}")
            
            return context
                
        except Exception as e:
            self.logger.exception(f"Error gathering turn context: {str(e)}")
            return {}
    
    def _calculate_messaging_thresholds(self) -> tuple:
        """
        Calculate personalized messaging thresholds for this agent to prevent synchronized messaging.
        Uses agent name to generate consistent but varied thresholds.
        
        Returns:
            tuple: (mandatory_threshold, urgent_threshold, encourage_threshold)
        """
        # Use agent name hash to generate consistent random values for this agent
        name_hash = hash(self.name)
        
        # Mandatory messaging: 8-12 turns since last message
        mandatory_threshold = 8 + (name_hash % 5)
        
        # Urgent messaging: 6-10 turns since last message  
        urgent_threshold = 6 + ((name_hash >> 8) % 5)
        
        # Encouraged messaging: 3-6 turns since last message
        encourage_threshold = 3 + ((name_hash >> 16) % 4)
        
        return mandatory_threshold, urgent_threshold, encourage_threshold
    
    def _get_player_name_lookup(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a lookup map from player ID to player name using context data.
        
        Args:
            context: The full context dictionary from gather_turn_context()
            
        Returns:
            dict: Mapping from player_id to player_name
        """
        player_id_to_name = {}
        
        # Try to get players list from multiple possible locations in context
        game_status = context.get("game_status", {})
        board_state = context.get("board_state", {})
        
        players_list = None
        if game_status.get("players"):
            players_list = game_status["players"]
        elif board_state.get("players"):
            players_list = board_state["players"]
        
        if players_list:
            for player in players_list:
                if isinstance(player, dict):
                    # Handle multiple possible field names for player ID and name
                    pid = player.get("player_id") or player.get("id")
                    pname = player.get("name") or player.get("player_name")
                    if pid and pname:
                        player_id_to_name[pid] = pname
        
        return player_id_to_name

    def _analyze_messaging_history(self, recent_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze recent actions to determine messaging patterns and urgency.
        
        Args:
            recent_actions: List of recent game actions
            
        Returns:
            dict: Messaging analysis with turns_since_last_message and urgency level
        """
        # Find the most recent message sent by this agent
        turns_since_last_message = 0
        last_message_turn = None
        
        # Look through recent actions for send_message actions by this agent
        for action in reversed(recent_actions):  # Start from most recent
            if (action.get("action_type") == "send_message" and 
                action.get("player_id") == self.player_id):
                last_message_turn = action.get("turn_number", 0)
                break
        
        # Calculate turns since last message
        if last_message_turn is not None:
            current_turn = max([action.get("turn_number", 0) for action in recent_actions] + [1])
            turns_since_last_message = current_turn - last_message_turn
        else:
            # No message found in recent history - assume it's been a long time
            turns_since_last_message = 15
        
        # Get personalized thresholds
        mandatory_threshold, urgent_threshold, encourage_threshold = self._calculate_messaging_thresholds()
        
        # Determine urgency level
        if turns_since_last_message >= mandatory_threshold:
            urgency = "MANDATORY"
        elif turns_since_last_message >= urgent_threshold:
            urgency = "URGENT"
        elif turns_since_last_message >= encourage_threshold:
            urgency = "ENCOURAGED"
        else:
            urgency = "NORMAL"
        
        return {
            "turns_since_last_message": turns_since_last_message,
            "last_message_turn": last_message_turn,
            "urgency": urgency,
            "mandatory_threshold": mandatory_threshold,
            "urgent_threshold": urgent_threshold,
            "encourage_threshold": encourage_threshold
        }

    def _generate_phase_guidance(self, current_phase: str, army_count: int, player_info: Dict[str, Any], game_status: Dict[str, Any]) -> str:
        """
        Generate phase-specific guidance for the agent.
        
        Args:
            current_phase: Current game phase
            army_count: Player's available army count
            player_info: Player information
            game_status: Game status information
            
        Returns:
            str: Formatted phase guidance
        """
        guidance = ["## 🎯 CURRENT PHASE STATUS & ACTIONS"]
        
        if current_phase == "setup":
            guidance.extend([
                "**📋 PHASE: SETUP**",
                "**Purpose**: Place your initial armies to establish starting positions",
                "**Phase Rules**: Each player places armies on their assigned territories",
                f"**Your Status**: {army_count} armies remaining to place",
                "",
                "**🎯 REQUIRED ACTIONS:**",
                "- ✅ **place_armies**: Place armies on your territories strategically",
                "- Focus on border territories and potential attack positions",
                "- Consider defensive positions and continent control",
                "",
                "**🚫 FORBIDDEN ACTIONS:**",
                "- attack_territory (not available in setup)",
                "- fortify_position (not available in setup)", 
                "- end_turn (setup advances automatically)",
                "",
                "**📈 STRATEGY TIPS:**",
                "- Strengthen territories that border enemies",
                "- Build up territories that can launch attacks",
                "- Consider continent control opportunities",
                "",
                "**➡️ NEXT PHASE**: Reinforcement (when all players finish setup)"
            ])
            
        elif current_phase == "reinforcement":
            cards = player_info.get('cards', [])
            guidance.extend([
                "**🛡️ PHASE: REINFORCEMENT**",
                "**Purpose**: Strengthen your position with new armies and card trades",
                "**Phase Rules**: Must place ALL armies before advancing to attack phase",
                f"**Your Status**: {army_count} armies to place, {len(cards)} cards in hand",
                "",
                "**🎯 REQUIRED ACTIONS:**",
                "- ✅ **place_armies**: MUST place all armies before ending turn",
                f"- You have {army_count} armies that MUST be placed",
                "- Choose territories strategically (vulnerable borders, attack positions)",
                "",
                "**🔄 OPTIONAL ACTIONS:**",
                "- 🎴 **trade_cards**: Trade 3+ cards for army bonus (do this FIRST!)",
                "- 💬 **send_message**: Diplomatic communication",
                "",
                "**🚫 FORBIDDEN ACTIONS:**",
                "- attack_territory (only available in attack phase)",
                "- fortify_position (only available in fortify phase)",
                f"- end_turn (WILL FAIL if army_count > 0)",
                "",
                "**⚠️ CRITICAL WARNING:**",
                f"- You CANNOT end your turn until army_count = 0",
                f"- Currently you have {army_count} unplaced armies",
                "- Calling end_turn with unplaced armies causes infinite loops!",
                "",
                "**➡️ NEXT PHASE**: Attack (after all armies placed)"
            ])
            
        elif current_phase == "attack":
            guidance.extend([
                "**⚔️ PHASE: ATTACK**",
                "**Purpose**: Conquer enemy territories to expand your empire",
                "**Phase Rules**: Attack adjacent enemy territories, earn cards for conquests",
                f"**Your Status**: {army_count} armies available, ready for combat",
                "",
                "**🎯 AVAILABLE ACTIONS:**",
                "- ⚔️ **attack_territory**: Attack adjacent enemy territories",
                "- 💬 **send_message**: Diplomatic communication",
                "- ⏭️ **end_turn**: Advance to fortify phase (or next player)",
                "",
                "**🔄 OPTIONAL ACTIONS:**",
                "- 🎴 **trade_cards**: Trade cards if you have 3+",
                "- 🛡️ **place_armies**: Place any remaining reinforcement armies",
                "",
                "**🚫 FORBIDDEN ACTIONS:**",
                "- fortify_position (only available in fortify phase)",
                "",
                "**📈 STRATEGY TIPS:**",
                "- Target weak territories (1-2 armies) for easy conquests",
                "- Earn cards by conquering at least one territory",
                "- Consider diplomatic alliances before attacking",
                "- Attack from territories with 3+ armies for better odds",
                "",
                "**➡️ NEXT PHASE**: Fortify (when you end turn)"
            ])
            
        elif current_phase == "fortify":
            guidance.extend([
                "**🏰 PHASE: FORTIFY**",
                "**Purpose**: Reposition armies for better strategic positioning",
                "**Phase Rules**: Move armies between your territories (optional)",
                f"**Your Status**: {army_count} armies available for repositioning",
                "",
                "**🎯 AVAILABLE ACTIONS:**",
                "- 🏰 **fortify_position**: Move armies between your territories",
                "- 💬 **send_message**: Diplomatic communication",
                "- ⏭️ **end_turn**: Complete your turn (advance to next player)",
                "",
                "**🔄 OPTIONAL ACTIONS:**",
                "- 🎴 **trade_cards**: Trade cards if you have 3+",
                "- 🛡️ **place_armies**: Place any remaining armies",
                "",
                "**🚫 FORBIDDEN ACTIONS:**",
                "- attack_territory (only available in attack phase)",
                "",
                "**📈 STRATEGY TIPS:**",
                "- Move armies from safe interior territories to borders",
                "- Strengthen territories that can launch future attacks",
                "- Consolidate forces for defensive positions",
                "- Fortification is optional - you can skip and end turn",
                "",
                "**➡️ NEXT PHASE**: Next player's reinforcement phase"
            ])
            
        elif current_phase == "game_over":
            guidance.extend([
                "**🏆 PHASE: GAME OVER**",
                "**Purpose**: Game has ended - someone achieved world domination!",
                "**Status**: No actions available - game is complete",
                "",
                "**🚫 ALL ACTIONS FORBIDDEN:**",
                "- Game has ended, no further actions possible",
                "- Check game results to see who won",
                "",
                "**🎉 GAME COMPLETE!**"
            ])
            
        else:
            # Unknown phase - provide generic guidance
            guidance.extend([
                f"**❓ PHASE: {current_phase.upper()}**",
                "**Status**: Unknown phase - this may indicate a game state issue",
                f"**Your Status**: {army_count} armies available",
                "",
                "**🎯 GENERAL ACTIONS:**",
                "- Try standard actions: place_armies, attack_territory, fortify_position",
                "- Use send_message for diplomatic communication",
                "- Use end_turn to advance the game",
                "",
                "**⚠️ WARNING:**",
                "- Unknown phase detected - game may be in unusual state",
                "- Proceed with caution and standard Risk actions"
            ])
        
        return "\n".join(guidance)

    async def format_turn_context(self, context: Dict[str, Any]) -> str:
        """
        Format the gathered context into a comprehensive, strategic prompt for the agent.
        Uses intelligent summarization when context becomes too large.
        
        Args:
            context: The context dictionary from gather_turn_context()
            
        Returns:
            str: Formatted context as a string for the agent prompt
        """
        # First, generate the full context
        sections = []
        
        # 1. Game Rules & Victory Conditions
        game_rules = [
            "## RISK GAME RULES & VICTORY",
            "**OBJECTIVE**: Conquer the world by eliminating all other players",
            "**VICTORY CONDITION**: Control ALL territories on the board",
            "",
            "**Core Mechanics**:",
            "- Each turn: Reinforcement -> Attack -> Fortify phases",
            "- Reinforcements: Get armies based on territories owned (minimum 3, or territories÷3)",
            "- Continent Bonuses: Control entire continents for extra armies each turn",
            "- Attacking: Roll dice, higher rolls win, attacker needs 2+ armies to attack",
            "- Cards: Earn cards by conquering territories, trade sets for army bonuses",
            "",
            "**Key Strategy Principles**:",
            "- Control continents for steady army income",
            "- Eliminate weak players to gain their cards",
            "- Form temporary alliances but be ready to break them",
            "- Fortify borders and chokepoints"
        ]
        sections.append("\n".join(game_rules))
        
        # 2. PHASE-SPECIFIC STATUS & GUIDANCE (NEW - PROMINENT SECTION)
        game_status = context.get("game_status", {})
        player_info = context.get("player_info", {})
        current_phase = game_status.get('phase', 'unknown')
        army_count = player_info.get('army_count', 0)
        
        phase_guidance = self._generate_phase_guidance(current_phase, army_count, player_info, game_status)
        sections.append(phase_guidance)
        
        # 3. Current Situation (Updated)
        current_turn_actions = context.get("current_turn_actions", {}).get("actions", [])
        
        current_situation = [
            "## CURRENT TURN STATUS",
            f"- Turn Number: {game_status.get('turn_number', 'Unknown')}",
            f"- Current Player: {game_status.get('current_player', 'Unknown')}"
        ]
        
        # Add actions taken this turn
        if current_turn_actions:
            current_situation.append("\nActions already taken this turn:")
            for action in current_turn_actions:
                action_type = action.get("action_type", "Unknown")
                action_data = json.dumps(action.get("action_data", {}))
                current_situation.append(f"- {action_type}: {action_data}")
        else:
            current_situation.append("\nNo actions taken yet this turn.")
            
        sections.append("\n".join(current_situation))
        
        # 4. Your Status (Enhanced)
        territories = player_info.get('territories', [])
        cards = player_info.get('cards', [])
        
        # Debug logging for cards data format
        self.logger.debug(f"Cards data type: {type(cards)}, value: {cards}")
        
        player_status = [
            "## YOUR STATUS",
            f"- Army Count: {player_info.get('army_count', 0)} (available to place)",
            f"- Territories Controlled: {len(territories)} (need 42 total to win)",
            f"- Cards in Hand: {len(cards)}"
        ]
        
        # Add detailed card information with defensive type checking
        # Handle case where cards might be a string (JSON string) or other unexpected format
        if isinstance(cards, str):
            try:
                cards = json.loads(cards) if cards else []
                self.logger.debug(f"Parsed cards from JSON string: {cards}")
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse cards JSON: {cards}")
                cards = []

        # Ensure cards is a list
        if not isinstance(cards, list):
            self.logger.warning(f"Cards is not a list, got: {type(cards)} - {cards}")
            cards = []

        if cards:
            card_counts = {}
            for card in cards:
                if isinstance(card, dict):
                    # Dictionary format: {"type": "Cavalry"}
                    card_type = card.get('type', 'Unknown')
                elif isinstance(card, str):
                    # String format: "Cavalry" (current game system format)
                    card_type = card
                else:
                    self.logger.warning(f"Invalid card format: {card} (type: {type(card)})")
                    continue
                    
                card_counts[card_type] = card_counts.get(card_type, 0) + 1
            
            player_status.append("\nCard Details:")
            for card_type, count in card_counts.items():
                player_status.append(f"  - {card_type}: {count}")
            
            # Check for tradeable sets
            if len(cards) >= 3:
                player_status.append("  ⚠️  You can trade cards for army bonus!")
        
        # Add continent bonuses
        continent_bonuses = player_info.get('continent_bonuses', {})
        if continent_bonuses:
            total_bonus = sum(continent_bonuses.values())
            player_status.append(f"\nContinent Bonuses (+{total_bonus} armies/turn):")
            for continent, bonus in continent_bonuses.items():
                player_status.append(f"  - {continent}: +{bonus} armies")
        else:
            player_status.append("\nContinent Bonuses: None (focus on controlling full continents!)")
                
        sections.append("\n".join(player_status))
        
        # 5. Board Analysis (New)
        board_state = context.get("board_state", {})
        board_analysis = self._analyze_board_state(board_state, player_info)
        sections.append(board_analysis)
        
        # 5. Opponent Intelligence (New)
        opponent_analysis = self._analyze_opponents(board_state, player_info)
        sections.append(opponent_analysis)
        
        # 6. Strategic Recommendations (New)
        strategic_recommendations = self._generate_strategic_recommendations(
            board_state, player_info, game_status, cards
        )
        sections.append(strategic_recommendations)
        
        # 7. Your Current Strategies
        strategies = context.get("strategies", {})
        short_term = strategies.get("short_term_strategy", "No strategy set")
        long_term = strategies.get("long_term_strategy", "No strategy set")
        
        # Get timestamps if available
        db_strategies = strategies.get("database_strategies", {})
        short_term_updated = db_strategies.get("short_term", {}).get("updated_at", "Unknown")
        long_term_updated = db_strategies.get("long_term", {}).get("updated_at", "Unknown")
        
        strategies_section = [
            "## YOUR CURRENT STRATEGIES",
            f"- Short-term: {short_term} (updated: {short_term_updated})",
            f"- Long-term: {long_term} (updated: {long_term_updated})"
        ]
        
        sections.append("\n".join(strategies_section))
        
        # 8. Recent History
        recent_actions = context.get("recent_actions", [])
        
        # Group actions by turn
        turns_actions = {}
        for action in recent_actions:
            turn = action.get("turn_number", 0)
            if turn not in turns_actions:
                turns_actions[turn] = []
            turns_actions[turn].append(action)
        
        # Take the 5 most recent turns
        recent_turns = sorted(turns_actions.keys(), reverse=True)[:5]
        
        if recent_turns:
            recent_history = ["## RECENT HISTORY (Last 5 turns)"]
            
            # Create a lookup map from player ID to player name
            player_id_to_name = {}
            
            # Try to get players list from multiple possible locations in context
            players_list = None
            if game_status.get("players"):
                players_list = game_status["players"]
            elif board_state.get("players"):
                players_list = board_state["players"]
            
            if players_list:
                for player in players_list:
                    if isinstance(player, dict):
                        pid = player.get("player_id") or player.get("id")
                        pname = player.get("name") or player.get("player_name")
                        if pid and pname:
                            player_id_to_name[pid] = pname
            
            for turn in recent_turns:
                actions = turns_actions[turn]
                if not actions:
                    continue
                    
                # Get the first action's player info for the turn
                player_id = actions[0].get("player_id", "Unknown")
                
                # Look up player name from our map, with fallbacks
                player_name = "Unknown"
                if player_id in player_id_to_name:
                    player_name = player_id_to_name[player_id]
                elif turn == game_status.get("turn_number"):
                    # Current turn fallback
                    player_name = game_status.get("current_player", "Unknown")
                elif player_id and player_id != "Unknown":
                    # Use player_id as name if we can't find the actual name
                    player_name = f"Player_{player_id}"
                
                recent_history.append(f"\nTurn {turn} - Player {player_name} ({player_id}):")
                
                # Summarize actions
                for action in actions:
                    action_type = action.get("action_type", "Unknown")
                    action_data = json.dumps(action.get("action_data", {}))
                    recent_history.append(f"  - {action_type}: {action_data}")
            
            sections.append("\n".join(recent_history))
        
        # 9. Recent Failed Actions & Loop Prevention (NEW)
        failure_warnings = self._generate_failure_warnings(game_status, player_info)
        if failure_warnings:
            sections.append(failure_warnings)
        
        # 10. Messaging Analysis & Diplomatic Status
        recent_actions = context.get("recent_actions", [])
        messaging_analysis = self._analyze_messaging_history(recent_actions)
        
        messaging_section = ["## DIPLOMATIC STATUS & MESSAGING"]
        messaging_section.append(f"- Turns since last message: {messaging_analysis['turns_since_last_message']}")
        messaging_section.append(f"- Messaging urgency: {messaging_analysis['urgency']}")
        
        # Add urgency-specific messaging guidance
        if messaging_analysis['urgency'] == "MANDATORY":
            messaging_section.append(f"- 🚨 **MANDATORY MESSAGING**: You MUST send a message this turn!")
            messaging_section.append(f"- You haven't messaged in {messaging_analysis['turns_since_last_message']} turns (threshold: {messaging_analysis['mandatory_threshold']})")
            messaging_section.append("- Think strategically about diplomacy and send a meaningful message")
        elif messaging_analysis['urgency'] == "URGENT":
            messaging_section.append(f"- ⚠️ **URGENT**: Consider sending a message soon")
            messaging_section.append(f"- Approaching mandatory threshold ({messaging_analysis['urgent_threshold']}/{messaging_analysis['mandatory_threshold']} turns)")
        elif messaging_analysis['urgency'] == "ENCOURAGED":
            messaging_section.append(f"- 💭 **ENCOURAGED**: Good time for diplomatic communication")
            messaging_section.append("- Consider your strategic position and diplomatic opportunities")
        else:
            messaging_section.append("- ✅ Recent diplomatic activity - messaging optional this turn")
        
        sections.append("\n".join(messaging_section))
        
        # 10. Diplomatic Messages
        messages = context.get("messages", [])
        if messages:
            recent_messages = ["## RECENT DIPLOMATIC MESSAGES"]
            
            for msg in messages[:10]:  # Limit to 10 most recent
                from_name = msg.get("from_player_name", "Unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                recent_messages.append(f"\nFrom {from_name} at {timestamp}:")
                recent_messages.append(f"  {content}")
                
            sections.append("\n".join(recent_messages))
        
        # 11. Action Priority Guide (Updated with Dynamic Messaging Priority)
        decision_workflow = ["## ACTION PRIORITY GUIDE"]
        
        # Dynamic priority based on messaging urgency
        if messaging_analysis['urgency'] == "MANDATORY":
            decision_workflow.extend([
                "🚨 **MANDATORY MESSAGING PRIORITY** 🚨",
                "You MUST send a diplomatic message this turn before other actions!",
                "",
                "**PRIORITY ORDER:**",
                "1. 💬 **Send Message** (MANDATORY) - Strategic diplomatic communication",
                "2. 🎴 **Trade Cards** (if you have 3+ cards) - Get army bonus",
                "3. 🛡️ **Place Armies** (if army_count > 0) - Reinforce territories",
                "4. ⚔️ **Attack** (if you have strong positions) - Target weak territories",
                "5. 🏰 **Fortify** (if needed) - Move armies strategically",
                "6. ⏹️ **End Turn** (if no other actions needed) - Complete your turn"
            ])
        else:
            decision_workflow.extend([
                "Use this guide to prioritize your actions:",
                "",
                "**PRIORITY ORDER:**",
                "1. 🎴 **Trade Cards** (if you have 3+ cards) - Get army bonus immediately",
                "2. 🛡️ **Place Armies** (if army_count > 0) - Reinforce vulnerable territories first"
            ])
            
            # Add messaging priority based on urgency
            if messaging_analysis['urgency'] == "URGENT":
                decision_workflow.append("3. 💬 **Send Message** (URGENT) - Important diplomatic communication")
                decision_workflow.append("4. ⚔️ **Attack** (if you have strong positions) - Target weak adjacent territories")
                decision_workflow.append("5. 🏰 **Fortify** (if needed) - Move armies to strategic positions")
                decision_workflow.append("6. ⏹️ **End Turn** (if no other actions needed) - Complete your turn")
            elif messaging_analysis['urgency'] == "ENCOURAGED":
                decision_workflow.append("3. ⚔️ **Attack** (if you have strong positions) - Target weak adjacent territories")
                decision_workflow.append("4. 💬 **Send Message** (ENCOURAGED) - Consider diplomatic opportunities")
                decision_workflow.append("5. 🏰 **Fortify** (if needed) - Move armies to strategic positions")
                decision_workflow.append("6. ⏹️ **End Turn** (if no other actions needed) - Complete your turn")
            else:
                decision_workflow.append("3. ⚔️ **Attack** (if you have strong positions) - Target weak adjacent territories")
                decision_workflow.append("4. 🏰 **Fortify** (if needed) - Move armies to strategic positions")
                decision_workflow.append("5. 💬 **Send Message** (optional) - Diplomatic communication")
                decision_workflow.append("6. ⏹️ **End Turn** (if no other actions needed) - Complete your turn")
        
        decision_workflow.extend([
            "",
            "**STRATEGY UPDATES:**",
            "- Only update strategies if they say 'No strategy set' or are clearly outdated",
            "- Don't spend excessive time on strategy updates - focus on action",
            "- Your current strategies are shown above for reference",
            "",
            "**DIPLOMATIC STRATEGY:**",
            "- Think creatively about diplomatic opportunities - WHO, WHEN, WHY, and HOW",
            "- Consider innovative negotiation approaches beyond simple non-aggression pacts",
            "- Explore strategic partnerships, information trading, coordinated attacks, temporary truces",
            "- Use timing and leverage to your advantage - negotiate from positions of strength",
            "- Think outside the box: betrayals, double-crosses, and shifting alliances are all valid",
            "- Diplomacy can be as powerful as armies - use it to manipulate the game state",
            "",
            "**REMEMBER:**",
            "- Every turn MUST include at least one concrete game action",
            "- Don't get stuck in analysis paralysis - make decisions quickly",
            "- Focus on actions that advance your position toward world domination",
            "- Strategy updates are optional, game actions are mandatory"
        ])
        
        sections.append("\n".join(decision_workflow))
        
        # 12. Phase Reminder (NEW - Bottom reinforcement)
        phase_reminder = self._generate_phase_reminder(current_phase, army_count, player_info, game_status)
        sections.append(phase_reminder)
        
        # 13. Character Reminder (FINAL - Keep agent in character)
        character_reminder = self._create_character_reminder()
        sections.append(character_reminder)
        
        # Generate the full context
        full_context = "\n\n".join(sections)
        
        # Check if we should use context summarization
        if context_summarizer and context_summarizer.should_summarize(full_context):
            try:
                self.logger.info("Context size exceeds threshold, applying intelligent summarization")
                
                # Get current turn number for summarization
                current_turn = game_status.get('turn_number', 1)
                
                # Use the context summarizer to create an optimized version
                optimized_context = await context_summarizer.summarize_context(
                    self.game_id,
                    full_context,
                    context,
                    current_turn
                )
                
                self.logger.info("Context summarization completed successfully")
                return optimized_context
                
            except Exception as e:
                self.logger.warning(f"Context summarization failed: {e}, using original context")
                return full_context
        else:
            # Context is within limits or summarization not available
            if context_summarizer:
                self.logger.debug("Context size within limits, no summarization needed")
            else:
                self.logger.debug("Context summarizer not available")
            return full_context
    
    def _analyze_board_state(self, board_state: Dict[str, Any], player_info: Dict[str, Any]) -> str:
        """
        Analyze the current board state to provide tactical insights.
        
        Args:
            board_state: Board state data from API
            player_info: Current player's information
            
        Returns:
            str: Formatted board analysis
        """
        analysis = ["## BOARD ANALYSIS"]
        
        territories = board_state.get("territories", {})
        my_territories = set(player_info.get("territories", []))
        
        if not territories:
            analysis.append("Board state unavailable")
            return "\n".join(analysis)
        
        # Analyze your territories
        vulnerable_territories = []
        strong_territories = []
        border_territories = []
        
        for territory_name in my_territories:
            territory = territories.get(territory_name, {})
            army_count = territory.get("army_count", 0)
            adjacent = territory.get("adjacent_territories", [])
            
            # Check if it borders enemy territories
            borders_enemy = any(
                territories.get(adj_name, {}).get("owner") != player_info.get("player_id")
                for adj_name in adjacent
            )
            
            if army_count <= 1:
                vulnerable_territories.append(f"{territory_name} ({army_count} armies)")
            elif army_count >= 5:
                strong_territories.append(f"{territory_name} ({army_count} armies)")
            
            if borders_enemy:
                border_territories.append(f"{territory_name} ({army_count} armies)")
        
        # Your territory analysis
        analysis.append(f"\n**Your Territory Status:**")
        analysis.append(f"- Total Territories: {len(my_territories)}")
        
        if vulnerable_territories:
            analysis.append(f"- Vulnerable (≤1 army): {', '.join(vulnerable_territories[:5])}")
        
        if strong_territories:
            analysis.append(f"- Strong Points (≥5 armies): {', '.join(strong_territories[:5])}")
        
        if border_territories:
            analysis.append(f"- Border Territories: {', '.join(border_territories[:5])}")
        
        # Attack opportunities
        attack_opportunities = []
        for territory_name in my_territories:
            territory = territories.get(territory_name, {})
            army_count = territory.get("army_count", 0)
            adjacent = territory.get("adjacent_territories", [])
            
            if army_count > 1:  # Can attack
                for adj_name in adjacent:
                    adj_territory = territories.get(adj_name, {})
                    adj_owner = adj_territory.get("owner")
                    adj_armies = adj_territory.get("army_count", 0)
                    
                    if adj_owner != player_info.get("player_id") and adj_armies <= 2:
                        attack_opportunities.append(
                            f"{territory_name}({army_count}) -> {adj_name}({adj_armies})"
                        )
        
        if attack_opportunities:
            analysis.append(f"\n**Attack Opportunities:**")
            for opp in attack_opportunities[:5]:  # Limit to top 5
                analysis.append(f"- {opp}")
        else:
            analysis.append(f"\n**Attack Opportunities:** None immediately available")
        
        return "\n".join(analysis)
    
    def _analyze_opponents(self, board_state: Dict[str, Any], player_info: Dict[str, Any]) -> str:
        """
        Analyze opponent positions and threats.
        
        Args:
            board_state: Board state data from API
            player_info: Current player's information
            
        Returns:
            str: Formatted opponent analysis
        """
        analysis = ["## OPPONENT INTELLIGENCE"]
        
        territories = board_state.get("territories", {})
        players = board_state.get("players", [])
        my_player_id = player_info.get("player_id")
        
        if not territories or not players:
            analysis.append("Opponent data unavailable")
            return "\n".join(analysis)
        
        # Analyze each opponent
        opponent_stats = []
        for player in players:
            if player.get("player_id") == my_player_id:
                continue
            
            player_territories = []
            total_armies = 0
            
            # Count territories and armies for this player
            for territory_name, territory in territories.items():
                if territory.get("owner") == player.get("player_id"):
                    player_territories.append(territory_name)
                    total_armies += territory.get("army_count", 0)
            
            opponent_stats.append({
                "name": player.get("name", "Unknown"),
                "player_id": player.get("player_id"),
                "territories": len(player_territories),
                "estimated_armies": total_armies,
                "territory_list": player_territories
            })
        
        # Sort by threat level (territories + armies)
        opponent_stats.sort(key=lambda x: x["territories"] + x["estimated_armies"], reverse=True)
        
        analysis.append("\n**Player Rankings (by threat level):**")
        for i, opponent in enumerate(opponent_stats, 1):
            threat_level = "🔴 HIGH" if opponent["territories"] > 15 else "🟡 MEDIUM" if opponent["territories"] > 8 else "🟢 LOW"
            analysis.append(
                f"{i}. {opponent['name']}: {opponent['territories']} territories, "
                f"~{opponent['estimated_armies']} armies [{threat_level}]"
            )
        
        # Identify immediate threats (opponents bordering your territories)
        immediate_threats = []
        my_territories = set(player_info.get("territories", []))
        
        for territory_name in my_territories:
            territory = territories.get(territory_name, {})
            adjacent = territory.get("adjacent_territories", [])
            
            for adj_name in adjacent:
                adj_territory = territories.get(adj_name, {})
                adj_owner = adj_territory.get("owner")
                adj_armies = adj_territory.get("army_count", 0)
                
                if adj_owner and adj_owner != my_player_id:
                    # Find opponent name
                    opponent_name = "Unknown"
                    for opponent in opponent_stats:
                        if opponent["player_id"] == adj_owner:
                            opponent_name = opponent["name"]
                            break
                    
                    immediate_threats.append(
                        f"{adj_name}({adj_armies}) owned by {opponent_name}"
                    )
        
        if immediate_threats:
            analysis.append(f"\n**Immediate Border Threats:**")
            threat_limit = int(os.getenv('AGENT_THREAT_ANALYSIS_LIMIT', '8'))
            for threat in list(set(immediate_threats))[:threat_limit]:  # Remove duplicates, configurable limit
                analysis.append(f"- {threat}")
        
        # Provide strategic diplomatic intelligence (let agent decide who to negotiate with)
        if len(opponent_stats) > 1:
            analysis.append(f"\n**Diplomatic Intelligence:**")
            analysis.append("- Analyze each player's position and consider strategic negotiations")
            analysis.append("- Think about timing, leverage, and mutual benefits in any diplomatic approach")
            analysis.append("- Consider various negotiation types: alliances, truces, information sharing, coordinated attacks")
        
        return "\n".join(analysis)
    
    def _generate_strategic_recommendations(
        self, 
        board_state: Dict[str, Any], 
        player_info: Dict[str, Any], 
        game_status: Dict[str, Any],
        cards: List[Dict[str, Any]]
    ) -> str:
        """
        Generate strategic recommendations based on current game state.
        
        Args:
            board_state: Board state data
            player_info: Current player's information
            game_status: Current game status
            cards: Player's cards
            
        Returns:
            str: Formatted strategic recommendations
        """
        recommendations = ["## STRATEGIC RECOMMENDATIONS"]
        
        territories = board_state.get("territories", {})
        my_territories = set(player_info.get("territories", []))
        army_count = player_info.get("army_count", 0)
        game_phase = game_status.get("phase", "unknown")
        
        # Card trading recommendation
        if len(cards) >= 3:
            recommendations.append("\n**🎴 CARD TRADING:**")
            recommendations.append("- ⚡ PRIORITY: Trade your cards for army bonus immediately!")
            recommendations.append("- This will give you significant army advantage")
        
        # Reinforcement recommendations
        if army_count > 0 and game_phase in ["setup", "reinforcement"]:
            recommendations.append(f"\n**🛡️ REINFORCEMENT STRATEGY ({army_count} armies to place):**")
            
            # Find vulnerable territories that need reinforcement
            vulnerable = []
            strategic = []
            
            for territory_name in my_territories:
                territory = territories.get(territory_name, {})
                army_count_territory = territory.get("army_count", 0)
                adjacent = territory.get("adjacent_territories", [])
                
                # Check if borders enemies
                borders_enemy = any(
                    territories.get(adj_name, {}).get("owner") != player_info.get("player_id")
                    for adj_name in adjacent
                )
                
                if army_count_territory <= 1 and borders_enemy:
                    vulnerable.append(territory_name)
                elif borders_enemy and army_count_territory < 5:
                    strategic.append(territory_name)
            
            if vulnerable:
                recommendations.append(f"- 🚨 URGENT: Reinforce vulnerable borders: {', '.join(vulnerable[:3])}")
            if strategic:
                recommendations.append(f"- 📈 BUILD UP: Strengthen attack positions: {', '.join(strategic[:3])}")
        
        # Attack recommendations
        if game_phase == "attack":
            recommendations.append(f"\n**⚔️ ATTACK STRATEGY:**")
            
            # Find good attack targets
            good_targets = []
            for territory_name in my_territories:
                territory = territories.get(territory_name, {})
                army_count_territory = territory.get("army_count", 0)
                adjacent = territory.get("adjacent_territories", [])
                
                if army_count_territory > 1:  # Can attack
                    for adj_name in adjacent:
                        adj_territory = territories.get(adj_name, {})
                        adj_owner = adj_territory.get("owner")
                        adj_armies = adj_territory.get("army_count", 0)
                        
                        if adj_owner != player_info.get("player_id"):
                            success_chance = "HIGH" if army_count_territory > adj_armies * 2 else "MEDIUM" if army_count_territory > adj_armies else "LOW"
                            good_targets.append(f"{adj_name}({adj_armies}) - {success_chance} chance")
            
            if good_targets:
                recommendations.append("- 🎯 PRIORITY TARGETS:")
                for target in good_targets[:3]:
                    recommendations.append(f"  • {target}")
            else:
                recommendations.append("- 🛡️ DEFENSIVE TURN: Focus on reinforcement and fortification")
        
        # Fortification recommendations
        if game_phase == "fortify":
            recommendations.append(f"\n**🏰 FORTIFICATION STRATEGY:**")
            recommendations.append("- Move armies from safe interior territories to borders")
            recommendations.append("- Strengthen territories that can launch future attacks")
            recommendations.append("- Consolidate forces for next turn's offensive")
        
        # Continent control analysis
        continent_bonuses = player_info.get('continent_bonuses', {})
        recommendations.append(f"\n**🌍 CONTINENT CONTROL:**")
        
        if continent_bonuses:
            total_bonus = sum(continent_bonuses.values())
            recommendations.append(f"- ✅ CURRENT BONUSES: +{total_bonus} armies/turn from {list(continent_bonuses.keys())}")
            recommendations.append("- 🛡️ DEFEND: Protect your continent borders at all costs!")
        else:
            recommendations.append("- 🎯 FOCUS: Work toward controlling a complete continent")
            recommendations.append("- 💡 TIP: Australia (2 armies) and South America (2 armies) are easiest")
        
        # Victory progress
        total_territories = len(territories)
        my_territory_count = len(my_territories)
        progress = (my_territory_count / total_territories) * 100
        
        recommendations.append(f"\n**🏆 VICTORY PROGRESS:**")
        recommendations.append(f"- World Domination: {my_territory_count}/{total_territories} territories ({progress:.1f}%)")
        
        if progress < 25:
            recommendations.append("- 📊 EARLY GAME: Focus on continent control and steady expansion")
        elif progress < 50:
            recommendations.append("- ⚡ MID GAME: Eliminate weak players, form strategic alliances")
        else:
            recommendations.append("- 🔥 END GAME: All-out assault! Victory is within reach!")
        
        return "\n".join(recommendations)
    
    def _generate_phase_reminder(self, current_phase: str, army_count: int, player_info: Dict[str, Any], game_status: Dict[str, Any]) -> str:
        """
        Generate a concise phase reminder for the bottom of the context.
        
        Args:
            current_phase: Current game phase
            army_count: Player's available army count
            player_info: Player information
            game_status: Game status information
            
        Returns:
            str: Formatted phase reminder
        """
        reminder = ["## ⚠️ FINAL PHASE REMINDER - READ BEFORE ACTING!"]
        
        if current_phase == "setup":
            reminder.extend([
                "",
                "**CURRENT PHASE: SETUP**",
                "- ✅ ALLOWED: place_armies, send_message",
                "- ❌ FORBIDDEN: attack_territory, fortify_position, end_turn",
                f"- 🚨 CRITICAL: You have {army_count} armies that MUST be placed",
                "- ➡️ NEXT: Reinforcement phase (when all players finish setup)",
                "",
                "**REMEMBER: Place armies strategically on border territories!**"
            ])
            
        elif current_phase == "reinforcement":
            cards = player_info.get('cards', [])
            reminder.extend([
                "",
                "**CURRENT PHASE: REINFORCEMENT**",
                "- ✅ ALLOWED: place_armies, trade_cards, send_message",
                "- ❌ FORBIDDEN: attack_territory, fortify_position, end_turn (until army_count = 0)",
                f"- 🚨 CRITICAL: You have {army_count} armies that MUST be placed before ending turn",
                f"- 🎴 CARDS: {len(cards)} cards in hand - trade if you have 3+",
                "- ➡️ NEXT: Attack phase (after all armies placed)",
                "",
                "**REMEMBER: Check your army_count before every action!**"
            ])
            
        elif current_phase == "attack":
            reminder.extend([
                "",
                "**CURRENT PHASE: ATTACK**",
                "- ✅ ALLOWED: attack_territory, trade_cards, place_armies, send_message, end_turn",
                "- ❌ FORBIDDEN: fortify_position",
                f"- 💪 STATUS: {army_count} armies available for combat",
                "- 🎯 GOAL: Conquer territories to earn cards and expand empire",
                "- ➡️ NEXT: Fortify phase (when you end turn)",
                "",
                "**REMEMBER: Attack from territories with 2+ armies!**"
            ])
            
        elif current_phase == "fortify":
            reminder.extend([
                "",
                "**CURRENT PHASE: FORTIFY**",
                "- ✅ ALLOWED: fortify_position, trade_cards, place_armies, send_message, end_turn",
                "- ❌ FORBIDDEN: attack_territory",
                f"- 🏰 STATUS: {army_count} armies available for repositioning",
                "- 🎯 GOAL: Move armies to strategic positions (optional)",
                "- ➡️ NEXT: Next player's reinforcement phase",
                "",
                "**REMEMBER: Fortification is optional - you can skip and end turn!**"
            ])
            
        elif current_phase == "game_over":
            reminder.extend([
                "",
                "**CURRENT PHASE: GAME OVER**",
                "- 🚫 ALL ACTIONS FORBIDDEN: Game has ended",
                "- 🏆 STATUS: Someone achieved world domination",
                "- 🎉 GAME COMPLETE!",
                "",
                "**REMEMBER: No further actions possible!**"
            ])
            
        else:
            # Unknown phase
            reminder.extend([
                "",
                f"**CURRENT PHASE: {current_phase.upper()}**",
                "- ⚠️ WARNING: Unknown phase detected",
                f"- 💪 STATUS: {army_count} armies available",
                "- 🎯 TRY: Standard Risk actions (place_armies, attack_territory, etc.)",
                "",
                "**REMEMBER: Proceed with caution!**"
            ])
        
        return "\n".join(reminder)
    
    def _create_character_reminder(self) -> str:
        """Create final character reminder section."""
        reminder = [
            "## 🎭 CHARACTER REMINDER - REMEMBER WHO YOU ARE!",
            "",
            f"**You are {self.name}** - stay in character throughout this turn.",
            "",
            "**Your Playing Style & Character Instructions:**",
            f"{self.custom_instructions}",
            "",
            "**CHARACTER CONSISTENCY REMINDERS:**",
            "- Make decisions that align with your established playing style",
            "- Your personality should influence HOW you play, not just WHAT you play",
            "- Stay consistent with your character's strategic preferences and behavioral patterns",
            "- When communicating with other players, maintain your character's voice and approach",
            "- If you send any messages, make sure you stay true to your character, language and style",
            f"**Now make your move as {self.name}, keeping this character guidance in mind!**"
        ]
        return "\n".join(reminder)
    
    def _generate_failure_warnings(self, game_status: Dict[str, Any], player_info: Dict[str, Any]) -> str:
        """
        Generate warnings about recent failed actions to prevent loops.
        
        Args:
            game_status: Current game status
            player_info: Player information
            
        Returns:
            str: Formatted failure warnings or empty string if none
        """
        if not self.game_id or not self.player_id:
            return ""
        
        try:
            # Get failure patterns for this player
            failure_patterns = action_tracker.get_player_failure_patterns(self.game_id, self.player_id)
            
            if not failure_patterns:
                return ""
            
            warnings = ["## 🚨 RECENT FAILED ACTIONS - AVOID REPEATING!"]
            warnings.append("")
            warnings.append("**⚠️ CRITICAL: You have recently failed these actions:**")
            
            for pattern in failure_patterns[:3]:  # Show top 3 failure patterns
                action_type = pattern['action_type']
                consecutive_failures = pattern['consecutive_failures']
                last_failure_message = pattern['last_failure_message']
                intervention_triggered = pattern.get('intervention_triggered', False)
                
                if consecutive_failures >= 5:
                    if intervention_triggered:
                        warnings.append(f"- ✅ **{action_type}**: Failed {consecutive_failures}× - INTERVENTION APPLIED")
                        warnings.append(f"  └─ System auto-recovery was triggered for this issue")
                    else:
                        warnings.append(f"- 🔴 **{action_type}**: Failed {consecutive_failures}× - LOOP DETECTED!")
                        warnings.append(f"  └─ Error: {last_failure_message}")
                        warnings.append(f"  └─ 🚨 URGENT: Fix the underlying issue before trying again!")
                elif consecutive_failures >= 3:
                    warnings.append(f"- 🟡 **{action_type}**: Failed {consecutive_failures}× - WARNING")
                    warnings.append(f"  └─ Error: {last_failure_message}")
                    warnings.append(f"  └─ ⚠️ Address the cause before attempting again")
                else:
                    warnings.append(f"- 🟠 **{action_type}**: Failed {consecutive_failures}× recently")
                    warnings.append(f"  └─ Error: {last_failure_message}")
            
            warnings.append("")
            warnings.append("**🎯 LOOP PREVENTION STRATEGY:**")
            
            # Specific guidance for common failure patterns
            for pattern in failure_patterns:
                action_type = pattern['action_type']
                consecutive_failures = pattern['consecutive_failures']
                
                if action_type == "end_turn" and consecutive_failures >= 3:
                    warnings.append("- 🛡️ **end_turn failures**: Check your army_count BEFORE calling end_turn")
                    warnings.append("  └─ If army_count > 0, use place_armies first, then end_turn")
                    warnings.append("  └─ NEVER call end_turn with unplaced armies!")
                
                elif action_type == "place_armies" and consecutive_failures >= 3:
                    warnings.append("- 🏰 **place_armies failures**: Verify territory ownership and army availability")
                    warnings.append("  └─ Check that you own the territory and have armies to place")
                
                elif action_type == "attack_territory" and consecutive_failures >= 3:
                    warnings.append("- ⚔️ **attack_territory failures**: Verify attack requirements")
                    warnings.append("  └─ Need 2+ armies in attacking territory, territories must be adjacent")
            
            warnings.append("")
            warnings.append("**🔧 IMMEDIATE ACTION REQUIRED:**")
            warnings.append("- 📊 **Analyze the errors above** - understand WHY each action failed")
            warnings.append("- 🎯 **Fix the root cause** - don't just retry the same action")
            warnings.append("- ✅ **Verify conditions** - check game state before taking actions")
            warnings.append("- 🚫 **Avoid repetition** - if an action failed 3+ times, try a different approach")
            
            return "\n".join(warnings)
            
        except Exception as e:
            self.logger.error(f"Error generating failure warnings: {e}")
            return ""
    
    async def check_is_my_turn(self) -> bool:
        """
        Check if it's this agent's turn to play using direct API call.
        This is much more efficient than using LLM for a simple state check.
        
        Returns:
            bool: True if it's this agent's turn, False otherwise
        """
        if not self.game_id or not self.player_id:
            self.logger.warning("Can't check turn - not in a game")
            return False
        
        try:
            # Direct API call to check game status with timeout and retry
            timeout = httpx.Timeout(10.0)  # 10 second timeout
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/status",
                    params={"player_id": self.player_id}
                )
                
                if response.status_code != 200:
                    self.logger.error(f"Failed to get game status: HTTP {response.status_code} - {response.text}")
                    return False
                
                data = response.json()
                game_status = data.get("game_status", {})
                
                # Check both current_player_id and current_player (name) for compatibility
                current_player_id = game_status.get("current_player_id")
                current_player_name = game_status.get("current_player")
                
                # Log the current game state for debugging
                turn_number = game_status.get("turn_number", "Unknown")
                phase = game_status.get("phase", "Unknown")
                game_state = game_status.get("state", "Unknown")
                
                self.logger.debug(f"Game state: Turn {turn_number}, Phase {phase}, State {game_state}, Current player ID: {current_player_id}, Current player name: {current_player_name}")
                
                # Check if it's our turn by ID or name
                is_my_turn = (current_player_id == self.player_id) or (current_player_name == self.name)
                
                if is_my_turn:
                    self.logger.info(f"✅ It's {self.name}'s turn to play! (Turn {turn_number}, Phase: {phase})")
                else:
                    self.logger.debug(f"⏳ Not {self.name}'s turn. Current player: {current_player_name or current_player_id}")
                
                return is_my_turn
                
        except httpx.TimeoutException:
            self.logger.warning(f"Timeout checking turn status for {self.name}")
            return False
        except httpx.RequestError as e:
            self.logger.warning(f"Network error checking turn status for {self.name}: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error checking turn status for {self.name}: {str(e)}")
            return False
    
    async def track_decision_start(self, context: Dict[str, Any], formatted_context: str, prompt: str) -> int:
        """
        Start tracking the agent's decision process in the database.
        
        Returns:
            int: Decision ID for later updating with results
        """
        # Make sure we have valid game_id and player_id
        if not self.game_id or not self.player_id:
            self.logger.error("Cannot track decision: missing game_id or player_id")
            return -1
            
        # Get the current turn number
        turn_number = context.get("game_status", {}).get("turn_number", 1)
        
        # Start tracking the decision
        decision_id = action_tracker.track_agent_decision_start(
            self.game_id,
            self.player_id,
            self.name,
            turn_number,
            context,
            formatted_context,
            prompt
        )
        
        self.logger.info(f"Started tracking decision {decision_id} in database")
        return decision_id
    
    async def track_decision_complete(
        self,
        decision_id: int,
        response: Optional[Dict[str, Any]],
        reasoning: Optional[str],
        tools_used: List[str],
        decision_time: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Complete tracking of the agent's decision in the database.
        """
        # Extract actual tools used and reasoning from the response if not provided
        extracted_tools = []
        extracted_reasoning = ""
        
        if response and "messages" in response:
            for message in response["messages"]:
                # Check for tool calls in the message using proper LangChain structure
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        try:
                            # Access tool call properties according to LangChain docs
                            tool_name = tool_call['name'] if isinstance(tool_call, dict) else tool_call.name
                            tool_args = tool_call['args'] if isinstance(tool_call, dict) else tool_call.args
                            tool_id = tool_call.get('id', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'id', 'unknown')
                            
                            extracted_tools.append(tool_name)
                            self.logger.info(f"Tool call detected: {tool_name} with args: {tool_args} (id: {tool_id})")
                        except (KeyError, AttributeError) as e:
                            self.logger.warning(f"Failed to parse tool call: {tool_call}, error: {e}")
                            extracted_tools.append('malformed_tool_call')
                
                # Check for invalid tool calls
                if hasattr(message, 'invalid_tool_calls') and message.invalid_tool_calls:
                    for invalid_call in message.invalid_tool_calls:
                        try:
                            invalid_name = invalid_call.get('name', 'unknown') if isinstance(invalid_call, dict) else getattr(invalid_call, 'name', 'unknown')
                            error_msg = invalid_call.get('error', 'unknown error') if isinstance(invalid_call, dict) else getattr(invalid_call, 'error', 'unknown error')
                            self.logger.warning(f"Invalid tool call detected: {invalid_name}, error: {error_msg}")
                            extracted_tools.append(f'invalid_{invalid_name}')
                        except (AttributeError, TypeError) as e:
                            self.logger.warning(f"Failed to parse invalid tool call: {invalid_call}, error: {e}")
                
                # Extract reasoning from AI messages
                if hasattr(message, 'content') and hasattr(message, 'type') and message.type == 'ai':
                    content = str(message.content) if message.content else ""
                    if content.strip():  # Only add non-empty content
                        extracted_reasoning += content + "\n"
        
        # Use provided parameters if available, otherwise use extracted values
        final_tools_used = tools_used if tools_used else extracted_tools
        final_reasoning = reasoning if reasoning else extracted_reasoning.strip()
        
        # Get the full response content
        response_content = ""
        if response and "messages" in response:
            response_content = "\n".join(
                [str(m.content if hasattr(m, 'content') else m) for m in response["messages"]]
            )
        
        # Log what we're tracking
        self.logger.info(f"Tracking decision {decision_id}: tools_used={final_tools_used}, success={success}")
        if final_reasoning:
            self.logger.debug(f"Decision reasoning: {final_reasoning[:200]}...")
        
        # Complete tracking the decision
        result = action_tracker.track_agent_decision_complete(
            decision_id,
            response_content,
            final_reasoning,
            list(set(final_tools_used)),  # Remove duplicates
            decision_time,
            success,
            error_message
        )
        
        self.logger.info(f"Completed tracking decision {decision_id} in database, success: {success}")
        return result
    
    def _get_strategy_guidance_text(self) -> str:
        """
        Get the strategy guidance text as a separate method to avoid string escaping issues.
        
        Returns:
            str: Strategy guidance text
        """
        return '''
            **MANDATORY STRATEGY UPDATES (You MUST do these):**
            - If ANY strategy says "No strategy set" -> UPDATE IT IMMEDIATELY
            - If ANY strategy is >5 turns old -> UPDATE IT (likely outdated)
            - If your position has changed significantly -> UPDATE BOTH strategies
            
            **STRATEGY EFFECTIVENESS REFLECTION:**
            Look at your recent actions and ask yourself:
            - Are my current strategies actually guiding my decisions?
            - Have my strategies helped me gain territory/armies in recent turns?
            - Do my strategies reflect the current game state and opponents?
            - Are there major threats/opportunities my strategies do not address?
            
            **WHEN TO UPDATE STRATEGIES (High Priority):**
            - Missing strategies (says "No strategy set")
            - Outdated strategies (>5 turns old or irrelevant to current situation)
            - Major game state changes (new continent control, player eliminated, etc.)
            - Strategies that have not been effective (not helping you win)
            - Your position has significantly improved or worsened
            
            **STRATEGY QUALITY CRITERIA:**
            Good strategies should be:
            - **Specific**: Name exact territories, continents, or players
            - **Actionable**: Clear actions you can take in next 1-3 turns
            - **Relevant**: Address current threats and opportunities
            - **Adaptive**: Account for opponent positions and recent events
            
            **STRATEGY UPDATE EXAMPLES:**
            - Short-term: "Reinforce Alaska with 3+ armies, then attack Kamchatka to secure North America"
            - Long-term: "Control North America and Australia, then eliminate Bob who is weakest"
            
            **BAD STRATEGY EXAMPLES (These need updating):**
            - "Play defensively" (too vague)
            - "Attack enemies" (not specific enough)
            - "Control Asia" (if Asia is already controlled by someone else)
            - Old strategies that do not reflect current board state
            
            **REFLECTION QUESTIONS:**
            Before proceeding, honestly assess:
            1. Do my strategies clearly guide what I should do this turn?
            2. Are they specific enough to be actionable?
            3. Do they address current threats and opportunities?
            4. Have they been effective in recent turns?
            
            If ANY answer is "no" -> UPDATE your strategies!
            
            Strategy update tools (use when needed):
            - "update_short_term_strategy" - for next 1-3 turns specific goals
            - "update_long_term_strategy" - for overall game objectives and win condition
            - "update_both_strategies" - when both need updating (common)
            
            **IMPORTANT**: Strategy updates are REQUIRED if missing or outdated, not optional!'''

    async def play_turn(self):
        """Play a turn when it's this agent's turn"""
        if not self.game_id or not self.player_id:
            self.logger.warning("Can't play turn - not properly initialized")
            return False
        
        # CRITICAL: Double-check it's actually our turn before doing expensive operations
        is_my_turn = await self.check_is_my_turn()
        if not is_my_turn:
            self.logger.debug(f"Not {self.name}'s turn - skipping LLM call")
            return False
        
        decision_id = -1
        start_time = time.time()
        
        try:
            # Make sure agent is initialized
            if not self.agent:
                await self.initialize()
            
            # Double check agent is initialized
            if not self.agent:
                self.logger.error("Failed to initialize agent")
                return False
            
            # 1. Gather rich context for decision making
            self.logger.info("Gathering context for turn decision")
            context = await self.gather_turn_context()
            
            if not context:
                self.logger.error("Failed to gather context")
                return False
                
            # 2. Format the context into structured prompt
            formatted_context = await self.format_turn_context(context)
            
            # 3. Create circuit breaker warning if needed
            circuit_breaker_warning = ""
            if self.consecutive_strategy_updates >= self.max_consecutive_strategy_updates:
                circuit_breaker_warning = f"""
            🚨 CIRCUIT BREAKER ACTIVATED 🚨
            You have updated strategies {self.consecutive_strategy_updates} times in a row.
            You are FORBIDDEN from using strategy update tools this turn.
            SKIP Step 1 completely and go directly to Step 2 (TAKE GAME ACTION).
            Focus on concrete game actions only!
            """
            
            # 4. Get strategy guidance text
            strategy_guidance = self._get_strategy_guidance_text() if self.consecutive_strategy_updates < self.max_consecutive_strategy_updates else ""
            
            # 5. Create the full prompt with agent instructions and rich context
            prompt = f"""
            You are {self.name}, an AI agent playing a game of Risk.
            Game ID: {self.game_id}
            Your Player ID: {self.player_id}
            
            ## YOUR PLAYING STYLE & INSTRUCTIONS:
            {self.custom_instructions}
            
            {circuit_breaker_warning}
            
            Below is the current game context with all the information you need to make an informed decision:
            
            {formatted_context}
            
            IMPORTANT: You must complete your turn by taking at least ONE concrete game action. Follow this decision workflow:
            
            ## STEP 1: STRATEGY REVIEW (MANDATORY if strategies missing, otherwise assess and update if needed)
            {"🚫 SKIP THIS STEP - Circuit breaker active" if self.consecutive_strategy_updates >= self.max_consecutive_strategy_updates else "**STRATEGY ASSESSMENT CHECKLIST** - You MUST assess your strategies:"}
            {strategy_guidance}
            
            ## STEP 2: MANDATORY TURN SEQUENCE (FOLLOW EXACTLY)
            **CRITICAL: You MUST follow this exact sequence:**

            **PHASE 1: ARMY PLACEMENT (REQUIRED IF army_count > 0)**
            - Check your army_count in the status above
            - If army_count > 0: You MUST use "place_armies" tool
            - Place armies on vulnerable territories or attack positions  
            - Repeat until army_count = 0
            - NEVER skip this step!

            **PHASE 2: CARD TRADING (RECOMMENDED IF you have 3+ cards)**
            - If you have 3+ cards: Use "trade_cards" tool for army bonus
            - This gives you MORE armies to place (return to Phase 1)

            **PHASE 3: COMBAT & MOVEMENT (OPTIONAL)**
            - Use "attack_territory" to conquer enemy territories
            - Use "fortify_position" to move armies strategically
            - Use "send_message" for diplomatic communication

            **PHASE 4: END TURN (REQUIRED TO COMPLETE TURN)**
            - ONLY call "end_turn" when army_count = 0
            - This advances the game to the next player
            - Will FAIL if you still have unplaced armies

            **CRITICAL VALIDATION CHECKLIST:**
            Before ending turn, verify:
            ✅ army_count = 0 (all armies placed)
            ✅ No mandatory actions remaining
            ✅ Strategic actions completed (optional)

            **COMMON MISTAKE TO AVOID:**
            ❌ Calling end_turn with army_count > 0
            ✅ Always place armies first, then end turn
            
            **LOOP PREVENTION:**
            - Always check army_count before ending turn
            - If end_turn fails, check if you have unplaced armies
            - Place armies first, then try end_turn again
            
            REMEMBER: 
            - Strategy updates are OPTIONAL and should be quick
            - Game actions are MANDATORY - you must take action every turn
            - Do not overthink - make decisions and execute them
            - Every action should advance your position toward world domination
            
            Think briefly, then ACT using the appropriate MCP tools.
            """
            
            # Track the decision start in the database
            decision_id = await self.track_decision_start(context, formatted_context, prompt)
            
            # Log the decision process
            self.logger.info(f"Making a move in the game using MCP tools")
            
            # Query the agent - it will use tools automatically
            try:
                response = await self.agent.ainvoke({
                    "messages": [HumanMessage(content=prompt)]
                })
            except Exception as llm_error:
                # Check for 402 error (token exhaustion)
                error_str = str(llm_error).lower()
                if "402" in error_str or "payment required" in error_str or "insufficient funds" in error_str or "out of tokens" in error_str:
                    self.logger.error(f"Token exhaustion detected for {self.name}: {llm_error}")
                    await self._handle_token_exhaustion()
                    
                    # Calculate time taken and complete tracking
                    end_time = time.time()
                    time_taken = round(end_time - start_time, 2)
                    
                    if decision_id >= 0:
                        await self.track_decision_complete(
                            decision_id,
                            None,
                            "",
                            [],
                            time_taken,
                            False,
                            f"Token exhaustion: {str(llm_error)}"
                        )
                    
                    return False
                else:
                    # Re-raise other errors
                    raise llm_error
            
            # Calculate time taken
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # Log the agent's response and time taken
            self.logger.info(f"Agent made a decision in {time_taken} seconds")
            
            # Extract the agent's final response
            final_message = response["messages"][-1]
            self.logger.info(f"Agent response: {final_message.content[:200]}...")  # Log first 200 chars
            
            # Complete the decision tracking
            await self.track_decision_complete(
                decision_id,
                response,
                "",  # reasoning - will be extracted from response
                [],  # tools_used - will be extracted from response
                time_taken,
                True,  # Mark as successful
                None   # No error
            )
            
            return True
            
        except Exception as e:
            # Calculate time taken even in case of error
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # Log the error
            self.logger.exception(f"Error playing turn: {str(e)}")
            
            # Complete the decision tracking with error
            if decision_id >= 0:
                await self.track_decision_complete(
                    decision_id,
                    None,  # No response
                    "",    # Empty reasoning
                    [],    # No tools used
                    time_taken,
                    False,  # Not successful
                    str(e)  # Error message
                )
            
            return False
    
    async def run(self, check_interval=15):
        """
        Legacy run method - now deprecated in favor of turn-based coordination.
        This method is kept for backward compatibility but should not be used
        with the new GameRunner turn-based system.
        
        Args:
            check_interval: Seconds to wait between turn checks
        """
        self.logger.warning(f"Agent {self.name} is using deprecated polling-based run() method. "
                           "Consider using turn-based coordination via GameRunner instead.")
        
        if not self.agent:
            await self.initialize()
        
        self.is_running = True
        self.logger.info(f"Starting agent {self.name}'s main loop (legacy mode)")
        
        while self.is_running:
            try:
                is_my_turn = await self.check_is_my_turn()
                
                if is_my_turn:
                    # It's our turn, play
                    self.logger.info(f"It's {self.name}'s turn - playing now")
                    await self.play_turn()
                else:
                    # Not our turn, wait
                    await asyncio.sleep(check_interval)
            
            except Exception as e:
                self.logger.exception(f"Error in main agent loop: {str(e)}")
                await asyncio.sleep(check_interval)  # Wait before retrying
    
    async def _handle_token_exhaustion(self):
        """Handle token exhaustion by notifying tournament manager and pausing game."""
        self.logger.error(f"Token exhaustion detected for agent {self.name} - notifying tournament manager")
        
        try:
            # Try to notify the tournament manager about token exhaustion
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/tournament/token-exhausted",
                    json={
                        "agent_name": self.name,
                        "player_id": self.player_id,
                        "game_id": self.game_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully notified tournament manager of token exhaustion")
                else:
                    self.logger.error(f"Failed to notify tournament manager: HTTP {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying tournament manager of token exhaustion: {e}")
        
        # Stop the agent from making further LLM calls
        self.is_running = False
        self.logger.info(f"Agent {self.name} paused due to token exhaustion")

    def stop(self):
        """Stop the agent's main loop"""
        self.logger.info(f"Stopping agent {self.name}")
        self.is_running = False
