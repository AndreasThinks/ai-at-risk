# -*- coding: utf-8 -*-
import asyncio
import logging
import traceback
import re
from typing import List
import httpx

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from .risk_agent import RiskAgent

class GameRunner:
    """
    Manages a Risk game between multiple AI agents.
    Creates a game, adds agents, and monitors the game progress.
    """
    
    def __init__(self, mcp_client: MultiServerMCPClient, player_configs=None, agent_names: List[str] = None, llm_config=None):
        """
        Initialize a game runner.
        
        Args:
            mcp_client: MCP client for connecting to the Risk server
            player_configs: List of PlayerConfig objects (new preferred method)
            agent_names: List of agent names to create (legacy compatibility)
            llm_config: Optional LLM configuration to pass to all agents (legacy compatibility)
        """
        self.client = mcp_client
        
        # Handle both new PlayerConfig system and legacy agent_names
        if player_configs:
            self.player_configs = player_configs
            self.agent_names = [config.name for config in player_configs]
            self.llm_config = None  # Individual configs used instead
        else:
            # Legacy mode - convert to PlayerConfig objects
            self.agent_names = agent_names or []
            self.llm_config = llm_config
            self.player_configs = self._create_legacy_configs(agent_names, llm_config)
        self.agents = []
        self.game_id = None
        self.running_tasks = []
        
        # Turn management and loop prevention
        self.last_player_id = None
        self.consecutive_same_player_count = 0
        self.max_consecutive_turns = 1  # Maximum consecutive turns for same player
        self.turn_timeout = 300  # 5 minutes per turn maximum
        self.stuck_player_skip_count = {}  # Track how many times each player has been skipped
        self.max_skip_count = 3  # Maximum times to skip a player before declaring them inactive
        
        # Enhanced turn management
        self.player_rotation = []  # Backup turn order
        self.current_rotation_index = 0
        self.emergency_same_player_limit = 100  # Absolute maximum consecutive turns
        self.emergency_total_loop_count = 0
        self.eliminated_players = set()  # Track eliminated players
        self.force_turn_attempts = {}  # Track force turn attempts per player
        self.max_force_attempts = 5  # Maximum force turn attempts before elimination
        
        # Game completion callbacks
        self.game_completion_callbacks = []
        
        # Set up logging
        self.logger = logging.getLogger("game_runner")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Try to create file handler with fallback to console-only
        try:
            # Try multiple log directories in order of preference
            log_paths = [
                "/app/data/logs/game_runner.log",  # Preferred: data/logs directory
                "/tmp/game_runner.log",            # Fallback: tmp directory
                "./data/logs/game_runner.log",     # Local development
                "./game_runner.log"                # Last resort: current directory
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
                    self.logger.info(f"Logging to file: {log_path}")
                    file_handler_created = True
                    break
                except (PermissionError, OSError) as e:
                    continue  # Try next path
            
            if not file_handler_created:
                self.logger.warning("Could not create file handler, using console logging only")
                
        except Exception as e:
            self.logger.warning(f"Error setting up file logging: {e}, using console logging only")
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def add_game_completion_callback(self, callback):
        """Add a callback to be called when the game completes."""
        if callback not in self.game_completion_callbacks:
            self.game_completion_callbacks.append(callback)
    
    def _create_legacy_configs(self, agent_names: List[str], llm_config):
        """Create PlayerConfig objects from legacy parameters for backward compatibility."""
        if not agent_names:
            return []
        
        # Import here to avoid circular imports
        from .player_config import PlayerConfig
        import os
        
        # Get API credentials from environment or llm_config
        api_key = ""
        base_url = ""
        model_name = "gpt-3.5-turbo"
        temperature = 0.7
        
        if llm_config:
            if llm_config.get("model_type") == "openai":
                api_key = llm_config.get("api_key", "")
                base_url = llm_config.get("base_url", "")
                model_name = llm_config.get("model_name", "gpt-3.5-turbo")
                temperature = llm_config.get("temperature", 0.7)
        
        # Fall back to environment variables
        if not api_key:
            api_key = os.getenv('RISK_API_KEY', '')
        if not base_url:
            base_url = os.getenv('RISK_BASE_URL', '')
        
        # Create PlayerConfig objects for each agent
        configs = []
        for name in agent_names:
            config = PlayerConfig(
                name=name,
                model_name=model_name,
                temperature=temperature,
                custom_instructions="Play strategically, balance offense and defense, and adapt to the game situation",
                api_key=api_key,
                base_url=base_url
            )
            configs.append(config)
        
        return configs
    
    async def create_game(self, num_players: int):
        """
        Create a new Risk game using direct API calls.
        
        Args:
            num_players: Number of players in the game
        
        Returns:
            bool: True if game created successfully, False otherwise
        """
        self.logger.info(f"Creating a new game with {num_players} players")
        
        try:
            # Direct API call to create game
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/games", 
                    json={"num_players": num_players}
                )
                
                if response.status_code != 200:
                    self.logger.error(f"Failed to create game: HTTP {response.status_code} - {response.text}")
                    return False
                
                result = response.json()
                if result.get("success") and result.get("game_id"):
                    self.game_id = result["game_id"]
                    self.logger.info(f"Game created successfully with ID: {self.game_id}")
                    return True
                else:
                    self.logger.error(f"Failed to create game: {result}")
                    return False
                
        except Exception as e:
            self.logger.exception(f"Error creating game: {str(e)}")
            return False
    
    async def initialize_agents(self):
        """Initialize all agents and have them join the game"""
        self.logger.info(f"Initializing {len(self.player_configs)} agents")
        
        try:
            # Create agents using player configurations
            for i, config in enumerate(self.player_configs):
                self.logger.info(f"Creating agent {i+1}/{len(self.player_configs)}: {config.name}")
                
                try:
                    # Convert PlayerConfig to LLM config format
                    llm_config = {
                        "model_type": "openai",
                        "api_key": config.api_key,
                        "base_url": config.base_url,
                        "model_name": config.model_name,
                        "temperature": config.temperature
                    }
                    
                    # Create agent with individual configuration
                    agent = RiskAgent(
                        self.client, 
                        config.name, 
                        llm_config=llm_config,
                        custom_instructions=config.custom_instructions
                    )
                    
                    self.logger.info(f"Initializing agent {config.name}...")
                    await agent.initialize()
                    self.agents.append(agent)
                    self.logger.info(f"✅ Created agent: {config.name} (model: {config.model_name}, temp: {config.temperature})")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to create agent {config.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            self.logger.info(f"✅ All {len(self.agents)} agents created successfully")
            
            # Verify game exists before trying to join
            self.logger.info(f"Verifying game {self.game_id} exists...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    # Check if game exists
                    response = await client.get("http://localhost:8080/api/games")
                    if response.status_code == 200:
                        games_data = response.json()
                        games = games_data.get("games", [])
                        game_exists = any(game.get("game_id") == self.game_id for game in games)
                        
                        if game_exists:
                            self.logger.info(f"✅ Game {self.game_id} exists and is ready for players")
                        else:
                            self.logger.error(f"❌ Game {self.game_id} not found in active games list")
                            self.logger.info(f"Available games: {[g.get('game_id') for g in games]}")
                            return False
                    else:
                        self.logger.warning(f"Could not verify game existence: HTTP {response.status_code}")
                        # Continue anyway - game might exist but API call failed
                
                except Exception as e:
                    self.logger.warning(f"Could not verify game existence: {e}")
                    # Continue anyway - game might exist but verification failed
                
                # Have agents join the game directly via API
                self.logger.info(f"Having {len(self.agents)} agents join game {self.game_id}...")
                
                for i, agent in enumerate(self.agents):
                    self.logger.info(f"Agent {i+1}/{len(self.agents)}: {agent.name} attempting to join game...")
                    
                    try:
                        # Join game via direct API call with timeout
                        response = await client.post(
                            "http://localhost:8080/api/games/join",
                            json={"game_id": self.game_id, "player_name": agent.name},
                            timeout=10.0
                        )
                        
                        self.logger.info(f"Join game API response for {agent.name}: HTTP {response.status_code}")
                        
                        if response.status_code != 200:
                            self.logger.error(f"❌ {agent.name} failed to join game: HTTP {response.status_code}")
                            self.logger.error(f"Response text: {response.text}")
                            return False
                        
                        result = response.json()
                        self.logger.info(f"Join game result for {agent.name}: {result}")
                        
                        if result.get("success") and result.get("player_id"):
                            # Store game and player information in the agent
                            agent.game_id = self.game_id
                            agent.player_id = result["player_id"]
                            self.logger.info(f"✅ Agent {agent.name} joined game {self.game_id} as player {agent.player_id}")
                            
                            # Track the model assignment in the action tracker
                            try:
                                from src.persistence.action_tracker import get_action_tracker
                                tracker = get_action_tracker()
                                if tracker:
                                    # Get the player config for this agent
                                    player_config = self.player_configs[i] if i < len(self.player_configs) else None
                                    if player_config:
                                        tracker.track_player_model(
                                            self.game_id,
                                            agent.player_id,
                                            agent.name,
                                            player_config.model_name,
                                            player_config.temperature
                                        )
                                        self.logger.info(f"✅ Tracked model assignment: {agent.name} -> {player_config.model_name}")
                                    else:
                                        self.logger.warning(f"⚠️ No player config found for agent {agent.name}, using default model tracking")
                            except Exception as e:
                                self.logger.error(f"❌ Failed to track model assignment for {agent.name}: {e}")
                        else:
                            self.logger.error(f"❌ {agent.name} join game failed: {result}")
                            return False
                            
                    except httpx.TimeoutException:
                        self.logger.error(f"❌ {agent.name} join game timed out")
                        return False
                    except Exception as e:
                        self.logger.error(f"❌ {agent.name} join game error: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
            
            self.logger.info(f"🎉 All {len(self.agents)} agents successfully joined game {self.game_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in initialize_agents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_game(self):
        """Run the game with turn-based coordination"""
        self.logger.info(f"Starting game {self.game_id} with {len(self.agents)} agents")
        
        try:
            # Initialize all agents first
            for agent in self.agents:
                if not agent.agent:
                    await agent.initialize()
                    self.logger.info(f"Initialized agent {agent.name}")
            
            # Run the turn-based game loop
            game_running = True
            turn_timeout = 300  # 5 minutes per turn maximum
            
            while game_running:
                try:
                    # Check game status
                    game_status = await self._get_game_status()
                    if not game_status:
                        self.logger.error("Failed to get game status")
                        break
                    
                    # Check if game is over
                    if game_status.get("phase") == "game_over":
                        self.logger.info("Game has ended!")
                        
                        # Try to determine the winner
                        winner_info = None
                        if "winner" in game_status:
                            winner_data = game_status["winner"]
                            winner_info = {
                                'player_id': winner_data.get('player_id'),
                                'player_name': winner_data.get('player_name'),
                                'reason': 'victory'
                            }
                            self.logger.info(f"Winner: {winner_info['player_name']}")
                        
                        # Send final messages before ending
                        await self.send_final_messages()
                        
                        # Stop game with winner information
                        await self.stop_game(winner_info)
                        break
                    
                    # Find the current player
                    current_player_id = game_status.get("current_player_id")
                    if not current_player_id:
                        self.logger.warning("No current player found, waiting...")
                        await asyncio.sleep(5)
                        continue
                    
                    # EMERGENCY CIRCUIT BREAKER: Check absolute limit
                    if self.consecutive_same_player_count > self.emergency_same_player_limit:
                        self.logger.critical(f"🚨 EMERGENCY: Player {current_player_id} has played {self.consecutive_same_player_count} consecutive turns - TERMINATING GAME")
                        await self.stop_game({"reason": "emergency_infinite_loop", "player_id": current_player_id})
                        break
                    
                    # TURN LOOP PREVENTION: Check for consecutive same player
                    if self.last_player_id == current_player_id:
                        self.consecutive_same_player_count += 1
                        self.emergency_total_loop_count += 1
                        self.logger.warning(f"Same player {current_player_id} playing consecutive turn #{self.consecutive_same_player_count}")
                        
                        if self.consecutive_same_player_count > self.max_consecutive_turns:
                            self.logger.error(f"🚨 TURN LOOP DETECTED! Player {current_player_id} has played {self.consecutive_same_player_count} consecutive turns")
                            
                            # Track skip count for this player
                            if current_player_id not in self.stuck_player_skip_count:
                                self.stuck_player_skip_count[current_player_id] = 0
                            self.stuck_player_skip_count[current_player_id] += 1
                            
                            # Check if we should skip this player
                            if self.stuck_player_skip_count[current_player_id] <= self.max_skip_count:
                                self.logger.warning(f"⏭️ SKIPPING stuck player {current_player_id} (skip #{self.stuck_player_skip_count[current_player_id]}/{self.max_skip_count})")
                                
                                # Try to force end their turn via API
                                success = await self._force_end_turn(current_player_id)
                                await asyncio.sleep(3)  # Give time for turn to end
                                
                                # Verify the turn actually advanced
                                new_status = await self._get_game_status()
                                if new_status and new_status.get("current_player_id") == current_player_id:
                                    self.logger.warning(f"⚠️ Force end turn didn't advance - trying advanced recovery")
                                    await self._force_advance_to_next_player(current_player_id)
                                
                                continue
                            else:
                                self.logger.error(f"❌ Player {current_player_id} has been skipped {self.max_skip_count} times - FORCING TURN ADVANCEMENT")
                                
                                # Try force phase advancement first
                                phase_advanced = await self._force_phase_advancement(current_player_id, "stuck_player")
                                if phase_advanced:
                                    self.logger.info(f"✅ Successfully advanced phase for stuck player {current_player_id}")
                                    # Reset counters since we advanced
                                    self.consecutive_same_player_count = 0
                                    self.last_player_id = None
                                    continue
                                
                                # If phase advancement failed, try player advancement
                                success = await self._force_advance_to_next_player(current_player_id)
                                if success:
                                    self.logger.info(f"✅ Successfully advanced past stuck player {current_player_id}")
                                    # Reset counters since we advanced
                                    self.consecutive_same_player_count = 0
                                    self.last_player_id = None
                                    continue
                                else:
                                    self.logger.critical(f"💥 FAILED to advance past stuck player {current_player_id} - eliminating player")
                                    await self._eliminate_stuck_player(current_player_id)
                                    continue
                    else:
                        # Different player, reset counter
                        self.consecutive_same_player_count = 0
                        self.last_player_id = current_player_id
                        
                        # Initialize player rotation if not set
                        if not self.player_rotation:
                            self.player_rotation = [agent.player_id for agent in self.agents if agent.player_id not in self.eliminated_players]
                            self.logger.info(f"Initialized player rotation: {len(self.player_rotation)} active players")
                    
                    # Find the agent for the current player
                    current_agent = None
                    for agent in self.agents:
                        if agent.player_id == current_player_id:
                            current_agent = agent
                            break
                    
                    if not current_agent:
                        self.logger.error(f"No agent found for current player {current_player_id}")
                        await asyncio.sleep(5)
                        continue
                    
                    # Let the current agent play their turn
                    self.logger.info(f"It's {current_agent.name}'s turn (Player ID: {current_player_id})")
                    
                    # Run the agent's turn with a timeout
                    turn_success = False
                    try:
                        await asyncio.wait_for(
                            current_agent.play_turn(),
                            timeout=self.turn_timeout
                        )
                        self.logger.info(f"✅ {current_agent.name} completed their turn")
                        turn_success = True
                        
                        # Reset skip count on successful turn
                        if current_player_id in self.stuck_player_skip_count:
                            del self.stuck_player_skip_count[current_player_id]
                            
                    except asyncio.TimeoutError:
                        self.logger.warning(f"⏰ {current_agent.name}'s turn timed out after {self.turn_timeout} seconds")
                        # Force end turn on timeout
                        await self._force_end_turn(current_player_id)
                        
                    except Exception as e:
                        self.logger.exception(f"💥 Error during {current_agent.name}'s turn: {str(e)}")
                        # Force end turn on error
                        await self._force_end_turn(current_player_id)
                    
                    # Brief pause before checking for next turn
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.exception(f"Error in game loop: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retrying
                
        except Exception as e:
            self.logger.exception(f"Error running game: {str(e)}")
    
    async def _get_game_status(self):
        """Get current game status via API call"""
        try:
            # Use the first agent's player_id for the status check
            # This is needed because the API endpoint requires a player_id parameter
            player_id = self.agents[0].player_id if self.agents and self.agents[0].player_id else "system"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:8080/api/games/{self.game_id}/status",
                    params={"player_id": player_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("game_status", {})
                else:
                    self.logger.error(f"Failed to get game status: HTTP {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            self.logger.exception(f"Error getting game status: {str(e)}")
            return None
    
    async def send_final_messages(self):
        """Have each agent send one final message when the game ends"""
        self.logger.info("Starting final message phase - each agent will send one last message")
        
        for agent in self.agents:
            try:
                self.logger.info(f"Requesting final message from {agent.name}")
                
                # Create a simple prompt for the final message
                final_message_prompt = f"""
                The Risk game has ended! This is your final opportunity to send one message to all other players.
                
                You should:
                - Congratulate the winner (if there is one)
                - Reflect briefly on the game
                - Share any final thoughts about the strategies used
                - Keep it concise and respectful
                
                Use the "send_message" tool to send your final message to all players.
                """
                
                # Have the agent send their final message
                response = await agent.agent.ainvoke({
                    "messages": [HumanMessage(content=final_message_prompt)]
                })
                
                self.logger.info(f"{agent.name} sent their final message")
                
            except Exception as e:
                self.logger.exception(f"Error getting final message from {agent.name}: {str(e)}")
                # Continue with other agents even if one fails
                continue
        
        self.logger.info("Final message phase completed")
        
    async def _force_end_turn(self, player_id: str):
        """Force end a player's turn via direct API call"""
        try:
            self.logger.info(f"🔄 Force ending turn for player {player_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/games/end-turn",
                    json={"game_id": self.game_id, "player_id": player_id}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        self.logger.info(f"✅ Successfully force-ended turn for player {player_id}")
                        return True
                    else:
                        self.logger.warning(f"⚠️ Force end turn failed: {result.get('message', 'Unknown error')}")
                        return False
                else:
                    self.logger.error(f"❌ Force end turn API call failed: HTTP {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            self.logger.exception(f"💥 Error force ending turn for player {player_id}: {str(e)}")
            return False
    
    def get_next_player_in_rotation(self, current_player_id: str):
        """Get the next player in rotation, skipping eliminated players"""
        try:
            # Initialize player rotation if not set
            if not self.player_rotation:
                self.player_rotation = [agent.player_id for agent in self.agents if agent.player_id not in self.eliminated_players]
                self.logger.info(f"Initialized player rotation with {len(self.player_rotation)} active players")
            
            # Filter out eliminated players
            active_players = [p for p in self.player_rotation if p not in self.eliminated_players]
            
            if not active_players:
                self.logger.error("No active players remaining in rotation")
                return None
            
            # Find current player index
            try:
                current_index = active_players.index(current_player_id)
                next_index = (current_index + 1) % len(active_players)
                next_player = active_players[next_index]
                self.logger.info(f"Next player in rotation: {next_player} (index {next_index}/{len(active_players)})")
                return next_player
            except ValueError:
                # Current player not in rotation, return first active player
                self.logger.warning(f"Current player {current_player_id} not in rotation, returning first active player")
                return active_players[0]
                
        except Exception as e:
            self.logger.exception(f"Error getting next player in rotation: {e}")
            return None
    
    async def _force_advance_to_next_player(self, stuck_player_id: str):
        """Try multiple methods to advance past a stuck player"""
        try:
            self.logger.info(f"🔄 Attempting to force advance past stuck player {stuck_player_id}")
            
            # Track force attempts
            if stuck_player_id not in self.force_turn_attempts:
                self.force_turn_attempts[stuck_player_id] = 0
            self.force_turn_attempts[stuck_player_id] += 1
            
            # Method 1: Multiple force end turn attempts
            self.logger.info("Method 1: Multiple force end turn attempts")
            for attempt in range(3):
                self.logger.info(f"Force end turn attempt {attempt + 1}/3")
                success = await self._force_end_turn(stuck_player_id)
                if success:
                    await asyncio.sleep(2)  # Give time for turn to advance
                    new_status = await self._get_game_status()
                    if new_status and new_status.get("current_player_id") != stuck_player_id:
                        self.logger.info(f"✅ Method 1 successful: Turn advanced to {new_status.get('current_player_id')}")
                        return True
                    else:
                        self.logger.warning(f"Force end turn succeeded but player {stuck_player_id} is still current")
                else:
                    self.logger.warning(f"Force end turn attempt {attempt + 1} failed")
                    await asyncio.sleep(1)
            
            # Method 2: Try to manually set next player (if we can determine it)
            self.logger.info("Method 2: Attempting to determine and set next player")
            next_player = self.get_next_player_in_rotation(stuck_player_id)
            if next_player:
                self.logger.info(f"Identified next player in rotation: {next_player}")
                # For now, we'll use repeated force end turn as our "set next player" method
                # since we don't have a direct API endpoint for this
                for attempt in range(5):
                    await self._force_end_turn(stuck_player_id)
                    await asyncio.sleep(1)
                    new_status = await self._get_game_status()
                    if new_status and new_status.get("current_player_id") == next_player:
                        self.logger.info(f"✅ Method 2 successful: Advanced to expected next player {next_player}")
                        return True
                    elif new_status and new_status.get("current_player_id") != stuck_player_id:
                        self.logger.info(f"✅ Method 2 partial success: Advanced to different player {new_status.get('current_player_id')}")
                        return True
            
            # Method 3: If all else fails, eliminate the stuck player
            self.logger.warning("Method 3: All force advancement methods failed, will eliminate stuck player")
            return False
            
        except Exception as e:
            self.logger.exception(f"💥 Error in force advance to next player: {e}")
            return False
    
    async def _eliminate_stuck_player(self, stuck_player_id: str):
        """Eliminate a stuck player from the game rotation"""
        try:
            self.logger.critical(f"🚫 ELIMINATING stuck player {stuck_player_id} from game rotation")
            
            # Add to eliminated players set
            self.eliminated_players.add(stuck_player_id)
            
            # Remove from player rotation
            if stuck_player_id in self.player_rotation:
                self.player_rotation.remove(stuck_player_id)
            
            # Find and mark agent as eliminated
            for agent in self.agents:
                if agent.player_id == stuck_player_id:
                    agent.eliminated = True
                    self.logger.info(f"Marked agent {agent.name} as eliminated")
                    break
            
            # Reset turn tracking for this player
            self.consecutive_same_player_count = 0
            self.last_player_id = None
            
            # Clear skip count
            if stuck_player_id in self.stuck_player_skip_count:
                del self.stuck_player_skip_count[stuck_player_id]
            
            remaining_players = len([a for a in self.agents if a.player_id not in self.eliminated_players])
            self.logger.info(f"Player {stuck_player_id} eliminated. {remaining_players} players remaining")
            
            # Check if we need to end the game due to insufficient players
            if remaining_players < 2:
                self.logger.critical("Less than 2 players remaining - ending game")
                await self.stop_game({"reason": "insufficient_players"})
                return True
            
            # Force end the eliminated player's turn one more time
            await self._force_end_turn(stuck_player_id)
            await asyncio.sleep(3)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"💥 Error eliminating stuck player: {e}")
            return False

    async def stop_game(self, winner_info: dict = None):
        """Stop the game and all agents with completion statistics"""
        self.logger.info("Stopping all agents and game")
        
        # Import action_tracker here to avoid circular imports
        try:
            from src.persistence.action_tracker import action_tracker
            
            # Store completion statistics if we have game info
            if hasattr(self, 'game_id') and self.game_id:
                if winner_info:
                    success = action_tracker.finish_game_with_stats(
                        self.game_id,
                        winner_player_id=winner_info.get('player_id'),
                        winner_player_name=winner_info.get('player_name'),
                        completion_reason=winner_info.get('reason', 'completed')
                    )
                else:
                    success = action_tracker.finish_game_with_stats(
                        self.game_id,
                        completion_reason='stopped'
                    )
                
                if success:
                    self.logger.info(f"Game {self.game_id} completed with comprehensive statistics")
                else:
                    self.logger.warning(f"Failed to store completion statistics for game {self.game_id}")
        except Exception as e:
            self.logger.error(f"Error storing completion statistics: {e}")
        
        # Notify tournament manager if this is a tournament game
        if hasattr(self, 'game_completion_callbacks') and self.game_completion_callbacks:
            for callback in self.game_completion_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(winner_info or {'reason': 'stopped'})
                    else:
                        callback(winner_info or {'reason': 'stopped'})
                except Exception as e:
                    self.logger.error(f"Error in game completion callback: {e}")
        
        # Stop all agents
        for agent in self.agents:
            agent.stop()
        
        # Cancel all tasks
        for task in self.running_tasks:
            task.cancel()
        
        self.logger.info("Game stopped")
    
    async def _handle_phase_progression(self, game_status: dict):
        """Handle automatic phase progression based on game state."""
        try:
            current_phase = game_status.get("phase", "unknown")
            current_player_id = game_status.get("current_player_id")
            
            if not current_player_id:
                return
            
            # Check if we need to auto-advance phase via API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/games/auto-advance-phase",
                    json={"game_id": self.game_id, "player_id": current_player_id}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("phase_advanced"):
                        old_phase = result.get("old_phase", "unknown")
                        new_phase = result.get("new_phase", "unknown")
                        self.logger.info(f"🔄 Auto-advanced phase: {old_phase} → {new_phase}")
                else:
                    self.logger.debug(f"Phase progression check: {response.status_code}")
                    
        except Exception as e:
            self.logger.warning(f"Error in phase progression handler: {e}")
    
    async def _force_phase_advancement(self, player_id: str, reason: str = "stuck"):
        """Force phase advancement for a stuck player."""
        try:
            self.logger.warning(f"🔄 Forcing phase advancement for player {player_id}: {reason}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/api/games/force-advance-phase",
                    json={"game_id": self.game_id, "player_id": player_id, "reason": reason}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        self.logger.info(f"✅ Successfully forced phase advancement: {result.get('message', 'Unknown result')}")
                        return True
                    else:
                        self.logger.warning(f"⚠️ Phase advancement failed: {result.get('message', 'Unknown error')}")
                        return False
                else:
                    self.logger.error(f"❌ Force phase advancement API call failed: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            self.logger.exception(f"💥 Error forcing phase advancement: {e}")
            return False
