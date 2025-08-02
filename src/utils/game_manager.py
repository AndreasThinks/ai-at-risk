import json
import os
from typing import Dict, Optional, List
from datetime import datetime

from game.game_state import GameState
from persistence.game_persistence import game_persistence

class GameManager:
    """Manages multiple Risk games."""
    
    def __init__(self):
        self.games: Dict[str, GameState] = {}
        self.territory_data = self._load_territory_data()
        self.tournament_manager = None  # Will be initialized if tournament mode is enabled
        self._needs_tournament_start = False  # Flag to start tournament when event loop is ready
        self._load_persisted_games()
        self._initialize_tournament_mode()
    
    def _load_territory_data(self) -> dict:
        """Load territory data from JSON file."""
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.dirname(__file__))
        territory_file = os.path.join(current_dir, 'data', 'territories.json')
        
        try:
            with open(territory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Territory data file not found: {territory_file}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in territory data file: {territory_file}")
    
    def create_game(self, num_players: int, game_name: Optional[str] = None) -> tuple[bool, str, str]:
        """
        Create a new game.
        Returns (success, game_id, message).
        """
        if num_players < 2 or num_players > 6:
            return False, "", "Number of players must be between 2 and 6"
        
        # Generate unique game ID
        import uuid
        game_id = str(uuid.uuid4())[:8]  # Short ID for easy reference
        
        # Create game state
        game_state = GameState(game_id, num_players, self.territory_data)
        self.games[game_id] = game_state
        
        game_name_str = f" '{game_name}'" if game_name else ""
        message = f"Game{game_name_str} created with ID: {game_id}. Waiting for {num_players} players to join."
        
        return True, game_id, message
    
    def get_game(self, game_id: str) -> Optional[GameState]:
        """Get a game by ID."""
        return self.games.get(game_id)
    
    def join_game(self, game_id: str, player_name: str) -> tuple[bool, str, str]:
        """
        Join a game.
        Returns (success, player_id, message).
        """
        game = self.get_game(game_id)
        if not game:
            return False, "", "Game not found"
        
        return game.add_player(player_name)
    
    def list_active_games(self) -> List[dict]:
        """List all active games."""
        games_info = []
        for game_id, game in self.games.items():
            # Convert players dict to a list of player objects for the dashboard
            players_list = [player.to_dict() for player in game.players.values()]
            
            games_info.append({
                'game_id': game_id,
                'phase': game.game_phase.value,
                'players': players_list,  # Now includes actual player data
                'max_players': game.num_players,
                'turn_number': game.turn_number,
                'created_at': game.created_at.isoformat()
            })
        return games_info
    
    def cleanup_finished_games(self) -> int:
        """Remove finished games from memory. Returns number of games removed."""
        from game.game_state import GamePhase
        
        finished_games = [
            game_id for game_id, game in self.games.items() 
            if game.game_phase == GamePhase.GAME_OVER
        ]
        
        for game_id in finished_games:
            del self.games[game_id]
        
        return len(finished_games)
    
    def cleanup_old_games(self, hours: int = 24) -> int:
        """Remove games older than specified hours. Returns number of games removed."""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        old_games = [
            game_id for game_id, game in self.games.items()
            if game.created_at < cutoff_time
        ]
        
        for game_id in old_games:
            del self.games[game_id]
        
        return len(old_games)
    
    def get_game_count(self) -> int:
        """Get the total number of active games."""
        return len(self.games)
    
    def get_player_games(self, player_name: str) -> List[dict]:
        """Get all games that a player is participating in."""
        player_games = []
        for game_id, game in self.games.items():
            for player in game.players.values():
                if player.name.lower() == player_name.lower():
                    player_games.append({
                        'game_id': game_id,
                        'phase': game.game_phase.value,
                        'player_id': player.player_id,
                        'is_current_player': game.get_current_player() == player,
                        'turn_number': game.turn_number
                    })
                    break
        return player_games
    
    def _load_persisted_games(self) -> None:
        """Load persisted games on startup."""
        try:
            saved_game_ids = game_persistence.list_saved_games()
            loaded_count = 0
            
            for game_id in saved_game_ids:
                try:
                    game_state_data = game_persistence.load_game_state(game_id)
                    if game_state_data:
                        # Create a new GameState instance from saved data
                        # Note: This is a simplified restoration - in practice you'd need
                        # to properly deserialize all game components
                        game_state = GameState(game_id, game_state_data.get('max_players', 2), self.territory_data)
                        self.games[game_id] = game_state
                        loaded_count += 1
                except Exception as e:
                    print(f"Failed to load game {game_id}: {e}")
                    
            if loaded_count > 0:
                print(f"Loaded {loaded_count} persisted games")
        except Exception as e:
            print(f"Failed to load persisted games: {e}")
    
    def save_game(self, game_id: str) -> bool:
        """Save a game to persistent storage."""
        game = self.get_game(game_id)
        if not game:
            return False
        
        # Serialize game state
        game_state_data = {
            'game_id': game_id,
            'num_players': game.num_players,
            'game_phase': game.game_phase.value if hasattr(game.game_phase, 'value') else str(game.game_phase),
            'turn_number': getattr(game, 'turn_number', 0),
            'created_at': game.created_at.isoformat(),
            'players': {
                pid: {
                    'player_id': player.player_id,
                    'name': player.name,
                    'color': getattr(player, 'color', 'Unknown'),
                    'territories': list(getattr(player, 'territories', [])),
                    'army_count': getattr(player, 'army_count', 0),
                    'cards': getattr(player, 'cards', []),
                    'is_eliminated': getattr(player, 'is_eliminated', False)
                }
                for pid, player in game.players.items()
            }
        }
        
        return game_persistence.save_game_state(game_id, game_state_data)
    
    def auto_save_all_games(self) -> int:
        """Auto-save all games that need saving. Returns number of games saved."""
        saved_count = 0
        for game_id in self.games.keys():
            if game_persistence.should_auto_save(game_id):
                if self.save_game(game_id):
                    saved_count += 1
        return saved_count
    
    def _initialize_tournament_mode(self) -> None:
        """Initialize tournament mode if enabled."""
        try:
            import sys
            import os
            
            tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
            
            if tournament_mode:
                # Import tournament manager
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from tournament.tournament_manager import TournamentManager
                
                # Create tournament config from environment
                config = {
                    'db_path': os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db'),
                    'submit_duration': int(os.getenv('TOURNAMENT_SUBMIT_PHASE_DURATION', '900')),
                    'voting_duration': int(os.getenv('TOURNAMENT_VOTING_PHASE_DURATION', '900')),
                    'game_duration': int(os.getenv('TOURNAMENT_GAME_PHASE_DURATION', '3600')),
                    'end_screen_duration': int(os.getenv('TOURNAMENT_END_SCREEN_DURATION', '300')),
                    'auto_restart': os.getenv('TOURNAMENT_AUTO_RESTART', 'true').lower() == 'true',
                    'max_submissions': int(os.getenv('TOURNAMENT_MAX_SUBMISSIONS', '20')),
                    'selected_characters': int(os.getenv('TOURNAMENT_SELECTED_CHARACTERS', '4'))
                }
                
                self.tournament_manager = TournamentManager(config)
                
                # Add callback to handle tournament phase changes
                self.tournament_manager.add_phase_change_callback(self._on_tournament_phase_change)
                
                print(f"Tournament mode initialized with config: {config}")
                
                # Mark that we need to auto-start tournament when event loop is available
                self._needs_tournament_start = True
                
        except Exception as e:
            print(f"Failed to initialize tournament mode: {e}")
            self.tournament_manager = None
    
    async def _on_tournament_phase_change(self, tournament_status: dict) -> None:
        """Handle tournament phase changes."""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from tournament.tournament_manager import TournamentPhase
            
            phase = tournament_status.get('phase')
            print(f"Tournament phase changed to: {phase}")
            
            # If transitioning to game phase, start the tournament game
            if phase == TournamentPhase.GAME.value and self.tournament_manager:
                await self._start_tournament_game()
                
        except Exception as e:
            print(f"Error handling tournament phase change: {e}")
    
    async def _start_tournament_game(self) -> None:
        """Start a tournament game with selected characters using simple runner pattern with retry logic."""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                print(f"Starting tournament game (attempt {attempt}/{max_attempts})")
                
                if not self.tournament_manager:
                    print("Tournament manager not available")
                    return
                
                # Get selected characters for the game
                selected_characters = self.tournament_manager.get_selected_characters_for_game()
                
                if not selected_characters:
                    print("No characters selected for tournament game")
                    return
                
                print(f"Starting tournament game with {len(selected_characters)} characters")
                
                # Call the actual game creation logic
                success = await self._create_tournament_game_internal(selected_characters)
                
                if success:
                    print(f"Tournament game created successfully on attempt {attempt}")
                    return
                else:
                    print(f"Tournament game creation failed on attempt {attempt}")
                    
            except Exception as e:
                print(f"Error on tournament game creation attempt {attempt}: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait before retry (exponential backoff)
            if attempt < max_attempts:
                delay = min(3 * (2 ** (attempt - 1)), 15)  # 3s, 6s, 12s max
                print(f"Waiting {delay}s before retry...")
                import asyncio
                await asyncio.sleep(delay)
        
        print(f"All {max_attempts} tournament game creation attempts failed")
    
    async def _create_tournament_game_internal(self, selected_characters) -> bool:
        """Internal method to create tournament game. Returns True on success."""
        try:
            
            # Import required modules
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from agents.player_config import PlayerConfig
            from agents.game_runner import GameRunner
            from langchain_mcp_adapters.client import MultiServerMCPClient
            import asyncio
            
            # Get API credentials from environment variables (like simple_runner does)
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('RISK_API_KEY', '')
            base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('RISK_BASE_URL', '')
            
            if not api_key or not base_url:
                print(f"Warning: Missing API credentials. Using defaults.")
                api_key = api_key or "dummy-key"
                base_url = base_url or "http://localhost:11434"
            
            # Convert tournament character configs to PlayerConfig objects
            player_configs = []
            for char in selected_characters:
                # Build custom instructions from personality and custom_instructions
                personality = char.get('personality', '') if isinstance(char, dict) else char.personality
                custom_instructions = char.get('custom_instructions', '') if isinstance(char, dict) else char.custom_instructions
                
                if personality and custom_instructions:
                    full_instructions = f"{personality}\n\n{custom_instructions}"
                elif personality:
                    full_instructions = personality
                elif custom_instructions:
                    full_instructions = custom_instructions
                else:
                    char_name = char.get('name') if isinstance(char, dict) else char.name
                    full_instructions = f"You are {char_name}, a strategic Risk player."
                
                # Handle both dict and TournamentCharacter object formats
                char_name = char.get('name') if isinstance(char, dict) else char.name
                char_model = char.get('model_name') if isinstance(char, dict) else char.model_name
                char_temp = char.get('temperature') if isinstance(char, dict) else char.temperature
                
                player_config = PlayerConfig(
                    name=char_name,
                    model_name=char_model,
                    temperature=char_temp,
                    custom_instructions=full_instructions,
                    api_key=api_key,
                    base_url=base_url
                )
                player_configs.append(player_config)
                print(f"  - {char_name}: {char_model} (temp: {char_temp})")
            
            # Create MCP client for the game runner
            mcp_client = MultiServerMCPClient({
                "risk": {
                    "transport": "sse",
                    "url": "http://localhost:8080/mcp"
                }
            })
            
            # Connect to MCP server and get tools
            print("Connecting to MCP server for tournament game...")
            tools = await mcp_client.get_tools()
            print(f"Connected to MCP server, loaded {len(tools)} tools for tournament game")
            
            # Create the game runner with player configs (like simple_runner does)
            game_runner = GameRunner(
                mcp_client=mcp_client,
                player_configs=player_configs
            )
            
            # Create the game (this will create a new game and set game_runner.game_id)
            print(f"Creating tournament game with {len(player_configs)} players...")
            create_success = await game_runner.create_game(len(player_configs))
            if not create_success:
                print("Failed to create tournament game via GameRunner")
                return
            
            # Get the game ID that was created
            game_id = game_runner.game_id
            print(f"Tournament game created with ID: {game_id}")
            
            # Set the game ID in tournament manager
            self.tournament_manager.set_game_id(game_id)
            
            # Initialize agents and have them join the game (like simple_runner does)
            print("Initializing agents and joining tournament game...")
            init_success = await game_runner.initialize_agents()
            if not init_success:
                print("Failed to initialize agents for tournament game")
                return
            
            print(f"All agents successfully joined tournament game {game_id}")
            
            # Track model assignments in the action tracker
            await self._track_tournament_model_assignments(game_id, player_configs)
            
            # Add callback to handle game completion
            if hasattr(game_runner, 'add_game_completion_callback'):
                game_runner.add_game_completion_callback(self._on_tournament_game_complete)
            
            # Start the game in the background (like simple_runner does)
            print("Starting tournament game loop...")
            asyncio.create_task(game_runner.run_game())
            
            print(f"Tournament game {game_id} started successfully with AI agents")
            return True
            
        except Exception as e:
            print(f"Error starting tournament game: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _force_tournament_game_completion(self, game_id: str, game_result: dict) -> None:
        """Force completion of a tournament game and store statistics."""
        try:
            print(f"Forcing completion of tournament game {game_id}")
            print(f"Game result received: {game_result}")
            
            # Import required modules
            from persistence.action_tracker import get_action_tracker
            
            # Get action tracker
            tracker = get_action_tracker()
            if not tracker:
                print("Action tracker not available")
                return
            
            # Extract winner information from game result
            winner_player_id = game_result.get('winner_player_id')
            winner_player_name = game_result.get('winner_player_name')
            completion_reason = game_result.get('completion_reason', 'game_completed')
            
            print(f"Extracted winner info: player_id={winner_player_id}, player_name={winner_player_name}, reason={completion_reason}")
            
            # Force the game to finish with comprehensive statistics
            success = tracker.finish_game_with_stats(
                game_id=game_id,
                winner_player_id=winner_player_id,
                winner_player_name=winner_player_name,
                completion_reason=completion_reason,
                status='completed'
            )
            
            if success:
                print(f"Successfully stored completion statistics for tournament game {game_id}")
                
                # Remove the game from active games to stop background processes
                if game_id in self.games:
                    print(f"Removing tournament game {game_id} from active games")
                    del self.games[game_id]
                else:
                    print(f"Tournament game {game_id} was not in active games list")
                    
            else:
                print(f"Failed to store completion statistics for tournament game {game_id}")
                
        except Exception as e:
            print(f"Error forcing tournament game completion: {e}")
            import traceback
            traceback.print_exc()
    
    async def _on_tournament_game_complete(self, game_result: dict) -> None:
        """Handle tournament game completion."""
        try:
            if not self.tournament_manager:
                return
            
            print(f"Tournament game completed: {game_result}")
            
            # Set the game result in tournament manager
            self.tournament_manager.set_game_result(game_result)
            
            # Force game completion and cleanup immediately
            current_game_id = self.tournament_manager.get_current_game_id()
            if current_game_id:
                await self._force_tournament_game_completion(current_game_id, game_result)
            
            # The tournament manager will handle the transition to end screen phase
            
        except Exception as e:
            print(f"Error handling tournament game completion: {e}")
    
    async def _auto_start_tournament(self) -> None:
        """Auto-start tournament if not already running."""
        try:
            if not self.tournament_manager:
                return
            
            # Ensure callback is registered (in case it was missed during initialization)
            self.tournament_manager.add_phase_change_callback(self._on_tournament_phase_change)
            
            # Check if tournament is already active
            if self.tournament_manager.is_tournament_active():
                print("Tournament already active")
                return
            
            # Start a new tournament
            tournament_id = await self.tournament_manager.start_tournament()
            print(f"Auto-started tournament: {tournament_id}")
            
        except Exception as e:
            print(f"Error auto-starting tournament: {e}")
    
    async def start_tournament_if_needed(self) -> None:
        """Start tournament if it was marked as needed during initialization."""
        if self._needs_tournament_start and self.tournament_manager:
            self._needs_tournament_start = False  # Reset flag
            await self._auto_start_tournament()
    
    async def _track_tournament_model_assignments(self, game_id: str, player_configs: List) -> None:
        """Track model assignments for tournament players in the action tracker."""
        try:
            from persistence.action_tracker import get_action_tracker
            
            tracker = get_action_tracker()
            if not tracker:
                print("Action tracker not available for model tracking")
                return
            
            print(f"Tracking model assignments for tournament game {game_id}")
            
            # Get the actual game to map player names to player IDs
            game = self.get_game(game_id)
            if not game:
                print(f"Game {game_id} not found for model tracking")
                return
            
            # Track each player's model assignment
            for config in player_configs:
                # Find the player ID for this player name
                player_id = None
                for pid, player in game.players.items():
                    if player.name == config.name:
                        player_id = pid
                        break
                
                if player_id:
                    success = tracker.track_player_model(
                        game_id=game_id,
                        player_id=player_id,
                        player_name=config.name,
                        model_name=config.model_name,
                        temperature=config.temperature
                    )
                    
                    if success:
                        print(f"  Tracked: {config.name} -> {config.model_name} (temp: {config.temperature})")
                    else:
                        print(f"  Failed to track: {config.name} -> {config.model_name}")
                else:
                    print(f"  Could not find player ID for {config.name}")
            
            print(f"Model tracking completed for tournament game {game_id}")
            
        except Exception as e:
            print(f"Error tracking tournament model assignments: {e}")
            import traceback
            traceback.print_exc()
    
    async def _create_tournament_game_with_configs(self, player_configs: List[Dict]) -> bool:
        """Create tournament game with provided player configurations. Used by validation logic."""
        try:
            print(f"Creating tournament game with {len(player_configs)} player configs")
            
            # Import required modules
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from agents.player_config import PlayerConfig
            from agents.game_runner import GameRunner
            from langchain_mcp_adapters.client import MultiServerMCPClient
            import asyncio
            
            # Convert dict configs to PlayerConfig objects
            processed_configs = []
            for config in player_configs:
                if isinstance(config, dict):
                    player_config = PlayerConfig(
                        name=config['name'],
                        model_name=config['model_name'],
                        temperature=config['temperature'],
                        custom_instructions=config['custom_instructions'],
                        api_key=config['api_key'],
                        base_url=config['base_url']
                    )
                    processed_configs.append(player_config)
                    print(f"  - {config['name']}: {config['model_name']} (temp: {config['temperature']})")
                else:
                    processed_configs.append(config)
            
            # Create MCP client for the game runner
            mcp_client = MultiServerMCPClient({
                "risk": {
                    "transport": "sse",
                    "url": "http://localhost:8080/mcp"
                }
            })
            
            # Connect to MCP server and get tools
            print("Connecting to MCP server for tournament game...")
            tools = await mcp_client.get_tools()
            print(f"Connected to MCP server, loaded {len(tools)} tools for tournament game")
            
            # Create the game runner with player configs
            game_runner = GameRunner(
                mcp_client=mcp_client,
                player_configs=processed_configs
            )
            
            # Create the game (this will create a new game and set game_runner.game_id)
            print(f"Creating tournament game with {len(processed_configs)} players...")
            create_success = await game_runner.create_game(len(processed_configs))
            if not create_success:
                print("Failed to create tournament game via GameRunner")
                return False
            
            # Get the game ID that was created
            game_id = game_runner.game_id
            print(f"Tournament game created with ID: {game_id}")
            
            # Set the game ID in tournament manager
            if self.tournament_manager:
                self.tournament_manager.set_game_id(game_id)
            
            # Initialize agents and have them join the game
            print("Initializing agents and joining tournament game...")
            init_success = await game_runner.initialize_agents()
            if not init_success:
                print("Failed to initialize agents for tournament game")
                return False
            
            print(f"All agents successfully joined tournament game {game_id}")
            
            # Track model assignments in the action tracker
            await self._track_tournament_model_assignments(game_id, processed_configs)
            
            # Add callback to handle game completion
            if hasattr(game_runner, 'add_game_completion_callback'):
                game_runner.add_game_completion_callback(self._on_tournament_game_complete)
            
            # Start the game in the background
            print("Starting tournament game loop...")
            asyncio.create_task(game_runner.run_game())
            
            print(f"Tournament game {game_id} started successfully with AI agents")
            return True
            
        except Exception as e:
            print(f"Error creating tournament game with configs: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def cleanup_tournament_games(self) -> None:
        """Clean up any existing tournament games before starting a new tournament."""
        try:
            # Get current tournament game ID if any
            current_game_id = None
            if self.tournament_manager:
                current_game_id = self.tournament_manager.get_current_game_id()
            
            # Remove tournament game from active games
            if current_game_id and current_game_id in self.games:
                print(f"Cleaning up tournament game: {current_game_id}")
                del self.games[current_game_id]
            
            # Clean up any other games that might be tournament-related
            # (This is a safety measure in case there are orphaned games)
            tournament_games = []
            for game_id, game in self.games.items():
                # Check if this might be a tournament game by looking at player names
                # Tournament games typically have character names as player names
                if hasattr(game, 'players') and len(game.players) == 4:  # Tournament games have 4 players
                    tournament_games.append(game_id)
            
            # Remove identified tournament games
            for game_id in tournament_games:
                print(f"Cleaning up potential tournament game: {game_id}")
                del self.games[game_id]
            
            # Reset tournament manager game ID
            if self.tournament_manager:
                self.tournament_manager.set_game_id(None)
            
            print(f"Tournament game cleanup completed. Removed {len(tournament_games) + (1 if current_game_id else 0)} games")
            
        except Exception as e:
            print(f"Error during tournament game cleanup: {e}")
    
    async def ensure_tournament_game_exists(self) -> bool:
        """Health check method to ensure tournament game exists when it should."""
        try:
            if not self.tournament_manager:
                return True  # No tournament mode, nothing to check
            
            tournament_status = self.tournament_manager.get_tournament_status()
            if not tournament_status:
                return True  # No active tournament
                
            # Check if we're in game phase but missing a game
            if (tournament_status.get('phase') == 'game' and 
                not tournament_status.get('game_id')):
                
                print("Health check: Tournament in game phase but no game_id found. Creating game...")
                await self._start_tournament_game()
                
                # Verify game was created
                updated_status = self.tournament_manager.get_tournament_status()
                if updated_status and updated_status.get('game_id'):
                    print("Health check: Game successfully created")
                    return True
                else:
                    print("Health check: Failed to create game")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in tournament game health check: {e}")
            import traceback
            traceback.print_exc()
            return False

# Global game manager instance
game_manager = GameManager()
