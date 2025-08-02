"""Tournament manager for handling tournament phases and lifecycle."""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.game_state import GamePhase
from tournament.character_manager import CharacterManager, TournamentCharacter
from tournament.model_manager import ModelManager

class TournamentPhase(Enum):
    """Tournament phases."""
    SUBMIT = "submit"
    VOTING = "voting"
    STARTING_GAME = "starting_game"
    GAME = "game"
    TOKEN_EXHAUSTED = "token_exhausted"
    END_SCREEN = "end_screen"

class TournamentManager:
    """Manages tournament lifecycle, phases, and timers."""
    
    def __init__(self, config: Dict):
        """
        Initialize tournament manager with configuration.
        
        Args:
            config: Dictionary containing tournament configuration
        """
        self.config = config
        self.character_manager = CharacterManager(config.get('db_path', './data/tournament.db'))
        self.model_manager = ModelManager(config.get('db_path', './data/tournament.db'))
        
        # Tournament state
        self.current_tournament_id = None
        self.current_phase = TournamentPhase.SUBMIT
        self.phase_start_time = None
        self.phase_timer_task = None
        self.auto_restart = config.get('auto_restart', False)
        
        # Phase durations (in seconds)
        self.phase_durations = {
            TournamentPhase.SUBMIT: config.get('submit_duration', 900),  # 15 minutes
            TournamentPhase.VOTING: config.get('voting_duration', 900),  # 15 minutes
            TournamentPhase.GAME: config.get('game_duration', 3600),     # 1 hour
            TournamentPhase.END_SCREEN: config.get('end_screen_duration', 300)  # 5 minutes
        }
        
        # Tournament settings
        self.max_submissions = config.get('max_submissions', 20)
        self.selected_characters_count = config.get('selected_characters', 4)
        
        # Game state
        self.current_game_id = None
        self.selected_characters = []
        self.game_result = None
        
        # Callbacks for phase changes
        self.phase_change_callbacks = []
        
    def add_phase_change_callback(self, callback):
        """Add a callback to be called when phase changes."""
        # Prevent duplicate callbacks
        if callback not in self.phase_change_callbacks:
            self.phase_change_callbacks.append(callback)
    
    async def start_tournament(self) -> str:
        """Start a new tournament cycle."""
        self.current_tournament_id = str(uuid.uuid4())
        self.current_phase = TournamentPhase.SUBMIT
        self.phase_start_time = datetime.now(timezone.utc)
        self.selected_characters = []
        self.game_result = None
        
        # Clear previous votes for this tournament
        self.character_manager.clear_votes_for_tournament(self.current_tournament_id)
        
        # Start phase timer
        await self._start_phase_timer()
        
        # Notify callbacks
        await self._notify_phase_change()
        
        return self.current_tournament_id
    
    async def _start_phase_timer(self):
        """Start the timer for the current phase."""
        if self.phase_timer_task:
            self.phase_timer_task.cancel()
        
        duration = self.phase_durations[self.current_phase]
        self.phase_timer_task = asyncio.create_task(self._phase_timer(duration))
    
    async def _phase_timer(self, duration: int):
        """Timer task for phase transitions."""
        try:
            await asyncio.sleep(duration)
            await self._advance_phase()
        except asyncio.CancelledError:
            pass  # Timer was cancelled
    
    async def _advance_phase(self):
        """Advance to the next phase."""
        if self.current_phase == TournamentPhase.SUBMIT:
            await self._transition_to_voting()
        elif self.current_phase == TournamentPhase.VOTING:
            await self._transition_to_game()
        elif self.current_phase == TournamentPhase.GAME:
            await self._transition_to_end_screen()
        elif self.current_phase == TournamentPhase.END_SCREEN:
            if self.auto_restart:
                await self.start_tournament()
            else:
                await self._stop_tournament()
    
    async def _transition_to_voting(self):
        """Transition from submit phase to voting phase."""
        self.current_phase = TournamentPhase.VOTING
        self.phase_start_time = datetime.now(timezone.utc)
        
        await self._start_phase_timer()
        await self._notify_phase_change()
    
    async def _transition_to_game(self):
        """Transition from voting phase to starting game phase."""
        # Select characters based on voting results
        self.selected_characters = self.character_manager.select_characters_for_game(
            self.current_tournament_id, 
            self.selected_characters_count
        )
        
        self.current_phase = TournamentPhase.STARTING_GAME
        self.phase_start_time = datetime.now(timezone.utc)
        
        # No timer for STARTING_GAME phase - it will advance when game is ready
        await self._notify_phase_change()
        
        # Start game creation process
        await self._create_tournament_game_with_validation()
    
    async def _transition_to_actual_game(self):
        """Transition from starting game phase to actual game phase."""
        self.current_phase = TournamentPhase.GAME
        self.phase_start_time = datetime.now(timezone.utc)
        
        await self._start_phase_timer()
        await self._notify_phase_change()
    
    async def _transition_to_end_screen(self):
        """Transition from game phase to end screen phase."""
        self.current_phase = TournamentPhase.END_SCREEN
        self.phase_start_time = datetime.now(timezone.utc)
        
        # IMPORTANT: Determine winner BEFORE forcing game completion
        # This ensures the game is still available for analysis
        if not self.game_result or 'winner_character_id' not in self.game_result:
            await self._determine_time_based_winner()
        
        # Force game completion after winner determination
        await self._force_game_completion()
        
        # Update character statistics
        await self._update_character_stats()
        
        # Only start timer if auto_restart is enabled
        # In tournament mode, we want to wait for manual restart button press
        if self.auto_restart:
            await self._start_phase_timer()
        
        await self._notify_phase_change()
    
    async def _stop_tournament(self):
        """Stop the tournament (when auto-restart is disabled)."""
        if self.phase_timer_task:
            self.phase_timer_task.cancel()
            self.phase_timer_task = None
        
        self.current_tournament_id = None
        await self._notify_phase_change()
    
    async def _update_character_stats(self):
        """Update character statistics after a game."""
        if not self.selected_characters:
            print("No selected characters to update stats for")
            return
        
        print(f"Updating character stats for {len(self.selected_characters)} characters")
        
        # Update play count for all selected characters
        for character in self.selected_characters:
            if character.id:
                print(f"Updating play count for character {character.name} (ID: {character.id})")
                self.character_manager.update_character_stats(character.id, played=True)
        
        # Update win count for winner if we have game result
        if self.game_result and 'winner_character_id' in self.game_result:
            winner_id = self.game_result['winner_character_id']
            print(f"Updating win count for winner character ID: {winner_id}")
            self.character_manager.update_character_stats(winner_id, won=True)
        else:
            print(f"No winner to update stats for. Game result: {self.game_result}")
    
    async def _notify_phase_change(self):
        """Notify all callbacks about phase change."""
        print(f"Notifying {len(self.phase_change_callbacks)} callbacks about phase change to {self.current_phase.value}")
        for i, callback in enumerate(self.phase_change_callbacks):
            try:
                print(f"Calling callback {i}: {callback}")
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.get_tournament_status())
                else:
                    callback(self.get_tournament_status())
                print(f"Callback {i} completed successfully")
            except Exception as e:
                print(f"Error in phase change callback {i}: {e}")
                import traceback
                traceback.print_exc()
    
    def get_tournament_status(self) -> Dict:
        """Get current tournament status."""
        time_remaining = 0
        phase_end_time = None
        
        # Only calculate time remaining if we have a timer running
        # In END_SCREEN phase with auto_restart=False, there's no timer
        if (self.phase_start_time and self.current_phase in self.phase_durations and 
            not (self.current_phase == TournamentPhase.END_SCREEN and not self.auto_restart)):
            
            # Use UTC for consistent time calculations
            now_utc = datetime.now(timezone.utc)
            
            # Ensure phase_start_time is timezone-aware
            if self.phase_start_time.tzinfo is None:
                # If phase_start_time is naive, assume it's UTC
                phase_start_utc = self.phase_start_time.replace(tzinfo=timezone.utc)
            else:
                phase_start_utc = self.phase_start_time
            
            elapsed = (now_utc - phase_start_utc).total_seconds()
            duration = self.phase_durations[self.current_phase]
            time_remaining = max(0, duration - elapsed)
            
            # Calculate absolute end time for client-side countdown
            phase_end_time = phase_start_utc + timedelta(seconds=duration)
        
        return {
            'tournament_id': self.current_tournament_id,
            'phase': self.current_phase.value if self.current_phase else None,
            'phase_start_time': self.phase_start_time.isoformat() if self.phase_start_time else None,
            'phase_end_time': phase_end_time.isoformat() if phase_end_time else None,
            'time_remaining': int(time_remaining),
            'selected_characters': [char.to_dict() for char in self.selected_characters],
            'game_id': self.current_game_id,
            'game_result': self.game_result,
            'auto_restart': self.auto_restart
        }
    
    def submit_character(self, name: str, model_name: str, temperature: float, 
                        personality: str, custom_instructions: str, 
                        submitted_by: str) -> Tuple[bool, str]:
        """
        Submit a character to the character library.
        Characters can now be submitted at any time, not just during the submit phase.
        Returns (success, message).
        """
        # Check submission limit
        all_characters = self.character_manager.get_all_characters()
        if len(all_characters) >= self.max_submissions:
            return False, f"Maximum of {self.max_submissions} characters allowed"
        
        character = TournamentCharacter(
            id=None,
            name=name,
            model_name=model_name,
            temperature=temperature,
            personality=personality,
            custom_instructions=custom_instructions,
            created_at=datetime.now(),
            submitted_by=submitted_by
        )
        
        return self.character_manager.submit_character(character)[:2]  # Return only success and message
    
    def submit_vote(self, character_id: int, voter_ip: str = None, voter_session: str = None) -> Tuple[bool, str]:
        """
        Submit a vote during the voting phase.
        Returns (success, message).
        """
        if self.current_phase != TournamentPhase.VOTING:
            return False, "Voting is only allowed during the voting phase"
        
        if not self.current_tournament_id:
            return False, "No active tournament"
        
        return self.character_manager.submit_vote(
            self.current_tournament_id, 
            character_id, 
            voter_ip, 
            voter_session
        )
    
    def get_characters_for_voting(self) -> List[Dict]:
        """Get all characters available for voting."""
        characters = self.character_manager.get_all_characters()
        return [char.to_dict() for char in characters]
    
    def get_vote_results(self) -> List[Dict]:
        """Get current voting results."""
        if not self.current_tournament_id:
            return []
        
        return self.character_manager.get_vote_results(self.current_tournament_id)
    
    def has_user_voted(self, voter_ip: str = None, voter_session: str = None) -> bool:
        """Check if a user has already voted in the current tournament."""
        if not self.current_tournament_id:
            return False
        
        return self.character_manager.has_user_voted(
            self.current_tournament_id, 
            voter_ip, 
            voter_session
        )
    
    def get_character_stats(self) -> List[Dict]:
        """Get character statistics."""
        return self.character_manager.get_character_stats()
    
    def set_game_id(self, game_id: str):
        """Set the current game ID."""
        self.current_game_id = game_id
    
    def get_current_game_id(self) -> Optional[str]:
        """Get the current game ID."""
        return self.current_game_id
    
    def set_game_result(self, result: Dict):
        """Set the game result."""
        self.game_result = result
        
        # Try to map winner to character ID
        if 'winner_player_name' in result and self.selected_characters:
            for character in self.selected_characters:
                if character.name == result['winner_player_name']:
                    result['winner_character_id'] = character.id
                    print(f"Mapped winner {result['winner_player_name']} to character ID {character.id}")
                    break
            
            # If we couldn't find the character ID, log it
            if 'winner_character_id' not in result:
                print(f"Warning: Could not map winner '{result['winner_player_name']}' to character ID")
                print(f"Available characters: {[char.name for char in self.selected_characters]}")
        
        print(f"Tournament game result set: {result.get('winner_player_name', 'Unknown')} won by {result.get('winner_determination_method', 'unknown method')}")
    
    def capture_natural_game_end(self, game_id: str, winner_info: Dict):
        """Capture game result when a game ends naturally (not forced by tournament timer)."""
        if game_id != self.current_game_id:
            print(f"Ignoring game end for {game_id} - not current tournament game ({self.current_game_id})")
            return
        
        print(f"Capturing natural game end for tournament game {game_id}")
        
        # Create comprehensive game result from provided winner_info
        game_result = {
            'winner_player_id': winner_info.get('winner_player_id', 'unknown'),
            'winner_player_name': winner_info.get('winner_player_name', 'Unknown'),
            'completion_reason': winner_info.get('completion_reason', 'victory'),
            'winner_determination_method': winner_info.get('winner_determination_method', 'victory'),
            'winner_criteria': winner_info.get('winner_criteria', 'Complete victory'),
            'total_turns': winner_info.get('total_turns', 0),
            'total_duration_seconds': winner_info.get('total_duration_seconds', 0),
            'total_actions': winner_info.get('total_actions', 0),
            'battles_fought': winner_info.get('battles_fought', 0),
            'final_scores': winner_info.get('final_scores', []),
            'game_phase_at_end': winner_info.get('game_phase_at_end', 'completed')
        }
        
        # Try to enhance with comprehensive stats from database if available
        try:
            from persistence.action_tracker import get_action_tracker
            
            tracker = get_action_tracker()
            if tracker:
                # Check if comprehensive stats are already stored in database
                completion_stats = tracker.get_game_completion_stats(game_id)
                if completion_stats:
                    # Merge database stats with winner info (database stats take precedence for comprehensive metrics)
                    game_result.update({
                        'total_actions': completion_stats.get('total_actions', game_result.get('total_actions', 0)),
                        'total_duration_seconds': completion_stats.get('total_duration_seconds', game_result.get('total_duration_seconds', 0)),
                        'battles_fought': completion_stats.get('battles_fought', game_result.get('battles_fought', 0)),
                        'diplomatic_messages': completion_stats.get('diplomatic_messages', 0),
                        'territories_conquered': completion_stats.get('territories_conquered', 0),
                        'successful_actions': completion_stats.get('successful_actions', 0),
                        'failed_actions': completion_stats.get('failed_actions', 0),
                        'action_type_breakdown': completion_stats.get('action_type_breakdown', {}),
                        'models_used': completion_stats.get('models_used', []),
                        'avg_decision_time_seconds': completion_stats.get('avg_decision_time_seconds', 0.0),
                        'player_statistics': completion_stats.get('player_statistics', {})
                    })
                    print(f"Enhanced natural game end with comprehensive stats: {completion_stats.get('total_actions', 0)} actions, {completion_stats.get('battles_fought', 0)} battles")
                else:
                    print("No comprehensive stats found in database for natural game end")
        except Exception as e:
            print(f"Error retrieving comprehensive stats for natural game end: {e}")
        
        # Set the result using the existing method (which handles character ID mapping)
        self.set_game_result(game_result)
    
    async def force_advance_phase(self) -> bool:
        """Force advance to the next phase (admin function)."""
        if self.phase_timer_task:
            self.phase_timer_task.cancel()
        
        await self._advance_phase()
        return True
    
    def get_selected_characters_for_game(self) -> List[Dict]:
        """Get the characters selected for the current game."""
        if self.current_phase not in [TournamentPhase.STARTING_GAME, TournamentPhase.GAME]:
            return []
        
        print(f"\n=== TOURNAMENT MODEL ASSIGNMENT ===")
        print(f"Tournament ID: {self.current_tournament_id}")
        print(f"Selected characters: {[char.name for char in self.selected_characters]}")
        
        # Get available models from database
        available_models = self._get_available_models()
        print(f"Available models from database: {available_models}")
        
        # Use weighted model selection to bias toward underrepresented models
        selected_models = self._select_models_weighted(available_models, len(self.selected_characters))
        print(f"Selected models after weighting: {selected_models}")
        
        # Convert TournamentCharacter objects to PlayerConfig format
        player_configs = []
        print(f"\nCharacter -> Model assignments:")
        for i, character in enumerate(self.selected_characters):
            # Get API credentials from environment
            api_key = os.getenv('RISK_API_KEY', os.getenv('OPENAI_API_KEY', ''))
            base_url = os.getenv('RISK_BASE_URL', os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
            
            # Assign the weighted-selected model
            assigned_model = selected_models[i] if i < len(selected_models) else available_models[0]
            print(f"  - {character.name} -> {assigned_model} (temp: {character.temperature})")
            
            config = {
                'name': character.name,
                'model_name': assigned_model,
                'temperature': character.temperature,
                'custom_instructions': character.custom_instructions,
                'api_key': api_key,
                'base_url': base_url
            }
            player_configs.append(config)
        
        print(f"=== END MODEL ASSIGNMENT ===\n")
        return player_configs
    
    def is_tournament_active(self) -> bool:
        """Check if a tournament is currently active."""
        return self.current_tournament_id is not None
    
    def get_phase_progress(self) -> float:
        """Get the progress of the current phase (0.0 to 1.0)."""
        if not self.phase_start_time or self.current_phase not in self.phase_durations:
            return 0.0
        
        # Use UTC for consistent time calculations
        now_utc = datetime.now(timezone.utc)
        
        # Ensure phase_start_time is timezone-aware
        if self.phase_start_time.tzinfo is None:
            phase_start_utc = self.phase_start_time.replace(tzinfo=timezone.utc)
        else:
            phase_start_utc = self.phase_start_time
        
        elapsed = (now_utc - phase_start_utc).total_seconds()
        duration = self.phase_durations[self.current_phase]
        
        return min(1.0, elapsed / duration)
    
    def get_time_remaining_formatted(self) -> str:
        """Get formatted time remaining in current phase."""
        status = self.get_tournament_status()
        time_remaining = status.get('time_remaining', 0)
        
        if time_remaining <= 0:
            return "00:00"
        
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)
        
        return f"{minutes:02d}:{seconds:02d}"
    
    def _get_available_models(self) -> List[str]:
        """Get available AI models from tournament database."""
        try:
            # Get active models from the database
            active_models = self.model_manager.get_active_models()
            
            if active_models:
                model_names = [model.name for model in active_models]
                print(f"Available models from database: {model_names}")
                return model_names
            else:
                print("No active models found in database, using fallback")
                # Fallback to environment if no models in database
                default_model = os.getenv('RISK_MODEL_NAME', 'gpt-3.5-turbo')
                return [default_model]
            
        except Exception as e:
            print(f"Error loading models from database: {e}")
            import traceback
            traceback.print_exc()
            # Ultimate fallback
            default_model = os.getenv('RISK_MODEL_NAME', 'gpt-3.5-turbo')
            return [default_model]
    
    def _select_models_weighted(self, available_models: List[str], num_needed: int) -> List[str]:
        """
        Select models using weighted selection to bias toward underrepresented models.
        
        Args:
            available_models: List of available model names
            num_needed: Number of models to select
            
        Returns:
            List of selected model names (no duplicates, ensuring one model per player)
        """
        import random
        
        try:
            # Get model usage statistics from the last 30 days
            from persistence.action_tracker import get_action_tracker
            
            tracker = get_action_tracker()
            if not tracker:
                print("Action tracker not available, falling back to random selection")
                return self._select_models_random(available_models, num_needed)
            
            usage_stats = tracker.get_model_usage_stats(days=30)
            
            # If no usage data, fall back to random selection
            if not usage_stats:
                print("No model usage data available, using random selection")
                return self._select_models_random(available_models, num_needed)
            
            # Calculate weights for each available model
            model_weights = {}
            total_usage = sum(usage_stats.values())
            # IMPORTANT: Calculate average based on available models, not just models with usage stats
            # This ensures unused models (like new models) get proper underrepresentation bonuses
            avg_usage = total_usage / len(available_models) if available_models else 0
            
            # Bias factor (how much to favor underrepresented models)
            # Higher values = stronger bias toward unused models
            bias_factor = float(os.getenv('TOURNAMENT_MODEL_BIAS_FACTOR', '2.0'))
            
            print(f"Model usage stats (last 30 days): {usage_stats}")
            print(f"Total usage: {total_usage}, Average: {avg_usage:.1f}, Bias factor: {bias_factor}")
            
            for model in available_models:
                usage_count = usage_stats.get(model, 0)
                
                # Base weight
                base_weight = 1.0
                
                # Underrepresentation bonus
                if avg_usage > 0:
                    underrepresentation_bonus = max(0, (avg_usage - usage_count) / avg_usage)
                else:
                    # If no average usage, treat all as equally underrepresented
                    underrepresentation_bonus = 1.0
                
                # Final weight calculation
                final_weight = base_weight + (underrepresentation_bonus * bias_factor)
                model_weights[model] = final_weight
                
                print(f"Model {model}: usage={usage_count}, bonus={underrepresentation_bonus:.2f}, weight={final_weight:.2f}")
            
            # Select models using weighted random selection without replacement
            selected_models = []
            remaining_models = available_models.copy()
            remaining_weights = model_weights.copy()
            
            for _ in range(min(num_needed, len(available_models))):
                if not remaining_models:
                    break
                
                # Create weighted list for selection
                weights_list = [remaining_weights[m] for m in remaining_models]
                
                # Use weighted random choice
                selected_model = random.choices(remaining_models, weights=weights_list, k=1)[0]
                selected_models.append(selected_model)
                
                # Remove selected model to ensure no duplicates
                remaining_models.remove(selected_model)
                del remaining_weights[selected_model]
                
                print(f"Selected model {selected_model} (weight: {model_weights[selected_model]:.2f})")
            
            # If we need more models than available, cycle through them
            while len(selected_models) < num_needed:
                selected_models.extend(available_models[:num_needed - len(selected_models)])
            
            print(f"Final model selection: {selected_models}")
            return selected_models[:num_needed]
            
        except Exception as e:
            print(f"Error in weighted model selection: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to random selection
            return self._select_models_random(available_models, num_needed)
    
    def _select_models_random(self, available_models: List[str], num_needed: int) -> List[str]:
        """
        Fallback random model selection.
        
        Args:
            available_models: List of available model names
            num_needed: Number of models to select
            
        Returns:
            List of randomly selected model names (no duplicates)
        """
        import random
        
        if len(available_models) >= num_needed:
            # Random selection without replacement
            return random.sample(available_models, num_needed)
        else:
            # Need more models than available, cycle through them
            selected = available_models.copy()
            while len(selected) < num_needed:
                selected.extend(available_models[:num_needed - len(selected)])
            return selected[:num_needed]
    
    async def _determine_time_based_winner(self):
        """Determine winner based on current game state when time expires."""
        if not self.current_game_id or not self.selected_characters:
            print("No game ID or selected characters for time-based winner determination")
            await self._create_fallback_winner_result("No game ID or selected characters")
            return
        
        try:
            print(f"Determining time-based winner for game {self.current_game_id}")
            
            # Import game manager to access current game state
            from utils.game_manager import game_manager
            from persistence.action_tracker import get_action_tracker
            
            # First try to get the game from active games
            game = game_manager.get_game(self.current_game_id)
            
            # If game not found in active games, try to get completion stats from database
            if not game:
                print(f"Game {self.current_game_id} not found in active games, checking database...")
                
                tracker = get_action_tracker()
                if tracker:
                    # Try to get game completion stats from database
                    completion_stats = tracker.get_game_completion_stats(self.current_game_id)
                    if completion_stats and completion_stats.get('winner_player_id'):
                        print(f"Found winner in database: {completion_stats.get('winner_player_name')}")
                        
                        # Use database winner information
                        winner_player_name = completion_stats.get('winner_player_name', 'Unknown')
                        winner_character_id = None
                        
                        # Map winner to character ID
                        for character in self.selected_characters:
                            if character.name == winner_player_name:
                                winner_character_id = character.id
                                break
                        
                        self.game_result = {
                            'winner_player_id': completion_stats.get('winner_player_id'),
                            'winner_player_name': winner_player_name,
                            'winner_character_id': winner_character_id,
                            'completion_reason': completion_stats.get('completion_reason', 'victory'),
                            'winner_determination_method': completion_stats.get('winner_determination_method', 'victory'),
                            'winner_criteria': completion_stats.get('winner_criteria', 'Game completed successfully'),
                            'total_turns': completion_stats.get('total_turns', 0),
                            'total_duration_seconds': completion_stats.get('total_duration_seconds', 0),
                            'game_phase_at_end': completion_stats.get('final_phase', 'completed')
                        }
                        
                        print(f"Winner determined from database: {winner_player_name}")
                        return
                
                # If still no winner found, create fallback
                print(f"Game {self.current_game_id} not found in active games or database")
                await self._create_fallback_winner_result("Game completed but not found in memory")
                return
            
            # Game found in active games - analyze current state
            winner_analysis = self._analyze_game_state_for_winner(game)
            
            if winner_analysis:
                # Create game result with time-based winner
                self.game_result = {
                    'winner_player_id': winner_analysis['player_id'],
                    'winner_player_name': winner_analysis['player_name'],
                    'winner_character_id': winner_analysis['character_id'],
                    'completion_reason': 'time_expired',
                    'winner_determination_method': winner_analysis['method'],
                    'winner_criteria': winner_analysis['criteria'],
                    'final_scores': winner_analysis['scores'],
                    'total_turns': getattr(game, 'turn_number', 0),
                    'game_phase_at_end': game.game_phase.value if hasattr(game.game_phase, 'value') else str(game.game_phase)
                }
                
                print(f"Time-based winner determined: {winner_analysis['player_name']} by {winner_analysis['method']}")
                print(f"Criteria: {winner_analysis['criteria']}")
            else:
                print("Could not determine winner from game state, creating fallback")
                await self._create_fallback_winner_result("Unable to analyze game state")
                
        except Exception as e:
            print(f"Error determining time-based winner: {e}")
            import traceback
            traceback.print_exc()
            await self._create_fallback_winner_result(f"Error: {str(e)}")
    
    def _analyze_game_state_for_winner(self, game):
        """Analyze current game state to determine the leading player."""
        try:
            if not hasattr(game, 'players') or not game.players:
                return None
            
            player_scores = []
            
            for player_id, player in game.players.items():
                if getattr(player, 'is_eliminated', False):
                    continue  # Skip eliminated players
                
                # Calculate score based on multiple criteria
                territory_count = len(getattr(player, 'territories', []))
                army_count = getattr(player, 'army_count', 0)
                
                # Calculate continent bonuses (if territory data is available)
                continent_bonus = 0
                if hasattr(game, 'territory_data') and hasattr(player, 'territories'):
                    continent_bonus = self._calculate_continent_bonus(player.territories, game.territory_data)
                
                # Total score calculation (weighted)
                total_score = (territory_count * 10) + army_count + (continent_bonus * 20)
                
                # Find corresponding character
                character_id = None
                character_name = player.name
                for character in self.selected_characters:
                    if character.name == player.name:
                        character_id = character.id
                        break
                
                player_scores.append({
                    'player_id': player_id,
                    'player_name': player.name,
                    'character_id': character_id,
                    'territory_count': territory_count,
                    'army_count': army_count,
                    'continent_bonus': continent_bonus,
                    'total_score': total_score
                })
            
            if not player_scores:
                return None
            
            # Sort by total score (descending)
            player_scores.sort(key=lambda x: x['total_score'], reverse=True)
            
            winner = player_scores[0]
            
            # Determine the method used for winner selection
            if len(player_scores) == 1:
                method = "sole_survivor"
                criteria = f"Only remaining player"
            elif winner['total_score'] > player_scores[1]['total_score']:
                if winner['territory_count'] > player_scores[1]['territory_count']:
                    method = "territory_control"
                    criteria = f"{winner['territory_count']} territories vs {player_scores[1]['territory_count']}"
                elif winner['continent_bonus'] > player_scores[1]['continent_bonus']:
                    method = "continent_control"
                    criteria = f"{winner['continent_bonus']} continent bonuses vs {player_scores[1]['continent_bonus']}"
                else:
                    method = "army_strength"
                    criteria = f"{winner['army_count']} armies vs {player_scores[1]['army_count']}"
            else:
                # Tie - use random selection but note it was tied
                method = "tied_random_selection"
                criteria = f"Tied with {len([p for p in player_scores if p['total_score'] == winner['total_score']])} players"
            
            return {
                'player_id': winner['player_id'],
                'player_name': winner['player_name'],
                'character_id': winner['character_id'],
                'method': method,
                'criteria': criteria,
                'scores': player_scores
            }
            
        except Exception as e:
            print(f"Error analyzing game state: {e}")
            return None
    
    def _calculate_continent_bonus(self, player_territories, territory_data):
        """Calculate continent control bonus for a player."""
        try:
            if not territory_data or 'continents' not in territory_data:
                return 0
            
            continent_bonus = 0
            continents = territory_data['continents']
            
            for continent_name, continent_info in continents.items():
                continent_territories = continent_info.get('territories', [])
                if continent_territories and all(territory in player_territories for territory in continent_territories):
                    continent_bonus += continent_info.get('bonus', 0)
            
            return continent_bonus
            
        except Exception as e:
            print(f"Error calculating continent bonus: {e}")
            return 0
    
    async def _force_game_completion(self):
        """Force the current tournament game to complete and store statistics."""
        if not self.current_game_id:
            print("No current game to force completion")
            return
        
        try:
            print(f"Forcing completion of tournament game {self.current_game_id}")
            
            # Import required modules
            from utils.game_manager import game_manager
            from persistence.action_tracker import get_action_tracker
            
            # Get the game instance
            game = game_manager.get_game(self.current_game_id)
            if not game:
                print(f"Game {self.current_game_id} not found in game manager")
                return
            
            # Get action tracker
            tracker = get_action_tracker()
            if not tracker:
                print("Action tracker not available")
                return
            
            # Determine winner information
            winner_player_id = None
            winner_player_name = None
            completion_reason = "tournament_time_expired"
            
            if self.game_result:
                winner_player_id = self.game_result.get('winner_player_id')
                winner_player_name = self.game_result.get('winner_player_name')
                completion_reason = self.game_result.get('completion_reason', 'tournament_time_expired')
            
            # Force the game to finish with comprehensive statistics
            success = tracker.finish_game_with_stats(
                game_id=self.current_game_id,
                winner_player_id=winner_player_id,
                winner_player_name=winner_player_name,
                completion_reason=completion_reason,
                status='completed'
            )
            
            if success:
                print(f"Successfully stored completion statistics for game {self.current_game_id}")
                
                # IMPORTANT: Retrieve the comprehensive stats and merge into game_result
                completion_stats = tracker.get_game_completion_stats(self.current_game_id)
                if completion_stats and self.game_result:
                    # Merge comprehensive stats into existing game_result
                    self.game_result.update({
                        'total_actions': completion_stats.get('total_actions', 0),
                        'total_duration_seconds': completion_stats.get('total_duration_seconds', 0),
                        'battles_fought': completion_stats.get('battles_fought', 0),
                        'diplomatic_messages': completion_stats.get('diplomatic_messages', 0),
                        'territories_conquered': completion_stats.get('territories_conquered', 0),
                        'successful_actions': completion_stats.get('successful_actions', 0),
                        'failed_actions': completion_stats.get('failed_actions', 0),
                        'action_type_breakdown': completion_stats.get('action_type_breakdown', {}),
                        'models_used': completion_stats.get('models_used', []),
                        'avg_decision_time_seconds': completion_stats.get('avg_decision_time_seconds', 0.0),
                        'player_statistics': completion_stats.get('player_statistics', {})
                    })
                    print(f"Enhanced game result with comprehensive stats: {completion_stats.get('total_actions', 0)} actions, {completion_stats.get('battles_fought', 0)} battles, {completion_stats.get('total_duration_seconds', 0)}s duration")
                elif completion_stats:
                    print("Retrieved completion stats but no existing game_result to merge with")
                else:
                    print("Failed to retrieve completion stats from database")
                
                # Remove the game from active games to stop background processes
                if self.current_game_id in game_manager.games:
                    print(f"Removing game {self.current_game_id} from active games")
                    del game_manager.games[self.current_game_id]
                    
            else:
                print(f"Failed to store completion statistics for game {self.current_game_id}")
                
        except Exception as e:
            print(f"Error forcing game completion: {e}")
            import traceback
            traceback.print_exc()
    
    async def _create_tournament_game_with_validation(self):
        """Create and validate tournament game with retry logic."""
        max_attempts = int(os.getenv('TOURNAMENT_MAX_CREATION_ATTEMPTS', '3'))
        timeout = int(os.getenv('TOURNAMENT_GAME_CREATION_TIMEOUT', '180'))  # 3 minutes
        
        print(f"Starting tournament game creation with validation (max {max_attempts} attempts, {timeout}s timeout)")
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n=== GAME CREATION ATTEMPT {attempt}/{max_attempts} ===")
            
            try:
                # Step 1: Cleanup all existing games
                print("Step 1: Cleaning up existing games...")
                await self._cleanup_all_games()
                print("✓ Game cleanup completed")
                
                # Step 2: Create new tournament game
                print("Step 2: Creating new tournament game...")
                success = await self._create_tournament_game_internal()
                if not success:
                    print("✗ Failed to create tournament game")
                    continue
                print(f"✓ Tournament game created: {self.current_game_id}")
                
                # Step 3: Validate game and players
                print("Step 3: Validating game and players...")
                validation_success = await self._validate_tournament_game()
                if not validation_success:
                    print("✗ Game validation failed")
                    await self._cleanup_failed_game()
                    continue
                print("✓ Game validation successful - all 4 players loaded")
                
                # Step 4: Transition to actual game phase
                print("Step 4: Transitioning to game phase...")
                await self._transition_to_actual_game()
                print("✓ Successfully transitioned to game phase")
                
                print(f"=== GAME CREATION SUCCESS (attempt {attempt}) ===\n")
                return True
                
            except asyncio.TimeoutError:
                print(f"✗ Game creation attempt {attempt} timed out")
                await self._cleanup_failed_game()
                
            except Exception as e:
                print(f"✗ Game creation attempt {attempt} failed: {e}")
                import traceback
                traceback.print_exc()
                await self._cleanup_failed_game()
        
        # All attempts failed - implement continuous retry with backoff
        print(f"\n=== ALL GAME CREATION ATTEMPTS FAILED ===")
        print("Implementing continuous retry with exponential backoff...")
        await self._implement_continuous_retry()
        return False
    
    async def _cleanup_all_games(self):
        """Clean up all existing games to ensure fresh start."""
        try:
            from utils.game_manager import game_manager
            
            # Get list of all active games
            active_games = list(game_manager.games.keys())
            
            if not active_games:
                print("No active games to clean up")
                return
            
            print(f"Cleaning up {len(active_games)} active games: {active_games}")
            
            # Kill all active games
            for game_id in active_games:
                try:
                    print(f"Terminating game {game_id}")
                    if game_id in game_manager.games:
                        del game_manager.games[game_id]
                        
                    # Track termination in database
                    from persistence.action_tracker import get_action_tracker
                    tracker = get_action_tracker()
                    if tracker:
                        tracker.track_game_end(
                            game_id=game_id,
                            completion_reason="tournament_cleanup",
                            winner_player_id=None,
                            winner_player_name=None
                        )
                        
                except Exception as e:
                    print(f"Error terminating game {game_id}: {e}")
            
            print("Game cleanup completed")
            
        except Exception as e:
            print(f"Error during game cleanup: {e}")
            raise
    
    async def _create_tournament_game_internal(self):
        """Create the actual tournament game."""
        try:
            from utils.game_manager import game_manager
            
            # Get player configurations for selected characters
            player_configs = self.get_selected_characters_for_game()
            
            if len(player_configs) != 4:
                print(f"Error: Expected 4 player configs, got {len(player_configs)}")
                return False
            
            print(f"Creating tournament game with {len(player_configs)} players")
            
            # Use game manager to create tournament game
            success = await game_manager._create_tournament_game_with_configs(player_configs)
            
            if success:
                # Get the created game ID from game manager
                self.current_game_id = game_manager.tournament_manager.get_current_game_id()
                print(f"Tournament game created successfully: {self.current_game_id}")
                return True
            else:
                print("Failed to create tournament game")
                return False
                
        except Exception as e:
            print(f"Error creating tournament game: {e}")
            return False
    
    async def _validate_tournament_game(self):
        """Validate that the tournament game has all players properly loaded."""
        if not self.current_game_id:
            print("No game ID to validate")
            return False
        
        try:
            from utils.game_manager import game_manager
            
            # Check if game exists in game manager
            game = game_manager.get_game(self.current_game_id)
            if not game:
                print(f"Game {self.current_game_id} not found in game manager")
                return False
            
            # Check if game has exactly 4 players
            if not hasattr(game, 'players') or len(game.players) != 4:
                print(f"Game has {len(game.players) if hasattr(game, 'players') else 0} players, expected 4")
                return False
            
            # Check if all players are properly initialized
            for player_id, player in game.players.items():
                if not hasattr(player, 'name') or not player.name:
                    print(f"Player {player_id} is not properly initialized")
                    return False
                
                # Verify player name matches selected characters
                player_found = False
                for character in self.selected_characters:
                    if character.name == player.name:
                        player_found = True
                        break
                
                if not player_found:
                    print(f"Player {player.name} not found in selected characters")
                    return False
            
            # Check if game is in correct initial phase
            if not hasattr(game, 'game_phase'):
                print("Game does not have game_phase attribute")
                return False
            
            print(f"Game validation successful: {len(game.players)} players properly loaded")
            return True
            
        except Exception as e:
            print(f"Error validating tournament game: {e}")
            return False
    
    async def _cleanup_failed_game(self):
        """Clean up a failed game creation attempt."""
        if not self.current_game_id:
            return
        
        try:
            print(f"Cleaning up failed game: {self.current_game_id}")
            
            from utils.game_manager import game_manager
            
            # Remove from active games
            if self.current_game_id in game_manager.games:
                del game_manager.games[self.current_game_id]
            
            # Track cleanup in database
            from persistence.action_tracker import get_action_tracker
            tracker = get_action_tracker()
            if tracker:
                tracker.track_game_end(
                    game_id=self.current_game_id,
                    completion_reason="creation_failed",
                    winner_player_id=None,
                    winner_player_name=None
                )
            
            # Clear game ID
            self.current_game_id = None
            
            print("Failed game cleanup completed")
            
        except Exception as e:
            print(f"Error cleaning up failed game: {e}")
    
    async def _transition_back_to_voting_with_error(self):
        """Transition back to voting phase when game creation fails."""
        try:
            print("Transitioning back to voting phase due to game creation failure")
            
            # Reset phase to voting
            self.current_phase = TournamentPhase.VOTING
            self.phase_start_time = datetime.now(timezone.utc)
            
            # Extend voting time to give more opportunity
            extended_duration = int(os.getenv('TOURNAMENT_VOTING_EXTENDED_DURATION', '600'))  # 10 minutes
            self.phase_durations[TournamentPhase.VOTING] = extended_duration
            
            # Start timer for extended voting
            await self._start_phase_timer()
            
            # Notify about the error and extended voting
            await self._notify_phase_change()
            
            print(f"Returned to voting phase with extended duration: {extended_duration}s")
            
        except Exception as e:
            print(f"Error transitioning back to voting: {e}")
    
    async def _implement_continuous_retry(self):
        """Implement continuous retry with exponential backoff for game creation."""
        print("Starting continuous retry mechanism...")
        
        # Start background task for continuous retries
        retry_task = asyncio.create_task(self._continuous_retry_loop())
        
        # Don't await the task - let it run in background
        # The retry loop will handle transitioning to game phase when successful
        print("Continuous retry task started in background")
    
    async def _continuous_retry_loop(self):
        """Background loop for continuous game creation retries."""
        retry_count = 0
        base_delay = float(os.getenv('TOURNAMENT_RETRY_BASE_DELAY', '10.0'))  # 10 seconds
        max_delay = float(os.getenv('TOURNAMENT_RETRY_MAX_DELAY', '300.0'))  # 5 minutes
        max_retries = int(os.getenv('TOURNAMENT_RETRY_MAX_ATTEMPTS', '0'))  # 0 = infinite
        
        print(f"Starting continuous retry loop: base_delay={base_delay}s, max_delay={max_delay}s, max_retries={max_retries}")
        
        while True:
            retry_count += 1
            
            # Check if we've exceeded max retries (if set)
            if max_retries > 0 and retry_count > max_retries:
                print(f"Exceeded maximum retry attempts ({max_retries}), giving up")
                # Could implement fallback behavior here
                break
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
            
            print(f"\n=== CONTINUOUS RETRY ATTEMPT {retry_count} ===")
            print(f"Waiting {delay}s before retry...")
            
            try:
                await asyncio.sleep(delay)
                
                # Check if we're still in the starting game phase
                if self.current_phase != TournamentPhase.STARTING_GAME:
                    print(f"Phase changed to {self.current_phase.value}, stopping retry loop")
                    break
                
                print(f"Attempting game creation (retry {retry_count})...")
                
                # Step 1: Cleanup all existing games
                print("Step 1: Cleaning up existing games...")
                await self._cleanup_all_games()
                print("✓ Game cleanup completed")
                
                # Step 2: Create new tournament game
                print("Step 2: Creating new tournament game...")
                success = await self._create_tournament_game_internal()
                if not success:
                    print("✗ Failed to create tournament game")
                    continue
                print(f"✓ Tournament game created: {self.current_game_id}")
                
                # Step 3: Validate game and players
                print("Step 3: Validating game and players...")
                validation_success = await self._validate_tournament_game()
                if not validation_success:
                    print("✗ Game validation failed")
                    await self._cleanup_failed_game()
                    continue
                print("✓ Game validation successful - all 4 players loaded")
                
                # Step 4: Transition to actual game phase
                print("Step 4: Transitioning to game phase...")
                await self._transition_to_actual_game()
                print("✓ Successfully transitioned to game phase")
                
                print(f"=== CONTINUOUS RETRY SUCCESS (attempt {retry_count}) ===\n")
                break  # Success! Exit the retry loop
                
            except asyncio.CancelledError:
                print("Continuous retry loop cancelled")
                break
                
            except Exception as e:
                print(f"✗ Retry attempt {retry_count} failed: {e}")
                import traceback
                traceback.print_exc()
                await self._cleanup_failed_game()
                
                # Continue to next retry
                continue
        
        print("Continuous retry loop ended")
    
    async def _create_fallback_winner_result(self, reason):
        """Create a fallback winner result when normal determination fails."""
        try:
            if not self.selected_characters:
                print("No selected characters for fallback winner")
                return
            
            # Pick a random character as winner
            import random
            winner_character = random.choice(self.selected_characters)
            
            self.game_result = {
                'winner_player_id': 'unknown',
                'winner_player_name': winner_character.name,
                'winner_character_id': winner_character.id,
                'completion_reason': 'time_expired_fallback',
                'winner_determination_method': 'random_fallback',
                'winner_criteria': f'Random selection due to: {reason}',
                'fallback_reason': reason,
                'total_turns': 0,
                'game_phase_at_end': 'unknown'
            }
            
            print(f"Fallback winner selected: {winner_character.name} (reason: {reason})")
            
        except Exception as e:
            print(f"Error creating fallback winner: {e}")
    
    async def handle_token_exhaustion(self):
        """Handle token exhaustion by pausing the game and transitioning to TOKEN_EXHAUSTED phase."""
        print("Token exhaustion detected - pausing tournament game")
        
        try:
            # Cancel any running phase timer
            if self.phase_timer_task:
                self.phase_timer_task.cancel()
            
            # Store the current game state for potential resumption
            if self.current_game_id:
                print(f"Storing paused game state for game {self.current_game_id}")
                # The game will remain in memory but stop processing
                
            # Transition to token exhausted phase
            self.current_phase = TournamentPhase.TOKEN_EXHAUSTED
            self.phase_start_time = datetime.now(timezone.utc)
            
            # No timer for TOKEN_EXHAUSTED phase - it will remain until admin restart
            await self._notify_phase_change()
            
            print("Tournament transitioned to TOKEN_EXHAUSTED phase")
            
        except Exception as e:
            print(f"Error handling token exhaustion: {e}")
            import traceback
            traceback.print_exc()
    
    async def restart_from_token_exhaustion(self):
        """Restart the tournament from token exhausted state (admin function)."""
        if self.current_phase != TournamentPhase.TOKEN_EXHAUSTED:
            return False, "Tournament is not in token exhausted state"
        
        try:
            print("Admin restarting tournament from token exhaustion")
            
            # Check if we have a paused game to resume
            if self.current_game_id:
                print(f"Attempting to resume game {self.current_game_id}")
                
                # Try to resume the existing game
                from utils.game_manager import game_manager
                game = game_manager.get_game(self.current_game_id)
                
                if game and hasattr(game, 'players') and len(game.players) == 4:
                    # Game is still valid, resume it
                    print("Resuming existing tournament game")
                    self.current_phase = TournamentPhase.GAME
                    self.phase_start_time = datetime.now(timezone.utc)
                    
                    # Resume the game timer
                    await self._start_phase_timer()
                    await self._notify_phase_change()
                    
                    return True, "Tournament resumed successfully"
                else:
                    print("Existing game not found or invalid, creating new game")
                    # Fall through to create new game
            
            # Create a new game with the same characters
            if not self.selected_characters:
                return False, "No selected characters available for restart"
            
            # Transition back to starting game phase
            self.current_phase = TournamentPhase.STARTING_GAME
            self.phase_start_time = datetime.now(timezone.utc)
            
            await self._notify_phase_change()
            
            # Start game creation process
            success = await self._create_tournament_game_with_validation()
            
            if success:
                return True, "Tournament restarted with new game successfully"
            else:
                # If game creation fails, stay in token exhausted state
                self.current_phase = TournamentPhase.TOKEN_EXHAUSTED
                await self._notify_phase_change()
                return False, "Failed to create new game - remaining in token exhausted state"
                
        except Exception as e:
            print(f"Error restarting from token exhaustion: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Error restarting tournament: {str(e)}"

    async def cleanup(self):
        """Cleanup tournament manager resources."""
        if self.phase_timer_task:
            self.phase_timer_task.cancel()
            try:
                await self.phase_timer_task
            except asyncio.CancelledError:
                pass
