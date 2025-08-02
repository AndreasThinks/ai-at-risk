import json
import random
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .player import Player
from .territory import TerritoryManager
from .combat import CombatEngine, CombatResult
from .cards import CardManager
from .notepad import NotepadManager

class GamePhase(Enum):
    SETUP = "setup"
    REINFORCEMENT = "reinforcement"
    ATTACK = "attack"
    FORTIFY = "fortify"
    GAME_OVER = "game_over"
    
    # Tournament phases
    TOURNAMENT_SUBMIT = "tournament_submit"
    TOURNAMENT_VOTING = "tournament_voting"
    TOURNAMENT_END_SCREEN = "tournament_end_screen"

class GameState:
    """Main game state manager for Risk."""
    
    def __init__(self, game_id: str, num_players: int, territory_data: dict):
        self.game_id = game_id
        self.num_players = num_players
        self.players: Dict[str, Player] = {}
        self.player_order: List[str] = []
        self.current_player_index = 0
        self.game_phase = GamePhase.SETUP
        self.turn_number = 1
        self.created_at = datetime.now()
        
        # Game systems
        self.territory_manager = TerritoryManager(territory_data)
        self.card_manager = CardManager()
        self.notepad_manager = NotepadManager()
        
        # Game log
        self.game_log: List[str] = []
        
        # Setup tracking
        self.setup_armies_remaining = 0
        self.initial_territories_assigned = False
        
        # Available player colors
        self.available_colors = [
            "Red", "Blue", "Green", "Yellow", "Black", "Orange"
        ]
        random.shuffle(self.available_colors)
    
    def add_player(self, player_name: str) -> Tuple[bool, str, str]:
        """
        Add a player to the game.
        Returns (success, player_id, message).
        """
        if len(self.players) >= self.num_players:
            return False, "", "Game is full"
        
        if self.game_phase != GamePhase.SETUP:
            return False, "", "Game has already started"
        
        # Check for duplicate names
        if any(p.name.lower() == player_name.lower() for p in self.players.values()):
            return False, "", "Player name already taken"
        
        player_id = str(uuid.uuid4())
        color = self.available_colors[len(self.players)]
        
        player = Player(
            player_id=player_id,
            name=player_name,
            color=color
        )
        
        self.players[player_id] = player
        self.player_order.append(player_id)
        
        self.log_event(f"{player_name} ({color}) joined the game")
        
        # Start setup if we have enough players
        if len(self.players) == self.num_players:
            self._start_setup()
        
        return True, player_id, f"Joined as {player_name} ({color})"
    
    def _start_setup(self) -> None:
        """Start the game setup phase."""
        self.log_event("Game setup started - assigning territories")
        
        # Shuffle player order
        random.shuffle(self.player_order)
        
        # Assign territories randomly
        territories = list(self.territory_manager.territories.keys())
        random.shuffle(territories)
        
        # Distribute territories evenly
        for i, territory_name in enumerate(territories):
            player_id = self.player_order[i % len(self.players)]
            territory = self.territory_manager.get_territory(territory_name)
            if territory:
                territory.set_owner(player_id)
                territory.add_armies(1)  # Each territory starts with 1 army
                
                player = self.players[player_id]
                player.add_territory(territory_name)
        
        # Calculate initial armies for each player
        initial_armies = self._calculate_initial_armies()
        territories_per_player = len(territories) // len(self.players)
        
        for player in self.players.values():
            # Total initial armies minus what's already placed
            player.army_count = initial_armies - territories_per_player
        
        self.setup_armies_remaining = sum(p.army_count for p in self.players.values())
        self.initial_territories_assigned = True
        
        self.log_event(f"Territories assigned. Each player has {initial_armies} armies to place.")
        self.log_event(f"Turn order: {', '.join(self.players[pid].name for pid in self.player_order)}")
    
    def _calculate_initial_armies(self) -> int:
        """Calculate initial army count based on number of players."""
        army_counts = {2: 40, 3: 35, 4: 30, 5: 25, 6: 20}
        return army_counts.get(len(self.players), 20)
    
    def get_current_player(self) -> Optional[Player]:
        """Get the current player whose turn it is."""
        if not self.player_order:
            return None
        return self.players[self.player_order[self.current_player_index]]
    
    def advance_turn(self) -> None:
        """Advance to the next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % len(self.player_order)
        
        # If we've cycled through all players, increment turn number
        if self.current_player_index == 0:
            self.turn_number += 1
        
        # Skip eliminated players
        current_player = self.get_current_player()
        if current_player and current_player.is_eliminated:
            self.advance_turn()
    
    def place_armies(self, player_id: str, territory_name: str, army_count: int) -> Tuple[bool, str]:
        """Place armies on a territory."""
        player = self.players.get(player_id)
        territory = self.territory_manager.get_territory(territory_name)
        
        if not player or not territory:
            return False, "Invalid player or territory"
        
        if not territory.is_owned_by(player_id):
            return False, "You don't own this territory"
        
        if army_count <= 0:
            return False, "Must place at least 1 army"
        
        if self.game_phase == GamePhase.SETUP:
            # During setup, can only place available armies
            if army_count > player.army_count:
                return False, f"You only have {player.army_count} armies to place"
            
            player.army_count -= army_count
            territory.add_armies(army_count)
            
            self.setup_armies_remaining -= army_count
            
            self.log_event(f"{player.name} placed {army_count} armies on {territory_name}")
            
            # Check if setup is complete
            if self.setup_armies_remaining == 0:
                self._start_main_game()
            else:
                self.advance_turn()
            
            return True, f"Placed {army_count} armies on {territory_name}"
        
        elif self.game_phase == GamePhase.REINFORCEMENT:
            # During reinforcement, can place reinforcement armies
            current_player = self.get_current_player()
            if not current_player or current_player.player_id != player_id:
                return False, "Not your turn"
            
            if army_count > player.army_count:
                return False, f"You only have {player.army_count} armies to place"
            
            player.army_count -= army_count
            territory.add_armies(army_count)
            
            self.log_event(f"{player.name} placed {army_count} armies on {territory_name}")
            
            return True, f"Placed {army_count} armies on {territory_name}"
        
        else:
            return False, "Cannot place armies during this phase"
    
    def _start_main_game(self) -> None:
        """Start the main game after setup is complete."""
        self.game_phase = GamePhase.REINFORCEMENT
        self.current_player_index = 0
        
        # Calculate reinforcements for the first player
        current_player = self.get_current_player()
        if current_player:
            self._calculate_reinforcements(current_player)
        
        self.log_event("Setup complete! Main game begins.")
    
    def _calculate_reinforcements(self, player: Player) -> None:
        """Calculate and assign reinforcement armies for a player."""
        # Base reinforcement: territories / 3 (minimum 3)
        base_armies = max(3, len(player.territories) // 3)
        
        # Continent bonuses
        continent_bonuses = self.territory_manager.get_player_continent_bonuses(player.player_id)
        bonus_armies = sum(continent_bonuses.values())
        
        total_armies = base_armies + bonus_armies
        player.army_count += total_armies
        
        bonus_str = ""
        if continent_bonuses:
            bonus_details = [f"{cont}: +{bonus}" for cont, bonus in continent_bonuses.items()]
            bonus_str = f" (including continent bonuses: {', '.join(bonus_details)})"
        
        self.log_event(f"{player.name} receives {total_armies} reinforcement armies{bonus_str}")
    
    def start_attack_phase(self, player_id: str) -> Tuple[bool, str]:
        """Start the attack phase for the current player."""
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn"
        
        if self.game_phase != GamePhase.REINFORCEMENT:
            return False, "Must complete reinforcement phase first"
        
        if current_player.army_count > 0:
            return False, "Must place all reinforcement armies first"
        
        self.game_phase = GamePhase.ATTACK
        self.log_event(f"{current_player.name} begins attack phase")
        
        return True, "Attack phase started"
    
    def attack_territory(self, player_id: str, from_territory: str, to_territory: str) -> Tuple[bool, str, Optional[CombatResult]]:
        """Execute an attack between territories."""
        if self.game_phase != GamePhase.ATTACK:
            return False, "Not in attack phase", None
        
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn", None
        
        # Validate territories
        from_terr = self.territory_manager.get_territory(from_territory)
        to_terr = self.territory_manager.get_territory(to_territory)
        
        if not from_terr or not to_terr:
            return False, "Invalid territory", None
        
        if not from_terr.is_owned_by(player_id):
            return False, "You don't own the attacking territory", None
        
        if to_terr.is_owned_by(player_id):
            return False, "Cannot attack your own territory", None
        
        if not from_terr.can_attack_from():
            return False, "Need more than 1 army to attack", None
        
        if not self.territory_manager.are_adjacent(from_territory, to_territory):
            return False, "Territories are not adjacent", None
        
        # Conduct battle
        result = CombatEngine.conduct_battle(from_terr.army_count, to_terr.army_count)
        
        # Apply losses
        from_terr.remove_armies(result.attacker_losses)
        to_terr.remove_armies(result.defender_losses)
        
        # Check if territory was conquered
        if result.territory_conquered:
            # Transfer ownership
            defender = None
            if to_terr.owner:
                defender = self.players[to_terr.owner]
                defender.remove_territory(to_territory)
            
            current_player.add_territory(to_territory)
            to_terr.set_owner(player_id)
            to_terr.add_armies(1)  # Minimum 1 army must occupy
            from_terr.remove_armies(1)  # Move 1 army to conquered territory
            
            # Award Risk card
            card = self.card_manager.draw_card()
            if card:
                current_player.add_card(card)
                self.log_event(f"{current_player.name} receives a {card} card")
            
            # Check if defender is eliminated
            if defender and len(defender.territories) == 0:
                defender.is_eliminated = True
                self.log_event(f"{defender.name} has been eliminated!")
                
                # Transfer all cards to attacker
                current_player.cards.extend(defender.cards)
                defender.cards.clear()
            
            # Check for victory
            if len(current_player.territories) == len(self.territory_manager.territories):
                self._end_game(current_player)
        
        # Log the battle
        defender_name = self.players[to_terr.owner].name if to_terr.owner else "Unknown"
        battle_log = CombatEngine.format_battle_result(
            result, current_player.name, defender_name, from_territory, to_territory
        )
        self.log_event(battle_log)
        
        return True, "Attack completed", result
    
    def start_fortify_phase(self, player_id: str) -> Tuple[bool, str]:
        """Start the fortify phase for the current player."""
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn"
        
        if self.game_phase != GamePhase.ATTACK:
            return False, "Must be in attack phase"
        
        self.game_phase = GamePhase.FORTIFY
        self.log_event(f"{current_player.name} begins fortify phase")
        
        return True, "Fortify phase started"
    
    def fortify_position(self, player_id: str, from_territory: str, to_territory: str, army_count: int) -> Tuple[bool, str]:
        """Move armies between connected territories."""
        if self.game_phase != GamePhase.FORTIFY:
            return False, "Not in fortify phase"
        
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn"
        
        # Validate territories
        from_terr = self.territory_manager.get_territory(from_territory)
        to_terr = self.territory_manager.get_territory(to_territory)
        
        if not from_terr or not to_terr:
            return False, "Invalid territory"
        
        if not from_terr.is_owned_by(player_id) or not to_terr.is_owned_by(player_id):
            return False, "You don't own both territories"
        
        if army_count <= 0:
            return False, "Must move at least 1 army"
        
        if from_terr.army_count <= army_count:
            return False, "Must leave at least 1 army in the source territory"
        
        # For simplicity, allow fortification between any owned territories
        # In classic Risk, they must be connected, but this is more flexible
        
        from_terr.remove_armies(army_count)
        to_terr.add_armies(army_count)
        
        self.log_event(f"{current_player.name} moved {army_count} armies from {from_territory} to {to_territory}")
        
        return True, f"Moved {army_count} armies from {from_territory} to {to_territory}"
    
    def end_turn(self, player_id: str) -> Tuple[bool, str]:
        """End the current player's turn."""
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn"
        
        if self.game_phase == GamePhase.REINFORCEMENT and current_player.army_count > 0:
            # Track this failure for loop detection
            try:
                # Import here to avoid circular imports
                from src.persistence.action_tracker import action_tracker
                consecutive_failures, intervention_needed = action_tracker.track_action_failure(
                    self.game_id,
                    player_id,
                    current_player.name,
                    "end_turn",
                    "Must place all reinforcement armies before ending turn"
                )
                
                # If intervention is needed, auto-place remaining armies
                if intervention_needed:
                    self.log_event(f"🚨 INTERVENTION: Auto-placing {current_player.army_count} armies for stuck player {current_player.name}")
                    self._auto_place_remaining_armies(current_player)
                    action_tracker.mark_intervention_triggered(self.game_id, player_id, "end_turn")
                    self.log_event(f"✅ RECOVERY: Auto-placed armies, {current_player.name} can now end turn")
                    # Continue with normal end turn logic below
                else:
                    return False, "Must place all reinforcement armies before ending turn"
            except Exception as e:
                self.log_event(f"Error in loop detection: {e}")
                return False, "Must place all reinforcement armies before ending turn"
        
        # Clear any failure tracking for successful end_turn
        try:
            from src.persistence.action_tracker import action_tracker
            action_tracker.clear_failure_tracking(self.game_id, player_id, "end_turn")
        except Exception:
            pass  # Don't fail the turn end if tracking fails
        
        # Advance to next player
        self.advance_turn()
        next_player = self.get_current_player()
        
        if next_player:
            self.game_phase = GamePhase.REINFORCEMENT
            self._calculate_reinforcements(next_player)
            self.log_event(f"Turn {self.turn_number}: {next_player.name}'s turn begins")
        
        return True, f"Turn ended. Next player: {next_player.name if next_player else 'Unknown'}"
    
    def auto_advance_phase(self) -> bool:
        """
        Automatically advance to the next appropriate phase based on game state.
        Returns True if phase was advanced, False if no advancement needed.
        """
        current_player = self.get_current_player()
        if not current_player:
            return False
            
        if self.game_phase == GamePhase.REINFORCEMENT:
            # Auto-advance from reinforcement to attack when all armies are placed
            if current_player.army_count == 0:
                self.game_phase = GamePhase.ATTACK
                self.log_event(f"{current_player.name} enters attack phase (auto-advanced)")
                return True
                
        elif self.game_phase == GamePhase.ATTACK:
            # Attack phase can transition to fortify when agent calls end_turn
            # or when no valid attacks are possible (handled by game runner)
            pass
            
        elif self.game_phase == GamePhase.FORTIFY:
            # Fortify phase can transition to next player when agent calls end_turn
            # or when no valid fortifications are possible (handled by game runner)
            pass
            
        return False
    
    def can_perform_action(self, player_id: str, action_type: str) -> Tuple[bool, str]:
        """
        Check if a player can perform a specific action in the current phase.
        Returns (can_perform, reason).
        """
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return False, "Not your turn"
            
        if self.game_phase == GamePhase.SETUP:
            if action_type == "place_armies":
                return True, "Can place armies during setup"
            return False, f"Cannot {action_type} during setup phase"
            
        elif self.game_phase == GamePhase.REINFORCEMENT:
            if action_type in ["place_armies", "trade_cards"]:
                return True, f"Can {action_type} during reinforcement phase"
            return False, f"Cannot {action_type} during reinforcement phase"
            
        elif self.game_phase == GamePhase.ATTACK:
            if action_type in ["attack_territory", "trade_cards", "place_armies"]:
                return True, f"Can {action_type} during attack phase"
            return False, f"Cannot {action_type} during attack phase"
            
        elif self.game_phase == GamePhase.FORTIFY:
            if action_type in ["fortify_position", "trade_cards", "place_armies"]:
                return True, f"Can {action_type} during fortify phase"
            return False, f"Cannot {action_type} during fortify phase"
            
        elif self.game_phase == GamePhase.GAME_OVER:
            return False, "Game is over"
            
        return False, "Unknown game phase"
    
    def get_available_actions(self, player_id: str) -> List[str]:
        """Get list of actions available to the player in the current phase."""
        current_player = self.get_current_player()
        if not current_player or current_player.player_id != player_id:
            return []
            
        if self.game_phase == GamePhase.SETUP:
            return ["place_armies"] if current_player.army_count > 0 else []
            
        elif self.game_phase == GamePhase.REINFORCEMENT:
            actions = []
            if current_player.army_count > 0:
                actions.append("place_armies")
            if len(current_player.cards) >= 3:
                actions.append("trade_cards")
            return actions
            
        elif self.game_phase == GamePhase.ATTACK:
            actions = ["end_turn"]  # Can always end turn
            if current_player.army_count > 0:
                actions.append("place_armies")
            if len(current_player.cards) >= 3:
                actions.append("trade_cards")
            
            # Check if any attacks are possible
            if self._can_attack_somewhere(player_id):
                actions.append("attack_territory")
            
            return actions
            
        elif self.game_phase == GamePhase.FORTIFY:
            actions = ["end_turn"]  # Can always end turn
            if current_player.army_count > 0:
                actions.append("place_armies")
            if len(current_player.cards) >= 3:
                actions.append("trade_cards")
            
            # Check if any fortifications are possible
            if self._can_fortify_somewhere(player_id):
                actions.append("fortify_position")
            
            return actions
            
        return []
    
    def _can_attack_somewhere(self, player_id: str) -> bool:
        """Check if the player can attack from any of their territories."""
        for territory_name in self.players[player_id].territories:
            territory = self.territory_manager.get_territory(territory_name)
            if territory and territory.can_attack_from():
                # Check if any adjacent territories are owned by enemies
                for adj_name in territory.adjacent_territories:
                    adj_territory = self.territory_manager.get_territory(adj_name)
                    if adj_territory and not adj_territory.is_owned_by(player_id):
                        return True
        return False
    
    def _can_fortify_somewhere(self, player_id: str) -> bool:
        """Check if the player can fortify between any of their territories."""
        player_territories = self.players[player_id].territories
        for from_territory_name in player_territories:
            from_territory = self.territory_manager.get_territory(from_territory_name)
            if from_territory and from_territory.army_count > 1:
                # Can move armies from this territory
                return True
        return False
    
    def force_advance_phase(self, reason: str = "forced") -> bool:
        """
        Force advancement to the next phase when agent gets stuck.
        Returns True if phase was advanced, False if no advancement possible.
        """
        current_player = self.get_current_player()
        if not current_player:
            return False
            
        if self.game_phase == GamePhase.REINFORCEMENT:
            # Force advance to attack phase
            self.game_phase = GamePhase.ATTACK
            self.log_event(f"{current_player.name} forced to attack phase ({reason})")
            return True
            
        elif self.game_phase == GamePhase.ATTACK:
            # Force advance to fortify phase
            self.game_phase = GamePhase.FORTIFY
            self.log_event(f"{current_player.name} forced to fortify phase ({reason})")
            return True
            
        elif self.game_phase == GamePhase.FORTIFY:
            # Force end turn and advance to next player
            self.advance_turn()
            next_player = self.get_current_player()
            
            if next_player:
                self.game_phase = GamePhase.REINFORCEMENT
                self._calculate_reinforcements(next_player)
                self.log_event(f"Turn {self.turn_number}: {next_player.name}'s turn begins (forced advancement)")
            return True
            
        return False
    
    def trade_cards(self, player_id: str, card_indices: List[int]) -> Tuple[bool, str]:
        """Trade in a set of Risk cards for army bonus."""
        player = self.players.get(player_id)
        if not player:
            return False, "Invalid player"
        
        if len(card_indices) != 3:
            return False, "Must trade exactly 3 cards"
        
        # Validate indices
        if any(idx < 0 or idx >= len(player.cards) for idx in card_indices):
            return False, "Invalid card indices"
        
        # Get the cards to trade
        cards_to_trade = [player.cards[idx] for idx in card_indices]
        
        # Attempt trade
        success, army_bonus, message = self.card_manager.trade_cards(cards_to_trade)
        
        if success:
            # Remove cards from player's hand
            player.remove_cards(card_indices)
            player.army_count += army_bonus
            self.log_event(f"{player.name} traded cards for {army_bonus} armies")
        
        return success, message
    
    def _auto_place_remaining_armies(self, player: Player) -> None:
        """
        Auto-place remaining armies for a stuck player.
        Distributes armies strategically across player's territories.
        """
        if player.army_count <= 0:
            return
        
        armies_to_place = player.army_count
        player_territories = list(player.territories)
        
        if not player_territories:
            self.log_event(f"ERROR: Player {player.name} has no territories for auto-placement")
            player.army_count = 0  # Clear to prevent infinite loop
            return
        
        # Strategy: Place armies on border territories first (territories adjacent to enemies)
        border_territories = []
        interior_territories = []
        
        for territory_name in player_territories:
            territory = self.territory_manager.get_territory(territory_name)
            if not territory:
                continue
                
            is_border = False
            for adj_name in territory.adjacent_territories:
                adj_territory = self.territory_manager.get_territory(adj_name)
                if adj_territory and not adj_territory.is_owned_by(player.player_id):
                    is_border = True
                    break
            
            if is_border:
                border_territories.append((territory_name, territory))
            else:
                interior_territories.append((territory_name, territory))
        
        # Prioritize border territories, then interior
        priority_territories = border_territories + interior_territories
        
        # Distribute armies evenly, with preference for weaker territories
        armies_placed = 0
        while armies_placed < armies_to_place and priority_territories:
            for territory_name, territory in priority_territories:
                if armies_placed >= armies_to_place:
                    break
                
                # Place 1 army at a time to distribute evenly
                territory.add_armies(1)
                armies_placed += 1
                
                if armies_placed == 1:  # Log first placement
                    self.log_event(f"🤖 AUTO-PLACEMENT: Placing armies on {player.name}'s territories")
        
        # Update player's army count
        player.army_count = 0
        
        self.log_event(f"🤖 AUTO-PLACEMENT: Placed {armies_placed} armies across {len(priority_territories)} territories")
        
        # Log final distribution for transparency
        territory_summary = []
        for territory_name, territory in priority_territories[:5]:  # Show first 5
            territory_summary.append(f"{territory_name}({territory.army_count})")
        
        if len(priority_territories) > 5:
            territory_summary.append(f"...and {len(priority_territories) - 5} more")
        
        self.log_event(f"🤖 DISTRIBUTION: {', '.join(territory_summary)}")
    
    def detect_stuck_player(self, player_id: str) -> bool:
        """
        Detect if a player appears to be stuck in a loop.
        Returns True if intervention is recommended.
        """
        try:
            from src.persistence.action_tracker import action_tracker
            failure_patterns = action_tracker.get_player_failure_patterns(self.game_id, player_id)
            
            # Check for any action with 5+ consecutive failures
            for pattern in failure_patterns:
                if pattern['consecutive_failures'] >= 5 and not pattern['intervention_triggered']:
                    return True
            
            return False
            
        except Exception as e:
            self.log_event(f"Error detecting stuck player: {e}")
            return False
    
    def get_stuck_players_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all stuck players in the game."""
        try:
            from src.persistence.action_tracker import action_tracker
            return action_tracker.get_stuck_players(self.game_id)
        except Exception as e:
            self.log_event(f"Error getting stuck players: {e}")
            return []
    
    def _end_game(self, winner: Player) -> None:
        """End the game with a winner."""
        self.game_phase = GamePhase.GAME_OVER
        self.log_event(f"🎉 {winner.name} has conquered the world and won the game!")
        
        # Notify tournament manager if this is a tournament game
        self._notify_tournament_game_end(winner)
    
    def _notify_tournament_game_end(self, winner: Player) -> None:
        """Notify tournament manager when game ends naturally."""
        try:
            # Import here to avoid circular imports
            from utils.game_manager import game_manager
            
            # Check if this is a tournament game
            if hasattr(game_manager, 'tournament_manager') and game_manager.tournament_manager:
                tournament_manager = game_manager.tournament_manager
                
                # Create comprehensive winner information
                winner_info = {
                    'winner_player_id': winner.player_id,
                    'winner_player_name': winner.name,
                    'completion_reason': 'victory',
                    'winner_determination_method': 'complete_conquest',
                    'winner_criteria': 'Conquered all territories',
                    'total_turns': self.turn_number,
                    'total_duration_seconds': (datetime.now() - self.created_at).total_seconds(),
                    'total_actions': len(self.game_log),
                    'battles_fought': len([log for log in self.game_log if 'attacks' in log.lower()]),
                    'final_scores': self._calculate_final_scores(),
                    'game_phase_at_end': self.game_phase.value
                }
                
                # Notify tournament manager
                tournament_manager.capture_natural_game_end(self.game_id, winner_info)
                
                self.log_event(f"Tournament notified of natural game end: {winner.name} wins")
                
        except Exception as e:
            self.log_event(f"Error notifying tournament of game end: {e}")
    
    def _calculate_final_scores(self) -> List[Dict]:
        """Calculate final scores for all players."""
        scores = []
        for player in self.players.values():
            territory_count = len(player.territories)
            army_count = sum(
                self.territory_manager.get_territory(t).army_count 
                for t in player.territories 
                if self.territory_manager.get_territory(t)
            )
            continent_bonuses = self.territory_manager.get_player_continent_bonuses(player.player_id)
            
            scores.append({
                'player_id': player.player_id,
                'player_name': player.name,
                'territory_count': territory_count,
                'army_count': army_count,
                'continent_bonuses': sum(continent_bonuses.values()),
                'is_eliminated': player.is_eliminated,
                'cards_held': len(player.cards)
            })
        
        # Sort by territory count (winner first)
        scores.sort(key=lambda x: x['territory_count'], reverse=True)
        return scores
    
    def log_event(self, event: str) -> None:
        """Add an event to the game log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.game_log.append(f"[{timestamp}] {event}")
    
    def get_game_status(self) -> dict:
        """Get current game status."""
        current_player = self.get_current_player()
        
        return {
            'game_id': self.game_id,
            'phase': self.game_phase.value,
            'turn_number': self.turn_number,
            'current_player': current_player.name if current_player else None,
            'current_player_id': current_player.player_id if current_player else None,
            'players': [p.to_dict() for p in self.players.values()],
            'setup_armies_remaining': self.setup_armies_remaining,
            'created_at': self.created_at.isoformat()
        }
    
    def get_player_info(self, player_id: str) -> Optional[dict]:
        """Get detailed information for a specific player."""
        player = self.players.get(player_id)
        if not player:
            return None
        
        continent_bonuses = self.territory_manager.get_player_continent_bonuses(player_id)
        
        return {
            **player.to_dict(),
            'continent_bonuses': continent_bonuses,
            'total_continent_bonus': sum(continent_bonuses.values()),
            'message_summary': self.notepad_manager.get_message_summary(player_id)
        }
