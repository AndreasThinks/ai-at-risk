"""
Game Narrator for generating intelligent turn updates and game state changes.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging


@dataclass
class GameStateChange:
    """Represents a change in the game state."""
    category: str  # 'territories', 'troops', 'cards', 'phase', 'combat', etc.
    description: str
    details: Dict[str, Any]


class GameNarrator:
    """Generates narrative updates about game state changes for agent conversations."""
    
    def __init__(self):
        self.previous_states: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("game_narrator")
    
    def generate_turn_update(
        self,
        game_id: str,
        player_id: str,
        current_state: Dict[str, Any],
        current_phase: str,
        last_action: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a narrative update about what changed this turn.
        
        Args:
            game_id: Game identifier
            player_id: Player identifier  
            current_state: Current game state
            current_phase: Current game phase
            last_action: Last action taken by any player
            
        Returns:
            Formatted narrative update string
        """
        state_key = f"{game_id}_{player_id}"
        previous_state = self.previous_states.get(state_key)
        
        # Store current state for next comparison
        self.previous_states[state_key] = current_state.copy()
        
        if previous_state is None:
            return self._generate_game_start_update(current_state, current_phase)
        
        changes = self._detect_changes(previous_state, current_state)
        return self._format_turn_update(changes, current_phase, last_action)
    
    def _generate_game_start_update(self, state: Dict[str, Any], phase: str) -> str:
        """Generate update for game start."""
        player_info = state.get('player_info', {})
        territories = player_info.get('territories', [])
        total_troops = player_info.get('army_count', 0)
        cards = player_info.get('cards', [])
        
        return f"""=== GAME START ===
Current Phase: {phase.upper()}
Your territories: {len(territories)} 
Your total troops: {total_troops}
Cards in hand: {len(cards)}

The game has begun! You are ready to take your turn."""
    
    def _detect_changes(
        self, 
        previous: Dict[str, Any], 
        current: Dict[str, Any]
    ) -> List[GameStateChange]:
        """Detect changes between two game states."""
        changes = []
        
        # Get player info from both states
        prev_player = previous.get('player_info', {})
        curr_player = current.get('player_info', {})
        
        # Check territory changes
        prev_territories = set(prev_player.get('territories', []))
        curr_territories = set(curr_player.get('territories', []))
        
        # Territories gained
        gained = curr_territories - prev_territories
        for territory in gained:
            changes.append(GameStateChange(
                category='territory_gained',
                description=f"Gained control of {territory}",
                details={'territory': territory}
            ))
        
        # Territories lost
        lost = prev_territories - curr_territories
        for territory in lost:
            changes.append(GameStateChange(
                category='territory_lost',
                description=f"Lost control of {territory}",
                details={'territory': territory}
            ))
        
        # Check troop changes
        prev_troops = prev_player.get('army_count', 0)
        curr_troops = curr_player.get('army_count', 0)
        if prev_troops != curr_troops:
            change = curr_troops - prev_troops
            if change > 0:
                changes.append(GameStateChange(
                    category='troops_gained',
                    description=f"Gained {change} troops (now have {curr_troops})",
                    details={'change': change, 'total': curr_troops}
                ))
            else:
                changes.append(GameStateChange(
                    category='troops_lost',
                    description=f"Lost {abs(change)} troops (now have {curr_troops})",
                    details={'change': change, 'total': curr_troops}
                ))
        
        # Check card changes
        prev_cards = len(prev_player.get('cards', []))
        curr_cards = len(curr_player.get('cards', []))
        if prev_cards != curr_cards:
            change = curr_cards - prev_cards
            if change > 0:
                changes.append(GameStateChange(
                    category='cards_gained',
                    description=f"Gained {change} cards (now have {curr_cards})",
                    details={'change': change, 'total': curr_cards}
                ))
            else:
                changes.append(GameStateChange(
                    category='cards_lost',
                    description=f"Lost {abs(change)} cards (now have {curr_cards})",
                    details={'change': change, 'total': curr_cards}
                ))
        
        # Check continent bonus changes
        prev_bonuses = prev_player.get('continent_bonuses', {})
        curr_bonuses = curr_player.get('continent_bonuses', {})
        
        for continent, bonus in curr_bonuses.items():
            if continent not in prev_bonuses:
                changes.append(GameStateChange(
                    category='continent_gained',
                    description=f"Gained control of {continent} (+{bonus} armies/turn)",
                    details={'continent': continent, 'bonus': bonus}
                ))
        
        for continent, bonus in prev_bonuses.items():
            if continent not in curr_bonuses:
                changes.append(GameStateChange(
                    category='continent_lost',
                    description=f"Lost control of {continent} (-{bonus} armies/turn)",
                    details={'continent': continent, 'bonus': bonus}
                ))
        
        return changes
    
    def _format_turn_update(
        self, 
        changes: List[GameStateChange], 
        current_phase: str, 
        last_action: Optional[Dict[str, Any]]
    ) -> str:
        """Format the turn update with changes."""
        if not changes:
            return f"=== TURN UPDATE ===\nPhase: {current_phase.upper()}\nNo significant changes since last turn.\n"
        
        lines = [f"=== TURN UPDATE ===", f"Phase: {current_phase.upper()}", ""]
        
        # Group changes by category
        territory_changes = [c for c in changes if c.category.startswith('territory')]
        troop_changes = [c for c in changes if c.category.startswith('troops')]
        card_changes = [c for c in changes if c.category.startswith('cards')]
        continent_changes = [c for c in changes if c.category.startswith('continent')]
        
        if territory_changes:
            lines.append("🏴 TERRITORIAL CHANGES:")
            for change in territory_changes:
                if change.category == 'territory_gained':
                    lines.append(f"  ✅ {change.description}")
                else:
                    lines.append(f"  ❌ {change.description}")
            lines.append("")
        
        if continent_changes:
            lines.append("🌍 CONTINENT CONTROL:")
            for change in continent_changes:
                if change.category == 'continent_gained':
                    lines.append(f"  🎯 {change.description}")
                else:
                    lines.append(f"  💔 {change.description}")
            lines.append("")
        
        if troop_changes:
            lines.append("⚔️ ARMY CHANGES:")
            for change in troop_changes:
                if change.category == 'troops_gained':
                    lines.append(f"  📈 {change.description}")
                else:
                    lines.append(f"  📉 {change.description}")
            lines.append("")
        
        if card_changes:
            lines.append("🎴 CARD CHANGES:")
            for change in card_changes:
                if change.category == 'cards_gained':
                    lines.append(f"  🎁 {change.description}")
                else:
                    lines.append(f"  💸 {change.description}")
            lines.append("")
        
        # Add action context if available
        if last_action:
            action_type = last_action.get('action_type', 'unknown')
            action_player = last_action.get('player_id', 'unknown')
            lines.append(f"Last action: {action_type} by player {action_player}")
        
        return "\n".join(lines)
    
    def generate_phase_transition_update(self, old_phase: str, new_phase: str) -> str:
        """Generate update for phase transitions."""
        phase_descriptions = {
            'setup': 'Initial army placement',
            'reinforcement': 'Receive and place reinforcement armies',
            'attack': 'Attack enemy territories',
            'fortify': 'Move armies between your territories',
            'game_over': 'Game has ended'
        }
        
        old_desc = phase_descriptions.get(old_phase, old_phase)
        new_desc = phase_descriptions.get(new_phase, new_phase)
        
        return f"""=== PHASE TRANSITION ===
{old_phase.upper()} → {new_phase.upper()}
Previous: {old_desc}
Now: {new_desc}

The game has moved to a new phase. Adjust your strategy accordingly!"""
