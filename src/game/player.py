from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    from_player_id: str
    from_player_name: str
    to_player_id: str
    content: str
    timestamp: datetime
    is_read: bool = False

@dataclass
class Player:
    player_id: str
    name: str
    color: str
    territories: List[str] = field(default_factory=list)
    army_count: int = 0
    cards: List[str] = field(default_factory=list)  # Risk card types: Infantry, Cavalry, Artillery, Wild
    private_notes: str = ""
    is_eliminated: bool = False
    short_term_strategy: str = ""
    long_term_strategy: str = ""
    
    def add_territory(self, territory: str) -> None:
        """Add a territory to this player's control."""
        if territory not in self.territories:
            self.territories.append(territory)
    
    def remove_territory(self, territory: str) -> None:
        """Remove a territory from this player's control."""
        if territory in self.territories:
            self.territories.remove(territory)
    
    def add_card(self, card_type: str) -> None:
        """Add a Risk card to this player's hand."""
        self.cards.append(card_type)
    
    def remove_cards(self, card_indices: List[int]) -> List[str]:
        """Remove cards by index and return the removed cards."""
        removed_cards = []
        # Sort indices in reverse order to avoid index shifting issues
        for idx in sorted(card_indices, reverse=True):
            if 0 <= idx < len(self.cards):
                removed_cards.append(self.cards.pop(idx))
        return removed_cards[::-1]  # Return in original order
    
    def get_territory_count(self) -> int:
        """Get the number of territories controlled by this player."""
        return len(self.territories)
    
    def controls_continent(self, continent_territories: List[str]) -> bool:
        """Check if player controls all territories in a continent."""
        return all(territory in self.territories for territory in continent_territories)
    
    def update_private_notes(self, notes: str) -> None:
        """Update the player's private notes."""
        self.private_notes = notes
    
    def update_short_term_strategy(self, strategy: str) -> None:
        """Update the player's short term strategy."""
        self.short_term_strategy = strategy
    
    def update_long_term_strategy(self, strategy: str) -> None:
        """Update the player's long term strategy."""
        self.long_term_strategy = strategy
    
    def to_dict(self) -> dict:
        """Convert player to dictionary for serialization."""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'color': self.color,
            'territories': self.territories.copy(),
            'army_count': self.army_count,
            'cards': self.cards.copy(),
            'territory_count': self.get_territory_count(),
            'is_eliminated': self.is_eliminated,
            'short_term_strategy': self.short_term_strategy,
            'long_term_strategy': self.long_term_strategy
        }
