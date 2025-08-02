import random
from typing import List, Dict, Tuple, Optional
from enum import Enum

class CardType(Enum):
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry"
    ARTILLERY = "Artillery"
    WILD = "Wild"

class CardManager:
    """Manages Risk cards and set trading."""
    
    def __init__(self):
        self.deck = self._create_deck()
        self.discard_pile = []
        random.shuffle(self.deck)
        self.sets_traded = 0  # Track number of sets traded globally for bonus calculation
    
    def _create_deck(self) -> List[str]:
        """Create a standard Risk card deck."""
        deck = []
        
        # Standard deck has cards for each territory plus wild cards
        territories = [
            "Alaska", "Northwest Territory", "Alberta", "Ontario", "Western United States",
            "Eastern United States", "Quebec", "Central America", "Greenland",
            "Venezuela", "Brazil", "Peru", "Argentina",
            "Iceland", "Great Britain", "Scandinavia", "Northern Europe", "Western Europe",
            "Southern Europe", "Ukraine", "North Africa", "Egypt", "East Africa",
            "Congo", "South Africa", "Madagascar", "Middle East", "Afghanistan",
            "Ural", "Siberia", "China", "Mongolia", "Irkutsk", "Yakutsk",
            "Kamchatka", "Japan", "India", "Siam", "Indonesia", "New Guinea",
            "Western Australia", "Eastern Australia"
        ]
        
        # Assign card types cyclically to territories
        card_types = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        for i, territory in enumerate(territories):
            card_type = card_types[i % 3]
            deck.append(card_type.value)
        
        # Add 2 wild cards
        deck.extend([CardType.WILD.value, CardType.WILD.value])
        
        return deck
    
    def draw_card(self) -> Optional[str]:
        """Draw a card from the deck. Returns None if deck is empty."""
        if not self.deck:
            # Reshuffle discard pile if deck is empty
            if self.discard_pile:
                self.deck = self.discard_pile.copy()
                self.discard_pile.clear()
                random.shuffle(self.deck)
            else:
                return None
        
        return self.deck.pop()
    
    def discard_cards(self, cards: List[str]) -> None:
        """Add cards to discard pile."""
        self.discard_pile.extend(cards)
    
    def is_valid_set(self, cards: List[str]) -> bool:
        """Check if the given cards form a valid set for trading."""
        if len(cards) != 3:
            return False
        
        # Count card types
        card_counts = {card_type.value: 0 for card_type in CardType}
        for card in cards:
            if card in card_counts:
                card_counts[card] += 1
        
        wild_count = card_counts[CardType.WILD.value]
        non_wild_cards = [card for card in cards if card != CardType.WILD.value]
        
        # Valid sets:
        # 1. Three of the same type (including wilds as substitutes)
        # 2. One of each type (Infantry, Cavalry, Artillery)
        # 3. Any combination with wild cards
        
        if wild_count >= 1:
            # With wild cards, almost any combination works
            return True
        
        # Without wild cards, check for three of a kind or one of each
        unique_types = set(non_wild_cards)
        
        if len(unique_types) == 1:
            # Three of the same type
            return True
        elif len(unique_types) == 3:
            # One of each type
            expected_types = {CardType.INFANTRY.value, CardType.CAVALRY.value, CardType.ARTILLERY.value}
            return unique_types == expected_types
        
        return False
    
    def calculate_set_bonus(self) -> int:
        """Calculate army bonus for trading in a set."""
        # Standard Risk progression: 4, 6, 8, 10, 12, 15, then +5 for each subsequent set
        if self.sets_traded == 0:
            return 4
        elif self.sets_traded == 1:
            return 6
        elif self.sets_traded == 2:
            return 8
        elif self.sets_traded == 3:
            return 10
        elif self.sets_traded == 4:
            return 12
        elif self.sets_traded == 5:
            return 15
        else:
            # After 6th set, each subsequent set gives 5 more than the previous
            return 15 + (self.sets_traded - 5) * 5
    
    def trade_cards(self, cards: List[str]) -> Tuple[bool, int, str]:
        """
        Trade in a set of cards for army bonus.
        Returns (success, army_bonus, message).
        """
        if not self.is_valid_set(cards):
            return False, 0, "Invalid card set. You need 3 cards that are either all the same type or one of each type."
        
        bonus = self.calculate_set_bonus()
        self.sets_traded += 1
        self.discard_cards(cards)
        
        return True, bonus, f"Set traded successfully! You receive {bonus} armies."
    
    def get_card_summary(self, cards: List[str]) -> Dict[str, int]:
        """Get a summary count of card types."""
        summary = {card_type.value: 0 for card_type in CardType}
        for card in cards:
            if card in summary:
                summary[card] += 1
        return summary
    
    def format_cards(self, cards: List[str]) -> str:
        """Format a list of cards for display."""
        if not cards:
            return "No cards"
        
        summary = self.get_card_summary(cards)
        parts = []
        for card_type, count in summary.items():
            if count > 0:
                parts.append(f"{count} {card_type}")
        
        return ", ".join(parts)
    
    def get_possible_sets(self, cards: List[str]) -> List[List[int]]:
        """Get all possible valid sets from the given cards (returns indices)."""
        valid_sets = []
        
        # Generate all combinations of 3 cards
        from itertools import combinations
        for combo in combinations(range(len(cards)), 3):
            card_combo = [cards[i] for i in combo]
            if self.is_valid_set(card_combo):
                valid_sets.append(list(combo))
        
        return valid_sets
