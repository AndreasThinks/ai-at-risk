from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Territory:
    name: str
    continent: str
    adjacent_territories: List[str]
    owner: Optional[str] = None  # player_id
    army_count: int = 0
    
    def is_owned_by(self, player_id: str) -> bool:
        """Check if territory is owned by the specified player."""
        return self.owner == player_id
    
    def is_adjacent_to(self, territory_name: str) -> bool:
        """Check if this territory is adjacent to another territory."""
        return territory_name in self.adjacent_territories
    
    def can_attack_from(self) -> bool:
        """Check if this territory can launch attacks (has more than 1 army)."""
        return self.army_count > 1
    
    def can_be_attacked(self) -> bool:
        """Check if this territory can be attacked (has an owner)."""
        return self.owner is not None
    
    def add_armies(self, count: int) -> None:
        """Add armies to this territory."""
        self.army_count += count
    
    def remove_armies(self, count: int) -> int:
        """Remove armies from this territory, return actual amount removed."""
        actual_removed = min(count, self.army_count)
        self.army_count -= actual_removed
        return actual_removed
    
    def set_owner(self, player_id: str) -> None:
        """Set the owner of this territory."""
        self.owner = player_id
    
    def clear_owner(self) -> None:
        """Remove the owner of this territory."""
        self.owner = None
        self.army_count = 0
    
    def to_dict(self) -> dict:
        """Convert territory to dictionary for serialization."""
        return {
            'name': self.name,
            'continent': self.continent,
            'adjacent_territories': self.adjacent_territories.copy(),
            'owner': self.owner,
            'army_count': self.army_count
        }

class TerritoryManager:
    """Manages all territories and their relationships."""
    
    def __init__(self, territory_data: dict):
        self.territories = {}
        self.continents = territory_data['continents']
        
        # Initialize all territories
        for name, data in territory_data['territories'].items():
            self.territories[name] = Territory(
                name=name,
                continent=data['continent'],
                adjacent_territories=data['adjacent']
            )
    
    def get_territory(self, name: str) -> Optional[Territory]:
        """Get a territory by name."""
        return self.territories.get(name)
    
    def get_all_territories(self) -> List[Territory]:
        """Get all territories."""
        return list(self.territories.values())
    
    def get_territories_by_continent(self, continent: str) -> List[Territory]:
        """Get all territories in a continent."""
        return [t for t in self.territories.values() if t.continent == continent]
    
    def get_territories_by_owner(self, player_id: str) -> List[Territory]:
        """Get all territories owned by a player."""
        return [t for t in self.territories.values() if t.owner == player_id]
    
    def are_adjacent(self, territory1: str, territory2: str) -> bool:
        """Check if two territories are adjacent."""
        t1 = self.get_territory(territory1)
        if t1:
            return t1.is_adjacent_to(territory2)
        return False
    
    def get_continent_bonus(self, continent: str) -> int:
        """Get the army bonus for controlling a continent."""
        return self.continents.get(continent, {}).get('bonus', 0)
    
    def get_continent_territories(self, continent: str) -> List[str]:
        """Get all territory names in a continent."""
        return self.continents.get(continent, {}).get('territories', [])
    
    def player_controls_continent(self, player_id: str, continent: str) -> bool:
        """Check if a player controls all territories in a continent."""
        continent_territories = self.get_continent_territories(continent)
        return all(
            self.territories[t_name].owner == player_id 
            for t_name in continent_territories
        )
    
    def get_player_continent_bonuses(self, player_id: str) -> dict:
        """Get all continent bonuses for a player."""
        bonuses = {}
        for continent in self.continents:
            if self.player_controls_continent(player_id, continent):
                bonuses[continent] = self.get_continent_bonus(continent)
        return bonuses
    
    def to_dict(self) -> dict:
        """Convert all territories to dictionary for serialization."""
        return {
            name: territory.to_dict() 
            for name, territory in self.territories.items()
        }
