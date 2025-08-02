import random
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class CombatResult:
    attacker_losses: int
    defender_losses: int
    attacker_dice: List[int]
    defender_dice: List[int]
    territory_conquered: bool = False
    
    def to_dict(self) -> dict:
        """Convert combat result to dictionary."""
        return {
            'attacker_losses': self.attacker_losses,
            'defender_losses': self.defender_losses,
            'attacker_dice': self.attacker_dice,
            'defender_dice': self.defender_dice,
            'territory_conquered': self.territory_conquered
        }

class CombatEngine:
    """Handles dice rolling and combat resolution for Risk."""
    
    @staticmethod
    def roll_dice(count: int) -> List[int]:
        """Roll the specified number of dice and return sorted results (highest first)."""
        dice = [random.randint(1, 6) for _ in range(count)]
        return sorted(dice, reverse=True)
    
    @staticmethod
    def get_attacker_dice_count(army_count: int) -> int:
        """Determine how many dice the attacker can roll based on army count."""
        if army_count >= 4:
            return 3
        elif army_count >= 3:
            return 2
        elif army_count >= 2:
            return 1
        else:
            return 0  # Cannot attack
    
    @staticmethod
    def get_defender_dice_count(army_count: int) -> int:
        """Determine how many dice the defender can roll based on army count."""
        if army_count >= 2:
            return 2
        elif army_count >= 1:
            return 1
        else:
            return 0  # No armies to defend
    
    @staticmethod
    def resolve_combat(attacker_dice: List[int], defender_dice: List[int]) -> Tuple[int, int]:
        """
        Resolve combat between attacker and defender dice.
        Returns (attacker_losses, defender_losses).
        """
        attacker_losses = 0
        defender_losses = 0
        
        # Compare dice in pairs, highest vs highest
        comparisons = min(len(attacker_dice), len(defender_dice))
        
        for i in range(comparisons):
            if attacker_dice[i] > defender_dice[i]:
                defender_losses += 1
            else:
                attacker_losses += 1
        
        return attacker_losses, defender_losses
    
    @classmethod
    def conduct_battle(cls, attacking_armies: int, defending_armies: int) -> CombatResult:
        """
        Conduct a single battle between attacking and defending armies.
        Returns detailed combat result.
        """
        # Determine dice counts
        attacker_dice_count = cls.get_attacker_dice_count(attacking_armies)
        defender_dice_count = cls.get_defender_dice_count(defending_armies)
        
        if attacker_dice_count == 0:
            # Cannot attack
            return CombatResult(
                attacker_losses=0,
                defender_losses=0,
                attacker_dice=[],
                defender_dice=[]
            )
        
        # Roll dice
        attacker_dice = cls.roll_dice(attacker_dice_count)
        defender_dice = cls.roll_dice(defender_dice_count)
        
        # Resolve combat
        attacker_losses, defender_losses = cls.resolve_combat(attacker_dice, defender_dice)
        
        # Check if territory is conquered
        territory_conquered = (defending_armies - defender_losses) <= 0
        
        return CombatResult(
            attacker_losses=attacker_losses,
            defender_losses=defender_losses,
            attacker_dice=attacker_dice,
            defender_dice=defender_dice,
            territory_conquered=territory_conquered
        )
    
    @classmethod
    def simulate_full_attack(cls, attacking_armies: int, defending_armies: int) -> List[CombatResult]:
        """
        Simulate a full attack until one side has no armies left.
        Returns list of all battle results.
        """
        battles = []
        current_attacking = attacking_armies
        current_defending = defending_armies
        
        while current_attacking > 1 and current_defending > 0:
            result = cls.conduct_battle(current_attacking, current_defending)
            battles.append(result)
            
            current_attacking -= result.attacker_losses
            current_defending -= result.defender_losses
            
            if result.territory_conquered:
                break
        
        return battles
    
    @staticmethod
    def format_battle_result(result: CombatResult, attacker_name: str, defender_name: str, 
                           from_territory: str, to_territory: str) -> str:
        """Format a battle result into a readable string."""
        attacker_dice_str = ", ".join(map(str, result.attacker_dice))
        defender_dice_str = ", ".join(map(str, result.defender_dice))
        
        message = f"Battle: {attacker_name} attacks {to_territory} from {from_territory}\n"
        message += f"Attacker rolled: [{attacker_dice_str}]\n"
        message += f"Defender rolled: [{defender_dice_str}]\n"
        message += f"Losses - Attacker: {result.attacker_losses}, Defender: {result.defender_losses}\n"
        
        if result.territory_conquered:
            message += f"🎉 {to_territory} has been conquered!"
        
        return message
