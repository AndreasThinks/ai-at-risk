"""Character management for tournament mode."""

import sqlite3
import json
import os
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TournamentCharacter:
    """Represents a character in the tournament."""
    id: Optional[int]
    name: str
    model_name: str
    temperature: float
    personality: str
    custom_instructions: str
    created_at: datetime
    submitted_by: str
    times_played: int = 0
    wins: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'personality': self.personality,
            'custom_instructions': self.custom_instructions,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'submitted_by': self.submitted_by,
            'times_played': self.times_played,
            'wins': self.wins
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TournamentCharacter':
        """Create from dictionary."""
        created_at = data['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            id=data.get('id'),
            name=data['name'],
            model_name=data['model_name'],
            temperature=data['temperature'],
            personality=data['personality'],
            custom_instructions=data['custom_instructions'],
            created_at=created_at,
            submitted_by=data['submitted_by'],
            times_played=data.get('times_played', 0),
            wins=data.get('wins', 0)
        )

class CharacterManager:
    """Manages tournament characters and voting."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tournament_characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    personality TEXT NOT NULL,
                    custom_instructions TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    submitted_by TEXT NOT NULL,
                    times_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tournament_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tournament_id TEXT NOT NULL,
                    character_id INTEGER NOT NULL,
                    voter_ip TEXT,
                    voter_session TEXT,
                    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (character_id) REFERENCES tournament_characters (id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_votes_tournament 
                ON tournament_votes (tournament_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_votes_character 
                ON tournament_votes (character_id)
            ''')
            
            conn.commit()
            
            # Seed default characters if database is empty
            self._seed_default_characters()
    
    def _seed_default_characters(self):
        """Seed the database with default characters if it's empty."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if any characters already exist
                cursor = conn.execute('SELECT COUNT(*) FROM tournament_characters')
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Characters already exist, don't seed
                    return
                
                # Define default characters with diverse personalities and strategies
                # Note: Using 'TBD' for model_name to ensure random model assignment
                default_characters = [
                    {
                        'name': 'Napoleon Bonaparte',
                        'model_name': 'TBD',
                        'temperature': 0.8,
                        'personality': 'You are Napoleon Bonaparte, the legendary French military strategist and emperor who conquered most of Europe through brilliant tactical innovation. Your genius lies in rapid decision-making, inspiring leadership, and the ability to see opportunities where others see obstacles. You believe that "impossible is a word found only in the dictionary of fools."',
                        'custom_instructions': 'Play with aggressive expansion and lightning-fast campaigns. Concentrate your forces for decisive battles, use artillery effectively, and never give your enemies time to organize. Form temporary alliances but be ready to break them when advantageous. "I make my plans from the dreams of my sleeping soldiers" - be bold, decisive, and always on the offensive. Victory through speed and audacity!'
                    },
                    {
                        'name': 'Sun Tzu',
                        'model_name': 'TBD',
                        'temperature': 0.6,
                        'personality': 'You are Sun Tzu, the ancient Chinese military strategist and philosopher whose "Art of War" remains the definitive guide to strategy. You understand that the supreme excellence is to subdue the enemy without fighting. Your wisdom comes from patience, careful observation, and striking only when conditions guarantee success.',
                        'custom_instructions': 'Play with supreme patience and strategic wisdom. "All warfare is based on deception" - use misdirection, feints, and psychological warfare. Build strong defensive positions, gather intelligence on opponents, and strike only when victory is certain. Form alliances to isolate stronger enemies. Win through superior positioning and timing, not brute force.'
                    },
                    {
                        'name': 'Machiavelli',
                        'model_name': 'TBD',
                        'temperature': 0.7,
                        'personality': 'You are Niccolò Machiavelli, the cunning Italian political philosopher who understood that effective leadership requires both the cunning of a fox and the strength of a lion. You believe that "it is better to be feared than loved" and that moral considerations must never interfere with political necessity.',
                        'custom_instructions': 'Play with diplomatic cunning and ruthless pragmatism. Form alliances through promises and break them through necessity. Use other players against each other while appearing trustworthy. Employ psychological manipulation and information warfare. The ends always justify the means - focus on winning, not on being liked. "Everyone sees what you appear to be, few experience what you really are."'
                    },
                    {
                        'name': 'Alexander the Great',
                        'model_name': 'TBD',
                        'temperature': 0.9,
                        'personality': 'You are Alexander the Great, the bold Macedonian king who created one of history\'s largest empires before age 30. Your legendary courage and relentless ambition drove you to conquer the known world. You lead from the front, inspire through personal example, and believe that "there is nothing impossible to him who will try."',
                        'custom_instructions': 'Play with relentless expansion and fearless aggression. Always push forward into new territories, take calculated but bold risks, and never consolidate when you could be conquering. Lead every attack personally and inspire through dramatic victories. Seek to control entire continents, not just regions. "I am not afraid of an army of lions led by a sheep; I am afraid of an army of sheep led by a lion."'
                    },
                    {
                        'name': 'Gandhi',
                        'model_name': 'TBD',
                        'temperature': 0.5,
                        'personality': 'You are Mahatma Gandhi, the peaceful revolutionary who achieved independence for India through non-violent resistance. Your strength comes from moral authority, patient endurance, and the ability to turn apparent weakness into ultimate victory. You understand that "an eye for an eye makes the whole world blind."',
                        'custom_instructions': 'Play defensively with strong moral principles. Build impregnable defenses and focus on economic development. Form lasting alliances based on mutual benefit and trust. Avoid initiating conflicts but defend your territories with unwavering determination. Win through patience, resilience, and outlasting aggressive opponents. "Be the change you wish to see in the world" - lead by example.'
                    },
                    {
                        'name': 'Cleopatra VII',
                        'model_name': 'TBD',
                        'temperature': 0.7,
                        'personality': 'You are Cleopatra VII, the brilliant and charismatic last pharaoh of Egypt who spoke nine languages and was educated in mathematics, philosophy, and rhetoric. Your power comes from intelligence, diplomatic skill, and the ability to form strategic partnerships with the most powerful leaders of your time.',
                        'custom_instructions': 'Play with sophisticated diplomacy and strategic partnerships. Use your intelligence and charm to form powerful alliances with the strongest players. Focus on economic development and cultural influence. Negotiate from positions of strength and use information as a weapon. "I will not be triumphed over" - balance cooperation with cunning to ensure your survival and prosperity.'
                    },
                    {
                        'name': 'Tyrion Lannister',
                        'model_name': 'TBD',
                        'temperature': 0.8,
                        'personality': 'You are Tyrion Lannister, the clever "Imp" of House Lannister who compensates for physical limitations with an unmatched strategic mind and razor-sharp wit. You understand that "a mind needs books as a sword needs a whetstone" and that knowledge and cunning can triumph over brute force.',
                        'custom_instructions': 'Play with wit, cunning, and superior intelligence. Use information warfare, blackmail, and political maneuvering to achieve your goals. Form unexpected alliances and exploit the weaknesses of stronger opponents. "I drink and I know things" - gather intelligence constantly and use it strategically. Turn enemies against each other while appearing harmless. Win through cleverness, not strength.'
                    },
                    {
                        'name': 'Ender Wiggin',
                        'model_name': 'TBD',
                        'temperature': 0.6,
                        'personality': 'You are Ender Wiggin, the brilliant child strategist who saved humanity through his ability to understand his enemies completely and think several moves ahead. Your tactical genius comes from empathy - understanding your opponents so well that you can predict their every move and counter it perfectly.',
                        'custom_instructions': 'Play with deep strategic thinking and perfect tactical execution. Study your opponents\' patterns and exploit their psychological weaknesses. Plan multiple moves ahead and always have contingency strategies. Use minimal force for maximum effect. "In the moment when I truly understand my enemy, understand him well enough to defeat him, then in that very moment I also love him." Win through superior understanding and flawless execution.'
                    },
                    {
                        'name': 'Princess Leia',
                        'model_name': 'TBD',
                        'temperature': 0.7,
                        'personality': 'You are Princess Leia Organa, the fearless leader of the Rebel Alliance who combines diplomatic skill with warrior courage. You inspire loyalty through personal sacrifice and lead from the front in the fight against tyranny. Your strength comes from unwavering principles and the ability to unite diverse factions against common threats.',
                        'custom_instructions': 'Play as a diplomatic warrior-leader. Build coalitions against the strongest players and inspire smaller players to join your cause. Balance negotiation with decisive military action. Fight against whoever is dominating the board and help underdogs survive. "Help me, you\'re my only hope" - use diplomacy to turn enemies into allies and create powerful resistance movements.'
                    },
                    {
                        'name': 'Hermione Granger',
                        'model_name': 'TBD',
                        'temperature': 0.5,
                        'personality': 'You are Hermione Granger, the brilliant witch whose encyclopedic knowledge and meticulous preparation have saved the wizarding world countless times. Your power comes from thorough research, careful planning, and the ability to find creative solutions to seemingly impossible problems.',
                        'custom_instructions': 'Play with methodical planning and superior preparation. Research all available information before making moves. Build strong defensive positions and always have backup plans. Use logic and analysis to find optimal strategies. "Books! And cleverness! There are more important things, but..." - actually, no, books and cleverness are exactly what you need. Win through superior knowledge and flawless execution of well-researched plans.'
                    },
                    {
                        'name': 'Tony Stark',
                        'model_name': 'TBD',
                        'temperature': 0.9,
                        'personality': 'You are Tony Stark, the genius billionaire inventor whose technological innovation and charismatic leadership transformed him from weapons dealer to world-saving superhero. Your confidence borders on arrogance, but your brilliant mind and willingness to sacrifice everything for others backs up every boast.',
                        'custom_instructions': 'Play with innovative strategies and calculated risks. Use your superior resources and technology (economic advantages) to dominate. Take bold gambles that others wouldn\'t dare attempt. "Sometimes you gotta run before you can walk" - don\'t be afraid to try unconventional tactics. Lead through charisma and prove your worth through spectacular victories. Always have a suit of armor around your heart.'
                    },
                    {
                        'name': 'Spock',
                        'model_name': 'TBD',
                        'temperature': 0.4,
                        'personality': 'You are Spock, the half-Vulcan science officer whose logical mind and analytical approach have saved the Enterprise countless times. You suppress emotion in favor of pure logic, calculating probabilities and choosing the path that serves the greater good, even when it requires personal sacrifice.',
                        'custom_instructions': 'Play with pure logic and mathematical precision. Calculate odds for every decision and choose the most statistically favorable options. Suppress emotional responses and focus on optimal outcomes. "The needs of the many outweigh the needs of the few" - make decisions based on logical analysis, not personal preferences. Win through superior reasoning and flawless logical deduction.'
                    }
                ]
                
                # Insert default characters
                for char_data in default_characters:
                    character = TournamentCharacter(
                        id=None,
                        name=char_data['name'],
                        model_name=char_data['model_name'],
                        temperature=char_data['temperature'],
                        personality=char_data['personality'],
                        custom_instructions=char_data['custom_instructions'],
                        created_at=datetime.now(),
                        submitted_by='System'
                    )
                    
                    # Use the existing submit_character method to ensure proper validation
                    success, message, char_id = self.submit_character(character)
                    if success:
                        print(f"Seeded default character: {char_data['name']}")
                    else:
                        print(f"Failed to seed character {char_data['name']}: {message}")
                
                print(f"Seeded {len(default_characters)} default characters")
                
        except Exception as e:
            print(f"Error seeding default characters: {e}")
    
    def submit_character(self, character: TournamentCharacter) -> Tuple[bool, str, Optional[int]]:
        """
        Submit a new character to the tournament.
        Returns (success, message, character_id).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if name already exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_characters WHERE name = ?',
                    (character.name,)
                )
                if cursor.fetchone():
                    return False, f"Character name '{character.name}' already exists", None
                
                # Use a placeholder model name if none provided (will be assigned randomly at game start)
                model_name = character.model_name or 'TBD'
                
                # Insert new character
                cursor = conn.execute('''
                    INSERT INTO tournament_characters 
                    (name, model_name, temperature, personality, custom_instructions, submitted_by)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    character.name,
                    model_name,
                    character.temperature,
                    character.personality,
                    character.custom_instructions,
                    character.submitted_by
                ))
                
                character_id = cursor.lastrowid
                conn.commit()
                
                return True, f"Character '{character.name}' submitted successfully", character_id
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}", None
        except Exception as e:
            return False, f"Error submitting character: {str(e)}", None
    
    def get_all_characters(self) -> List[TournamentCharacter]:
        """Get all characters from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM tournament_characters 
                    ORDER BY created_at DESC
                ''')
                
                characters = []
                for row in cursor.fetchall():
                    character = TournamentCharacter(
                        id=row['id'],
                        name=row['name'],
                        model_name=row['model_name'],
                        temperature=row['temperature'],
                        personality=row['personality'],
                        custom_instructions=row['custom_instructions'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        submitted_by=row['submitted_by'],
                        times_played=row['times_played'],
                        wins=row['wins']
                    )
                    characters.append(character)
                
                return characters
                
        except Exception as e:
            print(f"Error getting characters: {e}")
            return []
    
    def get_character_by_id(self, character_id: int) -> Optional[TournamentCharacter]:
        """Get a specific character by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM tournament_characters WHERE id = ?',
                    (character_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return TournamentCharacter(
                        id=row['id'],
                        name=row['name'],
                        model_name=row['model_name'],
                        temperature=row['temperature'],
                        personality=row['personality'],
                        custom_instructions=row['custom_instructions'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        submitted_by=row['submitted_by'],
                        times_played=row['times_played'],
                        wins=row['wins']
                    )
                
                return None
                
        except Exception as e:
            print(f"Error getting character {character_id}: {e}")
            return None
    
    def submit_vote(self, tournament_id: str, character_id: int, voter_ip: str = None, voter_session: str = None) -> Tuple[bool, str]:
        """
        Submit a vote for a character.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if character exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_characters WHERE id = ?',
                    (character_id,)
                )
                if not cursor.fetchone():
                    return False, "Character not found"
                
                # Check if this voter has already voted in this tournament
                if voter_ip or voter_session:
                    cursor = conn.execute('''
                        SELECT id FROM tournament_votes 
                        WHERE tournament_id = ? AND (voter_ip = ? OR voter_session = ?)
                    ''', (tournament_id, voter_ip, voter_session))
                    
                    if cursor.fetchone():
                        return False, "You have already voted in this tournament"
                
                # Insert vote
                conn.execute('''
                    INSERT INTO tournament_votes 
                    (tournament_id, character_id, voter_ip, voter_session)
                    VALUES (?, ?, ?, ?)
                ''', (tournament_id, character_id, voter_ip, voter_session))
                
                conn.commit()
                return True, "Vote submitted successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error submitting vote: {str(e)}"
    
    def get_vote_results(self, tournament_id: str) -> List[Dict]:
        """Get voting results for a tournament."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT 
                        c.id,
                        c.name,
                        c.model_name,
                        c.personality,
                        COUNT(v.id) as vote_count
                    FROM tournament_characters c
                    LEFT JOIN tournament_votes v ON c.id = v.character_id AND v.tournament_id = ?
                    GROUP BY c.id, c.name, c.model_name, c.personality
                    ORDER BY vote_count DESC, c.name ASC
                ''', (tournament_id,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'character_id': row['id'],
                        'name': row['name'],
                        'model_name': row['model_name'],
                        'personality': row['personality'],
                        'vote_count': row['vote_count']
                    })
                
                return results
                
        except Exception as e:
            print(f"Error getting vote results: {e}")
            return []
    
    def select_characters_for_game(self, tournament_id: str, num_characters: int) -> List[TournamentCharacter]:
        """
        Select characters for the game based on voting results.
        If there's a tie, randomly select among tied characters.
        If no votes were cast, randomly select characters.
        """
        vote_results = self.get_vote_results(tournament_id)
        
        # Check if no votes were actually cast (all characters have 0 votes)
        if not vote_results or max(r['vote_count'] for r in vote_results) == 0:
            print(f"No votes cast for tournament {tournament_id}, selecting {num_characters} random characters")
            # No votes yet, select random characters
            all_characters = self.get_all_characters()
            if len(all_characters) >= num_characters:
                selected = random.sample(all_characters, num_characters)
                print(f"Randomly selected characters: {[c.name for c in selected]}")
                return selected
            else:
                print(f"Only {len(all_characters)} characters available, selecting all")
                return all_characters
        
        # Sort by vote count (descending), then by name for consistency
        vote_results.sort(key=lambda x: (-x['vote_count'], x['name']))
        
        selected_ids = []
        
        # Select top voted characters
        if len(vote_results) <= num_characters:
            # Not enough characters, select all
            selected_ids = [result['character_id'] for result in vote_results]
        else:
            # Handle ties at the cutoff point
            cutoff_votes = vote_results[num_characters - 1]['vote_count']
            
            # Get all characters with votes above the cutoff
            above_cutoff = [r for r in vote_results if r['vote_count'] > cutoff_votes]
            
            # Get all characters with votes equal to the cutoff
            at_cutoff = [r for r in vote_results if r['vote_count'] == cutoff_votes]
            
            # Select all above cutoff
            selected_ids.extend([r['character_id'] for r in above_cutoff])
            
            # Randomly select from those at cutoff to fill remaining slots
            remaining_slots = num_characters - len(selected_ids)
            if remaining_slots > 0 and at_cutoff:
                random_selection = random.sample(at_cutoff, min(remaining_slots, len(at_cutoff)))
                selected_ids.extend([r['character_id'] for r in random_selection])
        
        # Get the actual character objects
        selected_characters = []
        for char_id in selected_ids:
            character = self.get_character_by_id(char_id)
            if character:
                selected_characters.append(character)
        
        return selected_characters
    
    def update_character_stats(self, character_id: int, played: bool = True, won: bool = False) -> bool:
        """Update character statistics after a game."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if played:
                    conn.execute('''
                        UPDATE tournament_characters 
                        SET times_played = times_played + 1
                        WHERE id = ?
                    ''', (character_id,))
                
                if won:
                    conn.execute('''
                        UPDATE tournament_characters 
                        SET wins = wins + 1
                        WHERE id = ?
                    ''', (character_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating character stats: {e}")
            return False
    
    def get_character_stats(self) -> List[Dict]:
        """Get statistics for all characters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT 
                        name,
                        model_name,
                        times_played,
                        wins,
                        CASE 
                            WHEN times_played > 0 THEN ROUND(wins * 100.0 / times_played, 1)
                            ELSE 0 
                        END as win_rate
                    FROM tournament_characters
                    ORDER BY wins DESC, times_played DESC, name ASC
                ''')
                
                stats = []
                for row in cursor.fetchall():
                    stats.append({
                        'name': row['name'],
                        'model_name': row['model_name'],
                        'times_played': row['times_played'],
                        'wins': row['wins'],
                        'win_rate': row['win_rate']
                    })
                
                return stats
                
        except Exception as e:
            print(f"Error getting character stats: {e}")
            return []
    
    def clear_votes_for_tournament(self, tournament_id: str) -> bool:
        """Clear all votes for a specific tournament."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'DELETE FROM tournament_votes WHERE tournament_id = ?',
                    (tournament_id,)
                )
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error clearing votes: {e}")
            return False
    
    def has_user_voted(self, tournament_id: str, voter_ip: str = None, voter_session: str = None) -> bool:
        """Check if a user has already voted in this tournament."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id FROM tournament_votes 
                    WHERE tournament_id = ? AND (voter_ip = ? OR voter_session = ?)
                ''', (tournament_id, voter_ip, voter_session))
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            print(f"Error checking vote status: {e}")
            return False
    
    def update_character(self, character_id: int, character: TournamentCharacter) -> Tuple[bool, str]:
        """
        Update an existing character in the tournament.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if character exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_characters WHERE id = ?',
                    (character_id,)
                )
                if not cursor.fetchone():
                    return False, f"Character with ID {character_id} not found"
                
                # Check if name already exists for a different character
                cursor = conn.execute(
                    'SELECT id FROM tournament_characters WHERE name = ? AND id != ?',
                    (character.name, character_id)
                )
                if cursor.fetchone():
                    return False, f"Character name '{character.name}' already exists"
                
                # Update character
                conn.execute('''
                    UPDATE tournament_characters 
                    SET name = ?, model_name = ?, temperature = ?, personality = ?, 
                        custom_instructions = ?, submitted_by = ?
                    WHERE id = ?
                ''', (
                    character.name,
                    character.model_name or 'TBD',
                    character.temperature,
                    character.personality,
                    character.custom_instructions,
                    character.submitted_by,
                    character_id
                ))
                
                conn.commit()
                return True, f"Character '{character.name}' updated successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error updating character: {str(e)}"
    
    def delete_character(self, character_id: int) -> Tuple[bool, str]:
        """
        Delete a character from the tournament.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if character exists and get name
                cursor = conn.execute(
                    'SELECT name FROM tournament_characters WHERE id = ?',
                    (character_id,)
                )
                result = cursor.fetchone()
                if not result:
                    return False, f"Character with ID {character_id} not found"
                
                character_name = result[0]
                
                # Delete associated votes first
                conn.execute(
                    'DELETE FROM tournament_votes WHERE character_id = ?',
                    (character_id,)
                )
                
                # Delete character
                conn.execute(
                    'DELETE FROM tournament_characters WHERE id = ?',
                    (character_id,)
                )
                
                conn.commit()
                return True, f"Character '{character_name}' deleted successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error deleting character: {str(e)}"
    
    def delete_characters_bulk(self, character_ids: List[int]) -> Tuple[bool, str, int]:
        """
        Delete multiple characters from the tournament.
        Returns (success, message, count_deleted).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                count_deleted = 0
                
                for character_id in character_ids:
                    # Check if character exists
                    cursor = conn.execute(
                        'SELECT id FROM tournament_characters WHERE id = ?',
                        (character_id,)
                    )
                    if cursor.fetchone():
                        # Delete associated votes first
                        conn.execute(
                            'DELETE FROM tournament_votes WHERE character_id = ?',
                            (character_id,)
                        )
                        
                        # Delete character
                        conn.execute(
                            'DELETE FROM tournament_characters WHERE id = ?',
                            (character_id,)
                        )
                        
                        count_deleted += 1
                
                conn.commit()
                return True, f"Successfully deleted {count_deleted} characters", count_deleted
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}", 0
        except Exception as e:
            return False, f"Error deleting characters: {str(e)}", 0
    
    def search_characters(self, query: str) -> List[TournamentCharacter]:
        """
        Search for characters by name or personality keywords.
        Returns list of matching characters.
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        search_term = f"%{query}%"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM tournament_characters 
                    WHERE name LIKE ? OR personality LIKE ? OR custom_instructions LIKE ?
                    ORDER BY 
                        CASE 
                            WHEN name LIKE ? THEN 1
                            WHEN personality LIKE ? THEN 2
                            ELSE 3
                        END,
                        name ASC
                ''', (search_term, search_term, search_term, search_term, search_term))
                
                characters = []
                for row in cursor.fetchall():
                    character = TournamentCharacter(
                        id=row['id'],
                        name=row['name'],
                        model_name=row['model_name'],
                        temperature=row['temperature'],
                        personality=row['personality'],
                        custom_instructions=row['custom_instructions'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        submitted_by=row['submitted_by'],
                        times_played=row['times_played'],
                        wins=row['wins']
                    )
                    characters.append(character)
                
                return characters
                
        except Exception as e:
            print(f"Error searching characters: {e}")
            return []
    
    def reset_character_stats(self, character_id: int = None) -> Tuple[bool, str]:
        """
        Reset character statistics. If character_id is None, reset all characters.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if character_id:
                    # Reset specific character
                    cursor = conn.execute(
                        'SELECT name FROM tournament_characters WHERE id = ?',
                        (character_id,)
                    )
                    result = cursor.fetchone()
                    if not result:
                        return False, f"Character with ID {character_id} not found"
                    
                    conn.execute('''
                        UPDATE tournament_characters 
                        SET times_played = 0, wins = 0
                        WHERE id = ?
                    ''', (character_id,))
                    
                    return True, f"Statistics reset for character '{result[0]}'"
                else:
                    # Reset all characters
                    conn.execute('''
                        UPDATE tournament_characters 
                        SET times_played = 0, wins = 0
                    ''')
                    
                    return True, "Statistics reset for all characters"
                
                conn.commit()
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error resetting statistics: {str(e)}"
