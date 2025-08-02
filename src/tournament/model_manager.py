"""Model management for tournament mode."""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TournamentModel:
    """Represents an AI model available for tournament characters."""
    id: Optional[int]
    name: str
    display_name: str
    default_temperature: float
    description: str
    is_active: bool
    created_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'default_temperature': self.default_temperature,
            'description': self.description,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TournamentModel':
        """Create from dictionary."""
        created_at = data['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            id=data.get('id'),
            name=data['name'],
            display_name=data['display_name'],
            default_temperature=data['default_temperature'],
            description=data['description'],
            is_active=data['is_active'],
            created_at=created_at
        )

class ModelManager:
    """Manages AI models for tournament characters."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tournament_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    default_temperature REAL NOT NULL DEFAULT 0.7,
                    description TEXT NOT NULL DEFAULT '',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            
            # Seed models from environment if database is empty
            self._seed_models_from_env()
    
    def _seed_models_from_env(self):
        """Seed the database with models from environment variable if database is empty."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if any models already exist
                cursor = conn.execute('SELECT COUNT(*) FROM tournament_models')
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Models already exist, don't seed
                    return
                
                # Load predefined models from environment variable
                models_json = os.getenv('RISK_PREDEFINED_MODELS', '{}')
                try:
                    predefined_models = json.loads(models_json)
                except json.JSONDecodeError:
                    print("⚠️  Warning: Invalid RISK_PREDEFINED_MODELS format, using empty models")
                    predefined_models = {}
                
                if not predefined_models:
                    print("No predefined models found in environment")
                    return
                
                # Insert models from environment
                for display_name, model_info in predefined_models.items():
                    model = TournamentModel(
                        id=None,
                        name=model_info.get('name', display_name),
                        display_name=display_name,
                        default_temperature=model_info.get('default_temp', 0.7),
                        description=model_info.get('description', ''),
                        is_active=True,
                        created_at=datetime.now()
                    )
                    
                    # Use the existing add_model method to ensure proper validation
                    success, message = self.add_model(model)
                    if success:
                        print(f"Seeded model: {display_name} ({model.name})")
                    else:
                        print(f"Failed to seed model {display_name}: {message}")
                
                print(f"Seeded {len(predefined_models)} models from environment")
                
        except Exception as e:
            print(f"Error seeding models from environment: {e}")
    
    def get_all_models(self) -> List[TournamentModel]:
        """Get all models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM tournament_models 
                    ORDER BY display_name ASC
                ''')
                
                models = []
                for row in cursor.fetchall():
                    model = TournamentModel(
                        id=row['id'],
                        name=row['name'],
                        display_name=row['display_name'],
                        default_temperature=row['default_temperature'],
                        description=row['description'],
                        is_active=bool(row['is_active']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                    models.append(model)
                
                return models
                
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
    
    def get_active_models(self) -> List[TournamentModel]:
        """Get only active models."""
        return [model for model in self.get_all_models() if model.is_active]
    
    def get_model_by_id(self, model_id: int) -> Optional[TournamentModel]:
        """Get a specific model by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM tournament_models WHERE id = ?',
                    (model_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return TournamentModel(
                        id=row['id'],
                        name=row['name'],
                        display_name=row['display_name'],
                        default_temperature=row['default_temperature'],
                        description=row['description'],
                        is_active=bool(row['is_active']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                
                return None
                
        except Exception as e:
            print(f"Error getting model {model_id}: {e}")
            return None
    
    def get_model_by_name(self, name: str) -> Optional[TournamentModel]:
        """Get a model by its name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM tournament_models WHERE name = ?',
                    (name,)
                )
                
                row = cursor.fetchone()
                if row:
                    return TournamentModel(
                        id=row['id'],
                        name=row['name'],
                        display_name=row['display_name'],
                        default_temperature=row['default_temperature'],
                        description=row['description'],
                        is_active=bool(row['is_active']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                
                return None
                
        except Exception as e:
            print(f"Error getting model by name {name}: {e}")
            return None
    
    def add_model(self, model: TournamentModel) -> Tuple[bool, str]:
        """
        Add a new model to the tournament.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if name already exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_models WHERE name = ?',
                    (model.name,)
                )
                if cursor.fetchone():
                    return False, f"Model name '{model.name}' already exists"
                
                # Check if display name already exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_models WHERE display_name = ?',
                    (model.display_name,)
                )
                if cursor.fetchone():
                    return False, f"Display name '{model.display_name}' already exists"
                
                # Insert new model
                cursor = conn.execute('''
                    INSERT INTO tournament_models 
                    (name, display_name, default_temperature, description, is_active)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    model.name,
                    model.display_name,
                    model.default_temperature,
                    model.description,
                    model.is_active
                ))
                
                model_id = cursor.lastrowid
                conn.commit()
                
                return True, f"Model '{model.display_name}' added successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error adding model: {str(e)}"
    
    def update_model(self, model_id: int, model: TournamentModel) -> Tuple[bool, str]:
        """
        Update an existing model.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if model exists
                cursor = conn.execute(
                    'SELECT id FROM tournament_models WHERE id = ?',
                    (model_id,)
                )
                if not cursor.fetchone():
                    return False, f"Model with ID {model_id} not found"
                
                # Check if name already exists for a different model
                cursor = conn.execute(
                    'SELECT id FROM tournament_models WHERE name = ? AND id != ?',
                    (model.name, model_id)
                )
                if cursor.fetchone():
                    return False, f"Model name '{model.name}' already exists"
                
                # Check if display name already exists for a different model
                cursor = conn.execute(
                    'SELECT id FROM tournament_models WHERE display_name = ? AND id != ?',
                    (model.display_name, model_id)
                )
                if cursor.fetchone():
                    return False, f"Display name '{model.display_name}' already exists"
                
                # Update model
                conn.execute('''
                    UPDATE tournament_models 
                    SET name = ?, display_name = ?, default_temperature = ?, 
                        description = ?, is_active = ?
                    WHERE id = ?
                ''', (
                    model.name,
                    model.display_name,
                    model.default_temperature,
                    model.description,
                    model.is_active,
                    model_id
                ))
                
                conn.commit()
                return True, f"Model '{model.display_name}' updated successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error updating model: {str(e)}"
    
    def delete_model(self, model_id: int) -> Tuple[bool, str]:
        """
        Delete a model from the tournament.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if model exists and get name
                cursor = conn.execute(
                    'SELECT display_name FROM tournament_models WHERE id = ?',
                    (model_id,)
                )
                result = cursor.fetchone()
                if not result:
                    return False, f"Model with ID {model_id} not found"
                
                model_name = result[0]
                
                # Check if any characters are using this model
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM tournament_characters WHERE model_name IN (SELECT name FROM tournament_models WHERE id = ?)',
                    (model_id,)
                )
                character_count = cursor.fetchone()[0]
                
                if character_count > 0:
                    return False, f"Cannot delete model '{model_name}' - {character_count} characters are using it"
                
                # Delete model
                conn.execute(
                    'DELETE FROM tournament_models WHERE id = ?',
                    (model_id,)
                )
                
                conn.commit()
                return True, f"Model '{model_name}' deleted successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"
    
    def toggle_model_status(self, model_id: int) -> Tuple[bool, str]:
        """
        Toggle the active status of a model.
        Returns (success, message).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current status
                cursor = conn.execute(
                    'SELECT display_name, is_active FROM tournament_models WHERE id = ?',
                    (model_id,)
                )
                result = cursor.fetchone()
                if not result:
                    return False, f"Model with ID {model_id} not found"
                
                model_name, is_active = result
                new_status = not bool(is_active)
                
                # Update status
                conn.execute(
                    'UPDATE tournament_models SET is_active = ? WHERE id = ?',
                    (new_status, model_id)
                )
                
                conn.commit()
                status_text = "activated" if new_status else "deactivated"
                return True, f"Model '{model_name}' {status_text} successfully"
                
        except sqlite3.Error as e:
            return False, f"Database error: {str(e)}"
        except Exception as e:
            return False, f"Error toggling model status: {str(e)}"
    
    def get_model_usage_stats(self) -> List[Dict]:
        """Get usage statistics for all models using actual runtime assignments."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # First, try to get data from action tracker if available
                try:
                    from persistence.action_tracker import get_action_tracker
                    
                    tracker = get_action_tracker()
                    if tracker:
                        # Get model usage from the last 30 days from action tracker
                        tracker_stats = tracker.get_model_usage_stats(days=30)
                        
                        # Get all models from our database
                        cursor = conn.execute('''
                            SELECT id, name, display_name, is_active
                            FROM tournament_models
                            ORDER BY display_name ASC
                        ''')
                        
                        stats = []
                        for row in cursor.fetchall():
                            model_name = row['name']
                            usage_count = tracker_stats.get(model_name, 0)
                            
                            # For character count, we'll count how many different characters 
                            # have been assigned this model (even though characters store 'TBD')
                            character_count = 0
                            if tracker:
                                # Get unique character names that have used this model
                                try:
                                    character_assignments = tracker.get_model_character_assignments(model_name)
                                    character_count = len(set(character_assignments))
                                except:
                                    character_count = 0
                            
                            # Estimate wins based on average win rate (since we don't track wins per model directly)
                            # This is a rough approximation - we could improve this later
                            estimated_wins = usage_count // 4 if usage_count > 0 else 0  # Assume ~25% win rate
                            win_rate = (estimated_wins / usage_count * 100) if usage_count > 0 else 0
                            
                            stats.append({
                                'id': row['id'],
                                'name': row['name'],
                                'display_name': row['display_name'],
                                'is_active': bool(row['is_active']),
                                'character_count': character_count,
                                'total_games': usage_count,
                                'total_wins': estimated_wins,
                                'win_rate': round(win_rate, 1)
                            })
                        
                        return stats
                        
                except ImportError:
                    print("Action tracker not available, falling back to character-based stats")
                except Exception as e:
                    print(f"Error getting tracker stats: {e}, falling back to character-based stats")
                
                # Fallback to original method if action tracker unavailable
                cursor = conn.execute('''
                    SELECT 
                        m.id,
                        m.name,
                        m.display_name,
                        m.is_active,
                        COUNT(CASE WHEN c.model_name != 'TBD' THEN c.id END) as character_count,
                        COALESCE(SUM(CASE WHEN c.model_name != 'TBD' THEN c.times_played ELSE 0 END), 0) as total_games,
                        COALESCE(SUM(CASE WHEN c.model_name != 'TBD' THEN c.wins ELSE 0 END), 0) as total_wins,
                        CASE 
                            WHEN SUM(CASE WHEN c.model_name != 'TBD' THEN c.times_played ELSE 0 END) > 0 THEN 
                                ROUND(SUM(CASE WHEN c.model_name != 'TBD' THEN c.wins ELSE 0 END) * 100.0 / 
                                      SUM(CASE WHEN c.model_name != 'TBD' THEN c.times_played ELSE 0 END), 1)
                            ELSE 0 
                        END as win_rate
                    FROM tournament_models m
                    LEFT JOIN tournament_characters c ON m.name = c.model_name
                    GROUP BY m.id, m.name, m.display_name, m.is_active
                    ORDER BY total_games DESC, m.display_name ASC
                ''')
                
                stats = []
                for row in cursor.fetchall():
                    stats.append({
                        'id': row['id'],
                        'name': row['name'],
                        'display_name': row['display_name'],
                        'is_active': bool(row['is_active']),
                        'character_count': row['character_count'],
                        'total_games': row['total_games'],
                        'total_wins': row['total_wins'],
                        'win_rate': row['win_rate']
                    })
                
                return stats
                
        except Exception as e:
            print(f"Error getting model usage stats: {e}")
            return []
    
    def sync_with_environment(self) -> Tuple[bool, str, int]:
        """
        Sync models with current environment variable.
        Returns (success, message, models_updated).
        """
        try:
            # Load current models from environment
            models_json = os.getenv('RISK_PREDEFINED_MODELS', '{}')
            try:
                env_models = json.loads(models_json)
            except json.JSONDecodeError:
                return False, "Invalid RISK_PREDEFINED_MODELS format", 0
            
            models_updated = 0
            
            with sqlite3.connect(self.db_path) as conn:
                for display_name, model_info in env_models.items():
                    model_name = model_info.get('name', display_name)
                    
                    # Check if model exists
                    cursor = conn.execute(
                        'SELECT id FROM tournament_models WHERE name = ?',
                        (model_name,)
                    )
                    
                    if cursor.fetchone():
                        # Update existing model
                        conn.execute('''
                            UPDATE tournament_models 
                            SET display_name = ?, default_temperature = ?, description = ?
                            WHERE name = ?
                        ''', (
                            display_name,
                            model_info.get('default_temp', 0.7),
                            model_info.get('description', ''),
                            model_name
                        ))
                    else:
                        # Add new model
                        conn.execute('''
                            INSERT INTO tournament_models 
                            (name, display_name, default_temperature, description, is_active)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            model_name,
                            display_name,
                            model_info.get('default_temp', 0.7),
                            model_info.get('description', ''),
                            True
                        ))
                    
                    models_updated += 1
                
                conn.commit()
            
            return True, f"Successfully synced {models_updated} models with environment", models_updated
            
        except Exception as e:
            return False, f"Error syncing with environment: {str(e)}", 0
