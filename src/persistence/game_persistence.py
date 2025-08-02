#!/usr/bin/env python3

import json
import os
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from pathlib import Path

from utils.logger import risk_logger

# Handle imports for both Docker and local environments
try:
    # Try Docker-style import first (when running from /app directory)
    from persistence.action_tracker import get_default_data_dir
except ImportError:
    # Fall back to local-style import (when running from project root)
    from src.persistence.action_tracker import get_default_data_dir

class GamePersistence:
    """Handles persistent storage of game state."""
    
    def __init__(self, data_dir: Optional[str] = None):
        # Use the provided directory, or get the default
        self.data_dir = Path(data_dir) if data_dir else Path(get_default_data_dir())
        self.games_dir = self.data_dir / 'games'
        self.lock = threading.RLock()
        
        # Ensure directories exist
        try:
            self.games_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            risk_logger.log_error(f"Cannot create games directory {self.games_dir}: {e}")
            # Fall back to a temp directory if we can't create the games directory
            import tempfile
            self.data_dir = Path(tempfile.gettempdir()) / 'risk_data'
            self.games_dir = self.data_dir / 'games'
            self.games_dir.mkdir(parents=True, exist_ok=True)
            risk_logger.log_warning(f"Falling back to temporary directory: {self.data_dir}")
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.save_interval = 30  # seconds
        self.last_save_time = {}
        
        risk_logger.log_info(f"Game persistence initialized: {self.games_dir}")
    
    def get_game_file_path(self, game_id: str) -> Path:
        """Get the file path for a game's persistent state."""
        return self.games_dir / f"{game_id}.json"
    
    def save_game_state(self, game_id: str, game_state: Dict[str, Any]) -> bool:
        """Save game state to persistent storage."""
        try:
            with self.lock:
                game_file = self.get_game_file_path(game_id)
                temp_file = game_file.with_suffix('.tmp')
                
                # Add metadata
                save_data = {
                    'game_id': game_id,
                    'saved_at': datetime.utcnow().isoformat(),
                    'version': '1.0',
                    'game_state': game_state
                }
                
                # Atomic write using temporary file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, default=str)
                
                # Move temp file to final location
                temp_file.replace(game_file)
                
                self.last_save_time[game_id] = time.time()
                risk_logger.log_info(f"Game state saved: {game_id}")
                return True
                
        except Exception as e:
            risk_logger.log_error(f"Failed to save game state {game_id}: {e}")
            return False
    
    def load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Load game state from persistent storage."""
        try:
            with self.lock:
                game_file = self.get_game_file_path(game_id)
                
                if not game_file.exists():
                    return None
                
                with open(game_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # Validate save data
                if 'game_state' not in save_data:
                    risk_logger.log_error(f"Invalid save data for {game_id}: missing game_state")
                    return None
                
                risk_logger.log_info(f"Game state loaded: {game_id}")
                return save_data['game_state']
                
        except Exception as e:
            risk_logger.log_error(f"Failed to load game state {game_id}: {e}")
            return None
    
    def delete_game_state(self, game_id: str) -> bool:
        """Delete game state from persistent storage."""
        try:
            with self.lock:
                game_file = self.get_game_file_path(game_id)
                
                if game_file.exists():
                    game_file.unlink()
                    risk_logger.log_info(f"Game state deleted: {game_id}")
                
                # Clean up tracking
                self.last_save_time.pop(game_id, None)
                return True
                
        except Exception as e:
            risk_logger.log_error(f"Failed to delete game state {game_id}: {e}")
            return False
    
    def list_saved_games(self) -> list[str]:
        """List all saved game IDs."""
        try:
            with self.lock:
                game_files = self.games_dir.glob("*.json")
                return [f.stem for f in game_files]
        except Exception as e:
            risk_logger.log_error(f"Failed to list saved games: {e}")
            return []
    
    def should_auto_save(self, game_id: str) -> bool:
        """Check if game should be auto-saved based on time interval."""
        if not self.auto_save_enabled:
            return False
        
        last_save = self.last_save_time.get(game_id, 0)
        return (time.time() - last_save) >= self.save_interval
    
    def cleanup_old_saves(self, max_age_days: int = 30) -> int:
        """Clean up old save files."""
        try:
            with self.lock:
                cutoff_time = time.time() - (max_age_days * 24 * 3600)
                cleaned = 0
                
                for game_file in self.games_dir.glob("*.json"):
                    if game_file.stat().st_mtime < cutoff_time:
                        game_file.unlink()
                        cleaned += 1
                        risk_logger.log_info(f"Cleaned up old save: {game_file.stem}")
                
                return cleaned
                
        except Exception as e:
            risk_logger.log_error(f"Failed to cleanup old saves: {e}")
            return 0
    
    def get_save_info(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a saved game."""
        try:
            with self.lock:
                game_file = self.get_game_file_path(game_id)
                
                if not game_file.exists():
                    return None
                
                with open(game_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                return {
                    'game_id': save_data.get('game_id'),
                    'saved_at': save_data.get('saved_at'),
                    'version': save_data.get('version'),
                    'file_size': game_file.stat().st_size,
                    'file_path': str(game_file)
                }
                
        except Exception as e:
            risk_logger.log_error(f"Failed to get save info for {game_id}: {e}")
            return None

# Global persistence instance
game_persistence = GamePersistence()
