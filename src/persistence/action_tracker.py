#!/usr/bin/env python3

import os
import json
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from utils.logger import risk_logger

def get_default_data_dir() -> str:
    """
    Determine the most appropriate default data directory based on environment.
    This allows the system to work both inside Docker and in a local development environment.
    """
    # Check if RISK_DATA_DIR is explicitly set
    if data_dir := os.getenv('RISK_DATA_DIR'):
        try:
            os.makedirs(data_dir, exist_ok=True)
            return data_dir
        except (PermissionError, OSError):
            pass  # Fall through to other options
        
    # Check if we're in Docker
    if os.path.exists('/app'):
        data_dir = '/app/data'
        try:
            os.makedirs(data_dir, exist_ok=True)
            # Test write access
            test_file = os.path.join(data_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return data_dir
        except (PermissionError, OSError):
            pass  # Fall through to temp directory
    
    # Check if we're in the project root
    current_dir = Path.cwd()
    if (current_dir / 'src').exists():
        data_dir = str(current_dir / 'data')
        try:
            os.makedirs(data_dir, exist_ok=True)
            return data_dir
        except (PermissionError, OSError):
            pass  # Fall through to other options
    
    # Fall back to data directory relative to this file's location
    src_dir = Path(__file__).parent.parent
    project_root = src_dir.parent
    data_dir = str(project_root / 'data')
    try:
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    except (PermissionError, OSError):
        pass  # Fall through to temp directory
    
    # Final fallback to temp directory
    import tempfile
    temp_dir = os.path.join(tempfile.gettempdir(), 'risk_data')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

class ActionTracker:
    """
    SQLite database tracker for game actions, turns, and results.
    Tracks whose turn it is, turn number, actions taken, and their outputs.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the action tracker with database connection."""
        # Use the provided directory, or get the default
        self.data_dir = Path(data_dir) if data_dir else Path(get_default_data_dir())
        self.db_path = self.data_dir / 'risk_actions.db'
        self.lock = threading.RLock()
        
        # Ensure directory exists
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            risk_logger.log_error(f"Cannot create data directory {self.data_dir}: {e}")
            # Fall back to a temp directory if we can't create the data directory
            import tempfile
            self.data_dir = Path(tempfile.gettempdir()) / 'risk_data'
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.data_dir / 'risk_actions.db'
            risk_logger.log_warning(f"Falling back to temporary directory: {self.data_dir}")
        
        # Initialize database
        self._init_db()
        
        risk_logger.log_info(f"Action tracker initialized: {self.db_path}")
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn
    
    def _init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        with self.lock:
            conn = self._get_db_connection()
            try:
                # Game sessions table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS game_sessions (
                    game_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    current_player_id TEXT,
                    current_turn_number INTEGER NOT NULL DEFAULT 1,
                    game_phase TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    num_players INTEGER NOT NULL
                )
                """)
                
                # Player models table (NEW - for tracking model assignments)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS player_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL,
                    assigned_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id),
                    UNIQUE(game_id, player_id)
                )
                """)
                
                # Player turns table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS player_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    turn_started_at TIMESTAMP NOT NULL,
                    turn_ended_at TIMESTAMP,
                    actions_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Game actions table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS game_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    action_result TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Player strategies table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS player_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('short_term', 'long_term')),
                    strategy_content TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Agent decisions table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    decision_timestamp TIMESTAMP NOT NULL,
                    context_data TEXT NOT NULL,        -- Full JSON context 
                    formatted_context TEXT NOT NULL,   -- The formatted context string
                    agent_prompt TEXT NOT NULL,        -- Complete prompt sent to LLM
                    agent_response TEXT,               -- Full LLM response
                    agent_reasoning TEXT,              -- Extracted reasoning
                    tools_used TEXT,                   -- JSON array of tools used
                    decision_time_seconds REAL,        -- Time taken for decision
                    success BOOLEAN NOT NULL,          -- Whether decision completed successfully
                    error_message TEXT,                -- Any error that occurred
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Context summaries table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS context_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    summary_type TEXT NOT NULL CHECK (summary_type IN ('strategic', 'diplomatic', 'battle_history', 'game_evolution', 'full_context')),
                    content TEXT NOT NULL,
                    turn_range_start INTEGER NOT NULL,
                    turn_range_end INTEGER NOT NULL,
                    original_tokens INTEGER NOT NULL,
                    summary_tokens INTEGER NOT NULL,
                    tokens_saved INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    is_current BOOLEAN DEFAULT 1,
                    version INTEGER DEFAULT 1,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Action failure tracking table (NEW - for loop detection)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS action_failure_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    consecutive_failures INTEGER NOT NULL DEFAULT 1,
                    last_failure_message TEXT NOT NULL,
                    last_failure_timestamp TIMESTAMP NOT NULL,
                    first_failure_timestamp TIMESTAMP NOT NULL,
                    intervention_triggered BOOLEAN DEFAULT 0,
                    intervention_timestamp TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Game completion statistics table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS game_completion_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL UNIQUE,
                    completed_at TIMESTAMP NOT NULL,
                    total_duration_seconds INTEGER NOT NULL,
                    total_turns INTEGER NOT NULL,
                    winner_player_id TEXT,
                    winner_player_name TEXT,
                    completion_reason TEXT NOT NULL,
                    
                    -- Player statistics (JSON)
                    player_statistics TEXT NOT NULL,
                    
                    -- Action statistics
                    total_actions INTEGER NOT NULL,
                    successful_actions INTEGER NOT NULL,
                    failed_actions INTEGER NOT NULL,
                    action_type_breakdown TEXT NOT NULL,
                    
                    -- Model/AI statistics
                    models_used TEXT NOT NULL,
                    avg_decision_time_seconds REAL,
                    total_summarizations INTEGER DEFAULT 0,
                    total_tokens_saved INTEGER DEFAULT 0,
                    
                    -- Game-specific metrics
                    diplomatic_messages INTEGER DEFAULT 0,
                    territories_conquered INTEGER DEFAULT 0,
                    battles_fought INTEGER DEFAULT 0,
                    
                    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
                )
                """)
                
                # Create indexes for better performance
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_summaries_game_current 
                ON context_summaries(game_id, is_current, summary_type)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_summaries_turn_range 
                ON context_summaries(game_id, turn_range_start, turn_range_end)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_completion_stats_completed_at 
                ON game_completion_stats(completed_at)
                """)
                
                conn.commit()
            except Exception as e:
                risk_logger.log_error(f"Failed to initialize action tracker database: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def track_game_start(self, game_id: str, num_players: int, game_phase: str) -> bool:
        """Track a new game session start."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    conn.execute(
                        """
                        INSERT INTO game_sessions
                        (game_id, created_at, game_phase, num_players)
                        VALUES (?, ?, ?, ?)
                        """,
                        (game_id, now, game_phase, num_players)
                    )
                    conn.commit()
                    risk_logger.log_info(f"Tracked game start: {game_id}")
                    return True
                except sqlite3.IntegrityError:
                    risk_logger.log_warning(f"Game already exists in tracker: {game_id}")
                    conn.rollback()
                    return False
                except Exception as e:
                    risk_logger.log_error(f"Failed to track game start {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking game start: {e}")
            return False
    
    def track_turn_start(self, game_id: str, player_id: str, player_name: str, turn_number: int) -> bool:
        """Track the start of a player's turn."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    # First update the game session
                    conn.execute(
                        """
                        UPDATE game_sessions
                        SET current_player_id = ?, current_turn_number = ?
                        WHERE game_id = ?
                        """,
                        (player_id, turn_number, game_id)
                    )
                    
                    # Then record the turn start
                    conn.execute(
                        """
                        INSERT INTO player_turns
                        (game_id, player_id, player_name, turn_number, turn_started_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (game_id, player_id, player_name, turn_number, now)
                    )
                    
                    conn.commit()
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to track turn start for game {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking turn start: {e}")
            return False
    
    def track_turn_end(self, game_id: str, player_id: str, turn_number: int) -> bool:
        """Track the end of a player's turn."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    # Update the player turn
                    conn.execute(
                        """
                        UPDATE player_turns
                        SET turn_ended_at = ?
                        WHERE game_id = ? AND player_id = ? AND turn_number = ? AND turn_ended_at IS NULL
                        """,
                        (now, game_id, player_id, turn_number)
                    )
                    
                    conn.commit()
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to track turn end for game {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking turn end: {e}")
            return False
    
    def track_action(
        self, 
        game_id: str, 
        player_id: str, 
        turn_number: int, 
        action_type: str, 
        action_data: Dict[str, Any], 
        action_result: Dict[str, Any]
    ) -> bool:
        """Track a game action with its input parameters and results."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    # Get the current sequence number for this turn
                    cursor = conn.execute(
                        """
                        SELECT MAX(sequence_number) as max_seq 
                        FROM game_actions 
                        WHERE game_id = ? AND player_id = ? AND turn_number = ?
                        """,
                        (game_id, player_id, turn_number)
                    )
                    result = cursor.fetchone()
                    sequence_number = (result['max_seq'] or 0) + 1
                    
                    # Insert the action
                    conn.execute(
                        """
                        INSERT INTO game_actions
                        (game_id, player_id, turn_number, action_type, action_data, action_result, timestamp, sequence_number)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id, 
                            player_id, 
                            turn_number, 
                            action_type, 
                            json.dumps(action_data), 
                            json.dumps(action_result), 
                            now, 
                            sequence_number
                        )
                    )
                    
                    # Update action count in the player turn
                    conn.execute(
                        """
                        UPDATE player_turns
                        SET actions_count = actions_count + 1
                        WHERE game_id = ? AND player_id = ? AND turn_number = ?
                        """,
                        (game_id, player_id, turn_number)
                    )
                    
                    conn.commit()
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to track action for game {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking action: {e}")
            return False
    
    def update_game_phase(self, game_id: str, game_phase: str) -> bool:
        """Update the current game phase."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    conn.execute(
                        """
                        UPDATE game_sessions
                        SET game_phase = ?
                        WHERE game_id = ?
                        """,
                        (game_phase, game_id)
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to update game phase for {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error updating game phase: {e}")
            return False
    
    def track_player_model(
        self, 
        game_id: str, 
        player_id: str, 
        player_name: str, 
        model_name: str, 
        temperature: Optional[float] = None
    ) -> bool:
        """Track model assignment for a player."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO player_models
                        (game_id, player_id, player_name, model_name, temperature, assigned_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (game_id, player_id, player_name, model_name, temperature, now)
                    )
                    conn.commit()
                    risk_logger.log_info(f"Tracked model assignment: {player_name} -> {model_name}")
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to track player model for {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking player model: {e}")
            return False
    
    def get_player_model(self, game_id: str, player_id: str) -> Optional[str]:
        """Get the model assigned to a player."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT model_name
                        FROM player_models
                        WHERE game_id = ? AND player_id = ?
                        """,
                        (game_id, player_id)
                    )
                    row = cursor.fetchone()
                    return row['model_name'] if row else None
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get player model: {e}")
            return None
    
    def get_game_models(self, game_id: str) -> Dict[str, str]:
        """Get all model assignments for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT player_id, player_name, model_name, temperature
                        FROM player_models
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    models = {}
                    for row in cursor.fetchall():
                        models[row['player_id']] = {
                            'player_name': row['player_name'],
                            'model_name': row['model_name'],
                            'temperature': row['temperature']
                        }
                    return models
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get game models: {e}")
            return {}

    def finish_game(self, game_id: str, status: str = 'completed') -> bool:
        """Mark a game as completed or abandoned."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    conn.execute(
                        """
                        UPDATE game_sessions
                        SET status = ?
                        WHERE game_id = ?
                        """,
                        (status, game_id)
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to finish game {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error finishing game: {e}")
            return False
    
    def get_game_action_history(self, game_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent action history for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT id, player_id, turn_number, action_type, action_data, 
                               action_result, timestamp, sequence_number
                        FROM game_actions
                        WHERE game_id = ?
                        ORDER BY turn_number DESC, timestamp DESC
                        LIMIT ?
                        """,
                        (game_id, limit)
                    )
                    
                    actions = []
                    for row in cursor.fetchall():
                        action = dict(row)
                        action['action_data'] = json.loads(action['action_data'])
                        action['action_result'] = json.loads(action['action_result'])
                        actions.append(action)
                    
                    return actions
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get action history for {game_id}: {e}")
            return []
    
    def get_turn_summary(self, game_id: str, turn_number: int) -> Dict[str, Any]:
        """Get a summary of a specific turn in a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get turn details
                    cursor = conn.execute(
                        """
                        SELECT pt.*, COUNT(ga.id) as action_count
                        FROM player_turns pt
                        LEFT JOIN game_actions ga ON pt.game_id = ga.game_id 
                            AND pt.player_id = ga.player_id 
                            AND pt.turn_number = ga.turn_number
                        WHERE pt.game_id = ? AND pt.turn_number = ?
                        GROUP BY pt.id
                        """,
                        (game_id, turn_number)
                    )
                    
                    turn_row = cursor.fetchone()
                    if not turn_row:
                        return {}
                    
                    turn_data = dict(turn_row)
                    
                    # Get all actions for this turn
                    cursor = conn.execute(
                        """
                        SELECT id, player_id, action_type, action_data, 
                               action_result, timestamp, sequence_number
                        FROM game_actions
                        WHERE game_id = ? AND turn_number = ?
                        ORDER BY timestamp ASC
                        """,
                        (game_id, turn_number)
                    )
                    
                    actions = []
                    for row in cursor.fetchall():
                        action = dict(row)
                        action['action_data'] = json.loads(action['action_data'])
                        action['action_result'] = json.loads(action['action_result'])
                        actions.append(action)
                    
                    turn_data['actions'] = actions
                    
                    return turn_data
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get turn summary for game {game_id}, turn {turn_number}: {e}")
            return {}
    
    def get_player_actions(self, game_id: str, player_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent actions for a specific player in a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT id, turn_number, action_type, action_data, 
                               action_result, timestamp, sequence_number
                        FROM game_actions
                        WHERE game_id = ? AND player_id = ?
                        ORDER BY turn_number DESC, timestamp DESC
                        LIMIT ?
                        """,
                        (game_id, player_id, limit)
                    )
                    
                    actions = []
                    for row in cursor.fetchall():
                        action = dict(row)
                        action['action_data'] = json.loads(action['action_data'])
                        action['action_result'] = json.loads(action['action_result'])
                        actions.append(action)
                    
                    return actions
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get player actions for {game_id}, player {player_id}: {e}")
            return []
    
    def get_current_turn_info(self, game_id: str) -> Dict[str, Any]:
        """Get information about the current turn in a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT current_player_id, current_turn_number, game_phase
                        FROM game_sessions
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    session_row = cursor.fetchone()
                    if not session_row:
                        return {}
                    
                    session_data = dict(session_row)
                    
                    if session_data['current_player_id'] and session_data['current_turn_number']:
                        cursor = conn.execute(
                            """
                            SELECT *
                            FROM player_turns
                            WHERE game_id = ? AND player_id = ? AND turn_number = ?
                            """,
                            (game_id, session_data['current_player_id'], session_data['current_turn_number'])
                        )
                        
                        turn_row = cursor.fetchone()
                        if turn_row:
                            session_data['current_turn'] = dict(turn_row)
                    
                    return session_data
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get current turn info for {game_id}: {e}")
            return {}

    def update_player_strategy(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        strategy_type: str,
        strategy_content: str
    ) -> bool:
        """Update a player's strategy and store in history."""
        if strategy_type not in ('short_term', 'long_term'):
            risk_logger.log_error(f"Invalid strategy type: {strategy_type}")
            return False

        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    # Insert new strategy entry
                    conn.execute(
                        """
                        INSERT INTO player_strategies
                        (game_id, player_id, player_name, strategy_type, strategy_content, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (game_id, player_id, player_name, strategy_type, strategy_content, now)
                    )
                    
                    conn.commit()
                    risk_logger.log_info(f"Updated {strategy_type} strategy for player {player_name} in game {game_id}")
                    return True
                except Exception as e:
                    risk_logger.log_error(f"Failed to update player strategy for {game_id}, player {player_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error updating player strategy: {e}")
            return False

    def get_current_player_strategies(self, game_id: str, player_id: str) -> Dict[str, Any]:
        """Get the current (latest) strategies for a player."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    strategies = {}
                    
                    # Get latest short_term strategy
                    cursor = conn.execute(
                        """
                        SELECT strategy_content, updated_at
                        FROM player_strategies
                        WHERE game_id = ? AND player_id = ? AND strategy_type = 'short_term'
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (game_id, player_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        strategies['short_term'] = {
                            'content': row['strategy_content'],
                            'updated_at': row['updated_at']
                        }
                    
                    # Get latest long_term strategy
                    cursor = conn.execute(
                        """
                        SELECT strategy_content, updated_at
                        FROM player_strategies
                        WHERE game_id = ? AND player_id = ? AND strategy_type = 'long_term'
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (game_id, player_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        strategies['long_term'] = {
                            'content': row['strategy_content'],
                            'updated_at': row['updated_at']
                        }
                    
                    return strategies
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get current strategies for game {game_id}, player {player_id}: {e}")
            return {}

    def get_player_strategy_history(self, game_id: str, player_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get full history of strategy updates for a player."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    history = {
                        'short_term': [],
                        'long_term': []
                    }
                    
                    # Get all strategy updates
                    cursor = conn.execute(
                        """
                        SELECT strategy_type, strategy_content, updated_at, player_name
                        FROM player_strategies
                        WHERE game_id = ? AND player_id = ?
                        ORDER BY updated_at DESC
                        """,
                        (game_id, player_id)
                    )
                    
                    for row in cursor.fetchall():
                        strategy_type = row['strategy_type']
                        entry = {
                            'content': row['strategy_content'],
                            'updated_at': row['updated_at'],
                            'player_name': row['player_name']
                        }
                        history[strategy_type].append(entry)
                    
                    return history
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get strategy history for game {game_id}, player {player_id}: {e}")
            return {'short_term': [], 'long_term': []}

    def get_all_current_strategies(self, game_id: str) -> Dict[str, Dict[str, Any]]:
        """Get current strategies for all players in a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get all player IDs in this game with strategies
                    cursor = conn.execute(
                        """
                        SELECT DISTINCT player_id, player_name
                        FROM player_strategies
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    all_strategies = {}
                    
                    for player_row in cursor.fetchall():
                        player_id = player_row['player_id']
                        player_name = player_row['player_name']
                        
                        strategies = self.get_current_player_strategies(game_id, player_id)
                        all_strategies[player_id] = {
                            'player_name': player_name,
                            'strategies': strategies
                        }
                    
                    return all_strategies
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get all current strategies for game {game_id}: {e}")
            return {}

    def get_all_strategy_history(self, game_id: str) -> Dict[str, Any]:
        """Get complete strategy history for all players in a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get all strategy updates for this game
                    cursor = conn.execute(
                        """
                        SELECT player_id, player_name, strategy_type, strategy_content, updated_at
                        FROM player_strategies
                        WHERE game_id = ?
                        ORDER BY updated_at DESC
                        """,
                        (game_id,)
                    )
                    
                    history = {}
                    
                    for row in cursor.fetchall():
                        player_id = row['player_id']
                        
                        if player_id not in history:
                            history[player_id] = {
                                'player_name': row['player_name'],
                                'short_term': [],
                                'long_term': []
                            }
                        
                        strategy_type = row['strategy_type']
                        entry = {
                            'content': row['strategy_content'],
                            'updated_at': row['updated_at']
                        }
                        history[player_id][strategy_type].append(entry)
                    
                    return history
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get strategy history for game {game_id}: {e}")
            return {}
            
    # Agent Decision Tracking Methods
    
    def track_agent_decision_start(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        turn_number: int,
        context_data: Dict[str, Any],
        formatted_context: str,
        agent_prompt: str
    ) -> int:
        """
        Start tracking an agent's decision process.
        Returns the decision ID for later updating with results.
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    cursor = conn.execute(
                        """
                        INSERT INTO agent_decisions
                        (game_id, player_id, player_name, turn_number, decision_timestamp,
                         context_data, formatted_context, agent_prompt, success)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id,
                            player_id,
                            player_name,
                            turn_number,
                            now,
                            json.dumps(context_data),
                            formatted_context,
                            agent_prompt,
                            False  # Mark as not successful yet
                        )
                    )
                    
                    conn.commit()
                    decision_id = cursor.lastrowid
                    risk_logger.log_info(f"Started tracking agent decision {decision_id} for player {player_name}")
                    return decision_id
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to track agent decision start: {e}")
                    conn.rollback()
                    return -1
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking agent decision start: {e}")
            return -1
    
    def track_agent_decision_complete(
        self,
        decision_id: int,
        agent_response: str,
        agent_reasoning: str,
        tools_used: List[str],
        decision_time_seconds: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Complete tracking of an agent's decision by updating with results.
        """
        if decision_id < 0:
            return False
            
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    conn.execute(
                        """
                        UPDATE agent_decisions
                        SET agent_response = ?,
                            agent_reasoning = ?,
                            tools_used = ?,
                            decision_time_seconds = ?,
                            success = ?,
                            error_message = ?
                        WHERE id = ?
                        """,
                        (
                            agent_response or "",
                            agent_reasoning or "",
                            json.dumps(tools_used),
                            decision_time_seconds,
                            success,
                            error_message or "",
                            decision_id
                        )
                    )
                    
                    conn.commit()
                    risk_logger.log_info(f"Completed tracking agent decision {decision_id}, success: {success}")
                    return True
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to complete agent decision tracking: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error completing agent decision: {e}")
            return False
    
    def get_agent_decisions(
        self,
        game_id: str,
        player_id: Optional[str] = None,
        turn_number: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get agent decisions with optional filtering by player or turn.
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    query = """
                        SELECT id, game_id, player_id, player_name, turn_number, 
                               decision_timestamp, agent_reasoning, tools_used,
                               decision_time_seconds, success, error_message
                        FROM agent_decisions
                        WHERE game_id = ?
                    """
                    params = [game_id]
                    
                    if player_id:
                        query += " AND player_id = ?"
                        params.append(player_id)
                        
                    if turn_number:
                        query += " AND turn_number = ?"
                        params.append(turn_number)
                        
                    query += " ORDER BY decision_timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    
                    decisions = []
                    for row in cursor.fetchall():
                        decision = dict(row)
                        # Parse JSON fields
                        if decision['tools_used']:
                            decision['tools_used'] = json.loads(decision['tools_used'])
                        else:
                            decision['tools_used'] = []
                            
                        decisions.append(decision)
                    
                    return decisions
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get agent decisions: {e}")
            return []
    
    def get_agent_decision_detail(self, decision_id: int) -> Dict[str, Any]:
        """
        Get complete details for a specific agent decision.
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT *
                        FROM agent_decisions
                        WHERE id = ?
                        """,
                        (decision_id,)
                    )
                    
                    row = cursor.fetchone()
                    if not row:
                        return {}
                        
                    decision = dict(row)
                    
                    # Parse JSON fields
                    if decision.get('context_data'):
                        try:
                            decision['context_data'] = json.loads(decision['context_data'])
                        except:
                            decision['context_data'] = {}
                            
                    if decision.get('tools_used'):
                        try:
                            decision['tools_used'] = json.loads(decision['tools_used'])
                        except:
                            decision['tools_used'] = []
                    
                    return decision
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get agent decision detail: {e}")
            return {}
    
    def get_agent_decision_analytics(self, game_id: str) -> Dict[str, Any]:
        """
        Get analytics about agent decisions for a game.
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get decision stats by player
                    cursor = conn.execute(
                        """
                        SELECT player_id, player_name, 
                               COUNT(*) as decision_count,
                               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                               AVG(decision_time_seconds) as avg_decision_time
                        FROM agent_decisions
                        WHERE game_id = ?
                        GROUP BY player_id, player_name
                        """,
                        (game_id,)
                    )
                    
                    player_stats = {}
                    for row in cursor.fetchall():
                        player_stats[row['player_id']] = {
                            'player_name': row['player_name'],
                            'decision_count': row['decision_count'],
                            'success_count': row['success_count'],
                            'avg_decision_time': row['avg_decision_time']
                        }
                    
                    # Get tool usage stats
                    tool_usage = {}
                    cursor = conn.execute(
                        """
                        SELECT tools_used
                        FROM agent_decisions
                        WHERE game_id = ? AND tools_used IS NOT NULL AND tools_used != ''
                        """,
                        (game_id,)
                    )
                    
                    for row in cursor.fetchall():
                        if row['tools_used']:
                            try:
                                tools = json.loads(row['tools_used'])
                                for tool in tools:
                                    if tool not in tool_usage:
                                        tool_usage[tool] = 0
                                    tool_usage[tool] += 1
                            except:
                                pass
                    
                    return {
                        'player_stats': player_stats,
                        'tool_usage': tool_usage,
                        'total_decisions': sum(s['decision_count'] for s in player_stats.values()),
                        'successful_decisions': sum(s['success_count'] for s in player_stats.values())
                    }
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get agent decision analytics: {e}")
            return {}
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """Get statistics about a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get basic game stats
                    cursor = conn.execute(
                        """
                        SELECT g.*, 
                            COUNT(DISTINCT pt.player_id) as player_count,
                            COUNT(DISTINCT pt.turn_number) as total_turns,
                            COUNT(ga.id) as total_actions
                        FROM game_sessions g
                        LEFT JOIN player_turns pt ON g.game_id = pt.game_id
                        LEFT JOIN game_actions ga ON g.game_id = ga.game_id
                        WHERE g.game_id = ?
                        GROUP BY g.game_id
                        """,
                        (game_id,)
                    )
                    
                    stats_row = cursor.fetchone()
                    if not stats_row:
                        return {}
                    
                    stats = dict(stats_row)
                    
                    # Get action type breakdown
                    cursor = conn.execute(
                        """
                        SELECT action_type, COUNT(*) as count
                        FROM game_actions
                        WHERE game_id = ?
                        GROUP BY action_type
                        """,
                        (game_id,)
                    )
                    
                    action_types = {}
                    for row in cursor.fetchall():
                        action_types[row['action_type']] = row['count']
                    
                    stats['action_types'] = action_types
                    
                    # Get player model assignments
                    stats['player_models'] = self.get_game_models(game_id)
                    
                    return stats
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get game stats for {game_id}: {e}")
            return {}

    # Context Summary Methods
    
    def store_context_summary(
        self,
        game_id: str,
        summary_type: str,
        content: str,
        turn_range_start: int,
        turn_range_end: int,
        original_tokens: int,
        summary_tokens: int
    ) -> bool:
        """Store a context summary in the database."""
        if summary_type not in ('strategic', 'diplomatic', 'battle_history', 'game_evolution', 'full_context'):
            risk_logger.log_error(f"Invalid summary type: {summary_type}")
            return False
            
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    tokens_saved = original_tokens - summary_tokens
                    
                    # Mark previous summaries of this type as not current
                    conn.execute(
                        """
                        UPDATE context_summaries
                        SET is_current = 0
                        WHERE game_id = ? AND summary_type = ? AND is_current = 1
                        """,
                        (game_id, summary_type)
                    )
                    
                    # Get next version number
                    cursor = conn.execute(
                        """
                        SELECT MAX(version) as max_version
                        FROM context_summaries
                        WHERE game_id = ? AND summary_type = ?
                        """,
                        (game_id, summary_type)
                    )
                    result = cursor.fetchone()
                    version = (result['max_version'] or 0) + 1
                    
                    # Insert new summary
                    conn.execute(
                        """
                        INSERT INTO context_summaries
                        (game_id, summary_type, content, turn_range_start, turn_range_end,
                         original_tokens, summary_tokens, tokens_saved, created_at, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id, summary_type, content, turn_range_start, turn_range_end,
                            original_tokens, summary_tokens, tokens_saved, now, version
                        )
                    )
                    
                    conn.commit()
                    risk_logger.log_info(f"Stored {summary_type} summary for game {game_id}, turns {turn_range_start}-{turn_range_end}, saved {tokens_saved} tokens")
                    return True
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to store context summary: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error storing context summary: {e}")
            return False
    
    def get_current_context_summary(self, game_id: str, summary_type: str) -> Optional[Dict[str, Any]]:
        """Get the current (most recent) summary of a specific type."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT *
                        FROM context_summaries
                        WHERE game_id = ? AND summary_type = ? AND is_current = 1
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (game_id, summary_type)
                    )
                    
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
                    return None
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get current context summary: {e}")
            return None
    
    def get_all_current_summaries(self, game_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all current summaries for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT *
                        FROM context_summaries
                        WHERE game_id = ? AND is_current = 1
                        ORDER BY summary_type, created_at DESC
                        """,
                        (game_id,)
                    )
                    
                    summaries = {}
                    for row in cursor.fetchall():
                        summary_data = dict(row)
                        summaries[summary_data['summary_type']] = summary_data
                    
                    return summaries
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get all current summaries: {e}")
            return {}
    
    def get_summary_history(self, game_id: str, summary_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get history of summaries for a game, optionally filtered by type."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    if summary_type:
                        cursor = conn.execute(
                            """
                            SELECT *
                            FROM context_summaries
                            WHERE game_id = ? AND summary_type = ?
                            ORDER BY created_at DESC
                            """,
                            (game_id, summary_type)
                        )
                    else:
                        cursor = conn.execute(
                            """
                            SELECT *
                            FROM context_summaries
                            WHERE game_id = ?
                            ORDER BY created_at DESC
                            """,
                            (game_id,)
                        )
                    
                    summaries = []
                    for row in cursor.fetchall():
                        summaries.append(dict(row))
                    
                    return summaries
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get summary history: {e}")
            return []
    
    def get_summary_stats(self, game_id: str) -> Dict[str, Any]:
        """Get statistics about summaries for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get summary counts and token savings by type
                    cursor = conn.execute(
                        """
                        SELECT summary_type,
                               COUNT(*) as summary_count,
                               SUM(tokens_saved) as total_tokens_saved,
                               AVG(tokens_saved) as avg_tokens_saved,
                               MAX(created_at) as last_created
                        FROM context_summaries
                        WHERE game_id = ?
                        GROUP BY summary_type
                        """,
                        (game_id,)
                    )
                    
                    stats_by_type = {}
                    total_summaries = 0
                    total_tokens_saved = 0
                    
                    for row in cursor.fetchall():
                        summary_type = row['summary_type']
                        stats_by_type[summary_type] = {
                            'count': row['summary_count'],
                            'total_tokens_saved': row['total_tokens_saved'],
                            'avg_tokens_saved': row['avg_tokens_saved'],
                            'last_created': row['last_created']
                        }
                        total_summaries += row['summary_count']
                        total_tokens_saved += row['total_tokens_saved'] or 0
                    
                    return {
                        'by_type': stats_by_type,
                        'total_summaries': total_summaries,
                        'total_tokens_saved': total_tokens_saved
                    }
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get summary stats: {e}")
            return {}
    
    def cleanup_old_summaries(self, game_id: str, keep_versions: int = 3) -> bool:
        """Clean up old summary versions, keeping only the most recent ones."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # For each summary type, keep only the most recent versions
                    summary_types = ['strategic', 'diplomatic', 'battle_history', 'game_evolution', 'full_context']
                    
                    for summary_type in summary_types:
                        # Get IDs of summaries to keep
                        cursor = conn.execute(
                            """
                            SELECT id
                            FROM context_summaries
                            WHERE game_id = ? AND summary_type = ?
                            ORDER BY created_at DESC
                            LIMIT ?
                            """,
                            (game_id, summary_type, keep_versions)
                        )
                        
                        keep_ids = [row['id'] for row in cursor.fetchall()]
                        
                        if keep_ids:
                            # Delete older summaries
                            placeholders = ','.join(['?'] * len(keep_ids))
                            conn.execute(
                                f"""
                                DELETE FROM context_summaries
                                WHERE game_id = ? AND summary_type = ? AND id NOT IN ({placeholders})
                                """,
                                [game_id, summary_type] + keep_ids
                            )
                    
                    conn.commit()
                    risk_logger.log_info(f"Cleaned up old summaries for game {game_id}")
                    return True
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to cleanup old summaries: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error cleaning up summaries: {e}")
            return False

    # Game Completion Statistics Methods
    
    def calculate_game_completion_stats(self, game_id: str) -> Dict[str, Any]:
        """Calculate comprehensive completion statistics for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get basic game info
                    cursor = conn.execute(
                        """
                        SELECT created_at, current_turn_number, num_players, status
                        FROM game_sessions
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    game_info = cursor.fetchone()
                    if not game_info:
                        risk_logger.log_error(f"Game {game_id} not found for completion stats")
                        return {}
                    
                    game_info = dict(game_info)
                    
                    # Calculate total duration
                    created_at = datetime.fromisoformat(game_info['created_at'])
                    completed_at = datetime.utcnow()
                    total_duration = int((completed_at - created_at).total_seconds())
                    
                    # Get all players who participated - check both player_turns and player_models
                    # to ensure we capture all tournament players even if they didn't complete turns
                    cursor = conn.execute(
                        """
                        SELECT DISTINCT player_id, player_name
                        FROM player_turns
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    players = {row['player_id']: row['player_name'] for row in cursor.fetchall()}
                    
                    # Also get players from player_models table (important for tournament games)
                    cursor = conn.execute(
                        """
                        SELECT DISTINCT player_id, player_name
                        FROM player_models
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    # Merge players from both sources
                    for row in cursor.fetchall():
                        if row['player_id'] not in players:
                            players[row['player_id']] = row['player_name']
                            print(f"Added player {row['player_name']} from player_models table")
                    
                    print(f"Total players found for game {game_id}: {len(players)} - {list(players.values())}")
                    
                    # Calculate player-specific statistics
                    player_statistics = {}
                    
                    for player_id, player_name in players.items():
                        player_stats = self._calculate_player_stats(conn, game_id, player_id, player_name)
                        player_statistics[player_id] = player_stats
                    
                    # Calculate overall action statistics
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) as total_actions,
                               SUM(CASE WHEN JSON_EXTRACT(action_result, '$.success') = 1 THEN 1 ELSE 0 END) as successful_actions,
                               action_type
                        FROM game_actions
                        WHERE game_id = ?
                        GROUP BY action_type
                        """,
                        (game_id,)
                    )
                    
                    action_type_breakdown = {}
                    total_actions = 0
                    successful_actions = 0
                    
                    for row in cursor.fetchall():
                        action_type = row['action_type']
                        count = row['total_actions']
                        success_count = row['successful_actions'] or 0
                        
                        action_type_breakdown[action_type] = {
                            'total': count,
                            'successful': success_count,
                            'failed': count - success_count,
                            'success_rate': success_count / count if count > 0 else 0
                        }
                        
                        total_actions += count
                        successful_actions += success_count
                    
                    failed_actions = total_actions - successful_actions
                    
                    # Get models used
                    models_used = self._get_models_used(conn, game_id)
                    
                    # Get decision analytics
                    cursor = conn.execute(
                        """
                        SELECT AVG(decision_time_seconds) as avg_decision_time
                        FROM agent_decisions
                        WHERE game_id = ? AND decision_time_seconds IS NOT NULL
                        """,
                        (game_id,)
                    )
                    
                    decision_row = cursor.fetchone()
                    avg_decision_time = decision_row['avg_decision_time'] if decision_row else 0.0
                    
                    # Get summarization stats
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) as total_summaries,
                               SUM(tokens_saved) as total_tokens_saved
                        FROM context_summaries
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    summary_row = cursor.fetchone()
                    total_summarizations = summary_row['total_summaries'] if summary_row else 0
                    total_tokens_saved = summary_row['total_tokens_saved'] if summary_row else 0
                    
                    # Calculate game-specific metrics
                    diplomatic_messages = action_type_breakdown.get('send_message', {}).get('total', 0)
                    territories_conquered = action_type_breakdown.get('attack_territory', {}).get('successful', 0)
                    battles_fought = action_type_breakdown.get('attack_territory', {}).get('total', 0)
                    
                    return {
                        'game_id': game_id,
                        'completed_at': completed_at.isoformat(),
                        'total_duration_seconds': total_duration,
                        'total_turns': game_info['current_turn_number'],
                        'num_players': game_info['num_players'],
                        'player_statistics': player_statistics,
                        'total_actions': total_actions,
                        'successful_actions': successful_actions,
                        'failed_actions': failed_actions,
                        'action_type_breakdown': action_type_breakdown,
                        'models_used': models_used,
                        'avg_decision_time_seconds': avg_decision_time,
                        'total_summarizations': total_summarizations,
                        'total_tokens_saved': total_tokens_saved,
                        'diplomatic_messages': diplomatic_messages,
                        'territories_conquered': territories_conquered,
                        'battles_fought': battles_fought
                    }
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to calculate completion stats for game {game_id}: {e}")
            return {}
    
    def _calculate_player_stats(self, conn: sqlite3.Connection, game_id: str, player_id: str, player_name: str) -> Dict[str, Any]:
        """Calculate statistics for a specific player."""
        print(f"Calculating stats for player {player_name} (ID: {player_id}) in game {game_id}")
        
        # Get player's actions
        cursor = conn.execute(
            """
            SELECT action_type, 
                   COUNT(*) as total,
                   SUM(CASE WHEN JSON_EXTRACT(action_result, '$.success') = 1 THEN 1 ELSE 0 END) as successful
            FROM game_actions
            WHERE game_id = ? AND player_id = ?
            GROUP BY action_type
            """,
            (game_id, player_id)
        )
        
        action_breakdown = {}
        total_actions = 0
        successful_actions = 0
        
        for row in cursor.fetchall():
            action_type = row['action_type']
            total = row['total']
            successful = row['successful'] or 0
            
            action_breakdown[action_type] = {
                'total': total,
                'successful': successful,
                'failed': total - successful,
                'success_rate': successful / total if total > 0 else 0
            }
            
            total_actions += total
            successful_actions += successful
        
        print(f"  Actions found: {total_actions} total, {successful_actions} successful")
        
        # Get player's decision analytics
        cursor = conn.execute(
            """
            SELECT COUNT(*) as decision_count,
                   AVG(decision_time_seconds) as avg_decision_time,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_decisions
            FROM agent_decisions
            WHERE game_id = ? AND player_id = ?
            """,
            (game_id, player_id)
        )
        
        decision_row = cursor.fetchone()
        decision_count = decision_row['decision_count'] if decision_row else 0
        avg_decision_time = decision_row['avg_decision_time'] if decision_row else 0.0
        successful_decisions = decision_row['successful_decisions'] if decision_row else 0
        
        print(f"  Decisions found: {decision_count} total, {successful_decisions} successful")
        
        # Get player's model info (from player_models table) - this is critical for tournament games
        cursor = conn.execute(
            """
            SELECT model_name, temperature
            FROM player_models
            WHERE game_id = ? AND player_id = ?
            """,
            (game_id, player_id)
        )
        
        model_used = "unknown"
        temperature = None
        model_row = cursor.fetchone()
        if model_row:
            model_used = model_row['model_name']
            temperature = model_row['temperature']
            print(f"  Model found: {model_used} (temp: {temperature})")
        else:
            print(f"  No model assignment found for player {player_name}")
        
        # Get player's turn count
        cursor = conn.execute(
            """
            SELECT COUNT(*) as turns_played
            FROM player_turns
            WHERE game_id = ? AND player_id = ?
            """,
            (game_id, player_id)
        )
        
        turns_row = cursor.fetchone()
        turns_played = turns_row['turns_played'] if turns_row else 0
        
        print(f"  Turns played: {turns_played}")
        
        # Get strategy updates count
        cursor = conn.execute(
            """
            SELECT COUNT(*) as strategy_updates
            FROM player_strategies
            WHERE game_id = ? AND player_id = ?
            """,
            (game_id, player_id)
        )
        
        strategy_row = cursor.fetchone()
        strategy_updates = strategy_row['strategy_updates'] if strategy_row else 0
        
        # Create comprehensive player statistics
        player_stats = {
            'player_name': player_name,
            'model_used': model_used,
            'temperature': temperature,
            'turns_played': turns_played,
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'failed_actions': total_actions - successful_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0.0,
            'action_breakdown': action_breakdown,
            'decision_count': decision_count,
            'successful_decisions': successful_decisions,
            'avg_decision_time': avg_decision_time,
            'strategy_updates': strategy_updates
        }
        
        print(f"  Final stats for {player_name}: {total_actions} actions, {turns_played} turns, model: {model_used}")
        
        return player_stats
    
    def _get_models_used(self, conn: sqlite3.Connection, game_id: str) -> List[str]:
        """Get list of unique models used in the game."""
        cursor = conn.execute(
            """
            SELECT DISTINCT context_data
            FROM agent_decisions
            WHERE game_id = ? AND context_data IS NOT NULL
            """,
            (game_id,)
        )
        
        models = set()
        for row in cursor.fetchall():
            if row['context_data']:
                try:
                    context_data = json.loads(row['context_data'])
                    model_name = context_data.get('model_name')
                    if model_name:
                        models.add(model_name)
                except:
                    pass
        
        return list(models) if models else ['unknown']
    
    def store_game_completion_stats(
        self,
        game_id: str,
        winner_player_id: Optional[str] = None,
        winner_player_name: Optional[str] = None,
        completion_reason: str = 'completed'
    ) -> bool:
        """Store comprehensive completion statistics for a game."""
        try:
            # Calculate all the stats
            stats = self.calculate_game_completion_stats(game_id)
            if not stats:
                risk_logger.log_error(f"Failed to calculate stats for game {game_id}")
                return False
            
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Store in completion stats table
                    conn.execute(
                        """
                        INSERT INTO game_completion_stats
                        (game_id, completed_at, total_duration_seconds, total_turns,
                         winner_player_id, winner_player_name, completion_reason,
                         player_statistics, total_actions, successful_actions, failed_actions,
                         action_type_breakdown, models_used, avg_decision_time_seconds,
                         total_summarizations, total_tokens_saved, diplomatic_messages,
                         territories_conquered, battles_fought)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id,
                            stats['completed_at'],
                            stats['total_duration_seconds'],
                            stats['total_turns'],
                            winner_player_id,
                            winner_player_name,
                            completion_reason,
                            json.dumps(stats['player_statistics']),
                            stats['total_actions'],
                            stats['successful_actions'],
                            stats['failed_actions'],
                            json.dumps(stats['action_type_breakdown']),
                            json.dumps(stats['models_used']),
                            stats['avg_decision_time_seconds'],
                            stats['total_summarizations'],
                            stats['total_tokens_saved'],
                            stats['diplomatic_messages'],
                            stats['territories_conquered'],
                            stats['battles_fought']
                        )
                    )
                    
                    conn.commit()
                    risk_logger.log_info(f"Stored completion stats for game {game_id}")
                    return True
                    
                except sqlite3.IntegrityError:
                    risk_logger.log_warning(f"Completion stats already exist for game {game_id}")
                    conn.rollback()
                    return False
                except Exception as e:
                    risk_logger.log_error(f"Failed to store completion stats for {game_id}: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
                    
        except Exception as e:
            risk_logger.log_error(f"Unexpected error storing completion stats: {e}")
            return False
    
    def get_game_completion_stats(self, game_id: str) -> Dict[str, Any]:
        """Get stored completion statistics for a game."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT *
                        FROM game_completion_stats
                        WHERE game_id = ?
                        """,
                        (game_id,)
                    )
                    
                    row = cursor.fetchone()
                    if not row:
                        return {}
                    
                    stats = dict(row)
                    
                    # Parse JSON fields
                    try:
                        stats['player_statistics'] = json.loads(stats['player_statistics'])
                        stats['action_type_breakdown'] = json.loads(stats['action_type_breakdown'])
                        stats['models_used'] = json.loads(stats['models_used'])
                    except:
                        pass
                    
                    return stats
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get completion stats for {game_id}: {e}")
            return {}
    
    def get_historical_game_stats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get completion statistics for multiple games for analysis."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT game_id, completed_at, total_duration_seconds, total_turns,
                               winner_player_name, completion_reason, total_actions,
                               successful_actions, failed_actions, models_used,
                               avg_decision_time_seconds, diplomatic_messages,
                               territories_conquered, battles_fought
                        FROM game_completion_stats
                        ORDER BY completed_at DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                    
                    games = []
                    for row in cursor.fetchall():
                        game_stats = dict(row)
                        
                        # Parse JSON fields
                        try:
                            game_stats['models_used'] = json.loads(game_stats['models_used'])
                        except:
                            game_stats['models_used'] = []
                        
                        games.append(game_stats)
                    
                    return games
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get historical game stats: {e}")
            return []
    
    def finish_game_with_stats(
        self,
        game_id: str,
        winner_player_id: Optional[str] = None,
        winner_player_name: Optional[str] = None,
        completion_reason: str = 'completed',
        status: str = 'completed'
    ) -> bool:
        """Enhanced finish_game that automatically calculates and stores completion stats."""
        try:
            # First mark the game as finished
            game_finished = self.finish_game(game_id, status)
            if not game_finished:
                return False
            
            # Then calculate and store completion statistics
            stats_stored = self.store_game_completion_stats(
                game_id, winner_player_id, winner_player_name, completion_reason
            )
            
            if stats_stored:
                risk_logger.log_info(f"Game {game_id} completed with comprehensive statistics")
            else:
                risk_logger.log_warning(f"Game {game_id} finished but stats storage failed")
            
            return True
            
        except Exception as e:
            risk_logger.log_error(f"Failed to finish game with stats {game_id}: {e}")
            return False

    # Action Failure Tracking Methods (NEW - for loop detection and prevention)
    
    def track_action_failure(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        action_type: str,
        error_message: str
    ) -> Tuple[int, bool]:
        """
        Track a failed action and detect if intervention is needed.
        Returns (consecutive_failures, intervention_needed).
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    # Check if we already have a failure record for this player/action combination
                    cursor = conn.execute(
                        """
                        SELECT id, consecutive_failures, first_failure_timestamp, intervention_triggered
                        FROM action_failure_tracking
                        WHERE game_id = ? AND player_id = ? AND action_type = ?
                        ORDER BY last_failure_timestamp DESC
                        LIMIT 1
                        """,
                        (game_id, player_id, action_type)
                    )
                    
                    existing_record = cursor.fetchone()
                    
                    if existing_record:
                        # Update existing record
                        record_id = existing_record['id']
                        consecutive_failures = existing_record['consecutive_failures'] + 1
                        first_failure_timestamp = existing_record['first_failure_timestamp']
                        intervention_triggered = existing_record['intervention_triggered']
                        
                        conn.execute(
                            """
                            UPDATE action_failure_tracking
                            SET consecutive_failures = ?,
                                last_failure_message = ?,
                                last_failure_timestamp = ?
                            WHERE id = ?
                            """,
                            (consecutive_failures, error_message, now, record_id)
                        )
                    else:
                        # Create new record
                        consecutive_failures = 1
                        first_failure_timestamp = now
                        intervention_triggered = False
                        
                        conn.execute(
                            """
                            INSERT INTO action_failure_tracking
                            (game_id, player_id, player_name, action_type, consecutive_failures,
                             last_failure_message, last_failure_timestamp, first_failure_timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (game_id, player_id, player_name, action_type, consecutive_failures,
                             error_message, now, first_failure_timestamp)
                        )
                    
                    conn.commit()
                    
                    # Determine if intervention is needed (3+ consecutive failures for faster response)
                    intervention_needed = consecutive_failures >= 3 and not intervention_triggered
                    
                    if intervention_needed:
                        risk_logger.log_warning(
                            f"LOOP DETECTED: Player {player_name} failed {action_type} "
                            f"{consecutive_failures} times consecutively - intervention needed!"
                        )
                    elif consecutive_failures >= 2:
                        risk_logger.log_warning(
                            f"Repeated failure: Player {player_name} failed {action_type} "
                            f"{consecutive_failures} times - monitoring for loop"
                        )
                    
                    return consecutive_failures, intervention_needed
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to track action failure: {e}")
                    conn.rollback()
                    return 1, False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error tracking action failure: {e}")
            return 1, False
    
    def mark_intervention_triggered(
        self,
        game_id: str,
        player_id: str,
        action_type: str
    ) -> bool:
        """Mark that intervention has been triggered for a specific failure pattern."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    now = datetime.utcnow().isoformat()
                    
                    conn.execute(
                        """
                        UPDATE action_failure_tracking
                        SET intervention_triggered = 1,
                            intervention_timestamp = ?
                        WHERE game_id = ? AND player_id = ? AND action_type = ?
                        """,
                        (now, game_id, player_id, action_type)
                    )
                    
                    conn.commit()
                    risk_logger.log_info(f"Marked intervention triggered for {player_id} {action_type}")
                    return True
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to mark intervention triggered: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error marking intervention: {e}")
            return False
    
    def clear_failure_tracking(
        self,
        game_id: str,
        player_id: str,
        action_type: Optional[str] = None
    ) -> bool:
        """
        Clear failure tracking for a player (called when action succeeds).
        If action_type is None, clears all failure tracking for the player.
        """
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    if action_type:
                        # Clear specific action type
                        conn.execute(
                            """
                            DELETE FROM action_failure_tracking
                            WHERE game_id = ? AND player_id = ? AND action_type = ?
                            """,
                            (game_id, player_id, action_type)
                        )
                        risk_logger.log_info(f"Cleared failure tracking for {player_id} {action_type}")
                    else:
                        # Clear all failure tracking for player
                        conn.execute(
                            """
                            DELETE FROM action_failure_tracking
                            WHERE game_id = ? AND player_id = ?
                            """,
                            (game_id, player_id)
                        )
                        risk_logger.log_info(f"Cleared all failure tracking for {player_id}")
                    
                    conn.commit()
                    return True
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to clear failure tracking: {e}")
                    conn.rollback()
                    return False
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Unexpected error clearing failure tracking: {e}")
            return False
    
    def get_player_failure_patterns(
        self,
        game_id: str,
        player_id: str
    ) -> List[Dict[str, Any]]:
        """Get current failure patterns for a player."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT action_type, consecutive_failures, last_failure_message,
                               last_failure_timestamp, first_failure_timestamp,
                               intervention_triggered, intervention_timestamp
                        FROM action_failure_tracking
                        WHERE game_id = ? AND player_id = ?
                        ORDER BY consecutive_failures DESC, last_failure_timestamp DESC
                        """,
                        (game_id, player_id)
                    )
                    
                    patterns = []
                    for row in cursor.fetchall():
                        patterns.append(dict(row))
                    
                    return patterns
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get failure patterns: {e}")
            return []
    
    def get_stuck_players(self, game_id: str) -> List[Dict[str, Any]]:
        """Get list of players who appear to be stuck (5+ consecutive failures)."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT player_id, player_name, action_type, consecutive_failures,
                               last_failure_message, last_failure_timestamp,
                               intervention_triggered
                        FROM action_failure_tracking
                        WHERE game_id = ? AND consecutive_failures >= 5
                        ORDER BY consecutive_failures DESC, last_failure_timestamp DESC
                        """,
                        (game_id,)
                    )
                    
                    stuck_players = []
                    for row in cursor.fetchall():
                        stuck_players.append(dict(row))
                    
                    return stuck_players
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get stuck players: {e}")
            return []
    
    def get_failure_analytics(self, game_id: str) -> Dict[str, Any]:
        """Get analytics about action failures and interventions."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Get failure counts by action type
                    cursor = conn.execute(
                        """
                        SELECT action_type, 
                               COUNT(*) as failure_instances,
                               SUM(consecutive_failures) as total_failures,
                               AVG(consecutive_failures) as avg_consecutive_failures,
                               SUM(CASE WHEN intervention_triggered = 1 THEN 1 ELSE 0 END) as interventions
                        FROM action_failure_tracking
                        WHERE game_id = ?
                        GROUP BY action_type
                        ORDER BY total_failures DESC
                        """,
                        (game_id,)
                    )
                    
                    failure_by_action = {}
                    total_interventions = 0
                    
                    for row in cursor.fetchall():
                        action_type = row['action_type']
                        failure_by_action[action_type] = {
                            'failure_instances': row['failure_instances'],
                            'total_failures': row['total_failures'],
                            'avg_consecutive_failures': row['avg_consecutive_failures'],
                            'interventions': row['interventions']
                        }
                        total_interventions += row['interventions']
                    
                    # Get failure counts by player
                    cursor = conn.execute(
                        """
                        SELECT player_id, player_name,
                               COUNT(*) as failure_patterns,
                               SUM(consecutive_failures) as total_failures,
                               MAX(consecutive_failures) as max_consecutive_failures,
                               SUM(CASE WHEN intervention_triggered = 1 THEN 1 ELSE 0 END) as interventions
                        FROM action_failure_tracking
                        WHERE game_id = ?
                        GROUP BY player_id, player_name
                        ORDER BY total_failures DESC
                        """,
                        (game_id,)
                    )
                    
                    failure_by_player = {}
                    for row in cursor.fetchall():
                        player_id = row['player_id']
                        failure_by_player[player_id] = {
                            'player_name': row['player_name'],
                            'failure_patterns': row['failure_patterns'],
                            'total_failures': row['total_failures'],
                            'max_consecutive_failures': row['max_consecutive_failures'],
                            'interventions': row['interventions']
                        }
                    
                    return {
                        'failure_by_action': failure_by_action,
                        'failure_by_player': failure_by_player,
                        'total_interventions': total_interventions,
                        'stuck_players_count': len([p for p in failure_by_player.values() 
                                                  if p['max_consecutive_failures'] >= 5])
                    }
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get failure analytics: {e}")
            return {}
    
    def get_model_usage_stats(self, days: int = 30) -> Dict[str, int]:
        """Get model usage statistics for the last N days."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    # Calculate the cutoff date
                    from datetime import datetime, timedelta
                    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                    
                    # Query model usage from player_models table
                    cursor = conn.execute(
                        """
                        SELECT model_name, COUNT(*) as usage_count
                        FROM player_models
                        WHERE assigned_at >= ?
                        GROUP BY model_name
                        ORDER BY usage_count DESC
                        """,
                        (cutoff_date,)
                    )
                    
                    usage_stats = {}
                    for row in cursor.fetchall():
                        usage_stats[row['model_name']] = row['usage_count']
                    
                    risk_logger.log_info(f"Retrieved model usage stats for last {days} days: {usage_stats}")
                    return usage_stats
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get model usage stats: {e}")
            return {}
    
    def get_detailed_model_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed model analytics including usage patterns and performance."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    from datetime import datetime, timedelta
                    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                    
                    # Get basic usage stats
                    cursor = conn.execute(
                        """
                        SELECT 
                            pm.model_name,
                            COUNT(*) as games_played,
                            COUNT(DISTINCT pm.game_id) as unique_games,
                            AVG(pm.temperature) as avg_temperature,
                            MIN(pm.assigned_at) as first_used,
                            MAX(pm.assigned_at) as last_used
                        FROM player_models pm
                        WHERE pm.assigned_at >= ?
                        GROUP BY pm.model_name
                        ORDER BY games_played DESC
                        """,
                        (cutoff_date,)
                    )
                    
                    model_analytics = {}
                    total_usage = 0
                    
                    for row in cursor.fetchall():
                        model_name = row['model_name']
                        games_played = row['games_played']
                        total_usage += games_played
                        
                        model_analytics[model_name] = {
                            'games_played': games_played,
                            'unique_games': row['unique_games'],
                            'avg_temperature': row['avg_temperature'],
                            'first_used': row['first_used'],
                            'last_used': row['last_used'],
                            'usage_percentage': 0.0  # Will calculate after getting total
                        }
                    
                    # Calculate usage percentages
                    for model_name in model_analytics:
                        if total_usage > 0:
                            model_analytics[model_name]['usage_percentage'] = (
                                model_analytics[model_name]['games_played'] / total_usage * 100
                            )
                    
                    # Add summary statistics
                    avg_usage = total_usage / len(model_analytics) if model_analytics else 0
                    
                    return {
                        'models': model_analytics,
                        'summary': {
                            'total_usage': total_usage,
                            'unique_models_used': len(model_analytics),
                            'avg_usage_per_model': avg_usage,
                            'analysis_period_days': days,
                            'cutoff_date': cutoff_date
                        }
                    }
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get detailed model analytics: {e}")
            return {'models': {}, 'summary': {}}
    
    def get_model_character_assignments(self, model_name: str) -> List[str]:
        """Get list of character names that have been assigned to a specific model."""
        try:
            with self.lock:
                conn = self._get_db_connection()
                try:
                    cursor = conn.execute(
                        """
                        SELECT DISTINCT player_name
                        FROM player_models
                        WHERE model_name = ?
                        ORDER BY player_name
                        """,
                        (model_name,)
                    )
                    
                    character_names = [row['player_name'] for row in cursor.fetchall()]
                    return character_names
                    
                finally:
                    conn.close()
        except Exception as e:
            risk_logger.log_error(f"Failed to get character assignments for model {model_name}: {e}")
            return []

# Global instance - lazy initialization to avoid startup failures
action_tracker = None

def get_action_tracker():
    """Get the global action tracker instance, creating it if needed."""
    global action_tracker
    if action_tracker is None:
        try:
            action_tracker = ActionTracker()
        except Exception as e:
            risk_logger.log_error(f"Failed to initialize action tracker: {e}")
            # Create a minimal fallback that won't crash the system
            action_tracker = None
            raise
    return action_tracker

# Initialize on first import attempt
try:
    action_tracker = ActionTracker()
except Exception as e:
    risk_logger.log_error(f"Failed to initialize action tracker on import: {e}")
    action_tracker = None
