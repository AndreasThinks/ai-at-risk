#!/usr/bin/env python3

import os
import sys
import asyncio
import argparse
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory and project root to the Python path
sys.path.insert(0, os.path.dirname(__file__))  # Add src/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Add project root directory

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi_mcp import FastApiMCP
import uvicorn

from utils.game_manager import game_manager
from utils.logger import risk_logger
from persistence.game_persistence import game_persistence
from persistence.action_tracker import get_action_tracker

# Helper function for player resolution
def resolve_player(game, player_id: Optional[str] = None, player_name: Optional[str] = None):
    """
    Resolve player from either ID or name.
    Returns: (resolved_player_id, player_object) or raises HTTPException
    """
    if not player_id and not player_name:
        raise HTTPException(status_code=400, detail="Either player_id or player_name must be provided")
    
    # If player_id provided, use it directly
    if player_id:
        player = game.players.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail=f"Player with ID '{player_id}' not found in this game")
        return player_id, player
    
    # If player_name provided, find by name
    if player_name:
        for pid, player in game.players.items():
            if player.name.lower() == player_name.lower():
                return pid, player
        raise HTTPException(status_code=404, detail=f"Player with name '{player_name}' not found in this game")
    
    # This shouldn't happen due to validation, but just in case
    raise HTTPException(status_code=400, detail="No valid player identification provided")

# Pydantic models for API requests
class PlayerConfigRequest(BaseModel):
    name: str = Field(description="Player name")
    model_name: str = Field(description="AI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    temperature: float = Field(ge=0.0, le=2.0, description="Model temperature (0.0-2.0)")
    custom_instructions: str = Field(description="Custom personality instructions")
    api_key: str = Field(description="OpenAI API key")
    base_url: Optional[str] = Field(None, description="Optional custom API base URL")

class CreateGameRequest(BaseModel):
    num_players: int = Field(ge=2, le=6, description="Number of players (2-6)")
    game_name: Optional[str] = Field(None, description="Optional game name")
    player_configs: Optional[List[PlayerConfigRequest]] = Field(None, description="Optional player configurations")

class JoinGameRequest(BaseModel):
    game_id: str
    player_name: str

from pydantic import BaseModel, Field, model_validator

class PlaceArmiesRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    territory: str = Field(description="Territory to place armies on")
    army_count: int = Field(ge=1, description="Number of armies to place")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class AttackTerritoryRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    from_territory: str = Field(description="Territory to attack from")
    to_territory: str = Field(description="Territory to attack")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class FortifyPositionRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    from_territory: str = Field(description="Territory to move armies from")
    to_territory: str = Field(description="Territory to move armies to")
    army_count: int = Field(ge=1, description="Number of armies to move")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class TradeCardsRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    card_indices: List[int] = Field(min_length=3, max_length=3, description="Indices of 3 cards to trade")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class EndTurnRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class SendMessageRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Sender player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Sender player name (use this OR player_id)")
    to_player_name: str = Field(description="Recipient player name")
    message: str = Field(min_length=1, description="Message content (cannot be empty)")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class BroadcastMessageRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Sender player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Sender player name (use this OR player_id)")
    message: str = Field(min_length=1, description="Message content (cannot be empty)")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self


class UpdateNotesRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    notes: str = Field(description="Private notes content")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class UpdateStrategyRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    strategy: str = Field(description="Strategy content")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

class UpdateBothStrategiesRequest(BaseModel):
    game_id: str = Field(description="ID of the game")
    player_id: Optional[str] = Field(None, description="Player ID (use this OR player_name)")
    player_name: Optional[str] = Field(None, description="Player name (use this OR player_id)")
    short_term_strategy: Optional[str] = Field(None, description="Short-term strategy content")
    long_term_strategy: Optional[str] = Field(None, description="Long-term strategy content")
    
    @model_validator(mode='after')
    def validate_player_identification(self):
        if not self.player_id and not self.player_name:
            raise ValueError("Either player_id or player_name must be provided")
        return self

# Create FastAPI app
app = FastAPI(
    title="Risk Game API",
    description="HTTP API for the Risk board game server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create and mount MCP server with limited operations for agents
mcp = FastApiMCP(
    app,
    include_operations=[
        # Core game actions
        "place_armies",
        "attack_territory", 
        "fortify_position",
        "end_turn",
        "trade_cards",
        
        # Communication
        "send_message",
        "broadcast_message",
        
        # Strategy management
        "update_short_term_strategy",
        "update_long_term_strategy", 
        "update_both_strategies"
    ]
)
mcp.mount()

# Background task for auto-saving game state
async def auto_save_game_state(game_id: str):
    """Background task to save game state periodically."""
    try:
        game = game_manager.get_game(game_id)
        if game and game_persistence.should_auto_save(game_id):
            game_state = serialize_game_state(game)
            game_persistence.save_game_state(game_id, game_state)
    except Exception as e:
        risk_logger.log_error(f"Auto-save failed for {game_id}: {e}")

def serialize_game_state(game) -> Dict[str, Any]:
    """Serialize game state for persistence."""
    try:
        return {
            'game_id': game.game_id,
            'game_name': getattr(game, 'game_name', None),
            'max_players': game.num_players,  # Fixed: use num_players instead of max_players
            'game_phase': game.game_phase.value if hasattr(game.game_phase, 'value') else str(game.game_phase),
            'current_player_index': game.current_player_index,
            'turn_number': game.turn_number,
            'created_at': getattr(game, 'created_at', datetime.utcnow().isoformat()),
            'players': {
                player_id: {
                    'player_id': player.player_id,
                    'name': player.name,
                    'color': player.color,
                    'army_count': player.army_count,
                    'territories': list(player.territories),
                    'cards': player.cards,
                    'is_eliminated': player.is_eliminated,
                    'short_term_strategy': player.short_term_strategy,
                    'long_term_strategy': player.long_term_strategy
                }
                for player_id, player in game.players.items()
            },
            'territories': game.territory_manager.to_dict(),
            'game_log': game.game_log[-100:],  # Keep last 100 events
            'card_trade_count': getattr(game.card_manager, 'trade_count', 0)
        }
    except Exception as e:
        risk_logger.log_error(f"Failed to serialize game state: {e}")
        return {}
    
favicon_path = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

# Health check endpoint
@app.get("/health", operation_id="health_check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_games": len(game_manager.games)
    }

# Game Management Endpoints
@app.post("/api/games", operation_id="create_game")
async def create_game(request: CreateGameRequest, background_tasks: BackgroundTasks):
    """Create a new Risk game with optional player configurations."""
    try:
        # Create the basic game first
        success, game_id, message = game_manager.create_game(
            request.num_players, 
            request.game_name
        )
        
        if success:
            # If player configurations are provided, set up the game with AI agents
            if request.player_configs:
                try:
                    # Import here to avoid circular imports
                    from agents.player_config import PlayerConfig
                    from agents.game_runner import GameRunner
                    from langchain_mcp_adapters.client import MultiServerMCPClient
                    import asyncio
                    import os
                    
                    # Get API credentials from environment variables
                    default_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('RISK_API_KEY', '')
                    default_base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('RISK_BASE_URL', '')
                    
                    # Convert API request configs to PlayerConfig objects
                    player_configs = []
                    for config_req in request.player_configs:
                        # Use environment credentials as defaults, allow individual overrides
                        api_key = config_req.api_key or default_api_key
                        base_url = config_req.base_url or default_base_url
                        
                        player_config = PlayerConfig(
                            name=config_req.name,
                            model_name=config_req.model_name,
                            temperature=config_req.temperature,
                            custom_instructions=config_req.custom_instructions,
                            api_key=api_key,
                            base_url=base_url
                        )
                        player_configs.append(player_config)
                    
                    # Create MCP client for the game runner with proper server connection
                    mcp_client = MultiServerMCPClient({
                        "risk": {
                            "transport": "sse",
                            "url": "http://localhost:8080/mcp"
                        }
                    })
                    
                    # Connect to MCP server and get tools
                    risk_logger.log_info("Connecting to MCP server for AI agents...")
                    tools = await mcp_client.get_tools()
                    risk_logger.log_info(f"Connected to MCP server, loaded {len(tools)} tools for AI agents")
                    
                    # Create and initialize the game runner
                    game_runner = GameRunner(
                        mcp_client=mcp_client,
                        player_configs=player_configs
                    )
                    
                    # Set the game ID
                    game_runner.game_id = game_id
                    
                    # Initialize agents and have them join the game
                    await game_runner.initialize_agents()
                    
                    # Start the game in the background
                    asyncio.create_task(game_runner.run_game())
                    
                    message = f"Game created with {len(player_configs)} AI agents and started automatically"
                    
                except Exception as e:
                    risk_logger.log_error(f"Failed to set up AI agents for game {game_id}: {e}")
                    # Game was created successfully, but AI setup failed
                    message = f"Game created successfully, but AI agent setup failed: {str(e)}"
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, game_id)
            
            # Track in the action database
            game = game_manager.get_game(game_id)
            if game:
                try:
                    tracker = get_action_tracker()
                    if tracker:
                        tracker.track_game_start(
                            game_id, 
                            request.num_players, 
                            game.game_phase.value
                        )
                        
                        # If a player has already joined (which happens when AI agents are set up),
                        # initialize their turn tracking as well
                        current_player = game.get_current_player()
                        if current_player:
                            tracker.track_turn_start(
                                game_id,
                                current_player.player_id,
                                current_player.name,
                                game.turn_number
                            )
                except Exception as e:
                    risk_logger.log_error(f"Failed to track game start in action tracker: {e}")
            
            return {
                "success": True,
                "game_id": game_id,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        risk_logger.log_error(f"Create game failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/join", operation_id="join_game")
async def join_game( request: JoinGameRequest, background_tasks: BackgroundTasks):
    """Join an existing game."""
    try:
        game_id = request.game_id.strip()
        success, player_id, message = game_manager.join_game(
            request.game_id, 
            request.player_name
        )
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, game_id)
            
            # Track join in action database
            game = game_manager.get_game(game_id)
            if game and player_id:
                tracker = get_action_tracker()
                if tracker:
                    tracker.track_action(
                        game_id,
                        player_id,
                        game.turn_number,
                        "join_game",
                        {"player_name": request.player_name},
                        {"success": True, "message": message}
                    )
                    
                    # Note: Model tracking is handled by GameRunner when AI agents are created
                    # Human players don't have model assignments
            
            return {
                "success": True,
                "player_id": player_id,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        risk_logger.log_error(f"Join game failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/status", operation_id="get_game_status")
async def get_game_status(game_id: str, player_id: str):
    """Get current game status."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        status = game.get_game_status()
        player_info = game.get_player_info(player_id)
        
        if not player_info:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        return {
            "game_status": status,
            "player_info": player_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get game status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/board", operation_id="get_board_state")
async def get_board_state(game_id: str, player_id: str):
    """Get current board state."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        territories_dict = game.territory_manager.to_dict()
        
        # Group territories by continent
        continents = {}
        for territory_name, territory_data in territories_dict.items():
            continent = territory_data['continent']
            if continent not in continents:
                continents[continent] = []
            continents[continent].append(territory_data)
        
        return {
            "territories": territories_dict,
            "continents": continents
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get board state failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/player/{player_id}", operation_id="get_player_info")
async def get_player_info(game_id: str, player_id: str):
    """Get detailed player information."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        player_info = game.get_player_info(player_id)
        if not player_info:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        return player_info
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gameplay Action Endpoints
@app.post("/api/games/place-armies", operation_id="place_armies")
async def place_armies(request: PlaceArmiesRequest, background_tasks: BackgroundTasks):
    """
    Place armies on a territory during the reinforcement phase.
    
    This endpoint allows players to deploy their available armies to strengthen
    territories they control. This is typically the first action in each turn.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `territory` (str, required): Name of territory to place armies on
    - `army_count` (int, required): Number of armies to place (≥1)
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "territory": "Alaska",
        "army_count": 3
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Placed 3 armies on Alaska"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification, invalid territory, insufficient armies
    - 404: Game not found, player not found, territory not found
    - 403: Not your turn, don't own territory, wrong game phase
    - 500: Game state update failed
    
    **Game Rules:**
    - Can only place armies on territories you control
    - Must be your turn and in reinforcement phase
    - Cannot place more armies than you have available
    - Minimum 1 army per placement
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        success, message = game.place_armies(
            player_id,
            request.territory,
            request.army_count
        )
        
        # Track action in the database
        action_data = {
            "territory": request.territory,
            "army_count": request.army_count
        }
        action_result = {
            "success": success,
            "message": message
        }
        
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                player_id,
                game.turn_number,
                "place_armies",
                action_data,
                action_result
            )
        
        if success:
            # Try auto-advance phase after placing armies
            game.auto_advance_phase()
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Place armies failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/attack-territory", operation_id="attack_territory")
async def attack_territory(request: AttackTerritoryRequest, background_tasks: BackgroundTasks):
    """
    Attack an enemy territory during the attack phase.
    
    This endpoint allows players to launch attacks against adjacent enemy territories.
    Combat is resolved automatically using dice rolls, with results returned immediately.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `from_territory` (str, required): Territory to attack from (must own, >=2 armies)
    - `to_territory` (str, required): Territory to attack (must be adjacent, enemy-owned)
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "from_territory": "Alaska",
        "to_territory": "Kamchatka"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Attack successful! Conquered Kamchatka",
        "combat_result": {
            "attacker_dice": [6, 4],
            "defender_dice": [3],
            "attacker_losses": 0,
            "defender_losses": 1,
            "territory_conquered": true
        }
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification, invalid territories, insufficient armies
    - 404: Game not found, player not found, territories not found
    - 403: Not your turn, don't own from_territory, territories not adjacent
    - 500: Combat resolution failed
    
    **Game Rules:**
    - Must own from_territory with >=2 armies (1 must stay behind)
    - to_territory must be adjacent and enemy-owned
    - Combat uses standard Risk dice rules
    - If territory conquered, you earn a card (if available)
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        success, message, combat_result = game.attack_territory(
            player_id,
            request.from_territory,
            request.to_territory
        )
        
        # Track attack action
        action_data = {
            "from_territory": request.from_territory,
            "to_territory": request.to_territory
        }
        
        action_result = {
            "success": success,
            "message": message
        }
        
        # Add combat result if available
        if combat_result:
            action_result["combat_result"] = {
                "attacker_dice": combat_result.attacker_dice,
                "defender_dice": combat_result.defender_dice,
                "attacker_losses": combat_result.attacker_losses,
                "defender_losses": combat_result.defender_losses,
                "territory_conquered": combat_result.territory_conquered
            }
        
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                player_id,
                game.turn_number,
                "attack_territory",
                action_data,
                action_result
            )
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            result = {
                "success": True,
                "message": message
            }
            
            if combat_result:
                result["combat_result"] = {
                    "attacker_dice": combat_result.attacker_dice,
                    "defender_dice": combat_result.defender_dice,
                    "attacker_losses": combat_result.attacker_losses,
                    "defender_losses": combat_result.defender_losses,
                    "territory_conquered": combat_result.territory_conquered
                }
            
            return result
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Attack territory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/fortify-position", operation_id="fortify_position")
async def fortify_position(request: FortifyPositionRequest, background_tasks: BackgroundTasks):
    """Move armies between territories."""
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        success, message = game.fortify_position(
            player_id,
            request.from_territory,
            request.to_territory,
            request.army_count
        )
        
        # Track fortify action
        action_data = {
            "from_territory": request.from_territory,
            "to_territory": request.to_territory,
            "army_count": request.army_count
        }
        action_result = {
            "success": success,
            "message": message
        }
        
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                player_id,
                game.turn_number,
                "fortify_position",
                action_data,
                action_result
            )
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Fortify position failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/end-turn", operation_id="end_turn")
async def end_turn(request: EndTurnRequest, background_tasks: BackgroundTasks):
    """
    End the current player's turn and advance to the next player.
    
    ⚠️  **CRITICAL PREREQUISITE**: You MUST place ALL reinforcement armies first!
    
    This endpoint completes the current player's turn and transitions the game
    to the next player. It handles turn cleanup, phase transitions, and
    reinforcement calculations for the next turn.
    
    **🚨 MANDATORY ARMY PLACEMENT CHECK:**
    Before calling this endpoint, you MUST ensure:
    1. Check your `army_count` in player_info from get_game_status
    2. If `army_count > 0`: Use `place_armies` tool until `army_count = 0`
    3. Only then call `end_turn` - it will FAIL if armies remain unplaced
    
    **Common Error Prevention:**
    - ❌ **ERROR**: "Must place all reinforcement armies before ending turn"
    - ✅ **SOLUTION**: Use `place_armies` tool to place all unplaced armies first
    - ⚠️ **WARNING**: Calling end_turn with unplaced armies causes infinite loops!
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Turn ended. Next player: Bob"
    }
    ```
    
    **Error Conditions:**
    - 400: "Must place all reinforcement armies before ending turn" (army_count > 0)
    - 400: Missing player identification, not your turn
    - 404: Game not found, player not found
    - 500: Turn transition failed
    
    **Turn Completion Checklist:**
    Before calling end_turn, verify:
    ✅ `army_count = 0` (all armies placed)
    ✅ Cards traded if desired (optional)
    ✅ Attacks completed (optional)
    ✅ Fortifications done (optional)
    ✅ Messages sent (optional)
    
    **Game Rules:**
    - Only the current player can end their turn
    - ALL reinforcement armies must be placed first
    - Automatically calculates reinforcements for next player
    - Advances turn counter and updates game phase
    - Triggers any end-of-turn game state updates
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents MUST ensure
    army_count = 0 before calling this tool to avoid infinite loops.
    
    **Agent Implementation Pattern:**
    ```python
    # CORRECT: Check army count first
    if player_info["army_count"] > 0:
        # Place armies first
        await place_armies(territory="target", army_count=player_info["army_count"])
    
    # Then end turn
    await end_turn()
    ```
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        # Get the current turn number before ending the turn
        current_turn = game.turn_number
        
        success, message = game.end_turn(player_id)
        
        # Track turn end
        action_data = {
            "player_name": player.name
        }
        
        action_result = {
            "success": success,
            "message": message,
            "next_turn": game.turn_number if success else current_turn
        }
        
        # Track the end of this turn
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                player_id,
                current_turn,
                "end_turn",
                action_data,
                action_result
            )
        
        # End the current turn in the tracker
        if success and tracker:
            tracker.track_turn_end(
                request.game_id,
                player_id,
                current_turn
            )
            
            # If we moved to a new player, track their turn start
            next_player = game.get_current_player()
            if next_player:
                tracker.track_turn_start(
                    request.game_id,
                    next_player.player_id,
                    next_player.name,
                    game.turn_number
                )
                
                # Update game phase in the tracker
                tracker.update_game_phase(
                    request.game_id,
                    game.game_phase.value
                )
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"End turn failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Card Management Endpoints
@app.get("/api/games/{game_id}/cards", operation_id="get_cards")
async def get_cards(game_id: str, player_id: str):
    """Get player's Risk cards."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        player = game.players.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        possible_sets = game.card_manager.get_possible_sets(player.cards)
        next_bonus = game.card_manager.calculate_set_bonus()
        
        return {
            "cards": [{"index": i, "card": card} for i, card in enumerate(player.cards)],
            "possible_sets": possible_sets,
            "next_set_bonus": next_bonus
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get cards failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/trade-cards", operation_id="trade_cards")
async def trade_cards(request: TradeCardsRequest, background_tasks: BackgroundTasks):
    """
    Trade in Risk cards for army bonus during reinforcement phase.
    
    This endpoint allows players to exchange sets of 3 cards for additional armies.
    Card trading provides significant army bonuses that increase with each trade,
    making it a crucial strategic element in Risk.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `card_indices` (List[int], required): Exactly 3 card indices to trade (0-based)
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "card_indices": [0, 1, 2]
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Traded 3 cards for 4 armies"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification, invalid card indices, invalid set
    - 404: Game not found, player not found
    - 403: Not your turn, insufficient cards, wrong game phase
    - 500: Card trade processing failed
    
    **Game Rules:**
    - Must have at least 3 cards to trade
    - Cards must form a valid set (3 of same type OR 1 of each type)
    - Army bonus increases with each trade (4, 6, 8, 10, 12, 15, 20, 25...)
    - Can only trade during reinforcement phase
    - Traded cards return to deck
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        # Get current player cards before trading
        cards_before = player.cards.copy() if player else []
        
        success, message = game.trade_cards(
            player_id,
            request.card_indices
        )
        
        # Track card trading action
        action_data = {
            "card_indices": request.card_indices,
            "cards_before": [str(card) for card in cards_before] if cards_before else []
        }
        
        action_result = {
            "success": success,
            "message": message
        }
        
        # If successful, also include the player's cards after trading
        if success and player:
            action_result["cards_after"] = [str(card) for card in player.cards]
        
        # Track the action in the database
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                player_id,
                game.turn_number,
                "trade_cards",
                action_data,
                action_result
            )
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Trade cards failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Communication Endpoints
@app.get("/api/games/{game_id}/messages", operation_id="get_messages")
async def get_messages(game_id: str, player_id: str):
    """Get messages for a player."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        messages = game.notepad_manager.get_messages(player_id, mark_as_read=True)
        unread_count = game.notepad_manager.get_unread_count(player_id)
        
        return {
            "messages": messages,
            "unread_count": unread_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/send-message", operation_id="send_message")
async def send_message(request: SendMessageRequest, background_tasks: BackgroundTasks):
    """
    Send a direct message to another player for diplomatic communication.
    
    This endpoint enables private diplomatic communication between players.
    Messages are delivered to the recipient's inbox and can be used for
    negotiations, alliances, threats, or intelligence gathering.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Sender's unique identifier 
    - `player_name` (str, optional): Sender's display name
    - `to_player_name` (str, required): Recipient's display name
    - `message` (str, required): Message content (cannot be empty)
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the sender and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "to_player_name": "Bob",
        "message": "I propose an alliance against Charlie. What do you think?"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Message sent to Bob"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification, empty message, sending to self
    - 404: Game not found, sender not found, recipient not found
    - 500: Message delivery failed
    
    **Diplomatic Strategy:**
    - Use for alliance formation and negotiation
    - Share intelligence about other players
    - Coordinate attacks or defensive strategies
    - Negotiate territory exchanges or truces
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve sender from either ID or name
        from_player_id, from_player = resolve_player(game, request.player_id, request.player_name)
        
        # Helper function to find player by name
        def find_player_by_name(name: str):
            for player_id, player in game.players.items():
                if player.name.lower() == name.lower():
                    return player_id, player
            return None, None
        
        # Find recipient
        to_player_id, to_player = find_player_by_name(request.to_player_name)
        
        if not to_player:
            raise HTTPException(status_code=404, detail=f"Recipient player '{request.to_player_name}' not found in this game")
        
        if to_player_id == from_player_id:
            raise HTTPException(status_code=400, detail="Cannot send message to yourself")
        
        success = game.notepad_manager.send_message(
            from_player_id,
            from_player.name,
            to_player_id,
            request.message
        )
        
        if success:
            # Log diplomatic message
            risk_logger.log_diplomatic_message(
                from_player.name, 
                to_player.name, 
                request.message, 
                request.game_id
            )
            
            # Track the messaging action in the database
            tracker = get_action_tracker()
            if tracker:
                tracker.track_action(
                    request.game_id,
                    from_player_id,
                    game.turn_number,
                    "send_message",
                    {
                        "to_player_name": to_player.name,
                        "to_player_id": to_player_id,
                        "message": request.message,
                        "message_length": len(request.message),
                        "broadcast": False
                    },
                    {
                        "success": True,
                        "message": f"Message sent to {to_player.name}"
                    }
                )
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": f"Message sent to {to_player.name}"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to send message (message was empty or rejected)")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Send message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/broadcast-message", operation_id="broadcast_message")
async def broadcast_message(request: BroadcastMessageRequest, background_tasks: BackgroundTasks):
    """Send a broadcast message to all other players in the game."""
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve sender from either ID or name
        from_player_id, from_player = resolve_player(game, request.player_id, request.player_name)
        
        success_count = 0
        failed_recipients = []
        
        for target_player_id, target_player in game.players.items():
            if target_player_id != from_player_id:  # Don't send to self
                success = game.notepad_manager.send_message(
                    from_player_id,
                    from_player.name,
                    target_player_id,
                    request.message
                )
                
                if success:
                    success_count += 1
                    # Log diplomatic message
                    risk_logger.log_diplomatic_message(
                        from_player.name, 
                        target_player.name, 
                        request.message, 
                        request.game_id
                    )
                else:
                    failed_recipients.append(target_player.name)
        
        # Track the broadcast action in the database
        tracker = get_action_tracker()
        if tracker:
            tracker.track_action(
                request.game_id,
                from_player_id,
                game.turn_number,
                "broadcast_message",
                {
                    "message": request.message,
                    "message_length": len(request.message),
                    "broadcast": True,
                    "recipients": success_count
                },
                {
                    "success": success_count > 0,
                    "message": f"Broadcast message sent to {success_count} players",
                    "failed_recipients": failed_recipients if failed_recipients else None
                }
            )
        
        if success_count > 0:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": f"Broadcast message sent to {success_count} players",
                "recipients": success_count,
                "failed": failed_recipients if failed_recipients else None
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to send broadcast message")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Broadcast message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/notes", operation_id="get_private_notes")
async def get_private_notes(game_id: str, player_id: str):
    """Get player's private notes."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        notes = game.notepad_manager.get_private_notes(player_id)
        
        return {
            "notes": notes or ""
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get private notes failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/{game_id}/notes", operation_id="update_private_notes")
async def update_private_notes(game_id: str, request: UpdateNotesRequest):
    """Update player's private notes."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game.notepad_manager.update_private_notes(request.player_id, request.notes)
        
        return {
            "success": True,
            "message": "Private notes updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Update private notes failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Player Strategy Endpoints
@app.get("/api/games/{game_id}/player/{player_id}/strategies", operation_id="get_player_strategies")
async def get_player_strategies(game_id: str, player_id: str):
    """Get a player's current strategies."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        player = game.players.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        # Get current strategies from database
        tracker = get_action_tracker()
        strategies = tracker.get_current_player_strategies(game_id, player_id) if tracker else {}
        
        return {
            "player_name": player.name,
            "short_term_strategy": player.short_term_strategy,
            "long_term_strategy": player.long_term_strategy,
            "database_strategies": strategies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player strategies failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/player/{player_id}/strategies/history", operation_id="get_player_strategy_history")
async def get_player_strategy_history(game_id: str, player_id: str):
    """Get a player's strategy history."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        player = game.players.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        # Get strategy history from database
        tracker = get_action_tracker()
        history = tracker.get_player_strategy_history(game_id, player_id) if tracker else []
        
        return {
            "player_name": player.name,
            "history": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player strategy history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/games/update-short-term-strategy", operation_id="update_short_term_strategy")
async def update_short_term_strategy(request: UpdateStrategyRequest, background_tasks: BackgroundTasks):
    """
    Update a player's short-term strategy (1-3 turns ahead).
    
    This endpoint allows agents to update their tactical planning for immediate actions.
    Short-term strategies should focus on specific territories, immediate threats, and
    concrete actions to be taken in the next few turns.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `strategy` (str, required): New short-term strategy content
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "strategy": "Reinforce Alaska with 5+ armies, then attack Kamchatka to secure North America"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Short-term strategy updated successfully"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification (neither player_id nor player_name provided)
    - 404: Game not found or player not found in game
    - 500: Database update failed
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        # Update in-memory player object
        player.update_short_term_strategy(request.strategy)
        
        # Store in database
        tracker = get_action_tracker()
        success = tracker.update_player_strategy(
            request.game_id,
            player_id,
            player.name,
            "short_term",
            request.strategy
        ) if tracker else True
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": "Short-term strategy updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update short-term strategy")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Update short-term strategy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/games/update-long-term-strategy", operation_id="update_long_term_strategy")
async def update_long_term_strategy(request: UpdateStrategyRequest, background_tasks: BackgroundTasks):
    """
    Update a player's long-term strategy (overall game objectives).
    
    This endpoint allows agents to update their strategic planning for overall game victory.
    Long-term strategies should focus on continent control, win conditions, alliance plans,
    and multi-turn objectives that lead to world domination.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `strategy` (str, required): New long-term strategy content
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Request Body Example:**
    ```json
    {
        "game_id": "f60b0cb2",
        "player_name": "Alice",
        "strategy": "Control Australia and South America for steady army income, then eliminate weakest player to gain cards"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Long-term strategy updated successfully"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification (neither player_id nor player_name provided)
    - 404: Game not found or player not found in game
    - 500: Database update failed
    
    **Strategy Guidelines:**
    - Focus on continent control and army bonuses
    - Plan for player elimination and card acquisition
    - Consider alliance formation and betrayal timing
    - Adapt to changing game dynamics
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. Agents should include
    their player context when calling this tool.
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        # Update in-memory player object
        player.update_long_term_strategy(request.strategy)
        
        # Store in database
        tracker = get_action_tracker()
        success = tracker.update_player_strategy(
            request.game_id,
            player_id,
            player.name,
            "long_term",
            request.strategy
        ) if tracker else True
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": "Long-term strategy updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update long-term strategy")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Update long-term strategy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/games/update-both-strategies", operation_id="update_both_strategies")
async def update_both_strategies(request: UpdateBothStrategiesRequest, background_tasks: BackgroundTasks):
    """
    Update both short-term and long-term strategies in a single request.
    
    This endpoint allows agents to efficiently update both tactical (short-term) and 
    strategic (long-term) planning simultaneously. This is the most commonly used
    strategy endpoint as agents typically want to coordinate both levels of planning.
    
    **Request Body Requirements:**
    - `game_id` (str, required): ID of the game session
    - `player_id` (str, optional): Player's unique identifier 
    - `player_name` (str, optional): Player's display name
    - `short_term_strategy` (str, optional): New short-term strategy (1-3 turns)
    - `long_term_strategy` (str, optional): New long-term strategy (overall objectives)
    
    **Player Identification:**
    You MUST provide either `player_id` OR `player_name` (not both).
    The system will resolve the player and validate they exist in the game.
    
    **Strategy Fields:**
    At least one strategy field must be provided. You can update:
    - Only short-term strategy
    - Only long-term strategy  
    - Both strategies simultaneously
    
    **Request Body Examples:**
    
    Update both strategies:
    ```json
    {
        "game_id": "f60b0cb2",
        "player_id": "player_123",
        "short_term_strategy": "Reinforce Alaska with 5+ armies, then attack Kamchatka to secure North America",
        "long_term_strategy": "Control Australia and South America for steady army income, then eliminate weakest player"
    }
    ```
    
    Update only short-term:
    ```json
    {
        "game_id": "f60b0cb2",
        "player_name": "Alice",
        "short_term_strategy": "Focus on defending Europe borders this turn"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "success": true,
        "message": "Strategies updated successfully"
    }
    ```
    
    **Error Conditions:**
    - 400: Missing player identification (neither player_id nor player_name provided)
    - 400: No strategy content provided (both fields are null/empty)
    - 404: Game not found or player not found in game
    - 500: Database update failed
    
    **Strategy Guidelines:**
    - **Short-term**: Specific territories, immediate threats, next 1-3 turns
    - **Long-term**: Continent control, win conditions, alliance plans, multi-turn objectives
    - Keep strategies current and actionable
    - Update when game situation changes significantly
    
    **MCP Tool Usage:**
    This endpoint is exposed as an MCP tool for AI agents. This is the primary
    strategy update tool that agents should use. Agents MUST include their player
    context (player_id or player_name) when calling this tool.
    
    **Common Usage Pattern:**
    ```python
    # Agent calling this tool should include:
    {
        "game_id": "current_game_id",
        "player_id": "agent_player_id",  # or "player_name": "agent_name"
        "short_term_strategy": "Specific tactical plan",
        "long_term_strategy": "Overall strategic objectives"
    }
    ```
    """
    try:
        game = game_manager.get_game(request.game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Resolve player from either ID or name
        player_id, player = resolve_player(game, request.player_id, request.player_name)
        
        success = True
        
        # Update short-term strategy if provided
        if request.short_term_strategy is not None:
            player.update_short_term_strategy(request.short_term_strategy)
            tracker = get_action_tracker()
            if tracker:
                short_term_success = tracker.update_player_strategy(
                    request.game_id,
                    player_id,
                    player.name,
                    "short_term",
                    request.short_term_strategy
                )
                success = success and short_term_success
        
        # Update long-term strategy if provided
        if request.long_term_strategy is not None:
            player.update_long_term_strategy(request.long_term_strategy)
            tracker = get_action_tracker()
            if tracker:
                long_term_success = tracker.update_player_strategy(
                    request.game_id,
                    player_id,
                    player.name,
                    "long_term",
                    request.long_term_strategy
                )
                success = success and long_term_success
        
        if success:
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, request.game_id)
            
            return {
                "success": True,
                "message": "Strategies updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update strategies")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Update both strategies failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/strategies", operation_id="get_all_player_strategies")
async def get_all_player_strategies(game_id: str):
    """Get current strategies for all players in a game."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get all strategies from database
        tracker = get_action_tracker()
        all_strategies = tracker.get_all_current_strategies(game_id) if tracker else {}
        
        # Combine with in-memory player data
        result = {}
        for player_id, player_data in all_strategies.items():
            player = game.players.get(player_id)
            if player:
                result[player_id] = {
                    "player_name": player.name,
                    "current": {
                        "short_term": player.short_term_strategy,
                        "long_term": player.long_term_strategy
                    },
                    "database": player_data["strategies"]
                }
            else:
                result[player_id] = player_data
        
        return {
            "strategies": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get all strategies failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/strategies/history", operation_id="get_all_strategy_history")
async def get_all_strategy_history(game_id: str):
    """Get strategy history for all players in a game."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        history = tracker.get_all_strategy_history(game_id) if tracker else []
        
        return {
            "history": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get all strategy history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Game History and Utility Endpoints
@app.get("/api/games/{game_id}/history", operation_id="get_game_history")
async def get_game_history(game_id: str, last_n_events: int = 10):
    """Get recent game events."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        recent_events = game.game_log[-last_n_events:] if game.game_log else []
        
        return {
            "events": recent_events,
            "total_events": len(game.game_log)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get game history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games", operation_id="list_active_games")
async def list_active_games():
    """List all active games."""
    try:
        games = game_manager.list_active_games()
        
        return {
            "games": games,
            "total_games": len(games)
        }
        
    except Exception as e:
        risk_logger.log_error(f"List active games failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Action Tracker Endpoints
@app.get("/api/games/{game_id}/actions", operation_id="get_game_action_history")
async def get_game_action_history(game_id: str, limit: int = 50):
    """Get a game's action history from the tracker database."""
    try:
        # Check if game exists in memory or in persistence
        game = game_manager.get_game(game_id)
        if not game:
            # Check if game exists in persistence
            saved_games = game_persistence.list_saved_games()
            if game_id not in saved_games:
                raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        actions = tracker.get_game_action_history(game_id, limit) if tracker else []
        
        return {
            "actions": actions,
            "total_actions": len(actions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get action history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/turns/{turn_number}", operation_id="get_turn_summary")
async def get_turn_summary(game_id: str, turn_number: int):
    """Get a summary of a specific turn with all its actions."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
            
        turn_data = tracker.get_turn_summary(game_id, turn_number)
        if not turn_data:
            raise HTTPException(status_code=404, detail=f"Turn {turn_number} not found in tracking database")
        
        return turn_data
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get turn summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/current-turn", operation_id="get_current_turn_info")
async def get_current_turn_info(game_id: str):
    """Get information about the current turn from the tracker database."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
            
        current_turn = tracker.get_current_turn_info(game_id)
        if not current_turn:
            raise HTTPException(status_code=404, detail="Current turn info not found in tracking database")
        
        return current_turn
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get current turn info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/player/{player_id}/actions", operation_id="get_player_actions")
async def get_player_actions(game_id: str, player_id: str, limit: int = 20):
    """Get a player's recent actions in a game."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        player = game.players.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        tracker = get_action_tracker()
        actions = tracker.get_player_actions(game_id, player_id, limit) if tracker else []
        
        return {
            "player_name": player.name,
            "actions": actions,
            "total_actions": len(actions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player actions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/stats", operation_id="get_game_stats")
async def get_game_stats(game_id: str):
    """Get statistical information about a game from the tracker database."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
            
        stats = tracker.get_game_stats(game_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Game stats not found in tracking database")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get game stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints for Statistical Comparisons
@app.post("/api/analytics/compare", operation_id="get_analytics_comparison")
async def get_analytics_comparison(request: dict):
    """Get detailed analytics comparison data with raw statistics for confidence intervals."""
    try:
        comparison_type = request.get('type')  # 'characters' or 'models'
        entity_names = request.get('entities', [])  # List of character/model names to compare
        
        if not comparison_type or comparison_type not in ['characters', 'models']:
            raise HTTPException(status_code=400, detail="Invalid comparison type. Must be 'characters' or 'models'")
        
        if not entity_names or len(entity_names) < 2:
            raise HTTPException(status_code=400, detail="Must provide at least 2 entities to compare")
        
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
        
        # Get all completed games for analysis
        completed_games = tracker.get_historical_game_stats(limit=1000)
        
        if not completed_games:
            return {
                "comparison_type": comparison_type,
                "entities": entity_names,
                "data": [],
                "raw_data": {},
                "message": "No completed games found for analysis"
            }
        
        # Collect raw data for each entity
        entity_raw_data = {}
        
        for game in completed_games:
            game_id = game['game_id']
            detailed_stats = tracker.get_game_completion_stats(game_id)
            
            if not detailed_stats or 'player_statistics' not in detailed_stats:
                continue
            
            player_statistics = detailed_stats['player_statistics']
            winner_player_id = detailed_stats.get('winner_player_id')
            
            for player_id, player_data in player_statistics.items():
                # Determine entity name based on comparison type
                if comparison_type == 'characters':
                    entity_name = player_data['player_name']
                else:  # models
                    entity_name = player_data.get('model_used', 'unknown')
                
                # Skip if this entity is not in our comparison list
                if entity_name not in entity_names:
                    continue
                
                if entity_name not in entity_raw_data:
                    entity_raw_data[entity_name] = {
                        'games': [],
                        'wins': [],
                        'success_rates': [],
                        'attacks_per_game': [],
                        'messages_per_game': [],
                        'card_trades_per_game': [],
                        'fortifications_per_game': [],
                        'army_placements_per_game': [],
                        'decision_times': []
                    }
                
                # Record win/loss (1 for win, 0 for loss)
                is_winner = 1 if player_id == winner_player_id else 0
                entity_raw_data[entity_name]['wins'].append(is_winner)
                
                # Record success rate for this game
                total_actions = player_data.get('total_actions', 0)
                successful_actions = player_data.get('successful_actions', 0)
                success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0
                entity_raw_data[entity_name]['success_rates'].append(success_rate)
                
                # Record action counts per game
                action_breakdown = player_data.get('action_breakdown', {})
                entity_raw_data[entity_name]['attacks_per_game'].append(
                    action_breakdown.get('attack_territory', {}).get('total', 0)
                )
                
                messages_count = (
                    action_breakdown.get('send_message', {}).get('total', 0) +
                    action_breakdown.get('broadcast_message', {}).get('total', 0)
                )
                entity_raw_data[entity_name]['messages_per_game'].append(messages_count)
                entity_raw_data[entity_name]['card_trades_per_game'].append(
                    action_breakdown.get('trade_cards', {}).get('total', 0)
                )
                entity_raw_data[entity_name]['fortifications_per_game'].append(
                    action_breakdown.get('fortify_position', {}).get('total', 0)
                )
                entity_raw_data[entity_name]['army_placements_per_game'].append(
                    action_breakdown.get('place_armies', {}).get('total', 0)
                )
                
                # Record successful action counts for rate-based metrics
                successful_attacks = action_breakdown.get('attack_territory', {}).get('successful', 0)
                successful_messages = (
                    action_breakdown.get('send_message', {}).get('successful', 0) +
                    action_breakdown.get('broadcast_message', {}).get('successful', 0)
                )
                successful_card_trades = action_breakdown.get('trade_cards', {}).get('successful', 0)
                successful_fortifications = action_breakdown.get('fortify_position', {}).get('successful', 0)
                successful_army_placements = action_breakdown.get('place_armies', {}).get('successful', 0)
                
                # Calculate rates (per successful action) for this game
                if successful_actions > 0:
                    attack_rate = (successful_attacks / successful_actions) * 100
                    message_rate = (successful_messages / successful_actions) * 100
                    card_trade_rate = (successful_card_trades / successful_actions) * 100
                    fortification_rate = (successful_fortifications / successful_actions) * 100
                    army_placement_rate = (successful_army_placements / successful_actions) * 100
                else:
                    attack_rate = message_rate = card_trade_rate = fortification_rate = army_placement_rate = 0
                
                # Add rate-based data tracking
                if 'attack_rates' not in entity_raw_data[entity_name]:
                    entity_raw_data[entity_name].update({
                        'attack_rates': [],
                        'message_rates': [],
                        'card_trade_rates': [],
                        'fortification_rates': [],
                        'army_placement_rates': []
                    })
                
                entity_raw_data[entity_name]['attack_rates'].append(attack_rate)
                entity_raw_data[entity_name]['message_rates'].append(message_rate)
                entity_raw_data[entity_name]['card_trade_rates'].append(card_trade_rate)
                entity_raw_data[entity_name]['fortification_rates'].append(fortification_rate)
                entity_raw_data[entity_name]['army_placement_rates'].append(army_placement_rate)
                
                # Record decision time (for models)
                decision_time = player_data.get('avg_decision_time', 0) or 0
                if decision_time > 0:
                    entity_raw_data[entity_name]['decision_times'].append(decision_time)
        
        # Calculate statistics with confidence intervals
        import statistics
        import math
        
        def calculate_confidence_interval(data, confidence=0.95):
            """Calculate mean and 95% confidence interval for a dataset."""
            if not data:
                return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'sample_size': 0}
            
            n = len(data)
            mean = statistics.mean(data)
            
            if n == 1:
                return {'mean': mean, 'ci_lower': mean, 'ci_upper': mean, 'sample_size': n}
            
            # Calculate standard error
            std_dev = statistics.stdev(data)
            std_error = std_dev / math.sqrt(n)
            
            # Use t-distribution for small samples, normal for large
            if n < 30:
                # Approximate t-value for 95% confidence (t_0.025 for various df)
                t_values = {
                    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                    8: 2.365, 9: 2.306, 10: 2.262, 15: 2.131, 20: 2.086, 25: 2.064, 29: 2.045
                }
                df = n - 1
                if df in t_values:
                    t_val = t_values[df]
                elif df >= 30:
                    t_val = 1.96  # Normal approximation
                else:
                    # Find closest df
                    closest_df = min(t_values.keys(), key=lambda x: abs(x - df))
                    t_val = t_values[closest_df]
            else:
                t_val = 1.96  # Normal distribution
            
            margin_error = t_val * std_error
            
            return {
                'mean': round(mean, 3),
                'ci_lower': round(mean - margin_error, 3),
                'ci_upper': round(mean + margin_error, 3),
                'sample_size': n,
                'std_dev': round(std_dev, 3),
                'std_error': round(std_error, 3)
            }
        
        # Process data for each entity
        comparison_data = []
        
        for entity_name in entity_names:
            if entity_name not in entity_raw_data:
                # Entity not found in data
                comparison_data.append({
                    'name': entity_name,
                    'games_played': 0,
                    'win_rate': calculate_confidence_interval([]),
                    'success_rate': calculate_confidence_interval([]),
                    'attacks_per_game': calculate_confidence_interval([]),
                    'messages_per_game': calculate_confidence_interval([]),
                    'card_trades_per_game': calculate_confidence_interval([]),
                    'fortifications_per_game': calculate_confidence_interval([]),
                    'army_placements_per_game': calculate_confidence_interval([]),
                    'avg_decision_time': calculate_confidence_interval([])
                })
                continue
            
            raw_data = entity_raw_data[entity_name]
            
            # Convert win rate to percentage for CI calculation
            win_rates_pct = [w * 100 for w in raw_data['wins']]
            
            entity_stats = {
                'name': entity_name,
                'games_played': len(raw_data['wins']),
                'win_rate': calculate_confidence_interval(win_rates_pct),
                'success_rate': calculate_confidence_interval(raw_data['success_rates']),
                'attacks_per_game': calculate_confidence_interval(raw_data['attacks_per_game']),
                'messages_per_game': calculate_confidence_interval(raw_data['messages_per_game']),
                'card_trades_per_game': calculate_confidence_interval(raw_data['card_trades_per_game']),
                'fortifications_per_game': calculate_confidence_interval(raw_data['fortifications_per_game']),
                'army_placements_per_game': calculate_confidence_interval(raw_data['army_placements_per_game']),
                'avg_decision_time': calculate_confidence_interval(raw_data['decision_times']),
                # Rate-based metrics (behavior preferences)
                'attack_rate': calculate_confidence_interval(raw_data.get('attack_rates', [])),
                'message_rate': calculate_confidence_interval(raw_data.get('message_rates', [])),
                'card_trade_rate': calculate_confidence_interval(raw_data.get('card_trade_rates', [])),
                'fortification_rate': calculate_confidence_interval(raw_data.get('fortification_rates', [])),
                'army_placement_rate': calculate_confidence_interval(raw_data.get('army_placement_rates', []))
            }
            
            comparison_data.append(entity_stats)
        
        return {
            "comparison_type": comparison_type,
            "entities": entity_names,
            "data": comparison_data,
            "total_games_analyzed": len(completed_games),
            "metrics": [
                'win_rate', 'success_rate', 'attacks_per_game', 'messages_per_game',
                'card_trades_per_game', 'fortifications_per_game', 'army_placements_per_game', 'avg_decision_time'
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        risk_logger.log_error(f"Analytics comparison failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/entities", operation_id="get_available_entities")
async def get_available_entities():
    """Get all available characters and models for analytics comparison."""
    try:
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
        
        # Get completed games to find unique characters and models
        completed_games = tracker.get_historical_game_stats(limit=1000)
        
        characters = set()
        models = set()
        
        for game in completed_games:
            game_id = game['game_id']
            detailed_stats = tracker.get_game_completion_stats(game_id)
            
            if not detailed_stats or 'player_statistics' not in detailed_stats:
                continue
            
            player_statistics = detailed_stats['player_statistics']
            
            for player_data in player_statistics.values():
                character_name = player_data['player_name']
                model_name = player_data.get('model_used', 'unknown')
                
                characters.add(character_name)
                if model_name != 'unknown':
                    models.add(model_name)
        
        return {
            "characters": sorted(list(characters)),
            "models": sorted(list(models)),
            "total_characters": len(characters),
            "total_models": len(models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get available entities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Leaderboard Endpoints
@app.get("/api/leaderboard/characters", operation_id="get_character_leaderboard")
async def get_character_leaderboard(limit: int = 50):
    """Get character leaderboard with comprehensive action statistics."""
    try:
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
        
        # Get all completed games
        completed_games = tracker.get_historical_game_stats(limit=1000)
        
        if not completed_games:
            return {
                "characters": [],
                "total_characters": 0,
                "games_analyzed": 0
            }
        
        # Aggregate character statistics
        character_stats = {}
        
        for game in completed_games:
            game_id = game['game_id']
            
            # Get detailed completion stats for this game
            detailed_stats = tracker.get_game_completion_stats(game_id)
            if not detailed_stats or 'player_statistics' not in detailed_stats:
                continue
            
            player_statistics = detailed_stats['player_statistics']
            winner_player_id = detailed_stats.get('winner_player_id')
            
            for player_id, player_data in player_statistics.items():
                character_name = player_data['player_name']  # Character name
                
                if character_name not in character_stats:
                    character_stats[character_name] = {
                        'character_name': character_name,
                        'games_played': 0,
                        'wins': 0,
                        'total_actions': 0,
                        'successful_actions': 0,
                        'failed_actions': 0,
                        'action_types': {},
                        'total_turns': 0,
                        'avg_decision_time': 0,
                        'models_used': set()
                    }
                
                stats = character_stats[character_name]
                stats['games_played'] += 1
                
                # Check if this character won
                if player_id == winner_player_id:
                    stats['wins'] += 1
                
                # Aggregate action statistics
                stats['total_actions'] += player_data.get('total_actions', 0)
                stats['successful_actions'] += player_data.get('successful_actions', 0)
                stats['failed_actions'] += player_data.get('failed_actions', 0)
                stats['total_turns'] += player_data.get('turns_played', 0)
                
                # Track models used
                model_used = player_data.get('model_used', 'unknown')
                if model_used != 'unknown':
                    stats['models_used'].add(model_used)
                
                # Aggregate action type breakdown
                action_breakdown = player_data.get('action_breakdown', {})
                for action_type, action_data in action_breakdown.items():
                    if action_type not in stats['action_types']:
                        stats['action_types'][action_type] = {
                            'total': 0,
                            'successful': 0,
                            'failed': 0
                        }
                    
                    stats['action_types'][action_type]['total'] += action_data.get('total', 0)
                    stats['action_types'][action_type]['successful'] += action_data.get('successful', 0)
                    stats['action_types'][action_type]['failed'] += action_data.get('failed', 0)
                
                # Update average decision time (weighted average)
                player_avg_time = player_data.get('avg_decision_time', 0) or 0
                if player_avg_time and player_avg_time > 0:
                    current_avg = stats['avg_decision_time'] or 0
                    games_count = stats['games_played']
                    if games_count > 0:
                        stats['avg_decision_time'] = ((current_avg * (games_count - 1)) + player_avg_time) / games_count
        
        # Calculate derived statistics and convert to list
        leaderboard = []
        for character_name, stats in character_stats.items():
            # Calculate rates
            win_rate = (stats['wins'] / stats['games_played']) * 100 if stats['games_played'] > 0 else 0
            success_rate = (stats['successful_actions'] / stats['total_actions']) * 100 if stats['total_actions'] > 0 else 0
            
            # Calculate action type averages per game
            attacks_per_game = stats['action_types'].get('attack_territory', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            messages_per_game = (stats['action_types'].get('send_message', {}).get('total', 0) + stats['action_types'].get('broadcast_message', {}).get('total', 0)) / stats['games_played'] if stats['games_played'] > 0 else 0
            card_trades_per_game = stats['action_types'].get('trade_cards', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            fortifications_per_game = stats['action_types'].get('fortify_position', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            army_placements_per_game = stats['action_types'].get('place_armies', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            
            leaderboard.append({
                'character_name': character_name,
                'games_played': stats['games_played'],
                'wins': stats['wins'],
                'win_rate': round(win_rate, 1),
                'success_rate': round(success_rate, 1),
                'attacks_per_game': round(attacks_per_game, 1),
                'messages_per_game': round(messages_per_game, 1),
                'card_trades_per_game': round(card_trades_per_game, 1),
                'fortifications_per_game': round(fortifications_per_game, 1),
                'army_placements_per_game': round(army_placements_per_game, 1),
                'avg_decision_time': round(stats['avg_decision_time'], 2),
                'models_used': list(stats['models_used']) if stats['models_used'] else ['unknown']
            })
        
        # Sort by win rate, then by games played
        leaderboard.sort(key=lambda x: (x['win_rate'], x['games_played']), reverse=True)
        
        # Limit results
        leaderboard = leaderboard[:limit]
        
        return {
            "characters": leaderboard,
            "total_characters": len(character_stats),
            "games_analyzed": len(completed_games)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get character leaderboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leaderboard/players", operation_id="get_player_leaderboard")
async def get_player_leaderboard(limit: int = 50):
    """Get player leaderboard with comprehensive statistics (deprecated - use characters instead)."""
    try:
        # Redirect to character leaderboard for consistency
        return await get_character_leaderboard(limit)
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player leaderboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leaderboard/models", operation_id="get_model_leaderboard")
async def get_model_leaderboard(limit: int = 20):
    """Get model leaderboard with comprehensive action statistics."""
    try:
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
        
        # Get all completed games
        completed_games = tracker.get_historical_game_stats(limit=1000)
        
        if not completed_games:
            return {
                "models": [],
                "total_models": 0,
                "games_analyzed": 0
            }
        
        # Aggregate model statistics
        model_stats = {}
        
        for game in completed_games:
            game_id = game['game_id']
            
            # Get detailed completion stats for this game
            detailed_stats = tracker.get_game_completion_stats(game_id)
            if not detailed_stats or 'player_statistics' not in detailed_stats:
                continue
            
            player_statistics = detailed_stats['player_statistics']
            winner_player_id = detailed_stats.get('winner_player_id')
            
            for player_id, player_data in player_statistics.items():
                model_used = player_data.get('model_used', 'unknown')
                
                if model_used not in model_stats:
                    model_stats[model_used] = {
                        'model_name': model_used,
                        'games_played': 0,
                        'wins': 0,
                        'total_actions': 0,
                        'successful_actions': 0,
                        'failed_actions': 0,
                        'action_types': {},
                        'total_turns': 0,
                        'total_decision_time': 0,
                        'decision_count': 0,
                        'players_using': set()
                    }
                
                stats = model_stats[model_used]
                stats['games_played'] += 1
                stats['players_using'].add(player_data['player_name'])
                
                # Check if this model won
                if player_id == winner_player_id:
                    stats['wins'] += 1
                
                # Aggregate action statistics
                stats['total_actions'] += player_data.get('total_actions', 0)
                stats['successful_actions'] += player_data.get('successful_actions', 0)
                stats['failed_actions'] += player_data.get('failed_actions', 0)
                stats['total_turns'] += player_data.get('turns_played', 0)
                
                # Aggregate decision time
                player_decision_time = player_data.get('avg_decision_time', 0) or 0
                player_decision_count = player_data.get('decision_count', 0) or 0
                if player_decision_time > 0 and player_decision_count > 0:
                    stats['total_decision_time'] += player_decision_time * player_decision_count
                    stats['decision_count'] += player_decision_count
                
                # Aggregate action type breakdown
                action_breakdown = player_data.get('action_breakdown', {})
                for action_type, action_data in action_breakdown.items():
                    if action_type not in stats['action_types']:
                        stats['action_types'][action_type] = {
                            'total': 0,
                            'successful': 0,
                            'failed': 0
                        }
                    
                    stats['action_types'][action_type]['total'] += action_data.get('total', 0)
                    stats['action_types'][action_type]['successful'] += action_data.get('successful', 0)
                    stats['action_types'][action_type]['failed'] += action_data.get('failed', 0)
        
        # Calculate derived statistics and convert to list
        leaderboard = []
        for model_name, stats in model_stats.items():
            # Calculate rates
            win_rate = (stats['wins'] / stats['games_played']) * 100 if stats['games_played'] > 0 else 0
            success_rate = (stats['successful_actions'] / stats['total_actions']) * 100 if stats['total_actions'] > 0 else 0
            
            # Calculate action type averages per game (same as character leaderboard)
            attacks_per_game = stats['action_types'].get('attack_territory', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            messages_per_game = (stats['action_types'].get('send_message', {}).get('total', 0) + stats['action_types'].get('broadcast_message', {}).get('total', 0)) / stats['games_played'] if stats['games_played'] > 0 else 0
            card_trades_per_game = stats['action_types'].get('trade_cards', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            fortifications_per_game = stats['action_types'].get('fortify_position', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            army_placements_per_game = stats['action_types'].get('place_armies', {}).get('total', 0) / stats['games_played'] if stats['games_played'] > 0 else 0
            
            # Calculate average decision time
            avg_decision_time = stats['total_decision_time'] / stats['decision_count'] if stats['decision_count'] > 0 else 0
            
            # Convert players set to count
            unique_players = len(stats['players_using'])
            
            leaderboard.append({
                'model_name': model_name,
                'games_played': stats['games_played'],
                'unique_players': unique_players,
                'wins': stats['wins'],
                'win_rate': round(win_rate, 1),
                'success_rate': round(success_rate, 1),
                'attacks_per_game': round(attacks_per_game, 1),
                'messages_per_game': round(messages_per_game, 1),
                'card_trades_per_game': round(card_trades_per_game, 1),
                'fortifications_per_game': round(fortifications_per_game, 1),
                'army_placements_per_game': round(army_placements_per_game, 1),
                'avg_decision_time': round(avg_decision_time, 2)
            })
        
        # Sort by win rate, then by games played
        leaderboard.sort(key=lambda x: (x['win_rate'], x['games_played']), reverse=True)
        
        # Limit results
        leaderboard = leaderboard[:limit]
        
        return {
            "models": leaderboard,
            "total_models": len(model_stats),
            "games_analyzed": len(completed_games)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get model leaderboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent Decision Endpoints
@app.get("/api/games/{game_id}/agent-decisions", operation_id="get_agent_decisions")
async def get_agent_decisions(game_id: str, player_id: Optional[str] = None, turn_number: Optional[int] = None, limit: int = 20):
    """Get agent decisions for a game with optional player and turn filters."""
    try:
        # Check if game exists in memory or in persistence
        game = game_manager.get_game(game_id)
        if not game:
            # Check if game exists in persistence
            saved_games = game_persistence.list_saved_games()
            if game_id not in saved_games:
                raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        decisions = tracker.get_agent_decisions(
            game_id=game_id,
            player_id=player_id,
            turn_number=turn_number,
            limit=limit
        ) if tracker else []
        
        return {
            "decisions": decisions,
            "total_count": len(decisions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get agent decisions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/agent-decisions/{decision_id}", operation_id="get_agent_decision_detail")
async def get_agent_decision_detail(game_id: str, decision_id: int):
    """Get detailed information about a specific agent decision."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        tracker = get_action_tracker()
        if not tracker:
            raise HTTPException(status_code=503, detail="Action tracking service unavailable")
            
        decision = tracker.get_agent_decision_detail(decision_id)
        if not decision:
            raise HTTPException(status_code=404, detail=f"Agent decision {decision_id} not found")
        
        if decision.get('game_id') != game_id:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} does not belong to game {game_id}")
            
        return decision
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get agent decision detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/player/{player_id}/agent-decisions", operation_id="get_player_agent_decisions")
async def get_player_agent_decisions(game_id: str, player_id: str, turn_number: Optional[int] = None, limit: int = 20):
    """Get agent decisions for a specific player in a game."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        if player_id not in game.players:
            raise HTTPException(status_code=404, detail="Player not found in this game")
        
        action_tracker = get_action_tracker()
        decisions = action_tracker.get_agent_decisions(
            game_id=game_id,
            player_id=player_id,
            turn_number=turn_number,
            limit=limit
        )
        
        return {
            "player_name": game.players[player_id].name,
            "decisions": decisions,
            "total_count": len(decisions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get player agent decisions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/{game_id}/agent-decisions/analytics", operation_id="get_agent_decision_analytics")
async def get_agent_decision_analytics(game_id: str):
    """Get analytics about agent decisions for a game."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        

        action_tracker = get_action_tracker()
        analytics = action_tracker.get_agent_decision_analytics(game_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get agent decision analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Territory Reference Endpoints
@app.get("/api/territories", operation_id="get_territory_reference")
async def get_territory_reference():
    """Get territory and continent reference data."""
    try:
        import json
        
        # Load territory data from the authoritative source
        territories_file = os.path.join(os.path.dirname(__file__), 'data', 'territories.json')
        
        with open(territories_file, 'r') as f:
            territory_data = json.load(f)
        
        return {
            "territories": territory_data.get("territories", {}),
            "continents": territory_data.get("continents", {}),
            "total_territories": len(territory_data.get("territories", {}))
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Territory reference data not found")
    except Exception as e:
        risk_logger.log_error(f"Get territory reference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Territory Information Endpoint
@app.get("/api/games/{game_id}/territories/{territory_name}", operation_id="get_territory_info")
async def get_territory_info(game_id: str, territory_name: str):
    """Get detailed information about a territory."""
    try:
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        territory = game.territory_manager.get_territory(territory_name)
        if not territory:
            raise HTTPException(status_code=404, detail="Territory not found")
        
        owner_info = None
        if territory.owner:
            owner_player = game.players.get(territory.owner)
            if owner_player:
                owner_info = {
                    "player_id": owner_player.player_id,
                    "name": owner_player.name,
                    "color": owner_player.color
                }
        
        # Get adjacent territories with their info
        adjacent_territories = []
        for adj_name in territory.adjacent_territories:
            adj_territory = game.territory_manager.get_territory(adj_name)
            if adj_territory:
                adj_owner_info = None
                if adj_territory.owner:
                    adj_owner_player = game.players.get(adj_territory.owner)
                    if adj_owner_player:
                        adj_owner_info = {
                            "player_id": adj_owner_player.player_id,
                            "name": adj_owner_player.name,
                            "color": adj_owner_player.color
                        }
                
                adjacent_territories.append({
                    "name": adj_territory.name,
                    "continent": adj_territory.continent,
                    "army_count": adj_territory.army_count,
                    "owner": adj_owner_info
                })
        
        return {
            "name": territory.name,
            "continent": territory.continent,
            "army_count": territory.army_count,
            "owner": owner_info,
            "adjacent_territories": adjacent_territories
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get territory info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Phase Management Endpoints
@app.post("/api/games/auto-advance-phase", operation_id="auto_advance_phase")
async def auto_advance_phase(request: dict, background_tasks: BackgroundTasks):
    """
    Auto-advance game phase based on current game state conditions.
    
    This endpoint checks if the current game state allows for automatic phase progression
    and advances the phase if appropriate. It's called by the GameRunner to handle
    automatic phase flow without relying on agents.
    
    **Request Body:**
    - `game_id` (str): ID of the game
    - `player_id` (str): Current player ID
    
    **Response:**
    - `phase_advanced` (bool): Whether phase was advanced
    - `old_phase` (str): Previous phase
    - `new_phase` (str): New phase (if advanced)
    - `message` (str): Description of what happened
    """
    try:
        game_id = request.get("game_id")
        player_id = request.get("player_id")
        
        if not game_id or not player_id:
            raise HTTPException(status_code=400, detail="Missing game_id or player_id")
        
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get current phase
        old_phase = game.game_phase.value
        
        # Try to auto-advance phase
        phase_advanced = game.auto_advance_phase()
        
        if phase_advanced:
            new_phase = game.game_phase.value
            
            # Update game phase in action tracker#
            action_tracker = get_action_tracker()
            action_tracker.update_game_phase(game_id, new_phase)
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, game_id)
            
            return {
                "phase_advanced": True,
                "old_phase": old_phase,
                "new_phase": new_phase,
                "message": f"Auto-advanced from {old_phase} to {new_phase}"
            }
        else:
            return {
                "phase_advanced": False,
                "old_phase": old_phase,
                "new_phase": old_phase,
                "message": "No phase advancement needed"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Auto advance phase failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/games/force-advance-phase", operation_id="force_advance_phase")
async def force_advance_phase(request: dict, background_tasks: BackgroundTasks):
    """
    Force advance game phase when a player gets stuck.
    
    This endpoint forces phase progression when normal game flow fails.
    It's used by the GameRunner as a recovery mechanism for stuck players.
    
    **Request Body:**
    - `game_id` (str): ID of the game
    - `player_id` (str): Player ID that's stuck
    - `reason` (str): Reason for forcing advancement
    
    **Response:**
    - `success` (bool): Whether advancement was successful
    - `old_phase` (str): Previous phase
    - `new_phase` (str): New phase (if successful)
    - `message` (str): Description of what happened
    """
    try:
        game_id = request.get("game_id")
        player_id = request.get("player_id")
        reason = request.get("reason", "forced")
        
        if not game_id or not player_id:
            raise HTTPException(status_code=400, detail="Missing game_id or player_id")
        
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get current phase
        old_phase = game.game_phase.value
        
        # Force advance phase
        phase_advanced = game.force_advance_phase(reason)
        
        if phase_advanced:
            new_phase = game.game_phase.value
            
            # Update game phase in action tracker
            action_tracker = get_action_tracker()
            action_tracker.update_game_phase(game_id, new_phase)
            
            # Track the forced advancement
            action_tracker.track_action(
                game_id,
                player_id,
                game.turn_number,
                "force_advance_phase",
                {"reason": reason, "old_phase": old_phase, "new_phase": new_phase},
                {"success": True, "message": f"Forced advancement from {old_phase} to {new_phase}"}
            )
            
            # Schedule auto-save
            background_tasks.add_task(auto_save_game_state, game_id)
            
            return {
                "success": True,
                "old_phase": old_phase,
                "new_phase": new_phase,
                "message": f"Forced advancement from {old_phase} to {new_phase} ({reason})"
            }
        else:
            return {
                "success": False,
                "old_phase": old_phase,
                "new_phase": old_phase,
                "message": f"Could not force advance from {old_phase} phase"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Force advance phase failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/api/configs", operation_id="list_configurations")
async def list_configurations():
    """List all saved game configurations."""
    try:
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        
        if not os.path.exists(configs_dir):
            return {"configurations": []}
        
        configurations = []
        for filename in os.listdir(configs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(configs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                    
                    # Extract metadata
                    config_info = {
                        "filename": filename,
                        "name": config.get("gameName", filename.replace('.json', '')),
                        "playerCount": config.get("playerCount", len(config.get("players", []))),
                        "created": os.path.getctime(filepath),
                        "modified": os.path.getmtime(filepath)
                    }
                    configurations.append(config_info)
                    
                except Exception as e:
                    risk_logger.log_error(f"Error reading config {filename}: {e}")
                    continue
        
        # Sort by modification time (newest first)
        configurations.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"configurations": configurations}
        
    except Exception as e:
        risk_logger.log_error(f"List configurations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs/{filename}", operation_id="get_configuration")
async def get_configuration(filename: str):
    """Get a specific configuration by filename."""
    try:
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        filepath = os.path.join(configs_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return {"configuration": config}
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configs", operation_id="save_configuration")
async def save_configuration(request: dict):
    """Save a game configuration to the server."""
    try:
        config_name = request.get("name", "unnamed-config")
        config_data = request.get("configuration", {})
        
        if not config_data:
            raise HTTPException(status_code=400, detail="Configuration data is required")
        
        # Sanitize filename
        import re
        safe_name = re.sub(r'[^\w\-_\.]', '-', config_name)
        filename = f"{safe_name}.json"
        
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        filepath = os.path.join(configs_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        risk_logger.log_info(f"Configuration saved: {filename}")
        
        return {
            "success": True,
            "filename": filename,
            "message": f"Configuration '{config_name}' saved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Save configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/configs/{filename}", operation_id="delete_configuration")
async def delete_configuration(filename: str):
    """Delete a configuration file."""
    try:
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        filepath = os.path.join(configs_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        os.remove(filepath)
        
        risk_logger.log_info(f"Configuration deleted: {filename}")
        
        return {
            "success": True,
            "message": f"Configuration '{filename}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Delete configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config", operation_id="get_frontend_config")
async def get_frontend_config():
    """Get frontend configuration including API base URL with robust fallbacks."""
    try:
        # Priority order for determining API base URL:
        # 1. Railway provided domain (production)
        # 2. Custom environment variable override
        # 3. Host detection from environment
        # 4. Localhost fallback (development)
        
        railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN')
        custom_api_base = os.getenv('RISK_API_BASE_URL')
        
        if railway_domain:
            # Railway deployment - use HTTPS with provided domain
            api_base = f"https://{railway_domain}/api"
            environment = "railway"
        elif custom_api_base:
            # Custom override (useful for other cloud providers)
            api_base = custom_api_base
            environment = "custom"
        else:
            # Fallback: try to detect from environment or use localhost
            host = os.getenv('HOST', 'localhost')
            port = os.getenv('PORT', '8080')
            
            if host == 'localhost' or host == '127.0.0.1':
                # Local development
                api_base = f"http://{host}:{port}/api"
                environment = "development"
            else:
                # Deployed somewhere else - assume HTTPS for security
                if port in ['80', '443']:
                    api_base = f"https://{host}/api"
                else:
                    api_base = f"https://{host}:{port}/api"
                environment = "production"
        
        return {
            "apiBase": api_base,
            "environment": environment,
            "railwayDomain": railway_domain,
            "detectedHost": host if 'host' in locals() else None,
            "detectedPort": port if 'port' in locals() else None
        }
        
    except Exception as e:
        risk_logger.log_error(f"Get frontend config failed: {e}")
        # Ultimate fallback
        return {
            "apiBase": "http://localhost:8080/api",
            "environment": "fallback",
            "error": str(e)
        }

@app.get("/api/support-config", operation_id="get_support_config")
async def get_support_config():
    """Get support configuration including payment links from environment variables."""
    try:
        paypal_url = os.getenv('PAYPAL_DONATION_URL', '')
        bitcoin_address = os.getenv('BITCOIN_ADDRESS', '')
        
        return {
            "paypal_url": paypal_url,
            "bitcoin_address": bitcoin_address,
            "has_paypal": bool(paypal_url and paypal_url != 'https://www.paypal.com/donate/?hosted_button_id=YOUR_PAYPAL_BUTTON_ID'),
            "has_bitcoin": bool(bitcoin_address and bitcoin_address != 'bc1qexamplebitcoinaddresshere')
        }
        
    except Exception as e:
        risk_logger.log_error(f"Get support config failed: {e}")
        return {
            "paypal_url": "",
            "bitcoin_address": "",
            "has_paypal": False,
            "has_bitcoin": False,
            "error": str(e)
        }

@app.get("/api/health/config", operation_id="health_check_config")
async def health_check_config():
    """Health check that includes configuration validation."""
    try:
        config = await get_frontend_config()
        
        return {
            "status": "healthy",
            "config": config,
            "timestamp": datetime.utcnow().isoformat(),
            "environment_vars": {
                "RAILWAY_PUBLIC_DOMAIN": bool(os.getenv('RAILWAY_PUBLIC_DOMAIN')),
                "RISK_API_BASE_URL": bool(os.getenv('RISK_API_BASE_URL')),
                "HOST": os.getenv('HOST', 'not_set'),
                "PORT": os.getenv('PORT', 'not_set')
            }
        }
        
    except Exception as e:
        risk_logger.log_error(f"Config health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/models", operation_id="get_available_models")
async def get_available_models():
    """Get available AI models from environment configuration."""
    try:
        import json
        
        # Get predefined models from environment
        models_json = os.getenv('RISK_PREDEFINED_MODELS', '{}')
        predefined_models = json.loads(models_json)
        
        # Format for frontend consumption
        models = {}
        for key, model_info in predefined_models.items():
            models[key] = {
                'name': model_info.get('name', key),
                'description': model_info.get('description', 'AI model'),
                'defaultTemp': model_info.get('default_temp', 0.7),
                'specs': model_info.get('description', 'AI model')  # Use description as specs
            }
        
        return {
            "models": models,
            "default_model": os.getenv('RISK_MODEL_NAME', 'gpt-3.5-turbo')
        }
        
    except Exception as e:
        risk_logger.log_error(f"Get available models failed: {e}")
        # Fallback to basic models if environment parsing fails
        return {
            "models": {
                "gpt-3.5-turbo": {
                    "name": "GPT-3.5 Turbo",
                    "description": "Fast and efficient model",
                    "defaultTemp": 0.8,
                    "specs": "Cost-effective choice"
                }
            },
            "default_model": "gpt-3.5-turbo"
        }

# General Character Management Endpoints (Phase-Independent)
@app.post("/api/characters", operation_id="submit_character")
async def submit_character(request: dict):
    """Submit a character (bypasses tournament phase restrictions)."""
    try:
        # Import character manager directly
        from tournament.character_manager import CharacterManager, TournamentCharacter
        
        # Use tournament database path from environment
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        # Extract character data
        name = request.get('name', '').strip()
        temperature = request.get('temperature', 0.7)
        personality = request.get('personality', '').strip()
        custom_instructions = request.get('custom_instructions', '').strip()
        submitted_by = request.get('submitted_by', 'user').strip()
        
        # Validate required fields
        if not name or not personality:
            raise HTTPException(status_code=400, detail="Name and personality are required")
        
        # Validate temperature
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid temperature value")
        
        # Create character (model_name will be assigned randomly during game)
        character = TournamentCharacter(
            id=None,
            name=name,
            model_name='TBD',  # Will be assigned randomly during game
            temperature=temperature,
            personality=personality,
            custom_instructions=custom_instructions,
            created_at=datetime.now(),
            submitted_by=submitted_by
        )
        
        # Submit character directly to character manager (bypassing tournament phases)
        success, message, character_id = char_manager.submit_character(character)
        
        if success:
            return {
                "success": True,
                "message": message,
                "character_id": character_id
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Submit character failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/characters", operation_id="get_all_characters")
async def get_all_characters():
    """Get all characters (bypasses tournament phase restrictions)."""
    try:
        # Import character manager directly
        from tournament.character_manager import CharacterManager
        
        # Use tournament database path from environment
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        characters = char_manager.get_all_characters()
        
        return {
            "characters": [char.to_dict() for char in characters],
            "total": len(characters)
        }
        
    except Exception as e:
        risk_logger.log_error(f"Get all characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/characters/search", operation_id="search_characters")
async def search_characters(q: str = ""):
    """Search for characters by name or personality keywords (bypasses tournament phase restrictions)."""
    try:
        # Import character manager directly
        from tournament.character_manager import CharacterManager
        
        # Use tournament database path from environment
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        # Search characters
        characters = char_manager.search_characters(q)
        
        # Convert to dictionaries for JSON response
        character_dicts = [char.to_dict() for char in characters]
        
        return {
            "characters": character_dicts,
            "total": len(character_dicts),
            "query": q
        }
        
    except Exception as e:
        risk_logger.log_error(f"Search characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/characters/stats", operation_id="get_character_stats")
async def get_character_stats():
    """Get character statistics (bypasses tournament phase restrictions)."""
    try:
        # Import character manager directly
        from tournament.character_manager import CharacterManager
        
        # Use tournament database path from environment
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        stats = char_manager.get_character_stats()
        
        return {
            "stats": stats,
            "total_characters": len(stats)
        }
        
    except Exception as e:
        risk_logger.log_error(f"Get character stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Tournament Mode Endpoints
@app.get("/api/tournament/status", operation_id="get_tournament_status")
async def get_tournament_status():
    """Get current tournament status."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        
        if not tournament_mode:
            return {
                "tournament_mode": False,
                "message": "Tournament mode is disabled"
            }
        
        # Get tournament manager instance (we'll need to add this to game_manager)
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            return {
                "tournament_mode": True,
                "active": False,
                "message": "Tournament not started"
            }
        
        status = game_manager.tournament_manager.get_tournament_status()
        status["tournament_mode"] = True
        status["active"] = game_manager.tournament_manager.is_tournament_active()
        
        return status
        
    except Exception as e:
        risk_logger.log_error(f"Get tournament status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/characters", operation_id="submit_tournament_character")
async def submit_tournament_character(request: dict):
    """Submit a character for the tournament."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        # Extract character data
        name = request.get('name', '').strip()
        temperature = request.get('temperature', 0.7)
        personality = request.get('personality', '').strip()
        custom_instructions = request.get('custom_instructions', '').strip()
        submitted_by = request.get('submitted_by', 'anonymous').strip()
        
        # Validate required fields (model_name no longer required)
        if not name or not personality:
            raise HTTPException(status_code=400, detail="Name and personality are required")
        
        # Validate temperature
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid temperature value")
        
        # Submit character (model_name will be assigned randomly at game start)
        success, message = game_manager.tournament_manager.submit_character(
            name=name,
            model_name=None,  # Will be assigned randomly
            temperature=temperature,
            personality=personality,
            custom_instructions=custom_instructions,
            submitted_by=submitted_by
        )
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Submit tournament character failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/characters", operation_id="get_tournament_characters")
async def get_tournament_characters():
    """Get all characters available for voting."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        characters = game_manager.tournament_manager.get_characters_for_voting()
        
        return {
            "characters": characters,
            "total": len(characters)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get tournament characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/characters/search", operation_id="search_tournament_characters")
async def search_tournament_characters(q: str = ""):
    """Search for characters by name or personality keywords."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Import character manager to use directly
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        # Search characters
        characters = char_manager.search_characters(q)
        
        # Convert to dictionaries for JSON response
        character_dicts = [char.to_dict() for char in characters]
        
        return {
            "characters": character_dicts,
            "total": len(character_dicts),
            "query": q
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Search tournament characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/vote", operation_id="submit_tournament_vote")
async def submit_tournament_vote(request: dict):
    """Submit a vote for a character."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        character_id = request.get('character_id')
        voter_ip = request.get('voter_ip')
        voter_session = request.get('voter_session')
        
        if not character_id:
            raise HTTPException(status_code=400, detail="character_id is required")
        
        try:
            character_id = int(character_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid character_id")
        
        # Submit vote
        success, message = game_manager.tournament_manager.submit_vote(
            character_id=character_id,
            voter_ip=voter_ip,
            voter_session=voter_session
        )
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Submit tournament vote failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/votes", operation_id="get_tournament_vote_results")
async def get_tournament_vote_results():
    """Get current voting results."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        results = game_manager.tournament_manager.get_vote_results()
        
        return {
            "results": results,
            "total_votes": sum(result.get('vote_count', 0) for result in results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get tournament vote results failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/stats", operation_id="get_tournament_character_stats")
async def get_tournament_character_stats():
    """Get character statistics."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        stats = game_manager.tournament_manager.get_character_stats()
        
        return {
            "stats": stats,
            "total_characters": len(stats)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Get tournament character stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/debug", operation_id="get_tournament_debug_info")
async def get_tournament_debug_info():
    """Get debug information about tournament state."""
    try:
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        
        debug_info = {
            "tournament_mode_enabled": tournament_mode,
            "game_manager_has_tournament_manager": hasattr(game_manager, 'tournament_manager'),
            "tournament_manager_exists": hasattr(game_manager, 'tournament_manager') and game_manager.tournament_manager is not None,
            "active_games": len(game_manager.games),
            "game_ids": list(game_manager.games.keys())
        }
        
        if hasattr(game_manager, 'tournament_manager') and game_manager.tournament_manager:
            try:
                status = game_manager.tournament_manager.get_tournament_status()
                debug_info["tournament_status"] = status
                debug_info["tournament_active"] = game_manager.tournament_manager.is_tournament_active()
            except Exception as e:
                debug_info["tournament_status_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        risk_logger.log_error(f"Get tournament debug info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/start", operation_id="start_tournament")
async def start_tournament():
    """Start a new tournament cycle (admin function)."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Initialize tournament manager if not exists
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            # Import tournament manager
            from tournament.tournament_manager import TournamentManager
            
            # Create tournament config from environment
            config = {
                'db_path': os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db'),
                'submit_duration': int(os.getenv('TOURNAMENT_SUBMIT_PHASE_DURATION', '900')),
                'voting_duration': int(os.getenv('TOURNAMENT_VOTING_PHASE_DURATION', '900')),
                'game_duration': int(os.getenv('TOURNAMENT_GAME_PHASE_DURATION', '3600')),
                'end_screen_duration': int(os.getenv('TOURNAMENT_END_SCREEN_DURATION', '300')),
                'auto_restart': os.getenv('TOURNAMENT_AUTO_RESTART', 'true').lower() == 'true',
                'max_submissions': int(os.getenv('TOURNAMENT_MAX_SUBMISSIONS', '20')),
                'selected_characters': int(os.getenv('TOURNAMENT_SELECTED_CHARACTERS', '4'))
            }
            
            game_manager.tournament_manager = TournamentManager(config)
        
        # Start tournament
        tournament_id = await game_manager.tournament_manager.start_tournament()
        
        return {
            "success": True,
            "tournament_id": tournament_id,
            "message": "Tournament started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Start tournament failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/advance-phase", operation_id="force_advance_tournament_phase")
async def force_advance_tournament_phase():
    """Force advance to the next tournament phase (admin function)."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        success = await game_manager.tournament_manager.force_advance_phase()
        
        if success:
            return {
                "success": True,
                "message": "Tournament phase advanced successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to advance tournament phase")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Force advance tournament phase failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/create-game", operation_id="force_create_tournament_game")
async def force_create_tournament_game():
    """Force create tournament game (admin function)."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        # Check if we're in game phase
        status = game_manager.tournament_manager.get_tournament_status()
        if status.get('phase') != 'game':
            raise HTTPException(status_code=400, detail=f"Tournament not in game phase (current: {status.get('phase')})")
        
        # Check if game already exists
        current_game_id = game_manager.tournament_manager.get_current_game_id()
        if current_game_id:
            return {
                "success": True,
                "message": f"Tournament game already exists: {current_game_id}",
                "game_id": current_game_id
            }
        
        # Manually trigger game creation
        await game_manager._start_tournament_game()
        
        # Check if game was created
        new_game_id = game_manager.tournament_manager.get_current_game_id()
        if new_game_id:
            return {
                "success": True,
                "message": f"Tournament game created successfully: {new_game_id}",
                "game_id": new_game_id
            }
        else:
            raise HTTPException(status_code=500, detail="Tournament game creation failed - no game ID set")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Force create tournament game failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/token-exhausted", operation_id="handle_token_exhaustion")
async def handle_token_exhaustion(request: dict):
    """Handle token exhaustion notification from agents."""
    try:
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        agent_name = request.get('agent_name', 'Unknown')
        player_id = request.get('player_id', 'Unknown')
        game_id = request.get('game_id', 'Unknown')
        timestamp = request.get('timestamp', datetime.now().isoformat())
        
        risk_logger.log_error(f"Token exhaustion reported by agent {agent_name} (player {player_id}) in game {game_id} at {timestamp}")
        
        # Notify tournament manager to handle token exhaustion
        await game_manager.tournament_manager.handle_token_exhaustion()
        
        return {
            "success": True,
            "message": "Token exhaustion handled - tournament paused",
            "agent_name": agent_name,
            "timestamp": timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Handle token exhaustion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/restart-match", operation_id="restart_tournament_match")
async def restart_tournament_match(request: dict):
    """Restart tournament match after token exhaustion (admin function)."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        # Restart from token exhaustion
        success, message = await game_manager.tournament_manager.restart_from_token_exhaustion()
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            return {
                "success": False,
                "message": message
            }
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Restart tournament match failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tournament/force-create-game", operation_id="emergency_create_tournament_game")
async def emergency_create_tournament_game():
    """Emergency tournament game creation endpoint that bypasses normal flow."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            raise HTTPException(status_code=400, detail="Tournament not active")
        
        # Emergency game creation - bypasses phase checks
        await game_manager._start_tournament_game()
        
        # Check if game was created
        new_game_id = game_manager.tournament_manager.get_current_game_id()
        if new_game_id:
            return {
                "success": True,
                "message": f"Emergency tournament game created successfully: {new_game_id}",
                "game_id": new_game_id
            }
        else:
            raise HTTPException(status_code=500, detail="Emergency tournament game creation failed")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Emergency create tournament game failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tournament/health", operation_id="tournament_health_check")
async def tournament_health_check():
    """Tournament health check endpoint that returns detailed diagnostics."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            return {
                "healthy": False,
                "tournament_mode": False,
                "message": "Tournament mode is disabled"
            }
        
        # Get tournament manager
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            return {
                "healthy": False,
                "tournament_mode": True,
                "tournament_active": False,
                "message": "Tournament manager not initialized"
            }
        
        # Get tournament status
        status = game_manager.tournament_manager.get_tournament_status()
        is_active = game_manager.tournament_manager.is_tournament_active()
        
        # Check for specific issues
        issues = []
        phase = status.get('phase')
        game_id = status.get('game_id')
        
        # Check for race condition issue
        if phase == 'game' and not game_id:
            issues.append("Tournament in game phase but no game_id present")
        
        # Check if game exists in game manager
        if game_id and game_id not in game_manager.games:
            issues.append(f"Game {game_id} not found in active games")
        
        # Run health check
        health_check_passed = await game_manager.ensure_tournament_game_exists()
        
        return {
            "healthy": len(issues) == 0 and health_check_passed,
            "tournament_mode": True,
            "tournament_active": is_active,
            "current_phase": phase,
            "current_game_id": game_id,
            "game_exists_in_manager": game_id in game_manager.games if game_id else None,
            "active_games_count": len(game_manager.games),
            "issues": issues,
            "health_check_passed": health_check_passed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        risk_logger.log_error(f"Tournament health check failed: {e}")
        return {
            "healthy": False,
            "tournament_mode": tournament_mode,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/tournament/restart", operation_id="restart_tournament")
async def restart_tournament():
    """Restart tournament with proper cleanup of previous games."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        if not tournament_mode:
            raise HTTPException(status_code=400, detail="Tournament mode is disabled")
        
        # Initialize tournament manager if not exists
        if not hasattr(game_manager, 'tournament_manager') or not game_manager.tournament_manager:
            # Import tournament manager
            from tournament.tournament_manager import TournamentManager
            
            # Create tournament config from environment
            config = {
                'db_path': os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db'),
                'submit_duration': int(os.getenv('TOURNAMENT_SUBMIT_PHASE_DURATION', '900')),
                'voting_duration': int(os.getenv('TOURNAMENT_VOTING_PHASE_DURATION', '900')),
                'game_duration': int(os.getenv('TOURNAMENT_GAME_PHASE_DURATION', '3600')),
                'end_screen_duration': int(os.getenv('TOURNAMENT_END_SCREEN_DURATION', '300')),
                'auto_restart': False,  # Manual restart mode
                'max_submissions': int(os.getenv('TOURNAMENT_MAX_SUBMISSIONS', '20')),
                'selected_characters': int(os.getenv('TOURNAMENT_SELECTED_CHARACTERS', '4'))
            }
            
            game_manager.tournament_manager = TournamentManager(config)
        
        # Clean up any existing tournament games
        await game_manager.cleanup_tournament_games()
        
        # Start new tournament
        tournament_id = await game_manager.tournament_manager.start_tournament()
        
        return {
            "success": True,
            "tournament_id": tournament_id,
            "message": "Tournament restarted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Restart tournament failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin Authentication Helper
def verify_admin_password(password: str) -> bool:
    """Verify admin password."""
    admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
    return password == admin_password

# Admin API Endpoints
@app.post("/api/admin/auth", operation_id="admin_authenticate")
async def admin_authenticate(request: dict):
    """Authenticate admin user."""
    try:
        password = request.get('password', '')
        
        if verify_admin_password(password):
            return {
                "success": True,
                "message": "Authentication successful"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid password")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Character Management Endpoints
@app.get("/api/admin/characters", operation_id="admin_get_characters")
async def admin_get_characters(password: str):
    """Get all characters for admin management."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get character manager
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        characters = char_manager.get_all_characters()
        
        return {
            "characters": [char.to_dict() for char in characters],
            "total": len(characters)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin get characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/characters/{character_id}", operation_id="admin_update_character")
async def admin_update_character(character_id: int, request: dict):
    """Update a character."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get character manager
        from tournament.character_manager import CharacterManager, TournamentCharacter
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        # Create updated character
        character = TournamentCharacter(
            id=character_id,
            name=request.get('name', ''),
            model_name=request.get('model_name', ''),
            temperature=float(request.get('temperature', 0.7)),
            personality=request.get('personality', ''),
            custom_instructions=request.get('custom_instructions', ''),
            created_at=datetime.now(),
            submitted_by=request.get('submitted_by', 'admin')
        )
        
        success, message = char_manager.update_character(character_id, character)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin update character failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/characters/{character_id}", operation_id="admin_delete_character")
async def admin_delete_character(character_id: int, password: str):
    """Delete a character."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get character manager
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        success, message = char_manager.delete_character(character_id)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin delete character failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/characters/bulk-delete", operation_id="admin_bulk_delete_characters")
async def admin_bulk_delete_characters(request: dict):
    """Delete multiple characters."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        character_ids = request.get('character_ids', [])
        if not character_ids:
            raise HTTPException(status_code=400, detail="No character IDs provided")
        
        # Get character manager
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        success, message, count = char_manager.delete_characters_bulk(character_ids)
        
        if success:
            return {
                "success": True,
                "message": message,
                "deleted_count": count
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin bulk delete characters failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/characters/reset-stats", operation_id="admin_reset_character_stats")
async def admin_reset_character_stats(request: dict):
    """Reset character statistics."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        character_id = request.get('character_id')  # None means reset all
        
        # Get character manager
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        success, message = char_manager.reset_character_stats(character_id)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin reset character stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Management Endpoints
@app.get("/api/admin/models", operation_id="admin_get_models")
async def admin_get_models(password: str):
    """Get all models for admin management."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        models = model_manager.get_all_models()
        usage_stats = model_manager.get_model_usage_stats()
        
        # Combine models with their usage stats
        models_with_stats = []
        for model in models:
            model_dict = model.to_dict()
            # Find matching usage stats
            for stat in usage_stats:
                if stat['id'] == model.id:
                    model_dict.update(stat)
                    break
            models_with_stats.append(model_dict)
        
        return {
            "models": models_with_stats,
            "total": len(models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin get models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/models", operation_id="admin_add_model")
async def admin_add_model(request: dict):
    """Add a new model."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager, TournamentModel
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        # Create new model
        model = TournamentModel(
            id=None,
            name=request.get('name', ''),
            display_name=request.get('display_name', ''),
            default_temperature=float(request.get('default_temperature', 0.7)),
            description=request.get('description', ''),
            is_active=request.get('is_active', True),
            created_at=datetime.now()
        )
        
        success, message = model_manager.add_model(model)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin add model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/models/{model_id}", operation_id="admin_update_model")
async def admin_update_model(model_id: int, request: dict):
    """Update a model."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager, TournamentModel
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        # Create updated model
        model = TournamentModel(
            id=model_id,
            name=request.get('name', ''),
            display_name=request.get('display_name', ''),
            default_temperature=float(request.get('default_temperature', 0.7)),
            description=request.get('description', ''),
            is_active=request.get('is_active', True),
            created_at=datetime.now()
        )
        
        success, message = model_manager.update_model(model_id, model)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin update model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/models/{model_id}", operation_id="admin_delete_model")
async def admin_delete_model(model_id: int, password: str):
    """Delete a model."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        success, message = model_manager.delete_model(model_id)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin delete model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/models/{model_id}/toggle", operation_id="admin_toggle_model")
async def admin_toggle_model(model_id: int, request: dict):
    """Toggle model active status."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        success, message = model_manager.toggle_model_status(model_id)
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin toggle model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/models/sync", operation_id="admin_sync_models")
async def admin_sync_models(request: dict):
    """Sync models with environment variable."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get model manager
        from tournament.model_manager import ModelManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        model_manager = ModelManager(db_path)
        
        success, message, count = model_manager.sync_with_environment()
        
        if success:
            return {
                "success": True,
                "message": message,
                "models_updated": count
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin sync models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Tournament Control Endpoints
@app.post("/api/admin/tournament/clear-votes", operation_id="admin_clear_votes")
async def admin_clear_votes(request: dict):
    """Clear all tournament votes."""
    try:
        password = request.get('password', '')
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Get character manager
        from tournament.character_manager import CharacterManager
        db_path = os.getenv('TOURNAMENT_DB_PATH', './data/tournament.db')
        char_manager = CharacterManager(db_path)
        
        # Clear votes for current tournament (using a dummy tournament ID)
        tournament_id = request.get('tournament_id', 'current')
        success = char_manager.clear_votes_for_tournament(tournament_id)
        
        if success:
            return {
                "success": True,
                "message": "Votes cleared successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to clear votes")
            
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin clear votes failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Game Management Endpoints
@app.get("/api/admin/games", operation_id="admin_get_games")
async def admin_get_games(password: str):
    """Get all active games for admin management."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        games = game_manager.list_active_games()
        
        # Enhance games with additional admin info
        enhanced_games = []
        for game in games:
            enhanced_game = game.copy()
            
            # Determine if this is a tournament game
            is_tournament = False
            tournament_phase = None
            
            if hasattr(game_manager, 'tournament_manager') and game_manager.tournament_manager:
                current_tournament_game_id = game_manager.tournament_manager.get_current_game_id()
                if game['game_id'] == current_tournament_game_id:
                    is_tournament = True
                    tournament_status = game_manager.tournament_manager.get_tournament_status()
                    tournament_phase = tournament_status.get('phase', 'unknown')
            
            enhanced_game['is_tournament'] = is_tournament
            enhanced_game['tournament_phase'] = tournament_phase
            enhanced_game['player_count'] = len(game['players'])
            enhanced_game['active_player'] = next((p['name'] for p in game['players'] if not p.get('is_eliminated', False)), 'None')
            
            enhanced_games.append(enhanced_game)
        
        return {
            "games": enhanced_games,
            "total": len(enhanced_games)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin get games failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/games/{game_id}", operation_id="admin_kill_game")
async def admin_kill_game(game_id: str, password: str):
    """Kill/terminate a specific game."""
    try:
        if not verify_admin_password(password):
            raise HTTPException(status_code=401, detail="Invalid admin password")
        
        # Check if game exists
        game = game_manager.get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Check if this is a tournament game and handle accordingly
        is_tournament_game = False
        if hasattr(game_manager, 'tournament_manager') and game_manager.tournament_manager:
            current_tournament_game_id = game_manager.tournament_manager.get_current_game_id()
            if game_id == current_tournament_game_id:
                is_tournament_game = True
                
                # Clear tournament game reference
                game_manager.tournament_manager.set_game_id(None)
                
                # If tournament was in game phase, it might need to be advanced or reset
                tournament_status = game_manager.tournament_manager.get_tournament_status()
                current_phase = tournament_status.get('phase')
                
                risk_logger.log_info(f"Killed tournament game {game_id} during {current_phase} phase")
        
        # Remove the game from active games
        if game_id in game_manager.games:
            del game_manager.games[game_id]
        
        # Track the game termination
        try:
            tracker = get_action_tracker()
            if tracker:
                tracker.track_game_end(
                    game_id=game_id,
                    completion_reason="admin_terminated",
                    winner_player_id=None,
                    winner_player_name=None
                )
        except Exception as e:
            risk_logger.log_error(f"Failed to track game termination in action tracker: {e}")
        
        message = f"Game {game_id} terminated successfully"
        if is_tournament_game:
            message += " (was tournament game)"
        
        return {
            "success": True,
            "message": message,
            "was_tournament": is_tournament_game
        }
        
    except HTTPException:
        raise
    except Exception as e:
        risk_logger.log_error(f"Admin kill game failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Static file serving endpoints
@app.get("/", operation_id="serve_game_setup")
async def serve_game_setup():
    """Serve the game setup page."""
    try:
        # Check if tournament mode is enabled
        tournament_mode = os.getenv('TOURNAMENT_MODE', 'false').lower() == 'true'
        
        if tournament_mode:
            # In tournament mode, serve tournament interface
            tournament_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tournament_interface.html")
            if os.path.exists(tournament_file):
                return FileResponse(tournament_file, media_type="text/html")
            else:
                raise HTTPException(status_code=404, detail="Tournament interface not found")
        
        # Look for game_setup.html in the project root
        setup_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "game_setup.html")
        if os.path.exists(setup_file):
            return FileResponse(setup_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Game setup page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving game setup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/setup", operation_id="serve_game_setup_alt")
async def serve_game_setup_alt():
    """Alternative route for game setup page."""
    return await serve_game_setup()

@app.get("/dashboard", operation_id="serve_dashboard")
async def serve_dashboard():
    """Serve the game dashboard page."""
    try:
        # Look for risk_dashboard.html in the project root
        dashboard_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "risk_dashboard.html")
        if os.path.exists(dashboard_file):
            return FileResponse(dashboard_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Dashboard page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk_dashboard.html", operation_id="serve_dashboard_direct")
async def serve_dashboard_direct():
    """Direct route for dashboard HTML file."""
    return await serve_dashboard()

@app.get("/risk_dashboard_briefing.html", operation_id="serve_briefing_dashboard_direct")
async def serve_briefing_dashboard_direct():
    """Direct route for briefing dashboard HTML file."""
    try:
        # Look for risk_dashboard_briefing.html in the project root
        briefing_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "risk_dashboard_briefing.html")
        if os.path.exists(briefing_file):
            return FileResponse(briefing_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Briefing dashboard page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving briefing dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game_setup.html", operation_id="serve_setup_direct")
async def serve_setup_direct():
    """Direct route for setup HTML file."""
    return await serve_game_setup()

@app.get("/tournament_leaderboard.html", operation_id="serve_tournament_leaderboard_direct")
async def serve_tournament_leaderboard_direct():
    """Direct route for tournament leaderboard HTML file."""
    try:
        leaderboard_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tournament_leaderboard.html")
        if os.path.exists(leaderboard_file):
            return FileResponse(leaderboard_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Tournament leaderboard page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving tournament leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tournament_interface.html", operation_id="serve_tournament_interface_direct")
async def serve_tournament_interface_direct():
    """Direct route for tournament interface HTML file."""
    try:
        tournament_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tournament_interface.html")
        if os.path.exists(tournament_file):
            return FileResponse(tournament_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Tournament interface page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving tournament interface: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/character_library.html", operation_id="serve_character_library_direct")
async def serve_character_library_direct():
    """Direct route for character library HTML file."""
    try:
        character_library_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "character_library.html")
        if os.path.exists(character_library_file):
            return FileResponse(character_library_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Character library page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving character library: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin", operation_id="serve_admin_panel")
async def serve_admin_panel():
    """Serve the admin panel page."""
    try:
        admin_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "admin.html")
        if os.path.exists(admin_file):
            return FileResponse(admin_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Admin panel not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving admin panel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin.html", operation_id="serve_admin_panel_direct")
async def serve_admin_panel_direct():
    """Direct route for admin panel HTML file."""
    return await serve_admin_panel()

@app.get("/about_this.html", operation_id="serve_about_this_direct")
async def serve_about_this_direct():
    """Direct route for about this HTML file."""
    try:
        about_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "about_this.html")
        if os.path.exists(about_file):
            return FileResponse(about_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="About page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving about page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/support_me.html", operation_id="serve_support_me_direct")
async def serve_support_me_direct():
    """Direct route for support me HTML file."""
    try:
        support_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "support_me.html")
        if os.path.exists(support_file):
            return FileResponse(support_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Support page not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving support page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embed_characters.html", operation_id="serve_embed_characters_direct")
async def serve_embed_characters_direct():
    """Direct route for embeddable character leaderboard HTML file."""
    try:
        embed_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embed_characters.html")
        if os.path.exists(embed_file):
            return FileResponse(embed_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Embeddable character leaderboard not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving embeddable character leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embed_models.html", operation_id="serve_embed_models_direct")
async def serve_embed_models_direct():
    """Direct route for embeddable model leaderboard HTML file."""
    try:
        embed_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embed_models.html")
        if os.path.exists(embed_file):
            return FileResponse(embed_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Embeddable model leaderboard not found")
    except Exception as e:
        risk_logger.log_error(f"Error serving embeddable model leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

mcp.setup_server()

@app.on_event("startup")
async def startup_event():
    """Handle server startup tasks."""
    try:
        # Start tournament if needed
        await game_manager.start_tournament_if_needed()
        risk_logger.log_info("Server startup completed")
    except Exception as e:
        risk_logger.log_error(f"Error during server startup: {e}")

def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Risk Game HTTP API Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Override with environment variables if set
    port = int(os.getenv('RISK_API_PORT', args.port))
    host = os.getenv('RISK_API_HOST', args.host)
    
    risk_logger.log_info(f"Starting Risk Game API server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
