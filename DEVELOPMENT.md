# Risk MCP Development Guide

This document provides technical details about the Risk MCP Server architecture, features, and development setup.

## 🏗️ Architecture Overview

### Core Components

```
risk_mcp/
├── src/
│   ├── api_server.py           # FastAPI server with MCP integration
│   ├── game/                   # Core game logic
│   │   ├── game_state.py       # Main game state management
│   │   ├── player.py           # Player data and strategies
│   │   ├── territory.py        # Territory and map management
│   │   ├── combat.py           # Combat system and dice rolling
│   │   ├── cards.py            # Risk card system
│   │   └── notepad.py          # Messaging and notes
│   ├── persistence/            # Data persistence layer
│   │   ├── action_tracker.py   # Game actions and decision tracking
│   │   └── game_persistence.py # Game state persistence
│   └── utils/
│       ├── game_manager.py     # Multi-game management
│       └── logger.py           # Structured logging
├── agents/                     # AI agent system
│   ├── risk_agent.py           # Individual AI agent
│   ├── game_runner.py          # Game orchestration
│   └── simple_runner.py        # Agent runner with display
├── data/                       # Persistent data directory
│   ├── risk_actions.db         # SQLite database
│   └── games/                  # JSON game saves
└── main.py                     # Entry point
```

## 🧠 Player Strategy System

### Overview
Each player now has both short-term and long-term strategies that can be updated during gameplay and are tracked in the database.

### Database Schema
```sql
CREATE TABLE player_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('short_term', 'long_term')),
    strategy_content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
);
```

### API Endpoints

#### Get Player Strategies
```http
GET /api/games/{game_id}/player/{player_id}/strategies
```
Returns current strategies and database history.

#### Update Short-term Strategy
```http
PUT /api/games/{game_id}/player/{player_id}/strategies/short-term
Content-Type: application/json

{
    "player_id": "player123",
    "strategy": "Focus on controlling Australia for early continent bonus"
}
```

#### Update Long-term Strategy
```http
PUT /api/games/{game_id}/player/{player_id}/strategies/long-term
Content-Type: application/json

{
    "player_id": "player123", 
    "strategy": "Build up in Asia, then push into Europe for final assault"
}
```

#### Update Both Strategies
```http
PUT /api/games/{game_id}/player/{player_id}/strategies
Content-Type: application/json

{
    "player_id": "player123",
    "short_term_strategy": "Secure North America",
    "long_term_strategy": "Control all of Asia and Europe"
}
```

#### Get Strategy History
```http
GET /api/games/{game_id}/player/{player_id}/strategies/history
```

#### Get All Player Strategies
```http
GET /api/games/{game_id}/strategies
```

### Implementation Details

**Player Class Updates:**
```python
@dataclass
class Player:
    # ... existing fields ...
    short_term_strategy: str = ""
    long_term_strategy: str = ""
    
    def update_short_term_strategy(self, strategy: str) -> None:
        self.short_term_strategy = strategy
    
    def update_long_term_strategy(self, strategy: str) -> None:
        self.long_term_strategy = strategy
```

**ActionTracker Methods:**
- `update_player_strategy()` - Store strategy updates with timestamps
- `get_current_player_strategies()` - Get latest strategies for a player
- `get_player_strategy_history()` - Get full strategy change history
- `get_all_current_strategies()` - Get strategies for all players in a game

## 🤖 Agent Decision Tracking

### Overview
The system now tracks detailed information about AI agent decision-making processes, including context, reasoning, tools used, and timing.

### Database Schema
```sql
CREATE TABLE agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    decision_timestamp TIMESTAMP NOT NULL,
    context_data TEXT NOT NULL,        -- Full JSON context
    formatted_context TEXT NOT NULL,   -- Human-readable context
    agent_prompt TEXT NOT NULL,        -- Complete LLM prompt
    agent_response TEXT,               -- Full LLM response
    agent_reasoning TEXT,              -- Extracted reasoning
    tools_used TEXT,                   -- JSON array of tools
    decision_time_seconds REAL,        -- Time taken
    success BOOLEAN NOT NULL,          -- Success/failure
    error_message TEXT,                -- Error details
    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
);
```

### API Endpoints

#### Get Agent Decisions
```http
GET /api/games/{game_id}/agent-decisions?player_id={player_id}&turn_number={turn}&limit={limit}
```

#### Get Decision Details
```http
GET /api/games/{game_id}/agent-decisions/{decision_id}
```

#### Get Player Agent Decisions
```http
GET /api/games/{game_id}/player/{player_id}/agent-decisions?limit={limit}
```

#### Get Decision Analytics
```http
GET /api/games/{game_id}/agent-decisions/analytics
```

### Implementation Details

**RiskAgent Decision Tracking:**
```python
class RiskAgent:
    async def play_turn(self):
        # Start decision tracking
        decision_id = await self.track_decision_start(context, formatted_context, prompt)
        
        try:
            # Make LLM decision
            response = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            
            # Extract reasoning and tools
            reasoning = self.extract_reasoning(response)
            tools_used = self.extract_tools_used(response)
            
            # Complete tracking
            await self.track_decision_complete(
                decision_id, response, reasoning, tools_used, 
                time_taken, True, None
            )
        except Exception as e:
            # Track failure
            await self.track_decision_complete(
                decision_id, None, None, [], time_taken, False, str(e)
            )
```

**ActionTracker Methods:**
- `track_agent_decision_start()` - Begin tracking a decision
- `track_agent_decision_complete()` - Complete decision tracking
- `get_agent_decisions()` - Query decisions with filtering
- `get_agent_decision_detail()` - Get full decision details
- `get_agent_decision_analytics()` - Get decision analytics

## 🌍 Environment Compatibility

### Smart Data Directory Detection

The system automatically detects the appropriate data directory:

```python
def get_default_data_dir() -> str:
    # Check if RISK_DATA_DIR is explicitly set
    if data_dir := os.getenv('RISK_DATA_DIR'):
        return data_dir
    
    # Check if we're in the project root
    current_dir = Path.cwd()
    if (current_dir / 'src').exists():
        data_dir = str(current_dir / 'data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    # Fall back to data directory relative to this file's location
    src_dir = Path(__file__).parent.parent
    project_root = src_dir.parent
    data_dir = str(project_root / 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
```

### Directory Structure

**Local Environment:**
- Working Directory: `./` (project root)
- Data Directory: `./data`
- Games: `./data/games/`
- Database: `./data/risk_actions.db`

**Fallback Environment:**
- Uses system temporary directory if others fail
- Logs warning about fallback location

## 🔧 Error Handling

### 404 Error Handling
Agent context gathering now properly handles 404 errors for missing turn data:

```python
# In RiskAgent.gather_turn_context()
if response.status_code == 200:
    context["current_turn_actions"] = response.json()
elif response.status_code == 404:
    # Turn not found in database yet - this is normal for new games/turns
    self.logger.info(f"Turn {current_turn} not yet recorded in database (this is normal for new turns)")
    context["current_turn_actions"] = {"actions": []}
else:
    self.logger.warning(f"Failed to get current turn actions: HTTP {response.status_code}")
    context["current_turn_actions"] = {"actions": []}
```

### Database Initialization
Robust error handling for database and directory creation:

```python
try:
    self.data_dir.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError) as e:
    risk_logger.log_error(f"Cannot create data directory {self.data_dir}: {e}")
    # Fall back to a temp directory if we can't create the data directory
    import tempfile
    self.data_dir = Path(tempfile.gettempdir()) / 'risk_data'
    self.data_dir.mkdir(parents=True, exist_ok=True)
    risk_logger.log_warning(f"Falling back to temporary directory: {self.data_dir}")
```

## 🎮 AI Agents System

### RiskAgent Architecture

Each AI agent is a complete autonomous player that:
1. **Monitors** game state for their turn
2. **Gathers** rich context about the game situation
3. **Formats** context into structured prompts
4. **Queries** LLM for strategic decisions
5. **Executes** decisions using MCP tools
6. **Tracks** all decision-making in the database

### Context Gathering

Agents collect comprehensive context:
- Current game status and turn information
- Board state with all territories and armies
- Recent game history and player actions
- Player's current strategies (short and long-term)
- Diplomatic messages from other players
- Available cards and trading opportunities

### Decision Process

```python
async def play_turn(self):
    # 1. Gather rich context
    context = await self.gather_turn_context()
    
    # 2. Format context for LLM
    formatted_context = self.format_turn_context(context)
    
    # 3. Create strategic prompt
    prompt = f"""
    You are {self.name}, playing Risk.
    
    {formatted_context}
    
    Make a strategic decision:
    1. UPDATE STRATEGIES
    2. SEND DIPLOMATIC MESSAGES  
    3. TAKE GAME ACTIONS
    
    Think carefully, then execute using MCP tools.
    """
    
    # 4. Track decision start
    decision_id = await self.track_decision_start(context, formatted_context, prompt)
    
    # 5. Query LLM
    response = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    
    # 6. Track completion
    await self.track_decision_complete(decision_id, response, reasoning, tools_used, time_taken, True)
```

### Agent Display System

The `simple_runner.py` provides a beautiful terminal interface showing:
- Colorful startup banner
- Real-time turn notifications
- Context information displayed to agents
- Agent reasoning and decision processes
- Tools used and timing information
- Color-coded output for different types of events

## 🗄️ Database Schema

### Complete Schema Overview

```sql
-- Game sessions
CREATE TABLE game_sessions (
    game_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    current_player_id TEXT,
    current_turn_number INTEGER NOT NULL DEFAULT 1,
    game_phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    num_players INTEGER NOT NULL
);

-- Player turns
CREATE TABLE player_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    turn_started_at TIMESTAMP NOT NULL,
    turn_ended_at TIMESTAMP,
    actions_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
);

-- Game actions
CREATE TABLE game_actions (
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
);

-- Player strategies
CREATE TABLE player_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('short_term', 'long_term')),
    strategy_content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
);

-- Agent decisions
CREATE TABLE agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    decision_timestamp TIMESTAMP NOT NULL,
    context_data TEXT NOT NULL,
    formatted_context TEXT NOT NULL,
    agent_prompt TEXT NOT NULL,
    agent_response TEXT,
    agent_reasoning TEXT,
    tools_used TEXT,
    decision_time_seconds REAL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    FOREIGN KEY (game_id) REFERENCES game_sessions(game_id)
);
```

## 🚀 Development Setup

### Local Development
```bash
# Install dependencies
uv sync

# Run the server
python main.py

# Or run API server
python src/api_server.py

# Run agents
python agents/simple_runner.py
```

### Testing
```bash
# Test action tracker
python test_action_tracker.py

# Health check
python healthcheck.py
```

## 📊 Monitoring & Analytics

### Game Analytics
- Player strategy evolution tracking
- Agent decision success rates
- Tool usage patterns
- Turn duration analysis
- Combat outcome statistics

### Database Queries

**Get agent performance:**
```sql
SELECT player_name, 
       COUNT(*) as total_decisions,
       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_decisions,
       AVG(decision_time_seconds) as avg_decision_time
FROM agent_decisions 
WHERE game_id = ?
GROUP BY player_name;
```

**Get strategy evolution:**
```sql
SELECT player_name, strategy_type, strategy_content, updated_at
FROM player_strategies 
WHERE game_id = ?
ORDER BY updated_at DESC;
```

**Get turn performance:**
```sql
SELECT turn_number, player_name, 
       turn_started_at, turn_ended_at,
       (julianday(turn_ended_at) - julianday(turn_started_at)) * 24 * 60 as turn_minutes
FROM player_turns
WHERE game_id = ?
ORDER BY turn_number;
```

## 🔌 MCP Integration

### FastAPI-MCP Bridge
The system uses `fastapi-mcp` to automatically expose HTTP API endpoints as MCP tools:

```python
from fastapi_mcp import FastApiMCP

app = FastAPI()
mcp = FastApiMCP(app)
mcp.mount()  # Automatically creates MCP tools from endpoints
```

### Available MCP Tools
All HTTP API endpoints are automatically available as MCP tools with the same parameters and responses.

### MCP Server Endpoint
- **HTTP API**: `http://localhost:8080/api/*`
- **MCP Server**: `http://localhost:8080/mcp`
- **Documentation**: `http://localhost:8080/docs`

## 🛠️ Extending the System

### Adding New API Endpoints
1. Add endpoint to `src/api_server.py`
2. Use appropriate Pydantic models for request/response
3. Add error handling and logging
4. Tool becomes automatically available via MCP

### Adding New Game Features
1. Update core game classes in `src/game/`
2. Add database schema changes to `ActionTracker._init_db()`
3. Add API endpoints for the new feature
4. Update agent context gathering if needed

### Adding New Agent Capabilities
1. Update `RiskAgent.gather_turn_context()` for new data
2. Update `RiskAgent.format_turn_context()` for LLM prompt
3. Add new tool extraction patterns if needed
4. Update decision tracking as required

---

This development guide covers the major architectural components and recent enhancements to the Risk MCP Server. The system is designed to be modular, extensible, and compatible across different deployment environments.
