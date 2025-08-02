## Tournament Flow Diagram

```mermaid
graph TD
    subgraph Tournament Cycle
        A[Start Tournament] --> B{Submit Phase};
        B -- 15 min timer --> C{Voting Phase};
        C -- 15 min timer --> D{Starting Game};
        D -- Game Creation --> E{Game Phase};
        E -- 1 hour timer / Game Over --> F{End Screen};
        F -- 5 min timer --> G{Auto-Restart?};
        G -- Yes --> A;
        G -- No --> H[End of Tournament];
    end

    subgraph User Interactions
        U1[User] -->|Submit Character| B;
        U2[Community] -->|Vote for Characters| C;
        U3[User] -->|View Live Game| E;
        U4[User] -->|View Results| F;
    end

    subgraph System Processes
        D --> P1[Select Top 4 Characters];
        P1 --> P2[Weighted Model Selection];
        P2 --> P3[Create AI Agents];
        P3 --> P4[Start GameRunner];
        E --> P5[Game Ends Naturally];
        P5 --> F;
        F --> P6[Update Character Stats];
    end

    subgraph Database Interactions
        B --> DB1[Write to characters.db];
        C --> DB2[Read/Write to votes.db];
        P1 --> DB3[Read from votes.db];
        P2 --> DB4[Read model_usage_stats];
        P6 --> DB5[Update character_stats.db];
    end

    subgraph Error Handling
        D -- Game Creation Fails --> D_ERR{Continuous Retry};
        D_ERR -- Exponential Backoff --> D;
        E -- Token Exhaustion --> E_ERR{Token Exhausted Phase};
        E_ERR -- Admin Restart --> D;
    end

    style A fill:#cde4ff,stroke:#333,stroke-width:2px
    style H fill:#ffcdd2,stroke:#333,stroke-width:2px
    style U1 fill:#fff2cc,stroke:#333,stroke-width:1px
    style U2 fill:#fff2cc,stroke:#333,stroke-width:1px
    style U3 fill:#fff2cc,stroke:#333,stroke-width:1px
    style U4 fill:#fff2cc,stroke:#333,stroke-width:1px
    style DB1 fill:#e1d5e7,stroke:#333,stroke-width:1px
    style DB2 fill:#e1d5e7,stroke:#333,stroke-width:1px
    style DB3 fill:#e1d5e7,stroke:#333,stroke-width:1px
    style DB4 fill:#e1d5e7,stroke:#333,stroke-width:1px
    style DB5 fill:#e1d5e7,stroke:#333,stroke-width:1px
    style D_ERR fill:#ffeb3b,stroke:#333,stroke-width:2px
    style E_ERR fill:#ffeb3b,stroke:#333,stroke-width:2px
```

## Complete Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend & User Interaction"
        UI[Browser Interface]
        UI -- HTTP/SSE --> API[FastAPI Server]
        UI -- "HTML/CSS/JS" --> Dashboard[Live Dashboard]
        UI -- "HTML/CSS/JS" --> TournamentUI[Tournament Interface]
    end

    subgraph "Backend: FastAPI & UV"
        API -- "/api" --> RestEndpoints[REST API]
        API -- "/mcp" --> MCPServer[MCP Server]
        API -- Serves --> Dashboard
        API -- Serves --> TournamentUI
    end

    subgraph "Core Logic"
        GameManager[Game Manager]
        TournamentManager[Tournament Manager]
        GameEngine[Game Engine]
        AIAgents[AI Agents]
    end

    subgraph "AI & LLM Layer (LangChain)"
        AIAgents -- "Uses" --> PlayerConfig[Player Config]
        AIAgents -- "Uses" --> Memory[Memory Management]
        AIAgents -- "Uses" --> LLM[LLM Integration]
        LLM -- "API Calls" --> ExternalLLMs[OpenAI, Claude, etc.]
    end

    subgraph "Data Persistence (SQLite)"
        DB1[Game State DB]
        DB2[Action Tracker DB]
        DB3[Tournament DB]
        DB4[Context Summaries DB]
    end

    subgraph "Deployment (Railway)"
        Railway[Railway.app]
        Railway -- "Hosts" --> API
    end

    %% Connections
    RestEndpoints -- "Manages" --> GameManager
    MCPServer -- "Provides Tools" --> AIAgents
    GameManager -- "Creates/Manages" --> GameEngine
    GameManager -- "Interacts with" --> TournamentManager
    TournamentManager -- "Orchestrates" --> AIAgents
    TournamentManager -- "Uses" --> CharacterManager[Character Manager]
    TournamentManager -- "Uses" --> ModelManager[Model Manager]

    GameEngine -- "Persists State" --> DB1
    AIAgents -- "Log Actions" --> DB2
    TournamentManager -- "Manages Data" --> DB3
    Memory -- "Caches Summaries" --> DB4

    AIAgents -- "Are" --> RiskAgent[RiskAgent]
    RiskAgent -- "Controlled by" --> GameRunner[GameRunner]

    classDef backend fill:#cde4ff,stroke:#333,stroke-width:2px;
    classDef frontend fill:#fff2cc,stroke:#333,stroke-width:2px;
    classDef logic fill:#d5e8d4,stroke:#333,stroke-width:2px;
    classDef data fill:#e1d5e7,stroke:#333,stroke-width:2px;
    classDef deployment fill:#f8cecc,stroke:#333,stroke-width:2px;
    classDef ai fill:#dae8fc,stroke:#333,stroke-width:2px;

    class UI,Dashboard,TournamentUI frontend;
    class API,RestEndpoints,MCPServer backend;
    class GameManager,TournamentManager,GameEngine,AIAgents,CharacterManager,ModelManager,RiskAgent,GameRunner logic;
    class PlayerConfig,Memory,LLM,ExternalLLMs ai;
    class DB1,DB2,DB3,DB4 data;
    class Railway deployment;
