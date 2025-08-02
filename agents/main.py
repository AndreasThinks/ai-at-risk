#!/usr/bin/env python3
import asyncio
import logging
import signal
import sys
import os
import argparse
import traceback

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from game_runner import GameRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_agents.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

# Create the client with SSE transport configuration
client = MultiServerMCPClient({
    "risk": {
        "transport": "sse",
        "url": "http://localhost:8080/mcp"
    }
})

# Global variable for game runner to enable clean shutdown
game_runner = None

def handle_exit(sig, frame):
    """Handle exit signals to cleanly shut down agents"""
    logger.info("Received exit signal, shutting down...")
    
    if game_runner:
        # Create and run a new event loop for cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(game_runner.stop_game())
        loop.close()
    
    sys.exit(0)

async def main():
    """Main entry point for the Risk AI agents"""
    global game_runner
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Risk AI agents')
    parser.add_argument('--agent-names', nargs='+', default=['Alice', 'Bob'], 
                        help='Names for the agents (default: Alice and Bob)')
    parser.add_argument('--server-url', default='http://localhost:8080/mcp',
                        help='URL of the MCP server (default: http://localhost:8080/mcp)')
    parser.add_argument('--check-interval', type=int, default=15,
                        help='Seconds to wait between turn checks (default: 15)')
    parser.add_argument('--model', default='mistral-nemo:latest',
                        help='Ollama model to use (default: mistral-nemo:latest)')
    
    args = parser.parse_args()
    
    # Configure client with server URL
    client = MultiServerMCPClient({
        "risk": {
            "transport": "sse",
            "url": args.server_url
        }
    })
    
    logger.info("=== Starting Risk AI Agents ===")
    
    try:
        logger.info("1. Connecting to MCP server...")
        tools = await client.get_tools()
        logger.info(f"✅ Connected to MCP server, loaded {len(tools)} tools")
        
        # Create game runner with specified agents
        game_runner = GameRunner(client, args.agent_names)
        
        # Create a new game with the number of agents
        num_players = len(args.agent_names)
        logger.info(f"2. Creating a new game with {num_players} players...")
        success = await game_runner.create_game(num_players)
        if not success:
            logger.error("❌ Failed to create game")
            return
        
        # Initialize agents and have them join the game
        logger.info("3. Initializing agents...")
        success = await game_runner.initialize_agents()
        if not success:
            logger.error("❌ Failed to initialize agents")
            return
        
        # Run the game
        logger.info(f"4. Starting game loop (Agents will check for their turn every {args.check_interval} seconds)...")
        logger.info("   Press Ctrl+C to stop the game")
        await game_runner.run_game()
        
    except asyncio.CancelledError:
        logger.info("Game was cancelled")
    except Exception as e:
        logger.exception(f"Error in main: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
