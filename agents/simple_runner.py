#!/usr/bin/env python3
import asyncio
import argparse
import logging
import signal
import sys
import os
import json
import re
import colorama
from colorama import Fore, Back, Style
from datetime import datetime
from pathlib import Path

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Configure custom logging handler
class PrettyConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        if record.levelname == 'INFO':
            msg = f"{Fore.GREEN}{msg}{Style.RESET_ALL}"
        elif record.levelname == 'WARNING':
            msg = f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"
        elif record.levelname == 'ERROR':
            msg = f"{Fore.RED}{msg}{Style.RESET_ALL}"
        elif record.levelname == 'CRITICAL':
            msg = f"{Fore.RED}{Back.WHITE}{msg}{Style.RESET_ALL}"
        
        print(msg)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_agents.log")
    ]
)

# Add our pretty console handler
logger = logging.getLogger("simple_runner")
console_handler = PrettyConsoleHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Set up a specific logger for the pretty display
display_logger = logging.getLogger("display")
display_logger.setLevel(logging.INFO)
display_logger.addHandler(logging.FileHandler("display.log"))

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_mcp_adapters.client import MultiServerMCPClient
from agents.game_runner import GameRunner

# Pretty display functions
def display_header(title):
    """Display a pretty header with a title."""
    terminal_width = 80  # Default width
    try:
        # Try to get actual terminal width
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except:
        pass
    
    print("\n" + "=" * terminal_width)
    print(f"{Fore.YELLOW}{title.center(terminal_width)}{Style.RESET_ALL}")
    print("=" * terminal_width + "\n")

def display_section(title, content, color=Fore.CYAN):
    """Display a section with a colored title and content."""
    print(f"\n{color}{'='*40} {title} {'='*40}{Style.RESET_ALL}")
    print(content)
    print(f"{color}{'='*90}{Style.RESET_ALL}\n")

def display_complete_response(agent_name, response, duration, debug_mode=False):
    """Display the complete agent response with all messages and tool calls."""
    print(f"\n{Fore.GREEN}{'*'*30} COMPLETE RESPONSE FROM {agent_name.upper()} {'*'*30}{Style.RESET_ALL}")
    
    if not response or "messages" not in response:
        print(f"{Fore.RED}No response data available{Style.RESET_ALL}")
        return
    
    messages = response["messages"]
    
    for i, message in enumerate(messages):
        print(f"\n{Fore.CYAN}--- Message {i+1} ---{Style.RESET_ALL}")
        
        # Show message type
        msg_type = getattr(message, 'type', 'unknown')
        print(f"{Fore.YELLOW}Type:{Style.RESET_ALL} {msg_type}")
        
        # Show message content
        content = getattr(message, 'content', str(message))
        if content:
            print(f"{Fore.YELLOW}Content:{Style.RESET_ALL}")
            if debug_mode:
                # In debug mode, show the full content
                print(content)
            else:
                # In normal mode, truncate long content
                if len(str(content)) > 500:
                    print(str(content)[:500] + "...")
                else:
                    print(content)
        
        # Show tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"{Fore.YELLOW}Tool Calls:{Style.RESET_ALL}")
            for j, tool_call in enumerate(message.tool_calls):
                print(f"  {j+1}. {tool_call.get('name', 'unknown')}")
                if debug_mode:
                    print(f"     Args: {tool_call.get('args', {})}")
                    print(f"     ID: {tool_call.get('id', 'unknown')}")
        
        # Show additional attributes in debug mode
        if debug_mode:
            attrs = [attr for attr in dir(message) if not attr.startswith('_')]
            other_attrs = [attr for attr in attrs if attr not in ['type', 'content', 'tool_calls']]
            if other_attrs:
                print(f"{Fore.YELLOW}Other Attributes:{Style.RESET_ALL}")
                for attr in other_attrs:
                    try:
                        value = getattr(message, attr)
                        if value and not callable(value):
                            print(f"  {attr}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                    except:
                        pass
    
    print(f"\n{Fore.YELLOW}Decision Time:{Style.RESET_ALL} {duration:.2f} seconds")
    print(f"{Fore.GREEN}{'*'*90}{Style.RESET_ALL}\n")

def extract_tools_from_response(response):
    """Extract tools used from the agent response."""
    tools_used = []
    
    if not response or "messages" not in response:
        return tools_used
    
    for message in response["messages"]:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', 'unknown_tool')
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
    
    return tools_used

def extract_reasoning_from_response(response):
    """Extract reasoning from the agent response."""
    reasoning_parts = []
    
    if not response or "messages" not in response:
        return "No reasoning available"
    
    for message in response["messages"]:
        if hasattr(message, 'type') and message.type == 'ai':
            content = getattr(message, 'content', '')
            if content and not hasattr(message, 'tool_calls'):
                # This is likely reasoning text, not a tool call
                reasoning_parts.append(str(content))
    
    if reasoning_parts:
        return "\n".join(reasoning_parts)
    else:
        return "No explicit reasoning found"

# Hook into RiskAgent to monitor decisions
original_play_turn = None

async def monitored_play_turn(self):
    """Wrapper around play_turn to monitor and display agent decisions."""
    start_time = datetime.now()
    
    # Start the pretty display
    display_header(f"{self.name}'s Turn")
    
    # Store the original response for display
    self._monitoring_response = None
    self._monitoring_duration = 0
    
    # Store original method if not already done
    if not hasattr(self, '_original_ainvoke'):
        self._original_ainvoke = self.agent.ainvoke if self.agent else None
    
    # Wrap the agent invocation to capture the response
    async def capture_response(*args, **kwargs):
        response_start = datetime.now()
        response = await self._original_ainvoke(*args, **kwargs)
        response_end = datetime.now()
        
        # Store for monitoring
        self._monitoring_response = response
        self._monitoring_duration = (response_end - response_start).total_seconds()
        
        return response
    
    # Temporarily replace the agent's ainvoke method
    if self.agent and self._original_ainvoke:
        self.agent.ainvoke = capture_response
    
    try:
        # Call the original play_turn method
        if original_play_turn is None:
            logger.error("Original play_turn method is None - this should not happen")
            # Fallback to direct method
            from agents.risk_agent import RiskAgent
            result = await RiskAgent.play_turn(self)
        else:
            result = await original_play_turn(self)
        
        # Display the complete response if captured
        if hasattr(self, '_monitoring_response') and self._monitoring_response:
            debug_mode = getattr(self, '_debug_mode', False)
            display_complete_response(
                self.name, 
                self._monitoring_response, 
                self._monitoring_duration,
                debug_mode
            )
            
            # Extract and display tools used
            tools_used = extract_tools_from_response(self._monitoring_response)
            reasoning = extract_reasoning_from_response(self._monitoring_response)
            
            if tools_used:
                display_section("TOOLS USED", ", ".join(tools_used), Fore.MAGENTA)
            
            if reasoning and reasoning != "No explicit reasoning found":
                display_section("AGENT REASONING", reasoning[:1000] + ("..." if len(reasoning) > 1000 else ""), Fore.BLUE)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"{Fore.GREEN}Turn completed in {total_duration:.2f} seconds{Style.RESET_ALL}")
        
        return result
        
    finally:
        # Restore the original ainvoke method
        if self.agent and self._original_ainvoke:
            self.agent.ainvoke = self._original_ainvoke

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

def show_interactive_launcher():
    """Show the main interactive launcher menu."""
    display_header("RISK AI GAME LAUNCHER")
    
    print(f"{Fore.CYAN}Welcome to the Risk AI Game System!{Style.RESET_ALL}")
    print("Choose how you'd like to set up your game:\n")
    
    # Import here to avoid circular imports
    from agents.player_config import PlayerConfigManager
    config_manager = PlayerConfigManager()
    
    # Get available saved configurations
    saved_configs = config_manager.list_saved_configurations()
    
    print(f"{Fore.YELLOW}📋 CONFIGURATION OPTIONS:{Style.RESET_ALL}")
    print("1. 🎮 Interactive Configuration (create new setup)")
    print("2. 📁 Load Saved Configuration")
    
    if saved_configs:
        print(f"   Available configs: {', '.join(saved_configs)}")
    else:
        print("   (No saved configurations found)")
    
    print("3. 🚀 Quick Presets")
    print("   - gpt4_tournament: GPT-4 models with different strategies")
    print("   - mixed_models: Different model types for variety")
    print("   - beginner_friendly: Simple setup for testing")
    print("4. ⚙️  Advanced Options (use command-line arguments)")
    print("5. ❌ Exit")
    
    while True:
        try:
            choice = input(f"\n{Fore.GREEN}Choose option (1-5): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                return 'interactive'
            elif choice == '2':
                if not saved_configs:
                    print(f"{Fore.RED}❌ No saved configurations found{Style.RESET_ALL}")
                    continue
                return show_saved_configs_menu(saved_configs)
            elif choice == '3':
                return show_presets_menu()
            elif choice == '4':
                print(f"\n{Fore.YELLOW}💡 Advanced Options:{Style.RESET_ALL}")
                print("Use command-line arguments for advanced configuration:")
                print("  --agent-names Alice Bob Charlie")
                print("  --model-type openai")
                print("  --debug")
                print("  --help (for all options)")
                print("\nRestart with arguments, or choose another option above.")
                continue
            elif choice == '5':
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                sys.exit(0)
            else:
                print(f"{Fore.RED}❌ Please choose a number between 1 and 5{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            sys.exit(0)

def show_saved_configs_menu(saved_configs):
    """Show menu for selecting saved configurations."""
    print(f"\n{Fore.CYAN}📁 SAVED CONFIGURATIONS:{Style.RESET_ALL}")
    print("=" * 40)
    
    from agents.player_config import PlayerConfigManager
    config_manager = PlayerConfigManager()
    
    # Show available configs with preview
    for i, config_name in enumerate(saved_configs, 1):
        print(f"\n{i}. {config_name}")
        
        # Try to load and preview the config
        try:
            players = config_manager.load_configuration(config_name)
            if players:
                print(f"   👥 {len(players)} players: {', '.join(p.name for p in players)}")
                print(f"   🤖 Models: {', '.join(set(p.model_name for p in players))}")
            else:
                print(f"   ❌ Error loading configuration")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{len(saved_configs) + 1}. 🔙 Back to main menu")
    
    while True:
        try:
            choice = input(f"\n{Fore.GREEN}Choose configuration (1-{len(saved_configs) + 1}): {Style.RESET_ALL}").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(saved_configs):
                selected_config = saved_configs[choice_num - 1]
                
                # Load and preview the configuration
                players = config_manager.load_configuration(selected_config)
                if not players:
                    print(f"{Fore.RED}❌ Failed to load configuration{Style.RESET_ALL}")
                    continue
                
                # Show detailed preview
                print(f"\n{Fore.CYAN}📋 CONFIGURATION PREVIEW:{Style.RESET_ALL}")
                print(f"Config: {selected_config}")
                print(f"Players: {len(players)}")
                for i, player in enumerate(players, 1):
                    print(f"  {i}. {player.name} ({player.model_name}, temp: {player.temperature})")
                    if len(player.custom_instructions) > 50:
                        instructions = player.custom_instructions[:50] + "..."
                    else:
                        instructions = player.custom_instructions
                    print(f"     Instructions: {instructions}")
                
                # Confirm selection
                confirm = input(f"\n{Fore.GREEN}Use this configuration? (y/N): {Style.RESET_ALL}").strip().lower()
                if confirm in ['y', 'yes']:
                    return ('load_config', selected_config)
                else:
                    print("Selection cancelled, choose again:")
                    continue
                    
            elif choice_num == len(saved_configs) + 1:
                return 'interactive'  # Back to main menu
            else:
                print(f"{Fore.RED}❌ Please choose a number between 1 and {len(saved_configs) + 1}{Style.RESET_ALL}")
                
        except ValueError:
            print(f"{Fore.RED}❌ Please enter a valid number{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            sys.exit(0)

def show_presets_menu():
    """Show menu for selecting preset configurations."""
    print(f"\n{Fore.CYAN}🚀 QUICK PRESETS:{Style.RESET_ALL}")
    print("=" * 30)
    
    presets = {
        '1': ('gpt4_tournament', 'GPT-4 Tournament', 'GPT-4 models with aggressive, diplomatic, and balanced strategies'),
        '2': ('mixed_models', 'Mixed Models', 'Different model types for variety and testing'),
        '3': ('beginner_friendly', 'Beginner Friendly', 'Simple setup with GPT-3.5-turbo for testing')
    }
    
    for key, (preset_id, name, description) in presets.items():
        print(f"{key}. {name}")
        print(f"   {description}")
        print()
    
    print("4. 🔙 Back to main menu")
    
    while True:
        try:
            choice = input(f"\n{Fore.GREEN}Choose preset (1-4): {Style.RESET_ALL}").strip()
            
            if choice in presets:
                preset_id, name, description = presets[choice]
                
                # Get number of players
                while True:
                    try:
                        num_players = input(f"Number of players (2-6, default 3): ").strip()
                        if not num_players:
                            num_players = 3
                        else:
                            num_players = int(num_players)
                        
                        if 2 <= num_players <= 6:
                            break
                        else:
                            print(f"{Fore.RED}❌ Number of players must be between 2 and 6{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}❌ Please enter a valid number{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}Creating '{name}' preset with {num_players} players...{Style.RESET_ALL}")
                return ('preset', preset_id, num_players)
                
            elif choice == '4':
                return 'interactive'  # Back to main menu
            else:
                print(f"{Fore.RED}❌ Please choose a number between 1 and 4{Style.RESET_ALL}")
                
        except ValueError:
            print(f"{Fore.RED}❌ Please enter a valid number{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            sys.exit(0)

def detect_no_args_mode():
    """Detect if we should start in interactive launcher mode."""
    # Check if we have any meaningful arguments
    # If only the script name is provided, or only basic flags, start interactive mode
    
    # Get command line arguments
    args = sys.argv[1:]  # Exclude script name
    
    # If no arguments at all, definitely interactive mode
    if not args:
        return True
    
    # Check for help flags - let them through
    help_flags = ['--help', '-h', '--version']
    if any(flag in args for flag in help_flags):
        return False
    
    # Check for any configuration-related flags
    config_flags = ['--interactive-config', '--load-config', '--preset']
    if any(flag in args for flag in config_flags):
        return False
    
    # Check if only basic flags are provided (debug, no-pretty, etc.)
    basic_flags = ['--debug', '--no-pretty']
    remaining_args = [arg for arg in args if arg not in basic_flags]
    
    # If only basic flags remain, start interactive mode
    if not remaining_args:
        return True
    
    # If we have substantial arguments, don't start interactive mode
    return False

async def main():
    """Main entry point for the Risk AI agents"""
    global game_runner
    global original_play_turn
    
    # Check if we should start in interactive launcher mode
    if detect_no_args_mode():
        # Start interactive launcher
        try:
            launcher_result = show_interactive_launcher()
            
            # Handle the different launcher results
            if launcher_result == 'interactive':
                # Set up args for interactive configuration
                sys.argv = [sys.argv[0], '--interactive-config']
            elif isinstance(launcher_result, tuple) and launcher_result[0] == 'load_config':
                # Set up args for loading configuration
                config_filename = launcher_result[1]
                sys.argv = [sys.argv[0], '--load-config', config_filename]
            elif isinstance(launcher_result, tuple) and launcher_result[0] == 'preset':
                # Set up args for preset configuration
                preset_name = launcher_result[1]
                num_players = launcher_result[2]
                # Generate default names for the preset
                default_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
                agent_names = default_names[:num_players]
                sys.argv = [sys.argv[0], '--preset', preset_name, '--agent-names'] + agent_names
            else:
                # Unknown result, exit
                print(f"{Fore.RED}❌ Unknown launcher result: {launcher_result}{Style.RESET_ALL}")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            sys.exit(0)
        except Exception as e:
            print(f"{Fore.RED}❌ Error in interactive launcher: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Risk AI agents')
    parser.add_argument('--agent-names', nargs='+', default=['Alice', 'Bob', 'Charlie'], 
                        help='Names for the agents (default: Alice, Bob, and Charlie)')
    parser.add_argument('--server-url', default='http://localhost:8080/mcp',
                        help='URL of the MCP server (default: http://localhost:8080/mcp)')
    parser.add_argument('--check-interval', type=int, default=15,
                        help='Seconds to wait between turn checks (default: 15)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging and full message display')
    parser.add_argument('--no-pretty', action='store_true',
                        help='Disable pretty display')
    
    # LLM Configuration Arguments
    parser.add_argument('--model-type', choices=['ollama', 'openai'], default='ollama',
                        help='Type of LLM to use (default: ollama)')
    parser.add_argument('--api-key', 
                        help='API key for OpenAI-compatible models (required for --model-type openai)')
    parser.add_argument('--base-url', 
                        help='Base URL for OpenAI-compatible models (required for --model-type openai)')
    parser.add_argument('--model-name', 
                        help='Model name (default: qwen3:14b for ollama, gpt-3.5-turbo for openai)')
    parser.add_argument('--num-ctx', type=int, default=10240,
                        help='Context size for Ollama models (default: 10240)')
    
    # Player Configuration Arguments
    parser.add_argument('--interactive-config', action='store_true',
                        help='Use interactive player configuration system')
    parser.add_argument('--load-config', 
                        help='Load player configuration from JSON file')
    parser.add_argument('--preset', choices=['gpt4_tournament', 'mixed_models', 'beginner_friendly'],
                        help='Use a predefined player configuration preset')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create LLM configuration from arguments and environment variables
    llm_config = None
    
    # Check for environment variables first
    env_model_type = os.getenv('RISK_MODEL_TYPE', args.model_type)
    env_api_key = os.getenv('RISK_API_KEY', args.api_key)
    env_base_url = os.getenv('RISK_BASE_URL', args.base_url)
    env_model_name = os.getenv('RISK_MODEL_NAME', args.model_name)
    
    # Use environment variables if available, otherwise use command line args
    model_type = env_model_type
    api_key = env_api_key
    base_url = env_base_url
    model_name = env_model_name
    
    if model_type == 'openai':
        # Validate required parameters for OpenAI-compatible models
        if not api_key:
            logger.error("API key is required for OpenAI-compatible models. Use --api-key or set RISK_API_KEY environment variable.")
            return
        if not base_url:
            logger.error("Base URL is required for OpenAI-compatible models. Use --base-url or set RISK_BASE_URL environment variable.")
            return
        
        # Set default model name for OpenAI if not specified
        if not model_name:
            model_name = "gpt-3.5-turbo"
        
        llm_config = {
            "model_type": "openai",
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name
        }
        
        logger.info(f"Using OpenAI-compatible model: {model_name} at {base_url}")
        
    else:  # ollama (default)
        # Set default model name for Ollama if not specified
        if not model_name:
            model_name = "qwen3:14b"
        
        llm_config = {
            "model_type": "ollama",
            "model_name": model_name,
            "num_ctx": args.num_ctx
        }
        
        logger.info(f"Using Ollama model: {model_name}")
    
    # Set up data directory if not already set
    if 'RISK_DATA_DIR' not in os.environ:
        # Use the current working directory's data folder
        current_dir = Path.cwd()
        data_dir = current_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ['RISK_DATA_DIR'] = str(data_dir)
        logger.info(f"Setting RISK_DATA_DIR to {data_dir}")

    # Set up the pretty display
    display_header("RISK GAME AI AGENTS")
    logger.info("Starting Risk AI Agents")
    
    try:
        # Configure client with server URL
        client = MultiServerMCPClient({
            "risk": {
                "transport": "sse",
                "url": args.server_url
            }
        })
        
        # Connect to MCP server
        display_section("CONNECTING TO SERVER", f"Connecting to MCP server at {args.server_url}...", Fore.BLUE)
        tools = await client.get_tools()
        logger.info(f"Connected to MCP server, loaded {len(tools)} tools")
        
        # Display available tools in debug mode
        if args.debug:
            tool_names = [tool.name for tool in tools]
            display_section("AVAILABLE MCP TOOLS", "\n".join(f"- {name}" for name in tool_names), Fore.MAGENTA)
        
        # Handle player configuration
        player_configs = None
        
        if args.interactive_config:
            # Use interactive configuration system
            display_section("INTERACTIVE CONFIGURATION", "Starting interactive player configuration...", Fore.BLUE)
            from agents.player_config import PlayerConfigManager
            config_manager = PlayerConfigManager()
            player_configs = config_manager.configure_players_interactive()
            
            if not player_configs:
                logger.error("Interactive configuration cancelled or failed")
                return
                
        elif args.load_config:
            # Load configuration from file
            display_section("LOADING CONFIGURATION", f"Loading player configuration from {args.load_config}...", Fore.BLUE)
            from agents.player_config import PlayerConfigManager
            config_manager = PlayerConfigManager()
            player_configs = config_manager.load_configuration(args.load_config)
            
            if not player_configs:
                logger.error(f"Failed to load configuration from {args.load_config}")
                return
                
        elif args.preset:
            # Use preset configuration
            display_section("PRESET CONFIGURATION", f"Using preset: {args.preset}...", Fore.BLUE)
            from agents.player_config import PlayerConfigManager
            config_manager = PlayerConfigManager()
            player_configs = config_manager.create_quick_preset(args.preset, len(args.agent_names))
            
            if not player_configs:
                logger.error(f"Failed to create preset: {args.preset}")
                return
        
        # Create game runner with player configurations or legacy mode
        if player_configs:
            game_runner = GameRunner(client, player_configs=player_configs)
            num_players = len(player_configs)
            display_section("PLAYER CONFIGURATION", 
                f"Configured {num_players} players:\n" + 
                "\n".join(f"- {config.name}: {config.model_name} (temp: {config.temperature})" 
                         for config in player_configs), Fore.CYAN)
        else:
            # Legacy mode - use command line arguments
            game_runner = GameRunner(client, agent_names=args.agent_names, llm_config=llm_config)
            num_players = len(args.agent_names)
        display_section("CREATING GAME", f"Creating a new Risk game with {num_players} players...", Fore.BLUE)
        success = await game_runner.create_game(num_players)
        if not success:
            logger.error("Failed to create game")
            return
        
        # Initialize agents and have them join the game
        if player_configs:
            agent_names_display = [config.name for config in player_configs]
        else:
            agent_names_display = args.agent_names
        display_section("INITIALIZING AGENTS", f"Setting up {', '.join(agent_names_display)}...", Fore.BLUE)
        
        # Hook into RiskAgent.play_turn for monitoring if pretty display is enabled
        if not args.no_pretty:
            from agents.risk_agent import RiskAgent
            # Store the original method before replacing it
            original_play_turn = RiskAgent.play_turn
            # Replace with our monitoring version
            RiskAgent.play_turn = monitored_play_turn
            logger.info("Enhanced agent monitoring enabled")
        
        success = await game_runner.initialize_agents()
        if not success:
            logger.error("Failed to initialize agents")
            return
        
        # Set debug mode on agents if enabled
        if args.debug:
            for agent in game_runner.agents:
                agent._debug_mode = True
        
        # Run the game using the new optimized turn-based system
        display_section("STARTING GAME", 
            f"Game is running with optimized turn-based coordination!\n"
            f"Agents will only use LLM when it's their turn (95% efficiency improvement).\n"
            f"Debug mode: {'ON' if args.debug else 'OFF'}\n"
            f"Press Ctrl+C to stop the game.", Fore.GREEN)
        await game_runner.run_game()
        
    except asyncio.CancelledError:
        logger.info("Game was cancelled")
    except Exception as e:
        logger.exception(f"Error in main: {e}")


if __name__ == "__main__":
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
