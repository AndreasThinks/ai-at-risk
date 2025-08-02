# -*- coding: utf-8 -*-
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

class RiskLogger:
    """Enhanced logger for the Risk MCP server with colored output and activity tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        self.activity_feed = deque(maxlen=10)  # Keep last 10 activities
        self.game_stats = {
            'games_created': 0,
            'players_joined': 0,
            'battles_fought': 0,
            'messages_sent': 0,
            'cards_traded': 0
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging."""
        # Create logger
        self.logger = logging.getLogger('risk_mcp')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler with colors (stderr to not interfere with MCP stdio)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logs
        try:
            file_handler = logging.FileHandler('risk_server.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            pass  # Continue without file logging if not possible
    
    def display_startup_banner(self):
        """Display the startup banner."""
        banner = f"""
{Colors.BRIGHT_CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                     🎲 RISK MCP SERVER 🎲                     ║
║                                                              ║
║            Text-based Risk Board Game via MCP                ║
║                                                              ║
║  🌍 42 Territories  🎯 18 MCP Tools  👥 2-6 Players           ║
║  ⚔️  Combat System  🃏 Card Trading  📝 Messaging             ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
        print(banner, file=sys.stderr)
        
        # Server info
        print(f"{Colors.BRIGHT_GREEN}🚀 Starting Risk MCP Server...{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.CYAN}📡 Protocol: Model Context Protocol (MCP){Colors.RESET}", file=sys.stderr)
        print(f"{Colors.CYAN}🔧 Transport: STDIO{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.CYAN}⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}", file=sys.stderr)
        print(file=sys.stderr)
    
    def display_ready_message(self, tool_count: int):
        """Display server ready message."""
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}✅ SERVER READY!{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.GREEN}🛠️  {tool_count} MCP tools registered and available{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.GREEN}🎮 Ready to host Risk games!{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.DIM}{'='*60}{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.BRIGHT_YELLOW}📊 ACTIVITY MONITOR{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.DIM}{'='*60}{Colors.RESET}", file=sys.stderr)
        print(file=sys.stderr)
    
    def log_tool_call(self, tool_name: str, args: Dict[str, Any], success: bool = True):
        """Log MCP tool calls."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if success:
            color = Colors.BRIGHT_BLUE
            status = "✓"
        else:
            color = Colors.BRIGHT_RED
            status = "✗"
        
        # Extract relevant info for display
        display_info = ""
        if tool_name == "create_game" and "num_players" in args:
            display_info = f"({args['num_players']} players)"
        elif tool_name == "join_game" and "player_name" in args:
            display_info = f"(player: {args['player_name']})"
        elif "game_id" in args:
            game_id = args["game_id"][:8] if len(args["game_id"]) > 8 else args["game_id"]
            display_info = f"(game: {game_id})"
        
        message = f"{color}{status} [{timestamp}] {tool_name}{display_info}{Colors.RESET}"
        print(message, file=sys.stderr)
        
        # Add to activity feed
        self.activity_feed.append({
            'timestamp': timestamp,
            'tool': tool_name,
            'args': args,
            'success': success
        })
        
        # Update stats
        if success:
            if tool_name == "create_game":
                self.game_stats['games_created'] += 1
            elif tool_name == "join_game":
                self.game_stats['players_joined'] += 1
            elif tool_name == "attack_territory":
                self.game_stats['battles_fought'] += 1
            elif tool_name == "send_message":
                self.game_stats['messages_sent'] += 1
            elif tool_name == "trade_cards":
                self.game_stats['cards_traded'] += 1
    
    def log_game_event(self, event_type: str, message: str, game_id: Optional[str] = None):
        """Log significant game events."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Choose color based on event type
        color_map = {
            'game_created': Colors.BRIGHT_GREEN,
            'player_joined': Colors.BRIGHT_CYAN,
            'battle': Colors.BRIGHT_RED,
            'territory_conquered': Colors.BRIGHT_YELLOW,
            'player_eliminated': Colors.BRIGHT_MAGENTA,
            'game_won': Colors.BRIGHT_GREEN,
            'message_sent': Colors.BRIGHT_BLUE,
            'card_traded': Colors.YELLOW,
            'turn_ended': Colors.CYAN
        }
        
        color = color_map.get(event_type, Colors.WHITE)
        icon_map = {
            'game_created': '🎮',
            'player_joined': '👥',
            'battle': '⚔️',
            'territory_conquered': '🏆',
            'player_eliminated': '💀',
            'game_won': '🎉',
            'message_sent': '💬',
            'card_traded': '🃏',
            'turn_ended': '🔄'
        }
        
        icon = icon_map.get(event_type, '📢')
        game_info = f" [{game_id[:8]}]" if game_id else ""
        
        log_message = f"{color}{icon} [{timestamp}]{game_info} {message}{Colors.RESET}"
        print(log_message, file=sys.stderr)
        
        # Also log to file
        self.logger.info(f"{event_type.upper()}: {message} (game: {game_id})")
    
    def log_diplomatic_message(self, sender: str, recipient: str, message: str, game_id: str):
        """Log diplomatic messages with full content display."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Main message log
        main_message = f"💬 [{timestamp}] [{game_id[:8]}] DIPLOMATIC MESSAGE: {sender} → {recipient}"
        print(main_message, file=sys.stderr)
        
        # Message content with indentation and styling
        content_line = f"{Colors.DIM}   📝 Message: \"{message}\"{Colors.RESET}"
        print(content_line, file=sys.stderr)
        
        # Also log to file with full details
        self.logger.info(f"DIPLOMATIC_MESSAGE: {sender} → {recipient}: \"{message}\" (game: {game_id})")
        
        # Update stats
        self.game_stats['messages_sent'] += 1
    
    def log_combat_result(self, attacker: str, defender: str, from_territory: str, 
                         to_territory: str, result: Any, game_id: str):
        """Log combat results with detailed info."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        attacker_dice = ', '.join(map(str, result.attacker_dice))
        defender_dice = ', '.join(map(str, result.defender_dice))
        
        if result.territory_conquered:
            color = Colors.BRIGHT_YELLOW
            outcome = f"🎉 CONQUERED! {to_territory} taken by {attacker}"
        else:
            color = Colors.BRIGHT_RED  
            outcome = f"🛡️ DEFENDED! {to_territory} holds against {attacker}"
        
        message = f"{color}⚔️ [{timestamp}] BATTLE: {from_territory} → {to_territory}{Colors.RESET}"
        print(message, file=sys.stderr)
        print(f"{Colors.DIM}   Attacker ({attacker}): [{attacker_dice}] Lost: {result.attacker_losses}{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.DIM}   Defender ({defender}): [{defender_dice}] Lost: {result.defender_losses}{Colors.RESET}", file=sys.stderr)
        print(f"{color}   {outcome}{Colors.RESET}", file=sys.stderr)
        
        self.game_stats['battles_fought'] += 1
    
    def log_error(self, error: str, context: str = ""):
        """Log errors with prominent display."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        context_str = f" ({context})" if context else ""
        
        message = f"{Colors.BRIGHT_RED}❌ [{timestamp}] ERROR{context_str}: {error}{Colors.RESET}"
        print(message, file=sys.stderr)
        self.logger.error(f"ERROR{context_str}: {error}")
    
    def display_stats(self):
        """Display current server statistics."""
        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats_display = f"""
{Colors.DIM}{'─'*60}{Colors.RESET}
{Colors.BRIGHT_CYAN}📊 SERVER STATISTICS{Colors.RESET}
{Colors.DIM}{'─'*60}{Colors.RESET}
{Colors.GREEN}⏱️  Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}{Colors.RESET}
{Colors.CYAN}🎮 Games Created: {self.game_stats['games_created']}{Colors.RESET}
{Colors.CYAN}👥 Players Joined: {self.game_stats['players_joined']}{Colors.RESET}
{Colors.RED}⚔️  Battles Fought: {self.game_stats['battles_fought']}{Colors.RESET}
{Colors.BLUE}💬 Messages Sent: {self.game_stats['messages_sent']}{Colors.RESET}
{Colors.YELLOW}🃏 Cards Traded: {self.game_stats['cards_traded']}{Colors.RESET}
{Colors.DIM}{'─'*60}{Colors.RESET}
"""
        print(stats_display, file=sys.stderr)
    
    def log_info(self, message: str):
        """Log general information."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.CYAN}ℹ️  [{timestamp}] {message}{Colors.RESET}", file=sys.stderr)
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warnings."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.BRIGHT_YELLOW}⚠️  [{timestamp}] WARNING: {message}{Colors.RESET}", file=sys.stderr)
        self.logger.warning(message)

# Global logger instance
risk_logger = RiskLogger()
