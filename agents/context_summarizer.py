#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import tiktoken
except ImportError:
    tiktoken = None

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Handle imports for both Docker and local environments
try:
    # Try Docker-style import first (when running from /app directory)
    from persistence.action_tracker import action_tracker
    from utils.logger import risk_logger
except ImportError:
    # Fall back to local-style import (when running from project root)
    from src.persistence.action_tracker import action_tracker
    from src.utils.logger import risk_logger


class ContextSummarizer:
    """
    Intelligent context summarization system for Risk AI agents.
    Uses a separate LLM to condense game context when it becomes too large.
    """
    
    def __init__(self):
        """Initialize the context summarizer with configuration from environment."""
        self.logger = logging.getLogger("context_summarizer")
        self.logger.setLevel(logging.INFO)
        
        # Load configuration from environment
        self.enabled = os.getenv('SUMMARIZATION_ENABLED', 'true').lower() == 'true'
        self.threshold_tokens = int(os.getenv('SUMMARIZATION_THRESHOLD_TOKENS', '12000'))
        self.model_url = os.getenv('SUMMARIZATION_MODEL_URL', 'https://api.openai.com/v1')
        self.model_name = os.getenv('SUMMARIZATION_MODEL_NAME', 'gpt-3.5-turbo')
        self.api_key = os.getenv('SUMMARIZATION_API_KEY', '')
        self.temperature = float(os.getenv('SUMMARIZATION_TEMPERATURE', '0.3'))
        self.max_tokens = int(os.getenv('SUMMARIZATION_MAX_TOKENS', '2000'))
        self.keep_recent_turns = int(os.getenv('AGENT_RECENT_TURNS', '4'))
        self.message_limit = int(os.getenv('AGENT_MESSAGE_LIMIT', '8'))
        self.cache_enabled = os.getenv('SUMMARIZATION_CACHE_ENABLED', 'true').lower() == 'true'
        
        # Initialize LLM if enabled and configured
        self.llm = None
        if self.enabled and self.api_key:
            try:
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    base_url=self.model_url,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.logger.info(f"Context summarizer initialized with {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize summarization LLM: {e}")
                self.enabled = False
        else:
            if not self.api_key:
                self.logger.warning("Summarization disabled: no API key provided")
            self.enabled = False
        
        # Initialize token encoder if available
        self.token_encoder = None
        if tiktoken:
            try:
                self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception as e:
                self.logger.warning(f"Failed to initialize token encoder: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken if available, otherwise estimate."""
        if not text:
            return 0
            
        if self.token_encoder:
            try:
                return len(self.token_encoder.encode(text))
            except Exception as e:
                self.logger.warning(f"Token counting failed: {e}")
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def should_summarize(self, context: str) -> bool:
        """Check if context should be summarized based on token count."""
        if not self.enabled:
            return False
            
        token_count = self.count_tokens(context)
        should_summarize = token_count > self.threshold_tokens
        
        if should_summarize:
            self.logger.info(f"Context size {token_count} tokens exceeds threshold {self.threshold_tokens}, summarization needed")
        else:
            self.logger.debug(f"Context size {token_count} tokens is below threshold {self.threshold_tokens}")
            
        return should_summarize
    
    async def summarize_context(
        self,
        game_id: str,
        full_context: str,
        context_data: Dict[str, Any],
        current_turn: int
    ) -> str:
        """
        Main entry point for context summarization.
        Returns optimized context with summaries replacing verbose sections.
        """
        if not self.enabled or not self.llm:
            self.logger.warning("Summarization not available, returning original context")
            return full_context
        
        try:
            self.logger.info(f"Starting context summarization for game {game_id}, turn {current_turn}")
            
            # Get existing summaries from cache
            existing_summaries = action_tracker.get_all_current_summaries(game_id) if self.cache_enabled else {}
            
            # Analyze what needs to be summarized
            sections_to_summarize = self._identify_summarizable_sections(context_data, current_turn, existing_summaries)
            
            # Generate summaries for each section
            new_summaries = {}
            for section_type, section_data in sections_to_summarize.items():
                summary = await self._generate_section_summary(
                    game_id, section_type, section_data, current_turn
                )
                if summary:
                    new_summaries[section_type] = summary
            
            # Reconstruct optimized context
            optimized_context = self._reconstruct_context(
                context_data, new_summaries, existing_summaries, current_turn
            )
            
            # Log token savings
            original_tokens = self.count_tokens(full_context)
            optimized_tokens = self.count_tokens(optimized_context)
            tokens_saved = original_tokens - optimized_tokens
            
            self.logger.info(f"Context optimization complete: {original_tokens} → {optimized_tokens} tokens (saved {tokens_saved})")
            
            return optimized_context
            
        except Exception as e:
            self.logger.exception(f"Error during context summarization: {e}")
            return full_context  # Fallback to original context
    
    def _identify_summarizable_sections(
        self,
        context_data: Dict[str, Any],
        current_turn: int,
        existing_summaries: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Identify which sections of context need summarization."""
        sections_to_summarize = {}
        
        # Strategic history (strategy updates and decisions)
        if self._should_summarize_strategic_history(context_data, current_turn, existing_summaries):
            sections_to_summarize['strategic'] = {
                'strategies': context_data.get('strategies', {}),
                'recent_actions': context_data.get('recent_actions', []),
                'current_turn': current_turn
            }
        
        # Diplomatic history (messages and negotiations)
        if self._should_summarize_diplomatic_history(context_data, current_turn, existing_summaries):
            sections_to_summarize['diplomatic'] = {
                'messages': context_data.get('messages', []),
                'recent_actions': [a for a in context_data.get('recent_actions', []) if a.get('action_type') == 'send_message'],
                'current_turn': current_turn
            }
        
        # Battle history (attacks, conquests, losses)
        if self._should_summarize_battle_history(context_data, current_turn, existing_summaries):
            battle_actions = [
                a for a in context_data.get('recent_actions', [])
                if a.get('action_type') in ['attack_territory', 'place_armies', 'fortify_position']
            ]
            sections_to_summarize['battle_history'] = {
                'battle_actions': battle_actions,
                'board_state': context_data.get('board_state', {}),
                'current_turn': current_turn
            }
        
        # Game evolution (territory control changes, momentum shifts)
        if self._should_summarize_game_evolution(context_data, current_turn, existing_summaries):
            sections_to_summarize['game_evolution'] = {
                'recent_actions': context_data.get('recent_actions', []),
                'board_state': context_data.get('board_state', {}),
                'game_status': context_data.get('game_status', {}),
                'current_turn': current_turn
            }
        
        return sections_to_summarize
    
    def _should_summarize_strategic_history(
        self,
        context_data: Dict[str, Any],
        current_turn: int,
        existing_summaries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Check if strategic history needs summarization."""
        if 'strategic' in existing_summaries:
            last_summary = existing_summaries['strategic']
            # Summarize if it's been more than 5 turns since last summary
            return current_turn - last_summary.get('turn_range_end', 0) > 5
        
        # Summarize if we have strategy data and we're past turn 10
        return bool(context_data.get('strategies')) and current_turn > 10
    
    def _should_summarize_diplomatic_history(
        self,
        context_data: Dict[str, Any],
        current_turn: int,
        existing_summaries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Check if diplomatic history needs summarization."""
        messages = context_data.get('messages', [])
        if not messages:
            return False
        
        if 'diplomatic' in existing_summaries:
            last_summary = existing_summaries['diplomatic']
            # Summarize if we have more than 10 new messages since last summary
            return len(messages) > 10
        
        # Summarize if we have more than 15 messages total
        return len(messages) > 15
    
    def _should_summarize_battle_history(
        self,
        context_data: Dict[str, Any],
        current_turn: int,
        existing_summaries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Check if battle history needs summarization."""
        recent_actions = context_data.get('recent_actions', [])
        battle_actions = [
            a for a in recent_actions
            if a.get('action_type') in ['attack_territory', 'place_armies', 'fortify_position']
        ]
        
        if 'battle_history' in existing_summaries:
            last_summary = existing_summaries['battle_history']
            # Summarize if it's been more than 8 turns since last summary
            return current_turn - last_summary.get('turn_range_end', 0) > 8
        
        # Summarize if we have more than 20 battle actions
        return len(battle_actions) > 20
    
    def _should_summarize_game_evolution(
        self,
        context_data: Dict[str, Any],
        current_turn: int,
        existing_summaries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Check if game evolution needs summarization."""
        if 'game_evolution' in existing_summaries:
            last_summary = existing_summaries['game_evolution']
            # Summarize if it's been more than 10 turns since last summary
            return current_turn - last_summary.get('turn_range_end', 0) > 10
        
        # Summarize if we're past turn 15
        return current_turn > 15
    
    async def _generate_section_summary(
        self,
        game_id: str,
        section_type: str,
        section_data: Dict[str, Any],
        current_turn: int
    ) -> Optional[str]:
        """Generate a summary for a specific section type."""
        try:
            # Get the appropriate prompt for this section type
            prompt = self._get_summary_prompt(section_type, section_data)
            
            if not prompt:
                return None
            
            # Generate summary using LLM
            self.logger.info(f"Generating {section_type} summary for game {game_id}")
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            summary_content = response.content.strip()
            
            if not summary_content:
                self.logger.warning(f"Empty summary generated for {section_type}")
                return None
            
            # Store summary in database if caching is enabled
            if self.cache_enabled:
                # Estimate token counts
                original_tokens = self.count_tokens(json.dumps(section_data))
                summary_tokens = self.count_tokens(summary_content)
                
                # Determine turn range
                turn_range_start = max(1, current_turn - 20)  # Look back up to 20 turns
                turn_range_end = current_turn
                
                success = action_tracker.store_context_summary(
                    game_id=game_id,
                    summary_type=section_type,
                    content=summary_content,
                    turn_range_start=turn_range_start,
                    turn_range_end=turn_range_end,
                    original_tokens=original_tokens,
                    summary_tokens=summary_tokens
                )
                
                if success:
                    self.logger.info(f"Cached {section_type} summary for game {game_id}")
                else:
                    self.logger.warning(f"Failed to cache {section_type} summary")
            
            return summary_content
            
        except Exception as e:
            self.logger.exception(f"Error generating {section_type} summary: {e}")
            return None
    
    def _get_summary_prompt(self, section_type: str, section_data: Dict[str, Any]) -> Optional[str]:
        """Get the appropriate summarization prompt for a section type."""
        
        if section_type == 'strategic':
            return self._get_strategic_summary_prompt(section_data)
        elif section_type == 'diplomatic':
            return self._get_diplomatic_summary_prompt(section_data)
        elif section_type == 'battle_history':
            return self._get_battle_history_summary_prompt(section_data)
        elif section_type == 'game_evolution':
            return self._get_game_evolution_summary_prompt(section_data)
        else:
            self.logger.warning(f"Unknown section type for summarization: {section_type}")
            return None
    
    def _get_strategic_summary_prompt(self, section_data: Dict[str, Any]) -> str:
        """Generate prompt for strategic history summarization."""
        strategies = section_data.get('strategies', {})
        recent_actions = section_data.get('recent_actions', [])
        current_turn = section_data.get('current_turn', 0)
        
        # Filter for strategy-related actions
        strategy_actions = [
            a for a in recent_actions
            if a.get('action_type') in ['update_short_term_strategy', 'update_long_term_strategy']
        ]
        
        return f"""
You are summarizing the strategic evolution of a Risk game player. Create a concise but informative summary of their strategic thinking and decision patterns.

CURRENT TURN: {current_turn}

CURRENT STRATEGIES:
{json.dumps(strategies, indent=2)}

RECENT STRATEGY UPDATES:
{json.dumps(strategy_actions[:10], indent=2)}

Create a summary that includes:
1. **Strategic Evolution**: How the player's strategies have changed over time
2. **Key Strategic Decisions**: Major strategic shifts and their reasoning
3. **Decision Patterns**: Consistent patterns in the player's strategic thinking
4. **Strategic Effectiveness**: How well strategies have been executed

Keep the summary under 300 words but ensure it captures the essential strategic narrative.
Focus on insights that would help understand the player's strategic mindset and decision-making patterns.
"""
    
    def _get_diplomatic_summary_prompt(self, section_data: Dict[str, Any]) -> str:
        """Generate prompt for diplomatic history summarization."""
        messages = section_data.get('messages', [])
        current_turn = section_data.get('current_turn', 0)
        
        # Get recent messages (last 20)
        recent_messages = messages[:20]
        
        return f"""
You are summarizing the diplomatic history of a Risk game. Create a concise summary of key diplomatic events and relationships.

CURRENT TURN: {current_turn}

RECENT DIPLOMATIC MESSAGES:
{json.dumps(recent_messages, indent=2)}

Create a summary that includes:
1. **Key Alliances**: Important alliances formed or broken
2. **Diplomatic Turning Points**: Critical negotiations or betrayals
3. **Communication Patterns**: How players interact diplomatically
4. **Current Relationships**: Status of relationships between players
5. **Negotiation Outcomes**: Results of major diplomatic efforts

Keep the summary under 300 words but capture the essential diplomatic narrative.
Focus on relationships, trust levels, and strategic diplomatic moves that shaped the game.
"""
    
    def _get_battle_history_summary_prompt(self, section_data: Dict[str, Any]) -> str:
        """Generate prompt for battle history summarization."""
        battle_actions = section_data.get('battle_actions', [])
        current_turn = section_data.get('current_turn', 0)
        
        return f"""
You are summarizing the military history of a Risk game. Create a concise summary of major battles, conquests, and military campaigns.

CURRENT TURN: {current_turn}

RECENT BATTLE ACTIONS:
{json.dumps(battle_actions[:25], indent=2)}

Create a summary that includes:
1. **Major Campaigns**: Significant military operations and their outcomes
2. **Key Conquests**: Important territories gained or lost
3. **Battle Patterns**: Consistent military strategies and tactics
4. **Turning Points**: Critical battles that changed the game's momentum
5. **Military Effectiveness**: Success rates and strategic impact of attacks

Keep the summary under 300 words but capture the essential military narrative.
Focus on strategic military decisions and their impact on territorial control.
"""
    
    def _get_game_evolution_summary_prompt(self, section_data: Dict[str, Any]) -> str:
        """Generate prompt for game evolution summarization."""
        recent_actions = section_data.get('recent_actions', [])
        board_state = section_data.get('board_state', {})
        game_status = section_data.get('game_status', {})
        current_turn = section_data.get('current_turn', 0)
        
        return f"""
You are summarizing the overall evolution of a Risk game. Create a concise summary of how the game has progressed and changed over time.

CURRENT TURN: {current_turn}

GAME STATUS:
{json.dumps(game_status, indent=2)}

RECENT GAME ACTIONS (sample):
{json.dumps(recent_actions[:15], indent=2)}

Create a summary that includes:
1. **Game Progression**: How the game has evolved from early to current state
2. **Power Shifts**: Changes in player dominance and territorial control
3. **Momentum Changes**: Key moments that shifted game momentum
4. **Strategic Phases**: Different phases of the game (expansion, consolidation, endgame)
5. **Current State**: Where the game stands now and likely future direction

Keep the summary under 300 words but capture the essential game narrative.
Focus on the big picture of how the game has unfolded and where it's heading.
"""
    
    def _reconstruct_context(
        self,
        context_data: Dict[str, Any],
        new_summaries: Dict[str, str],
        existing_summaries: Dict[str, Dict[str, Any]],
        current_turn: int
    ) -> str:
        """Reconstruct optimized context using summaries."""
        sections = []
        
        # 1. Game Rules & Victory Conditions (always keep full)
        sections.append(self._get_game_rules_section())
        
        # 2. Current Situation (always keep full)
        sections.append(self._get_current_situation_section(context_data))
        
        # 3. Player Status (always keep full)
        sections.append(self._get_player_status_section(context_data))
        
        # 4. Board Analysis (always keep full)
        sections.append(self._get_board_analysis_section(context_data))
        
        # 5. Strategic Summary (use summary if available)
        if 'strategic' in new_summaries:
            sections.append(f"## STRATEGIC HISTORY SUMMARY\n{new_summaries['strategic']}")
        elif 'strategic' in existing_summaries:
            sections.append(f"## STRATEGIC HISTORY SUMMARY\n{existing_summaries['strategic']['content']}")
        
        # 6. Diplomatic Summary (use summary if available)
        if 'diplomatic' in new_summaries:
            sections.append(f"## DIPLOMATIC HISTORY SUMMARY\n{new_summaries['diplomatic']}")
        elif 'diplomatic' in existing_summaries:
            sections.append(f"## DIPLOMATIC HISTORY SUMMARY\n{existing_summaries['diplomatic']['content']}")
        else:
            # Keep recent messages if no summary
            sections.append(self._get_recent_messages_section(context_data, limit=self.message_limit))
        
        # 7. Battle History Summary (use summary if available)
        if 'battle_history' in new_summaries:
            sections.append(f"## BATTLE HISTORY SUMMARY\n{new_summaries['battle_history']}")
        elif 'battle_history' in existing_summaries:
            sections.append(f"## BATTLE HISTORY SUMMARY\n{existing_summaries['battle_history']['content']}")
        
        # 8. Game Evolution Summary (use summary if available)
        if 'game_evolution' in new_summaries:
            sections.append(f"## GAME EVOLUTION SUMMARY\n{new_summaries['game_evolution']}")
        elif 'game_evolution' in existing_summaries:
            sections.append(f"## GAME EVOLUTION SUMMARY\n{existing_summaries['game_evolution']['content']}")
        
        # 9. Recent History (always keep last few turns in detail)
        sections.append(self._get_recent_detailed_history(context_data, self.keep_recent_turns))
        
        # 10. Current Strategies (always keep full)
        sections.append(self._get_current_strategies_section(context_data))
        
        # 11. Action Priority Guide (always keep full)
        sections.append(self._get_action_priority_section(context_data))
        
        return "\n\n".join(sections)
    
    def _get_game_rules_section(self) -> str:
        """Get the game rules section (always full detail)."""
        return """## RISK GAME RULES & VICTORY
**OBJECTIVE**: Conquer the world by eliminating all other players
**VICTORY CONDITION**: Control ALL territories on the board

**Core Mechanics**:
- Each turn: Reinforcement → Attack → Fortify phases
- Reinforcements: Get armies based on territories owned (minimum 3, or territories÷3)
- Continent Bonuses: Control entire continents for extra armies each turn
- Attacking: Roll dice, higher rolls win, attacker needs 2+ armies to attack
- Cards: Earn cards by conquering territories, trade sets for army bonuses

**Key Strategy Principles**:
- Control continents for steady army income
- Eliminate weak players to gain their cards
- Form temporary alliances but be ready to break them
- Fortify borders and chokepoints"""
    
    def _get_current_situation_section(self, context_data: Dict[str, Any]) -> str:
        """Get current situation section (always full detail)."""
        game_status = context_data.get("game_status", {})
        current_turn_actions = context_data.get("current_turn_actions", {}).get("actions", [])
        
        lines = [
            "## CURRENT SITUATION",
            f"- Turn Number: {game_status.get('turn_number', 'Unknown')}",
            f"- Game Phase: {game_status.get('phase', 'Unknown')}",
            f"- Current Player: {game_status.get('current_player', 'Unknown')}"
        ]
        
        if current_turn_actions:
            lines.append("\nActions already taken this turn:")
            for action in current_turn_actions:
                action_type = action.get("action_type", "Unknown")
                action_data = json.dumps(action.get("action_data", {}))
                lines.append(f"- {action_type}: {action_data}")
        else:
            lines.append("\nNo actions taken yet this turn.")
        
        return "\n".join(lines)
    
    def _get_player_status_section(self, context_data: Dict[str, Any]) -> str:
        """Get player status section (always full detail)."""
        player_info = context_data.get("player_info", {})
        territories = player_info.get('territories', [])
        cards = player_info.get('cards', [])
        
        lines = [
            "## YOUR STATUS",
            f"- Army Count: {player_info.get('army_count', 0)} (available to place)",
            f"- Territories Controlled: {len(territories)} (need 42 total to win)",
            f"- Cards in Hand: {len(cards)}"
        ]
        
        if cards:
            card_counts = {}
            for card in cards:
                card_type = card.get('type', 'Unknown')
                card_counts[card_type] = card_counts.get(card_type, 0) + 1
            
            lines.append("\nCard Details:")
            for card_type, count in card_counts.items():
                lines.append(f"  - {card_type}: {count}")
            
            if len(cards) >= 3:
                lines.append("  ⚠️  You can trade cards for army bonus!")
        
        continent_bonuses = player_info.get('continent_bonuses', {})
        if continent_bonuses:
            total_bonus = sum(continent_bonuses.values())
            lines.append(f"\nContinent Bonuses (+{total_bonus} armies/turn):")
            for continent, bonus in continent_bonuses.items():
                lines.append(f"  - {continent}: +{bonus} armies")
        else:
            lines.append("\nContinent Bonuses: None (focus on controlling full continents!)")
        
        return "\n".join(lines)
    
    def _get_board_analysis_section(self, context_data: Dict[str, Any]) -> str:
        """Get board analysis section (always full detail)."""
        # This would be implemented similar to the original format_turn_context method
        return "## BOARD ANALYSIS\n(Current board state and tactical analysis)"
    
    def _get_recent_messages_section(self, context_data: Dict[str, Any], limit: int = 5) -> str:
        """Get recent messages section with limited entries."""
        messages = context_data.get("messages", [])
        if not messages:
            return ""
        
        lines = ["## RECENT DIPLOMATIC MESSAGES"]
        for msg in messages[:limit]:
            from_name = msg.get("from_player_name", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            lines.append(f"\nFrom {from_name} at {timestamp}:")
            lines.append(f"  {content}")
        
        if len(messages) > limit:
            lines.append(f"\n(... and {len(messages) - limit} more messages)")
        
        return "\n".join(lines)
    
    def _get_recent_detailed_history(self, context_data: Dict[str, Any], keep_turns: int) -> str:
        """Get recent detailed history for the last few turns."""
        recent_actions = context_data.get("recent_actions", [])
        
        # Group actions by turn
        turns_actions = {}
        for action in recent_actions:
            turn = action.get("turn_number", 0)
            if turn not in turns_actions:
                turns_actions[turn] = []
            turns_actions[turn].append(action)
        
        # Take the most recent turns
        recent_turns = sorted(turns_actions.keys(), reverse=True)[:keep_turns]
        
        if not recent_turns:
            return "## RECENT DETAILED HISTORY\nNo recent actions available"
        
        lines = [f"## RECENT DETAILED HISTORY (Last {keep_turns} turns)"]
        
        for turn in recent_turns:
            actions = turns_actions[turn]
            if not actions:
                continue
            
            player_id = actions[0].get("player_id", "Unknown")
            lines.append(f"\nTurn {turn} - Player {player_id}:")
            
            for action in actions:
                action_type = action.get("action_type", "Unknown")
                action_data = json.dumps(action.get("action_data", {}))
                lines.append(f"  - {action_type}: {action_data}")
        
        return "\n".join(lines)
    
    def _get_current_strategies_section(self, context_data: Dict[str, Any]) -> str:
        """Get current strategies section (always full detail)."""
        strategies = context_data.get("strategies", {})
        short_term = strategies.get("short_term_strategy", "No strategy set")
        long_term = strategies.get("long_term_strategy", "No strategy set")
        
        db_strategies = strategies.get("database_strategies", {})
        short_term_updated = db_strategies.get("short_term", {}).get("updated_at", "Unknown")
        long_term_updated = db_strategies.get("long_term", {}).get("updated_at", "Unknown")
        
        return f"""## YOUR CURRENT STRATEGIES
- Short-term: {short_term} (updated: {short_term_updated})
- Long-term: {long_term} (updated: {long_term_updated})"""
    
    def _get_action_priority_section(self, context_data: Dict[str, Any]) -> str:
        """Get action priority guide section (always full detail)."""
        return """## ACTION PRIORITY GUIDE
Use this guide to prioritize your actions:

**PRIORITY ORDER:**
1. 🎴 **Trade Cards** (if you have 3+ cards) - Get army bonus immediately
2. 🛡️ **Place Armies** (if army_count > 0) - Reinforce vulnerable territories first
3. ⚔️ **Attack** (if you have strong positions) - Target weak adjacent territories
4. 🏰 **Fortify** (if needed) - Move armies to strategic positions
5. 💬 **Send Message** (for diplomacy) - Strategic communication
6. ⏹️ **End Turn** (if no other actions needed) - Complete your turn

**REMEMBER:**
- Every turn MUST include at least one concrete game action
- Don't get stuck in analysis paralysis - make decisions quickly
- Focus on actions that advance your position toward world domination"""


# Global instance
context_summarizer = ContextSummarizer()
