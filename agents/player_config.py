#!/usr/bin/env python3
"""
Player Configuration System for Risk AI Agents

This module provides interactive configuration for setting up AI players
with different models, temperatures, and custom instructions.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try manual loading
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


@dataclass
class PlayerConfig:
    """Configuration for a single AI player."""
    name: str
    model_name: str
    temperature: float
    custom_instructions: str
    api_key: str
    base_url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerConfig':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


class PlayerConfigManager:
    """Manages player configuration for Risk AI games."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.predefined_models = self._load_predefined_models()
        self.personality_templates = self._load_personality_templates()
        self.api_key = os.getenv('RISK_API_KEY', '')
        self.base_url = os.getenv('RISK_BASE_URL', '')
    
    def _load_predefined_models(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined models from environment variable."""
        models_json = os.getenv('RISK_PREDEFINED_MODELS', '{}')
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            print("⚠️  Warning: Invalid RISK_PREDEFINED_MODELS format, using defaults")
            return {
                "gpt-4": {"name": "gpt-4", "default_temp": 0.7, "description": "Most capable, best for complex strategy"},
                "gpt-4-turbo": {"name": "gpt-4-turbo", "default_temp": 0.7, "description": "Fast and capable, good balance"},
                "gpt-3.5-turbo": {"name": "gpt-3.5-turbo", "default_temp": 0.8, "description": "Fast and reliable, good for testing"},
                "custom": {"name": "custom", "default_temp": 0.7, "description": "Enter your own model name"}
            }
    
    def _load_personality_templates(self) -> Dict[str, str]:
        """Load personality templates from environment variable."""
        templates_json = os.getenv('RISK_PERSONALITY_TEMPLATES', '{}')
        try:
            return json.loads(templates_json)
        except json.JSONDecodeError:
            print("⚠️  Warning: Invalid RISK_PERSONALITY_TEMPLATES format, using defaults")
            return {
                "balanced": "Adapt strategy based on game state and opportunities",
                "aggressive": "Play aggressively, focus on early expansion and eliminating weak players",
                "diplomatic": "Form alliances and use negotiation to advance your position",
                "defensive": "Control key territories and build strong defensive positions"
            }
    
    def _validate_api_credentials(self) -> bool:
        """Validate that API credentials are available."""
        if not self.api_key:
            print("❌ Error: RISK_API_KEY environment variable not set")
            print("   Please set your OpenAI-compatible API key in the .env file")
            return False
        
        if not self.base_url:
            print("❌ Error: RISK_BASE_URL environment variable not set")
            print("   Please set your OpenAI-compatible base URL in the .env file")
            return False
        
        return True
    
    def _get_model_selection(self) -> tuple[str, float]:
        """Interactive model selection with temperature."""
        print("\n📋 Available Models:")
        print("=" * 50)
        
        model_list = list(self.predefined_models.items())
        for i, (key, model_info) in enumerate(model_list, 1):
            name = model_info['name']
            desc = model_info['description']
            temp = model_info['default_temp']
            print(f"  {i}. {name} (temp: {temp}) - {desc}")
        
        while True:
            try:
                choice = input(f"\nChoose model (1-{len(model_list)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(model_list):
                    selected_key = list(self.predefined_models.keys())[choice_num - 1]
                    selected_model = self.predefined_models[selected_key]
                    
                    # Handle custom model entry
                    if selected_key == "custom":
                        custom_name = input("Enter custom model name: ").strip()
                        if not custom_name:
                            print("❌ Model name cannot be empty")
                            continue
                        model_name = custom_name
                    else:
                        model_name = selected_model['name']
                    
                    # Get temperature with smart default
                    default_temp = selected_model['default_temp']
                    temp_input = input(f"Temperature (default {default_temp}, press Enter to use): ").strip()
                    
                    if temp_input:
                        try:
                            temperature = float(temp_input)
                            if not 0.0 <= temperature <= 1.0:
                                print("❌ Temperature must be between 0.0 and 1.0")
                                continue
                        except ValueError:
                            print("❌ Invalid temperature format")
                            continue
                    else:
                        temperature = default_temp
                    
                    return model_name, temperature
                else:
                    print(f"❌ Please choose a number between 1 and {len(model_list)}")
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Configuration cancelled")
                exit(0)
    
    def _get_custom_instructions(self) -> str:
        """Get custom instructions with template options."""
        print("\n🎭 Player Instructions:")
        print("=" * 30)
        print("Choose a personality template or enter custom instructions:")
        print()
        
        # Show personality templates
        template_list = list(self.personality_templates.items())
        for i, (key, description) in enumerate(template_list, 1):
            print(f"  {i}. {key.title()}: {description}")
        
        print(f"  {len(template_list) + 1}. Custom: Enter your own instructions")
        print(f"  {len(template_list) + 2}. Default: Use standard Risk AI instructions")
        
        while True:
            try:
                choice = input(f"\nChoose option (1-{len(template_list) + 2}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(template_list):
                    # Use personality template
                    selected_key = list(self.personality_templates.keys())[choice_num - 1]
                    return self.personality_templates[selected_key]
                
                elif choice_num == len(template_list) + 1:
                    # Custom instructions
                    print("\nEnter custom instructions (press Enter twice to finish):")
                    lines = []
                    empty_lines = 0
                    while empty_lines < 2:
                        line = input()
                        if line.strip():
                            lines.append(line)
                            empty_lines = 0
                        else:
                            empty_lines += 1
                    
                    custom_instructions = "\n".join(lines).strip()
                    if not custom_instructions:
                        print("❌ Instructions cannot be empty")
                        continue
                    return custom_instructions
                
                elif choice_num == len(template_list) + 2:
                    # Default instructions
                    return "Play strategically, balance offense and defense, and adapt to the game situation"
                
                else:
                    print(f"❌ Please choose a number between 1 and {len(template_list) + 2}")
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Configuration cancelled")
                exit(0)
    
    def _validate_player_name(self, name: str, existing_names: List[str]) -> bool:
        """Validate player name."""
        if not name:
            print("❌ Player name cannot be empty")
            return False
        
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            print("❌ Player name must start with a letter and contain only letters, numbers, hyphens, and underscores")
            return False
        
        if name in existing_names:
            print(f"❌ Player name '{name}' is already taken")
            return False
        
        if len(name) > 20:
            print("❌ Player name must be 20 characters or less")
            return False
        
        return True
    
    def configure_single_player(self, player_num: int, existing_names: List[str]) -> PlayerConfig:
        """Configure a single player interactively."""
        print(f"\n🎮 PLAYER {player_num} CONFIGURATION")
        print("=" * 40)
        
        # Get player name
        while True:
            name = input(f"Player {player_num} name: ").strip()
            if self._validate_player_name(name, existing_names):
                break
        
        # Get model and temperature
        model_name, temperature = self._get_model_selection()
        
        # Get custom instructions
        custom_instructions = self._get_custom_instructions()
        
        # Create player configuration
        config = PlayerConfig(
            name=name,
            model_name=model_name,
            temperature=temperature,
            custom_instructions=custom_instructions,
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Show configuration summary
        print(f"\n✅ Player {player_num} configured:")
        print(f"   Name: {config.name}")
        print(f"   Model: {config.model_name}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Instructions: {config.custom_instructions[:60]}{'...' if len(config.custom_instructions) > 60 else ''}")
        
        return config
    
    def configure_players_interactive(self) -> List[PlayerConfig]:
        """Interactive configuration for all players."""
        print("\n🎮 RISK AI GAME CONFIGURATION")
        print("=" * 50)
        
        # Validate API credentials first
        if not self._validate_api_credentials():
            return []
        
        # Get number of players
        while True:
            try:
                num_players = int(input("How many players? (2-6): ").strip())
                if 2 <= num_players <= 6:
                    break
                else:
                    print("❌ Number of players must be between 2 and 6")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Configuration cancelled")
                return []
        
        # Configure each player
        players = []
        existing_names = []
        
        for i in range(1, num_players + 1):
            try:
                player_config = self.configure_single_player(i, existing_names)
                players.append(player_config)
                existing_names.append(player_config.name)
            except KeyboardInterrupt:
                print("\n\n👋 Configuration cancelled")
                return []
        
        # Show final summary
        print(f"\n🎉 CONFIGURATION COMPLETE!")
        print("=" * 30)
        print(f"Players configured: {len(players)}")
        for i, player in enumerate(players, 1):
            print(f"  {i}. {player.name} ({player.model_name}, temp: {player.temperature})")
        
        # Ask if user wants to save configuration
        save_choice = input("\nSave this configuration? (y/N): ").strip().lower()
        if save_choice in ['y', 'yes']:
            self.save_configuration(players)
        
        return players
    
    def save_configuration(self, players: List[PlayerConfig], filename: Optional[str] = None) -> bool:
        """Save player configuration to JSON file."""
        if not filename:
            filename = input("Configuration filename (without .json): ").strip()
            if not filename:
                filename = "risk_config"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create configs directory if it doesn't exist
        config_dir = Path('configs')
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / filename
        
        try:
            config_data = {
                'players': [player.to_dict() for player in players],
                'created_at': str(Path().cwd()),
                'api_key_env': 'RISK_API_KEY',
                'base_url_env': 'RISK_BASE_URL'
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def load_configuration(self, filename: str) -> List[PlayerConfig]:
        """Load player configuration from JSON file."""
        config_path = Path('configs') / filename
        
        if not config_path.exists():
            # Try without .json extension
            if not filename.endswith('.json'):
                config_path = Path('configs') / f"{filename}.json"
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return []
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            players = []
            for player_data in config_data.get('players', []):
                # Update API credentials from current environment
                player_data['api_key'] = self.api_key
                player_data['base_url'] = self.base_url
                players.append(PlayerConfig.from_dict(player_data))
            
            print(f"✅ Configuration loaded from {config_path}")
            print(f"   Players: {', '.join(p.name for p in players)}")
            
            return players
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return []
    
    def list_saved_configurations(self) -> List[str]:
        """List all saved configuration files."""
        config_dir = Path('configs')
        if not config_dir.exists():
            return []
        
        configs = []
        for config_file in config_dir.glob('*.json'):
            configs.append(config_file.stem)
        
        return sorted(configs)
    
    def create_quick_preset(self, preset_name: str, num_players: int = 3) -> List[PlayerConfig]:
        """Create a quick preset configuration."""
        presets = {
            'gpt4_tournament': {
                'models': ['gpt-4', 'gpt-4-turbo', 'gpt-4'],
                'personalities': ['aggressive', 'diplomatic', 'balanced']
            },
            'mixed_models': {
                'models': ['gpt-4', 'gpt-3.5-turbo', 'claude-3-sonnet'],
                'personalities': ['balanced', 'creative', 'analytical']
            },
            'beginner_friendly': {
                'models': ['gpt-3.5-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo'],
                'personalities': ['balanced', 'balanced', 'balanced']
            }
        }
        
        if preset_name not in presets:
            print(f"❌ Unknown preset: {preset_name}")
            return []
        
        preset = presets[preset_name]
        players = []
        
        default_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
        
        for i in range(num_players):
            model_name = preset['models'][i % len(preset['models'])]
            personality = preset['personalities'][i % len(preset['personalities'])]
            
            # Get model info for temperature
            model_info = self.predefined_models.get(model_name, {'default_temp': 0.7})
            
            config = PlayerConfig(
                name=default_names[i],
                model_name=model_name,
                temperature=model_info['default_temp'],
                custom_instructions=self.personality_templates.get(personality, "Play strategically"),
                api_key=self.api_key,
                base_url=self.base_url
            )
            players.append(config)
        
        print(f"✅ Created '{preset_name}' preset with {num_players} players")
        return players


def main():
    """Main function for testing the configuration system."""
    manager = PlayerConfigManager()
    
    print("🧪 Testing Player Configuration System")
    print("=" * 50)
    
    # Test interactive configuration
    players = manager.configure_players_interactive()
    
    if players:
        print(f"\n✅ Successfully configured {len(players)} players!")
        for player in players:
            print(f"   - {player.name}: {player.model_name} (temp: {player.temperature})")


if __name__ == "__main__":
    main()
