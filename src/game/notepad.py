from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Message:
    from_player_id: str
    from_player_name: str
    to_player_id: str
    content: str
    timestamp: datetime
    is_read: bool = False
    
    def to_dict(self) -> dict:
        """Convert message to dictionary for serialization."""
        return {
            'from_player_id': self.from_player_id,
            'from_player_name': self.from_player_name,
            'to_player_id': self.to_player_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'is_read': self.is_read
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create message from dictionary."""
        return cls(
            from_player_id=data['from_player_id'],
            from_player_name=data['from_player_name'],
            to_player_id=data['to_player_id'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            is_read=data['is_read']
        )

class NotepadManager:
    """Manages private notes and messages between players."""
    
    def __init__(self):
        self.private_notes: Dict[str, str] = {}  # player_id -> private notes
        self.messages: Dict[str, List[Message]] = {}  # player_id -> list of messages received
    
    def get_private_notes(self, player_id: str) -> str:
        """Get a player's private notes."""
        return self.private_notes.get(player_id, "")
    
    def update_private_notes(self, player_id: str, notes: str) -> None:
        """Update a player's private notes."""
        self.private_notes[player_id] = notes
    
    def send_message(self, from_player_id: str, from_player_name: str, 
                    to_player_id: str, content: str) -> bool:
        """Send a message from one player to another."""
        if not content.strip():
            return False
        
        message = Message(
            from_player_id=from_player_id,
            from_player_name=from_player_name,
            to_player_id=to_player_id,
            content=content.strip(),
            timestamp=datetime.now()
        )
        
        if to_player_id not in self.messages:
            self.messages[to_player_id] = []
        
        self.messages[to_player_id].append(message)
        return True
    
    def get_messages(self, player_id: str, mark_as_read: bool = True) -> List[Message]:
        """Get all messages for a player."""
        if player_id not in self.messages:
            return []
        
        messages = self.messages[player_id]
        
        if mark_as_read:
            for message in messages:
                message.is_read = True
        
        return messages
    
    def get_unread_messages(self, player_id: str) -> List[Message]:
        """Get only unread messages for a player."""
        if player_id not in self.messages:
            return []
        
        return [msg for msg in self.messages[player_id] if not msg.is_read]
    
    def get_unread_count(self, player_id: str) -> int:
        """Get the number of unread messages for a player."""
        return len(self.get_unread_messages(player_id))
    
    def mark_messages_read(self, player_id: str) -> None:
        """Mark all messages as read for a player."""
        if player_id in self.messages:
            for message in self.messages[player_id]:
                message.is_read = True
    
    def get_message_history(self, player1_id: str, player2_id: str) -> List[Message]:
        """Get conversation history between two players."""
        all_messages = []
        
        # Get messages from player1 to player2
        if player2_id in self.messages:
            all_messages.extend([
                msg for msg in self.messages[player2_id] 
                if msg.from_player_id == player1_id
            ])
        
        # Get messages from player2 to player1
        if player1_id in self.messages:
            all_messages.extend([
                msg for msg in self.messages[player1_id] 
                if msg.from_player_id == player2_id
            ])
        
        # Sort by timestamp
        all_messages.sort(key=lambda msg: msg.timestamp)
        return all_messages
    
    def format_messages(self, messages: List[Message], current_player_id: str) -> str:
        """Format messages for display."""
        if not messages:
            return "No messages."
        
        formatted = []
        for msg in messages:
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            read_indicator = "" if msg.is_read else "📩 "
            
            if msg.to_player_id == current_player_id:
                # Message received
                formatted.append(f"{read_indicator}[{timestamp_str}] From {msg.from_player_name}: {msg.content}")
            else:
                # Message sent (when showing conversation history)
                formatted.append(f"[{timestamp_str}] To {msg.from_player_name}: {msg.content}")
        
        return "\n".join(formatted)
    
    def get_message_summary(self, player_id: str) -> str:
        """Get a summary of messages for a player."""
        unread_count = self.get_unread_count(player_id)
        total_messages = len(self.messages.get(player_id, []))
        
        if total_messages == 0:
            return "No messages."
        
        summary = f"Messages: {total_messages} total"
        if unread_count > 0:
            summary += f", {unread_count} unread 📩"
        
        return summary
    
    def clear_player_data(self, player_id: str) -> None:
        """Clear all data for a player (used when player leaves game)."""
        self.private_notes.pop(player_id, None)
        self.messages.pop(player_id, None)
        
        # Remove messages sent by this player from other players' inboxes
        for other_messages in self.messages.values():
            other_messages[:] = [
                msg for msg in other_messages 
                if msg.from_player_id != player_id
            ]
    
    def to_dict(self) -> dict:
        """Convert notepad data to dictionary for serialization."""
        return {
            'private_notes': self.private_notes.copy(),
            'messages': {
                player_id: [msg.to_dict() for msg in messages]
                for player_id, messages in self.messages.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'NotepadManager':
        """Create notepad manager from dictionary."""
        manager = cls()
        manager.private_notes = data.get('private_notes', {}).copy()
        
        messages_data = data.get('messages', {})
        for player_id, msg_list in messages_data.items():
            manager.messages[player_id] = [
                Message.from_dict(msg_data) for msg_data in msg_list
            ]
        
        return manager
