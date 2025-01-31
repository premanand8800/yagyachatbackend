from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation"""
    timestamp: datetime = Field(default_factory=datetime.now)
    user_input: str
    input_type: str
    preferences: Dict[str, str] = Field(default_factory=dict)
    background_info: Dict[str, str] = Field(default_factory=dict)
    goals: List[str] = Field(default_factory=list)
    context_score: float = 0.0

class ConversationMemory(BaseModel):
    """Stores and manages conversation history"""
    turns: List[ConversationTurn] = Field(default_factory=list)
    max_turns: int = 10  # Maximum number of turns to remember
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn, maintaining max_turns limit"""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)  # Remove oldest turn
    
    def get_recent_context(self, num_turns: int = 3) -> List[ConversationTurn]:
        """Get the most recent conversation turns"""
        return self.turns[-num_turns:] if self.turns else []
    
    def get_all_preferences(self) -> Dict[str, str]:
        """Aggregate all preferences from conversation history"""
        preferences = {}
        for turn in self.turns:
            preferences.update(turn.preferences)
        return preferences
    
    def get_background_info(self) -> Dict[str, str]:
        """Aggregate background information from conversation history"""
        background = {}
        for turn in self.turns:
            background.update(turn.background_info)
        return background
    
    def get_goals(self) -> List[str]:
        """Get all unique goals from conversation history"""
        goals = set()
        for turn in self.turns:
            goals.update(turn.goals)
        return list(goals)
