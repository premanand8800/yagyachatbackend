from pydantic import BaseModel
from typing import Optional, Dict, Any

class UserInput(BaseModel):
    """Model for user input data"""
    raw_input: str
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
