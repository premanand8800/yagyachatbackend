from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class InputType(str, Enum):
    """Extended enumeration of input types including preferences"""
    NEW_QUERY = "new_query"
    REWRITE_REQUEST = "rewrite_request"
    INPUT_ENHANCEMENT = "input_enhancement"
    CLARIFICATION_RESPONSE = "clarification_response"
    PREFERENCE_UPDATE = "preference_update"
    PREFERENCE_REMOVAL = "preference_removal"
    PREFERENCE_QUERY = "preference_query"
    INVALID_INPUT = "invalid_input"

class ValidationResult(BaseModel):
    """Enhanced validation results including conversation context"""
    is_valid: bool = Field(description="Whether the input is valid")
    has_background: bool = Field(description="Whether background information is detected")
    has_goals: bool = Field(description="Whether goals are detected")
    background_completeness: float = Field(description="Score for background completeness (0-1)")
    goals_clarity: float = Field(description="Score for goals clarity (0-1)")
    clarity_score: float = Field(default=0.0, description="Overall clarity score (0-1)")
    safety_score: float = Field(default=1.0, description="Safety assessment score (0-1)")
    context_score: float = Field(default=0.0, description="How well input fits with conversation context (0-1)")
    input_type: InputType = Field(description="Type of input")
    error_message: Optional[str] = Field(default=None, description="Error message if validation fails")
    missing_elements: List[str] = Field(default_factory=list, description="Missing information in input")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions to clarify input")
    detected_preferences: Dict[str, str] = Field(default_factory=dict, description="Detected user preferences")
    background_info: Dict[str, str] = Field(default_factory=dict, description="Extracted background information")
    goals: List[str] = Field(default_factory=list, description="Extracted goals")
    validation_details: Dict[str, Any] = Field(default_factory=dict, description="Additional validation details")
    guardrails_result: Dict = Field(default_factory=dict, description="Results from content safety checks")
