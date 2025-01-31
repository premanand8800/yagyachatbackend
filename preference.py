# app/models/preference.py
from typing import List, Optional, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

class PreferenceCategory(str, Enum):
    """Categories of user preferences"""
    LEARNING_STYLE = "learning_style"
    CAREER_PATH = "career_path"
    INDUSTRY = "industry"
    TECHNOLOGY = "technology"
    WORK_ENVIRONMENT = "work_environment"
    LOCATION = "location"
    EXPERIENCE_LEVEL = "experience_level"
    ROLE_TYPE = "role_type"
    COMPANY_SIZE = "company_size"
    DOMAIN = "domain"

class PreferenceOperation(str, Enum):
    """Types of preference operations"""
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    QUERY = "query"

class PreferenceValue(BaseModel):
    """Structure for preference values"""
    category: PreferenceCategory
    value: Union[str, List[str]]
    priority: Optional[int] = None
    constraints: Optional[Dict] = None
    
    @validator('priority')
    def validate_priority(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Priority must be between 1 and 10")
        return v

class PreferenceUpdate(BaseModel):
    """Structure for preference updates"""
    operation: PreferenceOperation
    preferences: List[PreferenceValue]
    reason: Optional[str] = None

class PreferenceAnalysisResult(BaseModel):
    """Results from analyzing preference-related input"""
    detected_operation: PreferenceOperation
    detected_preferences: List[PreferenceValue]
    confidence_score: float
    needs_clarification: bool
    clarification_questions: List[str] = Field(default_factory=list)