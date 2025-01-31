# app/models/analysis_models.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class UserSummary(BaseModel):
    """Summary of user's background, goals, and interests"""
    background: str = Field(default="", description="User's background information")
    expertise_level: str = Field(default="", description="User's expertise level")
    goals: List[str] = Field(default_factory=list, description="User's goals")
    key_interests: List[str] = Field(default_factory=list, description="User's key interests")


class KeywordsAnalysis(BaseModel):
    """Analysis of keywords present in user's input"""
    domain_terms: List[str] = Field(default_factory=list, description="Primary domain keywords")
    technical_terms: List[str] = Field(default_factory=list, description="Technical/professional terms")
    relationships: List[str] = Field(default_factory=list, description="Relationships between keywords")


class NeedsAssessment(BaseModel):
    """Assessment of user's needs"""
    explicit: List[str] = Field(default_factory=list, description="Explicitly stated needs")
    implicit: List[str] = Field(default_factory=list, description="Implicit requirements")
    gaps: List[str] = Field(default_factory=list, description="Identified knowledge gaps")


class SegmentUnderstanding(BaseModel):
    """Understanding of user's segment"""
    professional_category: str = Field(default="", description="Primary professional category")
    experience_level: str = Field(default="", description="User's experience level")
    industry_context: str = Field(default="", description="User's industry context")
    role_characteristics: str = Field(default="", description="User's role characteristics")
    growth_stage: str = Field(default="", description="User's growth stage")
    impact_potential: str = Field(default="", description="User's impact potential")
class AnalysisResults(BaseModel):
    """Result of the comprehensive analysis"""
    user_summary: UserSummary = Field(default_factory=UserSummary)
    keywords: KeywordsAnalysis = Field(default_factory=KeywordsAnalysis)
    needs: NeedsAssessment = Field(default_factory=NeedsAssessment)
    segment: SegmentUnderstanding = Field(default_factory=SegmentUnderstanding)