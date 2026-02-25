"""
Persona-related Pydantic schemas for request/response validation.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PersonaDraftRequest(BaseModel):
    """Schema for persona draft generation request."""
    user_id: str = Field(..., description="User identifier")
    resume_text: str = Field(..., description="Extracted resume text")
    questions: list = Field(..., description="User onboarding questions")


class PersonaDraftResponse(BaseModel):
    """Schema for persona draft response."""
    success: bool = Field(..., description="Whether draft generation was successful")
    user_id: str = Field(..., description="User identifier")
    draft: Optional[Dict[str, Any]] = Field(None, description="Generated persona draft")
    message: str = Field(..., description="Response message")


class PersonaFinalizeRequest(BaseModel):
    """Schema for persona finalization request."""
    user_id: str = Field(..., description="User identifier")
    draft: Dict[str, Any] = Field(..., description="Persona draft to finalize")


class PersonaFinalizeResponse(BaseModel):
    """Schema for persona finalization response."""
    success: bool = Field(..., description="Whether finalization was successful")
    user_id: str = Field(..., description="User identifier")
    persona: Optional[Dict[str, Any]] = Field(None, description="Finalized persona")
    message: str = Field(..., description="Response message")


class PersonaMatchRequest(BaseModel):
    """Schema for persona matching request."""
    user_id: str = Field(..., description="User identifier")
    match_type: str = Field(..., description="Type of match (needs, personality)")
    limit: int = Field(10, description="Maximum number of matches to return")


class PersonaMatchResponse(BaseModel):
    """Schema for persona matching response."""
    success: bool = Field(..., description="Whether matching was successful")
    user_id: str = Field(..., description="User identifier")
    matches: list = Field(..., description="List of matched personas")
    message: str = Field(..., description="Response message")
