"""
Common Pydantic schemas used across the application.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check response."""
    success: bool = Field(..., description="Whether service is healthy")
    data: Dict[str, Any] = Field(..., description="Health check data")
    message: str = Field(..., description="Health check message")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
