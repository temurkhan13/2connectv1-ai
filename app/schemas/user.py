"""
User-related Pydantic schemas for request/response validation.
"""
import re
import html
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from uuid import UUID


# Constants for validation
MAX_FEEDBACK_LENGTH = 5000
MAX_TEMPLATE_LENGTH = 2000
MAX_CONTENT_LENGTH = 10000


def sanitize_html(text: str) -> str:
    """Escape HTML characters to prevent XSS attacks."""
    if text is None:
        return text
    return html.escape(str(text), quote=True)


class QuestionAnswer(BaseModel):
    """Schema for question-answer pairs."""
    prompt: str = Field(..., description="Question prompt text")
    answer: Union[str, int, float, dict, list] = Field(..., description="Answer value")


class UserRegistrationRequest(BaseModel):
    """Schema for user registration request."""
    user_id: str = Field(..., description="Unique user identifier")
    resume_link: Optional[str] = Field(None, description="S3 link to resume file")
    questions: List[QuestionAnswer] = Field(..., description="Array of question-answer pairs")
    update: Optional[bool] = Field(False, description="Is this an update request?")

    @field_validator('user_id')
    @classmethod
    def _validate_uuid(cls, v: str):
        UUID(v)
        return v


class UserRegistrationResponse(BaseModel):
    """Schema for user registration response."""
    success: bool = Field(..., description="Whether registration was successful")
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="Response message")


class UserProfileResponse(BaseModel):
    """Schema for user profile response."""
    user_id: str = Field(..., description="User identifier")
    profile: dict = Field(..., description="User profile data")
    resume_text: Optional[dict] = Field(None, description="Extracted resume text")
    persona: Optional[dict] = Field(None, description="Generated persona data")
    processing_status: str = Field(..., description="Current processing status")
    persona_status: str = Field(..., description="Current persona generation status")
    needs_matchmaking: str = Field(..., description="Whether user needs matchmaking processing ('true' or 'false')")


class ApproveSummaryRequest(BaseModel):
    """Schema for approve summary request."""
    user_id: str = Field(..., description="User identifier")

    @field_validator('user_id')
    @classmethod
    def validate_user_id_uuid(cls, v: str) -> str:
        """Validate UUID format for user_id."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v


class FeedbackRequest(BaseModel):
    """Schema for user feedback on matches/chats."""
    user_id: str = Field(..., description="User identifier")
    type: Literal["match", "chat"] = Field(..., description="Feedback type (match or chat)")
    id: str = Field(..., description="Match or chat identifier being reviewed")
    feedback: str = Field(..., min_length=1, max_length=MAX_FEEDBACK_LENGTH, description="User feedback text")

    @field_validator('user_id', 'id')
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        """Validate UUID format for identifiers."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v

    @field_validator('feedback')
    @classmethod
    def sanitize_feedback(cls, v: str) -> str:
        """Sanitize feedback text to prevent XSS."""
        return sanitize_html(v.strip())


class InitiateAIChatRequest(BaseModel):
    """Schema for initiating AI-to-AI chat."""
    initiator_id: str = Field(..., description="Initiator user ID")
    responder_id: str = Field(..., description="Responder user ID")
    match_id: str = Field(..., description="Match identifier")
    template: Optional[str] = Field(None, max_length=MAX_TEMPLATE_LENGTH, description="Optional custom message template")

    @field_validator('initiator_id', 'responder_id', 'match_id')
    @classmethod
    def validate_uuid_ids(cls, v: str) -> str:
        """Validate UUID format for all identifiers."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v

    @field_validator('template')
    @classmethod
    def sanitize_template(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize template to prevent injection."""
        if v is None:
            return v
        return sanitize_html(v.strip())


class InitiateAIChatResponse(BaseModel):
    """Schema for AI chat initiation response."""
    code: int = Field(..., description="Response code")
    message: str = Field(..., description="Response message")
    result: bool = Field(..., description="Success status")


class ConversationMessage(BaseModel):
    """Schema for individual conversation message."""
    sender_id: str = Field(..., description="Sender user ID")
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH, description="Message content")

    @field_validator('sender_id')
    @classmethod
    def validate_sender_uuid(cls, v: str) -> str:
        """Validate UUID format for sender_id."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v

    @field_validator('content')
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        """Sanitize message content to prevent XSS."""
        return sanitize_html(v.strip())


class AIChatWebhookPayload(BaseModel):
    """Schema for AI chat completion webhook payload."""
    initiator_id: str = Field(..., description="Initiator user ID")
    responder_id: str = Field(..., description="Responder user ID")
    match_id: str = Field(..., description="Match identifier")
    ai_remarks: str = Field(..., description="AI-generated remarks about the conversation")
    compatibility_score: int = Field(..., description="Compatibility score (0-100)")
    conversation_data: List[ConversationMessage] = Field(..., description="List of conversation messages")
