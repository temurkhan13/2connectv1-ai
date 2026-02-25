"""
Question modification schemas.
"""
import html
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator


# Constants for validation
MAX_PROMPT_LENGTH = 5000
MAX_RESPONSE_LENGTH = 10000
MAX_OPTION_LENGTH = 500


def sanitize_text(text: str) -> str:
    """Escape HTML characters to prevent XSS attacks."""
    if text is None:
        return text
    return html.escape(str(text), quote=True)


class OptionItem(BaseModel):
    """Schema for option items."""
    label: str = Field(..., min_length=1, max_length=MAX_OPTION_LENGTH)
    value: str = Field(..., min_length=1, max_length=MAX_OPTION_LENGTH)

    @field_validator('label', 'value')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace."""
        return v.strip() if v else v


class PreviousUserResponse(BaseModel):
    """Schema for previous user responses."""
    question_id: str = Field(..., min_length=1, max_length=100)
    ai_text: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    description: Optional[str] = Field(None, max_length=MAX_PROMPT_LENGTH)
    narration: Optional[str] = Field(None, max_length=MAX_PROMPT_LENGTH)
    suggestion_chips: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    options: Optional[Union[str, List[OptionItem]]] = None
    user_response: Optional[str] = Field(None, max_length=MAX_RESPONSE_LENGTH)

    @field_validator('user_response')
    @classmethod
    def sanitize_user_response(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize user response to prevent injection."""
        if v is None:
            return v
        return sanitize_text(v.strip())


class QuestionPayload(BaseModel):
    """Schema for question modification request."""
    previous_user_response: Optional[List[PreviousUserResponse]] = Field(default_factory=list)
    question_id: str = Field(..., min_length=1, max_length=100)
    code: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    narration: Optional[str] = Field(None, max_length=MAX_PROMPT_LENGTH)
    description: Optional[str] = Field(None, max_length=MAX_PROMPT_LENGTH)
    suggestion_chips: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    options: Optional[Union[str, List[OptionItem]]] = None

    @field_validator('previous_user_response')
    @classmethod
    def limit_previous_responses(cls, v: Optional[List[PreviousUserResponse]]) -> Optional[List[PreviousUserResponse]]:
        """Limit the number of previous responses to prevent payload abuse."""
        if v and len(v) > 50:
            raise ValueError("Too many previous responses (max 50)")
        return v


class QuestionResponse(BaseModel):
    """Schema for question modification response."""
    question_id: str
    ai_text: str
    suggestion_chips: str

