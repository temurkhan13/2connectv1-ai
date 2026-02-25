"""
Answer prediction schemas.
"""
import html
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# Constants for validation
MAX_OPTION_LENGTH = 500
MAX_RESPONSE_LENGTH = 2000
MAX_OPTIONS_COUNT = 50


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


class PredictAnswerPayload(BaseModel):
    """Schema for answer prediction request."""
    options: List[OptionItem] = Field(..., min_length=1, max_length=MAX_OPTIONS_COUNT)
    user_response: str = Field(..., min_length=1, max_length=MAX_RESPONSE_LENGTH)

    @field_validator('user_response')
    @classmethod
    def sanitize_response(cls, v: str) -> str:
        """Sanitize user response to prevent injection."""
        return sanitize_text(v.strip())


class PredictAnswerResponse(BaseModel):
    """Schema for answer prediction response."""
    predicted_answer: Optional[str] = None
    valid_answer: Optional[bool] = None
    fallback_text: str = ""

