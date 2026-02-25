"""
Unified Error Handling Middleware.

Provides consistent error handling across the application with
custom exceptions, error codes, and formatted responses.

Key features:
1. Custom exception hierarchy
2. Error code system
3. Consistent error responses
4. Error logging and tracking
5. Request context preservation
"""
import os
import sys
import traceback
import logging
from typing import Dict, Any, Optional, Type
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Application error codes."""
    # General errors (1xxx)
    INTERNAL_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    NOT_FOUND = "E1002"
    UNAUTHORIZED = "E1003"
    FORBIDDEN = "E1004"
    RATE_LIMITED = "E1005"
    BAD_REQUEST = "E1006"

    # User errors (2xxx)
    USER_NOT_FOUND = "E2001"
    USER_ALREADY_EXISTS = "E2002"
    INVALID_CREDENTIALS = "E2003"
    PROFILE_INCOMPLETE = "E2004"
    PERSONA_NOT_GENERATED = "E2005"

    # Matching errors (3xxx)
    NO_MATCHES_FOUND = "E3001"
    MATCH_NOT_FOUND = "E3002"
    ALREADY_MATCHED = "E3003"
    MATCH_EXPIRED = "E3004"
    INCOMPATIBLE_MATCH = "E3005"

    # Embedding errors (4xxx)
    EMBEDDING_GENERATION_FAILED = "E4001"
    EMBEDDING_NOT_FOUND = "E4002"
    EMBEDDING_STALE = "E4003"
    VECTOR_DIMENSION_MISMATCH = "E4004"

    # Conversation errors (5xxx)
    CONVERSATION_NOT_FOUND = "E5001"
    MESSAGE_SEND_FAILED = "E5002"
    CONVERSATION_CLOSED = "E5003"

    # External service errors (6xxx)
    DATABASE_ERROR = "E6001"
    CACHE_ERROR = "E6002"
    LLM_SERVICE_ERROR = "E6003"
    EXTERNAL_API_ERROR = "E6004"


# Error code to HTTP status mapping
ERROR_STATUS_MAP = {
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.USER_NOT_FOUND: 404,
    ErrorCode.USER_ALREADY_EXISTS: 409,
    ErrorCode.INVALID_CREDENTIALS: 401,
    ErrorCode.PROFILE_INCOMPLETE: 400,
    ErrorCode.PERSONA_NOT_GENERATED: 400,
    ErrorCode.NO_MATCHES_FOUND: 404,
    ErrorCode.MATCH_NOT_FOUND: 404,
    ErrorCode.ALREADY_MATCHED: 409,
    ErrorCode.MATCH_EXPIRED: 410,
    ErrorCode.INCOMPATIBLE_MATCH: 400,
    ErrorCode.EMBEDDING_GENERATION_FAILED: 500,
    ErrorCode.EMBEDDING_NOT_FOUND: 404,
    ErrorCode.EMBEDDING_STALE: 400,
    ErrorCode.VECTOR_DIMENSION_MISMATCH: 500,
    ErrorCode.CONVERSATION_NOT_FOUND: 404,
    ErrorCode.MESSAGE_SEND_FAILED: 500,
    ErrorCode.CONVERSATION_CLOSED: 400,
    ErrorCode.DATABASE_ERROR: 503,
    ErrorCode.CACHE_ERROR: 503,
    ErrorCode.LLM_SERVICE_ERROR: 503,
    ErrorCode.EXTERNAL_API_ERROR: 502,
}


@dataclass
class ErrorDetail:
    """Detailed error information."""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    field: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ErrorResponse:
    """Standardized error response."""
    error_id: str
    code: str
    message: str
    status_code: int
    timestamp: str
    path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "error": {
                "id": self.error_id,
                "code": self.code,
                "message": self.message,
                "timestamp": self.timestamp
            }
        }
        if self.path:
            result["error"]["path"] = self.path
        if self.details:
            result["error"]["details"] = self.details
        if self.suggestion:
            result["error"]["suggestion"] = self.suggestion
        return result


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.code = code
        self.message = message
        self.details = details
        self.suggestion = suggestion
        self.original_error = original_error
        self.status_code = ERROR_STATUS_MAP.get(code, 500)
        super().__init__(message)


class ValidationException(AppException):
    """Validation error exception."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
            suggestion="Please check your input and try again"
        )


class NotFoundException(AppException):
    """Resource not found exception."""

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        code: ErrorCode = ErrorCode.NOT_FOUND
    ):
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            code=code,
            message=message,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class UserException(AppException):
    """User-related exception."""
    pass


class MatchException(AppException):
    """Match-related exception."""
    pass


class EmbeddingException(AppException):
    """Embedding-related exception."""
    pass


class ConversationException(AppException):
    """Conversation-related exception."""
    pass


class ExternalServiceException(AppException):
    """External service exception."""

    def __init__(
        self,
        service_name: str,
        message: str,
        code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            code=code,
            message=f"{service_name}: {message}",
            details={"service": service_name},
            suggestion="Please try again later",
            original_error=original_error
        )


class ErrorTracker:
    """Tracks errors for monitoring and alerting."""

    def __init__(self):
        self._errors: Dict[str, list] = {}
        self._error_counts: Dict[str, int] = {}
        self.max_stored_errors = int(os.getenv("MAX_STORED_ERRORS", "1000"))

    def track(
        self,
        error_id: str,
        error_code: ErrorCode,
        message: str,
        request_path: Optional[str] = None,
        user_id: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> None:
        """Track an error occurrence."""
        error_record = {
            "error_id": error_id,
            "code": error_code.value,
            "message": message,
            "path": request_path,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "stack_trace": stack_trace
        }

        # Store by code
        code_key = error_code.value
        if code_key not in self._errors:
            self._errors[code_key] = []

        self._errors[code_key].append(error_record)
        if len(self._errors[code_key]) > self.max_stored_errors:
            self._errors[code_key] = self._errors[code_key][-self.max_stored_errors:]

        # Increment count
        self._error_counts[code_key] = self._error_counts.get(code_key, 0) + 1

        logger.error(
            f"Error tracked: {error_id} - {error_code.value}: {message}",
            extra={"error_id": error_id, "error_code": error_code.value}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self._error_counts.values()),
            "by_code": dict(self._error_counts),
            "recent_errors": self._get_recent_errors(10)
        }

    def _get_recent_errors(self, limit: int) -> list:
        """Get most recent errors across all codes."""
        all_errors = []
        for errors in self._errors.values():
            all_errors.extend(errors)
        all_errors.sort(key=lambda e: e["timestamp"], reverse=True)
        return all_errors[:limit]


# Global error tracker
error_tracker = ErrorTracker()


def create_error_response(
    error: AppException,
    request: Optional[Request] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    error_id = str(uuid4())

    # Track the error
    error_tracker.track(
        error_id=error_id,
        error_code=error.code,
        message=error.message,
        request_path=str(request.url) if request else None,
        stack_trace=traceback.format_exc() if error.original_error else None
    )

    return ErrorResponse(
        error_id=error_id,
        code=error.code.value,
        message=error.message,
        status_code=error.status_code,
        timestamp=datetime.utcnow().isoformat(),
        path=str(request.url.path) if request else None,
        details=error.details,
        suggestion=error.suggestion
    )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""

    async def dispatch(self, request: Request, call_next):
        """Handle request with error catching."""
        try:
            response = await call_next(request)
            return response

        except AppException as e:
            # Our custom exceptions
            error_response = create_error_response(e, request)
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.to_dict()
            )

        except HTTPException as e:
            # FastAPI HTTP exceptions
            error_response = ErrorResponse(
                error_id=str(uuid4()),
                code=f"HTTP_{e.status_code}",
                message=e.detail,
                status_code=e.status_code,
                timestamp=datetime.utcnow().isoformat(),
                path=str(request.url.path)
            )
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.to_dict()
            )

        except Exception as e:
            # Unexpected exceptions
            error_id = str(uuid4())
            logger.exception(f"Unexpected error {error_id}: {str(e)}")

            error_tracker.track(
                error_id=error_id,
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
                request_path=str(request.url),
                stack_trace=traceback.format_exc()
            )

            # Don't expose internal details in production
            is_debug = os.getenv("DEBUG", "false").lower() == "true"
            message = str(e) if is_debug else "An internal error occurred"

            error_response = ErrorResponse(
                error_id=error_id,
                code=ErrorCode.INTERNAL_ERROR.value,
                message=message,
                status_code=500,
                timestamp=datetime.utcnow().isoformat(),
                path=str(request.url.path),
                suggestion="Please try again later or contact support"
            )

            return JSONResponse(
                status_code=500,
                content=error_response.to_dict()
            )


def setup_error_handling(app):
    """Setup error handling for FastAPI app."""
    app.add_middleware(ErrorHandlingMiddleware)

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        error_response = create_error_response(exc, request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.to_dict()
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        error_id = str(uuid4())
        logger.exception(f"Unhandled exception {error_id}")

        is_debug = os.getenv("DEBUG", "false").lower() == "true"

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "id": error_id,
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": str(exc) if is_debug else "Internal server error",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )

    logger.info("Error handling middleware configured")
