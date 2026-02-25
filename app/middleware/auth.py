"""
Authentication middleware for API key validation.
"""
import os
import hmac
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Optional

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate X-API-KEY header for incoming requests."""

    def __init__(self, app, exclude_paths: Optional[list] = None):
        """
        Initialize API key middleware.

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from API key validation (e.g., ["/health", "/docs"])
        """
        super().__init__(app)
        self.api_key = os.getenv('API_KEY')
        self.environment = os.getenv('ENVIRONMENT', 'development').lower()
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/redoc", "/openapi.json"]

        # SECURITY: In production, API_KEY is REQUIRED
        if self.environment == 'production' and not self.api_key:
            raise ValueError("API_KEY environment variable is REQUIRED in production")

    def _should_bypass_auth(self) -> bool:
        """Allow explicit auth bypass for non-production environments (e.g., tests)."""
        bypass = os.getenv("AUTH_BYPASS", "").lower() == "true"
        if not bypass:
            return False
        # Never allow bypass in production
        return self.environment != "production"

    async def dispatch(self, request: Request, call_next):
        """
        Validate API key for incoming requests.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from next handler or 401/403 error
        """
        # Skip validation for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Allow explicit bypass in non-production environments
        if self._should_bypass_auth():
            return await call_next(request)

        # SECURITY: In production, always require API key
        # In development, allow requests if no API_KEY is configured (for local testing)
        if not self.api_key:
            if self.environment == 'production':
                # This should never happen due to __init__ check, but defense in depth
                logger.error("API_KEY not configured in production - blocking request")
                return JSONResponse(
                    status_code=500,
                    content={
                        "code": 500,
                        "message": "Server misconfiguration: authentication not properly configured",
                        "result": False
                    }
                )
            # Development only: allow without auth
            logger.warning("No API_KEY configured - allowing request (development mode only)")
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("X-API-KEY")
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "code": 401,
                    "message": "X-API-KEY header is required",
                    "result": False
                }
            )
        
        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(api_key, self.api_key):
            return JSONResponse(
                status_code=403,
                content={
                    "code": 403,
                    "message": "Invalid API key",
                    "result": False
                }
            )
        
        # API key is valid, proceed with request
        return await call_next(request)

