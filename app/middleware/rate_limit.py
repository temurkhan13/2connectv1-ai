"""
Rate limiting middleware using slowapi.
Protects API endpoints from abuse by limiting requests per time window.
"""
import os
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Configuration from environment variables with sensible defaults
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100/minute')
RATE_LIMIT_BURST = os.getenv('RATE_LIMIT_BURST', '200/minute')  # For high-traffic endpoints
RATE_LIMIT_STRICT = os.getenv('RATE_LIMIT_STRICT', '30/minute')  # For expensive operations

# Redis URL for distributed rate limiting (optional)
REDIS_URL = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL'))


def get_api_key_or_ip(request: Request) -> str:
    """
    Get rate limit key from API key header or fallback to IP address.
    This allows per-client rate limiting when API keys are used.
    """
    api_key = request.headers.get('X-API-KEY')
    if api_key:
        # Use first 16 chars of API key as identifier (for privacy)
        return f"apikey:{api_key[:16]}"
    return get_remote_address(request)


def create_limiter() -> Limiter:
    """
    Create a rate limiter instance with appropriate storage backend.
    Uses Redis if available, otherwise falls back to in-memory storage.
    """
    if REDIS_URL and RATE_LIMIT_ENABLED:
        try:
            # Use Redis for distributed rate limiting
            return Limiter(
                key_func=get_api_key_or_ip,
                default_limits=[RATE_LIMIT_DEFAULT],
                storage_uri=REDIS_URL,
                strategy="fixed-window",
                enabled=True
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for rate limiting: {e}. Using in-memory storage.")

    # Fallback to in-memory storage
    return Limiter(
        key_func=get_api_key_or_ip,
        default_limits=[RATE_LIMIT_DEFAULT],
        strategy="fixed-window",
        enabled=RATE_LIMIT_ENABLED
    )


# Create global limiter instance
limiter = create_limiter()


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Custom handler for rate limit exceeded errors.
    Returns a 429 status with helpful error message.
    """
    logger.warning(f"Rate limit exceeded for {get_api_key_or_ip(request)}: {exc.detail}")

    # Extract retry-after from the exception detail if available
    retry_after = getattr(exc, 'retry_after', 60)

    return JSONResponse(
        status_code=429,
        content={
            "code": 429,
            "message": "Rate limit exceeded. Please slow down your requests.",
            "detail": str(exc.detail),
            "retry_after_seconds": retry_after
        },
        headers={"Retry-After": str(retry_after)}
    )


# Rate limit decorators for different endpoint types
def limit_default(func):
    """Apply default rate limit (100/minute)."""
    return limiter.limit(RATE_LIMIT_DEFAULT)(func)


def limit_strict(func):
    """Apply strict rate limit for expensive operations (30/minute)."""
    return limiter.limit(RATE_LIMIT_STRICT)(func)


def limit_burst(func):
    """Apply higher rate limit for high-traffic endpoints (200/minute)."""
    return limiter.limit(RATE_LIMIT_BURST)(func)
