"""
Structured Logging Configuration.

Provides consistent JSON-structured logging with request correlation,
performance metrics, and context preservation.

Key features:
1. JSON-formatted logs for aggregation
2. Request correlation IDs
3. Performance timing
4. Context injection
5. Log level management
"""
import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from contextvars import ContextVar
from functools import wraps

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Context variables for request-scoped data
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
extra_context_var: ContextVar[Dict[str, Any]] = ContextVar("extra_context", default={})


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log formatter."""

    def __init__(self, include_stack: bool = False):
        super().__init__()
        self.include_stack = include_stack
        self.service_name = os.getenv("SERVICE_NAME", "reciprocity-api")
        self.environment = os.getenv("ENVIRONMENT", "development")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment
        }

        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id

        # Add extra context
        extra_context = extra_context_var.get()
        if extra_context:
            log_entry["context"] = extra_context

        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None
            }
            if self.include_stack:
                import traceback
                log_entry["exception"]["stack_trace"] = traceback.format_exception(*record.exc_info)

        # Add any extra fields from the record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that includes context in all messages."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log message."""
        extra = kwargs.get("extra", {})

        # Add request context
        extra["request_id"] = request_id_var.get()
        extra["user_id"] = user_id_var.get()

        kwargs["extra"] = extra
        return msg, kwargs


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    def __init__(self, app, logger_name: str = "api"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        """Log request and response with timing."""
        # Generate request ID
        request_id = str(uuid4())
        request_id_var.set(request_id)

        # Extract user ID if available
        user_id = request.headers.get("X-User-ID", "")
        user_id_var.set(user_id)

        # Start timing
        start_time = time.perf_counter()

        # Log request
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "extra_fields": {
                    "event": "request_started",
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            }
        )

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "extra_fields": {
                        "event": "request_completed",
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2)
                    }
                }
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "extra_fields": {
                        "event": "request_failed",
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(duration_ms, 2),
                        "error": str(e)
                    }
                },
                exc_info=True
            )
            raise


def setup_logging(
    level: str = None,
    json_format: bool = True,
    include_stack: bool = False
) -> None:
    """
    Setup application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting
        include_stack: Include stack traces in JSON
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Set formatter
    if json_format and os.getenv("ENVIRONMENT", "development") != "development":
        handler.setFormatter(StructuredFormatter(include_stack=include_stack))
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.info(f"Logging configured: level={level}, json_format={json_format}")


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger."""
    return ContextLogger(logging.getLogger(name), {})


def log_performance(operation_name: str = None):
    """Decorator to log function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = logging.getLogger(func.__module__)

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        "extra_fields": {
                            "event": "operation_completed",
                            "operation": op_name,
                            "duration_ms": round(duration_ms, 2)
                        }
                    }
                )
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Operation failed: {op_name}",
                    extra={
                        "extra_fields": {
                            "event": "operation_failed",
                            "operation": op_name,
                            "duration_ms": round(duration_ms, 2),
                            "error": str(e)
                        }
                    }
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = logging.getLogger(func.__module__)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        "extra_fields": {
                            "event": "operation_completed",
                            "operation": op_name,
                            "duration_ms": round(duration_ms, 2)
                        }
                    }
                )
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Operation failed: {op_name}",
                    extra={
                        "extra_fields": {
                            "event": "operation_failed",
                            "operation": op_name,
                            "duration_ms": round(duration_ms, 2),
                            "error": str(e)
                        }
                    }
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def set_log_context(**kwargs) -> None:
    """Set additional context for current request."""
    current = extra_context_var.get()
    extra_context_var.set({**current, **kwargs})


def clear_log_context() -> None:
    """Clear the log context."""
    extra_context_var.set({})
    request_id_var.set("")
    user_id_var.set("")


class LogContext:
    """Context manager for scoped logging context."""

    def __init__(self, **kwargs):
        self.context = kwargs
        self.previous_context = {}

    def __enter__(self):
        self.previous_context = extra_context_var.get()
        extra_context_var.set({**self.previous_context, **self.context})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        extra_context_var.set(self.previous_context)
        return False
