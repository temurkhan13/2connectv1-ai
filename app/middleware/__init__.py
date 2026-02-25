"""
Middleware package for FastAPI application.
"""
from app.middleware.auth import APIKeyMiddleware

__all__ = ['APIKeyMiddleware']

