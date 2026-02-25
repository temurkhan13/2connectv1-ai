"""
FastAPI application main module.
"""
import os
import json
import logging

# Initialize Sentry BEFORE importing anything else (for best error capture)
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
        ],
        traces_sample_rate=0.1,  # 10% of requests traced
        environment=os.getenv('ENVIRONMENT', 'development'),
        send_default_pii=False,
    )
    logging.info("Sentry initialized for error monitoring")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.routers.user import router as user_router
from app.routers.health import router as health_router
from app.routers.matching import router as matching_router
from app.routers.question import router as question_router
from app.routers.prediction import router as prediction_router
from app.routers.onboarding import router as onboarding_router
from app.routers.match import router as match_router
from app.routers.feedback import router as feedback_router
from app.routers.templates import router as templates_router
from app.middleware.auth import APIKeyMiddleware
from app.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from dotenv import load_dotenv

dotenv_override = os.getenv("DOTENV_OVERRIDE", "false").lower() == "true"
load_dotenv(override=dotenv_override)

# Configure logging based on LOG_LEVEL environment variable
log_level = os.getenv('LOG_LEVEL')
if log_level:
    log_level = log_level.upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
else:
    # Default logging configuration if LOG_LEVEL not set
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Get configuration from environment variables
app_name = os.getenv('APP_NAME')
app_version = os.getenv('APP_VERSION')
environment = os.getenv('ENVIRONMENT')

# Parse CORS origins from JSON string
cors_origins_str = os.getenv('CORS_ORIGINS')
try:
    cors_origins = json.loads(cors_origins_str) if cors_origins_str else None
except json.JSONDecodeError:
    cors_origins = None

# Parse allowed hosts from JSON string
allowed_hosts_str = os.getenv('ALLOWED_HOSTS')
try:
    allowed_hosts = json.loads(allowed_hosts_str) if allowed_hosts_str else None
except json.JSONDecodeError:
    allowed_hosts = None

# Validate required environment variables
required_vars = ['APP_NAME', 'APP_VERSION', 'CORS_ORIGINS', 'ALLOWED_HOSTS']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

if not cors_origins:
    raise ValueError("CORS_ORIGINS must be a valid JSON array")
if not allowed_hosts:
    raise ValueError("ALLOWED_HOSTS must be a valid JSON array")

# Log non-sensitive configuration (NEVER log secrets/credentials)
logger = logging.getLogger(__name__)
logger.info(f"Starting {app_name} v{app_version} in {environment} environment")
logger.info(f"CORS origins: {len(cors_origins)} configured")
logger.info(f"AWS region: {os.getenv('AWS_DEFAULT_REGION', 'not set')}")

app = FastAPI(
    title=app_name,
    description="AI-powered reciprocity platform",
    version=app_version,
    swagger_ui_parameters={
        "persistAuthorization": True
    }
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Configure OpenAPI schema to include API key authentication
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app_name,
        version=app_version,
        description="AI-powered reciprocity platform",
        routes=app.routes,
    )
    
    # Add API key security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-KEY"
        }
    }
    
    # Apply security to all endpoints except excluded ones
    excluded_paths = ["/health", "/verify", "/", "/docs", "/redoc", "/openapi.json"]
    for path in openapi_schema["paths"]:
        if not any(path.startswith(excluded) for excluded in excluded_paths):
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-KEY", "Accept"],
)

# Add rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# Add API key authentication middleware
# Excludes health check, verification, docs, and OpenAPI endpoints
app.add_middleware(
    APIKeyMiddleware,
    exclude_paths=["/health", "/verify", "/", "/docs", "/redoc", "/openapi.json"]
)

app.include_router(health_router)
app.include_router(user_router, prefix="/api/v1")
app.include_router(matching_router, prefix="/api/v1")
app.include_router(question_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(onboarding_router, prefix="/api/v1")
# Match router with /api/v1 prefix (backend's AI_SERVICE_URL includes /api/v1)
app.include_router(match_router, prefix="/api/v1")
# Feedback router for learning loop
app.include_router(feedback_router, prefix="/api/v1")
# Templates router for use case templates
app.include_router(templates_router, prefix="/api/v1")
