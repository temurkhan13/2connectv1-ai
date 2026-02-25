"""
Health check routes.
"""
from fastapi import APIRouter
from app.schemas.common import HealthResponse
from app.services.llm_slot_extractor import llm_slot_extractor
from app.adapters.postgresql import postgresql_adapter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint at /health."""
    return HealthResponse(
        success=True,
        data={"status": "healthy"},
        message="OK"
    )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check_v1():
    """Health check endpoint at /api/v1/health (for Render)."""
    return HealthResponse(
        success=True,
        data={"status": "healthy"},
        message="OK"
    )


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        success=True,
        data={"message": "Welcome to 2Connect AI API"},
        message="OK"
    )


@router.get("/verify")
async def verify_system():
    """
    Verification endpoint - checks all common issues at once.
    Use this BEFORE testing to ensure fixes are applied.
    """
    checks = {
        "llm_cleanup_function_exists": False,
        "llm_prompt_has_critical_rules": False,
        "postgresql_connection": False,
        "user_count": 0,
        "users_with_embeddings": 0,
    }
    issues = []

    # Check 1: LLM cleanup function exists
    try:
        if hasattr(llm_slot_extractor, '_clean_follow_up_question'):
            checks["llm_cleanup_function_exists"] = True
        else:
            issues.append("_clean_follow_up_question method missing - LLM may say 'The user is...'")
    except Exception as e:
        issues.append(f"LLM extractor check failed: {e}")

    # Check 2: PostgreSQL connection and user count
    try:
        stats = postgresql_adapter.get_embedding_stats()
        checks["postgresql_connection"] = True
        checks["user_count"] = stats.get("total_users", 0)
        checks["users_with_embeddings"] = stats.get("users_with_embeddings", 0)

        if checks["user_count"] == 0:
            issues.append("No users in database - onboarding may not be working")
        if checks["users_with_embeddings"] == 0:
            issues.append("No users with embeddings - matching will return empty")
    except Exception as e:
        issues.append(f"PostgreSQL check failed: {e}")

    # Check 3: Verify prompt has critical rules (check file content)
    try:
        import inspect
        source = inspect.getsource(llm_slot_extractor._build_system_prompt)
        if "CRITICAL: follow_up_question Rules" in source:
            checks["llm_prompt_has_critical_rules"] = True
        else:
            issues.append("LLM prompt missing 'CRITICAL: follow_up_question Rules' - may repeat user info")
    except Exception as e:
        issues.append(f"Prompt check failed: {e}")

    all_ok = len(issues) == 0

    return {
        "status": "ok" if all_ok else "issues_found",
        "checks": checks,
        "issues": issues,
        "recommendation": "All checks passed!" if all_ok else "Fix issues before testing"
    }
