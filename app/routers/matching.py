"""
Production matching API endpoints.
Uses SentenceTransformers + pgvector for all user matching.
"""
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from app.services.matching_service import matching_service, MatchResult, UserMatchesResponse
from app.services.multi_vector_matcher import MultiVectorMatcher, MatchTier
from app.adapters.dynamodb import UserProfile
from app.adapters.postgresql import postgresql_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matching", tags=["User Matching"])

# Timeout for DynamoDB operations (seconds)
DYNAMODB_TIMEOUT = 5

# Thread pool for running sync operations with timeout
_executor = ThreadPoolExecutor(max_workers=4)


# Helper Functions
def get_default_similarity_threshold() -> float:
    """Get the default similarity threshold from environment variable."""
    return float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))


def sanitize_threshold(threshold: Optional[float]) -> Optional[float]:
    """Sanitize similarity threshold, handling NaN and invalid values."""
    import math
    if threshold is None:
        return None
    # Handle NaN (sent by some backend implementations)
    if math.isnan(threshold):
        logger.warning("Received NaN similarity_threshold, using default")
        return None
    # Clamp to valid range
    if threshold < 0.0:
        return 0.0
    if threshold > 1.0:
        return 1.0
    return threshold


# Validation Helpers
def _sync_validate_user_profile(user_id: str) -> UserProfile:
    """Synchronous DynamoDB validation (called from thread pool)."""
    user_profile = UserProfile.get(user_id)
    if user_profile.persona_status != 'completed':
        raise ValueError(f"User persona not completed. Status: {user_profile.persona_status}")
    return user_profile


async def validate_user_profile(user_id: str, skip_dynamo: bool = False) -> Optional[UserProfile]:
    """
    Validate that user exists and has completed persona.

    Uses asyncio timeout to prevent hanging on DynamoDB issues.
    Falls back to PostgreSQL-only matching if DynamoDB times out.

    Args:
        user_id: The user ID to validate
        skip_dynamo: If True, skip DynamoDB validation entirely

    Returns:
        UserProfile: The validated user profile, or None if skipping

    Raises:
        HTTPException: If user not found or persona not completed
    """
    if skip_dynamo:
        # Check if user has embeddings in PostgreSQL (sufficient for matching)
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not embeddings.get('requirements') and not embeddings.get('offerings'):
            raise HTTPException(status_code=404, detail="User not found or has no embeddings")
        return None

    try:
        loop = asyncio.get_event_loop()
        # Run DynamoDB call in thread pool with timeout
        user_profile = await asyncio.wait_for(
            loop.run_in_executor(_executor, _sync_validate_user_profile, user_id),
            timeout=DYNAMODB_TIMEOUT
        )
        return user_profile
    except asyncio.TimeoutError:
        logger.warning(f"DynamoDB timeout for user {user_id}, falling back to PostgreSQL-only")
        # Fall back to PostgreSQL check
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not embeddings.get('requirements') and not embeddings.get('offerings'):
            raise HTTPException(status_code=404, detail="User not found or has no embeddings")
        return None
    except UserProfile.DoesNotExist:
        raise HTTPException(status_code=404, detail="User not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"DynamoDB error for user {user_id}: {e}, falling back to PostgreSQL-only")
        # Fall back to PostgreSQL check
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not embeddings.get('requirements') and not embeddings.get('offerings'):
            raise HTTPException(status_code=404, detail="User not found or has no embeddings")
        return None


def validate_user_requirements(user_profile: Optional[UserProfile], user_id: str) -> None:
    """
    Validate that user has requirements defined.

    Args:
        user_profile: The user profile to validate (can be None if skipped)
        user_id: The user ID for fallback check

    Raises:
        HTTPException: If user has no requirements
    """
    if user_profile:
        persona = user_profile.persona
        if not persona or not persona.requirements:
            raise HTTPException(
                status_code=400,
                detail="User does not have requirements defined"
            )
    else:
        # Fall back to PostgreSQL check
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not embeddings.get('requirements'):
            raise HTTPException(
                status_code=400,
                detail="User does not have requirements defined"
            )


def validate_user_offerings(user_profile: Optional[UserProfile], user_id: str) -> None:
    """
    Validate that user has offerings defined.

    Args:
        user_profile: The user profile to validate (can be None if skipped)
        user_id: The user ID for fallback check

    Raises:
        HTTPException: If user has no offerings
    """
    if user_profile:
        persona = user_profile.persona
        if not persona or not persona.offerings:
            raise HTTPException(
                status_code=400,
                detail="User does not have offerings defined"
            )
    else:
        # Fall back to PostgreSQL check
        embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not embeddings.get('offerings'):
            raise HTTPException(
                status_code=400,
                detail="User does not have offerings defined"
            )


# API Endpoints

@router.get("/{user_id}/matches", response_model=UserMatchesResponse)
async def get_user_matches(
    user_id: str,
    similarity_threshold: Optional[float] = Query(None, description="Minimum similarity score (0.0 to 1.0). Uses SIMILARITY_THRESHOLD env var if not specified."),
    skip_validation: bool = Query(False, description="Skip DynamoDB validation (use for faster queries)")
):
    """
    Get all matches for a user.

    Returns both requirements matches and offerings matches with explanations.
    Uses asyncio timeout to prevent hanging on DynamoDB issues.
    """
    try:
        # Validate user profile with timeout (can fall back to PostgreSQL-only)
        await validate_user_profile(user_id, skip_dynamo=skip_validation)

        # Sanitize and use environment default if threshold not specified or NaN
        sanitized = sanitize_threshold(similarity_threshold)
        threshold = sanitized if sanitized is not None else get_default_similarity_threshold()

        return matching_service.get_all_user_matches(user_id, threshold)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding matches for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{user_id}/requirements-matches")
async def get_user_requirements_matches(
    user_id: str,
    similarity_threshold: Optional[float] = Query(None, description="Minimum similarity score (0.0 to 1.0). Uses SIMILARITY_THRESHOLD env var if not specified."),
    skip_validation: bool = Query(False, description="Skip DynamoDB validation (use for faster queries)")
):
    """
    Get matches for user's REQUIREMENTS (what user needs vs others' OFFERINGS).

    Returns users whose offerings match what this user is looking for.
    """
    try:
        # Validate user profile and requirements with timeout
        user_profile = await validate_user_profile(user_id, skip_dynamo=skip_validation)
        validate_user_requirements(user_profile, user_id)

        # Sanitize and use environment default if threshold not specified or NaN
        sanitized = sanitize_threshold(similarity_threshold)
        threshold = sanitized if sanitized is not None else get_default_similarity_threshold()

        return matching_service.get_requirements_matches(user_id, threshold)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding requirements matches for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{user_id}/offerings-matches")
async def get_user_offerings_matches(
    user_id: str,
    similarity_threshold: Optional[float] = Query(None, description="Minimum similarity score (0.0 to 1.0). Uses SIMILARITY_THRESHOLD env var if not specified."),
    skip_validation: bool = Query(False, description="Skip DynamoDB validation (use for faster queries)")
):
    """
    Get matches for user's OFFERINGS (what user offers vs others' REQUIREMENTS).

    Returns users who need what this user is offering.
    """
    try:
        # Validate user profile and offerings with timeout
        user_profile = await validate_user_profile(user_id, skip_dynamo=skip_validation)
        validate_user_offerings(user_profile, user_id)

        # Sanitize and use environment default if threshold not specified or NaN
        sanitized = sanitize_threshold(similarity_threshold)
        threshold = sanitized if sanitized is not None else get_default_similarity_threshold()

        return matching_service.get_offerings_matches(user_id, threshold)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding offerings matches for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stats")
async def get_matching_stats():
    """Get matching system statistics."""
    try:
        return matching_service.get_matching_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# Multi-Vector Matching Endpoints

class MultiVectorMatchResponse(BaseModel):
    """Response model for multi-vector match results."""
    user_id: str
    total_score: float
    tier: str
    dimension_scores: List[Dict[str, Any]]
    explanation: Optional[str] = None


@router.get("/{user_id}/multi-vector-matches", response_model=List[MultiVectorMatchResponse])
async def get_multi_vector_matches(
    user_id: str,
    min_tier: str = Query("worth_exploring", description="Minimum match tier: perfect, strong, worth_exploring, low"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of matches to return"),
    skip_validation: bool = Query(False, description="Skip DynamoDB validation")
):
    """
    Get matches using multi-vector weighted similarity.

    Multi-vector matching considers 6 separate dimensions:
    - primary_goal (20%): User's main objective
    - industry (25%): Industry/sector focus
    - stage (20%): Investment/company stage
    - geography (15%): Geographic preferences
    - engagement_style (10%): Communication preferences
    - dealbreakers (10%): Hard exclusion criteria

    Returns matches sorted by weighted total score.
    """
    try:
        await validate_user_profile(user_id, skip_dynamo=skip_validation)

        # Parse tier
        try:
            tier_enum = MatchTier(min_tier.lower())
        except ValueError:
            tier_enum = MatchTier.WORTH_EXPLORING

        # Get multi-vector matches
        matcher = MultiVectorMatcher()
        matches = matcher.find_multi_vector_matches(
            user_id=user_id,
            min_tier=tier_enum,
            limit=limit
        )

        return [
            MultiVectorMatchResponse(
                user_id=m.user_id,
                total_score=m.total_score,
                tier=m.tier.value,
                dimension_scores=[
                    {
                        "dimension": ds.dimension,
                        "similarity": ds.similarity,
                        "weight": ds.weight,
                        "weighted_score": ds.weighted_score,
                        "matched": ds.matched
                    }
                    for ds in m.dimension_scores
                ],
                explanation=m.explanation
            )
            for m in matches
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding multi-vector matches for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
