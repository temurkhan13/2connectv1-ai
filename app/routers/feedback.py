"""
Feedback API endpoints.
Exposes feedback collection, analytics, and learning loop functionality.
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.feedback_loop import feedback_loop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class MatchFeedbackRequest(BaseModel):
    """Request model for submitting match feedback."""
    user_id: str = Field(..., description="User providing feedback")
    match_user_id: str = Field(..., description="User being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    comment: Optional[str] = Field(None, description="Optional comment")
    match_tier: Optional[str] = Field(None, description="Original match tier")
    match_score: Optional[float] = Field(None, description="Original match score")


class OutcomeFeedbackRequest(BaseModel):
    """Request model for connection outcome feedback."""
    user_id: str = Field(..., description="User providing feedback")
    match_user_id: str = Field(..., description="Connected user")
    outcome: str = Field(
        ...,
        description="Connection outcome: successful_deal, ongoing_relationship, "
                    "valuable_conversation, no_response, mutual_pass, negative_experience"
    )
    details: Optional[str] = Field(None, description="Optional details")


class DimensionFeedbackRequest(BaseModel):
    """Request model for dimension-specific feedback."""
    user_id: str = Field(..., description="User providing feedback")
    match_user_id: str = Field(..., description="Matched user")
    dimension_ratings: dict = Field(
        ...,
        description="Ratings per dimension (e.g., {'industry': 4, 'stage': 3})"
    )
    comment: Optional[str] = Field(None, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    feedback_id: str
    applied_learning: bool = False


@router.post("/match", response_model=FeedbackResponse)
async def submit_match_feedback(request: MatchFeedbackRequest):
    """
    Submit feedback on a match quality.

    This feeds into the learning loop to improve future matching.
    Ratings 1-2 (negative) or 4-5 (positive) trigger learning adjustments.
    """
    try:
        logger.info(f"Receiving match feedback from {request.user_id}")

        result = feedback_loop.submit_match_feedback(
            user_id=request.user_id,
            match_user_id=request.match_user_id,
            rating=request.rating,
            comment=request.comment,
            match_tier=request.match_tier,
            match_score=request.match_score
        )

        return FeedbackResponse(
            success=result["success"],
            feedback_id=result["feedback_id"],
            applied_learning=result.get("applied_learning", False)
        )

    except Exception as e:
        logger.error(f"Error submitting match feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.post("/outcome", response_model=FeedbackResponse)
async def submit_outcome_feedback(request: OutcomeFeedbackRequest):
    """
    Submit feedback on connection outcome.

    Called after users have connected to track what happened.
    Positive outcomes (successful_deal, ongoing_relationship) strengthen
    the matching patterns that led to this connection.
    """
    try:
        logger.info(f"Receiving outcome feedback from {request.user_id}")

        result = feedback_loop.submit_outcome_feedback(
            user_id=request.user_id,
            match_user_id=request.match_user_id,
            outcome=request.outcome,
            details=request.details
        )

        return FeedbackResponse(
            success=result["success"],
            feedback_id=result["feedback_id"]
        )

    except Exception as e:
        logger.error(f"Error submitting outcome feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.post("/dimensions")
async def submit_dimension_feedback(request: DimensionFeedbackRequest):
    """
    Submit feedback on specific matching dimensions.

    Allows users to rate how well specific aspects matched:
    - industry: How well industries aligned
    - stage: How well company stages matched
    - geography: How well location preferences matched
    - engagement_style: How well communication styles matched
    """
    try:
        logger.info(f"Receiving dimension feedback from {request.user_id}")

        feedback = feedback_loop.collector.collect_dimension_feedback(
            user_id=request.user_id,
            match_user_id=request.match_user_id,
            dimension_ratings=request.dimension_ratings,
            comment=request.comment
        )

        return {
            "success": True,
            "feedback_id": feedback.feedback_id,
            "dimensions_rated": list(request.dimension_ratings.keys())
        }

    except Exception as e:
        logger.error(f"Error submitting dimension feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/analytics")
async def get_feedback_analytics(days: int = 30):
    """
    Get feedback analytics for the specified period.

    Returns:
    - Overall statistics (total feedback, average rating)
    - Rating and outcome distributions
    - Detected patterns
    - Learning signals for algorithm improvement
    - Performance by match tier
    """
    try:
        logger.info(f"Fetching feedback analytics for {days} days")
        return feedback_loop.get_analytics(days)
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/recommendations")
async def get_improvement_recommendations():
    """
    Get recommendations for matching improvement.

    Analyzes recent feedback to suggest:
    - Weight adjustments for matching dimensions
    - Tier threshold recalibrations
    - Other algorithm improvements

    Recommendations are sorted by priority (high, medium, low).
    """
    try:
        logger.info("Fetching improvement recommendations")
        return feedback_loop.get_improvement_recommendations()
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/close-loop")
async def close_feedback_loop():
    """
    Close the feedback loop by applying accumulated learnings.

    This should be called periodically (e.g., daily by a cron job)
    to apply high-confidence learnings to the matching algorithm.

    Only applies changes with confidence >= 80% and >= 10 samples.
    """
    try:
        logger.info("Closing feedback loop")
        return feedback_loop.close_loop()
    except Exception as e:
        logger.error(f"Error closing feedback loop: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to close loop: {str(e)}")


@router.get("/user/{user_id}")
async def get_user_feedback_history(user_id: str, limit: int = 50):
    """
    Get feedback history for a specific user.

    Returns all feedback submitted by this user.
    """
    try:
        feedback_list = feedback_loop.collector.get_user_feedback(user_id, limit)

        return {
            "user_id": user_id,
            "feedback_count": len(feedback_list),
            "feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "feedback_type": f.feedback_type.value,
                    "rating": f.rating.value if f.rating else None,
                    "outcome": f.outcome.value if f.outcome else None,
                    "timestamp": f.timestamp.isoformat(),
                    "match_user_id": f.match_user_id
                }
                for f in feedback_list
            ]
        }
    except Exception as e:
        logger.error(f"Error getting user feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")
