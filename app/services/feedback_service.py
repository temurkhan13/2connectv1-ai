"""Feedback processing service.

This service handles feedback from the legacy /user/feedback endpoint.
For structured feedback with reasons, use /user/feedback-with-reasons endpoint.

Note: Feedback storage is handled by the backend (match_feedback table).
This service focuses only on embedding/persona adjustment via feedback_learner.
"""
import logging
from typing import Dict, Any

from app.schemas.user import FeedbackRequest
from app.services.feedback_learner import feedback_learner

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for processing user feedback on matches/chats."""

    def process_feedback(self, data: FeedbackRequest) -> Dict[str, Any]:
        """
        Process user feedback and update persona embeddings.

        Note: Feedback is NOT stored in DynamoDB anymore.
        The backend stores feedback in match_feedback table.
        This service only handles the AI learning/embedding adjustment.
        """
        try:
            logger.info(f"Processing feedback for user {data.user_id}, type: {data.type}")

            # Prepare context
            context = {
                "feedback_type": data.type,
                "target_id": data.id
            }

            # Process through feedback learner
            result = feedback_learner.process_feedback(
                user_id=data.user_id,
                feedback_text=data.feedback,
                feedback_type=data.type,
                match_context=context
            )

            if result.get("success"):
                logger.info(f"Feedback processed successfully for user {data.user_id}")
                return {
                    "success": True,
                    "message": "Feedback processed and persona updated",
                    "analysis": result.get("analysis", {})
                }
            else:
                logger.warning(f"Feedback processing returned failure: {result.get('message')}")
                return {
                    "success": True,  # Return success anyway to not block UI
                    "message": "Feedback received but embedding update skipped",
                    "reason": result.get("message")
                }

        except Exception as e:
            logger.error(f"Error processing feedback for user {data.user_id}: {e}")
            # Return success anyway (fire-and-forget pattern)
            return {
                "success": True,
                "message": "Feedback received",
                "error": str(e)
            }
