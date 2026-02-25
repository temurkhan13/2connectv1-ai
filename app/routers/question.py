"""
Question modification routes.
"""
import logging
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.question import QuestionPayload, QuestionResponse, PreviousUserResponse
from app.services.question_service import QuestionService

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache()
def get_question_service() -> QuestionService:
    """Dependency injection for QuestionService with caching."""
    return QuestionService()


@router.post("/modify-question", response_model=QuestionResponse)
async def modify_question(
    payload: QuestionPayload,
    question_service: QuestionService = Depends(get_question_service)
):
    """
    Modify a question's tone based on conversation context.

    This endpoint modifies questions to have a friendly, engaging tone.
    It can handle both first-time calls (no previous responses) and
    subsequent calls (with conversation context).
    """
    try:
        # Check if this is the first call
        # First call: previous_user_response array is empty OR the last item's user_response is empty/null
        is_first_call = True
        user_message = None
        context = None

        if payload.previous_user_response and len(payload.previous_user_response) > 0:
            # Get the last response
            last_response = payload.previous_user_response[-1]
            user_message = last_response.user_response if last_response.user_response else None

            # If user_response has a value, it's not the first call
            if user_message and user_message.strip():
                is_first_call = False
                # Build context from all previous responses (excluding empty ones)
                context = question_service.build_conversation_context(payload.previous_user_response)

        if is_first_call:
            # First time: just modify the prompt with friendly tone, including options if available
            modified_text = question_service.modify_question_tone(payload.prompt, options=payload.options)
        else:
            # Subsequent call: use user message and conversation context, including options if available
            # Use the user message and current prompt to create natural conversation flow
            modified_text = question_service.modify_question_tone(
                payload.prompt,
                user_message=user_message,
                context=context,
                options=payload.options
            )

        # Generate a follow-up question for suggestion_chips based on the modified ai_text
        try:
            suggestion_chips = question_service.generate_followup_question(modified_text, payload.prompt)
        except Exception as e:
            logger.warning(f"Failed to generate follow-up question: {str(e)}, using original suggestion_chips")
            # If generation fails, fallback to original suggestion_chips from payload
            suggestion_chips = payload.suggestion_chips

        return QuestionResponse(
            question_id=payload.question_id,
            ai_text=modified_text,
            suggestion_chips=suggestion_chips
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

