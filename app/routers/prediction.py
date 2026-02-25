"""
Answer prediction routes.
"""
import logging
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction import PredictAnswerPayload, PredictAnswerResponse, OptionItem
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache()
def get_prediction_service() -> PredictionService:
    """Dependency injection for PredictionService with caching."""
    return PredictionService()


@router.post("/predict-answer", response_model=PredictAnswerResponse)
async def predict_answer(
    payload: PredictAnswerPayload,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict and validate user answer against available options.

    This endpoint:
    - Uses fuzzy matching to find the closest option to user input
    - Returns predicted answer if a match is found
    - Generates a helpful fallback message using LLM if no match is found
    """
    try:
        # Convert Pydantic models to dicts for the service
        options_dict = [
            {
                "label": opt.label,
                "value": opt.value
            }
            for opt in payload.options
        ]

        # Get prediction result
        result = prediction_service.predict_answer(
            payload.user_response,
            options_dict
        )

        return PredictAnswerResponse(
            predicted_answer=result["predicted_answer"],
            valid_answer=result["valid_answer"],
            fallback_text=result["fallback_text"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

