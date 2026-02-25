"""Persona generation service using OpenAI."""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.adapters.postgresql import postgresql_adapter
from app.prompts.persona_prompts import build_persona_chain, combine_user_data, PERSONA_JSON_SPEC
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class PersonaService:
    """Service for generating personas using AI."""
    
    def __init__(self):
        """Initialize persona service."""
        self.llm_service = LLMService()
    
    def is_available(self) -> bool:
        """Check if persona service is available."""
        return self.llm_service.is_available()
    
    async def generate_persona(self, questions: List[Dict[str, Any]], resume_text: str) -> Optional[Dict[str, Any]]:
        """Generate persona with requirements and offerings from user data."""
        if not self.is_available():
            logger.warning("OpenAI API not available")
            return None
        
        try:
            # Combine the data
            combined_data = combine_user_data(questions, resume_text)
            
            if not combined_data.strip():
                logger.warning("No data provided for persona generation")
                return None
            
            # Log what data we're working with
            if resume_text:
                logger.info(f"Generating persona from questions and resume ({len(resume_text)} chars)")
            else:
                logger.info("Generating persona from questions only (no resume provided)")
            
            # Get LLM and build chain
            llm = self.llm_service.get_chat_model()
            chain = build_persona_chain(llm)
            
            logger.debug(f"Generating persona for data length: {len(combined_data)} characters")
            
            # Generate persona
            result = await chain.ainvoke({
                "combined_data": combined_data,
                "json_schema": PERSONA_JSON_SPEC,
            })
            
            # Extract the three parts from AI response
            persona_data = result.get("persona", {})
            requirements_text = result.get("requirements", "")
            offerings_text = result.get("offerings", "")
            
            # Add metadata to persona
            persona_data["generated_at"] = datetime.utcnow()
            
            # Validate required fields
            # Note: "strategy" is the role-agnostic field (replaces "investment_philosophy")
            persona_required_fields = [
                "name", "archetype", "designation", "experience", "focus", "profile_essence",
                "strategy", "what_theyre_looking_for", "engagement_style"
            ]
            missing_fields = [field for field in persona_required_fields if not persona_data.get(field)]
            if missing_fields or not requirements_text or not offerings_text:
                logger.warning(f"Invalid response - missing fields: {missing_fields}")
                return None
            
            # Return structured response
            full_response = {
                "persona": persona_data,
                "requirements": requirements_text,
                "offerings": offerings_text,
                "generated_at": datetime.utcnow()
            }
            
            logger.info(f"Successfully generated persona: {persona_data['name']}")
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            return None
    
    def generate_persona_sync(self, questions: List[Dict[str, Any]], resume_text: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of generate_persona for Celery workers."""
        if not self.is_available():
            logger.warning("OpenAI API not available")
            return None
        
        try:
            # Combine the data
            combined_data = combine_user_data(questions, resume_text)
            
            if not combined_data.strip():
                logger.warning("No data provided for persona generation")
                return None
            
            # Log what data we're working with
            if resume_text:
                logger.info(f"Generating persona from questions and resume ({len(resume_text)} chars)")
            else:
                logger.info("Generating persona from questions only (no resume provided)")
            
            # Get LLM and build chain
            llm = self.llm_service.get_chat_model()
            chain = build_persona_chain(llm)
            
            logger.debug(f"Generating persona for data length: {len(combined_data)} characters")
            
            # Generate persona synchronously
            result = chain.invoke({
                "combined_data": combined_data,
                "json_schema": PERSONA_JSON_SPEC,
            })
            
            # Extract the three parts from AI response
            persona_data = result.get("persona", {})
            requirements_text = result.get("requirements", "")
            offerings_text = result.get("offerings", "")
            
            # Add metadata to persona
            persona_data["generated_at"] = datetime.utcnow()
            
            # Validate required fields
            # Note: "strategy" is the role-agnostic field (replaces "investment_philosophy")
            persona_required_fields = [
                "name", "archetype", "designation", "experience", "focus", "profile_essence",
                "strategy", "what_theyre_looking_for", "engagement_style"
            ]
            missing_fields = [field for field in persona_required_fields if not persona_data.get(field)]
            if missing_fields or not requirements_text or not offerings_text:
                logger.warning(f"Invalid response - missing fields: {missing_fields}")
                return None
            
            # Return structured response
            full_response = {
                "persona": persona_data,
                "requirements": requirements_text,
                "offerings": offerings_text,
                "generated_at": datetime.utcnow()
            }
            
            logger.info(f"Successfully generated persona: {persona_data['name']}")
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            return None


def update_persona_vector_with_feedback(user_id: str, feedback: str, feedback_type: str, match_context: str = None):
    """Update user's persona embeddings based on feedback via FeedbackLearner."""
    # Import here to avoid circular imports
    from app.services.feedback_learner import feedback_learner

    try:
        # Use the new intelligent feedback learning system
        result = feedback_learner.process_feedback(
            user_id=user_id,
            feedback_text=feedback,
            feedback_type=feedback_type,
            match_context=match_context if isinstance(match_context, dict) else None
        )

        if result.get("success"):
            analysis = result.get("analysis", {})
            logger.info(f"Processed feedback for user {user_id}: sentiment={analysis.get('sentiment')}")
        else:
            logger.warning(f"Failed to process feedback for user {user_id}: {result.get('message')}")

    except Exception as e:
        logger.error(f"Error updating persona vector for user {user_id}: {e}")