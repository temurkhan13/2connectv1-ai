"""Persona generation service.

Apr-22 Phase 4 prompt caching:
Switched from LangChain PromptTemplate → RobustJsonOutputParser chain to direct
call_with_fallback invocation with cache_control on the stable system block.
PERSONA_SYSTEM_TEMPLATE (with {json_schema} filled) is byte-identical across all
persona calls site-wide — ONE cached prefix shared across all users. Only combined_data
varies per call, living in the user message (uncached).

Both sync and async methods use the same split; the async path wraps call_with_fallback
in asyncio.to_thread for compatibility with async callers (the Celery worker uses sync).

RobustJsonOutputParser retained for response parsing — preserves the Apr BUG-015 /
BUG-027 fallback behavior (extract JSON from markdown, degrade to _generate_fallback_persona
on complete parse failure).
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.adapters.postgresql import postgresql_adapter
from app.prompts.persona_prompts import (
    combine_user_data,
    PERSONA_JSON_SPEC,
    PERSONA_SYSTEM_TEMPLATE,
    PERSONA_USER_TEMPLATE,
    RobustJsonOutputParser,
)
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _build_cached_prefix_and_user_message(combined_data: str) -> tuple[list, str]:
    """Build the split system/user content for a persona generation call.

    Returns:
        (cached_system_blocks, user_message_text) ready to pass to call_with_fallback.

    The system block has cache_control: ephemeral. The cached prefix is
    PERSONA_SYSTEM_TEMPLATE with {json_schema} filled in — byte-identical across
    all users site-wide. The user message contains only combined_data + a short
    output-format reminder (preserves prompt recency effect on JSON output).
    """
    # str(PERSONA_JSON_SPEC) matches the original LangChain rendering behavior exactly.
    # The spec is a module-level dict with deterministic insertion order; str() produces
    # byte-identical output on every call, preserving cache key stability.
    system_text = PERSONA_SYSTEM_TEMPLATE.format(json_schema=str(PERSONA_JSON_SPEC))
    user_text = PERSONA_USER_TEMPLATE.format(combined_data=combined_data)

    cached_system = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return cached_system, user_text


def _validate_and_structure_persona(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate persona result and wrap with generated_at metadata.

    Shared between async and sync paths. Returns None if required fields are missing
    (matches previous behavior).
    """
    persona_data = result.get("persona", {})
    requirements_text = result.get("requirements", "")
    offerings_text = result.get("offerings", "")

    # Note: "strategy" is the role-agnostic field (replaces "investment_philosophy")
    persona_required_fields = [
        "name", "archetype", "designation", "experience", "focus", "profile_essence",
        "strategy", "what_theyre_looking_for", "engagement_style"
    ]
    missing_fields = [f for f in persona_required_fields if not persona_data.get(f)]
    if missing_fields or not requirements_text or not offerings_text:
        logger.warning(f"Invalid response - missing fields: {missing_fields}")
        return None

    persona_data["generated_at"] = datetime.utcnow()
    return {
        "persona": persona_data,
        "requirements": requirements_text,
        "offerings": offerings_text,
        "generated_at": datetime.utcnow(),
    }


class PersonaService:
    """Service for generating personas using AI."""

    def __init__(self):
        """Initialize persona service."""
        self.llm_service = LLMService()

    def is_available(self) -> bool:
        """Check if persona service is available."""
        return self.llm_service.is_available()

    async def generate_persona(self, questions: List[Dict[str, Any]], resume_text: str, conversation_text: str = "") -> Optional[Dict[str, Any]]:
        """Generate persona with requirements and offerings from user data (async variant)."""
        if not self.is_available():
            logger.warning("Anthropic API not available")
            return None

        try:
            from app.services.llm_fallback import call_with_fallback

            combined_data = combine_user_data(questions, resume_text, conversation_text)
            if not combined_data.strip():
                logger.warning("No data provided for persona generation")
                return None

            if resume_text:
                logger.info(f"Generating persona from questions and resume ({len(resume_text)} chars)")
            else:
                logger.info("Generating persona from questions only (no resume provided)")
            logger.debug(f"Generating persona for data length: {len(combined_data)} characters")

            cached_system, user_text = _build_cached_prefix_and_user_message(combined_data)

            # Run sync wrapper in a thread so the async API is preserved for callers.
            # [LLM Cache] observability line in llm_fallback.call_with_fallback logs
            # cache_creation / cache_read token counts per call.
            content = await asyncio.to_thread(
                call_with_fallback,
                service="matching",
                system_prompt=cached_system,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=4096,
                temperature=self.llm_service.temperature,
            )

            parser = RobustJsonOutputParser()
            result = parser.parse(content or "")

            structured = _validate_and_structure_persona(result)
            if structured:
                logger.info(f"Successfully generated persona: {structured['persona']['name']}")
            return structured

        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            return None

    def generate_persona_sync(self, questions: List[Dict[str, Any]], resume_text: str, conversation_text: str = "") -> Optional[Dict[str, Any]]:
        """Synchronous version of generate_persona for Celery workers (the primary production path)."""
        if not self.is_available():
            logger.warning("Anthropic API not available")
            return None

        try:
            from app.services.llm_fallback import call_with_fallback

            combined_data = combine_user_data(questions, resume_text, conversation_text)
            if not combined_data.strip():
                logger.warning("No data provided for persona generation")
                return None

            if resume_text:
                logger.info(f"Generating persona from questions and resume ({len(resume_text)} chars)")
            else:
                logger.info("Generating persona from questions only (no resume provided)")
            logger.debug(f"Generating persona for data length: {len(combined_data)} characters")

            cached_system, user_text = _build_cached_prefix_and_user_message(combined_data)

            content = call_with_fallback(
                service="matching",
                system_prompt=cached_system,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=4096,
                temperature=self.llm_service.temperature,
            )

            parser = RobustJsonOutputParser()
            result = parser.parse(content or "")

            structured = _validate_and_structure_persona(result)
            if structured:
                logger.info(f"Successfully generated persona: {structured['persona']['name']}")
            return structured

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