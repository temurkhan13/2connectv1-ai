"""
LLM-based Question Generator Service.

DEDICATED to generating personalized follow-up questions.
This is SEPARATE from slot extraction - clean separation of concerns.

BUG-071 FIX: Split extraction and question generation into separate services.
- Extraction Sonnet: ONLY extracts slots from user message
- Question Sonnet: ONLY generates personalized follow-up questions

This allows:
1. Independent prompt tuning for each task
2. Future parallelization (run both concurrently)
3. Different model choices per task if needed
"""
import os
import logging
from typing import Dict, List, Optional, Any
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMQuestionGenerator:
    """
    Dedicated service for generating personalized follow-up questions.

    Uses Claude Sonnet 4.5 - separate instance from slot extraction.
    Single responsibility: Generate engaging, contextual follow-up questions.
    """

    def __init__(self):
        # Lazy-load client to ensure API key is read at runtime
        self._client = None
        # Dedicated model for question generation (can be tuned independently)
        self.question_model = os.getenv("ANTHROPIC_QUESTION_MODEL", "claude-sonnet-4-5-20250929")
        # Session-specific pattern memory (prevents repetitive questions)
        self._session_patterns = {}

    @property
    def client(self) -> Anthropic:
        """Lazy-load Anthropic client."""
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            logger.info(f"[QuestionGenerator] Initializing Anthropic client")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def generate_followup_question(
        self,
        user_message: str,
        extracted_slots: Dict[str, Any],
        all_filled_slots: Dict[str, Any],
        missing_slots: List[str],
        user_type: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """
        Generate a personalized follow-up question based on context.

        Args:
            user_message: What the user just said
            extracted_slots: Slots extracted from THIS message
            all_filled_slots: ALL slots filled so far in session
            missing_slots: Slots we still need to collect
            user_type: Inferred user type (founder, investor, etc.)
            session_id: Session ID for pattern tracking
            conversation_history: Full conversation to avoid repeating questions

        Returns:
            Personalized follow-up question, or None on error
        """
        try:
            # Initialize session patterns if needed
            if session_id and session_id not in self._session_patterns:
                self._session_patterns[session_id] = {
                    'asked_questions': [],
                    'topic_counts': {},
                    'last_topic': None
                }
                logger.info(f"[QuestionGenerator] Initialized patterns for session {session_id[:8]}...")

            # Build context summary
            extracted_summary = ", ".join([
                f"{k}={v.get('value', v) if isinstance(v, dict) else v}"
                for k, v in extracted_slots.items()
            ]) if extracted_slots else "none this turn"

            filled_summary = ", ".join([
                f"{k}={v.get('value', v) if isinstance(v, dict) else v}"
                for k, v in all_filled_slots.items()
            ]) if all_filled_slots else "none yet"

            # Focus on top 3 missing slots
            priority_missing = missing_slots[:3] if missing_slots else []

            # Extract previous AI questions to prevent duplicates
            previous_questions = []
            if conversation_history:
                for turn in conversation_history:
                    if turn.get("role") == "assistant":
                        content = turn.get("content", "").strip()
                        if content and "?" in content:
                            previous_questions.append(content)

            # Get session patterns
            asked_before = []
            if session_id and session_id in self._session_patterns:
                asked_before = self._session_patterns[session_id].get('asked_questions', [])[-5:]

            # Build the prompt
            prompt = f"""You are a warm, professional onboarding assistant for a business networking platform.

## YOUR ONLY JOB
Generate ONE follow-up question to learn more about the user.

## CONTEXT
User just said: "{user_message}"
User type: {user_type}
Just extracted: {extracted_summary}
Already know: {filled_summary}
Still need: {', '.join(priority_missing) if priority_missing else 'nothing critical'}

## QUESTIONS ALREADY ASKED (NEVER REPEAT)
{chr(10).join([f'- "{q[:100]}..."' for q in previous_questions[-3:]]) if previous_questions else 'None yet'}

## RULES
1. Ask about ONE of the missing slots: {priority_missing}
2. Reference something SPECIFIC from their message (not generic acknowledgment)
3. Be conversational, not form-like
4. NEVER repeat a question already asked
5. Keep it SHORT (1-2 sentences max)

## WRONG EXAMPLES
- "That's interesting! What industries are you focused on?" (too generic)
- "Got it. Can you tell me more about your goals?" (robotic)
- "Thanks for sharing. What's your budget?" (abrupt topic change)

## GOOD EXAMPLES
- "A B2B SaaS for healthcare sounds promising - are you looking for investors who specialize in healthtech, or more generalist funds?"
- "Series A with strong MRR is a great position. What's driving your decision to raise now vs continue bootstrapping?"

Return ONLY the question. No preamble, no explanation."""

            logger.info(f"[QuestionGenerator] Generating question for session {session_id[:8] if session_id else 'unknown'}...")

            response = self.client.messages.create(
                model=self.question_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7  # Higher temperature for variety
            )

            question = response.content[0].text.strip()

            # Clean up any quotes if the model wrapped the question
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]

            # Track this question to avoid repeating
            if session_id and session_id in self._session_patterns:
                self._session_patterns[session_id]['asked_questions'].append(question)

            logger.info(f"[QuestionGenerator] Generated: {question[:80]}...")
            return question

        except Exception as e:
            logger.error(f"[QuestionGenerator] Failed to generate question: {e}")
            # Return a safe fallback
            if missing_slots:
                return f"Could you tell me more about your {missing_slots[0].replace('_', ' ')}?"
            return "What else would be helpful for me to know about what you're looking for?"

    def clear_session_patterns(self, session_id: str) -> None:
        """Clear patterns for a completed session."""
        if session_id in self._session_patterns:
            del self._session_patterns[session_id]
            logger.info(f"[QuestionGenerator] Cleared patterns for session {session_id[:8]}...")


# Singleton instance
_question_generator_instance = None

def get_question_generator() -> LLMQuestionGenerator:
    """Get singleton instance of question generator."""
    global _question_generator_instance
    if _question_generator_instance is None:
        _question_generator_instance = LLMQuestionGenerator()
    return _question_generator_instance
