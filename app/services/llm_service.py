"""
LLM service for handling OpenAI API interactions.
"""
from typing import Optional, Dict, Any, List
import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)  # Override shell env vars with .env values

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM interactions using OpenAI."""

    def __init__(self):
        """Initialize LLM service."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.model:
            raise ValueError("OPENAI_MODEL environment variable is required")

    def get_chat_model(self) -> ChatOpenAI:
        """Get OpenAI chat model instance."""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return bool(self.api_key and self.model)

    async def generate_match_explanation(
        self,
        user_a: Dict[str, Any],
        user_b: Dict[str, Any],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered match explanation from two user profiles.

        Args:
            user_a: First user's profile data
            user_b: Second user's profile data
            scores: Pre-computed alignment scores

        Returns:
            Dict with summary, synergy_areas, friction_points, talking_points
        """
        chat_model = self.get_chat_model()

        system_prompt = """You are an AI networking assistant that explains why two professionals are a good match.
Your goal is to provide insightful, specific analysis of how these two people can benefit from connecting.
Be concise, professional, and focus on actionable insights.
Always respond in valid JSON format."""

        user_prompt = f"""Analyze why these two users are matched and generate a helpful explanation.

USER A:
- Name: {user_a.get('name', 'Unknown')}
- Role: {user_a.get('user_type', 'Professional')}
- Industry: {user_a.get('industry', 'General')}
- Requirements (what they need): {user_a.get('requirements', 'Not specified')}
- Offerings (what they provide): {user_a.get('offerings', 'Not specified')}

USER B:
- Name: {user_b.get('name', 'Unknown')}
- Role: {user_b.get('user_type', 'Professional')}
- Industry: {user_b.get('industry', 'General')}
- Requirements (what they need): {user_b.get('requirements', 'Not specified')}
- Offerings (what they provide): {user_b.get('offerings', 'Not specified')}

ALIGNMENT SCORES:
- Requirements alignment: {scores.get('req_to_off', 0.5):.0%}
- Offerings alignment: {scores.get('off_to_req', 0.5):.0%}
- Industry match: {scores.get('industry_match', 0.5):.0%}

Respond with a JSON object containing:
{{
    "summary": "2-3 sentence explanation of the match value (mention specific shared interests/goals)",
    "synergy_areas": ["3-4 specific areas where they can collaborate or help each other"],
    "friction_points": ["1-2 potential challenges or gaps to be aware of"],
    "talking_points": ["3-4 specific conversation topics they could discuss"]
}}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await chat_model.ainvoke(messages)
            content = response.content.strip()

            # Parse JSON response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())

            # Validate response structure
            return {
                "summary": result.get("summary", "This match shows strong potential for collaboration."),
                "synergy_areas": result.get("synergy_areas", ["Complementary professional backgrounds"]),
                "friction_points": result.get("friction_points", ["No significant friction points identified"]),
                "talking_points": result.get("talking_points", ["Discuss potential collaboration opportunities"])
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return template fallback
            return self._fallback_explanation(user_a, user_b)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_explanation(user_a, user_b)

    def _fallback_explanation(self, user_a: Dict, user_b: Dict) -> Dict[str, Any]:
        """Fallback template-based explanation when LLM fails."""
        return {
            "summary": f"This match connects {user_a.get('user_type', 'a professional')} with {user_b.get('user_type', 'a professional')}. There's potential for mutual value exchange based on aligned interests.",
            "synergy_areas": [
                f"Shared industry focus: {user_a.get('industry', 'General')}",
                "Complementary professional backgrounds"
            ],
            "friction_points": ["Limited information to assess challenges"],
            "talking_points": [
                "Discuss potential collaboration opportunities",
                "Explore mutual connections or interests"
            ]
        }

    async def generate_ice_breakers(
        self,
        requesting_user: Dict[str, Any],
        other_user: Dict[str, Any],
        match_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate AI-powered conversation starters for a match.

        Args:
            requesting_user: The user who wants conversation starters
            other_user: The user they want to message
            match_context: Optional context about the match

        Returns:
            List of personalized ice breaker messages
        """
        chat_model = self.get_chat_model()

        system_prompt = """You are an AI networking assistant that generates personalized conversation starters.
Create messages that are warm, professional, and show genuine interest.
Each message should be different in approach - some direct, some question-based, some complimentary.
Keep messages under 100 words each. Sound natural, not robotic.
Always respond in valid JSON format."""

        user_prompt = f"""Generate 4 personalized first message options for starting a conversation.

SENDER:
- Name: {requesting_user.get('name', 'Unknown')}
- Role: {requesting_user.get('user_type', 'Professional')}
- Industry: {requesting_user.get('industry', 'General')}
- Looking for: {requesting_user.get('requirements', 'Not specified')}
- Can offer: {requesting_user.get('offerings', 'Not specified')}

RECIPIENT:
- Name: {other_user.get('name', 'Unknown')}
- Role: {other_user.get('user_type', 'Professional')}
- Industry: {other_user.get('industry', 'General')}
- Looking for: {other_user.get('requirements', 'Not specified')}
- Can offer: {other_user.get('offerings', 'Not specified')}

Generate 4 distinct opening messages. Each should:
1. Reference something specific about the recipient
2. Express genuine interest or offer value
3. End with a question or call to action
4. Sound like a real person, not a template

Respond with a JSON object:
{{
    "suggestions": ["message 1", "message 2", "message 3", "message 4"]
}}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await chat_model.ainvoke(messages)
            content = response.content.strip()

            # Parse JSON response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            suggestions = result.get("suggestions", [])

            if suggestions and len(suggestions) >= 2:
                return suggestions[:4]

            return self._fallback_ice_breakers(other_user)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM ice breakers response: {e}")
            return self._fallback_ice_breakers(other_user)
        except Exception as e:
            logger.error(f"LLM ice breakers generation failed: {e}")
            return self._fallback_ice_breakers(other_user)

    def _fallback_ice_breakers(self, other_user: Dict) -> List[str]:
        """Fallback template-based ice breakers when LLM fails."""
        other_name = other_user.get('name', 'there')
        if other_name == "Unknown":
            other_name = "there"
        industry = other_user.get('industry', 'your field')
        role = other_user.get('user_type', 'professional')

        return [
            f"Hi {other_name}! I noticed we're both in {industry}. What got you started in this space?",
            f"I see you're a {role}. I'd love to hear about your current focus and what you're most excited about.",
            "What's the most interesting project you're working on right now?",
            "I think there might be some valuable synergies between what we're each working on. Would love to explore!"
        ]


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
