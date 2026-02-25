"""
AI-Powered Ice Breakers Service.

Generates personalized conversation starters based on
matched profiles, finding common ground and interesting topics.

Key features:
1. Context-aware opener generation
2. Multiple opener styles (professional, casual, direct)
3. Topic extraction from both profiles
4. Personalization scoring
5. Fallback starters when AI unavailable
"""
import os
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OpenerStyle(str, Enum):
    """Style of ice breaker."""
    PROFESSIONAL = "professional"  # Formal, business-focused
    CASUAL = "casual"              # Friendly, conversational
    DIRECT = "direct"              # Straight to the point
    CURIOUS = "curious"            # Question-based
    COMPLIMENTARY = "complimentary"  # Starts with genuine praise


class OpenerCategory(str, Enum):
    """Category of what the opener references."""
    SHARED_INTEREST = "shared_interest"
    COMPLEMENTARY = "complementary"
    BACKGROUND = "background"
    INDUSTRY = "industry"
    GOAL = "goal"
    GENERIC = "generic"


@dataclass
class IceBreaker:
    """A generated ice breaker."""
    breaker_id: str
    text: str
    style: OpenerStyle
    category: OpenerCategory
    personalization_score: float  # 0-1, how personalized
    context_used: List[str]  # What info was used
    follow_up_suggestions: List[str] = field(default_factory=list)


@dataclass
class IceBreakerSet:
    """Set of ice breakers for a match."""
    match_user_id: str
    breakers: List[IceBreaker]
    generated_at: datetime
    common_topics: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IceBreakerGenerator:
    """
    Generates personalized ice breakers for matches.

    Uses profile analysis to find relevant talking points
    and generates contextual conversation starters.
    """

    # Templates by style and category
    TEMPLATES = {
        (OpenerStyle.PROFESSIONAL, OpenerCategory.SHARED_INTEREST): [
            "I noticed we both have a strong interest in {topic}. I'd love to hear your perspective on {related_question}.",
            "Your focus on {topic} caught my attention - it's an area I'm deeply invested in as well. Would you be open to discussing {related_aspect}?",
            "Given our shared interest in {topic}, I think there could be some interesting synergies. What's your current thinking on {trend}?"
        ],
        (OpenerStyle.PROFESSIONAL, OpenerCategory.COMPLEMENTARY): [
            "Your experience in {their_strength} aligns well with what I'm looking for. I'd appreciate the chance to learn more about your approach.",
            "I'm impressed by your work in {their_strength}. Given my focus on {my_need}, I believe we could have a valuable conversation.",
            "Your expertise in {their_strength} could be exactly what I need for {my_goal}. Would you be open to a brief chat?"
        ],
        (OpenerStyle.CASUAL, OpenerCategory.SHARED_INTEREST): [
            "Hey! Looks like we're both into {topic} - always great to meet someone else in the space. What got you started?",
            "Love seeing someone else passionate about {topic}! What's the most exciting thing you're working on right now?",
            "We seem to have {topic} in common! Would love to swap notes sometime."
        ],
        (OpenerStyle.CASUAL, OpenerCategory.INDUSTRY): [
            "Fellow {industry} enthusiast here! What trends are you most excited about this year?",
            "Great to connect with someone in {industry}. How's the landscape looking from your end?",
            "Nice to meet another {industry} person! What's keeping you busy these days?"
        ],
        (OpenerStyle.DIRECT, OpenerCategory.GOAL): [
            "I'm looking for {my_goal} and your profile suggests we might be a good fit. Interested in exploring?",
            "Based on your background, I think we could help each other. I'm focused on {my_goal} - does that align with what you're looking for?",
            "Let me be direct: I think there's potential here. You're strong in {their_strength}, I need {my_need}. Worth a conversation?"
        ],
        (OpenerStyle.CURIOUS, OpenerCategory.BACKGROUND): [
            "Your journey from {background_point} to {current_focus} is fascinating. What drove that transition?",
            "I'm curious about your experience with {background_point}. How has that shaped your current approach?",
            "Your background in {background_point} is intriguing. How do you see that influencing {current_focus}?"
        ],
        (OpenerStyle.COMPLIMENTARY, OpenerCategory.BACKGROUND): [
            "Really impressed by your work in {their_strength}. The approach you've taken to {specific_aspect} is exactly what the industry needs.",
            "Your profile stood out - not often you see someone with such depth in {their_strength}. Would love to learn more.",
            "I have to say, your experience with {background_point} is remarkable. I'd value the chance to hear your insights."
        ]
    }

    # Fallback generic openers
    GENERIC_OPENERS = [
        "Hi! I came across your profile and think we might have some interesting things to discuss. Would you be open to connecting?",
        "Hello! Based on our matching criteria, it seems we could have a valuable conversation. What do you think?",
        "Hi there! 2Connect suggested we might be a good match. I'd love to learn more about what you're working on.",
        "Hello! I'm always looking to connect with interesting people in the space. Your profile caught my attention - shall we chat?"
    ]

    def __init__(self):
        # Configuration
        self.num_breakers_per_match = int(os.getenv("NUM_ICE_BREAKERS", "3"))
        self.min_personalization = float(os.getenv("MIN_PERSONALIZATION_SCORE", "0.3"))

        # Keywords for topic extraction
        self.industry_keywords = {
            "fintech": ["fintech", "financial", "banking", "payments", "lending"],
            "healthtech": ["healthtech", "healthcare", "medical", "biotech", "health"],
            "saas": ["saas", "software", "b2b", "enterprise", "platform"],
            "ai": ["ai", "machine learning", "ml", "artificial intelligence", "deep learning"],
            "climate": ["climate", "sustainability", "green", "cleantech", "renewable"],
            "marketplace": ["marketplace", "platform", "two-sided", "network effects"]
        }

    def generate_ice_breakers(
        self,
        viewer_persona: Dict[str, Any],
        match_persona: Dict[str, Any],
        match_score: float = 0.7
    ) -> IceBreakerSet:
        """
        Generate ice breakers for a match.

        Args:
            viewer_persona: Persona of the user initiating
            match_persona: Persona of the matched user
            match_score: Overall match score

        Returns:
            IceBreakerSet with multiple breakers
        """
        match_user_id = match_persona.get("user_id", "unknown")
        breakers = []

        # Analyze both profiles
        common_topics = self._find_common_topics(viewer_persona, match_persona)
        complementary = self._find_complementary_aspects(viewer_persona, match_persona)
        their_strengths = self._extract_strengths(match_persona)
        my_goals = self._extract_goals(viewer_persona)

        context = {
            "common_topics": common_topics,
            "complementary": complementary,
            "their_strengths": their_strengths,
            "my_goals": my_goals,
            "match_score": match_score
        }

        # Generate breakers of different styles
        styles_to_use = self._select_styles(match_score)

        for style in styles_to_use:
            breaker = self._generate_single_breaker(
                style, context, viewer_persona, match_persona
            )
            if breaker:
                breakers.append(breaker)

        # Ensure we have enough breakers
        while len(breakers) < self.num_breakers_per_match:
            breakers.append(self._generate_fallback_breaker(context))

        # Sort by personalization score
        breakers.sort(key=lambda b: b.personalization_score, reverse=True)

        return IceBreakerSet(
            match_user_id=match_user_id,
            breakers=breakers[:self.num_breakers_per_match],
            generated_at=datetime.utcnow(),
            common_topics=common_topics,
            metadata={
                "context_analyzed": list(context.keys()),
                "styles_used": [b.style.value for b in breakers]
            }
        )

    def _find_common_topics(
        self,
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> List[str]:
        """Find topics both personas share."""
        common = []

        viewer_text = self._combine_text_fields(viewer)
        match_text = self._combine_text_fields(match)

        for topic, keywords in self.industry_keywords.items():
            viewer_has = any(kw in viewer_text for kw in keywords)
            match_has = any(kw in match_text for kw in keywords)
            if viewer_has and match_has:
                common.append(topic)

        return common

    def _find_complementary_aspects(
        self,
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Find where one offers what the other needs."""
        complementary = []

        viewer_needs = viewer.get("what_theyre_looking_for", "")
        match_offers = match.get("offerings", "")

        # Simple keyword matching for complementary
        if viewer_needs and match_offers:
            for topic, keywords in self.industry_keywords.items():
                needs_it = any(kw in viewer_needs.lower() for kw in keywords)
                has_it = any(kw in match_offers.lower() for kw in keywords)
                if needs_it and has_it:
                    complementary.append({
                        "need": topic,
                        "offered_by": "match"
                    })

        return complementary

    def _extract_strengths(self, persona: Dict[str, Any]) -> List[str]:
        """Extract notable strengths from persona."""
        strengths = []

        if persona.get("experience"):
            strengths.append(persona["experience"])

        if persona.get("focus"):
            strengths.append(persona["focus"])

        if persona.get("investment_philosophy"):
            strengths.append(persona["investment_philosophy"][:50])

        return strengths[:3]

    def _extract_goals(self, persona: Dict[str, Any]) -> List[str]:
        """Extract goals from persona."""
        goals = []

        if persona.get("what_theyre_looking_for"):
            goals.append(persona["what_theyre_looking_for"][:100])

        if persona.get("requirements"):
            goals.append(persona["requirements"][:100])

        return goals[:2]

    def _combine_text_fields(self, persona: Dict[str, Any]) -> str:
        """Combine relevant text fields for analysis."""
        fields = [
            persona.get("focus", ""),
            persona.get("profile_essence", ""),
            persona.get("investment_philosophy", ""),
            persona.get("what_theyre_looking_for", ""),
            persona.get("offerings", ""),
            persona.get("requirements", "")
        ]
        return " ".join(f.lower() for f in fields if f)

    def _select_styles(self, match_score: float) -> List[OpenerStyle]:
        """Select styles based on match quality."""
        if match_score >= 0.85:
            # High match - can be more direct
            return [OpenerStyle.DIRECT, OpenerStyle.PROFESSIONAL, OpenerStyle.CURIOUS]
        elif match_score >= 0.7:
            # Good match - mix of approaches
            return [OpenerStyle.PROFESSIONAL, OpenerStyle.CASUAL, OpenerStyle.COMPLIMENTARY]
        else:
            # Lower match - more exploratory
            return [OpenerStyle.CURIOUS, OpenerStyle.CASUAL, OpenerStyle.PROFESSIONAL]

    def _generate_single_breaker(
        self,
        style: OpenerStyle,
        context: Dict[str, Any],
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> Optional[IceBreaker]:
        """Generate a single ice breaker."""
        # Determine best category based on available context
        category, template_vars = self._select_category_and_vars(
            style, context, viewer, match
        )

        # Get templates for this style/category combo
        templates = self.TEMPLATES.get((style, category), [])

        if not templates:
            # Try alternative categories
            for alt_category in OpenerCategory:
                templates = self.TEMPLATES.get((style, alt_category), [])
                if templates:
                    category = alt_category
                    break

        if not templates:
            return None

        # Select and fill template
        template = random.choice(templates)
        try:
            text = template.format(**template_vars)
        except KeyError:
            # Missing vars, try with defaults
            text = self._fill_template_with_defaults(template, template_vars)

        # Calculate personalization score
        personalization = self._calculate_personalization(text, template_vars)

        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(category, context)

        breaker_id = f"ib_{datetime.utcnow().timestamp()}_{random.randint(1000, 9999)}"

        return IceBreaker(
            breaker_id=breaker_id,
            text=text,
            style=style,
            category=category,
            personalization_score=personalization,
            context_used=list(template_vars.keys()),
            follow_up_suggestions=follow_ups
        )

    def _select_category_and_vars(
        self,
        style: OpenerStyle,
        context: Dict[str, Any],
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> Tuple[OpenerCategory, Dict[str, str]]:
        """Select best category and prepare template variables."""
        vars = {}

        # Check for shared interests
        if context["common_topics"]:
            topic = context["common_topics"][0]
            vars["topic"] = topic.title()
            vars["related_question"] = f"the latest developments in {topic}"
            vars["related_aspect"] = f"how {topic} is evolving"
            vars["trend"] = f"where {topic} is heading"
            return OpenerCategory.SHARED_INTEREST, vars

        # Check for complementary aspects
        if context["complementary"]:
            comp = context["complementary"][0]
            vars["their_strength"] = match.get("focus", "your area")
            vars["my_need"] = comp["need"]
            vars["my_goal"] = viewer.get("what_theyre_looking_for", "my goals")[:50]
            return OpenerCategory.COMPLEMENTARY, vars

        # Use their strengths
        if context["their_strengths"]:
            strength = context["their_strengths"][0]
            vars["their_strength"] = strength[:50]
            vars["specific_aspect"] = "your approach"
            vars["background_point"] = strength[:30]
            vars["current_focus"] = match.get("focus", "your current work")
            return OpenerCategory.BACKGROUND, vars

        # Use industry
        industry = self._extract_industry(match)
        if industry:
            vars["industry"] = industry.title()
            return OpenerCategory.INDUSTRY, vars

        # Fallback to goal
        vars["my_goal"] = viewer.get("what_theyre_looking_for", "connecting with the right people")[:50]
        vars["their_strength"] = match.get("focus", "your expertise")
        vars["my_need"] = "valuable connections"
        return OpenerCategory.GOAL, vars

    def _extract_industry(self, persona: Dict[str, Any]) -> Optional[str]:
        """Extract primary industry from persona."""
        text = self._combine_text_fields(persona)
        for industry, keywords in self.industry_keywords.items():
            if any(kw in text for kw in keywords):
                return industry
        return None

    def _fill_template_with_defaults(
        self,
        template: str,
        vars: Dict[str, str]
    ) -> str:
        """Fill template with defaults for missing vars."""
        defaults = {
            "topic": "technology",
            "their_strength": "your experience",
            "my_need": "valuable insights",
            "my_goal": "meaningful connections",
            "industry": "the industry",
            "background_point": "your background",
            "current_focus": "your current work",
            "specific_aspect": "your approach",
            "related_question": "current trends",
            "related_aspect": "future directions",
            "trend": "market evolution"
        }

        merged = {**defaults, **vars}
        try:
            return template.format(**merged)
        except KeyError:
            return random.choice(self.GENERIC_OPENERS)

    def _calculate_personalization(
        self,
        text: str,
        vars: Dict[str, str]
    ) -> float:
        """Calculate how personalized the breaker is."""
        # More specific vars = higher personalization
        score = 0.3  # Base score

        # Bonus for specific context used
        if "topic" in vars and vars["topic"] != "technology":
            score += 0.2
        if "their_strength" in vars and len(vars.get("their_strength", "")) > 10:
            score += 0.2
        if "my_goal" in vars and len(vars.get("my_goal", "")) > 10:
            score += 0.15
        if "industry" in vars and vars["industry"] != "the industry":
            score += 0.15

        return min(1.0, score)

    def _generate_follow_ups(
        self,
        category: OpenerCategory,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up suggestions."""
        follow_ups = []

        if category == OpenerCategory.SHARED_INTEREST:
            follow_ups.append("What challenges are you facing in this area?")
            follow_ups.append("Have you seen any interesting developments recently?")

        elif category == OpenerCategory.COMPLEMENTARY:
            follow_ups.append("What would make this a valuable connection for you?")
            follow_ups.append("What's your current priority in this area?")

        elif category == OpenerCategory.BACKGROUND:
            follow_ups.append("What lessons from that experience still guide you?")
            follow_ups.append("How did that shape your current approach?")

        else:
            follow_ups.append("What are you most excited about right now?")
            follow_ups.append("What would make this a successful connection for you?")

        return follow_ups[:2]

    def _generate_fallback_breaker(
        self,
        context: Dict[str, Any]
    ) -> IceBreaker:
        """Generate a generic fallback breaker."""
        text = random.choice(self.GENERIC_OPENERS)

        return IceBreaker(
            breaker_id=f"ib_fallback_{datetime.utcnow().timestamp()}",
            text=text,
            style=OpenerStyle.PROFESSIONAL,
            category=OpenerCategory.GENERIC,
            personalization_score=0.2,
            context_used=[],
            follow_up_suggestions=[
                "What are you working on currently?",
                "What brought you to 2Connect?"
            ]
        )

    def breaker_set_to_dict(self, breaker_set: IceBreakerSet) -> Dict[str, Any]:
        """Convert breaker set to dictionary for API."""
        return {
            "match_user_id": breaker_set.match_user_id,
            "generated_at": breaker_set.generated_at.isoformat(),
            "common_topics": breaker_set.common_topics,
            "breakers": [
                {
                    "breaker_id": b.breaker_id,
                    "text": b.text,
                    "style": b.style.value,
                    "category": b.category.value,
                    "personalization_score": round(b.personalization_score, 2),
                    "follow_up_suggestions": b.follow_up_suggestions
                }
                for b in breaker_set.breakers
            ],
            "metadata": breaker_set.metadata
        }


# Global instance
ice_breaker_generator = IceBreakerGenerator()
