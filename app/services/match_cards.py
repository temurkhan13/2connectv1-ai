"""
Match Cards Service.

Generates rich match cards with confidence indicators, highlights,
and actionable information for display to users.

Key features:
1. Confidence visualization with tier badges
2. Key highlights extraction
3. Compatibility breakdown by dimension
4. Action suggestions
5. Quick facts extraction
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.services.multi_vector_matcher import MatchTier, MultiVectorMatch

logger = logging.getLogger(__name__)


class HighlightType(str, Enum):
    """Types of highlights to show on cards."""
    STRONG_FIT = "strong_fit"           # High alignment area
    SHARED_INTEREST = "shared_interest"  # Common ground
    COMPLEMENTARY = "complementary"      # They offer what you need
    UNIQUE_VALUE = "unique_value"        # Special quality
    CAUTION = "caution"                  # Potential mismatch area


class ActionType(str, Enum):
    """Types of actions available on match cards."""
    CONNECT = "connect"
    MESSAGE = "message"
    VIEW_PROFILE = "view_profile"
    SAVE = "save"
    SKIP = "skip"
    REPORT = "report"


@dataclass
class MatchHighlight:
    """A highlight to display on the match card."""
    highlight_type: HighlightType
    title: str
    description: str
    relevance_score: float  # 0-1, how relevant this highlight is
    icon: Optional[str] = None


@dataclass
class DimensionScore:
    """Score breakdown for a matching dimension."""
    dimension: str
    score: float
    label: str
    explanation: str


@dataclass
class QuickFact:
    """A quick fact to display on the card."""
    label: str
    value: str
    icon: Optional[str] = None


@dataclass
class MatchCard:
    """Complete match card for display."""
    card_id: str
    match_user_id: str
    display_name: str
    archetype: str
    designation: str

    # Scoring
    overall_score: float
    tier: MatchTier
    confidence_level: str  # "High", "Medium", "Low"

    # Content
    profile_summary: str
    highlights: List[MatchHighlight]
    dimension_scores: List[DimensionScore]
    quick_facts: List[QuickFact]

    # Actions
    primary_action: ActionType
    available_actions: List[ActionType]

    # Metadata
    generated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MatchCardGenerator:
    """
    Generates rich match cards from matching results.

    Transforms raw match data into user-friendly cards
    with highlights, explanations, and actions.
    """

    # Dimension display names
    DIMENSION_LABELS = {
        "primary_goal": "Goal Alignment",
        "industry_focus": "Industry Match",
        "stage_preference": "Stage Fit",
        "geography": "Location",
        "engagement_style": "Working Style",
        "dealbreakers": "Compatibility"
    }

    # Tier configurations
    TIER_CONFIG = {
        MatchTier.PERFECT: {
            "badge": "Perfect Match",
            "color": "#10B981",  # Green
            "confidence": "High",
            "description": "Exceptional alignment across all dimensions"
        },
        MatchTier.STRONG: {
            "badge": "Strong Match",
            "color": "#3B82F6",  # Blue
            "confidence": "High",
            "description": "Strong compatibility with great potential"
        },
        MatchTier.WORTH_EXPLORING: {
            "badge": "Good Potential",
            "color": "#F59E0B",  # Amber
            "confidence": "Medium",
            "description": "Interesting match worth exploring"
        },
        MatchTier.LOW: {
            "badge": "Possible Match",
            "color": "#6B7280",  # Gray
            "confidence": "Low",
            "description": "Some alignment, limited overlap"
        }
    }

    def __init__(self):
        # Configuration
        self.max_highlights = int(os.getenv("MAX_CARD_HIGHLIGHTS", "3"))
        self.max_quick_facts = int(os.getenv("MAX_QUICK_FACTS", "4"))
        self.show_dimension_scores = os.getenv(
            "SHOW_DIMENSION_SCORES", "true"
        ).lower() == "true"

    def generate_card(
        self,
        match: MultiVectorMatch,
        viewer_persona: Dict[str, Any],
        match_persona: Dict[str, Any]
    ) -> MatchCard:
        """
        Generate a match card from match data.

        Args:
            match: MultiVectorMatch result
            viewer_persona: Persona of the user viewing the card
            match_persona: Persona of the matched user

        Returns:
            MatchCard ready for display
        """
        card_id = f"card_{match.user_id}_{datetime.utcnow().timestamp()}"

        # Get tier configuration
        tier_config = self.TIER_CONFIG[match.tier]

        # Extract display info from persona
        display_name = match_persona.get("name", "Anonymous")
        archetype = match_persona.get("archetype", "Professional")
        designation = match_persona.get("designation", "")

        # Generate profile summary
        profile_summary = self._generate_summary(match_persona)

        # Generate highlights
        highlights = self._generate_highlights(
            match, viewer_persona, match_persona
        )

        # Generate dimension scores
        dimension_scores = self._generate_dimension_scores(match)

        # Extract quick facts
        quick_facts = self._extract_quick_facts(match_persona)

        # Determine actions
        primary_action, available_actions = self._determine_actions(match.tier)

        card = MatchCard(
            card_id=card_id,
            match_user_id=match.user_id,
            display_name=display_name,
            archetype=archetype,
            designation=designation,
            overall_score=match.overall_score,
            tier=match.tier,
            confidence_level=tier_config["confidence"],
            profile_summary=profile_summary,
            highlights=highlights,
            dimension_scores=dimension_scores,
            quick_facts=quick_facts,
            primary_action=primary_action,
            available_actions=available_actions,
            generated_at=datetime.utcnow(),
            metadata={
                "tier_badge": tier_config["badge"],
                "tier_color": tier_config["color"],
                "tier_description": tier_config["description"],
                "explanation": match.explanation
            }
        )

        logger.info(
            f"Generated card {card_id} for match {match.user_id} "
            f"(tier: {match.tier.value}, score: {match.overall_score:.2f})"
        )

        return card

    def _generate_summary(self, persona: Dict[str, Any]) -> str:
        """Generate a concise profile summary."""
        essence = persona.get("profile_essence", "")
        if essence:
            # Truncate if needed
            if len(essence) > 200:
                essence = essence[:197] + "..."
            return essence

        # Fallback: construct from other fields
        parts = []
        if persona.get("archetype"):
            parts.append(persona["archetype"])
        if persona.get("focus"):
            parts.append(f"focused on {persona['focus']}")
        if persona.get("experience"):
            parts.append(f"with {persona['experience']}")

        return " ".join(parts) if parts else "Profile information available"

    def _generate_highlights(
        self,
        match: MultiVectorMatch,
        viewer_persona: Dict[str, Any],
        match_persona: Dict[str, Any]
    ) -> List[MatchHighlight]:
        """Generate relevant highlights for the card."""
        highlights = []

        # Find strongest dimension matches
        if match.dimension_scores:
            sorted_dims = sorted(
                match.dimension_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for dim, score in sorted_dims[:2]:
                if score >= 0.8:
                    highlights.append(MatchHighlight(
                        highlight_type=HighlightType.STRONG_FIT,
                        title=f"Strong {self.DIMENSION_LABELS.get(dim, dim)}",
                        description=self._get_dimension_description(
                            dim, score, viewer_persona, match_persona
                        ),
                        relevance_score=score,
                        icon="check-circle"
                    ))

        # Check for complementary matches (offerings match requirements)
        viewer_reqs = viewer_persona.get("what_theyre_looking_for", "").lower()
        match_offerings = match_persona.get("offerings", "").lower()

        if viewer_reqs and match_offerings:
            overlap = self._find_keyword_overlap(viewer_reqs, match_offerings)
            if overlap:
                highlights.append(MatchHighlight(
                    highlight_type=HighlightType.COMPLEMENTARY,
                    title="They Offer What You Need",
                    description=f"Expertise in {', '.join(overlap[:2])}",
                    relevance_score=0.85,
                    icon="gift"
                ))

        # Check for shared interests
        viewer_focus = viewer_persona.get("focus", "").lower()
        match_focus = match_persona.get("focus", "").lower()

        if viewer_focus and match_focus:
            shared = self._find_keyword_overlap(viewer_focus, match_focus)
            if shared:
                highlights.append(MatchHighlight(
                    highlight_type=HighlightType.SHARED_INTEREST,
                    title="Shared Focus",
                    description=f"Both interested in {shared[0]}",
                    relevance_score=0.75,
                    icon="users"
                ))

        # Check for potential cautions (low dimension scores)
        if match.dimension_scores:
            for dim, score in match.dimension_scores.items():
                if score < 0.4 and dim != "dealbreakers":
                    highlights.append(MatchHighlight(
                        highlight_type=HighlightType.CAUTION,
                        title=f"Different {self.DIMENSION_LABELS.get(dim, dim)}",
                        description="May require discussion to align",
                        relevance_score=0.5,
                        icon="alert-circle"
                    ))
                    break  # Only one caution

        # Limit highlights
        return highlights[:self.max_highlights]

    def _get_dimension_description(
        self,
        dimension: str,
        score: float,
        viewer_persona: Dict[str, Any],
        match_persona: Dict[str, Any]
    ) -> str:
        """Generate description for a dimension match."""
        if dimension == "industry_focus":
            return "Industry experience aligns well with your focus"
        elif dimension == "stage_preference":
            return "Stage preferences match your criteria"
        elif dimension == "primary_goal":
            return "Goals are well aligned"
        elif dimension == "engagement_style":
            return "Working styles are compatible"
        elif dimension == "geography":
            return "Geographic focus overlaps"
        else:
            return f"Strong alignment in {dimension.replace('_', ' ')}"

    def _find_keyword_overlap(self, text1: str, text2: str) -> List[str]:
        """Find overlapping keywords between two texts."""
        # Simple keyword extraction (could be enhanced with NLP)
        keywords = [
            "fintech", "healthtech", "saas", "b2b", "b2c", "ai", "ml",
            "climate", "sustainability", "marketplace", "platform",
            "enterprise", "consumer", "mobile", "cloud", "data",
            "seed", "series a", "growth", "early stage"
        ]

        found = []
        for kw in keywords:
            if kw in text1 and kw in text2:
                found.append(kw.title())

        return found

    def _generate_dimension_scores(
        self,
        match: MultiVectorMatch
    ) -> List[DimensionScore]:
        """Generate dimension score breakdown."""
        if not self.show_dimension_scores:
            return []

        scores = []
        for dim, score in match.dimension_scores.items():
            # Convert to display label
            if score >= 0.8:
                label = "Excellent"
            elif score >= 0.6:
                label = "Good"
            elif score >= 0.4:
                label = "Fair"
            else:
                label = "Low"

            scores.append(DimensionScore(
                dimension=dim,
                score=score,
                label=label,
                explanation=self._get_score_explanation(dim, score)
            ))

        # Sort by score descending
        return sorted(scores, key=lambda x: x.score, reverse=True)

    def _get_score_explanation(self, dimension: str, score: float) -> str:
        """Generate explanation for a dimension score."""
        dim_label = self.DIMENSION_LABELS.get(dimension, dimension)

        if score >= 0.8:
            return f"{dim_label} is strongly aligned"
        elif score >= 0.6:
            return f"{dim_label} shows good compatibility"
        elif score >= 0.4:
            return f"{dim_label} has some overlap"
        else:
            return f"{dim_label} differs - may complement each other"

    def _extract_quick_facts(self, persona: Dict[str, Any]) -> List[QuickFact]:
        """Extract quick facts from persona."""
        facts = []

        # Experience
        if persona.get("experience"):
            facts.append(QuickFact(
                label="Experience",
                value=persona["experience"],
                icon="briefcase"
            ))

        # Focus
        if persona.get("focus"):
            focus = persona["focus"]
            if len(focus) > 50:
                focus = focus[:47] + "..."
            facts.append(QuickFact(
                label="Focus",
                value=focus,
                icon="target"
            ))

        # Investment style (if investor)
        if persona.get("investment_philosophy"):
            phil = persona["investment_philosophy"]
            if len(phil) > 50:
                phil = phil[:47] + "..."
            facts.append(QuickFact(
                label="Approach",
                value=phil,
                icon="lightbulb"
            ))

        # Engagement style
        if persona.get("engagement_style"):
            facts.append(QuickFact(
                label="Style",
                value=persona["engagement_style"],
                icon="message-circle"
            ))

        return facts[:self.max_quick_facts]

    def _determine_actions(
        self,
        tier: MatchTier
    ) -> Tuple[ActionType, List[ActionType]]:
        """Determine available actions based on match tier."""
        all_actions = [
            ActionType.CONNECT,
            ActionType.MESSAGE,
            ActionType.VIEW_PROFILE,
            ActionType.SAVE,
            ActionType.SKIP,
            ActionType.REPORT
        ]

        # Primary action varies by tier
        if tier in [MatchTier.PERFECT, MatchTier.STRONG]:
            primary = ActionType.CONNECT
        elif tier == MatchTier.WORTH_EXPLORING:
            primary = ActionType.VIEW_PROFILE
        else:
            primary = ActionType.VIEW_PROFILE

        return primary, all_actions

    def generate_cards_batch(
        self,
        matches: List[MultiVectorMatch],
        viewer_persona: Dict[str, Any],
        match_personas: Dict[str, Dict[str, Any]]
    ) -> List[MatchCard]:
        """
        Generate cards for multiple matches.

        Args:
            matches: List of match results
            viewer_persona: Persona of the viewing user
            match_personas: Dict mapping user_id to persona

        Returns:
            List of MatchCards
        """
        cards = []

        for match in matches:
            match_persona = match_personas.get(match.user_id, {})
            if match_persona:
                card = self.generate_card(match, viewer_persona, match_persona)
                cards.append(card)

        logger.info(f"Generated {len(cards)} match cards")
        return cards

    def card_to_dict(self, card: MatchCard) -> Dict[str, Any]:
        """Convert card to dictionary for API response."""
        return {
            "card_id": card.card_id,
            "match_user_id": card.match_user_id,
            "display_name": card.display_name,
            "archetype": card.archetype,
            "designation": card.designation,
            "overall_score": round(card.overall_score, 2),
            "tier": card.tier.value,
            "confidence_level": card.confidence_level,
            "profile_summary": card.profile_summary,
            "highlights": [
                {
                    "type": h.highlight_type.value,
                    "title": h.title,
                    "description": h.description,
                    "relevance_score": round(h.relevance_score, 2),
                    "icon": h.icon
                }
                for h in card.highlights
            ],
            "dimension_scores": [
                {
                    "dimension": d.dimension,
                    "score": round(d.score, 2),
                    "label": d.label,
                    "explanation": d.explanation
                }
                for d in card.dimension_scores
            ],
            "quick_facts": [
                {"label": f.label, "value": f.value, "icon": f.icon}
                for f in card.quick_facts
            ],
            "primary_action": card.primary_action.value,
            "available_actions": [a.value for a in card.available_actions],
            "generated_at": card.generated_at.isoformat(),
            "metadata": card.metadata
        }


# Global instance
match_card_generator = MatchCardGenerator()
