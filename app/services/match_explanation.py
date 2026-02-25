"""
Match Explanation Service.

Generates human-readable explanations for why two people were matched.
Breaks down the matching logic into understandable insights.

Key features:
1. Dimension-by-dimension explanation
2. Highlight key alignment factors
3. Mutual benefit articulation
4. Transparency about limitations
5. Adaptive detail level (summary vs detailed)
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.services.multi_vector_matcher import MatchTier, MultiVectorMatchResult as MultiVectorMatch

logger = logging.getLogger(__name__)


class ExplanationLevel(str, Enum):
    """Level of detail for explanation."""
    BRIEF = "brief"          # One sentence summary
    STANDARD = "standard"    # Paragraph with key points
    DETAILED = "detailed"    # Full breakdown


class AlignmentStrength(str, Enum):
    """Strength of alignment on a dimension."""
    STRONG = "strong"        # 80%+ match
    GOOD = "good"            # 60-79% match
    PARTIAL = "partial"      # 40-59% match
    WEAK = "weak"            # Below 40%


@dataclass
class DimensionExplanation:
    """Explanation for a single matching dimension."""
    dimension: str
    display_name: str
    score: float
    strength: AlignmentStrength
    explanation: str
    viewer_value: Optional[str] = None
    match_value: Optional[str] = None


@dataclass
class MutualBenefit:
    """Describes a mutual benefit between the matched users."""
    benefit_type: str
    for_viewer: str
    for_match: str
    confidence: float


@dataclass
class MatchExplanation:
    """Complete explanation for a match."""
    explanation_id: str
    viewer_user_id: str
    match_user_id: str
    tier: MatchTier

    # Core explanation
    summary: str
    detailed_explanation: str

    # Breakdowns
    dimension_explanations: List[DimensionExplanation]
    mutual_benefits: List[MutualBenefit]

    # Key points
    top_reasons: List[str]
    potential_concerns: List[str]

    # Metadata
    generated_at: datetime
    explanation_level: ExplanationLevel


class MatchExplainer:
    """
    Generates explanations for why matches were made.

    Provides transparency into the matching algorithm's
    reasoning in user-friendly language.
    """

    # Dimension display names and descriptions
    DIMENSION_INFO = {
        "primary_goal": {
            "name": "Goal Alignment",
            "description": "How well your objectives align",
            "strong_text": "You both have very similar goals on the platform",
            "weak_text": "Your goals differ, but this could bring diverse perspectives"
        },
        "industry_focus": {
            "name": "Industry Match",
            "description": "Overlap in industry interests",
            "strong_text": "Strong alignment in industry focus and expertise",
            "weak_text": "Different industry backgrounds - could be complementary"
        },
        "stage_preference": {
            "name": "Stage Fit",
            "description": "Alignment on company/investment stage",
            "strong_text": "Stage preferences align well",
            "weak_text": "Different stage focus - may still find common ground"
        },
        "geography": {
            "name": "Geographic Alignment",
            "description": "Location and market focus overlap",
            "strong_text": "Geographic focus overlaps significantly",
            "weak_text": "Different geographic priorities"
        },
        "engagement_style": {
            "name": "Working Style",
            "description": "How you prefer to work with others",
            "strong_text": "Compatible working and engagement styles",
            "weak_text": "Different working styles - communication is key"
        },
        "dealbreakers": {
            "name": "Compatibility",
            "description": "No dealbreaker conflicts",
            "strong_text": "No conflicts with stated preferences",
            "weak_text": "Some potential conflicts to discuss"
        }
    }

    # Templates for summary generation
    SUMMARY_TEMPLATES = {
        MatchTier.PERFECT: [
            "Exceptional match - strong alignment across all key dimensions including {top_dims}.",
            "Outstanding compatibility with particularly strong fit in {top_dims}.",
            "Highly compatible profiles with excellent alignment in {top_dims}."
        ],
        MatchTier.STRONG: [
            "Strong match with good alignment in {top_dims}, with room for interesting discussions in other areas.",
            "Solid compatibility, especially in {top_dims}. Worth exploring further.",
            "Good potential here - strong fit in {top_dims} suggests valuable connection."
        ],
        MatchTier.WORTH_EXPLORING: [
            "Interesting potential match. While {top_dims} align well, there's diversity in other areas that could be enriching.",
            "Promising connection with alignment in {top_dims}. Other differences could bring fresh perspectives.",
            "Worth exploring - {top_dims} show common ground, with opportunities to learn from differences."
        ],
        MatchTier.LOW: [
            "Limited overlap, but the connection in {top_dims} might still be valuable for specific needs.",
            "Different profiles with some alignment in {top_dims}. Could be worth a conversation to explore.",
            "Exploratory match - while overall alignment is limited, {top_dims} suggests possible value."
        ]
    }

    def __init__(self):
        self.default_level = ExplanationLevel(
            os.getenv("DEFAULT_EXPLANATION_LEVEL", "standard")
        )

    def explain_match(
        self,
        match: MultiVectorMatch,
        viewer_persona: Dict[str, Any],
        match_persona: Dict[str, Any],
        level: Optional[ExplanationLevel] = None
    ) -> MatchExplanation:
        """
        Generate explanation for a match.

        Args:
            match: The match result
            viewer_persona: Persona of the viewing user
            match_persona: Persona of the matched user
            level: Desired explanation detail level

        Returns:
            MatchExplanation with full breakdown
        """
        level = level or self.default_level
        explanation_id = f"exp_{match.user_id}_{datetime.utcnow().timestamp()}"

        # Generate dimension explanations
        dim_explanations = self._explain_dimensions(
            match.dimension_scores, viewer_persona, match_persona
        )

        # Identify mutual benefits
        mutual_benefits = self._identify_mutual_benefits(
            viewer_persona, match_persona, match.dimension_scores
        )

        # Extract top reasons
        top_reasons = self._extract_top_reasons(dim_explanations, mutual_benefits)

        # Identify potential concerns
        concerns = self._identify_concerns(dim_explanations)

        # Generate summary
        summary = self._generate_summary(match.tier, dim_explanations)

        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(
            dim_explanations, mutual_benefits, top_reasons, concerns
        )

        return MatchExplanation(
            explanation_id=explanation_id,
            viewer_user_id=viewer_persona.get("user_id", "unknown"),
            match_user_id=match.user_id,
            tier=match.tier,
            summary=summary,
            detailed_explanation=detailed,
            dimension_explanations=dim_explanations,
            mutual_benefits=mutual_benefits,
            top_reasons=top_reasons,
            potential_concerns=concerns,
            generated_at=datetime.utcnow(),
            explanation_level=level
        )

    def _explain_dimensions(
        self,
        scores: Dict[str, float],
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> List[DimensionExplanation]:
        """Generate explanations for each dimension."""
        explanations = []

        for dim, score in scores.items():
            info = self.DIMENSION_INFO.get(dim, {})

            # Determine strength
            if score >= 0.8:
                strength = AlignmentStrength.STRONG
                text = info.get("strong_text", f"Strong alignment in {dim}")
            elif score >= 0.6:
                strength = AlignmentStrength.GOOD
                text = f"Good alignment in {info.get('name', dim)}"
            elif score >= 0.4:
                strength = AlignmentStrength.PARTIAL
                text = f"Some overlap in {info.get('name', dim)}"
            else:
                strength = AlignmentStrength.WEAK
                text = info.get("weak_text", f"Different approaches to {dim}")

            # Try to extract actual values
            viewer_val, match_val = self._extract_dimension_values(
                dim, viewer, match
            )

            explanations.append(DimensionExplanation(
                dimension=dim,
                display_name=info.get("name", dim.replace("_", " ").title()),
                score=score,
                strength=strength,
                explanation=text,
                viewer_value=viewer_val,
                match_value=match_val
            ))

        # Sort by score descending
        return sorted(explanations, key=lambda e: e.score, reverse=True)

    def _extract_dimension_values(
        self,
        dimension: str,
        viewer: Dict[str, Any],
        match: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract actual values for a dimension from personas."""
        # Map dimensions to persona fields
        field_mapping = {
            "primary_goal": "what_theyre_looking_for",
            "industry_focus": "focus",
            "stage_preference": "experience",
            "geography": "focus",  # Often contains geography info
            "engagement_style": "engagement_style",
            "dealbreakers": "dealbreakers"
        }

        field_name = field_mapping.get(dimension)
        if not field_name:
            return None, None

        viewer_val = viewer.get(field_name)
        match_val = match.get(field_name)

        # Truncate long values
        if viewer_val and len(str(viewer_val)) > 50:
            viewer_val = str(viewer_val)[:47] + "..."
        if match_val and len(str(match_val)) > 50:
            match_val = str(match_val)[:47] + "..."

        return viewer_val, match_val

    def _identify_mutual_benefits(
        self,
        viewer: Dict[str, Any],
        match: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[MutualBenefit]:
        """Identify mutual benefits from the match."""
        benefits = []

        # Check for complementary needs/offerings
        viewer_needs = viewer.get("what_theyre_looking_for", "")
        viewer_offers = viewer.get("offerings", "")
        match_needs = match.get("what_theyre_looking_for", "")
        match_offers = match.get("offerings", "")

        # Industry expertise exchange
        if self._has_keyword_overlap(viewer_needs, match_offers):
            benefits.append(MutualBenefit(
                benefit_type="expertise_exchange",
                for_viewer=f"Access to expertise in {match.get('focus', 'their area')[:30]}",
                for_match=f"Connect with someone seeking {viewer_needs[:30]}",
                confidence=0.75
            ))

        if self._has_keyword_overlap(match_needs, viewer_offers):
            benefits.append(MutualBenefit(
                benefit_type="expertise_exchange",
                for_viewer=f"Opportunity to share your expertise",
                for_match=f"Access to your knowledge in {viewer_offers[:30]}",
                confidence=0.75
            ))

        # Network expansion
        if scores.get("geography", 0) < 0.5:
            benefits.append(MutualBenefit(
                benefit_type="network_expansion",
                for_viewer="Expand network into new geographic markets",
                for_match="Connect with contacts in your region",
                confidence=0.6
            ))

        # Learning opportunity
        if scores.get("industry_focus", 0) >= 0.6:
            benefits.append(MutualBenefit(
                benefit_type="learning",
                for_viewer="Learn from their industry perspective",
                for_match="Share insights and compare approaches",
                confidence=0.65
            ))

        return benefits[:3]  # Limit to top 3

    def _has_keyword_overlap(self, text1: str, text2: str) -> bool:
        """Check for meaningful keyword overlap."""
        keywords = [
            "fintech", "healthtech", "saas", "b2b", "ai", "ml",
            "climate", "sustainability", "marketplace", "platform",
            "seed", "series", "growth", "enterprise", "consumer"
        ]

        text1_lower = text1.lower() if text1 else ""
        text2_lower = text2.lower() if text2 else ""

        for kw in keywords:
            if kw in text1_lower and kw in text2_lower:
                return True
        return False

    def _extract_top_reasons(
        self,
        dimensions: List[DimensionExplanation],
        benefits: List[MutualBenefit]
    ) -> List[str]:
        """Extract top reasons for the match."""
        reasons = []

        # Add top dimension alignments
        for dim in dimensions[:3]:
            if dim.strength in [AlignmentStrength.STRONG, AlignmentStrength.GOOD]:
                reasons.append(f"{dim.display_name}: {dim.explanation}")

        # Add top benefits
        for benefit in benefits[:2]:
            reasons.append(f"Mutual benefit: {benefit.for_viewer}")

        return reasons[:4]

    def _identify_concerns(
        self,
        dimensions: List[DimensionExplanation]
    ) -> List[str]:
        """Identify potential concerns to mention."""
        concerns = []

        for dim in dimensions:
            if dim.strength == AlignmentStrength.WEAK:
                concerns.append(
                    f"{dim.display_name} differs - worth discussing early "
                    f"to ensure alignment"
                )

        return concerns[:2]  # Limit concerns shown

    def _generate_summary(
        self,
        tier: MatchTier,
        dimensions: List[DimensionExplanation]
    ) -> str:
        """Generate a summary sentence."""
        import random

        templates = self.SUMMARY_TEMPLATES.get(tier, self.SUMMARY_TEMPLATES[MatchTier.WORTH_EXPLORING])

        # Get top dimensions
        strong_dims = [
            d.display_name for d in dimensions[:3]
            if d.strength in [AlignmentStrength.STRONG, AlignmentStrength.GOOD]
        ]

        if strong_dims:
            top_dims = " and ".join(strong_dims[:2])
        else:
            top_dims = "some areas"

        template = random.choice(templates)
        return template.format(top_dims=top_dims)

    def _generate_detailed_explanation(
        self,
        dimensions: List[DimensionExplanation],
        benefits: List[MutualBenefit],
        reasons: List[str],
        concerns: List[str]
    ) -> str:
        """Generate detailed explanation paragraph."""
        parts = []

        # Opening with strongest alignment
        if dimensions and dimensions[0].strength == AlignmentStrength.STRONG:
            parts.append(
                f"This match shows particularly strong alignment in "
                f"{dimensions[0].display_name}. {dimensions[0].explanation}."
            )
        elif dimensions:
            parts.append(
                f"The matching analysis found notable alignment in "
                f"{dimensions[0].display_name}."
            )

        # Add benefit if available
        if benefits:
            benefit = benefits[0]
            parts.append(
                f"For you, this connection offers: {benefit.for_viewer}."
            )

        # Mention areas of difference constructively
        weak_dims = [d for d in dimensions if d.strength == AlignmentStrength.WEAK]
        if weak_dims and len(weak_dims) <= 2:
            dim_names = " and ".join(d.display_name for d in weak_dims)
            parts.append(
                f"While {dim_names} differ between you, this could bring "
                f"valuable diverse perspectives to the conversation."
            )

        # Closing recommendation
        if len(dimensions) > 0:
            avg_score = sum(d.score for d in dimensions) / len(dimensions)
            if avg_score >= 0.7:
                parts.append(
                    "Based on your profiles, this appears to be a connection "
                    "worth pursuing."
                )
            elif avg_score >= 0.5:
                parts.append(
                    "We recommend an exploratory conversation to see if there's "
                    "mutual value."
                )
            else:
                parts.append(
                    "While alignment is limited, specific shared interests may "
                    "still make this valuable."
                )

        return " ".join(parts)

    def explanation_to_dict(self, explanation: MatchExplanation) -> Dict[str, Any]:
        """Convert explanation to dictionary for API."""
        return {
            "explanation_id": explanation.explanation_id,
            "match_user_id": explanation.match_user_id,
            "tier": explanation.tier.value,
            "summary": explanation.summary,
            "detailed_explanation": explanation.detailed_explanation,
            "top_reasons": explanation.top_reasons,
            "potential_concerns": explanation.potential_concerns,
            "dimension_breakdown": [
                {
                    "dimension": d.dimension,
                    "display_name": d.display_name,
                    "score": round(d.score, 2),
                    "strength": d.strength.value,
                    "explanation": d.explanation,
                    "your_value": d.viewer_value,
                    "their_value": d.match_value
                }
                for d in explanation.dimension_explanations
            ],
            "mutual_benefits": [
                {
                    "type": b.benefit_type,
                    "for_you": b.for_viewer,
                    "for_them": b.for_match,
                    "confidence": round(b.confidence, 2)
                }
                for b in explanation.mutual_benefits
            ],
            "generated_at": explanation.generated_at.isoformat(),
            "explanation_level": explanation.explanation_level.value
        }


# Global instance
match_explainer = MatchExplainer()
