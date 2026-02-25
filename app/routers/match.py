"""
Match API endpoints.
Provides match explanation and ice breakers functionality.
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.match_explanation_service import match_explanation_service
from app.services.llm_service import get_llm_service
from app.adapters.dynamodb import UserProfile
from app.adapters.postgresql import postgresql_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/match", tags=["Match"])


class MatchExplanationRequest(BaseModel):
    """Request model for match explanation."""
    match_id: str = Field(..., description="UUID of the match")
    user_a_id: str = Field(..., description="UUID of user A")
    user_b_id: str = Field(..., description="UUID of user B")
    force_refresh: bool = Field(False, description="Force regeneration")


class ScoreDimension(BaseModel):
    """Individual dimension score with weight and explanation."""
    score: float = Field(0.5, description="Raw score (0-1)")
    weight: float = Field(0.166, description="Weight of this dimension")
    weighted_score: float = Field(0.083, description="Score * weight")
    explanation: str = Field("", description="Explanation of this score")


class ScoreBreakdown(BaseModel):
    """Score breakdown for match - matches frontend ScoreBreakdown interface."""
    objective_alignment: ScoreDimension = Field(default_factory=ScoreDimension)
    industry_match: ScoreDimension = Field(default_factory=ScoreDimension)
    timeline_compatibility: ScoreDimension = Field(default_factory=ScoreDimension)
    skill_complement: ScoreDimension = Field(default_factory=ScoreDimension)
    experience_level: ScoreDimension = Field(default_factory=ScoreDimension)
    communication_style: ScoreDimension = Field(default_factory=ScoreDimension)


class MatchExplanationResponse(BaseModel):
    """Response model for match explanation."""
    match_id: str
    summary: str
    synergy_areas: List[str]
    friction_points: List[str]
    talking_points: List[str]
    overall_score: float
    score_breakdown: ScoreBreakdown
    match_tier: str


class IceBreakersRequest(BaseModel):
    """Request model for ice breakers."""
    match_id: str = Field(..., description="UUID of the match")
    user_id: str = Field(..., description="UUID of the requesting user")
    other_user_id: str = Field(..., description="UUID of the other user in match")


class IceBreakersResponse(BaseModel):
    """Response model for ice breakers."""
    match_id: str
    suggestions: List[str]
    context_used: List[str]


def _extract_industry_from_focus(focus: str) -> str:
    """Extract industry from focus areas."""
    if not focus or focus == "Not specified":
        return "General"
    # Common industry keywords to look for
    industries = {
        "fintech": "Fintech",
        "healthtech": "Healthcare Technology",
        "healthcare": "Healthcare",
        "edtech": "Education Technology",
        "saas": "SaaS",
        "software": "Software",
        "ai": "Artificial Intelligence",
        "machine learning": "AI/ML",
        "data": "Data & Analytics",
        "e-commerce": "E-Commerce",
        "ecommerce": "E-Commerce",
        "real estate": "Real Estate",
        "proptech": "PropTech",
        "climate": "Climate Tech",
        "cleantech": "Clean Technology",
        "biotech": "Biotechnology",
        "crypto": "Crypto/Blockchain",
        "blockchain": "Blockchain",
        "web3": "Web3",
        "gaming": "Gaming",
        "media": "Media & Entertainment",
        "social": "Social Media",
        "marketplace": "Marketplace",
        "logistics": "Logistics",
        "supply chain": "Supply Chain",
        "hr": "HR Tech",
        "legal": "Legal Tech",
        "insurtech": "Insurance Technology",
        "agtech": "Agriculture Technology",
        "foodtech": "Food Technology",
    }
    focus_lower = focus.lower()
    for keyword, industry in industries.items():
        if keyword in focus_lower:
            return industry
    # Return first focus area as industry if no match
    first_focus = focus.split("|")[0].strip()
    return first_focus if first_focus else "General"


def _get_user_persona(user_id: str) -> dict:
    """Get user persona from DynamoDB with fallback to PostgreSQL."""
    try:
        user_profile = UserProfile.get(user_id)
        if user_profile and user_profile.persona:
            persona = user_profile.persona
            # Extract user_type from designation (e.g., "Software Engineer" -> "Software Engineer")
            user_type = persona.designation if persona.designation and persona.designation != "Not specified" else None
            if not user_type and persona.archetype:
                # Fall back to archetype (e.g., "Growth-Focused Founder")
                user_type = persona.archetype

            # Extract industry from focus areas
            industry = _extract_industry_from_focus(persona.focus or "")

            return {
                "name": persona.name or "Unknown",
                "user_type": user_type or "Professional",
                "industry": industry,
                "requirements": persona.requirements or "",
                "offerings": persona.offerings or "",
            }
    except Exception as e:
        logger.warning(f"DynamoDB lookup failed for {user_id}: {e}")

    # Fallback: try to get basic info from PostgreSQL user table
    try:
        conn = postgresql_adapter.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, onboarding_status FROM users WHERE id = %s",
            (user_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return {
                "name": row[0] or "Unknown",
                "user_type": "Professional",
                "industry": "General",
                "requirements": "",
                "offerings": "",
            }
    except Exception as e:
        logger.warning(f"PostgreSQL lookup failed for {user_id}: {e}")

    return {
        "name": "Unknown",
        "user_type": "Professional",
        "industry": "General",
        "requirements": "",
        "offerings": "",
    }


def _compute_match_tier(score: float) -> str:
    """Compute match tier from score."""
    if score >= 0.85:
        return "perfect"
    elif score >= 0.70:
        return "strong"
    elif score >= 0.55:
        return "worth_exploring"
    return "low"


def _format_user_type(user_type: str) -> str:
    """Format user type for display with correct article (a/an) handling."""
    if not user_type or user_type == "Unknown":
        return "professional"  # Better fallback than "Unknown"
    # Use "an" for vowel sounds
    if user_type[0].lower() in 'aeiou':
        return f"an {user_type}"
    return f"a {user_type}"


def _format_user_type_plain(user_type: str) -> str:
    """Format user type without article for direct use."""
    if not user_type or user_type == "Unknown":
        return "professional"
    return user_type


@router.post("/explanation", response_model=MatchExplanationResponse)
async def get_match_explanation(request: MatchExplanationRequest):
    """
    Generate AI explanation for why two users matched.

    Returns synergy areas, friction points, and suggested talking points.
    Uses real LLM to generate contextual explanations from both user profiles.
    """
    try:
        logger.info(f"Generating LLM explanation for match: {request.match_id}")

        # Get user personas
        user_a = _get_user_persona(request.user_a_id)
        user_b = _get_user_persona(request.user_b_id)

        # Calculate alignment scores (still used for scoring)
        req_to_off = 0.0
        off_to_req = 0.0

        if user_a["requirements"] and user_b["offerings"]:
            common_a = match_explanation_service.find_common_keywords(
                user_a["requirements"], user_b["offerings"]
            )
            req_to_off = min(len(common_a) / 5, 1.0) * 0.8 + 0.2

        if user_a["offerings"] and user_b["requirements"]:
            common_b = match_explanation_service.find_common_keywords(
                user_a["offerings"], user_b["requirements"]
            )
            off_to_req = min(len(common_b) / 5, 1.0) * 0.8 + 0.2

        # Industry match
        industry_match = 0.7 if user_a["industry"].lower() == user_b["industry"].lower() else 0.4

        # Calculate overall score
        overall_score = (req_to_off + off_to_req + industry_match) / 3
        match_tier = _compute_match_tier(overall_score)

        # Generate LLM-powered explanation
        llm_service = get_llm_service()
        scores = {
            "req_to_off": req_to_off,
            "off_to_req": off_to_req,
            "industry_match": industry_match
        }

        llm_result = await llm_service.generate_match_explanation(user_a, user_b, scores)

        # Build structured score breakdown matching frontend interface
        # Weights sum to 1.0 across 6 dimensions
        weights = {
            "objective_alignment": 0.20,
            "industry_match": 0.18,
            "timeline_compatibility": 0.15,
            "skill_complement": 0.20,
            "experience_level": 0.15,
            "communication_style": 0.12,
        }

        # Calculate scores for each dimension based on available data
        objective_score = (req_to_off + off_to_req) / 2  # Average of requirements alignment
        skill_score = max(req_to_off, off_to_req)  # Best skill alignment

        score_breakdown = ScoreBreakdown(
            objective_alignment=ScoreDimension(
                score=objective_score,
                weight=weights["objective_alignment"],
                weighted_score=objective_score * weights["objective_alignment"],
                explanation=f"Based on how well {user_a['name']}'s goals align with {user_b['name']}'s offerings"
            ),
            industry_match=ScoreDimension(
                score=industry_match,
                weight=weights["industry_match"],
                weighted_score=industry_match * weights["industry_match"],
                explanation=f"Both in {user_a['industry']}" if user_a['industry'].lower() == user_b['industry'].lower() else f"{user_a['industry']} + {user_b['industry']} cross-industry potential"
            ),
            timeline_compatibility=ScoreDimension(
                score=0.7,  # Default - would need timeline data to compute
                weight=weights["timeline_compatibility"],
                weighted_score=0.7 * weights["timeline_compatibility"],
                explanation="Timeline alignment based on availability"
            ),
            skill_complement=ScoreDimension(
                score=skill_score,
                weight=weights["skill_complement"],
                weighted_score=skill_score * weights["skill_complement"],
                explanation=f"Complementary skills between {user_a['user_type']} and {user_b['user_type']}"
            ),
            experience_level=ScoreDimension(
                score=0.65,  # Default - would need experience data
                weight=weights["experience_level"],
                weighted_score=0.65 * weights["experience_level"],
                explanation="Experience levels are compatible"
            ),
            communication_style=ScoreDimension(
                score=0.7,  # Default
                weight=weights["communication_style"],
                weighted_score=0.7 * weights["communication_style"],
                explanation="Communication preferences align"
            ),
        )

        # Recalculate overall score from weighted dimensions
        overall_score = sum([
            score_breakdown.objective_alignment.weighted_score,
            score_breakdown.industry_match.weighted_score,
            score_breakdown.timeline_compatibility.weighted_score,
            score_breakdown.skill_complement.weighted_score,
            score_breakdown.experience_level.weighted_score,
            score_breakdown.communication_style.weighted_score,
        ])
        match_tier = _compute_match_tier(overall_score)

        return MatchExplanationResponse(
            match_id=request.match_id,
            summary=llm_result["summary"],
            synergy_areas=llm_result["synergy_areas"],
            friction_points=llm_result["friction_points"],
            talking_points=llm_result["talking_points"],
            overall_score=overall_score,
            score_breakdown=score_breakdown,
            match_tier=match_tier
        )

    except Exception as e:
        logger.error(f"Error generating match explanation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@router.post("/ice-breakers", response_model=IceBreakersResponse)
async def get_ice_breakers(request: IceBreakersRequest):
    """
    Generate AI-powered conversation starters for a match.

    Uses real LLM to generate personalized ice breaker messages
    based on both users' profiles, requirements, and offerings.
    """
    try:
        logger.info(f"Generating LLM ice breakers for match: {request.match_id}")

        # Get both users' personas for context-aware generation
        requesting_user = _get_user_persona(request.user_id)
        other_user = _get_user_persona(request.other_user_id)

        # Generate LLM-powered ice breakers
        llm_service = get_llm_service()
        suggestions = await llm_service.generate_ice_breakers(
            requesting_user=requesting_user,
            other_user=other_user
        )

        context_used = [
            f"Sender: {requesting_user['name']} ({requesting_user['user_type']})",
            f"Recipient: {other_user['name']} ({other_user['user_type']})",
            f"Industry: {other_user['industry']}",
            f"Their needs: {other_user['requirements'][:100]}..." if other_user['requirements'] else "Their needs: Not specified",
        ]

        return IceBreakersResponse(
            match_id=request.match_id,
            suggestions=suggestions,
            context_used=context_used
        )

    except Exception as e:
        logger.error(f"Error generating ice breakers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate ice breakers: {str(e)}")
