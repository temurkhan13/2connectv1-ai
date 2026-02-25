"""
Use-Case Templates for Objective-Specific AI Interactions.

Provides differentiated AI prompts, success criteria, and verdict logic
based on the user's primary objective (fundraising, hiring, partnership, mentorship).

Each template tailors:
1. System prompts for AI chat
2. Key questions to ask during conversations
3. Success criteria for match evaluation
4. Verdict criteria for determining match quality
"""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ObjectiveType(str, Enum):
    """Primary objectives users can have on the platform."""
    FUNDRAISING = "fundraising"
    HIRING = "hiring"
    PARTNERSHIP = "partnership"
    MENTORSHIP = "mentorship"
    INVESTING = "investing"
    COFOUNDER = "cofounder"
    NETWORKING = "networking"


class VerdictLevel(str, Enum):
    """Match verdict levels."""
    STRONG_MATCH = "strong_match"
    GOOD_MATCH = "good_match"
    POTENTIAL_MATCH = "potential_match"
    WEAK_MATCH = "weak_match"
    NO_MATCH = "no_match"


class UseCaseTemplate(BaseModel):
    """Template for a specific use case/objective."""
    objective: ObjectiveType
    display_name: str
    description: str
    system_prompt: str
    success_criteria: List[str]
    key_questions: List[str]
    verdict_criteria: Dict[str, List[str]]
    onboarding_focus_slots: List[str]
    match_weight_overrides: Dict[str, float]

    class Config:
        use_enum_values = True


# Define all templates
TEMPLATES: Dict[ObjectiveType, UseCaseTemplate] = {
    ObjectiveType.FUNDRAISING: UseCaseTemplate(
        objective=ObjectiveType.FUNDRAISING,
        display_name="Seeking Investment",
        description="Founders looking to raise capital for their startup",
        system_prompt="""You are an expert AI facilitating a conversation between a startup founder seeking investment and a potential investor.

Your role is to:
1. Help both parties understand if there's alignment on investment thesis, stage, and sector
2. Explore the founder's traction, team, and vision
3. Understand the investor's check size, involvement style, and value-add
4. Identify potential deal-breakers early (stage mismatch, sector focus, check size)

Focus areas for this conversation:
- Investment thesis alignment with the startup's sector
- Check size fit with the founder's raise amount
- Stage preference alignment (pre-seed, seed, Series A, etc.)
- Geographic considerations
- Value-add beyond capital (network, expertise, operational support)

Be direct about misalignments. It's better to identify poor fits quickly than waste both parties' time.""",
        success_criteria=[
            "Investment thesis matches startup sector",
            "Check size within investor's typical range",
            "Stage preference aligns with company stage",
            "Geographic focus is compatible",
            "Timeline expectations align",
            "Value-add matches founder's needs"
        ],
        key_questions=[
            "What's your typical check size for investments at this stage?",
            "Which sectors do you focus on, and why?",
            "What stage companies are you most active in?",
            "How hands-on are you with portfolio companies?",
            "What's your typical decision-making timeline?",
            "What would make this an exciting opportunity for you?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "thesis_match", "size_fit", "stage_fit", "timeline_align"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "thesis_match", "size_fit", "stage_fit"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "thesis_match", "stage_fit"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "thesis_match"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "funding_need", "company_stage", "industry_focus", "geography", "timeline"
        ],
        match_weight_overrides={
            "check_size_alignment": 0.25,
            "stage_alignment": 0.25,
            "sector_alignment": 0.20,
            "value_add_fit": 0.15,
            "timeline_compatibility": 0.15
        }
    ),

    ObjectiveType.INVESTING: UseCaseTemplate(
        objective=ObjectiveType.INVESTING,
        display_name="Looking to Invest",
        description="Investors seeking quality deal flow and founders",
        system_prompt="""You are an expert AI facilitating a conversation between an investor and a startup founder seeking investment.

Your role is to:
1. Help the investor evaluate if this startup fits their investment thesis
2. Explore the founder's background, traction, and vision
3. Understand the startup's metrics, team, and market opportunity
4. Identify alignment on terms, timeline, and involvement expectations

Focus areas for this conversation:
- Does this startup fit the investor's thesis and focus areas?
- Are the metrics and traction compelling for this stage?
- Is the team strong enough to execute?
- Is the valuation and round size appropriate?
- Would the investor want to be involved with this company long-term?

Be thorough in your evaluation. Good investors pass on many deals to find the right ones.""",
        success_criteria=[
            "Startup fits investment thesis",
            "Traction is appropriate for stage",
            "Team has relevant experience",
            "Market opportunity is large enough",
            "Terms and valuation are reasonable",
            "Chemistry and working style align"
        ],
        key_questions=[
            "What's your current MRR or revenue traction?",
            "Tell me about your team's background and why you're the right team",
            "What's your competitive moat or unfair advantage?",
            "How are you thinking about this raise and use of funds?",
            "What does success look like in 12-18 months?",
            "What kind of investor involvement are you looking for?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "thesis_fit", "traction_good", "team_strong", "terms_reasonable"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "thesis_fit", "traction_good", "team_strong"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "thesis_fit", "team_strong"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "thesis_fit"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "check_size", "stage_preference", "industry_focus", "investment_thesis", "geography"
        ],
        match_weight_overrides={
            "thesis_alignment": 0.30,
            "stage_fit": 0.25,
            "sector_match": 0.20,
            "check_size_fit": 0.15,
            "geography_match": 0.10
        }
    ),

    ObjectiveType.HIRING: UseCaseTemplate(
        objective=ObjectiveType.HIRING,
        display_name="Hiring Talent",
        description="Companies looking to hire executives, advisors, or key team members",
        system_prompt="""You are an expert AI facilitating a conversation between a hiring company and a potential candidate.

Your role is to:
1. Help both parties understand if there's role fit and mutual interest
2. Explore the candidate's skills, experience, and career goals
3. Understand the company's culture, growth stage, and what success looks like
4. Identify compensation alignment and logistics (remote, location, start date)

Focus areas for this conversation:
- Skills and experience alignment with role requirements
- Culture and working style fit
- Compensation expectations vs. budget
- Growth opportunities and career trajectory
- Logistics: location, remote work, availability

Be straightforward about misalignments. Finding the right fit matters more than filling a role quickly.""",
        success_criteria=[
            "Required skills and experience present",
            "Compensation expectations align with budget",
            "Culture and values fit",
            "Location/remote preferences compatible",
            "Start date and availability work",
            "Growth trajectory aligns with opportunity"
        ],
        key_questions=[
            "What's your experience with [specific skill/technology]?",
            "What compensation range are you targeting?",
            "What's your ideal working environment (remote, hybrid, in-office)?",
            "When could you potentially start?",
            "What's most important to you in your next role?",
            "Where do you see yourself in 2-3 years?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "skills_match", "experience_fit", "comp_align", "culture_fit"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "skills_match", "experience_fit", "comp_align"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "skills_match", "experience_fit"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "skills_match"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "role_type", "team_size", "industry_focus", "geography", "engagement_style"
        ],
        match_weight_overrides={
            "skills_alignment": 0.30,
            "experience_level": 0.25,
            "compensation_fit": 0.20,
            "culture_alignment": 0.15,
            "logistics_fit": 0.10
        }
    ),

    ObjectiveType.PARTNERSHIP: UseCaseTemplate(
        objective=ObjectiveType.PARTNERSHIP,
        display_name="Business Partnership",
        description="Companies seeking strategic partnerships, integrations, or collaborations",
        system_prompt="""You are an expert AI facilitating a conversation between two potential business partners.

Your role is to:
1. Help both parties understand if there's strategic alignment
2. Explore the value each party brings to a partnership
3. Understand what success looks like for both sides
4. Identify potential collaboration models and next steps

Focus areas for this conversation:
- Strategic alignment and shared goals
- Complementary strengths and capabilities
- Value exchange and mutual benefit
- Partnership structure and commitment level
- Timeline and resource requirements

Partnerships work best when both sides gain clear value. Be direct about where alignment exists and where it doesn't.""",
        success_criteria=[
            "Strategic goals align",
            "Complementary capabilities exist",
            "Value exchange is clear and mutual",
            "Resource and timeline expectations match",
            "Decision-making authority is clear",
            "Both parties see concrete next steps"
        ],
        key_questions=[
            "What's the strategic goal you're trying to achieve with partnerships?",
            "What capabilities or resources are you looking for in a partner?",
            "What value can you bring to a partnership?",
            "How do you typically structure partnerships?",
            "What does success look like in 6-12 months?",
            "Who would be involved in making this decision?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "strategic_align", "value_clear", "capabilities_complement", "timeline_match"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "strategic_align", "value_clear", "capabilities_complement"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "strategic_align", "value_clear"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "strategic_align"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "primary_goal", "industry_focus", "company_stage", "geography", "engagement_style"
        ],
        match_weight_overrides={
            "strategic_alignment": 0.30,
            "capability_complement": 0.25,
            "value_exchange": 0.20,
            "resource_fit": 0.15,
            "timeline_compatibility": 0.10
        }
    ),

    ObjectiveType.MENTORSHIP: UseCaseTemplate(
        objective=ObjectiveType.MENTORSHIP,
        display_name="Seeking Mentorship",
        description="Individuals seeking mentors or advisors for guidance",
        system_prompt="""You are an expert AI facilitating a conversation between someone seeking mentorship and a potential mentor.

Your role is to:
1. Help both parties understand if there's a good mentorship fit
2. Explore what the mentee is hoping to learn and achieve
3. Understand the mentor's expertise and how they like to help
4. Identify expectations around time commitment and format

Focus areas for this conversation:
- Alignment between mentee's goals and mentor's expertise
- Mentoring style compatibility
- Time commitment expectations
- Communication preferences
- Clear definition of success

Good mentorship relationships are built on mutual respect and clear expectations. Help both parties establish these foundations.""",
        success_criteria=[
            "Mentor expertise matches mentee needs",
            "Mentoring style aligns with preferences",
            "Time commitment expectations match",
            "Communication format works for both",
            "Clear goals for the mentorship",
            "Chemistry and rapport exist"
        ],
        key_questions=[
            "What specific areas are you hoping to get guidance on?",
            "What's your experience and expertise in this area?",
            "How much time can you commit to this mentorship?",
            "How do you prefer to communicate (calls, async, in-person)?",
            "What would make this mentorship successful for you?",
            "What's your typical mentoring style?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "expertise_match", "style_fit", "time_align", "chemistry_good"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "expertise_match", "style_fit", "time_align"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "expertise_match", "style_fit"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "expertise_match"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "primary_goal", "industry_focus", "experience_years", "engagement_style"
        ],
        match_weight_overrides={
            "expertise_alignment": 0.35,
            "style_compatibility": 0.25,
            "time_commitment": 0.20,
            "communication_fit": 0.10,
            "chemistry": 0.10
        }
    ),

    ObjectiveType.COFOUNDER: UseCaseTemplate(
        objective=ObjectiveType.COFOUNDER,
        display_name="Finding Co-founder",
        description="Entrepreneurs seeking co-founders to build with",
        system_prompt="""You are an expert AI facilitating a conversation between potential co-founders.

Your role is to:
1. Help both parties understand if they could work together long-term
2. Explore complementary skills and shared vision
3. Understand working styles, commitment levels, and expectations
4. Identify potential conflicts early (equity, control, timeline)

Focus areas for this conversation:
- Complementary skills (technical + business, etc.)
- Shared vision for the company
- Alignment on commitment level (full-time, timeline)
- Working style compatibility
- Equity and role expectations
- Values and principles alignment

Co-founder relationships are like marriages. Be thorough in exploring compatibility - it's better to discover misalignment now than after building together.""",
        success_criteria=[
            "Skills are complementary",
            "Vision for company aligns",
            "Commitment level matches",
            "Working styles compatible",
            "Equity expectations reasonable",
            "Values and principles align"
        ],
        key_questions=[
            "What's your vision for this company in 5 years?",
            "What skills and experience do you bring?",
            "How do you handle disagreements or conflict?",
            "What's your commitment level and timeline?",
            "How do you think about equity splits?",
            "What are your non-negotiables in a co-founder?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "skills_complement", "vision_align", "commitment_match", "values_align"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "skills_complement", "vision_align", "commitment_match"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "skills_complement", "vision_align"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "skills_complement"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "primary_goal", "industry_focus", "company_stage", "engagement_style", "experience_years"
        ],
        match_weight_overrides={
            "skills_complement": 0.25,
            "vision_alignment": 0.25,
            "commitment_match": 0.20,
            "working_style": 0.15,
            "values_alignment": 0.15
        }
    ),

    ObjectiveType.NETWORKING: UseCaseTemplate(
        objective=ObjectiveType.NETWORKING,
        display_name="General Networking",
        description="Professionals looking to expand their network",
        system_prompt="""You are an expert AI facilitating a networking conversation between two professionals.

Your role is to:
1. Help both parties find common ground and shared interests
2. Explore how they might be helpful to each other
3. Identify potential future collaboration opportunities
4. Keep the conversation engaging and productive

Focus areas for this conversation:
- Shared interests and experiences
- Complementary networks or expertise
- Potential ways to help each other
- Common challenges or goals
- Natural conversation flow

Good networking is about genuine connection, not transactions. Help create an authentic conversation that both parties enjoy.""",
        success_criteria=[
            "Found common interests or experiences",
            "Identified ways to help each other",
            "Conversation flowed naturally",
            "Both parties engaged actively",
            "Clear follow-up potential exists"
        ],
        key_questions=[
            "What are you working on right now that excites you?",
            "What's a challenge you're currently facing?",
            "How can I potentially be helpful to you?",
            "What's the most interesting thing you've learned recently?",
            "What does your ideal connection look like?"
        ],
        verdict_criteria={
            VerdictLevel.STRONG_MATCH.value: [
                "shared_interests", "mutual_help", "good_chemistry", "follow_up_clear"
            ],
            VerdictLevel.GOOD_MATCH.value: [
                "shared_interests", "mutual_help", "good_chemistry"
            ],
            VerdictLevel.POTENTIAL_MATCH.value: [
                "shared_interests", "good_chemistry"
            ],
            VerdictLevel.WEAK_MATCH.value: [
                "shared_interests"
            ],
            VerdictLevel.NO_MATCH.value: []
        },
        onboarding_focus_slots=[
            "primary_goal", "industry_focus", "user_type", "geography"
        ],
        match_weight_overrides={
            "interest_overlap": 0.30,
            "industry_alignment": 0.25,
            "network_complement": 0.20,
            "geography_proximity": 0.15,
            "experience_level": 0.10
        }
    ),
}


def get_template(objective: str) -> UseCaseTemplate:
    """
    Get the template for a given objective.

    Args:
        objective: The objective string (can be various formats)

    Returns:
        UseCaseTemplate for the objective

    Falls back to NETWORKING template if objective not found.
    """
    # Normalize the objective string
    normalized = objective.lower().strip()

    # Try direct enum match
    try:
        obj_type = ObjectiveType(normalized)
        return TEMPLATES[obj_type]
    except ValueError:
        pass

    # Try keyword matching
    keyword_mapping = {
        "invest": ObjectiveType.INVESTING,
        "investor": ObjectiveType.INVESTING,
        "angel": ObjectiveType.INVESTING,
        "vc": ObjectiveType.INVESTING,
        "fund": ObjectiveType.FUNDRAISING,
        "raise": ObjectiveType.FUNDRAISING,
        "capital": ObjectiveType.FUNDRAISING,
        "seeking investment": ObjectiveType.FUNDRAISING,
        "hire": ObjectiveType.HIRING,
        "hiring": ObjectiveType.HIRING,
        "recruit": ObjectiveType.HIRING,
        "talent": ObjectiveType.HIRING,
        "partner": ObjectiveType.PARTNERSHIP,
        "partnership": ObjectiveType.PARTNERSHIP,
        "collaborate": ObjectiveType.PARTNERSHIP,
        "mentor": ObjectiveType.MENTORSHIP,
        "advisor": ObjectiveType.MENTORSHIP,
        "advice": ObjectiveType.MENTORSHIP,
        "guidance": ObjectiveType.MENTORSHIP,
        "cofounder": ObjectiveType.COFOUNDER,
        "co-founder": ObjectiveType.COFOUNDER,
        "founding team": ObjectiveType.COFOUNDER,
        "network": ObjectiveType.NETWORKING,
        "connect": ObjectiveType.NETWORKING,
    }

    for keyword, obj_type in keyword_mapping.items():
        if keyword in normalized:
            logger.info(f"Matched objective '{objective}' to {obj_type.value} via keyword '{keyword}'")
            return TEMPLATES[obj_type]

    # Default fallback
    logger.warning(f"No template match for objective '{objective}', defaulting to NETWORKING")
    return TEMPLATES[ObjectiveType.NETWORKING]


def get_system_prompt(objective: str) -> str:
    """Get the system prompt for a given objective."""
    template = get_template(objective)
    return template.system_prompt


def get_key_questions(objective: str) -> List[str]:
    """Get the key questions for a given objective."""
    template = get_template(objective)
    return template.key_questions


def get_success_criteria(objective: str) -> List[str]:
    """Get the success criteria for a given objective."""
    template = get_template(objective)
    return template.success_criteria


def get_verdict_criteria(objective: str) -> Dict[str, List[str]]:
    """Get the verdict criteria for a given objective."""
    template = get_template(objective)
    return template.verdict_criteria


def get_onboarding_slots(objective: str) -> List[str]:
    """Get the focus slots for onboarding based on objective."""
    template = get_template(objective)
    return template.onboarding_focus_slots


def get_match_weights(objective: str) -> Dict[str, float]:
    """Get the match weight overrides for a given objective."""
    template = get_template(objective)
    return template.match_weight_overrides


def evaluate_verdict(
    objective: str,
    criteria_met: List[str]
) -> VerdictLevel:
    """
    Evaluate the verdict level based on criteria met.

    Args:
        objective: The user's objective
        criteria_met: List of criteria that were satisfied

    Returns:
        VerdictLevel indicating match quality
    """
    template = get_template(objective)
    criteria_set = set(criteria_met)

    # Check from strongest to weakest
    for level in [VerdictLevel.STRONG_MATCH, VerdictLevel.GOOD_MATCH,
                  VerdictLevel.POTENTIAL_MATCH, VerdictLevel.WEAK_MATCH]:
        required = set(template.verdict_criteria.get(level.value, []))
        if required and required.issubset(criteria_set):
            return level

    return VerdictLevel.NO_MATCH


def list_objectives() -> List[Dict[str, str]]:
    """List all available objectives with display names."""
    return [
        {
            "code": obj.value,
            "display_name": TEMPLATES[obj].display_name,
            "description": TEMPLATES[obj].description
        }
        for obj in ObjectiveType
    ]
