"""
Matching Criteria - Extracted from Matching Scenario Document (Ryan Shared).

This module contains detailed matching criteria for each connection type,
based on the exhaustive matching intelligence document.

Created: Feb 2026
Source: Reciprocity old handover documents/Matching Scenario - Ryan Shared.md
"""
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class ConnectionType(str, Enum):
    """Connection types with complementary matching."""
    INVESTOR_FOUNDER = "investor_founder"
    FOUNDER_CTO = "founder_cto"
    MENTOR_MENTEE = "mentor_mentee"
    PARTNER_PARTNER = "partner_partner"
    RECRUITER_CANDIDATE = "recruiter_candidate"
    SALES_BIZDEV = "sales_bizdev"
    GENERAL_NETWORKING = "general_networking"


@dataclass
class MatchingCriterion:
    """A single matching criterion with weight and evaluation method."""
    name: str
    weight: float
    description: str
    keywords_positive: List[str] = field(default_factory=list)  # Keywords that indicate match
    keywords_negative: List[str] = field(default_factory=list)  # Keywords that indicate mismatch
    field_to_check: str = ""  # Persona field to check


@dataclass
class ConnectionCriteria:
    """Complete matching criteria for a connection type."""
    connection_type: ConnectionType
    display_name: str
    side_a_wants: List[str]  # What side A is looking for
    side_b_wants: List[str]  # What side B is looking for
    criteria: List[MatchingCriterion]
    complementary_pair: tuple = None  # (SideA_intent, SideB_intent)


# ============================================================================
# INVESTOR ↔ FOUNDER CRITERIA
# From Matching Scenario doc Section 1
# ============================================================================
INVESTOR_FOUNDER_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.INVESTOR_FOUNDER,
    display_name="Investor ↔ Founder",
    side_a_wants=[  # What Investor wants
        "Access to vetted, high-potential deals",
        "Founders who are coachable and decisive",
        "Aligned exit expectations",
        "Proof of traction or market validation",
        "Confidence in go-to-market plan",
        "Terms favorable to fund structure",
        "Opportunity to co-invest or lead round",
        "Alignment with personal or fund thesis",
        "Clear cap table and governance",
    ],
    side_b_wants=[  # What Founder wants
        "Capital with strategic value",
        "Investors who understand their industry",
        "Hands-off vs. hands-on clarity",
        "Access to investor's network",
        "Support during future funding rounds",
        "Fair terms and transparency",
        "Investors with good reputation",
        "Alignment on mission and values",
        "Long-term commitment or follow-on potential",
    ],
    criteria=[
        MatchingCriterion(
            name="stage_alignment",
            weight=0.25,
            description="Investment stage matches (seed, Series A, etc.)",
            keywords_positive=["seed", "series a", "series b", "pre-seed", "growth"],
            field_to_check="stage"
        ),
        MatchingCriterion(
            name="check_size_fit",
            weight=0.20,
            description="Check size matches raise amount",
            keywords_positive=["$500k", "$1m", "$2m", "$5m", "$10m"],
            field_to_check="funding_need"
        ),
        MatchingCriterion(
            name="sector_alignment",
            weight=0.20,
            description="Industry/sector focus matches",
            keywords_positive=["saas", "fintech", "healthcare", "ai", "b2b"],
            field_to_check="industry"
        ),
        MatchingCriterion(
            name="value_add_match",
            weight=0.15,
            description="Investor's value-add matches founder's needs",
            keywords_positive=["operational", "network", "board", "strategic", "hands-on"],
            field_to_check="engagement_style"
        ),
        MatchingCriterion(
            name="timeline_compatibility",
            weight=0.10,
            description="Fundraising timeline alignment",
            keywords_positive=["now", "immediate", "q1", "q2", "active"],
            field_to_check="timeline"
        ),
        MatchingCriterion(
            name="geography_match",
            weight=0.10,
            description="Geographic focus or remote compatibility",
            keywords_positive=["us", "europe", "global", "remote"],
            field_to_check="geography"
        ),
    ],
    complementary_pair=("investor_founder", "founder_investor")
)


# ============================================================================
# FOUNDER ↔ CTO CRITERIA
# From Matching Scenario doc Section 2
# ============================================================================
FOUNDER_CTO_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.FOUNDER_CTO,
    display_name="Founder ↔ CTO",
    side_a_wants=[  # What Founder wants
        "Technical partner who believes in vision",
        "Full-stack execution capabilities",
        "Vision to roadmap translation ability",
        "System architecture expertise",
        "Equity-based alignment",
        "Relevant technology experience",
        "Team-building and hiring ability",
    ],
    side_b_wants=[  # What CTO wants
        "Strong founder with product/market vision",
        "Clear direction and priorities",
        "Reasonable expectations and funding plan",
        "Early say in architecture and hiring",
        "Mutual trust and respect",
        "Long-term commitment and role clarity",
    ],
    criteria=[
        MatchingCriterion(
            name="technical_fit",
            weight=0.25,
            description="Technical skills match product needs",
            keywords_positive=["python", "react", "aws", "backend", "frontend", "full-stack"],
            field_to_check="skills"
        ),
        MatchingCriterion(
            name="equity_expectations",
            weight=0.20,
            description="Equity and compensation alignment",
            keywords_positive=["equity", "cofounder", "vesting", "ownership"],
            field_to_check="compensation"
        ),
        MatchingCriterion(
            name="commitment_level",
            weight=0.20,
            description="Full-time vs part-time commitment",
            keywords_positive=["full-time", "dedicated", "all-in"],
            keywords_negative=["part-time", "consulting", "freelance"],
            field_to_check="engagement_style"
        ),
        MatchingCriterion(
            name="vision_alignment",
            weight=0.15,
            description="Shared vision for company direction",
            keywords_positive=["vision", "mission", "product-market fit"],
            field_to_check="primary_goal"
        ),
        MatchingCriterion(
            name="working_style",
            weight=0.10,
            description="Communication and collaboration style",
            keywords_positive=["agile", "collaborative", "remote-friendly"],
            field_to_check="communication_style"
        ),
        MatchingCriterion(
            name="location_fit",
            weight=0.10,
            description="Geographic/remote work compatibility",
            keywords_positive=["remote", "hybrid", "co-located"],
            field_to_check="geography"
        ),
    ],
    complementary_pair=("cofounder", "cofounder")
)


# ============================================================================
# MENTOR ↔ MENTEE CRITERIA
# ============================================================================
MENTOR_MENTEE_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.MENTOR_MENTEE,
    display_name="Mentor ↔ Mentee",
    side_a_wants=[  # What Mentor wants
        "Mentees who are coachable",
        "Clear goals and commitment",
        "Someone with potential",
        "Respect for their time",
        "Willingness to implement feedback",
    ],
    side_b_wants=[  # What Mentee wants
        "Domain expertise in their area",
        "Track record of success",
        "Available and responsive",
        "Genuine interest in helping",
        "Network access",
        "Strategic guidance",
    ],
    criteria=[
        MatchingCriterion(
            name="expertise_alignment",
            weight=0.35,
            description="Mentor's expertise matches mentee's needs",
            keywords_positive=["expert", "years experience", "leadership", "advisor"],
            field_to_check="expertise"
        ),
        MatchingCriterion(
            name="industry_match",
            weight=0.20,
            description="Same or related industry experience",
            keywords_positive=["industry", "sector", "domain"],
            field_to_check="industry"
        ),
        MatchingCriterion(
            name="time_commitment",
            weight=0.20,
            description="Availability and time expectations",
            keywords_positive=["weekly", "monthly", "available", "responsive"],
            field_to_check="engagement_style"
        ),
        MatchingCriterion(
            name="communication_style",
            weight=0.15,
            description="Preferred communication format",
            keywords_positive=["calls", "async", "email", "video"],
            field_to_check="communication_style"
        ),
        MatchingCriterion(
            name="geography_proximity",
            weight=0.10,
            description="Location for in-person meetings if desired",
            keywords_positive=["local", "remote", "virtual"],
            field_to_check="geography"
        ),
    ],
    complementary_pair=("mentor_mentee", "mentee_mentor")
)


# ============================================================================
# RECRUITER ↔ CANDIDATE CRITERIA
# ============================================================================
RECRUITER_CANDIDATE_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.RECRUITER_CANDIDATE,
    display_name="Recruiter ↔ Candidate",
    side_a_wants=[  # What Recruiter/Company wants
        "Required skills and experience",
        "Culture fit",
        "Compensation alignment",
        "Availability to start",
        "Long-term potential",
    ],
    side_b_wants=[  # What Candidate wants
        "Competitive compensation",
        "Growth opportunities",
        "Good culture and team",
        "Remote/hybrid flexibility",
        "Mission alignment",
        "Work-life balance",
    ],
    criteria=[
        MatchingCriterion(
            name="skills_match",
            weight=0.30,
            description="Required skills and experience present",
            keywords_positive=["engineer", "developer", "manager", "executive"],
            field_to_check="skills"
        ),
        MatchingCriterion(
            name="compensation_fit",
            weight=0.25,
            description="Compensation expectations align",
            keywords_positive=["salary", "equity", "compensation", "total comp"],
            field_to_check="compensation"
        ),
        MatchingCriterion(
            name="experience_level",
            weight=0.20,
            description="Years of experience match role level",
            keywords_positive=["senior", "junior", "mid-level", "executive"],
            field_to_check="experience"
        ),
        MatchingCriterion(
            name="location_match",
            weight=0.15,
            description="Location/remote preferences compatible",
            keywords_positive=["remote", "hybrid", "onsite", "relocation"],
            field_to_check="geography"
        ),
        MatchingCriterion(
            name="availability",
            weight=0.10,
            description="Start date and availability",
            keywords_positive=["immediate", "two weeks", "available", "notice period"],
            field_to_check="timeline"
        ),
    ],
    complementary_pair=("talent_seeking", "opportunity_seeking")
)


# ============================================================================
# PARTNER ↔ PARTNER CRITERIA (Strategic Partnership)
# ============================================================================
PARTNER_PARTNER_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.PARTNER_PARTNER,
    display_name="Strategic Partnership",
    side_a_wants=[  # What Partner A wants
        "Strategic alignment",
        "Complementary capabilities",
        "Clear value exchange",
        "Resource and timeline match",
        "Decision-making authority",
    ],
    side_b_wants=[  # What Partner B wants
        "Strategic alignment",
        "Complementary capabilities",
        "Clear value exchange",
        "Resource and timeline match",
        "Decision-making authority",
    ],
    criteria=[
        MatchingCriterion(
            name="strategic_alignment",
            weight=0.30,
            description="Shared strategic goals",
            keywords_positive=["partnership", "strategic", "alliance", "joint"],
            field_to_check="primary_goal"
        ),
        MatchingCriterion(
            name="capability_complement",
            weight=0.25,
            description="Complementary strengths",
            keywords_positive=["technology", "distribution", "market access", "expertise"],
            field_to_check="offerings"
        ),
        MatchingCriterion(
            name="value_exchange",
            weight=0.20,
            description="Clear mutual value",
            keywords_positive=["value", "benefit", "mutual", "win-win"],
            field_to_check="requirements"
        ),
        MatchingCriterion(
            name="resource_fit",
            weight=0.15,
            description="Resource availability",
            keywords_positive=["resources", "team", "budget", "commitment"],
            field_to_check="engagement_style"
        ),
        MatchingCriterion(
            name="timeline_compatibility",
            weight=0.10,
            description="Project timeline alignment",
            keywords_positive=["timeline", "deadline", "milestone"],
            field_to_check="timeline"
        ),
    ],
    complementary_pair=("partnership", "partnership")
)


# ============================================================================
# GENERAL NETWORKING CRITERIA
# ============================================================================
GENERAL_NETWORKING_CRITERIA = ConnectionCriteria(
    connection_type=ConnectionType.GENERAL_NETWORKING,
    display_name="General Networking",
    side_a_wants=[
        "Meaningful connections",
        "Industry insights",
        "Potential future collaboration",
    ],
    side_b_wants=[
        "Meaningful connections",
        "Industry insights",
        "Potential future collaboration",
    ],
    criteria=[
        MatchingCriterion(
            name="industry_overlap",
            weight=0.30,
            description="Shared or related industry",
            keywords_positive=["industry", "sector", "space"],
            field_to_check="industry"
        ),
        MatchingCriterion(
            name="interest_alignment",
            weight=0.25,
            description="Shared interests or focus areas",
            keywords_positive=["interested", "passionate", "focus"],
            field_to_check="focus"
        ),
        MatchingCriterion(
            name="network_complement",
            weight=0.20,
            description="Complementary networks",
            keywords_positive=["network", "connections", "community"],
            field_to_check="offerings"
        ),
        MatchingCriterion(
            name="geography_proximity",
            weight=0.15,
            description="Geographic proximity for meetups",
            keywords_positive=["local", "city", "region"],
            field_to_check="geography"
        ),
        MatchingCriterion(
            name="experience_level",
            weight=0.10,
            description="Similar experience level",
            keywords_positive=["years", "experience", "senior", "junior"],
            field_to_check="experience"
        ),
    ],
    complementary_pair=("general", "general")
)


# ============================================================================
# REGISTRY - All criteria indexed by connection type
# ============================================================================
MATCHING_CRITERIA_REGISTRY: Dict[ConnectionType, ConnectionCriteria] = {
    ConnectionType.INVESTOR_FOUNDER: INVESTOR_FOUNDER_CRITERIA,
    ConnectionType.FOUNDER_CTO: FOUNDER_CTO_CRITERIA,
    ConnectionType.MENTOR_MENTEE: MENTOR_MENTEE_CRITERIA,
    ConnectionType.RECRUITER_CANDIDATE: RECRUITER_CANDIDATE_CRITERIA,
    ConnectionType.PARTNER_PARTNER: PARTNER_PARTNER_CRITERIA,
    ConnectionType.GENERAL_NETWORKING: GENERAL_NETWORKING_CRITERIA,
}


def get_criteria_for_connection(connection_type: str) -> ConnectionCriteria:
    """Get matching criteria for a connection type."""
    try:
        ct = ConnectionType(connection_type)
        return MATCHING_CRITERIA_REGISTRY.get(ct, GENERAL_NETWORKING_CRITERIA)
    except ValueError:
        return GENERAL_NETWORKING_CRITERIA


def get_criteria_weights(connection_type: str) -> Dict[str, float]:
    """Get criteria weights as a simple dict for a connection type."""
    criteria = get_criteria_for_connection(connection_type)
    return {c.name: c.weight for c in criteria.criteria}


def get_all_connection_types() -> List[Dict[str, str]]:
    """List all connection types with display names."""
    return [
        {"code": ct.value, "display_name": MATCHING_CRITERIA_REGISTRY[ct].display_name}
        for ct in ConnectionType
    ]
