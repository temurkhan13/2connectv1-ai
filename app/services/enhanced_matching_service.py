"""
Enhanced matching with bidirectional scoring and intent classification.
"""
import os
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.adapters.postgresql import postgresql_adapter
from app.services.embedding_service import embedding_service
# Import matching criteria from Matching Scenario document (Feb 2026)
from app.services.matching_criteria import (
    get_criteria_for_connection,
    get_criteria_weights,
    ConnectionType,
)

logger = logging.getLogger(__name__)


class MatchIntent(str, Enum):
    """Match intent types with different scoring strategies."""
    INVESTOR_FOUNDER = "investor_founder"      # Investor seeking founders
    FOUNDER_INVESTOR = "founder_investor"      # Founder seeking investment
    MENTOR_MENTEE = "mentor_mentee"            # Mentor seeking mentees
    MENTEE_MENTOR = "mentee_mentor"            # Mentee seeking mentor
    COFOUNDER = "cofounder"                    # Seeking co-founder
    TALENT_SEEKING = "talent_seeking"          # Company seeking talent
    OPPORTUNITY_SEEKING = "opportunity_seeking" # Talent seeking opportunity
    PARTNERSHIP = "partnership"                # B2B partnership
    GENERAL = "general"                        # General networking
    # BUG-040 FIX: Added recruiter and service provider intents
    RECRUITER = "recruiter"                    # Recruiter seeking clients AND candidates
    SERVICE_PROVIDER = "service_provider"      # Consultant/agency seeking clients


class IntentClassifier:
    """Classifies user intent from persona data."""

    # Keywords that indicate each intent type
    # CRITICAL FIX (Mar 27, 2026): Investor vs Founder keywords were overlapping.
    # "fund", "seed", "series", "capital" appeared in INVESTOR_FOUNDER but also
    # match founders seeking funding. This caused founders like Ryan Best to be
    # misclassified as investors, inverting their entire match direction.
    #
    # Fix: INVESTOR_FOUNDER keywords must be investor-SPECIFIC (deploying capital).
    #       FOUNDER_INVESTOR keywords must catch fundraising language broadly.
    INTENT_KEYWORDS = {
        MatchIntent.INVESTOR_FOUNDER: [
            # Words that ONLY an investor would use (deploying capital, not seeking it)
            "invest in", "investor", "angel investor", "vc", "venture capital",
            "portfolio companies", "deal flow", "deploy capital", "back founders",
            "fund startups", "write checks", "check size", "investment thesis",
            "due diligence", "term sheet review", "co-invest"
        ],
        MatchIntent.FOUNDER_INVESTOR: [
            # Words a founder seeking money would use
            "raising", "fundraising", "seeking investment", "looking for funding",
            "need funding", "need capital", "raise capital", "raise a round",
            "seed round", "series a", "series b", "pre-seed round",
            "seeking investors", "looking for investors", "pitch to investors",
            "investor relations", "funding round", "raise money",
            "need investment", "capital raise", "seeking seed", "seeking series",
            "seeking investment capital", "need investment capital",
            "looking for capital"
        ],
        MatchIntent.MENTOR_MENTEE: [
            "mentor others", "guide others", "advise founders", "coach",
            "teach", "share knowledge", "help others", "give back",
            "offer mentorship", "provide guidance", "advisory role"
        ],
        MatchIntent.MENTEE_MENTOR: [
            "learn from", "seeking mentor", "looking for guidance", "need advice",
            "want to learn", "need mentorship", "career guidance",
            "seeking advice", "find a mentor", "looking for a mentor"
        ],
        MatchIntent.COFOUNDER: [
            "co-founder", "cofounder", "founding team",
            "build together", "start together", "technical cofounder",
            "find a cofounder", "looking for cofounder"
        ],
        MatchIntent.TALENT_SEEKING: [
            "hiring", "recruit", "looking for talent", "need engineer",
            "seeking developer", "team expansion", "new hire",
            "build a team", "looking to hire", "seeking talent",
            "need senior engineer", "hiring for", "open role"
        ],
        MatchIntent.OPPORTUNITY_SEEKING: [
            "job", "position", "role", "career",
            "looking for work", "new opportunity", "next role",
            "seeking employment", "job search", "find a role",
            "looking for a job", "open to opportunities"
        ],
        MatchIntent.PARTNERSHIP: [
            "partner", "collaborate", "b2b", "enterprise", "strategic",
            "alliance", "joint venture", "integration",
            "strategic partner", "business development"
        ],
        MatchIntent.RECRUITER: [
            "recruiter", "recruiting", "talent acquisition", "headhunter",
            "staffing", "placing candidates", "hiring for clients",
            "talent partner", "executive search", "recruitment firm"
        ],
        MatchIntent.SERVICE_PROVIDER: [
            "consulting", "consultant", "agency", "service provider",
            "client work", "serving companies", "advisory services",
            "professional services", "boutique firm", "fractional"
        ]
    }

    # Direct mapping from primary_goal slot values to MatchIntent
    PRIMARY_GOAL_MAP = {
        "hire talent": MatchIntent.TALENT_SEEKING,
        "hire": MatchIntent.TALENT_SEEKING,
        "hiring": MatchIntent.TALENT_SEEKING,
        "looking to hire": MatchIntent.TALENT_SEEKING,
        "build team": MatchIntent.TALENT_SEEKING,
        "recruit": MatchIntent.RECRUITER,
        "find co-founder": MatchIntent.COFOUNDER,
        "find cofounder": MatchIntent.COFOUNDER,
        "raise funding": MatchIntent.FOUNDER_INVESTOR,
        "raise capital": MatchIntent.FOUNDER_INVESTOR,
        "raise a round": MatchIntent.FOUNDER_INVESTOR,
        "raise money": MatchIntent.FOUNDER_INVESTOR,
        "fundraising": MatchIntent.FOUNDER_INVESTOR,
        "seeking investment": MatchIntent.FOUNDER_INVESTOR,
        "seeking funding": MatchIntent.FOUNDER_INVESTOR,
        "seeking investors": MatchIntent.FOUNDER_INVESTOR,
        "looking for investors": MatchIntent.FOUNDER_INVESTOR,
        "looking for funding": MatchIntent.FOUNDER_INVESTOR,
        "need funding": MatchIntent.FOUNDER_INVESTOR,
        "need investment": MatchIntent.FOUNDER_INVESTOR,
        "need capital": MatchIntent.FOUNDER_INVESTOR,
        "seed round": MatchIntent.FOUNDER_INVESTOR,
        "series a": MatchIntent.FOUNDER_INVESTOR,
        "invest in startups": MatchIntent.INVESTOR_FOUNDER,
        "investing in": MatchIntent.INVESTOR_FOUNDER,
        "deploy capital": MatchIntent.INVESTOR_FOUNDER,
        "back founders": MatchIntent.INVESTOR_FOUNDER,
        "angel investing": MatchIntent.INVESTOR_FOUNDER,
        "explore partnerships": MatchIntent.PARTNERSHIP,
        "partnerships": MatchIntent.PARTNERSHIP,
        "find mentor": MatchIntent.MENTEE_MENTOR,
        "offer mentorship": MatchIntent.MENTEE_MENTOR,  # They ARE a mentor offering guidance
        "provide mentorship": MatchIntent.MENTEE_MENTOR,
        "give mentorship": MatchIntent.MENTEE_MENTOR,
        "seek mentorship": MatchIntent.MENTOR_MENTEE,   # They WANT a mentor
        "mentorship": MatchIntent.MENTOR_MENTEE,
        "find job": MatchIntent.OPPORTUNITY_SEEKING,
        "find new job": MatchIntent.OPPORTUNITY_SEEKING,
        "find a job": MatchIntent.OPPORTUNITY_SEEKING,
        "looking for a job": MatchIntent.OPPORTUNITY_SEEKING,
        "looking for job": MatchIntent.OPPORTUNITY_SEEKING,
        "job search": MatchIntent.OPPORTUNITY_SEEKING,
        "seeking job": MatchIntent.OPPORTUNITY_SEEKING,
        "seeking employment": MatchIntent.OPPORTUNITY_SEEKING,
        "find new role": MatchIntent.OPPORTUNITY_SEEKING,
        "find role": MatchIntent.OPPORTUNITY_SEEKING,
        "career change": MatchIntent.OPPORTUNITY_SEEKING,
        "new opportunity": MatchIntent.OPPORTUNITY_SEEKING,
        "offer services": MatchIntent.SERVICE_PROVIDER,
        "general networking": MatchIntent.GENERAL,
        "seek networking": MatchIntent.GENERAL,
        "networking": MatchIntent.GENERAL,
        "expand network": MatchIntent.GENERAL,
        "build network": MatchIntent.GENERAL,
        # Founder goals that don't imply investing (Mar 27, 2026)
        "find first customer": MatchIntent.FOUNDER_INVESTOR,
        "validate product": MatchIntent.PARTNERSHIP,
        "expand to new markets": MatchIntent.PARTNERSHIP,
        "launch product": MatchIntent.PARTNERSHIP,
        "scale business": MatchIntent.FOUNDER_INVESTOR,
        "grow revenue": MatchIntent.PARTNERSHIP,
    }

    def classify(self, persona_data: Dict[str, Any]) -> Tuple[MatchIntent, float]:
        """Classify user intent, returns (MatchIntent, confidence_score).

        Priority: primary_goal slot > keyword analysis of persona text.
        The primary_goal is the user's own stated objective — it should always win.
        """
        # PRIORITY 1: Use primary_goal slot if available (highest confidence)
        raw_goal = persona_data.get("primary_goal", "")
        # Handle case where primary_goal is a list (e.g. ['Validate Product', 'Expand to New Markets'])
        if isinstance(raw_goal, list):
            raw_goal = raw_goal[0] if raw_goal else ""
            logger.info(f"[IntentClassifier] primary_goal was a list, using first element: '{raw_goal}'")
        primary_goal = str(raw_goal).lower().strip()
        if primary_goal:
            # Sort by length descending so more specific matches win
            # e.g. "offer mentorship" matches before "mentorship"
            sorted_goals = sorted(self.PRIMARY_GOAL_MAP.items(), key=lambda x: len(x[0]), reverse=True)
            for goal_text, intent in sorted_goals:
                if goal_text in primary_goal or primary_goal in goal_text:
                    # FIX (Mar 30, 2026): Distinguish mentor from mentee
                    # "Seek Mentorship" is ambiguous — check user_type to determine side
                    if intent in (MatchIntent.MENTOR_MENTEE, MatchIntent.MENTEE_MENTOR):
                        user_type = str(persona_data.get("user_type", "")).lower()
                        archetype = str(persona_data.get("archetype", "")).lower()
                        combined_type = f"{user_type} {archetype}"
                        mentor_keywords = ["mentor", "advisor", "coach", "advisory", "guide"]
                        is_mentor = any(kw in combined_type for kw in mentor_keywords)
                        if is_mentor:
                            intent = MatchIntent.MENTEE_MENTOR  # They ARE a mentor, seeking mentees
                            logger.info(f"[IntentClassifier] Mentor detected from user_type '{user_type}' -> mentee_mentor")
                        else:
                            intent = MatchIntent.MENTOR_MENTEE  # They WANT a mentor
                            logger.info(f"[IntentClassifier] Mentee detected from user_type '{user_type}' -> mentor_mentee")
                    logger.info(f"[IntentClassifier] Resolved intent from primary_goal '{primary_goal}' -> {intent.value}")
                    return intent, 0.95

        # PRIORITY 2: Keyword analysis of persona text (fallback)
        text_to_analyze = " ".join([
            str(persona_data.get("what_theyre_looking_for", "")),
            str(persona_data.get("requirements", "")),
            str(persona_data.get("offerings", "")),
            str(persona_data.get("archetype", "")),
            str(persona_data.get("focus", ""))
        ]).lower()

        # Score each intent
        intent_scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_to_analyze)
            if score > 0:
                intent_scores[intent] = score / len(keywords)

        if not intent_scores:
            return MatchIntent.GENERAL, 0.5

        # Tiebreaker priority: when multiple intents score equally,
        # prefer actionable intents over vague ones.
        # e.g., "mentor" can appear in both mentor_mentee and talent_seeking contexts —
        # a founder "mentoring" their team while also hiring should be talent_seeking.
        INTENT_PRIORITY = {
            MatchIntent.TALENT_SEEKING: 6,       # Most actionable — hiring
            MatchIntent.OPPORTUNITY_SEEKING: 6,   # Most actionable — job seeking
            MatchIntent.INVESTOR_FOUNDER: 5,      # Deploying capital
            MatchIntent.FOUNDER_INVESTOR: 5,      # Raising capital
            MatchIntent.RECRUITER: 4,
            MatchIntent.SERVICE_PROVIDER: 4,
            MatchIntent.COFOUNDER: 3,
            MatchIntent.PARTNERSHIP: 3,
            MatchIntent.MENTOR_MENTEE: 2,         # Vague — often misclassified
            MatchIntent.MENTEE_MENTOR: 2,
            MatchIntent.GENERAL: 1,
        }

        # Return highest scoring intent, with priority tiebreaker
        best_intent = max(
            intent_scores,
            key=lambda i: (intent_scores[i], INTENT_PRIORITY.get(i, 0))
        )
        confidence = min(0.95, intent_scores[best_intent] * 2)  # Cap at 0.95

        return best_intent, confidence


@dataclass
class IntentScoringConfig:
    """Configuration for intent-specific match scoring."""
    experience_gap_weight: float = 0.0      # How much experience gap matters
    ideal_experience_gap: int = 0           # Ideal gap in years (for mentor/mentee)
    industry_match_weight: float = 1.0      # Industry alignment importance
    stage_match_weight: float = 1.0         # Company stage importance
    geography_weight: float = 0.5           # Location importance
    bidirectional_required: bool = True     # Both sides must match


INTENT_SCORING_CONFIGS = {
    MatchIntent.INVESTOR_FOUNDER: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.5,  # Investors care about industry focus
        stage_match_weight=3.0,     # Stage (seed/series) is critical — raised from 2.0 to catch seed↔Series A mismatches
        geography_weight=0.3,
        bidirectional_required=True
    ),
    MatchIntent.FOUNDER_INVESTOR: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.5,
        stage_match_weight=3.0,     # Raised from 2.0 — founders need stage-matched investors
        geography_weight=0.3,
        bidirectional_required=True
    ),
    MatchIntent.MENTOR_MENTEE: IntentScoringConfig(
        experience_gap_weight=1.5,  # Experience gap is important
        ideal_experience_gap=10,    # ~10 years ideal gap
        industry_match_weight=1.0,
        stage_match_weight=0.5,
        geography_weight=0.2,       # Remote mentorship OK
        bidirectional_required=True
    ),
    MatchIntent.MENTEE_MENTOR: IntentScoringConfig(
        experience_gap_weight=1.5,
        ideal_experience_gap=10,
        industry_match_weight=1.0,
        stage_match_weight=0.5,
        geography_weight=0.2,
        bidirectional_required=True
    ),
    MatchIntent.COFOUNDER: IntentScoringConfig(
        experience_gap_weight=0.3,  # Some gap OK, not too much
        ideal_experience_gap=3,
        industry_match_weight=1.5,
        stage_match_weight=0.5,     # Both early stage anyway
        geography_weight=1.0,       # Co-location matters
        bidirectional_required=True
    ),
    MatchIntent.TALENT_SEEKING: IntentScoringConfig(
        experience_gap_weight=0.5,
        industry_match_weight=1.0,
        stage_match_weight=0.8,
        geography_weight=0.7,
        bidirectional_required=True
    ),
    MatchIntent.OPPORTUNITY_SEEKING: IntentScoringConfig(
        experience_gap_weight=0.5,
        industry_match_weight=1.0,
        stage_match_weight=0.8,
        geography_weight=0.7,
        bidirectional_required=True
    ),
    MatchIntent.PARTNERSHIP: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.2,
        stage_match_weight=1.0,
        geography_weight=0.5,
        bidirectional_required=True
    ),
    MatchIntent.GENERAL: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.0,
        stage_match_weight=0.5,
        geography_weight=0.5,
        bidirectional_required=True
    ),
    # BUG-040 FIX: Recruiter scoring - industry matters most, can match across stages
    MatchIntent.RECRUITER: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.5,    # Industry alignment is key
        stage_match_weight=0.8,       # Stage matters for company clients
        geography_weight=0.6,         # Can work remotely but local helps
        bidirectional_required=False  # Recruiters can initiate matches
    ),
    # BUG-040 FIX: Service provider scoring - similar to partnership
    # Industry weight raised 1.3→2.0: out-of-industry consultants are a fundamental quality failure
    MatchIntent.SERVICE_PROVIDER: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=2.0,
        stage_match_weight=0.7,
        geography_weight=0.5,
        bidirectional_required=False  # Service providers seek clients
    )
}


@dataclass
class BidirectionalMatch:
    """Result of a bidirectional match calculation."""
    user_id: str
    forward_score: float          # My requirements → Their offerings
    reverse_score: float          # My offerings → Their requirements
    combined_score: float         # Geometric mean (core_score input)
    intent_match_quality: float   # How well intents align (0-1)
    activity_boost: float         # Boost from user activity
    temporal_boost: float         # Boost from recency
    final_score: float            # All factors combined
    match_reasons: List[str]      # Human-readable reasons
    potential_gaps: List[str]     # Potential issues
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Formula component scores (added for analysis)
    dimension_score: float = 0.5  # Layer 2: Dimensional alignment
    signal_score: float = 0.86    # Layer 4: Activity + recency combined


class EnhancedMatchingService:
    """Bidirectional matching with intent classification and activity weighting."""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.base_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

        # Activity decay settings
        self.activity_decay_days = int(os.getenv("ACTIVITY_DECAY_DAYS", "30"))
        self.new_user_boost_days = int(os.getenv("NEW_USER_BOOST_DAYS", "14"))

        # Cache settings (for pre-computed matches)
        self._match_cache: Dict[str, Dict] = {}
        self._cache_ttl_seconds = int(os.getenv("MATCH_CACHE_TTL", "3600"))

        # Hard filter settings (Feb 2026)
        self.enforce_hard_dealbreakers = os.getenv("ENFORCE_HARD_DEALBREAKERS", "true").lower() == "true"
        self.block_same_objective = os.getenv("BLOCK_SAME_OBJECTIVE", "true").lower() == "true"

    def _check_dealbreaker_violation(self, user_dealbreakers: List[str], candidate_persona) -> Tuple[bool, List[str]]:
        """
        Check if candidate violates any of user's hard dealbreakers.

        Returns:
            Tuple of (has_violation: bool, violated_dealbreakers: List[str])
        """
        if not user_dealbreakers:
            return False, []

        violated = []

        # Get candidate's profile text for matching
        candidate_text = " ".join([
            str(getattr(candidate_persona, "focus", "") or ""),
            str(getattr(candidate_persona, "archetype", "") or ""),
            str(getattr(candidate_persona, "requirements", "") or ""),
            str(getattr(candidate_persona, "offerings", "") or ""),
            str(getattr(candidate_persona, "what_theyre_looking_for", "") or ""),
        ]).lower()

        # Check each dealbreaker
        for dealbreaker in user_dealbreakers:
            dealbreaker_lower = str(dealbreaker).lower().strip()
            if not dealbreaker_lower:
                continue

            # Check if dealbreaker keyword appears in candidate's profile
            if dealbreaker_lower in candidate_text:
                violated.append(dealbreaker)
                continue

            # Check for common variations
            variations = self._get_dealbreaker_variations(dealbreaker_lower)
            for variation in variations:
                if variation in candidate_text:
                    violated.append(dealbreaker)
                    break

        return len(violated) > 0, violated

    def _get_dealbreaker_variations(self, dealbreaker: str) -> List[str]:
        """Get common variations of a dealbreaker term."""
        variations_map = {
            "crypto": ["cryptocurrency", "blockchain", "web3", "defi", "nft"],
            "blockchain": ["crypto", "cryptocurrency", "web3", "defi"],
            "gambling": ["casino", "betting", "gaming"],
            "tobacco": ["cigarette", "smoking", "vaping"],
            "alcohol": ["liquor", "beer", "wine", "spirits"],
            "early stage": ["pre-seed", "seed", "early-stage", "ideation"],
            "late stage": ["series c", "series d", "growth stage", "late-stage"],
            "remote": ["distributed", "work from home", "virtual"],
            "consumer": ["b2c", "direct to consumer", "d2c"],
            "enterprise": ["b2b", "corporate", "business to business"],
        }
        return variations_map.get(dealbreaker, [])

    def _is_same_objective_blocked(self, user_intent: MatchIntent, candidate_intent: MatchIntent) -> bool:
        """
        UPGRADED (Mar 30, 2026): Only block truly same-need pairs.

        Old model blocked 20+ pairs assuming rigid role-based incompatibility
        (e.g., investor↔service_provider, mentee↔partnership). Wrong — users
        can have any objective regardless of role.

        New model: ONLY block when both users have the exact same unidirectional
        need and neither can provide what the other seeks.
        Everything else is allowed — let IQ scores and profile quality decide ranking.
        """
        same_need_pairs = {
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.INVESTOR_FOUNDER),      # Both deploying capital, neither raising
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.FOUNDER_INVESTOR),      # Both raising, neither investing
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.OPPORTUNITY_SEEKING), # Both seeking jobs, neither hiring
            (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTEE_MENTOR),            # Both seeking mentors, neither mentoring
            (MatchIntent.TALENT_SEEKING, MatchIntent.TALENT_SEEKING),          # Both hiring, neither is a candidate
        }

        return (user_intent, candidate_intent) in same_need_pairs

    def _get_user_dealbreakers(self, persona) -> List[str]:
        """Extract dealbreakers from user's onboarding data."""
        if not persona:
            return []

        # Try different field names that might contain dealbreakers
        dealbreakers = []
        for field_name in ['dealbreakers', 'exclusions', 'not_interested_in', 'avoid']:
            value = getattr(persona, field_name, None)
            if value:
                if isinstance(value, list):
                    dealbreakers.extend(value)
                elif isinstance(value, str):
                    # Split by comma or semicolon
                    dealbreakers.extend([d.strip() for d in value.replace(';', ',').split(',') if d.strip()])

        return dealbreakers

    def find_bidirectional_matches(
        self,
        user_id: str,
        threshold: float = None,
        limit: int = 50,
        include_explanations: bool = True
    ) -> List[BidirectionalMatch]:
        """Find matches where both parties benefit."""
        threshold = threshold or self.base_threshold

        try:
            # Get user's embeddings
            user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
            if not user_embeddings or len(user_embeddings) == 0:
                logger.warning(f"No embeddings found for user {user_id}")
                return []

            # Get user's persona and onboarding slots for intent classification
            from app.adapters.supabase_profiles import UserProfile
            try:
                user_profile = UserProfile.get(user_id)
                user_persona = user_profile.persona
                user_intent, user_intent_confidence = self._classify_user_intent(user_persona, user_id)
            except Exception as e:
                logger.warning(f"Could not get persona for {user_id}: {e}")
                user_intent = MatchIntent.GENERAL
                user_intent_confidence = 0.5
                user_persona = None

            logger.info(f"User {user_id} classified as {user_intent.value} (confidence: {user_intent_confidence:.2f})")

            # Get user's dealbreakers for hard filtering
            user_dealbreakers = self._get_user_dealbreakers(user_persona) if user_persona else []
            if user_dealbreakers and self.enforce_hard_dealbreakers:
                logger.info(f"User {user_id} has {len(user_dealbreakers)} dealbreakers: {user_dealbreakers[:5]}...")

            # Step 1: Find forward matches (my requirements → their offerings)
            forward_matches = {}
            if user_embeddings.get('requirements'):
                req_vector = user_embeddings['requirements']['vector_data']
                forward_results = postgresql_adapter.find_similar_users(
                    query_vector=req_vector,
                    embedding_type='offerings',
                    threshold=threshold * 0.5,  # Lower threshold, will filter later
                    exclude_user_id=user_id
                )
                for match in forward_results:
                    forward_matches[match['user_id']] = match['similarity_score']

            # Step 2: Find reverse matches (my offerings → their requirements)
            reverse_matches = {}
            if user_embeddings.get('offerings'):
                off_vector = user_embeddings['offerings']['vector_data']
                reverse_results = postgresql_adapter.find_similar_users(
                    query_vector=off_vector,
                    embedding_type='requirements',
                    threshold=threshold * 0.5,
                    exclude_user_id=user_id
                )
                for match in reverse_results:
                    reverse_matches[match['user_id']] = match['similarity_score']

            # Step 3: Find bidirectional matches (appear in both)
            bidirectional_user_ids = set(forward_matches.keys()) & set(reverse_matches.keys())

            logger.info(
                f"Found {len(forward_matches)} forward, {len(reverse_matches)} reverse, "
                f"{len(bidirectional_user_ids)} bidirectional for user {user_id}"
            )

            # Step 4: Score and rank bidirectional matches
            bidirectional_matches = []
            scoring_config = INTENT_SCORING_CONFIGS.get(user_intent, INTENT_SCORING_CONFIGS[MatchIntent.GENERAL])

            # Counters for logging
            dealbreaker_filtered = 0
            same_objective_filtered = 0

            for match_user_id in bidirectional_user_ids:
                forward_score = forward_matches[match_user_id]
                reverse_score = reverse_matches[match_user_id]

                # Geometric mean (rewards balanced mutual matches)
                combined_score = math.sqrt(forward_score * reverse_score)

                # Fwd-Rev asymmetry penalty (Mar 27, 2026)
                fwd_rev_diff = abs(forward_score - reverse_score)
                if fwd_rev_diff > 0.15:
                    asymmetry_penalty = 1.0 - min(0.3, (fwd_rev_diff - 0.15) * 1.5)
                    combined_score *= asymmetry_penalty

                # Skip if below threshold
                if combined_score < threshold:
                    continue

                # Get matched user's data
                try:
                    match_profile = UserProfile.get(match_user_id)
                    match_persona = match_profile.persona
                    match_intent, match_intent_confidence = self._classify_user_intent(match_persona, match_user_id)
                except Exception:
                    match_persona = None
                    match_intent = MatchIntent.GENERAL
                    match_intent_confidence = 0.5

                # NOTE: Removed archetype-based intent override (Mar 30, 2026).
                # A founder who says "Find New Job" IS an opportunity seeker.
                # Primary goal always takes precedence over archetype.

                # HARD FILTER 1: Dealbreaker check (Feb 2026)
                if self.enforce_hard_dealbreakers and user_dealbreakers and match_persona:
                    has_violation, violated = self._check_dealbreaker_violation(user_dealbreakers, match_persona)
                    if has_violation:
                        dealbreaker_filtered += 1
                        logger.debug(f"Filtered {match_user_id}: dealbreaker violation {violated}")
                        continue

                # HARD FILTER 2: Same-objective blocking (Feb 2026)
                if self.block_same_objective:
                    if self._is_same_objective_blocked(user_intent, match_intent):
                        same_objective_filtered += 1
                        logger.debug(f"Filtered {match_user_id}: same objective ({user_intent.value})")
                        continue

                # PENALTY: Same-persona mirror match detection
                # Two users with identical archetypes have the same gaps — they can't help each other
                if user_persona and match_persona:
                    user_archetype = getattr(user_persona, 'archetype', None) or ''
                    match_archetype = getattr(match_persona, 'archetype', None) or ''
                    if user_archetype and match_archetype and user_archetype.lower() == match_archetype.lower():
                        combined_score *= 0.70  # 30% penalty for mirror matches
                        logger.debug(f"Mirror match penalty for {match_user_id}: same archetype '{user_archetype}'")

                # Same-ROLE mirror match penalty (expanded from archetype-only)
                # Corp dev matched to corp dev, advisor matched to advisor, etc.
                if user_persona and match_persona:
                    user_desig = (getattr(user_persona, 'designation', '') or '').lower()
                    cand_desig = (getattr(match_persona, 'designation', '') or '').lower()
                    # Check for role keyword overlap in designation
                    role_groups = [
                        ['corporate development', 'corp dev', 'business development', 'acquisitions'],
                        ['advisor', 'board member', 'advisory'],
                        ['consultant', 'consulting'],
                    ]
                    for group in role_groups:
                        user_has = any(kw in user_desig for kw in group)
                        cand_has = any(kw in cand_desig for kw in group)
                        if user_has and cand_has:
                            combined_score *= 0.50  # 50% penalty for same-role matches
                            logger.debug(f"Same-role penalty for {match_user_id}: designation overlap in group {group}")
                            break

                # Role-overlap penalty: a service provider matched with someone who already holds
                # that role gets penalized. E.g., fractional CTO matched with an existing CTO.
                if user_intent == MatchIntent.SERVICE_PROVIDER and match_persona:
                    user_offerings = (getattr(user_persona, 'offerings', '') or '').lower() if user_persona else ''
                    cand_designation = (getattr(match_persona, 'designation', '') or '').lower()
                    cand_arch = (getattr(match_persona, 'archetype', '') or '').lower()

                    # Check if the candidate already holds the role the service provider offers
                    role_keywords = []
                    if 'cto' in user_offerings or 'technical leadership' in user_offerings:
                        role_keywords = ['cto', 'chief technology', 'vp engineering', 'head of engineering']
                    elif 'cmo' in user_offerings or 'marketing' in user_offerings:
                        role_keywords = ['cmo', 'chief marketing', 'vp marketing', 'head of marketing']
                    elif 'cfo' in user_offerings or 'financial' in user_offerings:
                        role_keywords = ['cfo', 'chief financial', 'vp finance', 'head of finance']

                    if role_keywords:
                        cand_text = f"{cand_designation} {cand_arch}"
                        if any(kw in cand_text for kw in role_keywords):
                            combined_score *= 0.50  # 50% penalty — they already have this role
                            logger.debug(f"Role-overlap penalty for {match_user_id}: candidate already holds offered role")

                # Calculate intent match quality
                intent_quality = self._calculate_intent_match_quality(
                    user_intent, match_intent, user_intent_confidence, match_intent_confidence
                )

                # Calculate dimensional alignment (NEW — uses all 15 stored embeddings)
                dimension_score = self._calculate_dimensional_score(
                    user_id, match_user_id, scoring_config
                )

                # Calculate temporal and activity boosts
                activity_boost = self._calculate_activity_boost(match_profile if match_persona else None)
                temporal_boost = self._calculate_temporal_boost(match_profile if match_persona else None)

                # Seniority bonus for mentorship matches
                seniority_bonus = 1.0
                if user_intent in (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTOR_MENTEE):
                    seniority_bonus = self._calculate_mentorship_seniority_bonus(match_persona)

                # Product launch relevance bonus
                launch_bonus = self._calculate_product_launch_relevance_bonus(user_persona, match_persona)
                if launch_bonus > 1.0:
                    seniority_bonus = max(seniority_bonus, launch_bonus)  # Use the higher bonus, don't stack

                # Calculate final score with all factors (additive weighted)
                final_score, dim_score_stored, signal_score_stored = self._calculate_final_score(
                    combined_score=combined_score,
                    intent_quality=intent_quality,
                    activity_boost=activity_boost,
                    temporal_boost=temporal_boost,
                    config=scoring_config,
                    dimension_score=dimension_score,
                    user_intent=user_intent,
                    match_intent=match_intent,
                    seniority_bonus=seniority_bonus,
                )

                # Generate explanations
                match_reasons = []
                potential_gaps = []

                if include_explanations:
                    match_reasons, potential_gaps = self._generate_match_explanation(
                        user_persona=user_persona,
                        match_persona=match_persona,
                        user_intent=user_intent,
                        match_intent=match_intent,
                        forward_score=forward_score,
                        reverse_score=reverse_score
                    )

                bidirectional_matches.append(BidirectionalMatch(
                    user_id=match_user_id,
                    forward_score=forward_score,
                    reverse_score=reverse_score,
                    combined_score=combined_score,
                    intent_match_quality=intent_quality,
                    activity_boost=activity_boost,
                    temporal_boost=temporal_boost,
                    final_score=final_score,
                    match_reasons=match_reasons,
                    potential_gaps=potential_gaps,
                    metadata={
                        "user_intent": user_intent.value,
                        "match_intent": match_intent.value,
                        "scoring_config": scoring_config.__class__.__name__
                    },
                    dimension_score=dim_score_stored,
                    signal_score=signal_score_stored
                ))

            # Log filter results
            if dealbreaker_filtered > 0 or same_objective_filtered > 0:
                logger.info(
                    f"Hard filters applied for {user_id}: "
                    f"{dealbreaker_filtered} dealbreaker violations, "
                    f"{same_objective_filtered} same-objective blocks"
                )

            # Tiered sort: for transactional intents, perfect complementary matches
            # (IQ>=0.95) always rank above non-complementary, then by score within tier.
            transactional = user_intent in (MatchIntent.FOUNDER_INVESTOR, MatchIntent.INVESTOR_FOUNDER,
                                            MatchIntent.TALENT_SEEKING, MatchIntent.OPPORTUNITY_SEEKING)
            if transactional:
                bidirectional_matches.sort(key=lambda m: (-int(m.intent_match_quality >= 0.95), -m.final_score))
            else:
                bidirectional_matches.sort(key=lambda m: m.final_score, reverse=True)
            logger.info(f"Returning {len(bidirectional_matches[:limit])} matches for {user_id}")
            return bidirectional_matches[:limit]

        except Exception as e:
            logger.error(f"Error finding bidirectional matches for {user_id}: {e}")
            return []

    def _classify_user_intent(self, persona, user_id: str = None) -> Tuple[MatchIntent, float]:
        """Classify user intent from persona + onboarding slots.

        FIX (Mar 30, 2026): Also fetches primary_goal and user_type from
        onboarding_answers for accurate classification. Previously only
        used persona fields, resulting in low-confidence classifications
        and the scheduled cron returning 0 matches for all users.
        """
        if not persona:
            return MatchIntent.GENERAL, 0.5

        # Fetch primary_goal and user_type from onboarding for accurate classification
        primary_goal = ""
        user_type_slot = ""
        if user_id:
            try:
                from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
                adapter = SupabaseOnboardingAdapter()
                slots = adapter.get_user_slots_sync(user_id)
                if isinstance(slots, dict):
                    if "primary_goal" in slots:
                        pg = slots["primary_goal"]
                        primary_goal = pg.get("value", "") if isinstance(pg, dict) else str(pg)
                    if "user_type" in slots:
                        ut = slots["user_type"]
                        user_type_slot = ut.get("value", "") if isinstance(ut, dict) else str(ut)
            except Exception:
                pass

        persona_dict = {
            "primary_goal": primary_goal,
            "user_type": user_type_slot,
            "what_theyre_looking_for": getattr(persona, "what_theyre_looking_for", ""),
            "requirements": getattr(persona, "requirements", ""),
            "offerings": getattr(persona, "offerings", ""),
            "archetype": getattr(persona, "archetype", ""),
            "focus": getattr(persona, "focus", "")
        }

        return self.intent_classifier.classify(persona_dict)

    def _calculate_intent_match_quality(
        self,
        user_intent: MatchIntent,
        match_intent: MatchIntent,
        user_confidence: float,
        match_confidence: float
    ) -> float:
        """Calculate how well two user intents complement each other.

        UPGRADED (Mar 30, 2026): IQ = 1.0 for all non-blocked pairs.

        IQ only answers: "does this match type serve the user's objective?"
        If YES → 1.0. If same-need → blocked before reaching here.

        Profile quality (embedding similarity) handles whether the SPECIFIC
        person is right. A founder wanting a CTO cofounder won't match another
        business founder because their requirement/offering embeddings won't
        align — that's the embeddings' job, not IQ's.

        Same-need pairs (both raising, both job seeking, etc.) are blocked by
        _is_same_objective_blocked() before scoring, so everything here is valid.
        """
        return 1.0

    # Senior designations that make someone a strong mentor candidate
    SENIOR_DESIGNATIONS = [
        'vp ', 'vice president', 'director', 'head of', 'chief',
        'cto', 'cmo', 'cfo', 'coo', 'ceo', 'cpo', 'cro',
        'principal', 'partner', 'managing director', 'founder',
        'svp', 'evp', 'senior vice president', 'executive',
    ]

    def _calculate_mentorship_seniority_bonus(self, match_persona) -> float:
        """
        Score bonus for matches with senior designations when user is a mentee.
        Returns a multiplier: 1.0 (no bonus) to 1.30 (strong senior match).
        """
        if not match_persona:
            return 1.0

        designation = (getattr(match_persona, 'designation', '') or '').lower()
        archetype = (getattr(match_persona, 'archetype', '') or '').lower()
        combined = f"{designation} {archetype}"

        # Strong senior signal — VP/Director/C-level
        strong_senior = ['vp ', 'vice president', 'director', 'head of',
                         'cto', 'cmo', 'cfo', 'coo', 'ceo', 'cpo',
                         'svp', 'evp', 'principal', 'managing director']
        if any(kw in combined for kw in strong_senior):
            return 1.30  # 30% boost

        # Moderate senior signal — experienced professionals
        moderate_senior = ['senior', 'lead', 'manager', 'partner', 'founder']
        if any(kw in combined for kw in moderate_senior):
            return 1.15  # 15% boost

        return 1.0  # No bonus for junior/unclear roles

    # GTM/sales/distribution keywords that are highly relevant for product launch users
    PRODUCT_LAUNCH_KEYWORDS = [
        'go-to-market', 'gtm', 'sales', 'distribution', 'channel',
        'revenue operations', 'revenue ops', 'growth', 'marketing',
        'product marketing', 'demand gen', 'customer acquisition',
        'business development', 'partnerships', 'enterprise sales',
        'b2b sales', 'market entry', 'launch', 'scaling',
    ]

    def _calculate_product_launch_relevance_bonus(self, user_persona, match_persona) -> float:
        """
        Bonus for matches whose expertise is highly relevant to product launch.
        Only applies when user's goal is product launch.
        Returns multiplier: 1.0 (no bonus) to 1.25 (strong GTM relevance).
        """
        if not match_persona:
            return 1.0

        # Check user's goal
        user_goal = ''
        if user_persona:
            user_goal = (getattr(user_persona, 'primary_goal', '') or '').lower()
        if 'launch' not in user_goal and 'product' not in user_goal:
            return 1.0

        # Check match's focus/offerings for GTM keywords
        match_focus = (getattr(match_persona, 'focus', '') or '').lower()
        match_offerings = (getattr(match_persona, 'offerings', '') or '').lower()
        match_designation = (getattr(match_persona, 'designation', '') or '').lower()
        combined = f"{match_focus} {match_offerings} {match_designation}"

        matches = sum(1 for kw in self.PRODUCT_LAUNCH_KEYWORDS if kw in combined)
        if matches >= 3:
            return 1.25  # Strong GTM relevance
        elif matches >= 1:
            return 1.15  # Some GTM relevance
        return 1.0

    def _calculate_activity_boost(self, user_profile) -> float:
        """Active users get a boost, inactive users get a penalty."""
        if not user_profile:
            return 1.0

        try:
            # Check last updated time from profile
            last_updated = getattr(user_profile.profile, 'updated_at', None)
            if not last_updated:
                return 1.0

            days_since_active = (datetime.utcnow() - last_updated).days

            # Exponential decay
            decay_rate = 0.5 / self.activity_decay_days  # 50% decay over decay period
            activity_boost = math.exp(-decay_rate * days_since_active)

            # Clamp between 0.5 and 1.5
            return max(0.5, min(1.5, 0.5 + activity_boost))

        except Exception:
            return 1.0

    def _calculate_temporal_boost(self, user_profile) -> float:
        """New users get a boost to help them get initial matches."""
        if not user_profile:
            return 1.0

        try:
            created_at = getattr(user_profile.profile, 'created_at', None)
            if not created_at:
                return 1.0

            days_since_created = (datetime.utcnow() - created_at).days

            if days_since_created <= self.new_user_boost_days:
                # New user boost: starts at 1.3, decays to 1.0 over boost period
                boost = 1.3 - (0.3 * days_since_created / self.new_user_boost_days)
                return boost

            return 1.0

        except Exception:
            return 1.0

    def _calculate_dimensional_score(
        self,
        user_id: str,
        candidate_id: str,
        config: IntentScoringConfig
    ) -> float:
        """Calculate dimensional alignment score using all stored embeddings.

        UPGRADED (Mar 2026): Uses all 15 non-core embeddings with per-intent weights.
        Previously these embeddings were generated and stored but NEVER used in scoring.
        """
        try:
            user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
            candidate_embeddings = postgresql_adapter.get_user_embeddings(candidate_id)
        except Exception as e:
            logger.warning(f"Could not fetch embeddings for dimensional scoring: {e}")
            return 0.5  # Neutral fallback

        # Map config weights to dimension names
        weight_mapping = {
            "industry_combined": config.industry_match_weight * 2.0,  # Doubled — industry mismatch is a fundamental quality failure
            "focus_slot_industry_focus": config.industry_match_weight * 2.0,
            "stage_combined": config.stage_match_weight,
            "focus_slot_company_stage": config.stage_match_weight,
            "focus_slot_geography": config.geography_weight,
        }

        # All dimension embedding types to compare (excluding requirements/offerings — those are in core_score)
        dimension_types = [
            "skills_combined", "industry_combined", "stage_combined",
            "culture_combined", "traction_combined", "market_combined",
            "team_combined", "funding_combined",
            "focus_slot_geography", "focus_slot_timeline",
            "focus_slot_engagement_style", "focus_slot_funding_need",
            "focus_slot_company_stage", "focus_slot_industry_focus",
            "focus_slot_dealbreakers",
        ]

        weighted_total = 0.0
        weight_sum = 0.0

        for dim in dimension_types:
            user_emb = user_embeddings.get(dim)
            candidate_emb = candidate_embeddings.get(dim)

            if not user_emb or not candidate_emb:
                continue

            user_vec = user_emb.get('vector_data')
            candidate_vec = candidate_emb.get('vector_data')

            if not user_vec or not candidate_vec:
                continue

            # Cosine similarity
            try:
                similarity = self._cosine_similarity(user_vec, candidate_vec)
            except Exception:
                continue

            weight = weight_mapping.get(dim, 1.0)
            weighted_total += similarity * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.5  # No dimensional data available — neutral fallback

        return weighted_total / weight_sum

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_final_score(
        self,
        combined_score: float,
        intent_quality: float,
        activity_boost: float,
        temporal_boost: float,
        config: IntentScoringConfig,
        dimension_score: float = 0.5,
        user_intent: MatchIntent = None,
        match_intent: MatchIntent = None,
        seniority_bonus: float = 1.0
    ) -> tuple:
        """Combine all scoring factors into final score.

        UPGRADED (Mar 27, 2026): Intent as hard multiplier + asymmetry penalty + spread.
        UPGRADED (Mar 29, 2026): Mentorship seniority bonus + softened ceiling for mentees.

        Intent < 0.5 now CRUSHES the score (multiplier) instead of mild additive penalty.
        Forward-reverse asymmetry penalizes one-sided matches.
        Power scaling spreads scores apart for better ranking signal.

        Returns: (final_score, dimension_score, signal_score) tuple for analysis storage
        """
        import math as _math

        # Normalize activity and temporal to 0-1 range for additive formula
        activity_score = max(0.7, min(1.0, activity_boost))
        temporal_score = max(0.8, min(1.0, temporal_boost))
        signal_score = (activity_score * 0.6) + (temporal_score * 0.4)

        # Base formula (without intent)
        base_total = (
            combined_score   * 0.40 +    # Layer 1: Bidirectional core (boosted from 0.35)
            dimension_score  * 0.35 +    # Layer 2: Dimensional alignment (boosted from 0.30)
            signal_score     * 0.10      # Layer 3: Activity + recency
        )

        # Apply seniority bonus for mentorship matches (boosts base before intent)
        if seniority_bonus > 1.0:
            base_total = base_total * seniority_bonus

        # Intent multiplier: only truly bad pairs (< 0.50) get crushed.
        # Pairs in 0.50-0.65 range (service/partnership) get mild scaling, not destruction.
        if intent_quality < 0.50:
            intent_multiplier = intent_quality / 0.50  # Maps 0→0, 0.50→1.0
            final = base_total * intent_multiplier + intent_quality * 0.15
        elif intent_quality >= 0.95:
            # PERFECT pair: boost so good matches break above 75%
            final = base_total * 1.15 + intent_quality * 0.15
        else:
            # GOOD/MODERATE pair: standard additive formula
            final = base_total + intent_quality * 0.15

        # Score spread enhancement (power scaling)
        final = max(0.0, min(1.0, final))
        if final > 0:
            final = final ** 0.85

        # Intent ceiling — different behavior for mentorship vs transactional
        is_mentee = user_intent in (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTOR_MENTEE) if user_intent else False
        is_general = user_intent == MatchIntent.GENERAL if user_intent else False

        if (is_mentee or is_general) and intent_quality < 0.95:
            # SOFTENED ceiling for mentorship: profile quality can push past IQ
            # A VP of Product (high base) should score higher than a junior engineer
            # even when both have the same IQ score
            # Blend: 60% IQ ceiling + 40% profile quality
            hard_ceiling = min(final, intent_quality)
            soft_ceiling = intent_quality * 0.60 + base_total * 0.40
            final = max(hard_ceiling, min(final, soft_ceiling))
        else:
            # Standard hard ceiling for transactional intents
            final = min(final, intent_quality)

        return final, dimension_score, signal_score

    def _generate_match_explanation(
        self,
        user_persona,
        match_persona,
        user_intent: MatchIntent,
        match_intent: MatchIntent,
        forward_score: float,
        reverse_score: float
    ) -> Tuple[List[str], List[str]]:
        """Generate match reasons and potential gaps."""
        reasons = []
        gaps = []

        # Score-based reasons
        if forward_score >= 0.8:
            reasons.append("Excellent alignment: their offerings match what you're looking for")
        elif forward_score >= 0.6:
            reasons.append("Strong alignment between your requirements and their offerings")

        if reverse_score >= 0.8:
            reasons.append("Your offerings are highly relevant to what they need")
        elif reverse_score >= 0.6:
            reasons.append("Good fit: you offer what they're seeking")

        # Intent-based reasons
        intent_explanations = {
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.FOUNDER_INVESTOR):
                "Perfect investor-founder match: they're seeking investment, you're looking for deal flow",
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.INVESTOR_FOUNDER):
                "Ideal funding opportunity: they invest in founders like you",
            (MatchIntent.MENTOR_MENTEE, MatchIntent.MENTEE_MENTOR):
                "Great mentorship match: you can guide their career growth",
            (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTOR_MENTEE):
                "Perfect mentor connection: they have experience to share with you",
            (MatchIntent.TALENT_SEEKING, MatchIntent.OPPORTUNITY_SEEKING):
                "Hiring match: your open role aligns with their career goals",
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.TALENT_SEEKING):
                "Career opportunity: they're hiring for roles matching your skills",
            (MatchIntent.COFOUNDER, MatchIntent.COFOUNDER):
                "Co-founder potential: you're both seeking founding partners"
        }

        pair = (user_intent, match_intent)
        if pair in intent_explanations:
            reasons.append(intent_explanations[pair])

        # Extract persona-based insights
        if user_persona and match_persona:
            try:
                user_industry = str(getattr(user_persona, 'focus', '')).lower()
                match_industry = str(getattr(match_persona, 'focus', '')).lower()

                # Check for industry overlap
                if user_industry and match_industry:
                    if any(word in match_industry for word in user_industry.split() if len(word) > 4):
                        reasons.append(f"Industry alignment in focus areas")
            except Exception:
                pass

        # Potential gaps
        if forward_score < 0.5:
            gaps.append("Their offerings may not fully meet your requirements")
        if reverse_score < 0.5:
            gaps.append("Your offerings may not fully address their needs")
        if abs(forward_score - reverse_score) > 0.3:
            gaps.append("Match strength is unbalanced - one side benefits more")

        return reasons, gaps

    def get_match_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's matches."""
        try:
            matches = self.find_bidirectional_matches(user_id, include_explanations=False)

            if not matches:
                return {
                    "total_matches": 0,
                    "average_score": 0,
                    "score_distribution": {}
                }

            scores = [m.final_score for m in matches]

            return {
                "total_matches": len(matches),
                "average_score": sum(scores) / len(scores),
                "top_score": max(scores),
                "score_distribution": {
                    "excellent (0.8+)": len([s for s in scores if s >= 0.8]),
                    "strong (0.6-0.8)": len([s for s in scores if 0.6 <= s < 0.8]),
                    "good (0.4-0.6)": len([s for s in scores if 0.4 <= s < 0.6]),
                    "moderate (<0.4)": len([s for s in scores if s < 0.4])
                }
            }
        except Exception as e:
            logger.error(f"Error getting match stats: {e}")
            return {"error": str(e)}

    def format_matches_for_api(
        self,
        matches: List[BidirectionalMatch]
    ) -> List[Dict[str, Any]]:
        """Format bidirectional matches for API response."""
        return [
            {
                "user_id": m.user_id,
                "scores": {
                    "forward": round(m.forward_score, 3),
                    "reverse": round(m.reverse_score, 3),
                    "combined": round(m.combined_score, 3),
                    "final": round(m.final_score, 3)
                },
                "quality_factors": {
                    "intent_match": round(m.intent_match_quality, 3),
                    "activity_boost": round(m.activity_boost, 3),
                    "temporal_boost": round(m.temporal_boost, 3)
                },
                "explanation": {
                    "match_reasons": m.match_reasons,
                    "potential_gaps": m.potential_gaps
                },
                "metadata": m.metadata
            }
            for m in matches
        ]


enhanced_matching_service = EnhancedMatchingService()
