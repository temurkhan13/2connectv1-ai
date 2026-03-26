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
    INTENT_KEYWORDS = {
        MatchIntent.INVESTOR_FOUNDER: [
            "invest", "investor", "angel", "vc", "venture capital", "fund",
            "portfolio", "deal flow", "seed", "series", "pre-seed", "capital"
        ],
        MatchIntent.FOUNDER_INVESTOR: [
            "raising", "fundraising", "seeking investment", "looking for funding",
            "seed round", "series a", "need capital", "investor relations"
        ],
        MatchIntent.MENTOR_MENTEE: [
            "mentor", "guide", "advise", "coach", "teach", "share knowledge",
            "help others", "give back", "mentorship", "guidance"
        ],
        MatchIntent.MENTEE_MENTOR: [
            "learn from", "seeking mentor", "looking for guidance", "need advice",
            "want to learn", "mentorship", "career guidance", "growth"
        ],
        MatchIntent.COFOUNDER: [
            "co-founder", "cofounder", "founding team", "partner",
            "build together", "start together", "technical cofounder"
        ],
        MatchIntent.TALENT_SEEKING: [
            "hiring", "recruit", "looking for talent", "need engineer",
            "seeking developer", "team expansion", "new hire"
        ],
        MatchIntent.OPPORTUNITY_SEEKING: [
            "job", "opportunity", "position", "role", "career",
            "looking for work", "new opportunity", "next role"
        ],
        MatchIntent.PARTNERSHIP: [
            "partner", "collaborate", "b2b", "enterprise", "strategic",
            "alliance", "joint venture", "integration"
        ],
        # BUG-040 FIX: Added recruiter and service provider keywords
        MatchIntent.RECRUITER: [
            "recruiter", "recruiting", "talent acquisition", "headhunter",
            "staffing", "placing candidates", "hiring for clients",
            "talent partner", "executive search", "recruitment firm"
        ],
        MatchIntent.SERVICE_PROVIDER: [
            "consulting", "consultant", "agency", "service provider",
            "client work", "serving companies", "advisory services",
            "professional services", "boutique firm"
        ]
    }

    def classify(self, persona_data: Dict[str, Any]) -> Tuple[MatchIntent, float]:
        """Classify user intent, returns (MatchIntent, confidence_score)."""
        # Combine relevant persona fields
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

        # Return highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
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
        stage_match_weight=2.0,     # Stage (seed/series) is critical
        geography_weight=0.3,
        bidirectional_required=True
    ),
    MatchIntent.FOUNDER_INVESTOR: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.5,
        stage_match_weight=2.0,
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
    MatchIntent.SERVICE_PROVIDER: IntentScoringConfig(
        experience_gap_weight=0.0,
        industry_match_weight=1.3,
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
        Check if two users have the same objective and shouldn't be matched.

        Blocked same-side pairs (NOT complementary):
        - Investor ↔ Investor (both looking to invest)
        - Founder ↔ Founder seeking funding (both raising)
        - Talent ↔ Talent (both job seeking)
        - Mentee ↔ Mentee (both seeking mentors)
        - Mentor ↔ Mentor (both offering mentorship)

        Allowed same-side pairs:
        - Cofounder ↔ Cofounder (both seeking partners) - OK
        - Partnership ↔ Partnership (both seeking partners) - OK
        - General ↔ General (networking) - OK
        - BUG-040 FIX: Recruiter ↔ Recruiter (referral network) - OK
        - BUG-040 FIX: Service Provider ↔ Service Provider (referral network) - OK
        """
        blocked_same_pairs = {
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.INVESTOR_FOUNDER),
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.FOUNDER_INVESTOR),
            (MatchIntent.TALENT_SEEKING, MatchIntent.TALENT_SEEKING),
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.OPPORTUNITY_SEEKING),
            (MatchIntent.MENTOR_MENTEE, MatchIntent.MENTOR_MENTEE),
            (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTEE_MENTOR),
        }

        return (user_intent, candidate_intent) in blocked_same_pairs

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
            if not user_embeddings or not user_embeddings.get('requirements'):
                logger.warning(f"No embeddings found for user {user_id}")
                return []

            # Get user's persona for intent classification
            from app.adapters.supabase_profiles import UserProfile
            try:
                user_profile = UserProfile.get(user_id)
                user_persona = user_profile.persona
                user_intent, user_intent_confidence = self._classify_user_intent(user_persona)
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

                # Skip if below threshold
                if combined_score < threshold:
                    continue

                # Get matched user's data
                try:
                    match_profile = UserProfile.get(match_user_id)
                    match_persona = match_profile.persona
                    match_intent, match_intent_confidence = self._classify_user_intent(match_persona)
                except Exception:
                    match_persona = None
                    match_intent = MatchIntent.GENERAL
                    match_intent_confidence = 0.5

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

                # Calculate final score with all factors (additive weighted)
                final_score, dim_score_stored, signal_score_stored = self._calculate_final_score(
                    combined_score=combined_score,
                    intent_quality=intent_quality,
                    activity_boost=activity_boost,
                    temporal_boost=temporal_boost,
                    config=scoring_config,
                    dimension_score=dimension_score
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

            # Sort by final score and limit
            bidirectional_matches.sort(key=lambda m: m.final_score, reverse=True)
            logger.info(f"Returning {len(bidirectional_matches[:limit])} matches for {user_id}")
            return bidirectional_matches[:limit]

        except Exception as e:
            logger.error(f"Error finding bidirectional matches for {user_id}: {e}")
            return []

    def _classify_user_intent(self, persona) -> Tuple[MatchIntent, float]:
        """Classify user intent from persona."""
        if not persona:
            return MatchIntent.GENERAL, 0.5

        persona_dict = {
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

        UPGRADED (Mar 2026): Complete pair table with 0.7 default.
        Confidence scaling removed — was double-penalizing scores.
        """
        # Complete complementary pairs table
        complete_pairs = {
            # Perfect complementary pairs (1.0)
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.FOUNDER_INVESTOR): 1.0,
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.INVESTOR_FOUNDER): 1.0,
            (MatchIntent.MENTOR_MENTEE, MatchIntent.MENTEE_MENTOR): 1.0,
            (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTOR_MENTEE): 1.0,
            (MatchIntent.TALENT_SEEKING, MatchIntent.OPPORTUNITY_SEEKING): 1.0,
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.TALENT_SEEKING): 1.0,

            # Strong complementary pairs (0.85-0.95)
            (MatchIntent.RECRUITER, MatchIntent.TALENT_SEEKING): 0.95,
            (MatchIntent.TALENT_SEEKING, MatchIntent.RECRUITER): 0.95,
            (MatchIntent.COFOUNDER, MatchIntent.COFOUNDER): 0.9,
            (MatchIntent.RECRUITER, MatchIntent.OPPORTUNITY_SEEKING): 0.9,
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.RECRUITER): 0.9,

            # PRODUCT_LAUNCH / GENERAL pairs (PREVIOUSLY MISSING — all got 0.5)
            (MatchIntent.GENERAL, MatchIntent.INVESTOR_FOUNDER): 0.9,
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.GENERAL): 0.9,
            (MatchIntent.GENERAL, MatchIntent.PARTNERSHIP): 0.85,
            (MatchIntent.PARTNERSHIP, MatchIntent.GENERAL): 0.85,
            (MatchIntent.GENERAL, MatchIntent.SERVICE_PROVIDER): 0.8,
            (MatchIntent.SERVICE_PROVIDER, MatchIntent.GENERAL): 0.8,
            (MatchIntent.GENERAL, MatchIntent.MENTOR_MENTEE): 0.75,
            (MatchIntent.MENTOR_MENTEE, MatchIntent.GENERAL): 0.75,

            # Good matches (0.8-0.85)
            (MatchIntent.PARTNERSHIP, MatchIntent.PARTNERSHIP): 0.85,
            (MatchIntent.SERVICE_PROVIDER, MatchIntent.FOUNDER_INVESTOR): 0.85,
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.SERVICE_PROVIDER): 0.85,

            # COFOUNDER cross-pairs (PREVIOUSLY MISSING — all got 0.5)
            (MatchIntent.COFOUNDER, MatchIntent.INVESTOR_FOUNDER): 0.8,
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.COFOUNDER): 0.8,
            (MatchIntent.COFOUNDER, MatchIntent.SERVICE_PROVIDER): 0.75,
            (MatchIntent.SERVICE_PROVIDER, MatchIntent.COFOUNDER): 0.75,
            (MatchIntent.COFOUNDER, MatchIntent.MENTOR_MENTEE): 0.7,
            (MatchIntent.MENTOR_MENTEE, MatchIntent.COFOUNDER): 0.7,

            # NETWORKING pairs (PREVIOUSLY ALL 0.5)
            (MatchIntent.GENERAL, MatchIntent.GENERAL): 0.8,

            # SERVICE_PROVIDER pairs
            (MatchIntent.SERVICE_PROVIDER, MatchIntent.TALENT_SEEKING): 0.8,
            (MatchIntent.TALENT_SEEKING, MatchIntent.SERVICE_PROVIDER): 0.8,

            # Self-referral pairs
            (MatchIntent.RECRUITER, MatchIntent.RECRUITER): 0.7,
            (MatchIntent.SERVICE_PROVIDER, MatchIntent.SERVICE_PROVIDER): 0.6,
        }

        pair = (user_intent, match_intent)
        # Default 0.7 for unknown pairs (neutral, NOT punishing)
        return complete_pairs.get(pair, 0.7)

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
            "industry_combined": config.industry_match_weight,
            "focus_slot_industry_focus": config.industry_match_weight,
            "stage_combined": config.stage_match_weight,
            "focus_slot_company_stage": config.stage_match_weight,
            "focus_slot_geography": config.geography_weight,
            # Default weight 1.0 for dimensions without specific config
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
        dimension_score: float = 0.5
    ) -> tuple:
        """Combine all scoring factors into final score.

        UPGRADED (Mar 2026): Additive weighted formula replaces multiplicative chain.

        Old formula: combined × intent × activity × temporal (each factor can only reduce)
        New formula: core×0.35 + dimensions×0.40 + intent×0.15 + signals×0.10 (additive)

        Weights:
        - 35% Core: Bidirectional requirements ↔ offerings (the foundation)
        - 40% Dimensions: Industry, geography, stage, engagement, timeline, etc.
        - 15% Intent: User type complementarity (helpful context, not dominant)
        - 10% Signals: Activity + recency (tiebreakers, not determining factors)

        Returns: (final_score, dimension_score, signal_score) tuple for analysis storage
        """
        # Normalize activity and temporal to 0-1 range for additive formula
        # Activity: clamp to 0.7-1.0 (inactive users shouldn't be heavily penalized)
        activity_score = max(0.7, min(1.0, activity_boost))

        # Temporal: clamp to 0.8-1.0 (gentle decay)
        temporal_score = max(0.8, min(1.0, temporal_boost))

        # Combined signal score
        signal_score = (activity_score * 0.6) + (temporal_score * 0.4)

        # Additive weighted formula
        final = (
            combined_score   * 0.35 +    # Layer 1: Bidirectional core compatibility
            dimension_score  * 0.40 +    # Layer 2: Dimensional alignment (NEW)
            intent_quality   * 0.15 +    # Layer 3: Intent complementarity
            signal_score     * 0.10      # Layer 4: Activity + recency
        )

        # Clamp to 0-1 range
        return max(0.0, min(1.0, final)), dimension_score, signal_score

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
