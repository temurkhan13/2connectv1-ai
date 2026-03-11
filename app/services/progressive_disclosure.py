"""
Progressive Disclosure Service.

Implements intelligent question flow that reveals information
progressively to avoid overwhelming users during onboarding.

Key features:
1. Batched question presentation (1-3 at a time)
2. Adaptive pacing based on user engagement
3. Smart skip detection (infer answers from context)
4. Priority-based question ordering
5. Progress tracking and visualization
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.services.slot_extraction import SlotDefinition, SlotType, SlotSchema
from app.services.context_manager import (
    ContextManager, ConversationContext, ConversationPhase
)
from app.services.llm_slot_extractor import SEMANTIC_TOPIC_CLUSTERS

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-VECTOR DIMENSIONS (Critical for Match Quality)
# =============================================================================
# These dimensions are used by multi_vector_matcher.py for high-quality matching.
# Onboarding MUST collect at least 4/6 of these before allowing completion.

MULTI_VECTOR_DIMENSIONS = {
    "primary_goal": {
        "weight": 0.20,
        "required": True,
        "slot_name": "primary_goal",
        "description": "User's main objective on the platform"
    },
    "industry": {
        "weight": 0.25,
        "required": False,
        "slot_name": "industry_focus",
        "description": "Industry/sector focus"
    },
    "stage": {
        "weight": 0.20,
        "required": False,
        "slot_name": "stage_preference",
        "description": "Company stage preference"
    },
    "geography": {
        "weight": 0.15,
        "required": False,
        "slot_name": "geography",
        "description": "Geographic focus"
    },
    "engagement_style": {
        "weight": 0.10,
        "required": False,
        "slot_name": "engagement_style",
        "description": "Preferred engagement style"
    },
    "dealbreakers": {
        "weight": 0.10,
        "required": False,
        "slot_name": "dealbreakers",
        "description": "Deal-breaker criteria"
    }
}

# Minimum dimensions required before onboarding can complete
MIN_MULTI_VECTOR_DIMENSIONS = 4


class QuestionPriority(int, Enum):
    """Priority levels for questions."""
    CRITICAL = 1      # Must ask - core matching requirements
    HIGH = 2          # Important for quality matches
    MEDIUM = 3        # Improves match quality
    LOW = 4           # Optional enhancement
    OPTIONAL = 5      # Nice to have


class UserEngagementLevel(str, Enum):
    """Detected user engagement level."""
    HIGH = "high"           # Detailed responses, asks questions
    MODERATE = "moderate"   # Answers adequately
    LOW = "low"             # Brief responses, wants to finish
    FRUSTRATED = "frustrated"  # Signs of impatience


@dataclass
class QuestionCard:
    """A question to present to the user."""
    slot_name: str
    question_text: str
    question_type: SlotType
    priority: QuestionPriority
    options: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    help_text: Optional[str] = None
    can_skip: bool = False
    estimated_time_seconds: int = 30


@dataclass
class DisclosureBatch:
    """A batch of questions to present."""
    batch_id: str
    questions: List[QuestionCard]
    phase: ConversationPhase
    progress_percent: float
    estimated_remaining_minutes: float
    can_skip_batch: bool = False


@dataclass
class EngagementMetrics:
    """Metrics tracking user engagement."""
    avg_response_length: float = 0.0
    avg_response_time_seconds: float = 0.0
    questions_skipped: int = 0
    questions_answered: int = 0
    clarifications_requested: int = 0
    corrections_made: int = 0

    def get_engagement_level(self) -> UserEngagementLevel:
        """Determine engagement level from metrics."""
        # High engagement: detailed responses, few skips
        if self.avg_response_length > 100 and self.questions_skipped == 0:
            return UserEngagementLevel.HIGH

        # Frustrated: many skips, very short responses
        skip_rate = (self.questions_skipped /
                    max(1, self.questions_answered + self.questions_skipped))
        if skip_rate > 0.3 or self.avg_response_length < 10:
            if self.avg_response_time_seconds < 5:
                return UserEngagementLevel.FRUSTRATED
            return UserEngagementLevel.LOW

        # Moderate is the default
        return UserEngagementLevel.MODERATE


class ProgressiveDisclosure:
    """
    Manages progressive revelation of onboarding questions.

    Adapts to user behavior to maintain engagement while
    collecting necessary information for quality matching.
    """

    # BUG-047 FIX: Role-specific question text overrides
    # Generic questions like industry_focus use investor language by default.
    # This dict provides alternative phrasing for different user types.
    ROLE_QUESTION_OVERRIDES = {
        "job_seeker": {
            "industry_focus": "What industry or sector are you targeting for your next role?",
            "stage_preference": "What stage of company interests you? (startup vs established)",
            "engagement_style": "What kind of work environment suits you best?",
            "requirements": "What are you looking for in your next opportunity?",
            "offerings": "What skills and experience do you bring?",
            "dealbreakers": "What would be a clear 'no' for you in a role?",
        },
        "advisor": {
            "industry_focus": "What industries do you focus your advisory work on?",
            "stage_preference": "What stage companies do you prefer to advise?",
            "engagement_style": "How do you typically engage with your advisory clients?",
            "requirements": "What are you looking for in advisory relationships?",
            "offerings": "What expertise and value do you bring as an advisor?",
        },
        "service_provider": {
            "industry_focus": "What industries do you serve?",
            "stage_preference": "What stage companies are your ideal clients?",
            "engagement_style": "How do you typically work with clients?",
            "requirements": "What kind of clients are you looking for?",
            "offerings": "What services do you provide?",
        },
    }

    # Question templates by slot name
    # NOTE: Uses INDIRECT phrasing - conversational, not form-like
    QUESTION_TEMPLATES = {
        # Core slots - indirect, conversational phrasing
        "primary_goal": QuestionCard(
            slot_name="primary_goal",
            question_text="Tell me more about what success looks like for you here.",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.CRITICAL,
            options=["Find investment opportunities", "Find investors for my startup",
                    "Network with peers", "Find co-founders or partners"],
            help_text="Understanding your goals helps us find the right connections."
        ),
        "user_type": QuestionCard(
            slot_name="user_type",
            question_text="Tell me a bit about your background and what you do.",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.CRITICAL,
            options=["Investor", "Founder", "Advisor", "Service Provider"],
            help_text="This helps us personalize your experience."
        ),
        "industry_focus": QuestionCard(
            slot_name="industry_focus",
            question_text="What space gets you most excited these days?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Fintech", "Healthtech", "SaaS/B2B", "Consumer/D2C",
                    "Deep Tech", "Climate/Sustainability", "Other"],
            help_text="We'll use this to find people in similar spaces.",
            can_skip=False
        ),
        "stage_preference": QuestionCard(
            slot_name="stage_preference",
            question_text="What kind of companies do you love working with?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Pre-seed", "Seed", "Series A", "Series B+", "Growth/Late Stage"],
            help_text="Early-stage, growth, or somewhere in between?"
        ),
        "geography": QuestionCard(
            slot_name="geography",
            question_text="Where in the world are you focused?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,  # PROMOTED: Critical for multi-vector matching
            options=["UK", "US", "Europe", "Asia", "Global/Remote"],
            help_text="Helps us find people in your target markets.",
            can_skip=False  # CHANGED: Required for match quality
        ),
        "requirements": QuestionCard(
            slot_name="requirements",
            question_text="What are you hoping to find through 2Connect?",
            question_type=SlotType.FREE_TEXT,
            priority=QuestionPriority.CRITICAL,
            examples=["Investors for my Series A", "Technical co-founder",
                     "Startups in climate tech to invest in"],
            help_text="Understanding what you're looking for helps us match you better.",
            can_skip=False
        ),
        "offerings": QuestionCard(
            slot_name="offerings",
            question_text="What unique value do you bring to the table?",
            question_type=SlotType.FREE_TEXT,
            priority=QuestionPriority.CRITICAL,
            examples=["Industry expertise in fintech", "Network of enterprise buyers",
                     "Hands-on operational experience"],
            help_text="This helps potential connections understand what you offer.",
            can_skip=False
        ),

        # Investor slots - indirect phrasing
        "check_size": QuestionCard(
            slot_name="check_size",
            question_text="How do you typically think about investment size?",
            question_type=SlotType.RANGE,
            priority=QuestionPriority.HIGH,
            examples=["£25K-100K", "$500K-2M", "Varies by stage"],
            help_text="A range is fine. This helps founders know if there's a fit."
        ),
        "portfolio_size": QuestionCard(
            slot_name="portfolio_size",
            question_text="Tell me about your portfolio - how active have you been?",
            question_type=SlotType.NUMBER,
            priority=QuestionPriority.LOW,
            examples=["5", "20+", "Just starting"],
            can_skip=True
        ),
        "investment_thesis": QuestionCard(
            slot_name="investment_thesis",
            question_text="What kind of opportunities get you most excited?",
            question_type=SlotType.FREE_TEXT,
            priority=QuestionPriority.MEDIUM,
            examples=["B2B SaaS with strong unit economics",
                     "Climate tech with hardware component"],
            help_text="What patterns do you look for?",
            can_skip=True
        ),

        # Founder slots - indirect phrasing
        "company_stage": QuestionCard(
            slot_name="company_stage",
            question_text="Where are you on this journey with your company?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Idea stage", "Pre-seed", "Seed", "Series A", "Series B+"]
        ),
        "funding_need": QuestionCard(
            slot_name="funding_need",
            question_text="How are you thinking about your next funding milestone?",
            question_type=SlotType.RANGE,
            priority=QuestionPriority.HIGH,
            examples=["£250K", "$1-2M", "Not raising currently"],
            can_skip=True
        ),
        "team_size": QuestionCard(
            slot_name="team_size",
            question_text="Tell me about your team - who's building this with you?",
            question_type=SlotType.NUMBER,
            priority=QuestionPriority.LOW,
            examples=["Just me", "3 co-founders", "10 employees"],
            can_skip=True
        ),

        # Multi-vector dimension slots - PROMOTED for match quality
        "engagement_style": QuestionCard(
            slot_name="engagement_style",
            question_text="What kind of relationship would be most valuable for you?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,  # PROMOTED: Critical for multi-vector matching
            options=["Hands-on mentorship", "Strategic advice only",
                    "Introductions and network", "Purely financial"],
            can_skip=False  # CHANGED: Required for match quality
        ),
        "dealbreakers": QuestionCard(
            slot_name="dealbreakers",
            question_text="Anything that would be a clear 'not for me'?",
            question_type=SlotType.FREE_TEXT,
            priority=QuestionPriority.HIGH,  # PROMOTED: Critical for multi-vector matching
            examples=["No crypto projects", "Must have technical co-founder",
                     "No single-founder teams"],
            help_text="Things that would be an immediate no for you.",
            can_skip=False  # CHANGED: Required for match quality
        ),
        "experience_years": QuestionCard(
            slot_name="experience_years",
            question_text="How long have you been in this space?",
            question_type=SlotType.NUMBER,
            priority=QuestionPriority.OPTIONAL,
            can_skip=True
        ),

        # ====================================================================
        # HIRING slots - for companies seeking talent
        # ====================================================================
        "role_type": QuestionCard(
            slot_name="role_type",
            question_text="What kind of role are you looking to fill?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Engineering/Technical", "Product", "Sales/BD", "Marketing/Growth",
                    "Operations", "Executive/C-Suite", "Other"],
            help_text="This helps us match you with the right talent."
        ),
        "seniority_level": QuestionCard(
            slot_name="seniority_level",
            question_text="What level of experience are you targeting?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Junior", "Mid-level", "Senior", "Lead/Staff", "Director", "VP/C-Suite"],
            help_text="Helps us find candidates at the right stage of their career."
        ),
        "remote_preference": QuestionCard(
            slot_name="remote_preference",
            question_text="How do you think about work location for this role?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["Fully Remote", "Hybrid", "On-site Only", "Flexible"],
            can_skip=True
        ),
        "compensation_range": QuestionCard(
            slot_name="compensation_range",
            question_text="What's the compensation range you're thinking about?",
            question_type=SlotType.RANGE,
            priority=QuestionPriority.MEDIUM,
            examples=["$150K-200K", "£80K-120K", "Competitive + equity"],
            help_text="Helps ensure alignment early.",
            can_skip=True
        ),

        # ====================================================================
        # MENTORSHIP slots - for mentors and mentees
        # ====================================================================
        "mentorship_areas": QuestionCard(
            slot_name="mentorship_areas",
            question_text="What areas are you most interested in getting guidance on?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Leadership", "Technical", "Go-to-Market", "Fundraising",
                    "Hiring & Team", "Product", "Sales", "Marketing", "Career Growth"],
            help_text="Helps us match you with mentors who have relevant experience."
        ),
        "mentorship_format": QuestionCard(
            slot_name="mentorship_format",
            question_text="How would you prefer to connect with a mentor?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["Weekly calls", "Bi-weekly calls", "Monthly sessions",
                    "Async messaging", "Ad-hoc as needed"],
            can_skip=True
        ),
        "mentorship_commitment": QuestionCard(
            slot_name="mentorship_commitment",
            question_text="How much time can you dedicate to mentorship?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["1-2 hours/month", "3-5 hours/month", "5-10 hours/month",
                    "10+ hours/month", "Flexible"],
            can_skip=True
        ),

        # ====================================================================
        # COFOUNDER slots - for finding co-founders
        # ====================================================================
        "skills_have": QuestionCard(
            slot_name="skills_have",
            question_text="What skills and strengths do you bring to the table?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Technical/Engineering", "Product", "Sales/BD", "Marketing/Growth",
                    "Finance/Operations", "Design/UX", "Domain Expertise", "Fundraising"],
            help_text="This helps us find complementary co-founders."
        ),
        "skills_need": QuestionCard(
            slot_name="skills_need",
            question_text="What complementary skills are you looking for in a co-founder?",
            question_type=SlotType.MULTI_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Technical/Engineering", "Product", "Sales/BD", "Marketing/Growth",
                    "Finance/Operations", "Design/UX", "Domain Expertise", "Fundraising"],
            help_text="We'll match you with people who have these skills."
        ),
        "commitment_level": QuestionCard(
            slot_name="commitment_level",
            question_text="What kind of commitment are you expecting from a co-founder?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Full-time immediately", "Full-time after funding",
                    "Part-time initially", "Nights & weekends", "Flexible"],
            help_text="Ensures alignment on expectations upfront."
        ),
        "equity_expectations": QuestionCard(
            slot_name="equity_expectations",
            question_text="How do you think about equity for a co-founder?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["Equal split (50/50)", "Majority for existing founder",
                    "Based on contribution", "Open to discuss", "With vesting"],
            can_skip=True
        ),

        # ====================================================================
        # JOB SEEKER slots - for candidates seeking employment
        # BUG-046 FIX: Previously missing - Job Seekers fell back to investor questions
        # ====================================================================
        "target_role": QuestionCard(
            slot_name="target_role",
            question_text="What kind of role are you looking for?",
            question_type=SlotType.FREE_TEXT,
            priority=QuestionPriority.CRITICAL,
            examples=["Senior Product Manager", "Full Stack Engineer", "VP Engineering"],
            help_text="Job title or type of position you're targeting."
        ),
        "desired_seniority": QuestionCard(
            slot_name="desired_seniority",
            question_text="What level are you targeting for your next role?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Junior", "Mid-level", "Senior", "Lead/Staff", "Director", "VP/C-Suite"],
            help_text="Helps match you with the right opportunities."
        ),
        "salary_expectation": QuestionCard(
            slot_name="salary_expectation",
            question_text="What's your target compensation range?",
            question_type=SlotType.RANGE,
            priority=QuestionPriority.MEDIUM,
            examples=["$150K-200K", "£80K-120K + equity", "Flexible based on opportunity"],
            can_skip=True
        ),
        "work_preference": QuestionCard(
            slot_name="work_preference",
            question_text="What's your preferred work setup?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.HIGH,
            options=["Fully Remote", "Hybrid", "On-site", "Flexible"],
            help_text="Helps find companies that match your work style."
        ),
        "availability": QuestionCard(
            slot_name="availability",
            question_text="When could you start a new role?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["Immediately", "2 weeks", "1 month", "2-3 months", "3+ months"],
            can_skip=True
        ),
        "company_size_preference": QuestionCard(
            slot_name="company_size_preference",
            question_text="What size company are you interested in?",
            question_type=SlotType.SINGLE_SELECT,
            priority=QuestionPriority.MEDIUM,
            options=["Startup (1-20)", "Small (21-100)", "Medium (101-500)", "Large (500+)", "Any"],
            help_text="Early-stage startup or established company?"
        ),
    }

    def __init__(self, context_manager: Optional[ContextManager] = None):
        self.context_manager = context_manager or ContextManager()
        self.schema = SlotSchema()

        # Configuration
        # UX-004 FIX: Increased from 2/4 to 5/7 to extract more questions per turn
        # Users were experiencing only 2 questions per onboarding session (too few)
        self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "5"))
        self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "7"))
        self.min_batch_size = 1

        # Engagement tracking per session
        self._engagement_metrics: Dict[str, EngagementMetrics] = {}

        # P3 FIX: Slot to topic mapping for semantic coverage tracking
        self._slot_to_topic = {
            "primary_goal": "goals",
            "requirements": "needs",
            "offerings": "offers",
            "geography": "geography",
            "stage_preference": "stage",
            "industry_focus": "industry",
            "company_stage": "stage",
            "funding_range": "needs",
            "investment_range": "offers",
            "skills": "skills",
            "challenges": "challenges"
        }

    def _map_slot_to_topic(self, slot_name: str) -> Optional[str]:
        """P3 FIX: Map a slot name to its semantic topic."""
        return self._slot_to_topic.get(slot_name)

    def _get_role_aware_question(
        self,
        slot_name: str,
        user_type: Optional[str]
    ) -> Optional[QuestionCard]:
        """
        BUG-047 FIX: Get question template with role-appropriate phrasing.

        Generic questions like industry_focus use investor language by default.
        For other roles (Job Seeker, Advisor, etc.), we override the text.
        """
        template = self.QUESTION_TEMPLATES.get(slot_name)
        if not template:
            return None

        # Normalize user_type for lookup
        if not user_type:
            return template

        user_type_lower = user_type.lower().replace(" ", "_").replace("-", "_")

        # Check for job seeker variants
        if any(term in user_type_lower for term in ["job", "seeker", "candidate", "looking_for"]):
            role_key = "job_seeker"
        elif any(term in user_type_lower for term in ["advisor", "mentor", "consultant"]):
            role_key = "advisor"
        elif any(term in user_type_lower for term in ["service", "provider", "agency"]):
            role_key = "service_provider"
        else:
            # Investor/Founder use default templates
            return template

        # Check if we have an override for this role + slot
        role_overrides = self.ROLE_QUESTION_OVERRIDES.get(role_key, {})
        override_text = role_overrides.get(slot_name)

        if override_text:
            # Create new QuestionCard with overridden text
            return QuestionCard(
                slot_name=template.slot_name,
                question_text=override_text,  # Override!
                question_type=template.question_type,
                priority=template.priority,
                options=template.options,
                examples=template.examples,
                help_text=template.help_text,
                can_skip=template.can_skip,
                estimated_time_seconds=template.estimated_time_seconds
            )

        return template

    def _check_multi_vector_coverage(
        self,
        context: ConversationContext
    ) -> Tuple[int, int, List[str]]:
        """
        Check how many multi-vector dimensions are filled.

        Returns:
            Tuple of (filled_count, total_count, missing_dimension_names)
        """
        filled_count = 0
        missing_dimensions = []

        for dim_name, dim_config in MULTI_VECTOR_DIMENSIONS.items():
            slot_name = dim_config["slot_name"]
            slot = context.slots.get(slot_name)

            if slot and slot.status.value in ["filled", "confirmed"]:
                filled_count += 1
            else:
                missing_dimensions.append(dim_name)

        total_count = len(MULTI_VECTOR_DIMENSIONS)

        logger.info(
            f"Multi-vector coverage: {filled_count}/{total_count} "
            f"(missing: {missing_dimensions})"
        )

        return filled_count, total_count, missing_dimensions

    def can_complete_onboarding(self, session_id: str) -> Tuple[bool, str]:
        """
        Check if onboarding can be completed based on multi-vector coverage.

        Returns:
            Tuple of (can_complete, reason_if_not)
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return False, "Session not found"

        filled, total, missing = self._check_multi_vector_coverage(context)

        if filled < MIN_MULTI_VECTOR_DIMENSIONS:
            # Build helpful message about what's missing
            missing_readable = [
                MULTI_VECTOR_DIMENSIONS[dim]["description"]
                for dim in missing[:3]  # Top 3 missing
            ]
            reason = (
                f"Need at least {MIN_MULTI_VECTOR_DIMENSIONS}/{total} dimensions for quality matching. "
                f"Missing: {', '.join(missing_readable)}"
            )
            return False, reason

        return True, ""

    def get_multi_vector_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed multi-vector coverage status for UI/debugging.

        Returns:
            Dict with coverage details
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return {"error": "Session not found"}

        filled, total, missing = self._check_multi_vector_coverage(context)

        dimension_status = {}
        for dim_name, dim_config in MULTI_VECTOR_DIMENSIONS.items():
            slot_name = dim_config["slot_name"]
            slot = context.slots.get(slot_name)
            dimension_status[dim_name] = {
                "slot_name": slot_name,
                "weight": dim_config["weight"],
                "filled": slot is not None and slot.status.value in ["filled", "confirmed"],
                "value": slot.value if slot else None
            }

        return {
            "filled_count": filled,
            "total_count": total,
            "coverage_percent": round((filled / total) * 100, 1),
            "can_complete": filled >= MIN_MULTI_VECTOR_DIMENSIONS,
            "missing_dimensions": missing,
            "dimensions": dimension_status
        }

    def _detect_covered_topics(self, context: ConversationContext) -> List[str]:
        """
        P3 FIX: Detect which semantic topics have been covered in conversation.

        This prevents asking "Which regions?" after asking "Where are you based?"
        since both map to the "geography" topic cluster.
        """
        covered = set()

        # Check all assistant questions (what we asked)
        for turn in context.turns:
            if turn.turn_type.value == "assistant":
                content = turn.content.lower()

                # Check each topic cluster
                for topic_name, keywords in SEMANTIC_TOPIC_CLUSTERS.items():
                    if any(kw in content for kw in keywords):
                        covered.add(topic_name)

        return list(covered)

    def get_next_batch(
        self,
        session_id: str,
        force_size: Optional[int] = None
    ) -> Optional[DisclosureBatch]:
        """
        Get the next batch of questions to present.

        Adapts batch size based on user engagement and phase.

        Args:
            session_id: Session identifier
            force_size: Optional forced batch size

        Returns:
            DisclosureBatch or None if complete
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Get engagement metrics
        metrics = self._get_or_create_metrics(session_id)
        engagement = metrics.get_engagement_level()

        # Adapt batch size to engagement
        batch_size = self._calculate_batch_size(engagement, context.phase)
        if force_size:
            batch_size = force_size

        # Get pending questions for current phase
        pending = self._get_pending_questions(context)

        if not pending:
            # Check if we should advance phase
            if context.phase != ConversationPhase.COMPLETE:
                return self._create_phase_transition_batch(context)
            return None

        # Select questions for batch
        batch_questions = self._select_batch_questions(pending, batch_size, context)

        # Calculate progress
        progress = self._calculate_progress(context)

        batch = DisclosureBatch(
            batch_id=f"{session_id}_{datetime.utcnow().timestamp()}",
            questions=batch_questions,
            phase=context.phase,
            progress_percent=progress,
            estimated_remaining_minutes=self._estimate_remaining_time(context),
            can_skip_batch=all(q.can_skip for q in batch_questions)
        )

        logger.info(
            f"Created batch {batch.batch_id} with {len(batch_questions)} questions "
            f"(engagement: {engagement.value}, progress: {progress:.0f}%)"
        )

        return batch

    def _calculate_batch_size(
        self,
        engagement: UserEngagementLevel,
        phase: ConversationPhase
    ) -> int:
        """Calculate optimal batch size based on engagement."""
        if engagement == UserEngagementLevel.FRUSTRATED:
            return self.min_batch_size
        elif engagement == UserEngagementLevel.LOW:
            return self.min_batch_size
        elif engagement == UserEngagementLevel.HIGH:
            return min(self.base_batch_size + 1, self.max_batch_size)
        else:
            return self.base_batch_size

    def _get_pending_questions(
        self,
        context: ConversationContext
    ) -> List[QuestionCard]:
        """
        Get questions that haven't been answered yet.

        Uses DYNAMIC slot selection based on primary_goal (objective),
        falling back to user_type if objective not yet known.

        P3 FIX: Also filters out questions whose TOPIC has already been
        covered, even if the exact slot isn't filled.
        """
        pending = []

        # P3 FIX: Detect covered topics to prevent semantic repetition
        covered_topics = self._detect_covered_topics(context)
        logger.info(f"Covered topics: {covered_topics}")

        # Determine which slots to consider based on phase
        if context.phase == ConversationPhase.GREETING:
            return []  # No questions during greeting

        # Get primary_goal and user_type from context
        primary_goal_slot = context.slots.get("primary_goal")
        user_type_slot = context.slots.get("user_type")
        primary_goal = str(primary_goal_slot.value) if primary_goal_slot else None
        user_type = str(user_type_slot.value) if user_type_slot else None

        if context.phase == ConversationPhase.CORE_COLLECTION:
            # During core collection, ask ALL required CORE_SLOTS
            # FIX: Previously hardcoded to only 4 slots, missing requirements, offerings, stage_preference
            slot_names = [s.name for s in self.schema.CORE_SLOTS if s.required]

        elif context.phase == ConversationPhase.ROLE_SPECIFIC:
            # DYNAMIC SELECTION based on objective (primary_goal)
            if primary_goal:
                # Use objective-based slot selection
                objective_slots = self.schema.get_slots_for_objective(primary_goal, user_type)
                # Filter out core slots already asked in CORE_COLLECTION (all required CORE_SLOTS)
                core_names = {s.name for s in self.schema.CORE_SLOTS if s.required}
                slot_names = [s.name for s in objective_slots if s.name not in core_names]
            elif user_type:
                # Fallback to user_type selection (legacy behavior)
                if "investor" in user_type.lower():
                    slot_names = [s.name for s in self.schema.INVESTOR_SLOTS]
                elif "founder" in user_type.lower() or "entrepreneur" in user_type.lower():
                    slot_names = [s.name for s in self.schema.FOUNDER_SLOTS]
                elif "recruiter" in user_type.lower() or "hiring" in user_type.lower():
                    slot_names = [s.name for s in self.schema.HIRING_SLOTS]
                else:
                    slot_names = []
            else:
                slot_names = []

        elif context.phase == ConversationPhase.OPTIONAL_DETAILS:
            # Get optional slots relevant to objective
            if primary_goal:
                objective_slots = self.schema.get_slots_for_objective(primary_goal, user_type)
                # Only include optional slots (engagement_style, dealbreakers, experience_years)
                optional_names = {"engagement_style", "dealbreakers", "experience_years"}
                slot_names = [s.name for s in objective_slots if s.name in optional_names]
            else:
                slot_names = [s.name for s in self.schema.OPTIONAL_SLOTS]
        else:
            slot_names = []

        for slot_name in slot_names:
            # Skip if already filled
            existing = context.slots.get(slot_name)
            if existing and existing.status.value in ["filled", "confirmed", "skipped"]:
                continue

            # P3 FIX: Skip if topic already covered (semantic duplicate prevention)
            slot_topic = self._map_slot_to_topic(slot_name)
            if slot_topic and slot_topic in covered_topics:
                logger.info(f"Skipping slot '{slot_name}' - topic '{slot_topic}' already covered")
                continue

            # Get question template - BUG-047 FIX: use role-aware phrasing
            template = self._get_role_aware_question(slot_name, user_type)
            if template:
                pending.append(template)

        return pending

    def _select_batch_questions(
        self,
        pending: List[QuestionCard],
        batch_size: int,
        context: ConversationContext
    ) -> List[QuestionCard]:
        """Select questions for batch, prioritizing critical ones."""
        # Sort by priority
        sorted_pending = sorted(pending, key=lambda q: q.priority.value)

        # Take top N by priority
        selected = sorted_pending[:batch_size]

        return selected

    def _create_phase_transition_batch(
        self,
        context: ConversationContext
    ) -> Optional[DisclosureBatch]:
        """Create a batch indicating phase transition."""
        # This signals the UI to show a transition message
        return DisclosureBatch(
            batch_id=f"transition_{context.phase.value}",
            questions=[],
            phase=context.phase,
            progress_percent=self._calculate_progress(context),
            estimated_remaining_minutes=self._estimate_remaining_time(context),
            can_skip_batch=True
        )

    def _calculate_progress(self, context: ConversationContext) -> float:
        """
        Calculate overall onboarding progress percentage.

        IMPORTANT: Progress is monotonically increasing - never drops.
        Uses DYNAMIC slot selection based on primary_goal.

        UPDATED: Now factors in multi-vector dimension coverage (70% weight)
        plus traditional slot progress (30% weight) for accurate quality signal.
        """
        # Get primary_goal and user_type
        primary_goal_slot = context.slots.get("primary_goal")
        user_type_slot = context.slots.get("user_type")
        primary_goal = str(primary_goal_slot.value) if primary_goal_slot else None
        user_type = str(user_type_slot.value) if user_type_slot else None

        # Calculate traditional slot progress (30% weight)
        if primary_goal:
            # Use objective-based slot selection
            objective_slots = self.schema.get_slots_for_objective(primary_goal, user_type)
            # Only count required slots
            total_required = len([s for s in objective_slots if s.required])
        else:
            # Before primary_goal is known, just count core slots
            total_required = len([s for s in self.schema.CORE_SLOTS if s.required])

        # Count filled slots
        filled_slots = len([s for s in context.slots.values()
                          if s.status.value in ["filled", "confirmed"]])

        slot_progress = 0.0
        if total_required > 0:
            slot_progress = min(100.0, (filled_slots / total_required) * 100)

        # Calculate multi-vector coverage (70% weight)
        mv_filled, mv_total, _ = self._check_multi_vector_coverage(context)
        mv_progress = (mv_filled / mv_total) * 100 if mv_total > 0 else 0.0

        # Combined progress: 70% multi-vector + 30% traditional slots
        # This ensures users can't reach 100% without multi-vector coverage
        raw_progress = (mv_progress * 0.70) + (slot_progress * 0.30)

        # Cap at 95% if multi-vector coverage is insufficient
        if mv_filled < MIN_MULTI_VECTOR_DIMENSIONS:
            raw_progress = min(raw_progress, 95.0)

        # Get highest progress seen (stored in context metadata)
        highest_progress = context.metadata.get("highest_progress", 0.0)

        # Only increase, never decrease
        if raw_progress > highest_progress:
            context.metadata["highest_progress"] = raw_progress
            return raw_progress
        else:
            return highest_progress

    def _estimate_remaining_time(self, context: ConversationContext) -> float:
        """Estimate remaining time in minutes using objective-based slots."""
        pending_count = 0

        # Get primary_goal and user_type
        primary_goal_slot = context.slots.get("primary_goal")
        user_type_slot = context.slots.get("user_type")
        primary_goal = str(primary_goal_slot.value) if primary_goal_slot else None
        user_type = str(user_type_slot.value) if user_type_slot else None

        # Get relevant slots based on objective
        if primary_goal:
            all_slots = self.schema.get_slots_for_objective(primary_goal, user_type)
        else:
            all_slots = self.schema.CORE_SLOTS

        # Count pending required slots
        for slot in all_slots:
            if not slot.required:
                continue
            existing = context.slots.get(slot.name)
            if not existing or existing.status.value == "empty":
                pending_count += 1

        # Assume ~30 seconds per question
        return (pending_count * 30) / 60

    def _get_or_create_metrics(self, session_id: str) -> EngagementMetrics:
        """Get or create engagement metrics for session."""
        if session_id not in self._engagement_metrics:
            self._engagement_metrics[session_id] = EngagementMetrics()
        return self._engagement_metrics[session_id]

    def record_response(
        self,
        session_id: str,
        slot_name: str,
        response_length: int,
        response_time_seconds: float,
        was_skipped: bool = False
    ) -> None:
        """
        Record a user response for engagement tracking.

        Args:
            session_id: Session identifier
            slot_name: Slot that was answered
            response_length: Length of response in characters
            response_time_seconds: Time taken to respond
            was_skipped: Whether user skipped the question
        """
        metrics = self._get_or_create_metrics(session_id)

        if was_skipped:
            metrics.questions_skipped += 1
        else:
            metrics.questions_answered += 1

            # Update rolling averages
            total = metrics.questions_answered
            metrics.avg_response_length = (
                (metrics.avg_response_length * (total - 1) + response_length) / total
            )
            metrics.avg_response_time_seconds = (
                (metrics.avg_response_time_seconds * (total - 1) + response_time_seconds) / total
            )

        logger.debug(
            f"Recorded response for {slot_name}: "
            f"length={response_length}, time={response_time_seconds:.1f}s, "
            f"skipped={was_skipped}"
        )

    def get_progress_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of progress for UI display.

        Args:
            session_id: Session identifier

        Returns:
            Progress summary dictionary
        """
        context = self.context_manager.get_session(session_id)
        if not context:
            return {}

        metrics = self._get_or_create_metrics(session_id)
        progress = self._calculate_progress(context)

        return {
            "progress_percent": round(progress, 1),
            "phase": context.phase.value,
            "questions_answered": metrics.questions_answered,
            "questions_skipped": metrics.questions_skipped,
            "estimated_remaining_minutes": round(
                self._estimate_remaining_time(context), 1
            ),
            "engagement_level": metrics.get_engagement_level().value,
            "slots_filled": len([s for s in context.slots.values()
                               if s.status.value in ["filled", "confirmed"]])
        }

    def should_show_encouragement(self, session_id: str) -> Tuple[bool, str]:
        """
        Determine if we should show encouragement message.

        Returns:
            Tuple of (should_show, message)
        """
        metrics = self._get_or_create_metrics(session_id)
        context = self.context_manager.get_session(session_id)

        if not context:
            return False, ""

        progress = self._calculate_progress(context)
        engagement = metrics.get_engagement_level()

        # Milestone encouragements
        if 45 <= progress <= 55:
            return True, "You're halfway there! Just a few more questions."

        if 85 <= progress <= 95:
            return True, "Almost done! Just finishing up."

        # Engagement-based
        if engagement == UserEngagementLevel.FRUSTRATED:
            return True, "We appreciate your time. Feel free to skip optional questions."

        if engagement == UserEngagementLevel.HIGH and metrics.questions_answered >= 5:
            return True, "Great responses! This will help us find perfect matches for you."

        return False, ""

    def get_questions_for_goal(self, goal: str) -> List[str]:
        """
        Get relevant question slot names based on user's goal.

        Args:
            goal: User's primary goal (fundraising, hiring, etc.)

        Returns:
            List of slot names relevant to this goal
        """
        goal_questions = {
            'fundraising': ['funding_need', 'company_stage', 'industry_focus', 'geography', 'timeline'],
            'investing': ['check_size', 'portfolio_size', 'investment_thesis', 'stage_preference', 'industry_focus'],
            'hiring': ['role_type', 'team_size', 'industry_focus', 'geography', 'engagement_style'],
            'partnership': ['primary_goal', 'industry_focus', 'company_stage', 'engagement_style', 'geography'],
            'mentorship': ['primary_goal', 'industry_focus', 'experience_years', 'engagement_style'],
            'cofounder': ['primary_goal', 'industry_focus', 'company_stage', 'engagement_style', 'experience_years'],
        }

        # Default to all core slots if goal not recognized
        default_slots = [s.name for s in self.schema.CORE_SLOTS]
        return goal_questions.get(goal.lower(), default_slots)


# Global instance
progressive_disclosure = ProgressiveDisclosure()
