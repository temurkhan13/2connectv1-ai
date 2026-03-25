"""
Conversation Context Manager.

Tracks conversation state, extracted slots, and manages context
across multiple turns of the onboarding conversation.

Key features:
1. Slot state tracking across turns
2. Conversation history management
3. Dependency resolution for conditional slots
4. Context-aware question generation hints
5. Session persistence and recovery
"""
import os
import json
import logging
import ssl
import redis
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import uuid4

from app.services.slot_extraction import (
    SlotExtractor, SlotDefinition, SlotStatus, SlotType,
    SlotSchema, ExtractedSlot
)
from app.services.llm_slot_extractor import LLMSlotExtractor, LLMExtractionResult
from app.services.llm_question_generator import get_question_generator
from app.adapters.supabase_onboarding import supabase_onboarding_adapter

logger = logging.getLogger(__name__)


class ConversationPhase(str, Enum):
    """Phases of the onboarding conversation."""
    GREETING = "greeting"
    CORE_COLLECTION = "core_collection"
    ROLE_SPECIFIC = "role_specific"
    OPTIONAL_DETAILS = "optional_details"
    CONFIRMATION = "confirmation"
    COMPLETE = "complete"


class TurnType(str, Enum):
    """Type of conversation turn."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CLARIFICATION = "clarification"


@dataclass
class ConversationTurn:
    """Single turn in the conversation."""
    turn_id: str
    turn_type: TurnType
    content: str
    timestamp: datetime
    extracted_slots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "turn_type": self.turn_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "extracted_slots": self.extracted_slots,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            turn_id=data["turn_id"],
            turn_type=TurnType(data["turn_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            extracted_slots=data.get("extracted_slots", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationContext:
    """Full context for an onboarding conversation."""
    session_id: str
    user_id: str
    phase: ConversationPhase
    slots: Dict[str, ExtractedSlot]
    turns: List[ConversationTurn]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Serialize slots with datetime handling
        serialized_slots = {}
        for k, v in self.slots.items():
            slot_dict = asdict(v)
            # Convert datetime to ISO string
            if 'extracted_at' in slot_dict and slot_dict['extracted_at']:
                if hasattr(slot_dict['extracted_at'], 'isoformat'):
                    slot_dict['extracted_at'] = slot_dict['extracted_at'].isoformat()
            serialized_slots[k] = slot_dict

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "phase": self.phase.value,
            "slots": serialized_slots,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class ContextManager:
    """
    Manages conversation context for onboarding flow.

    Responsibilities:
    - Track slot extraction state across turns
    - Manage conversation history
    - Resolve slot dependencies
    - Determine next questions to ask
    - Support session persistence
    """

    def __init__(self, slot_extractor: Optional[SlotExtractor] = None):
        self.slot_extractor = slot_extractor or SlotExtractor()
        self.llm_extractor = LLMSlotExtractor()  # LLM-based extraction (ONLY extraction)
        self.question_generator = get_question_generator()  # BUG-071: Dedicated question generator
        self.schema = SlotSchema()

        # Session storage (in production, use Redis or DB)
        self._sessions: Dict[str, ConversationContext] = {}

        # Store LLM responses for contextual follow-ups
        self._llm_responses: Dict[str, LLMExtractionResult] = {}

        # Extraction cache to prevent redundant API calls (P2 fix)
        # Key: hash(content), Value: extracted_names list
        self._extraction_cache: Dict[str, List[str]] = {}

        # Configuration
        self.max_history_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "50"))
        self.session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
        self.max_questions = int(os.getenv("MAX_ONBOARDING_QUESTIONS", "5"))  # Prevent over-questioning

        # Redis persistence
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            # Support rediss:// URLs (Upstash, etc.)
            # For Upstash and other managed Redis services using TLS
            if self.redis_url.startswith("rediss://"):
                # Use ssl_cert_reqs as string "none" for newer redis-py versions
                self.redis = redis.from_url(
                    self.redis_url,
                    ssl_cert_reqs="none"  # Disable certificate verification for managed Redis
                )
            else:
                self.redis = redis.from_url(self.redis_url)
            self.redis.ping()  # Test connection
            self._use_redis = True
            logger.info("Redis connected for session persistence")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self._use_redis = False
            self.redis = None

    def _save_to_redis(self, session_id: str, context: ConversationContext) -> None:
        """Persist session to Redis."""
        if self._use_redis and self.redis:
            try:
                key = f"onboarding:session:{session_id}"
                data = json.dumps(context.to_dict())
                self.redis.setex(key, self.session_timeout_hours * 3600, data)
            except Exception as e:
                logger.error(f"Failed to save session to Redis: {e}")

    def _load_from_redis(self, session_id: str) -> Optional[ConversationContext]:
        """Load session from Redis."""
        if self._use_redis and self.redis:
            try:
                key = f"onboarding:session:{session_id}"
                data = self.redis.get(key)
                if data:
                    parsed = json.loads(data)
                    # Reconstruct ConversationContext from dict
                    return self._dict_to_context(parsed)
            except Exception as e:
                logger.error(f"Failed to load session from Redis: {e}")
        return None

    def _get_resume_text_from_redis(self, session_id: str) -> Optional[str]:
        """
        ISSUE-1 FIX: Fetch extracted resume text from Redis.

        The resume text is stored by the upload_resume endpoint immediately
        after upload, allowing it to be used during slot extraction.

        Args:
            session_id: The onboarding session ID

        Returns:
            Extracted resume text if available, None otherwise
        """
        if not self._use_redis or not self.redis:
            return None

        try:
            key = f"resume_text:{session_id}"
            resume_text = self.redis.get(key)
            if resume_text:
                # Redis returns bytes or string depending on decode_responses setting
                if isinstance(resume_text, bytes):
                    resume_text = resume_text.decode('utf-8')
                return resume_text
        except Exception as e:
            logger.debug(f"Could not fetch resume text from Redis: {e}")

        return None

    def _dict_to_context(self, data: dict) -> ConversationContext:
        """Convert dict back to ConversationContext."""
        from app.services.slot_extraction import SlotStatus

        slots = {}
        for name, slot_data in data.get("slots", {}).items():
            slots[name] = ExtractedSlot(
                name=name,
                value=slot_data.get("value"),
                confidence=slot_data.get("confidence", 0.0),
                status=SlotStatus(slot_data.get("status", "empty")),
                source_text=slot_data.get("source_text", ""),
                extracted_at=datetime.fromisoformat(slot_data["extracted_at"]) if slot_data.get("extracted_at") else datetime.utcnow()
            )

        turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]

        return ConversationContext(
            session_id=data["session_id"],
            user_id=data["user_id"],
            phase=ConversationPhase(data.get("phase", "greeting")),
            slots=slots,
            turns=turns,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )

    def create_session(self, user_id: str) -> ConversationContext:
        """
        Create a new conversation session.

        Args:
            user_id: User identifier

        Returns:
            New ConversationContext
        """
        session_id = str(uuid4())
        now = datetime.utcnow()

        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            phase=ConversationPhase.GREETING,
            slots={},
            turns=[],
            created_at=now,
            updated_at=now,
            metadata={"turn_count": 0}
        )

        self._sessions[session_id] = context
        self._save_to_redis(session_id, context)
        logger.info(f"Created session {session_id} for user {user_id}")

        return context

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        Retrieve an existing session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext or None if not found/expired
        """
        # Check in-memory first
        context = self._sessions.get(session_id)

        # If not in memory, try Redis
        if not context:
            context = self._load_from_redis(session_id)
            if context:
                self._sessions[session_id] = context  # Cache locally

        if context:
            # Check for expiration
            age = datetime.utcnow() - context.updated_at
            if age > timedelta(hours=self.session_timeout_hours):
                logger.info(f"Session {session_id} expired after {age}")
                del self._sessions[session_id]
                return None

        return context

    def save_session(self, session_id: str) -> bool:
        """
        BUG-100 FIX: Save session to Redis after external modifications.

        This method should be called after slots are restored from Supabase
        to ensure the updated context is persisted to Redis. Without this,
        subsequent requests would retrieve stale session data with empty slots.

        Args:
            session_id: Session identifier

        Returns:
            True if saved successfully, False otherwise
        """
        context = self._sessions.get(session_id)
        if context:
            self._save_to_redis(session_id, context)
            logger.info(f"BUG-100 FIX: Saved session {session_id[:8]}... to Redis after slot restoration")
            return True
        logger.warning(f"BUG-100: Cannot save session {session_id[:8]}... - not found in memory")
        return False

    async def add_turn(
        self,
        session_id: str,
        turn_type: TurnType,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Optional[ConversationTurn]:
        """
        Add a turn to the conversation and extract any slots.

        BUG-008 FIX: Made async to support async slot extraction and persistence.

        Args:
            session_id: Session identifier
            turn_type: Type of turn (user, assistant, etc.)
            content: Turn content
            metadata: Optional metadata

        Returns:
            ConversationTurn or None if session not found
        """
        context = self.get_session(session_id)
        if not context:
            logger.warning(f"Session not found: {session_id}")
            return None

        turn_id = str(uuid4())
        now = datetime.utcnow()

        # Extract slots from user turns
        # BUG-008 FIX: Await async slot extraction
        extracted_slot_names = []
        if turn_type == TurnType.USER:
            extracted_slot_names = await self._extract_slots_from_turn(context, content)

        turn = ConversationTurn(
            turn_id=turn_id,
            turn_type=turn_type,
            content=content,
            timestamp=now,
            extracted_slots=extracted_slot_names,
            metadata=metadata or {}
        )

        # Add to history (with size limit)
        context.turns.append(turn)
        if len(context.turns) > self.max_history_turns:
            context.turns = context.turns[-self.max_history_turns:]

        context.updated_at = now
        context.metadata["turn_count"] = len(context.turns)
        self._save_to_redis(session_id, context)

        # Update phase based on progress
        self._update_phase(context)

        logger.debug(f"Added turn {turn_id} to session {session_id}")
        return turn

    async def _extract_slots_from_turn(
        self,
        context: ConversationContext,
        content: str
    ) -> List[str]:
        """
        Extract slot values from user turn content using LLM comprehension.

        This uses an LLM to understand the user's message and extract
        structured slot data, rather than relying on regex patterns.

        Args:
            context: Conversation context
            content: User message content

        Returns:
            List of slot names that were extracted
        """
        # P2 FIX: Check extraction cache first to avoid redundant API calls
        # CRITICAL: Cache only prevents redundant slot STORAGE, not question generation
        # We must ALWAYS generate fresh follow-up questions for conversation context
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        cache_key = f"{context.session_id}:{content_hash}"

        use_cached_slots = cache_key in self._extraction_cache
        if use_cached_slots:
            logger.info(f"Cache hit for slot extraction (will still generate fresh question)")

        extracted_names = []

        # Build already-filled slots dict
        already_filled = {
            name: slot.value for name, slot in context.slots.items()
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
        }

        # Build conversation history for LLM context
        conversation_history = []
        for turn in context.turns[-6:]:  # Last 6 turns
            conversation_history.append({
                "role": "assistant" if turn.turn_type == TurnType.ASSISTANT else "user",
                "content": turn.content
            })

        try:
            logger.info(f"LLM extraction starting. Already filled: {list(already_filled.keys())}")

            # ISSUE-1 FIX: Fetch resume text from Redis if available
            # This allows the LLM to skip questions about info already in the resume
            resume_context = None
            try:
                resume_context = self._get_resume_text_from_redis(context.session_id)
                if resume_context:
                    logger.info(f"ISSUE-1 FIX: Found resume context for session ({len(resume_context)} chars)")
            except Exception as e:
                logger.debug(f"Could not fetch resume context: {e}")

            # Use LLM to extract slots with full comprehension
            # CRITICAL: Always call LLM even on cache hit to generate fresh personalized question
            llm_result = self.llm_extractor.extract_slots(
                user_message=content,
                conversation_history=conversation_history,
                already_filled_slots=already_filled,
                session_id=context.session_id,
                resume_context=resume_context
            )

            logger.info(f"LLM returned slots: {list(llm_result.extracted_slots.keys())}")

            # BUG-087 FIX: Calculate missing slots PROGRAMMATICALLY instead of trusting LLM
            # The LLM only sees current turn, not session requirements - can't know what's truly missing.
            from app.services.use_case_templates import get_onboarding_slots, get_template

            # Determine objective from primary_goal (user's stated objective), NOT user_type
            # A founder could be looking for co-founder, fundraising, partnership, etc.
            # primary_goal is the actual stated intent; user_type is just their role.
            primary_goal_value = already_filled.get("primary_goal", "")
            # Also check if primary_goal was just extracted in this turn
            if not primary_goal_value and "primary_goal" in llm_result.extracted_slots:
                pg_slot = llm_result.extracted_slots["primary_goal"]
                primary_goal_value = pg_slot.value if hasattr(pg_slot, 'value') else str(pg_slot)

            # Always set user_type from LLM inference (needed for question generator)
            user_type = llm_result.user_type_inference or "unknown"

            if primary_goal_value:
                # get_template handles keyword matching (e.g. "Looking to Invest" → investing)
                template = get_template(str(primary_goal_value))
                objective = template.objective.value  # e.g. "investing", "fundraising", "cofounder"
            else:
                # primary_goal not yet extracted — fall back to user_type inference
                objective = self._map_user_type_to_objective(user_type)

            # Get required slots for this objective
            required_slots = get_onboarding_slots(objective)
            logger.info(f"[BUG-087] Objective '{objective}' requires slots: {required_slots}")

            # Calculate what's ACTUALLY missing (required - already_filled - just_extracted)
            all_filled_now = set(already_filled.keys()) | set(llm_result.extracted_slots.keys())
            actual_missing_slots = [s for s in required_slots if s not in all_filled_now]
            logger.info(f"[BUG-087] Actually missing: {actual_missing_slots} (filled: {list(all_filled_now)})")

            # BUG-087 FIX: ALWAYS generate follow-up question using DEDICATED question generator
            # Previously gated on llm_result.missing_slots being non-empty, but that was unreliable.
            follow_up_question = None
            if not llm_result.is_off_topic:
                try:
                    # ALWAYS call question generator - pass calculated missing slots, not LLM's opinion
                    follow_up_question = self.question_generator.generate_followup_question(
                        user_message=content,
                        extracted_slots={k: {"value": v.value, "confidence": v.confidence} for k, v in llm_result.extracted_slots.items()},
                        all_filled_slots=already_filled,
                        missing_slots=actual_missing_slots if actual_missing_slots else ["engagement"],  # Fallback for engagement questions
                        user_type=user_type,
                        session_id=context.session_id,
                        conversation_history=conversation_history
                    )
                    logger.info(f"[QuestionGenerator] Generated follow-up: {follow_up_question[:80] if follow_up_question else 'None'}...")
                except Exception as qe:
                    logger.warning(f"[QuestionGenerator] Failed to generate question: {qe}")
                    follow_up_question = None

            # Create updated result with generated question AND calculated missing slots
            # BUG-092 FIX: If user wants to finish, skip question generation
            if llm_result.is_completion_signal:
                logger.info(f"[BUG-092] User signaled completion - skipping follow-up question")
                follow_up_question = ""

            llm_result_with_question = LLMExtractionResult(
                extracted_slots=llm_result.extracted_slots,
                user_type_inference=llm_result.user_type_inference,
                missing_slots=actual_missing_slots,  # BUG-087: Use calculated, not LLM's opinion
                understanding_summary=llm_result.understanding_summary,
                is_off_topic=llm_result.is_off_topic,
                is_completion_signal=llm_result.is_completion_signal,  # BUG-092: Propagate signal
                follow_up_question=follow_up_question or ""
            )

            # Store LLM result for response generation (CRITICAL for question generation)
            self._llm_responses[context.session_id] = llm_result_with_question

            # P0 FIX: ALWAYS persist slots to Supabase (even on cache hit)
            # This ensures dashboard shows correct slot count and enables regeneration
            # BUG-008 FIX: Await async persistence function
            if llm_result.extracted_slots:
                await self._persist_slots_to_supabase(context, llm_result, content)

            # If cache hit, use cached slot names to avoid redundant storage IN MEMORY
            # But we still persisted to Supabase above for dashboard visibility
            if use_cached_slots:
                cached_slots = self._extraction_cache[cache_key]
                logger.info(f"Using {len(cached_slots)} cached slots (question generated fresh, persisted to DB)")
                return cached_slots

            # Convert LLM extractions to our slot format
            for slot_name, llm_slot in llm_result.extracted_slots.items():
                # Skip already filled slots (unless it's a correction with high confidence)
                if slot_name in context.slots:
                    existing = context.slots[slot_name]
                    if existing.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                        # Only override if LLM is very confident (correction detection)
                        if llm_slot.confidence < 0.95:
                            continue

                # Create ExtractedSlot from LLM result
                extracted = ExtractedSlot(
                    name=slot_name,
                    value=llm_slot.value,
                    confidence=llm_slot.confidence,
                    status=SlotStatus.FILLED,
                    source_text=content,
                    extracted_at=datetime.utcnow()
                )

                context.slots[slot_name] = extracted
                extracted_names.append(slot_name)
                logger.info(f"LLM extracted slot {slot_name}: {llm_slot.value} (confidence: {llm_slot.confidence:.2f}, reason: {llm_slot.reasoning})")

            logger.info(f"LLM extraction complete: {len(extracted_names)} slots extracted, user_type_inference: {llm_result.user_type_inference}")

        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to regex: {e}")
            # Fallback to regex-based extraction
            fallback_result = self._extract_slots_regex_fallback(context, content)
            # Cache fallback result too
            self._extraction_cache[cache_key] = fallback_result
            return fallback_result

        # P2 FIX: Store in cache for future requests with same content
        self._extraction_cache[cache_key] = extracted_names
        logger.info(f"Cached extraction result: {len(extracted_names)} slots for content hash {content_hash[:8]}...")

        # NOTE: Persistence already handled above (after LLM call) for both cache hit and miss

        return extracted_names

    async def _persist_slots_to_supabase(
        self,
        context: ConversationContext,
        llm_result: LLMExtractionResult,
        content: str
    ) -> None:
        """
        P0 FIX: Persist extracted slots to Supabase.

        This ensures slots are visible in the admin dashboard and available
        for regeneration if needed.

        BUG-008 FIX: Made async to avoid event loop conflict when called
        from async FastAPI endpoints.

        BUG-022 FIX: Fail fast on persistence errors instead of silent failures.
        If slots can't be persisted, the error propagates to the API endpoint,
        user sees error message, and can retry (vs thinking onboarding succeeded
        when data was actually lost).

        Raises:
            SupabasePersistenceError: If slot persistence fails after retries
        """
        from app.adapters.supabase_onboarding import SupabasePersistenceError

        slots_to_save = [
            {
                "name": slot_name,
                "value": str(llm_slot.value),
                "confidence": llm_slot.confidence,
                "source_text": content[:500],  # Truncate to 500 chars
                "extraction_method": "llm",
                "status": "filled"
            }
            for slot_name, llm_slot in llm_result.extracted_slots.items()
        ]

        if slots_to_save:
            # BUG-022 FIX: Let SupabasePersistenceError propagate (fail fast)
            # save_slots_batch now retries 3x and validates persistence
            saved_count = await supabase_onboarding_adapter.save_slots_batch(
                user_id=context.user_id,
                slots=slots_to_save
            )
            logger.info(f"✅ Persisted {saved_count}/{len(slots_to_save)} slots to Supabase for user {context.user_id[:8]}...")
            # Note: If this succeeds, persistence was validated (slots in database)

    def _extract_slots_regex_fallback(
        self,
        context: ConversationContext,
        content: str
    ) -> List[str]:
        """Fallback to regex extraction if LLM fails."""
        extracted_names = []

        target_slots = self._get_phase_slots(context.phase, context)
        target_slot_names = [s.name for s in target_slots]

        existing_context = {
            name: slot.value for name, slot in context.slots.items()
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
        }

        extractions = self.slot_extractor.extract_from_text(
            content,
            target_slots=target_slot_names,
            context=existing_context
        )

        for slot_name, extracted in extractions.items():
            if slot_name in context.slots:
                existing = context.slots[slot_name]
                if existing.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                    continue

            if extracted and extracted.status != SlotStatus.EMPTY:
                context.slots[slot_name] = extracted
                extracted_names.append(slot_name)
                logger.info(f"Regex fallback extracted slot {slot_name}: {extracted.value}")

        return extracted_names

    def get_llm_response(self, session_id: str) -> Optional[LLMExtractionResult]:
        """Get the latest LLM extraction result for response generation."""
        return self._llm_responses.get(session_id)

    def _map_user_type_to_objective(self, user_type: str) -> str:
        """
        BUG-087 FIX: Map user_type to objective for getting required slots.

        Uses same mapping as onboarding.py _get_core_slot_names() (BUG-085 FIX).
        This ensures consistent objective detection across the system.

        Args:
            user_type: User type from LLM inference (founder, investor, job_seeker, etc.)

        Returns:
            Objective string matching ObjectiveType enum values
        """
        user_type_lower = (user_type or "").lower().strip()

        # BUG-085 FIX: Correct mappings (same as onboarding.py)
        type_to_objective = {
            "founder": "fundraising",
            "entrepreneur": "fundraising",
            "investor": "investing",
            "angel_investor": "investing",
            "vc_partner": "investing",
            "job_seeker": "job_search",
            "candidate": "job_search",
            "advisor": "mentorship",
            "mentor": "mentorship",
            "recruiter": "hiring",
            "service_provider": "services",
            "consultant": "services",
            "executive": "job_search",  # Executives looking for opportunities
        }

        # Try exact match first
        if user_type_lower in type_to_objective:
            return type_to_objective[user_type_lower]

        # Try partial match (e.g., "job_seeker/candidate" contains "job_seeker")
        for key, objective in type_to_objective.items():
            if key in user_type_lower:
                return objective

        # Default to networking for unknown types
        logger.warning(f"[BUG-087] Unknown user_type '{user_type}', defaulting to 'networking'")
        return "networking"

    def _get_phase_slots(
        self,
        phase: ConversationPhase,
        context: ConversationContext
    ) -> List[SlotDefinition]:
        """Get slot definitions relevant to current phase."""
        if phase in [ConversationPhase.GREETING, ConversationPhase.CORE_COLLECTION]:
            # Extract core slots during greeting and core collection
            return self.schema.CORE_SLOTS
        elif phase == ConversationPhase.ROLE_SPECIFIC:
            user_type = context.slots.get("user_type")
            if user_type and user_type.value:
                if "investor" in str(user_type.value).lower():
                    return self.schema.INVESTOR_SLOTS
                elif "founder" in str(user_type.value).lower():
                    return self.schema.FOUNDER_SLOTS
            return []
        elif phase == ConversationPhase.OPTIONAL_DETAILS:
            return self.schema.OPTIONAL_SLOTS
        else:
            return []

    def _check_dependencies(
        self,
        slot_def: SlotDefinition,
        context: ConversationContext
    ) -> bool:
        """
        Check if slot dependencies are satisfied.

        Args:
            slot_def: Slot definition with potential dependencies
            context: Current conversation context

        Returns:
            True if all dependencies are met
        """
        if not slot_def.depends_on:
            return True

        for dep in slot_def.depends_on:
            dep_slot = context.slots.get(dep)
            if not dep_slot or dep_slot.status == SlotStatus.EMPTY:
                return False

        return True

    def _update_phase(self, context: ConversationContext) -> None:
        """Update conversation phase based on slot completion."""
        filled_core = self._count_filled_slots(context, self.schema.CORE_SLOTS)
        total_core = len(self.schema.CORE_SLOTS)

        if context.phase == ConversationPhase.GREETING:
            # Move to core collection after greeting
            if len(context.turns) >= 2:
                context.phase = ConversationPhase.CORE_COLLECTION

        elif context.phase == ConversationPhase.CORE_COLLECTION:
            # Move to role-specific when core is mostly filled
            if filled_core >= total_core - 1:
                context.phase = ConversationPhase.ROLE_SPECIFIC

        elif context.phase == ConversationPhase.ROLE_SPECIFIC:
            # Move to optional when role-specific is done
            role_slots = self._get_phase_slots(ConversationPhase.ROLE_SPECIFIC, context)
            filled_role = self._count_filled_slots(context, role_slots)
            if filled_role >= len(role_slots) * 0.8:
                context.phase = ConversationPhase.OPTIONAL_DETAILS

        elif context.phase == ConversationPhase.OPTIONAL_DETAILS:
            # Move to confirmation when enough info gathered
            total_filled = len([s for s in context.slots.values()
                               if s.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]])
            if total_filled >= 8:  # Minimum slots for good matching
                context.phase = ConversationPhase.CONFIRMATION

    def _count_filled_slots(
        self,
        context: ConversationContext,
        slot_defs: List[SlotDefinition]
    ) -> int:
        """Count filled slots from a slot definition list."""
        count = 0
        for slot_def in slot_defs:
            slot_name = slot_def.name if hasattr(slot_def, 'name') else str(slot_def)
            slot = context.slots.get(slot_name)
            if slot and slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                count += 1
        return count

    def get_next_questions(
        self,
        session_id: str,
        max_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get the next questions to ask based on missing slots.

        Args:
            session_id: Session identifier
            max_questions: Maximum questions to return

        Returns:
            List of question hints with slot info
        """
        context = self.get_session(session_id)
        if not context:
            return []

        questions = []
        target_slots = self._get_phase_slots(context.phase, context)

        for slot_name, slot_def in target_slots.items():
            if len(questions) >= max_questions:
                break

            # Skip filled slots
            existing = context.slots.get(slot_name)
            if existing and existing.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                continue

            # Check dependencies
            if not self._check_dependencies(slot_def, context):
                continue

            # Generate question hint
            questions.append({
                "slot_name": slot_name,
                "slot_type": slot_def.slot_type.value,
                "prompt_hint": slot_def.prompt_hint,
                "required": slot_def.required,
                "options": slot_def.valid_options,
                "examples": slot_def.examples
            })

        return questions

    def get_slot_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of all extracted slots.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with slot summary
        """
        context = self.get_session(session_id)
        if not context:
            return {}

        summary = {
            "phase": context.phase.value,
            "total_slots": len(context.slots),
            "filled_count": 0,
            "confirmed_count": 0,
            "slots": {}
        }

        for name, slot in context.slots.items():
            summary["slots"][name] = {
                "value": slot.value,
                "status": slot.status.value,
                "confidence": slot.confidence
            }
            if slot.status == SlotStatus.FILLED:
                summary["filled_count"] += 1
            elif slot.status == SlotStatus.CONFIRMED:
                summary["confirmed_count"] += 1

        return summary

    def confirm_slot(
        self,
        session_id: str,
        slot_name: str
    ) -> bool:
        """
        Mark a slot as confirmed by user.

        Args:
            session_id: Session identifier
            slot_name: Name of slot to confirm

        Returns:
            True if confirmed successfully
        """
        context = self.get_session(session_id)
        if not context:
            return False

        slot = context.slots.get(slot_name)
        if slot:
            slot.status = SlotStatus.CONFIRMED
            context.updated_at = datetime.utcnow()
            logger.info(f"Confirmed slot {slot_name} in session {session_id}")
            return True

        return False

    def update_slot(
        self,
        session_id: str,
        slot_name: str,
        new_value: Any,
        confidence: float = 1.0
    ) -> bool:
        """
        Update a slot value (for corrections).

        Args:
            session_id: Session identifier
            slot_name: Name of slot to update
            new_value: New value for the slot
            confidence: Confidence score

        Returns:
            True if updated successfully
        """
        context = self.get_session(session_id)
        if not context:
            return False

        if slot_name in context.slots:
            context.slots[slot_name].value = new_value
            context.slots[slot_name].confidence = confidence
            context.slots[slot_name].status = SlotStatus.FILLED
            context.updated_at = datetime.utcnow()
            logger.info(f"Updated slot {slot_name} to {new_value}")
            return True

        return False

    def skip_slot(self, session_id: str, slot_name: str) -> bool:
        """
        Mark a slot as skipped.

        Args:
            session_id: Session identifier
            slot_name: Name of slot to skip

        Returns:
            True if marked successfully
        """
        context = self.get_session(session_id)
        if not context:
            return False

        # Create or update slot as skipped
        context.slots[slot_name] = ExtractedSlot(
            slot_name=slot_name,
            value=None,
            confidence=1.0,
            status=SlotStatus.SKIPPED,
            source_text=""
        )
        context.updated_at = datetime.utcnow()
        logger.info(f"Skipped slot {slot_name} in session {session_id}")
        return True

    def get_conversation_history(
        self,
        session_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for context.

        Args:
            session_id: Session identifier
            last_n: Optional limit on number of turns

        Returns:
            List of turn dictionaries
        """
        context = self.get_session(session_id)
        if not context:
            return []

        turns = context.turns
        if last_n:
            turns = turns[-last_n:]

        return [t.to_dict() for t in turns]

    def is_complete(self, session_id: str) -> bool:
        """
        Check if onboarding is complete.

        Returns True if:
        1. Phase is explicitly set to COMPLETE, OR
        2. All required slots are filled (progress >= 80%), OR
        3. User explicitly signals completion ("done", "that's all", etc.), OR
        4. Max question limit reached AND minimum viable profile exists (3+ slots)
        """
        context = self.get_session(session_id)
        if not context:
            return False

        # Explicit completion phase
        if context.phase == ConversationPhase.COMPLETE:
            return True

        # Check for explicit user completion signals in recent messages
        if self._user_signals_completion(context):
            logger.info(f"Session {session_id}: User explicitly signaled completion")
            context.phase = ConversationPhase.COMPLETE
            return True

        # Check if max questions reached (prevent over-questioning like Alex's 7 questions)
        if self._max_questions_reached(context):
            logger.info(f"Session {session_id}: Max questions ({self.max_questions}) reached, auto-completing")
            context.phase = ConversationPhase.COMPLETE
            return True

        # Check if enough required slots are filled
        return self._all_required_slots_filled(context)

    def _user_signals_completion(self, context: 'ConversationContext') -> bool:
        """
        Detect if user explicitly signals they want to finish onboarding.

        FIX A+C: Trust LLM only - removed all code-level phrase matching.
        The LLM (via is_completion_signal) understands context and won't false-positive
        on phrases like "I've done deals" or "done Africa work".

        Previously: Regex phrase matching caused false positives.
        Now: Delegates entirely to LLM's is_completion_signal detection.
        """
        # FIX A+C: Removed all phrase-based detection
        # LLM comprehension handles completion detection via is_completion_signal
        # This method now only checks if LLM flagged completion in the last extraction

        llm_result = self._llm_responses.get(context.session_id)
        if llm_result and llm_result.is_completion_signal:
            logger.info(f"FIX A+C: LLM detected completion signal for session {context.session_id}")
            return True

        return False

    def _max_questions_reached(self, context: 'ConversationContext') -> bool:
        """
        Check if we've asked too many questions.

        Prevents over-questioning scenarios like Alex (7 questions).
        Only auto-complete if we have minimum viable profile (3+ required slots).
        """
        # Count AI questions (assistant turns with question marks)
        ai_questions = [
            t for t in context.turns
            if t.turn_type == TurnType.ASSISTANT and "?" in t.content
        ]

        questions_asked = len(ai_questions)

        if questions_asked < self.max_questions:
            return False

        # Max questions reached - check if we have minimum viable profile
        filled_required_slots = [
            name for name, slot in context.slots.items()
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
            and name in ["user_type", "primary_goal", "requirements", "offerings", "industry_focus"]
        ]

        # Allow completion if we have at least 3 critical slots filled
        has_minimum_profile = len(filled_required_slots) >= 3

        if has_minimum_profile:
            logger.info(f"Max questions reached ({questions_asked}/{self.max_questions}), "
                       f"minimum profile exists ({len(filled_required_slots)} critical slots)")
            return True
        else:
            logger.warning(f"Max questions reached ({questions_asked}/{self.max_questions}), "
                          f"but only {len(filled_required_slots)} critical slots filled - continuing")
            return False

    def _all_required_slots_filled(self, context: 'ConversationContext') -> bool:
        """Check if all required slots are filled based on user type."""
        # Get required core slots
        required_slots = [s.name for s in self.schema.CORE_SLOTS if s.required]

        # Add role-specific required slots
        user_type_slot = context.slots.get("user_type")
        if user_type_slot and user_type_slot.value:
            user_type_value = str(user_type_slot.value).lower()
            if "investor" in user_type_value or "vc" in user_type_value:
                required_slots.extend([s.name for s in self.schema.INVESTOR_SLOTS if s.required])
            elif "founder" in user_type_value or "entrepreneur" in user_type_value:
                required_slots.extend([s.name for s in self.schema.FOUNDER_SLOTS if s.required])

        # Check if all required are filled
        filled_count = 0
        for slot_name in required_slots:
            slot = context.slots.get(slot_name)
            if slot and slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                filled_count += 1

        # FIX B: Strict completion requirements with minimum 5 questions
        # Changed from 3 turns to 5 turns to prevent premature completion
        # Ensures user has been asked enough questions before completing
        if not required_slots:
            return False

        completion_ratio = filled_count / len(required_slots)

        # Requirement 1: ALL required slots must be filled (100%)
        slots_complete = completion_ratio >= 1.0

        # FIX B: Minimum 5 conversation turns (user messages) - increased from 3
        user_turns = sum(1 for turn in context.turns if turn.turn_type == TurnType.USER)
        min_turns_met = user_turns >= 5  # FIX B: Changed from 3 to 5

        is_complete = slots_complete and min_turns_met

        if slots_complete and not min_turns_met:
            logger.info(f"Session {context.session_id}: {filled_count}/{len(required_slots)} slots filled (100%) but only {user_turns}/5 turns - FIX B: need 5 minimum")
        elif is_complete:
            logger.info(f"Session auto-complete: {filled_count}/{len(required_slots)} required slots filled (100%), {user_turns} user turns")

        return is_complete

    def finalize_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Finalize session and prepare data for persona generation.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with all collected data or None
        """
        context = self.get_session(session_id)
        if not context:
            return None

        # Mark as complete
        context.phase = ConversationPhase.COMPLETE
        context.updated_at = datetime.utcnow()

        # Compile all slot values
        collected_data = {}
        for name, slot in context.slots.items():
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                collected_data[name] = {
                    "value": slot.value,
                    "confidence": slot.confidence
                }

        result = {
            "session_id": session_id,
            "user_id": context.user_id,
            "collected_data": collected_data,
            "turn_count": len(context.turns),
            "created_at": context.created_at.isoformat(),
            "completed_at": context.updated_at.isoformat()
        }

        logger.info(f"Finalized session {session_id} with {len(collected_data)} slots")
        return result


# Global instance
context_manager = ContextManager()
