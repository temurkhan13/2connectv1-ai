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
        self.llm_extractor = LLMSlotExtractor()  # LLM-based extraction
        self.schema = SlotSchema()

        # Session storage (in production, use Redis or DB)
        self._sessions: Dict[str, ConversationContext] = {}

        # Store LLM responses for contextual follow-ups
        self._llm_responses: Dict[str, LLMExtractionResult] = {}

        # Configuration
        self.max_history_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "50"))
        self.session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))

        # Redis persistence
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            # Support rediss:// URLs (Upstash, etc.)
            redis_kwargs = {}
            if self.redis_url.startswith("rediss://"):
                redis_kwargs["ssl_cert_reqs"] = "CERT_NONE"
            self.redis = redis.from_url(self.redis_url, **redis_kwargs)
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

    def add_turn(
        self,
        session_id: str,
        turn_type: TurnType,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Optional[ConversationTurn]:
        """
        Add a turn to the conversation and extract any slots.

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
        extracted_slot_names = []
        if turn_type == TurnType.USER:
            extracted_slot_names = self._extract_slots_from_turn(context, content)

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

    def _extract_slots_from_turn(
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

            # Use LLM to extract slots with full comprehension
            llm_result = self.llm_extractor.extract_slots(
                user_message=content,
                conversation_history=conversation_history,
                already_filled_slots=already_filled
            )

            logger.info(f"LLM returned slots: {list(llm_result.extracted_slots.keys())}")

            # Store LLM result for response generation
            self._llm_responses[context.session_id] = llm_result

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
            return self._extract_slots_regex_fallback(context, content)

        return extracted_names

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
        3. User explicitly signals completion ("done", "that's all", etc.)
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

        # Check if enough required slots are filled
        return self._all_required_slots_filled(context)

    def _user_signals_completion(self, context: 'ConversationContext') -> bool:
        """
        Detect if user explicitly signals they want to finish onboarding.

        Phrases like "done", "that's all", "let's start matching", "no more", etc.
        """
        # Only check recent user turns
        recent_user_turns = [
            t for t in context.turns[-4:]
            if t.turn_type == TurnType.USER
        ]

        completion_phrases = [
            "done", "that's all", "that's everything", "lets start", "let's start",
            "start matching", "see my matches", "ready to match", "no more",
            "nothing else", "i'm ready", "im ready", "proceed", "move on",
            "next page", "next step", "finish", "complete", "that covers",
            "no, that", "no that", "no.", "nope"
        ]

        for turn in recent_user_turns:
            content_lower = turn.content.lower().strip()
            # Check for completion phrases
            for phrase in completion_phrases:
                if phrase in content_lower:
                    # Also verify we have minimum viable profile (at least 3 slots)
                    filled_slots = [
                        name for name, slot in context.slots.items()
                        if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
                    ]
                    if len(filled_slots) >= 3:
                        return True

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

        # Complete if 80%+ of required slots are filled (allows some flexibility)
        if not required_slots:
            return False

        completion_ratio = filled_count / len(required_slots)
        is_complete = completion_ratio >= 0.8

        if is_complete:
            logger.info(f"Session auto-complete: {filled_count}/{len(required_slots)} required slots filled ({completion_ratio*100:.0f}%)")

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
