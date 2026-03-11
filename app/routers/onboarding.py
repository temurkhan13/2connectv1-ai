"""
Onboarding Router - Chat-based conversational onboarding.

Provides endpoints for:
1. Starting a new onboarding session
2. Chat-based slot extraction
3. Session progress tracking
4. Session finalization
5. Resume upload during onboarding
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import uuid
import base64
import ssl
import redis
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import traceback

# REC-487: Thread pool for background onboarding tasks
# This allows heavy operations (embeddings, matching, persona) to run
# after the HTTP response is sent, preventing timeout errors
_onboarding_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="onboarding_bg")

# REC-487: Retry configuration for background tasks
_BG_MAX_RETRIES = 3
_BG_RETRY_DELAYS = [60, 120, 240]  # Exponential backoff: 1min, 2min, 4min


def _send_admin_alert(subject: str, message: str, user_id: str) -> None:
    """
    Send alert to admin when background onboarding tasks fail after all retries.
    Uses Sentry for error tracking (already configured in the AI service).
    """
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("alert_type", "onboarding_background_failure")
            scope.set_tag("user_id", user_id)
            scope.set_level("error")
            sentry_sdk.capture_message(f"[ADMIN ALERT] {subject}: {message}")
        logger.error(f"[ADMIN ALERT] {subject} for user {user_id}: {message}")
    except Exception as e:
        # If Sentry fails, at least log it
        logger.error(f"[ADMIN ALERT - Sentry failed] {subject} for user {user_id}: {message}. Sentry error: {e}")

from app.services.slot_extraction import SlotExtractor, ExtractedSlot, SlotStatus
from app.services.context_manager import ContextManager, TurnType
from app.services.progressive_disclosure import ProgressiveDisclosure
from app.services.use_case_templates import get_template, get_onboarding_slots
from app.adapters.supabase_onboarding import supabase_onboarding_adapter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/onboarding", tags=["onboarding"])

# Initialize services
slot_extractor = SlotExtractor()
context_manager = ContextManager(slot_extractor)
progressive_disclosure = ProgressiveDisclosure(context_manager)


# BUG-002 FIX: Slot-to-keyword mapping for hard filtering LLM questions
# This prevents LLM from asking about topics that already have filled slots
SLOT_QUESTION_KEYWORDS = {
    "primary_goal": ["goal", "objective", "looking for", "hoping to", "trying to", "want to achieve", "aim", "target"],
    "user_type": ["role", "describe yourself", "what do you do", "who are you", "your background", "professional"],
    "industry_focus": ["industry", "sector", "vertical", "market", "field", "domain", "space"],
    "stage_preference": ["stage", "seed", "series", "pre-seed", "early-stage", "growth", "maturity", "company stage"],
    "geography": ["geography", "location", "region", "country", "market focus", "based in", "where", "uk", "us", "europe"],
    "engagement_style": ["engagement", "collaborate", "work together", "partnership", "involvement", "hands-on", "passive"],
    "investment_range": ["invest", "investment", "ticket size", "check size", "capital", "funding amount", "how much"],
    "funding_range": ["raise", "raising", "funding", "investment needed", "capital required", "how much funding"],
    "team_size": ["team", "employees", "people", "headcount", "staff", "how many people"],
    "requirements": ["need", "looking for", "require", "want from", "seeking", "help with", "support"],
    "offerings": ["offer", "provide", "bring", "contribute", "expertise", "can help with", "value add"],
    "dealbreakers": ["dealbreaker", "won't work", "no-go", "avoid", "not interested in", "red flag"],
    "specialization": ["specialize", "specialization", "expertise", "focus area", "niche", "strength"],
    "target_clients": ["clients", "customers", "who do you serve", "target market", "who do you work with"],
    "years_experience": ["experience", "years", "how long", "background", "track record"],
}


def _question_covers_filled_slot(question: str, slots: Dict[str, Any]) -> Optional[str]:
    """
    BUG-002 FIX: Check if an LLM-generated question covers an already-filled slot.

    Args:
        question: The LLM-generated follow-up question
        slots: Dict of slot_name -> ExtractedSlot objects

    Returns:
        The name of the filled slot that the question covers, or None if no overlap.
    """
    if not question:
        return None

    question_lower = question.lower()

    # Get filled slot names
    filled_slots = set()
    for slot_name, slot in slots.items():
        if hasattr(slot, 'status'):
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                filled_slots.add(slot_name)
        elif isinstance(slot, dict) and slot.get('status') in ['filled', 'confirmed']:
            filled_slots.add(slot_name)

    if not filled_slots:
        return None

    # Check if question contains keywords for any filled slot
    for slot_name in filled_slots:
        keywords = SLOT_QUESTION_KEYWORDS.get(slot_name, [])
        for keyword in keywords:
            if keyword in question_lower:
                logger.info(f"BUG-002 FIX: LLM question '{question[:50]}...' covers filled slot '{slot_name}' (keyword: '{keyword}')")
                return slot_name

    return None


def _run_background_onboarding_tasks(
    user_id: str,
    session_id: str,
    slots: Dict[str, Any],
    diag_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    REC-487: Background processing for heavy onboarding operations.

    Runs AFTER the HTTP response is sent to user. User sees success immediately,
    these operations complete in background (~30-60s):
    1. Multi-vector embeddings (12+ API calls)
    2. Basic embeddings (2 API calls)
    3. Inline bidirectional matching (compares all users)
    4. Persona generation pipeline

    Returns dict with success status and any errors for retry logic.
    """
    thread_name = threading.current_thread().name
    logger.info(f"[BG:{thread_name}] Starting background onboarding tasks for user {user_id}")

    results = {
        "multi_vector_ok": False,
        "basic_embeddings_ok": False,
        "matching_ok": False,
        "persona_ok": False,
        "errors": []
    }

    # 1. Generate multi-vector embeddings for 6-dimension matching
    try:
        from app.services.multi_vector_matcher import multi_vector_matcher

        persona_data = {
            "primary_goal": slots.get("primary_goal", {}).get("value", ""),
            "objective": slots.get("user_type", {}).get("value", ""),
            "industry": slots.get("industry_focus", {}).get("value", ""),
            "stage": slots.get("stage_preference", {}).get("value", ""),
            "investment_stage": slots.get("investment_stage", {}).get("value", ""),
            "company_stage": slots.get("company_stage", {}).get("value", ""),
            "geography": slots.get("geography", {}).get("value", ""),
            "engagement_style": slots.get("engagement_style", {}).get("value", ""),
            "dealbreakers": slots.get("dealbreakers", {}).get("value", ""),
            "requirements": slots.get("requirements", {}).get("value", ""),
            "offerings": slots.get("offerings", {}).get("value", ""),
        }

        req_results = multi_vector_matcher.store_multi_vector_embeddings(
            user_id=user_id,
            persona_data=persona_data,
            direction="requirements"
        )
        off_results = multi_vector_matcher.store_multi_vector_embeddings(
            user_id=user_id,
            persona_data=persona_data,
            direction="offerings"
        )

        stored_count = sum(1 for v in {**req_results, **off_results}.values() if v)
        results["multi_vector_ok"] = stored_count > 0
        logger.info(f"[BG:{thread_name}] Stored {stored_count} multi-vector embeddings for {user_id}")
    except Exception as e:
        results["errors"].append(f"multi_vector: {e}")
        logger.warning(f"[BG:{thread_name}] Failed multi-vector embeddings for {user_id}: {e}")

    # 2. Generate basic 2-vector embeddings
    try:
        from app.services.embedding_service import embedding_service

        requirements_text = slots.get("requirements", {}).get("value", "")
        offerings_text = slots.get("offerings", {}).get("value", "")

        if requirements_text or offerings_text:
            basic_stored = embedding_service.store_user_embeddings(
                user_id=user_id,
                requirements=requirements_text,
                offerings=offerings_text
            )
            results["basic_embeddings_ok"] = basic_stored
            if basic_stored:
                logger.info(f"[BG:{thread_name}] Stored basic embeddings for {user_id}")
        else:
            results["basic_embeddings_ok"] = True  # Nothing to store is OK
    except Exception as e:
        results["errors"].append(f"basic_embeddings: {e}")
        logger.warning(f"[BG:{thread_name}] Failed basic embeddings for {user_id}: {e}")

    # 3. Inline bidirectional matching - CRITICAL for user experience
    try:
        from app.services.inline_matching_service import inline_matching_service
        match_result = inline_matching_service.calculate_and_sync_matches_bidirectional(user_id)

        if match_result.get('success'):
            results["matching_ok"] = True
            logger.info(
                f"[BG:{thread_name}] {user_id}: "
                f"{match_result.get('new_user_matches', 0)} matches, "
                f"{match_result.get('reciprocal_updates', 0)} reciprocal updates"
            )
        else:
            results["errors"].append(f"matching: {match_result.get('errors')}")
            logger.warning(f"[BG:{thread_name}] Match sync failed for {user_id}: {match_result.get('errors')}")
    except Exception as e:
        results["errors"].append(f"matching: {e}")
        logger.warning(f"[BG:{thread_name}] Failed inline matching for {user_id}: {e}")

    # 4. Trigger persona generation pipeline
    try:
        from celery import chain
        from app.workers.resume_processing import process_resume_task
        from app.workers.persona_processing import generate_persona_task

        # Check if resume was uploaded during session
        resume_data = None
        try:
            redis_client = _get_redis_client()
            resume_b64 = redis_client.get(f"resume:{session_id}")
            if resume_b64:
                resume_meta = redis_client.hgetall(f"resume_meta:{session_id}")
                resume_data = {
                    "content": resume_b64,
                    "filename": resume_meta.get("filename", "resume.pdf"),
                    "content_type": resume_meta.get("content_type", "application/pdf")
                }
                logger.info(f"[BG:{thread_name}] Found resume for session {session_id}")
                redis_client.delete(f"resume:{session_id}", f"resume_meta:{session_id}")
        except Exception as e:
            logger.warning(f"[BG:{thread_name}] Could not retrieve resume: {e}")

        workflow = chain(
            process_resume_task.s(user_id, resume_data),
            generate_persona_task.s()
        )
        result = workflow.apply_async()
        results["persona_ok"] = True
        logger.info(f"[BG:{thread_name}] Persona pipeline started for {user_id}, task_id: {result.id}")
    except Exception as e:
        results["errors"].append(f"persona: {e}")
        logger.warning(f"[BG:{thread_name}] Failed to start persona pipeline for {user_id}: {e}")

    logger.info(f"[BG:{thread_name}] Background onboarding tasks completed for {user_id}: {results}")
    return results


def _run_background_onboarding_with_retry(
    user_id: str,
    session_id: str,
    slots: Dict[str, Any],
    diag_dict: Dict[str, Any]
) -> None:
    """
    Wrapper that retries background tasks up to 3 times with exponential backoff.
    Sends admin alert if all retries fail.
    """
    thread_name = threading.current_thread().name
    last_error = None

    for attempt in range(_BG_MAX_RETRIES):
        try:
            results = _run_background_onboarding_tasks(user_id, session_id, slots, diag_dict)

            # Check if critical operations succeeded
            # Matching is most important - user needs to see matches on dashboard
            if results.get("matching_ok"):
                logger.info(f"[BG:{thread_name}] Background tasks succeeded for {user_id} on attempt {attempt + 1}")
                return  # Success!

            # If matching failed, treat as failure for retry
            if results.get("errors"):
                last_error = "; ".join(results["errors"])
                raise Exception(f"Critical operations failed: {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"[BG:{thread_name}] Background tasks failed for {user_id} "
                f"(attempt {attempt + 1}/{_BG_MAX_RETRIES}): {e}"
            )

            # If not the last attempt, wait before retrying
            if attempt < _BG_MAX_RETRIES - 1:
                delay = _BG_RETRY_DELAYS[attempt]
                logger.info(f"[BG:{thread_name}] Retrying in {delay}s...")
                time.sleep(delay)

    # All retries exhausted - send admin alert
    _send_admin_alert(
        subject="Onboarding Background Tasks Failed",
        message=f"All {_BG_MAX_RETRIES} retry attempts failed. Last error: {last_error}. "
                f"User may not have matches until cron job runs.",
        user_id=user_id
    )


class StartSessionRequest(BaseModel):
    """Request to start a new onboarding session."""
    user_id: str = Field(..., description="User identifier")
    objective: Optional[str] = Field(None, description="User's primary objective if known")


class StartSessionResponse(BaseModel):
    """Response after starting a session."""
    session_id: str
    greeting: str
    suggested_questions: List[str]
    progress_percent: float


class ChatMessageRequest(BaseModel):
    """Request for a chat message."""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID (creates new if not provided)")


class ChatMessageResponse(BaseModel):
    """Response to a chat message."""
    session_id: str
    ai_response: str
    extracted_slots: Dict[str, Any]
    all_slots: Dict[str, Any]
    completion_percent: float
    next_questions: List[str]
    phase: str
    is_complete: bool
    is_off_topic: bool = False  # True if user asked off-topic/general knowledge question


class SessionProgressResponse(BaseModel):
    """Session progress information."""
    session_id: str
    user_id: str
    phase: str
    progress_percent: float
    slots_filled: int
    total_required: int
    estimated_remaining_minutes: float
    is_complete: bool


class FinalizeSessionResponse(BaseModel):
    """Response after finalizing a session."""
    session_id: str
    user_id: str
    collected_data: Dict[str, Any]
    turn_count: int
    completed_at: str


@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """
    Start a new onboarding session.

    Creates a new conversation context and returns a greeting
    with suggested first questions.
    """
    try:
        # Create session
        context = context_manager.create_session(request.user_id)

        # BUG-029 FIX: Restore slots from Supabase if user had a previous expired session
        # This ensures returning users don't lose progress
        try:
            supabase_slots = await supabase_onboarding_adapter.get_user_slots(request.user_id)
            if supabase_slots:
                restored_count = 0
                for slot_name, slot_data in supabase_slots.items():
                    context.slots[slot_name] = ExtractedSlot(
                        name=slot_name,
                        value=slot_data.get("value"),
                        confidence=slot_data.get("confidence", 1.0),
                        status=SlotStatus.FILLED if slot_data.get("status") == "filled" else SlotStatus.CONFIRMED
                    )
                    restored_count += 1
                logger.info(f"BUG-029 FIX: Restored {restored_count} slots from Supabase on session start for user {request.user_id[:8]}...")
        except Exception as restore_error:
            logger.warning(f"BUG-029: Could not restore slots on session start: {restore_error}")

        # Get template-based questions if objective provided
        if request.objective:
            focus_slots = get_onboarding_slots(request.objective)
            context.metadata["objective"] = request.objective
            context.metadata["focus_slots"] = focus_slots

        # Generate greeting
        greeting = _generate_greeting(request.objective)

        # Get first batch of questions
        batch = progressive_disclosure.get_next_batch(context.session_id)
        suggested_questions = []
        if batch:
            suggested_questions = [q.question_text for q in batch.questions[:3]]

        logger.info(f"Started onboarding session {context.session_id} for user {request.user_id}")

        # BUG-029 FIX: Calculate actual progress from restored slots (not hardcoded 0.0)
        progress = progressive_disclosure.get_progress_summary(context.session_id)
        progress_percent = progress.get("progress_percent", 0.0)

        return StartSessionResponse(
            session_id=context.session_id,
            greeting=greeting,
            suggested_questions=suggested_questions,
            progress_percent=progress_percent
        )
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatMessageResponse)
async def chat(request: ChatMessageRequest):
    """
    Process a chat message and extract slots.

    This is the main conversational endpoint. It:
    1. Creates or retrieves the session
    2. Extracts slots from the user's message
    3. Generates a contextual AI response
    4. Returns progress and next questions
    """
    import hashlib

    try:
        # DEDUPLICATION CHECK (Sentry 7311102382): Prevent duplicate message processing
        # If same message from same user received within 10s, skip processing
        if request.session_id:
            message_hash = hashlib.md5(
                f"{request.user_id}:{request.session_id}:{request.message.strip().lower()}".encode()
            ).hexdigest()[:16]
            dedup_key = f"onboarding:dedup:{message_hash}"

            try:
                redis_client = _get_redis_client()
                # SETNX returns True if key was set (first request), False if exists (duplicate)
                is_first_request = redis_client.setnx(dedup_key, "1")
                if is_first_request:
                    # Set expiry - only the first request sets the key
                    redis_client.expire(dedup_key, 10)  # 10 second dedup window
                else:
                    # Duplicate request detected - return cached response from context
                    logger.warning(f"[DEDUP] Duplicate chat message detected for session {request.session_id}")
                    context = context_manager.get_session(request.session_id)
                    if context and context.turns:
                        # Return the last AI response
                        last_ai_turn = next(
                            (t for t in reversed(context.turns) if t.turn_type == TurnType.ASSISTANT),
                            None
                        )
                        if last_ai_turn:
                            all_slots = {
                                name: {"value": slot.value, "confidence": slot.confidence, "status": slot.status.value}
                                for name, slot in context.slots.items()
                            }
                            progress = progressive_disclosure.get_progress_summary(request.session_id)
                            return ChatMessageResponse(
                                session_id=request.session_id,
                                ai_response=last_ai_turn.content,
                                extracted_slots={},
                                all_slots=all_slots,
                                completion_percent=progress.get("progress_percent", 0.0),
                                next_questions=[],
                                phase=context.phase.value,
                                is_complete=context_manager.is_complete(request.session_id),
                                is_off_topic=False
                            )
            except Exception as e:
                # Redis error - log but continue processing (dedup is best-effort)
                logger.warning(f"[DEDUP] Redis dedup check failed: {e}")

        # Get or create session
        if request.session_id:
            context = context_manager.get_session(request.session_id)
            if not context:
                # BUG-029 FIX: Session expired - create new one AND restore slots from Supabase
                # Previously: New session created with empty slots, user stuck at in_progress
                # Fix: Load existing slots from Supabase so is_complete() can detect completion
                context = context_manager.create_session(request.user_id)
                logger.info(f"Session {request.session_id} expired, created new: {context.session_id}")

                # BUG-029: Restore slots from Supabase into new session
                try:
                    supabase_slots = await supabase_onboarding_adapter.get_user_slots(request.user_id)
                    if supabase_slots:
                        restored_count = 0
                        for slot_name, slot_data in supabase_slots.items():
                            context.slots[slot_name] = ExtractedSlot(
                                name=slot_name,
                                value=slot_data.get("value"),
                                confidence=slot_data.get("confidence", 1.0),
                                status=SlotStatus.FILLED if slot_data.get("status") == "filled" else SlotStatus.CONFIRMED
                            )
                            restored_count += 1
                        logger.info(f"BUG-029 FIX: Restored {restored_count} slots from Supabase for user {request.user_id[:8]}...")
                except Exception as restore_error:
                    logger.warning(f"BUG-029: Could not restore slots from Supabase (continuing with empty): {restore_error}")
        else:
            context = context_manager.create_session(request.user_id)

        session_id = context.session_id

        # Add user turn and extract slots
        # BUG-008 FIX: Await async add_turn
        turn = await context_manager.add_turn(
            session_id=session_id,
            turn_type=TurnType.USER,
            content=request.message
        )

        # Get newly extracted slots from this turn
        newly_extracted = {}
        if turn and turn.extracted_slots:
            for slot_name in turn.extracted_slots:
                slot = context.slots.get(slot_name)
                if slot:
                    newly_extracted[slot_name] = {
                        "value": slot.value,
                        "confidence": slot.confidence
                    }

        # Get all slots for response
        all_slots = {}
        for name, slot in context.slots.items():
            all_slots[name] = {
                "value": slot.value,
                "confidence": slot.confidence,
                "status": slot.status.value
            }

        # BUG-027 FIX: Persist slots to Supabase for session resilience
        # This ensures slots survive if Redis session expires before profile creation
        if newly_extracted and context.user_id:
            try:
                slots_to_save = [
                    {
                        "name": slot_name,
                        "value": str(slot_data.get("value", "")),
                        "confidence": slot_data.get("confidence", 1.0),
                        "source_text": request.message,
                        "extraction_method": "llm",
                        "status": "filled"
                    }
                    for slot_name, slot_data in newly_extracted.items()
                ]
                if slots_to_save:
                    saved_count = await supabase_onboarding_adapter.save_slots_batch(
                        user_id=context.user_id,
                        slots=slots_to_save
                    )
                    logger.info(f"BUG-027: Persisted {saved_count} newly extracted slots to Supabase for user {context.user_id[:8]}...")
            except Exception as persist_error:
                # Log but don't fail the request - in-memory session is primary
                logger.warning(f"BUG-027: Failed to persist slots to Supabase (non-fatal): {persist_error}")

        # Calculate progress
        progress = progressive_disclosure.get_progress_summary(session_id)
        completion = progress.get("progress_percent", 0.0)

        # BUG-001 FIX: Check if user signals completion BEFORE generating response
        # This prevents the AI from asking follow-up questions when user says "I'm done"
        is_complete = context_manager.is_complete(session_id)

        # Get LLM result early so we can access is_off_topic for the response
        llm_result = context_manager.get_llm_response(session_id)
        is_off_topic = llm_result.is_off_topic if llm_result else False

        if is_complete:
            # User signaled completion OR all required slots filled
            # Skip follow-up question generation entirely
            ai_response = "Perfect! I have everything I need to find your matches. Click the button below to complete your profile."
            logger.info(f"Session {session_id}: User completion detected, skipping follow-up")
        else:
            # Generate AI response using LLM result if available
            # NOTE: Only show follow_up_question to user - understanding_summary is internal
            if llm_result and llm_result.follow_up_question:
                # BUG-002 FIX: Hard filter - check if LLM question covers an already-filled slot
                # LLMs are unreliable at following negative constraints ("DO NOT ASK AGAIN")
                # so we enforce the constraint here in code
                covered_slot = _question_covers_filled_slot(llm_result.follow_up_question, context.slots)

                if covered_slot:
                    # LLM asked about a filled slot - replace with progressive_disclosure question
                    logger.info(f"BUG-002 FIX: Replacing LLM question (covers '{covered_slot}') with progressive_disclosure question")
                    batch = progressive_disclosure.get_next_batch(session_id)
                    if batch and batch.questions:
                        # Use the first question from progressive_disclosure (already filtered for filled slots)
                        ai_response = batch.questions[0].question_text
                        logger.info(f"BUG-002 FIX: Using progressive_disclosure question: {ai_response[:50]}...")
                    else:
                        # No more questions from progressive_disclosure - use fallback
                        ai_response = _generate_contextual_response(context, newly_extracted, turn.extracted_slots if turn else [], request.message)
                else:
                    # LLM question is valid - use it
                    ai_response = llm_result.follow_up_question
            else:
                # Fallback to template-based response
                ai_response = _generate_contextual_response(context, newly_extracted, turn.extracted_slots if turn else [], request.message)

        # Add assistant turn
        # BUG-008 FIX: Await async add_turn
        await context_manager.add_turn(
            session_id=session_id,
            turn_type=TurnType.ASSISTANT,
            content=ai_response
        )

        # Get next questions (only if not complete)
        next_questions = []
        if not is_complete:
            batch = progressive_disclosure.get_next_batch(session_id)
            if batch:
                next_questions = [q.question_text for q in batch.questions]

        # is_complete already computed above (BUG-001 fix)

        return ChatMessageResponse(
            session_id=session_id,
            ai_response=ai_response,
            extracted_slots=newly_extracted,
            all_slots=all_slots,
            completion_percent=completion,
            next_questions=next_questions,
            phase=context.phase.value,
            is_complete=is_complete,
            is_off_topic=is_off_topic
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{session_id}", response_model=SessionProgressResponse)
async def get_progress(session_id: str):
    """Get progress for an onboarding session."""
    context = context_manager.get_session(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")

    progress = progressive_disclosure.get_progress_summary(session_id)

    return SessionProgressResponse(
        session_id=session_id,
        user_id=context.user_id,
        phase=context.phase.value,
        progress_percent=progress.get("progress_percent", 0.0),
        slots_filled=progress.get("slots_filled", 0),
        total_required=len(context.slots) + 5,  # Estimate
        estimated_remaining_minutes=progress.get("estimated_remaining_minutes", 0.0),
        is_complete=context_manager.is_complete(session_id)
    )


@router.post("/finalize/{session_id}", response_model=FinalizeSessionResponse)
async def finalize_session(session_id: str):
    """
    Finalize an onboarding session.

    Marks the session as complete and returns all collected data
    ready for persona generation.
    """
    result = context_manager.finalize_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")

    return FinalizeSessionResponse(
        session_id=result["session_id"],
        user_id=result["user_id"],
        collected_data=result["collected_data"],
        turn_count=result["turn_count"],
        completed_at=result["completed_at"]
    )


@router.get("/slots/{session_id}")
async def get_slots(session_id: str):
    """Get all extracted slots for a session."""
    summary = context_manager.get_slot_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


def _get_redis_client():
    """Get Redis client for session storage."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Support rediss:// URLs (Upstash, etc.)
    redis_kwargs = {"decode_responses": True}
    if redis_url.startswith("rediss://"):
        redis_kwargs["ssl_cert_reqs"] = "none"  # String format for redis-py
    return redis.from_url(redis_url, **redis_kwargs)


class ResumeUploadResponse(BaseModel):
    """Response after uploading a resume."""
    success: bool
    message: str
    filename: str


@router.post("/upload-resume", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...)
):
    """
    Upload resume during conversational onboarding.

    The resume is stored temporarily in Redis with the session,
    and will be processed when onboarding completes.
    """
    # Validate file type
    allowed_types = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and Word documents are allowed"
        )

    # Read and validate size (5MB max)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must be under 5MB"
        )

    # Verify session exists
    context = context_manager.get_session(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")

    if context.user_id != user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch")

    try:
        # ISSUE-1 FIX: Extract resume text IMMEDIATELY during upload
        # This allows the text to be used for question filtering during onboarding
        # The full resume processing (persona generation) still happens after onboarding
        from app.services.resume_service import ResumeService
        resume_service = ResumeService()

        extraction_result = resume_service.extract_text_from_content(
            content=contents,
            filename=file.filename,
            content_type=file.content_type
        )

        extracted_text = ""
        if extraction_result.get("success"):
            extracted_text = extraction_result.get("text", "")
            logger.info(f"ISSUE-1 FIX: Extracted {len(extracted_text)} chars from resume during upload")
        else:
            # Log but don't fail - resume will be re-processed after onboarding
            logger.warning(f"ISSUE-1 FIX: Could not extract text during upload: {extraction_result.get('error')}")

        # Store resume content in Redis (base64 encoded) - for Celery processing later
        redis_client = _get_redis_client()
        redis_client.setex(
            f"resume:{session_id}",
            3600,  # 1 hour TTL
            base64.b64encode(contents).decode('utf-8')
        )

        # Store metadata in a separate key
        redis_client.hset(f"resume_meta:{session_id}", mapping={
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "user_id": user_id
        })
        redis_client.expire(f"resume_meta:{session_id}", 3600)

        # ISSUE-1 FIX: Store extracted text separately for use during onboarding
        if extracted_text:
            redis_client.setex(
                f"resume_text:{session_id}",
                3600,  # 1 hour TTL
                extracted_text
            )
            logger.info(f"ISSUE-1 FIX: Stored extracted resume text for session {session_id}")

            # BUG-043 FIX: Pre-extract slots from resume to accelerate onboarding
            # This allows the onboarding to skip questions about info already in resume
            try:
                from app.services.llm_slot_extractor import LLMSlotExtractor
                import json as json_module

                extractor = LLMSlotExtractor()
                pre_extracted_slots = extractor.extract_slots_from_resume(extracted_text)

                if pre_extracted_slots:
                    # Store pre-extracted slots in Redis for the session
                    redis_client.setex(
                        f"resume_slots:{session_id}",
                        3600,  # 1 hour TTL
                        json_module.dumps(pre_extracted_slots)
                    )
                    logger.info(f"BUG-043 FIX: Pre-extracted {len(pre_extracted_slots)} slots from resume: {list(pre_extracted_slots.keys())}")

                    # Also load them into the session context immediately
                    from app.services.slot_extraction import ExtractedSlot, SlotStatus
                    for slot_name, slot_data in pre_extracted_slots.items():
                        context.slots[slot_name] = ExtractedSlot(
                            name=slot_name,
                            value=slot_data.get("value"),
                            confidence=slot_data.get("confidence", 0.85),
                            status=SlotStatus.FILLED,
                            source_text="[from resume]",
                            extracted_at=datetime.utcnow()
                        )
                    logger.info(f"BUG-043 FIX: Loaded {len(pre_extracted_slots)} slots into session context")

                    # BUG-043: Also persist to Supabase for session resilience
                    try:
                        slots_to_save = [
                            {
                                "name": slot_name,
                                "value": str(slot_data.get("value", "")),
                                "confidence": slot_data.get("confidence", 0.85),
                                "source_text": "[extracted from resume]",
                                "extraction_method": "resume_llm",
                                "status": "filled"
                            }
                            for slot_name, slot_data in pre_extracted_slots.items()
                        ]
                        saved_count = await supabase_onboarding_adapter.save_slots_batch(
                            user_id=user_id,
                            slots=slots_to_save
                        )
                        logger.info(f"BUG-043 FIX: Persisted {saved_count} resume-extracted slots to Supabase")
                    except Exception as persist_error:
                        logger.warning(f"BUG-043: Could not persist resume slots to Supabase: {persist_error}")
            except Exception as e:
                # Don't fail the upload if pre-extraction fails
                logger.warning(f"BUG-043: Could not pre-extract slots from resume (non-fatal): {e}")

        logger.info(f"Resume uploaded for session {session_id}: {file.filename} ({len(contents)} bytes)")

        return ResumeUploadResponse(
            success=True,
            message="Resume uploaded successfully" + (f" - extracted {len(extracted_text)} characters" if extracted_text else ""),
            filename=file.filename
        )
    except Exception as e:
        logger.error(f"Failed to store resume: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store resume: {str(e)}")


class CompleteOnboardingRequest(BaseModel):
    """Request to complete onboarding and create user profile."""
    session_id: str = Field(..., description="Session ID to complete")
    user_id: Optional[str] = Field(None, description="User ID for the profile (optional - derived from session if not provided)")


class OnboardingDiagnostics(BaseModel):
    """Diagnostics for debugging onboarding issues."""
    dynamo_profile_created: bool = False
    postgres_status_updated: bool = False
    user_summary_created: bool = False
    multi_vector_embeddings_count: int = 0
    basic_embeddings_stored: bool = False
    matches_found: int = 0
    matches_synced: int = 0
    match_sync_error: Optional[str] = None


class CompleteOnboardingResponse(BaseModel):
    """Response after completing onboarding."""
    success: bool
    user_id: str
    message: str
    profile_created: bool
    persona_task_id: Optional[str] = None
    diagnostics: Optional[OnboardingDiagnostics] = None


@router.post("/complete", response_model=CompleteOnboardingResponse)
async def complete_onboarding(request: CompleteOnboardingRequest):
    """
    Complete onboarding by creating the user profile in DynamoDB.

    This endpoint:
    1. Retrieves the finalized session data (extracted slots)
    2. Converts slots to question/answer format
    3. Creates the user profile in DynamoDB
    4. Triggers persona generation pipeline

    Call this after the user reviews and approves their onboarding summary.
    """
    from celery import chain
    from app.adapters.supabase_profiles import UserProfile, QuestionAnswer
    from app.workers.persona_processing import generate_persona_task
    from app.workers.resume_processing import process_resume_task

    session_id = request.session_id

    logger.info(f"Completing onboarding for session {session_id}")

    try:
        # BUG-025 FIX: Add Supabase fallback for resilience
        # Try to get session data from in-memory context manager first
        slot_summary = context_manager.get_slot_summary(session_id)

        # Get user_id early (needed for both in-memory and Supabase fallback)
        user_id = request.user_id
        if not user_id:
            # Get user_id from the session context
            context = context_manager.get_session(session_id)
            if context and context.user_id:
                user_id = context.user_id
                logger.info(f"Derived user_id {user_id} from session {session_id}")
            else:
                # Last resort: user_id must be provided in request if session not in memory
                if not slot_summary or not slot_summary.get("slots"):
                    raise HTTPException(
                        status_code=400,
                        detail="user_id required when session not in memory"
                    )

        # BUG-025 FIX: Fallback to Supabase if in-memory session missing or empty
        if not slot_summary or not slot_summary.get("slots"):
            logger.warning(
                f"Session {session_id} not in memory or empty, "
                f"fetching slots from Supabase for user {user_id}"
            )

            from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
            supabase = SupabaseOnboardingAdapter()

            if not supabase.enabled:
                raise HTTPException(
                    status_code=500,
                    detail="Session expired and Supabase fallback not available"
                )

            # Fetch all slots for this user from Supabase
            supabase_slots = await supabase.get_user_slots(user_id)

            if not supabase_slots:
                raise HTTPException(
                    status_code=404,
                    detail=f"No slots found for user {user_id} in memory or Supabase"
                )

            # Convert Supabase format to slot_summary format
            # Supabase returns: {slot_name: {value, confidence, status, created_at}}
            # slot_summary expects: {slots: {slot_name: {value, confidence, status}}}
            slot_summary = {"slots": supabase_slots}

            logger.info(
                f"Loaded {len(supabase_slots)} slots from Supabase for user {user_id} "
                f"(session {session_id} not in memory)"
            )

        logger.info(f"Completing onboarding for user {user_id}, session {session_id}")

        # IDEMPOTENCY CHECK (Sentry 7311102382): Check if user already completed onboarding
        # If profile exists in DynamoDB, return success instead of failing
        try:
            existing_profile = UserProfile.get(user_id)
            if existing_profile:
                logger.info(f"[IDEMPOTENT] User {user_id} already has DynamoDB profile, returning success")
                return CompleteOnboardingResponse(
                    success=True,
                    user_id=user_id,
                    message="Profile already exists. Onboarding previously completed.",
                    profile_created=True,
                    persona_task_id=None,
                    diagnostics=OnboardingDiagnostics(
                        dynamo_profile_created=True,
                        postgres_status_updated=True,
                    )
                )
        except UserProfile.DoesNotExist:
            # Expected case - user doesn't have profile yet, continue with creation
            pass
        except Exception as e:
            # DynamoDB error - log but continue (profile creation will fail properly below)
            logger.warning(f"[IDEMPOTENT] Could not check existing profile: {e}")

        # Convert slots to question/answer format for DynamoDB
        questions = []
        slot_to_question_map = {
            # Core identity slots
            "name": "What is your name?",
            "role_title": "What is your job title or designation?",
            "experience_years": "How many years of experience do you have?",
            "company_name": "What is your company name?",
            # Goals and objectives
            "primary_goal": "What are you looking for?",
            "user_type": "What's your role or background?",
            "industry_focus": "What industry or sector are you in?",
            "experience_level": "What's your experience level?",
            "stage_preference": "What stage are you interested in?",
            "check_size": "What's your typical investment size or budget?",
            "geographic_focus": "What geographic regions are you focused on?",
            # Matching criteria
            "offerings": "What can you offer to connections?",
            "requirements": "What do you need from connections?",
            "timeline": "What's your timeline?",
            "skills": "What are your key skills?",
            "company_info": "Tell me about your company/project",
            # Founder-specific
            "company_stage": "What stage is your company at?",
            "funding_need": "How much funding are you seeking?",
            "team_size": "What is your team size?",
            # Investor-specific
            "investment_stage": "What stages do you invest in?",
        }

        # Internal slots that should NOT be stored as user questions
        # These are metadata/debug fields from the LLM extraction process
        INTERNAL_SLOTS_TO_SKIP = {
            "missing_important_slots",
            "follow_up_question",
            "understanding_summary",
            "user_type_inference",
            "extraction_confidence",
            "extracted_slots",  # This is a nested container, not a value
        }

        slots = slot_summary.get("slots", {})
        for slot_name, slot_data in slots.items():
            # Skip internal/metadata slots
            if slot_name in INTERNAL_SLOTS_TO_SKIP:
                continue

            if isinstance(slot_data, dict):
                value = slot_data.get("value", slot_data.get("raw_value", ""))
            else:
                value = str(slot_data)

            if value:
                prompt = slot_to_question_map.get(slot_name, f"What is your {slot_name.replace('_', ' ')}?")
                questions.append({"prompt": prompt, "answer": str(value)})

        if not questions:
            logger.warning(f"No slots extracted for session {session_id}")
            return CompleteOnboardingResponse(
                success=False,
                user_id=user_id,
                message="No data collected during onboarding. Please complete the conversation first.",
                profile_created=False
            )

        # Initialize diagnostics tracker
        diag = OnboardingDiagnostics()

        # Create user profile in DynamoDB
        try:
            user_profile = UserProfile.create_user(
                user_id=user_id,
                resume_link=None,
                questions=questions
            )
            user_profile.needs_matchmaking = "true"  # Enable scheduled matching
            user_profile.save()
            diag.dynamo_profile_created = True
            logger.info(f"Created DynamoDB profile for user {user_id} with {len(questions)} Q&A pairs")
        except Exception as e:
            logger.error(f"Failed to create DynamoDB profile: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create profile: {str(e)}")

        # Update PostgreSQL onboarding_status to 'completed'
        # This bridges AI service onboarding to backend user status for dashboard access
        try:
            from app.adapters.postgresql import postgresql_adapter
            status_updated = postgresql_adapter.update_user_onboarding_status(user_id, 'completed')
            if status_updated:
                diag.postgres_status_updated = True
                logger.info(f"Updated PostgreSQL onboarding_status='completed' for user {user_id}")
            else:
                logger.warning(f"Could not update PostgreSQL onboarding_status for user {user_id}")
        except Exception as e:
            # Log but don't fail - DynamoDB profile was created successfully
            logger.warning(f"Failed to update PostgreSQL onboarding_status: {e}")

        # Create user_summary for Discover page + AI Summary display
        # BUG-007 FIX: Generate markdown summary for frontend display instead of JSON
        try:
            import json
            # Build summary from collected slots
            summary_data = {
                "profile_type": slots.get("user_type", {}).get("value", "Unknown"),
                "industry": slots.get("industry_focus", {}).get("value", ""),
                "goal": slots.get("primary_goal", {}).get("value", ""),
                "stage": slots.get("stage_preference", {}).get("value", ""),
                "geography": slots.get("geographic_focus", {}).get("value", ""),
                "offerings": slots.get("offerings", {}).get("value", ""),
                "requirements": slots.get("requirements", {}).get("value", ""),
            }

            # BUG-013 FIX: Ensure offerings/requirements are strings, not lists
            # This prevents DynamoDB serialization errors and embedding generation crashes
            for key in ["offerings", "requirements"]:
                if isinstance(summary_data[key], list):
                    summary_data[key] = "; ".join(str(item).strip() for item in summary_data[key] if item)
                    logger.info(f"BUG-013 FIX: Converted {key} from list to string in summary_data")

            # Generate markdown summary for frontend display (AI Summary page)
            profile_type = summary_data["profile_type"] or "User"
            industry = summary_data["industry"] or "Not specified"
            goal = summary_data["goal"] or "Not specified"
            stage = summary_data["stage"] or "Not specified"
            geography = summary_data["geography"] or "Not specified"
            offerings = summary_data["offerings"] or "Not specified"
            requirements = summary_data["requirements"] or "Not specified"

            summary_markdown = f"""# {profile_type} Profile

## Primary Goal
{goal}

## Industry Focus
{industry}

## Stage/Experience
{stage}

## Geography
{geography}

## What I Can Offer
{offerings}

## What I'm Looking For
{requirements}

---

*This summary was generated based on your onboarding responses. You can update it anytime from your profile settings.*
"""

            summary_id = postgresql_adapter.create_user_summary(
                user_id=user_id,
                summary=summary_markdown,  # Store markdown instead of JSON
                status='draft',
                urgency='ongoing'
            )
            if summary_id:
                diag.user_summary_created = True
                logger.info(f"Created user_summary {summary_id} for user {user_id}")
            else:
                logger.warning(f"Could not create user_summary for user {user_id}")
        except Exception as e:
            # Log but don't fail - user can still use dashboard
            logger.warning(f"Failed to create user_summary: {e}")

        # REC-487: RETURN EARLY, PROCESS IN BACKGROUND
        # ==============================================
        # Previously, this endpoint ran all operations synchronously (~60s+):
        # - Multi-vector embeddings (12+ API calls)
        # - Basic embeddings (2 API calls)
        # - Inline bidirectional matching (compares ALL users)
        # - Persona generation pipeline
        #
        # This caused HTTP timeouts and "Failed to complete onboarding" errors.
        # User would have to click 3-4 times for it to work (idempotency check).
        #
        # NEW APPROACH: Return success immediately after profile creation (~3s),
        # then run heavy operations in background thread.
        # User lands on dashboard, matches appear within ~60s.

        # Log diagnostics before returning
        logger.info(f"Onboarding diagnostics (pre-background) for {user_id}: {diag.model_dump()}")

        # Schedule background tasks - runs AFTER this response is sent
        # Uses ThreadPoolExecutor to avoid blocking the HTTP response
        # Wrapper function handles retry (3x) + admin alert on failure
        try:
            _onboarding_executor.submit(
                _run_background_onboarding_with_retry,
                user_id,
                session_id,
                slots,
                diag.model_dump()
            )
            logger.info(f"[REC-487] Scheduled background onboarding tasks for {user_id}")
        except Exception as e:
            # If scheduling fails, log but don't fail the request
            # User still has their profile, matches will sync via cron job
            logger.error(f"[REC-487] Failed to schedule background tasks for {user_id}: {e}")

        # Return immediately - user sees success in ~3 seconds
        return CompleteOnboardingResponse(
            success=True,
            user_id=user_id,
            message=f"Profile created with {len(questions)} data points. Finding your matches...",
            profile_created=True,
            persona_task_id=None,  # Task runs in background thread, not Celery
            diagnostics=diag
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _detect_multiple_goals(user_message: str) -> List[str]:
    """
    Detect multiple goal signals in user's message.

    Returns a list of detected goal categories. If 2+ are detected,
    the system should ask a direct prioritization question.
    """
    message_lower = user_message.lower()
    detected_goals = []

    # Goal patterns - each tuple is (keywords, friendly_label)
    goal_patterns = [
        (["investor", "invest", "funding", "raise", "series", "vc", "venture capital", "angel"],
         "connecting with investors"),
        (["co-founder", "cofounder", "partner", "founding team", "technical partner"],
         "finding a co-founder"),
        (["mentor", "advisor", "advice", "guidance", "learn from"],
         "finding mentors"),
        (["sales", "customer", "client", "enterprise", "b2b", "go-to-market", "gtm"],
         "finding customers or sales help"),
        (["founder", "startup founder", "entrepreneur", "peer", "community"],
         "connecting with other founders"),
        (["partnership", "strategic partner", "collaborate", "collaboration"],
         "exploring partnerships"),
        (["hire", "hiring", "recruit", "talent", "engineer", "developer", "team member"],
         "hiring talent"),
    ]

    for keywords, label in goal_patterns:
        if any(kw in message_lower for kw in keywords):
            detected_goals.append(label)

    return detected_goals


def _generate_greeting(objective: Optional[str] = None) -> str:
    """Generate a greeting based on objective."""
    if objective:
        template = get_template(objective)
        return (
            f"Welcome! I see you're interested in {template.display_name.lower()}. "
            f"I'll ask you a few questions to understand your needs and find the best matches. "
            f"Feel free to give detailed answers - the more you share, the better I can help."
        )
    return (
        "Welcome to 2Connect! I'm here to help you find great connections. "
        "Let's start by learning a bit about you. Tell me about yourself - "
        "what you're working on and what kind of connections you're looking for."
    )


def _generate_contextual_response(
    context,
    newly_extracted: Dict[str, Any],
    slot_names: List[str],
    user_message: str = ""
) -> str:
    """
    Generate a contextual response based on extracted slots.

    IMPORTANT: Follows Indirect Elicitation principle.
    - Default mode is INDIRECT - follow user's natural thread
    - Never feel like a questionnaire
    - Only use direct questions when truly necessary
    - EXCEPTION: When user mentions MULTIPLE goals, ask direct prioritization question
    """
    # Check what slots are already filled vs still needed
    filled_slots = []
    missing_slots = []
    core_slot_names = ["primary_goal", "user_type", "industry_focus", "stage_preference", "funding_need"]

    for slot_name in core_slot_names:
        slot = context.slots.get(slot_name)
        if slot and slot.value:
            filled_slots.append(slot_name)
        else:
            missing_slots.append(slot_name)

    # MULTI-GOAL DETECTION: If primary_goal is missing but user mentioned multiple goals,
    # ask a DIRECT prioritization question instead of vague indirect prompts
    if "primary_goal" in missing_slots and user_message:
        goal_signals = _detect_multiple_goals(user_message)
        if len(goal_signals) >= 2:
            # User mentioned 2+ different goals - ask for prioritization
            goals_list = ", ".join(goal_signals[:-1]) + " and " + goal_signals[-1]
            logger.info(f"MULTI-GOAL: Detected {len(goal_signals)} goals: {goal_signals}")
            return (
                f"I can see you have several goals - {goals_list}. "
                "If you had to pick ONE primary focus for 2Connect right now, which would be most valuable for you?"
            )

    # INDIRECT follow-up questions - conversational, not form-like
    indirect_prompts = {
        "primary_goal": [
            "Tell me more about what success looks like for you here.",
            "What would make your time on 2Connect worthwhile for you?",
            "What's the main thing you're hoping to accomplish?",
        ],
        "user_type": [
            "Tell me a bit more about your background.",
            "What's your story? How did you get to where you are now?",
            "Walk me through your journey so far.",
        ],
        "industry_focus": [
            "What space gets you most excited?",
            "Tell me about the areas you're most passionate about.",
            "What industries do you find yourself drawn to?",
        ],
        "stage_preference": [
            "Where on this journey are you most comfortable working?",
            "What kind of companies do you love working with?",
            "Tell me about the stage of companies you find most interesting.",
        ],
        "funding_need": [
            "How are you thinking about your next chapter financially?",
            "What resources would help you get to the next milestone?",
            "Tell me about your growth plans.",
        ],
    }

    import random

    if not newly_extracted:
        # No new slots extracted - guide naturally
        if len(filled_slots) >= 3:
            if missing_slots:
                # Pick a random indirect prompt for the first missing slot
                slot = missing_slots[0]
                prompts = indirect_prompts.get(slot, [f"Tell me more about your {slot.replace('_', ' ')}."])
                return random.choice(prompts)
            else:
                return "This is really helpful context. Anything else you'd like to share, or shall we wrap up?"
        else:
            # Still need core info - open-ended invitation
            return (
                "I'd love to hear more. What's the story behind what you're building "
                "and what kind of connections would be most valuable for you?"
            )

    # Don't acknowledge/repeat what they said - just move forward naturally
    if missing_slots:
        # Pick a random indirect prompt for the first missing slot
        slot = missing_slots[0]
        prompts = indirect_prompts.get(slot, [f"Tell me more about your {slot.replace('_', ' ')}."])
        return random.choice(prompts)

    # All core slots filled - wrap up conversationally
    return "This is great context. Feel free to share anything else, or we can move forward whenever you're ready."


# =============================================================================
# DIAGNOSTIC/BACKFILL ENDPOINTS
# =============================================================================

class BackfillResponse(BaseModel):
    """Response from backfill operations."""
    success: bool
    total_users_checked: int
    users_missing_summaries: int
    summaries_created: int
    errors: List[str]


@router.get("/backfill/user-summaries", response_model=BackfillResponse)
async def backfill_user_summaries():
    """
    UX-003 FIX: Backfill missing UserSummaries for completed users.

    Users who complete onboarding but don't have UserSummaries won't appear
    on the Discover page. This endpoint finds and fixes those users.
    """
    from app.adapters.postgresql import PostgreSQLAdapter
    import json

    errors = []
    total_checked = 0
    missing_count = 0
    created_count = 0

    try:
        postgresql_adapter = PostgreSQLAdapter()
        conn = postgresql_adapter.get_backend_connection()
        cursor = conn.cursor()

        # Find completed users without summaries
        cursor.execute("""
            SELECT u.id, u.first_name, u.last_name, u.objective
            FROM users u
            LEFT JOIN user_summaries us ON us.user_id = u.id
            WHERE u.deleted_at IS NULL
            AND u.onboarding_status = 'completed'
            AND u.is_active = true
            AND us.id IS NULL
            ORDER BY u.created_at DESC
        """)
        users_without_summaries = cursor.fetchall()
        total_checked = len(users_without_summaries)
        missing_count = total_checked

        logger.info(f"Found {missing_count} users without summaries")

        for user in users_without_summaries:
            user_id, first_name, last_name, objective = user
            try:
                # Create a basic summary from available data
                summary_data = {
                    "profile_type": "professional",
                    "industry": "",
                    "stage": "",
                    "goal": objective or "seeking connections",
                    "geography": "",
                    "offerings": f"{first_name} is a professional seeking meaningful connections.",
                    "requirements": "",
                }
                summary_json = json.dumps(summary_data)

                summary_id = postgresql_adapter.create_user_summary(
                    user_id=user_id,
                    summary=summary_json,
                    status='draft',
                    urgency='ongoing'
                )

                if summary_id:
                    created_count += 1
                    logger.info(f"Created summary for user {user_id}")
                else:
                    errors.append(f"Failed to create summary for {user_id}")
            except Exception as e:
                errors.append(f"Error for {user_id}: {str(e)}")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Backfill error: {e}")
        errors.append(f"Connection error: {str(e)}")

    return BackfillResponse(
        success=len(errors) == 0,
        total_users_checked=total_checked,
        users_missing_summaries=missing_count,
        summaries_created=created_count,
        errors=errors[:10]  # Limit error list
    )


@router.get("/diagnostics/discover-eligibility")
async def check_discover_eligibility():
    """
    Diagnostic endpoint to check why users might not appear on Discover page.
    """
    from app.adapters.postgresql import PostgreSQLAdapter

    try:
        postgresql_adapter = PostgreSQLAdapter()
        conn = postgresql_adapter.get_backend_connection()
        cursor = conn.cursor()

        # Total users
        cursor.execute("SELECT COUNT(*) FROM users WHERE deleted_at IS NULL")
        total = cursor.fetchone()[0]

        # By onboarding status
        cursor.execute("""
            SELECT onboarding_status, COUNT(*)
            FROM users
            WHERE deleted_at IS NULL
            GROUP BY onboarding_status
        """)
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        # Discoverable (completed + active + not test)
        cursor.execute("""
            SELECT COUNT(*)
            FROM users
            WHERE deleted_at IS NULL
            AND onboarding_status = 'completed'
            AND is_active = true
            AND is_test = false
        """)
        discoverable = cursor.fetchone()[0]

        # With summaries
        cursor.execute("""
            SELECT COUNT(DISTINCT u.id)
            FROM users u
            JOIN user_summaries us ON us.user_id = u.id
            WHERE u.deleted_at IS NULL
            AND u.onboarding_status = 'completed'
            AND u.is_active = true
            AND u.is_test = false
        """)
        with_summaries = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return {
            "success": True,
            "data": {
                "total_users": total,
                "by_onboarding_status": by_status,
                "discoverable_users": discoverable,
                "with_summaries": with_summaries,
                "missing_summaries": discoverable - with_summaries,
                "issue": "Users missing summaries won't appear on Discover" if discoverable > with_summaries else None
            }
        }

    except Exception as e:
        logger.error(f"Diagnostics error: {e}")
        return {
            "success": False,
            "error": str(e),
            "hint": "Check RECIPROCITY_BACKEND_DB_URL environment variable"
        }
