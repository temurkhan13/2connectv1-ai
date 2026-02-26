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
import redis
import os

from app.services.slot_extraction import SlotExtractor, ExtractedSlot
from app.services.context_manager import ContextManager, TurnType
from app.services.progressive_disclosure import ProgressiveDisclosure
from app.services.use_case_templates import get_template, get_onboarding_slots

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/onboarding", tags=["onboarding"])

# Initialize services
slot_extractor = SlotExtractor()
context_manager = ContextManager(slot_extractor)
progressive_disclosure = ProgressiveDisclosure(context_manager)


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

        return StartSessionResponse(
            session_id=context.session_id,
            greeting=greeting,
            suggested_questions=suggested_questions,
            progress_percent=0.0
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
    try:
        # Get or create session
        if request.session_id:
            context = context_manager.get_session(request.session_id)
            if not context:
                # Session expired, create new one
                context = context_manager.create_session(request.user_id)
                logger.info(f"Session {request.session_id} expired, created new: {context.session_id}")
        else:
            context = context_manager.create_session(request.user_id)

        session_id = context.session_id

        # Add user turn and extract slots
        turn = context_manager.add_turn(
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

        # Calculate progress
        progress = progressive_disclosure.get_progress_summary(session_id)
        completion = progress.get("progress_percent", 0.0)

        # Generate AI response using LLM result if available
        # NOTE: Only show follow_up_question to user - understanding_summary is internal
        llm_result = context_manager.get_llm_response(session_id)
        if llm_result and llm_result.follow_up_question:
            # Use only the follow-up question - don't repeat back what user said
            ai_response = llm_result.follow_up_question
        else:
            # Fallback to template-based response
            ai_response = _generate_contextual_response(context, newly_extracted, turn.extracted_slots if turn else [])

        # Add assistant turn
        context_manager.add_turn(
            session_id=session_id,
            turn_type=TurnType.ASSISTANT,
            content=ai_response
        )

        # Get next questions
        batch = progressive_disclosure.get_next_batch(session_id)
        next_questions = []
        if batch:
            next_questions = [q.question_text for q in batch.questions]

        # Check if complete
        is_complete = context_manager.is_complete(session_id)

        return ChatMessageResponse(
            session_id=session_id,
            ai_response=ai_response,
            extracted_slots=newly_extracted,
            all_slots=all_slots,
            completion_percent=completion,
            next_questions=next_questions,
            phase=context.phase.value,
            is_complete=is_complete
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
        redis_kwargs["ssl_cert_reqs"] = "CERT_NONE"
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
        # Store resume content in Redis (base64 encoded)
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

        logger.info(f"Resume uploaded for session {session_id}: {file.filename} ({len(contents)} bytes)")

        return ResumeUploadResponse(
            success=True,
            message="Resume uploaded successfully",
            filename=file.filename
        )
    except Exception as e:
        logger.error(f"Failed to store resume: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store resume: {str(e)}")


class CompleteOnboardingRequest(BaseModel):
    """Request to complete onboarding and create user profile."""
    session_id: str = Field(..., description="Session ID to complete")
    user_id: str = Field(..., description="User ID for the profile")


class CompleteOnboardingResponse(BaseModel):
    """Response after completing onboarding."""
    success: bool
    user_id: str
    message: str
    profile_created: bool
    persona_task_id: Optional[str] = None


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
    from app.adapters.dynamodb import UserProfile, QuestionAnswer
    from app.workers.persona_processing import generate_persona_task
    from app.workers.resume_processing import process_resume_task

    session_id = request.session_id
    user_id = request.user_id

    logger.info(f"Completing onboarding for user {user_id}, session {session_id}")

    try:
        # Get session data
        slot_summary = context_manager.get_slot_summary(session_id)
        if not slot_summary:
            raise HTTPException(status_code=404, detail="Session not found")

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

        # Create user profile in DynamoDB
        try:
            user_profile = UserProfile.create_user(
                user_id=user_id,
                resume_link=None,
                questions=questions
            )
            user_profile.needs_matchmaking = "true"  # Enable scheduled matching
            user_profile.save()
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
                logger.info(f"Updated PostgreSQL onboarding_status='completed' for user {user_id}")
            else:
                logger.warning(f"Could not update PostgreSQL onboarding_status for user {user_id}")
        except Exception as e:
            # Log but don't fail - DynamoDB profile was created successfully
            logger.warning(f"Failed to update PostgreSQL onboarding_status: {e}")

        # Create user_summary for Discover page
        # Without this, users won't appear in search results
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
            summary_json = json.dumps(summary_data)
            summary_id = postgresql_adapter.create_user_summary(
                user_id=user_id,
                summary=summary_json,
                status='draft',
                urgency='ongoing'
            )
            if summary_id:
                logger.info(f"Created user_summary {summary_id} for user {user_id}")
            else:
                logger.warning(f"Could not create user_summary for user {user_id}")
        except Exception as e:
            # Log but don't fail - user can still use dashboard
            logger.warning(f"Failed to create user_summary: {e}")

        # Generate multi-vector embeddings for 6-dimension matching
        # This enables the weighted multi-dimensional matching algorithm
        try:
            from app.services.multi_vector_matcher import multi_vector_matcher

            # Build persona_data from slots in the format expected by multi_vector_matcher
            persona_data = {
                "primary_goal": slots.get("primary_goal", {}).get("value", ""),
                "objective": slots.get("user_type", {}).get("value", ""),
                "industry": slots.get("industry_focus", {}).get("value", ""),
                "stage": slots.get("stage_preference", {}).get("value", ""),
                "investment_stage": slots.get("investment_stage", {}).get("value", ""),
                "company_stage": slots.get("company_stage", {}).get("value", ""),
                "geography": slots.get("geographic_focus", {}).get("value", ""),
                "engagement_style": "",  # Not collected in current onboarding
                "dealbreakers": "",  # Not collected in current onboarding
                "requirements": slots.get("requirements", {}).get("value", ""),
                "offerings": slots.get("offerings", {}).get("value", ""),
            }

            # Store embeddings for what the user needs (requirements direction)
            req_results = multi_vector_matcher.store_multi_vector_embeddings(
                user_id=user_id,
                persona_data=persona_data,
                direction="requirements"
            )

            # Store embeddings for what the user offers (offerings direction)
            off_results = multi_vector_matcher.store_multi_vector_embeddings(
                user_id=user_id,
                persona_data=persona_data,
                direction="offerings"
            )

            stored_count = sum(1 for v in {**req_results, **off_results}.values() if v)
            logger.info(f"Stored {stored_count} multi-vector embeddings for user {user_id}")
        except Exception as e:
            # Log but don't fail - user can still use platform with basic matching
            logger.warning(f"Failed to generate multi-vector embeddings: {e}")

        # Generate basic 2-vector embeddings (requirements/offerings) for simple matching
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
                if basic_stored:
                    logger.info(f"Stored basic embeddings for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to generate basic embeddings: {e}")

        # Automatically sync matches to backend so they appear in frontend
        # This is the permanent fix - all future users get matches synced automatically
        try:
            from app.services.match_sync_service import match_sync_service
            sync_result = match_sync_service.sync_user_matches(user_id)
            if sync_result.get('success'):
                match_count = sync_result.get('count', 0)
                logger.info(f"Synced {match_count} matches to backend for user {user_id}")
            else:
                logger.warning(f"Match sync returned unsuccessful for {user_id}: {sync_result.get('error')}")
        except Exception as e:
            # Log but don't fail - user can still use platform, matches will sync later
            logger.warning(f"Failed to sync matches to backend: {e}")

        # Trigger persona generation pipeline
        try:
            # Check if resume was uploaded during session
            resume_data = None
            try:
                redis_client = _get_redis_client()
                resume_b64 = redis_client.get(f"resume:{session_id}")
                if resume_b64:
                    # Get metadata
                    resume_meta = redis_client.hgetall(f"resume_meta:{session_id}")
                    resume_data = {
                        "content": resume_b64,  # Already base64 encoded
                        "filename": resume_meta.get("filename", "resume.pdf"),
                        "content_type": resume_meta.get("content_type", "application/pdf")
                    }
                    logger.info(f"Found uploaded resume for session {session_id}: {resume_data['filename']}")
                    # Clean up Redis after retrieving
                    redis_client.delete(f"resume:{session_id}", f"resume_meta:{session_id}")
            except Exception as e:
                logger.warning(f"Could not retrieve resume from Redis: {e}")

            workflow = chain(
                process_resume_task.s(user_id, resume_data),  # Pass resume data if uploaded
                generate_persona_task.s()
            )
            result = workflow.apply_async()
            task_id = result.id
            logger.info(f"Persona pipeline started for user {user_id}, task_id: {task_id}")
        except Exception as e:
            logger.warning(f"Failed to start persona pipeline: {e}")
            task_id = None

        return CompleteOnboardingResponse(
            success=True,
            user_id=user_id,
            message=f"Profile created with {len(questions)} data points. Persona generation started.",
            profile_created=True,
            persona_task_id=task_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    slot_names: List[str]
) -> str:
    """
    Generate a contextual response based on extracted slots.

    IMPORTANT: Follows Indirect Elicitation principle.
    - Default mode is INDIRECT - follow user's natural thread
    - Never feel like a questionnaire
    - Only use direct questions when truly necessary
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
