"""
User-related FastAPI routes.
"""
import logging
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.schemas.user import (
    UserRegistrationRequest,
    UserRegistrationResponse,
    UserProfileResponse,
    ApproveSummaryRequest,
    FeedbackRequest,
    InitiateAIChatRequest,
    InitiateAIChatResponse
)
from app.schemas.common import ErrorResponse
from app.services.user_service import UserService
from app.services.feedback_service import FeedbackService
from app.workers.embedding_processing import generate_embeddings_task
from app.workers.ai_chat_processing import simulate_ai_chat_task
from app.workers.scheduled_matching import scheduled_matchmaking_task

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache()
def get_user_service() -> UserService:
    """Dependency injection for UserService with caching."""
    return UserService()


@lru_cache()
def get_feedback_service() -> FeedbackService:
    """Dependency injection for FeedbackService with caching."""
    return FeedbackService()


@router.post("/user/register")
async def register_user(
    request: UserRegistrationRequest,
    user_service: UserService = Depends(get_user_service)
):
    """Register a new user."""
    result = user_service.register_user(request)

    if result.success:
        return {
            "code": 200,
            "message": "success",
            "result": True
        }
    else:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": result.message,
                "result": False
            }
        )
@router.get("/user/run-scheduled-matchmaking")
async def run_scheduled_matchmaking():
    """Run scheduled matchmaking worker on demand (GET only)."""
    try:
        logger.info("Running scheduled matchmaking on demand")
        task_result = scheduled_matchmaking_task.apply_async()
        logger.info(f"Scheduled matchmaking task queued with task_id: {task_result.id}")
        return {
            "code": 200,
            "message": "success",
            "result": True
        }
    except Exception as e:
        logger.error(f"Error queuing scheduled matchmaking: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(e),
                "result": False
            }
        )

@router.get("/user/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
):
    """Get user profile by ID."""
    profile = user_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile


@router.post("/user/approve-summary")
async def approve_summary(request: ApproveSummaryRequest):
    """
    Trigger embeddings generation and matching for a user.
    
    This endpoint should be called after the user approves their persona summary.
    It triggers the embeddings generation task which will:
    1. Generate embeddings from the user's persona (requirements and offerings)
    2. Store embeddings in PostgreSQL
    3. Find matches and send notification to backend
    
    The task runs asynchronously in the background.
    """
    try:
        user_id = request.user_id
        logger.info(f"Triggering embeddings generation for user {user_id}")
        
        # Trigger the embeddings generation task asynchronously
        task_result = generate_embeddings_task.apply_async(args=[user_id])
        
        logger.info(f"Embeddings generation task queued for user {user_id} with task_id: {task_result.id}")
        
        return {
            "code": 200,
            "message": "success",
            "result": True
        }

    except Exception as e:
        logger.error(f"Error triggering embeddings generation for user {request.user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(e),
                "result": False
            }
        )


@router.post("/user/feedback")
async def submit_feedback(
    data: FeedbackRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Endpoint: Accepts feedback for a specific match/chat.
    Saves feedback and updates user's persona vector using feedback and match context.

    This endpoint allows users to provide feedback on matches or chats, which is used to
    refine their persona embeddings and improve future matching results.
    """
    try:
        result = feedback_service.process_feedback(data)

        if result["success"]:
            return {
                "code": 200,
                "message": "success",
                "result": True
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "message": result["message"],
                    "result": False
                }
            )
    except Exception as e:
        logger.error(f"Error processing feedback for user {data.user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(e),
                "result": False
            }
        )


@router.post("/user/initiate-ai-chat", response_model=InitiateAIChatResponse)
async def initiate_ai_chat(request: InitiateAIChatRequest):
    """
    Initiate AI-to-AI chat between two users.
    
    This endpoint triggers an asynchronous background task that:
    1. Simulates a full conversation between two user personas
    2. Stores the conversation in DynamoDB
    3. Sends a webhook notification to the backend with the results
    
    The endpoint immediately returns success, and the actual chat simulation
    happens in the background. The results are sent via webhook when ready.
    """
    try:
        logger.info(f"Initiating AI chat for match {request.match_id}")
        logger.info(f"Initiator: {request.initiator_id}, Responder: {request.responder_id}")
        
        # Trigger the AI chat simulation task asynchronously
        task_result = simulate_ai_chat_task.apply_async(
            args=[
                request.initiator_id,
                request.responder_id,
                request.match_id,
                request.template
            ]
        )
        
        logger.info(f"AI chat task queued for match {request.match_id} with task_id: {task_result.id}")
        
        return InitiateAIChatResponse(
            code=200,
            message="success",
            result=True
        )
        
    except Exception as e:
        logger.error(f"Error initiating AI chat for match {request.match_id}: {str(e)}")
        return InitiateAIChatResponse(
            code=500,
            message=str(e),
            result=False
        )


@router.post("/user/regenerate-persona")
async def regenerate_persona(
    request: ApproveSummaryRequest,  # Reuse - just needs user_id
    user_service: UserService = Depends(get_user_service)
):
    """
    Regenerate persona for an existing user.

    This endpoint:
    1. Retrieves the user's existing Q&A data from DynamoDB
    2. Regenerates the persona using the updated AI prompts
    3. Updates the user's persona in DynamoDB
    4. Triggers new embeddings generation

    Use this when:
    - User wants to refresh their AI summary
    - Code fixes require re-running persona generation
    - User has updated their onboarding answers
    """
    from app.workers.persona_processing import generate_persona_task
    from app.adapters.dynamodb import UserProfile

    try:
        user_id = request.user_id
        logger.info(f"Regenerating persona for user {user_id}")

        # Verify user exists
        try:
            user_profile = UserProfile.get(user_id)
        except UserProfile.DoesNotExist:
            raise HTTPException(status_code=404, detail="User not found")

        # Reset persona status to trigger regeneration
        user_profile.update(actions=[UserProfile.persona_status.set('pending')])

        # Trigger persona regeneration task
        task_result = generate_persona_task.apply_async(args=[user_id, True])

        logger.info(f"Persona regeneration task queued for user {user_id} with task_id: {task_result.id}")

        return {
            "code": 200,
            "message": "Persona regeneration started",
            "result": True,
            "task_id": task_result.id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating persona for user {request.user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(e),
                "result": False
            }
        )


@router.get("/admin/search-user")
async def search_user_by_email(email: str):
    """
    Debug endpoint to search for a user by email in the backend database.
    Returns the user ID if found.
    """
    import os
    import psycopg2

    try:
        conn = psycopg2.connect(
            os.getenv('RECIPROCITY_BACKEND_DB_URL',
                     'postgresql://postgres:postgres@localhost:5432/reciprocity_db')
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, onboarding_status, created_at FROM users WHERE email ILIKE %s",
            (f"%{email}%",)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        users = []
        for row in rows:
            users.append({
                "id": str(row[0]),
                "email": row[1],
                "onboarding_status": row[2],
                "created_at": str(row[3]) if row[3] else None
            })

        return {"code": 200, "message": "Search complete", "result": users, "count": len(users)}
    except Exception as e:
        logger.error(f"Error searching user: {e}")
        return {"code": 500, "message": str(e), "result": [], "count": 0}


@router.post("/admin/init-tables")
async def init_tables(request: dict):
    """
    Admin endpoint to create missing DynamoDB tables.
    Should be called once during initial deployment.
    """
    from app.adapters.dynamodb import UserProfile, UserMatches
    import os

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    results = {}

    # Create UserMatches table if it doesn't exist
    try:
        if not UserMatches.exists():
            UserMatches.create_table(
                read_capacity_units=5,
                write_capacity_units=5,
                wait=True,
                billing_mode='PAY_PER_REQUEST'
            )
            results['user_matches'] = 'created'
            logger.info("Created UserMatches table")
        else:
            results['user_matches'] = 'already exists'
    except Exception as e:
        results['user_matches'] = f'error: {str(e)}'
        logger.error(f"Error creating UserMatches table: {e}")

    # Check UserProfile table
    try:
        if UserProfile.exists():
            results['user_profiles'] = 'exists'
        else:
            results['user_profiles'] = 'missing'
    except Exception as e:
        results['user_profiles'] = f'error: {str(e)}'

    return {"code": 200, "message": "Table initialization complete", "result": results}


@router.post("/admin/import-profile")
async def import_profile(request: dict):
    """
    Admin endpoint to directly import a user profile with persona.
    Used for migrating data from LocalStack to production.

    Expects:
    {
        "user_id": "uuid",
        "profile": {"raw_questions": [...], ...},
        "persona": {"name": "...", "archetype": "...", ...},
        "persona_status": "completed"
    }
    """
    from app.adapters.dynamodb import UserProfile, PersonaData, ProfileData, QuestionAnswer, ResumeTextData
    from datetime import datetime
    import os

    # Simple API key check for admin endpoints
    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    try:
        user_id = request["user_id"]
        profile_data = request.get("profile", {})
        persona_data = request.get("persona", {})

        # Check if user already exists
        try:
            existing = UserProfile.get(user_id)
            logger.info(f"User {user_id} already exists, updating...")
        except UserProfile.DoesNotExist:
            existing = None

        now = datetime.utcnow()

        # Build profile
        profile = ProfileData(
            resume_link=profile_data.get("resume_link"),
            raw_questions=[],
            created_at=now,
            updated_at=now
        )
        for q in profile_data.get("raw_questions", []):
            profile.raw_questions.append(
                QuestionAnswer(prompt=q.get("prompt", ""), answer=q.get("answer", ""))
            )

        # Build persona
        persona = PersonaData(
            name=persona_data.get("name"),
            archetype=persona_data.get("archetype"),
            designation=persona_data.get("designation"),
            experience=persona_data.get("experience"),
            focus=persona_data.get("focus"),
            profile_essence=persona_data.get("profile_essence"),
            investment_philosophy=persona_data.get("investment_philosophy"),
            strategy=persona_data.get("strategy"),
            what_theyre_looking_for=persona_data.get("what_theyre_looking_for"),
            engagement_style=persona_data.get("engagement_style"),
            requirements=persona_data.get("requirements"),
            offerings=persona_data.get("offerings"),
            user_type=persona_data.get("user_type"),
            industry=persona_data.get("industry"),
            generated_at=now
        )

        if existing:
            # Update existing
            existing.profile = profile
            existing.persona = persona
            existing.persona_status = request.get("persona_status", "completed")
            existing.needs_matchmaking = "true"
            existing.save()
        else:
            # Create new
            user_profile = UserProfile(
                user_id=user_id,
                profile=profile,
                resume_text=ResumeTextData(text=None, extracted_at=None, extraction_method=None),
                persona=persona,
                processing_status="completed",
                persona_status=request.get("persona_status", "completed"),
                needs_matchmaking="true"
            )
            user_profile.save()

        logger.info(f"Imported profile for user {user_id}")
        return {"code": 200, "message": "Profile imported", "result": True, "user_id": user_id}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error importing profile: {e}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": str(e), "result": False}
        )


@router.get("/admin/user-diagnostics")
async def get_user_diagnostics(email: str):
    """
    Comprehensive diagnostic endpoint for user journey debugging.
    Returns complete status of: account, onboarding, persona, embeddings, matches.
    """
    import os
    import psycopg2
    from datetime import datetime
    from app.adapters.dynamodb import UserProfile, UserMatches

    diagnostics = {
        "email": email,
        "timestamp": datetime.utcnow().isoformat(),
        "account": {"status": "not_found", "details": None},
        "onboarding": {"status": "unknown", "details": None},
        "persona": {"status": "not_found", "details": None},
        "embeddings": {"status": "not_found", "count": 0, "details": None},
        "matches": {"status": "not_found", "count": 0, "details": None},
        "timeline": [],
        "issues": []
    }

    user_id = None

    # 1. Check backend PostgreSQL for account
    try:
        backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        if backend_db_url:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, email, first_name, last_name, onboarding_status,
                       is_email_verified, created_at, updated_at
                FROM users WHERE email ILIKE %s
            """, (email,))
            row = cursor.fetchone()

            if row:
                user_id = str(row[0])
                diagnostics["account"] = {
                    "status": "found",
                    "details": {
                        "user_id": user_id,
                        "email": row[1],
                        "name": f"{row[2] or ''} {row[3] or ''}".strip(),
                        "onboarding_status": row[4],
                        "email_verified": row[5],
                        "created_at": str(row[6]) if row[6] else None,
                        "updated_at": str(row[7]) if row[7] else None
                    }
                }
                diagnostics["onboarding"] = {
                    "status": row[4] or "unknown",
                    "details": {"backend_status": row[4]}
                }
                diagnostics["timeline"].append({
                    "event": "account_created",
                    "timestamp": str(row[6]) if row[6] else None
                })
            else:
                diagnostics["issues"].append("User not found in backend database")

            cursor.close()
            conn.close()
        else:
            diagnostics["issues"].append("RECIPROCITY_BACKEND_DB_URL not configured")
    except Exception as e:
        logger.error(f"Error checking backend DB: {e}")
        diagnostics["issues"].append(f"Backend DB error: {str(e)}")

    # 2. Check DynamoDB for profile and persona
    if user_id:
        try:
            profile = UserProfile.get(user_id)

            # Profile exists
            diagnostics["onboarding"]["details"] = diagnostics["onboarding"].get("details", {})
            diagnostics["onboarding"]["details"]["dynamodb_profile"] = True
            diagnostics["onboarding"]["details"]["processing_status"] = profile.processing_status
            diagnostics["onboarding"]["details"]["persona_status"] = profile.persona_status
            diagnostics["onboarding"]["details"]["needs_matchmaking"] = profile.needs_matchmaking

            # Count questions answered
            questions_count = len(profile.profile.raw_questions) if profile.profile and profile.profile.raw_questions else 0
            diagnostics["onboarding"]["details"]["questions_answered"] = questions_count

            # Check persona
            if profile.persona and profile.persona.name:
                diagnostics["persona"] = {
                    "status": profile.persona_status or "completed",
                    "details": {
                        "name": profile.persona.name,
                        "archetype": profile.persona.archetype,
                        "designation": profile.persona.designation,
                        "generated_at": str(profile.persona.generated_at) if profile.persona.generated_at else None,
                        "has_requirements": bool(profile.persona.requirements),
                        "has_offerings": bool(profile.persona.offerings)
                    }
                }
                if profile.persona.generated_at:
                    diagnostics["timeline"].append({
                        "event": "persona_generated",
                        "timestamp": str(profile.persona.generated_at)
                    })
            else:
                diagnostics["issues"].append("Persona not generated")

        except UserProfile.DoesNotExist:
            diagnostics["issues"].append("User profile not found in DynamoDB")
        except Exception as e:
            logger.error(f"Error checking DynamoDB: {e}")
            diagnostics["issues"].append(f"DynamoDB error: {str(e)}")

    # 3. Check pgvector for embeddings
    if user_id:
        try:
            ai_db_url = os.getenv('DATABASE_URL')
            if ai_db_url:
                conn = psycopg2.connect(ai_db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), MAX(created_at),
                           array_agg(DISTINCT content_type)
                    FROM user_embeddings WHERE user_id = %s
                """, (user_id,))
                row = cursor.fetchone()

                if row and row[0] > 0:
                    diagnostics["embeddings"] = {
                        "status": "found",
                        "count": row[0],
                        "details": {
                            "last_created": str(row[1]) if row[1] else None,
                            "content_types": row[2] if row[2] else []
                        }
                    }
                    if row[1]:
                        diagnostics["timeline"].append({
                            "event": "embeddings_created",
                            "timestamp": str(row[1])
                        })
                else:
                    diagnostics["issues"].append("No embeddings found - matching will fail")

                cursor.close()
                conn.close()
            else:
                diagnostics["issues"].append("DATABASE_URL not configured")
        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            diagnostics["issues"].append(f"Embeddings DB error: {str(e)}")

    # 4. Check matches in DynamoDB
    if user_id:
        try:
            matches = UserMatches.get(user_id)
            match_count = len(matches.matches) if matches.matches else 0
            diagnostics["matches"] = {
                "status": "found" if match_count > 0 else "empty",
                "count": match_count,
                "details": {
                    "stored_at": str(matches.created_at) if matches.created_at else None,
                    "algorithm": matches.algorithm if hasattr(matches, 'algorithm') else None
                }
            }
            if matches.created_at:
                diagnostics["timeline"].append({
                    "event": "matches_calculated",
                    "timestamp": str(matches.created_at)
                })
        except UserMatches.DoesNotExist:
            diagnostics["issues"].append("No matches stored yet")
        except Exception as e:
            logger.error(f"Error checking matches: {e}")
            diagnostics["issues"].append(f"Matches error: {str(e)}")

    # 5. Check backend matches table
    if user_id:
        try:
            backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
            if backend_db_url:
                conn = psycopg2.connect(backend_db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), MAX(created_at)
                    FROM matches WHERE user_id = %s
                """, (user_id,))
                row = cursor.fetchone()

                if row and row[0] > 0:
                    diagnostics["matches"]["details"] = diagnostics["matches"].get("details", {})
                    diagnostics["matches"]["details"]["backend_count"] = row[0]
                    diagnostics["matches"]["details"]["backend_last_sync"] = str(row[1]) if row[1] else None
                else:
                    if diagnostics["matches"]["count"] > 0:
                        diagnostics["issues"].append("Matches in DynamoDB but not synced to backend")

                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error checking backend matches: {e}")

    # Sort timeline
    diagnostics["timeline"] = sorted(
        [t for t in diagnostics["timeline"] if t.get("timestamp")],
        key=lambda x: x["timestamp"]
    )

    # Generate summary status
    if not user_id:
        diagnostics["summary"] = "USER_NOT_FOUND"
    elif diagnostics["embeddings"]["count"] == 0:
        diagnostics["summary"] = "EMBEDDINGS_MISSING"
    elif diagnostics["matches"]["count"] == 0:
        diagnostics["summary"] = "MATCHES_NOT_CALCULATED"
    elif len(diagnostics["issues"]) > 0:
        diagnostics["summary"] = "ISSUES_DETECTED"
    else:
        diagnostics["summary"] = "HEALTHY"

    return {"code": 200, "message": "Diagnostics complete", "result": diagnostics}


@router.post("/admin/regenerate-embeddings")
async def regenerate_embeddings(request: dict):
    """
    Trigger embedding regeneration for a user.
    Use after fixing the send_task() bug to backfill missing embeddings.
    """
    import os
    from app.workers.embedding_processing import generate_embeddings_task

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        # Trigger embedding generation
        logger.info(f"Manually triggering embedding generation for user: {user_id}")
        generate_embeddings_task.delay(user_id)
        return {"code": 200, "message": "Embedding generation triggered", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error triggering embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/regenerate-matches")
async def regenerate_matches(request: dict):
    """
    Trigger match recalculation for a user.
    """
    import os
    from app.services.match_sync_service import match_sync_service

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        logger.info(f"Manually triggering match calculation for user: {user_id}")
        result = match_sync_service.calculate_and_sync_matches(user_id)
        return {"code": 200, "message": "Match calculation complete", "result": result}
    except Exception as e:
        logger.error(f"Error calculating matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
