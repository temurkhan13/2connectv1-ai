"""
User-related FastAPI routes.
"""
import logging
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
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

            # Count slots filled (profile data points extracted from questions)
            slots_total = 11  # name, archetype, designation, experience, focus, profile_essence, strategy, seeking, style, requirements, offerings
            slots_filled = 0
            if profile.persona:
                if profile.persona.name: slots_filled += 1
                if profile.persona.archetype: slots_filled += 1
                if profile.persona.designation: slots_filled += 1
                if profile.persona.experience: slots_filled += 1
                if profile.persona.focus: slots_filled += 1
                if profile.persona.profile_essence: slots_filled += 1
                if profile.persona.investment_philosophy: slots_filled += 1
                if profile.persona.what_theyre_looking_for: slots_filled += 1
                if profile.persona.engagement_style: slots_filled += 1
                if profile.persona.requirements: slots_filled += 1
                if profile.persona.offerings: slots_filled += 1
            diagnostics["onboarding"]["details"]["slots_filled"] = slots_filled
            diagnostics["onboarding"]["details"]["slots_total"] = slots_total

            # Check persona - include all 11 slots
            if profile.persona and profile.persona.name:
                diagnostics["persona"] = {
                    "status": profile.persona_status or "completed",
                    "details": {
                        # Core slots (11 total)
                        "name": profile.persona.name,
                        "archetype": profile.persona.archetype,
                        "designation": profile.persona.designation,
                        "experience": profile.persona.experience,
                        "focus": profile.persona.focus,
                        "profile_essence": profile.persona.profile_essence,
                        "investment_philosophy": profile.persona.investment_philosophy,
                        "what_theyre_looking_for": profile.persona.what_theyre_looking_for,
                        "engagement_style": profile.persona.engagement_style,
                        "has_requirements": bool(profile.persona.requirements),
                        "has_offerings": bool(profile.persona.offerings),
                        # Metadata
                        "generated_at": str(profile.persona.generated_at) if profile.persona.generated_at else None
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
                # Get embedding count and types
                cursor.execute("""
                    SELECT COUNT(*), MAX(created_at), array_agg(DISTINCT embedding_type)
                    FROM user_embeddings WHERE user_id = %s
                """, (user_id,))
                row = cursor.fetchone()

                if row and row[0] > 0:
                    # Format embedding types nicely
                    embed_types = row[2] if row[2] else []
                    type_labels = {
                        'requirements': 'Requirements',
                        'offerings': 'Offerings',
                        'combined': 'Combined'
                    }
                    formatted_types = [type_labels.get(t, t) for t in embed_types if t]

                    diagnostics["embeddings"] = {
                        "status": "found",
                        "count": row[0],
                        "details": {
                            "last_created": str(row[1]) if row[1] else None,
                            "types": formatted_types
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
                    "stored_at": str(matches.last_updated) if matches.last_updated else None,
                    "algorithm": matches.algorithm if hasattr(matches, 'algorithm') else None
                }
            }
            if matches.last_updated:
                diagnostics["timeline"].append({
                    "event": "matches_calculated",
                    "timestamp": str(matches.last_updated)
                })
        except UserMatches.DoesNotExist:
            diagnostics["issues"].append("No matches stored yet")
        except Exception as e:
            logger.error(f"Error checking matches: {e}")
            diagnostics["issues"].append(f"Matches error: {str(e)}")

    # 5. Check backend matches table (uses user_a_id and user_b_id columns)
    if user_id:
        try:
            backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
            if backend_db_url:
                conn = psycopg2.connect(backend_db_url)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), MAX(created_at)
                    FROM matches WHERE user_a_id = %s OR user_b_id = %s
                """, (user_id, user_id))
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


@router.get("/admin/list-users")
async def list_all_users():
    """
    List all users with their onboarding/matching status.
    Returns summary status for each user without deep diagnostics.
    """
    import os
    import psycopg2
    from app.adapters.dynamodb import UserProfile

    users = []

    try:
        backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        ai_db_url = os.getenv('DATABASE_URL')

        if not backend_db_url:
            return {"code": 500, "message": "RECIPROCITY_BACKEND_DB_URL not configured", "result": []}

        # Get all users from backend
        conn = psycopg2.connect(backend_db_url)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, first_name, last_name, onboarding_status, created_at
            FROM users
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Get embedding counts and types in bulk
        embedding_counts = {}
        embedding_types = {}
        multi_vector_info = {}  # Track multi-vector embedding status

        # Multi-vector dimensions (6 total)
        MV_DIMENSIONS = ['primary_goal', 'industry', 'stage', 'geography', 'engagement_style', 'dealbreakers']

        if ai_db_url:
            try:
                ai_conn = psycopg2.connect(ai_db_url)
                ai_cursor = ai_conn.cursor()
                ai_cursor.execute("""
                    SELECT user_id, COUNT(*), array_agg(DISTINCT embedding_type)
                    FROM user_embeddings GROUP BY user_id
                """)
                for row in ai_cursor.fetchall():
                    uid = str(row[0])
                    embedding_counts[uid] = row[1]
                    # Format types nicely
                    types = row[2] if row[2] else []
                    type_labels = {'requirements': 'Requirements', 'offerings': 'Offerings', 'combined': 'Combined'}
                    embedding_types[uid] = [type_labels.get(t, t) for t in types if t]

                    # Calculate multi-vector coverage
                    mv_present = []
                    mv_missing = []
                    for dim in MV_DIMENSIONS:
                        # Check if both requirements and offerings exist for this dimension
                        has_req = f'requirements_{dim}' in types
                        has_off = f'offerings_{dim}' in types
                        if has_req and has_off:
                            mv_present.append(dim)
                        elif has_req or has_off:
                            mv_present.append(f'{dim}(partial)')
                        else:
                            mv_missing.append(dim)

                    multi_vector_info[uid] = {
                        'count': len([d for d in mv_present if '(partial)' not in d]),
                        'total': len(MV_DIMENSIONS),
                        'present': mv_present,
                        'missing': mv_missing,
                        'complete': len(mv_missing) == 0 and '(partial)' not in str(mv_present)
                    }
                ai_cursor.close()
                ai_conn.close()
            except Exception as e:
                logger.error(f"Error fetching embedding counts: {e}")

        # Get match counts from backend (matches table has user_a_id and user_b_id)
        match_counts = {}
        try:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()
            # Count matches where user appears as either user_a or user_b
            cursor.execute("""
                SELECT user_id, COUNT(*) FROM (
                    SELECT user_a_id as user_id FROM matches
                    UNION ALL
                    SELECT user_b_id as user_id FROM matches
                ) as all_matches
                GROUP BY user_id
            """)
            for row in cursor.fetchall():
                match_counts[str(row[0])] = row[1]
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching match counts: {e}")

        # Build user list with status
        for row in rows:
            user_id = str(row[0])
            embed_count = embedding_counts.get(user_id, 0)
            match_count = match_counts.get(user_id, 0)

            # Check persona and slots in DynamoDB
            persona_status = "unknown"
            slots_filled = 0
            filled_slots = []  # List of filled slot names
            questions_answered = 0
            persona_name = None
            try:
                profile = UserProfile.get(user_id)
                if profile.persona and profile.persona.name:
                    persona_status = "completed"
                    persona_name = profile.persona.name
                    # Count and track slots filled
                    if profile.persona.name:
                        slots_filled += 1
                        filled_slots.append("Name")
                    if profile.persona.archetype:
                        slots_filled += 1
                        filled_slots.append("Archetype")
                    if profile.persona.designation:
                        slots_filled += 1
                        filled_slots.append("Designation")
                    if profile.persona.experience:
                        slots_filled += 1
                        filled_slots.append("Experience")
                    if profile.persona.focus:
                        slots_filled += 1
                        filled_slots.append("Focus")
                    if profile.persona.profile_essence:
                        slots_filled += 1
                        filled_slots.append("Essence")
                    if profile.persona.investment_philosophy:
                        slots_filled += 1
                        filled_slots.append("Philosophy")
                    if profile.persona.what_theyre_looking_for:
                        slots_filled += 1
                        filled_slots.append("Seeking")
                    if profile.persona.engagement_style:
                        slots_filled += 1
                        filled_slots.append("Style")
                    if profile.persona.requirements:
                        slots_filled += 1
                        filled_slots.append("Requirements")
                    if profile.persona.offerings:
                        slots_filled += 1
                        filled_slots.append("Offerings")
                else:
                    persona_status = profile.persona_status or "pending"
                # Count questions answered
                if profile.profile and profile.profile.raw_questions:
                    questions_answered = len(profile.profile.raw_questions)
            except UserProfile.DoesNotExist:
                persona_status = "no_profile"
            except Exception:
                persona_status = "error"

            # Determine overall status
            onboarding_status = row[4] or "unknown"
            if onboarding_status != "completed":
                status = "onboarding_incomplete"
            elif persona_status != "completed":
                status = "persona_missing"
            elif embed_count == 0:
                status = "embeddings_missing"
            elif match_count == 0:
                status = "matches_missing"
            else:
                status = "healthy"

            # Get multi-vector info for this user
            mv_info = multi_vector_info.get(user_id, {
                'count': 0, 'total': 6, 'present': [], 'missing': MV_DIMENSIONS, 'complete': False
            })

            users.append({
                "user_id": user_id,
                "email": row[1],
                "name": f"{row[2] or ''} {row[3] or ''}".strip() or None,
                "onboarding_status": onboarding_status,
                "persona_status": persona_status,
                "persona_name": persona_name,
                "slots_filled": slots_filled,
                "slots_total": 11,
                "filled_slots": filled_slots,  # List of slot names
                "questions_answered": questions_answered,
                "embeddings_count": embed_count,
                "embedding_types": embedding_types.get(user_id, []),  # List of types
                "multi_vector": mv_info,  # Multi-vector embedding status
                "matches_count": match_count,
                "status": status,
                "created_at": str(row[5]) if row[5] else None
            })

        return {
            "code": 200,
            "message": f"Found {len(users)} users",
            "result": users,
            "summary": {
                "total": len(users),
                "healthy": len([u for u in users if u["status"] == "healthy"]),
                "embeddings_missing": len([u for u in users if u["status"] == "embeddings_missing"]),
                "matches_missing": len([u for u in users if u["status"] == "matches_missing"]),
                "onboarding_incomplete": len([u for u in users if u["status"] == "onboarding_incomplete"]),
                "multi_vector_complete": len([u for u in users if u.get("multi_vector", {}).get("complete", False)]),
                "multi_vector_partial": len([u for u in users if 0 < u.get("multi_vector", {}).get("count", 0) < 6]),
                "multi_vector_none": len([u for u in users if u.get("multi_vector", {}).get("count", 0) == 0])
            }
        }

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return {"code": 500, "message": str(e), "result": []}


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
    Trigger match recalculation for a user using the inline bidirectional matching service.

    This uses the same matching logic as onboarding completion:
    - Calculates matches for the user (threshold 0.5)
    - Updates reciprocal matches (other users' match lists)
    - Syncs to backend
    """
    import os
    from app.services.inline_matching_service import inline_matching_service

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    # Optional: allow custom threshold (default 0.5)
    threshold = request.get("threshold", 0.5)

    try:
        logger.info(f"Manually triggering bidirectional match calculation for user: {user_id} (threshold: {threshold})")
        result = inline_matching_service.calculate_and_sync_matches_bidirectional(user_id, threshold=threshold)
        return {"code": 200, "message": "Bidirectional match calculation complete", "result": result}
    except Exception as e:
        logger.error(f"Error calculating matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/regenerate-all-matches")
async def regenerate_all_matches(request: dict):
    """
    Batch recalculate matches for ALL completed users.

    This fixes the BUG-009 issue where everyone matched with everyone
    due to 0.0 threshold. Running this will:
    - Recalculate matches with proper threshold (0.5)
    - Update all reciprocal match lists
    - Sync all affected users to backend

    WARNING: This can take several minutes for large user bases.
    """
    import os
    from app.services.inline_matching_service import inline_matching_service
    from app.adapters.dynamodb import UserProfile

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    threshold = request.get("threshold", 0.5)
    dry_run = request.get("dry_run", False)

    try:
        # Get all users with completed persona (onboarding complete)
        completed_users = []
        for profile in UserProfile.scan():
            # persona_status == 'completed' means onboarding is done and persona generated
            if profile.persona_status == "completed":
                completed_users.append(profile.user_id)

        logger.info(f"Found {len(completed_users)} completed users for match recalculation")

        if dry_run:
            return {
                "code": 200,
                "message": "Dry run - no changes made",
                "users_to_process": len(completed_users),
                "threshold": threshold
            }

        results = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

        for user_id in completed_users:
            try:
                logger.info(f"Recalculating matches for user {user_id}")
                result = inline_matching_service.calculate_and_sync_matches_bidirectional(
                    user_id,
                    threshold=threshold
                )
                results["processed"] += 1
                if result.get("success"):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({"user_id": user_id, "error": result.get("errors", [])})
            except Exception as e:
                results["processed"] += 1
                results["failed"] += 1
                results["errors"].append({"user_id": user_id, "error": str(e)})
                logger.error(f"Error recalculating matches for {user_id}: {e}")

        return {
            "code": 200,
            "message": f"Batch match recalculation complete. {results['success']}/{results['processed']} succeeded.",
            "result": results
        }
    except Exception as e:
        logger.error(f"Error in batch match recalculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/regenerate-embeddings-sync")
async def regenerate_embeddings_sync(request: dict):
    """
    SYNCHRONOUSLY generate embeddings for a user.
    Bypasses Celery entirely - needed because CELERY_TASK_ALWAYS_EAGER
    doesn't work properly in Render's single-process environment.
    """
    import os
    from app.services.embedding_service import embedding_service
    from app.adapters.dynamodb import UserProfile, UserMatches, NotifiedMatchPairs

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        logger.info(f"[SYNC] Starting embedding generation for user: {user_id}")

        # Get user profile
        try:
            user_profile = UserProfile.get(user_id)
        except UserProfile.DoesNotExist:
            raise HTTPException(status_code=404, detail="User profile not found in DynamoDB")

        # Check persona status
        if user_profile.persona_status != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Persona not completed. Status: {user_profile.persona_status}"
            )

        # Get requirements and offerings
        requirements = user_profile.persona.requirements if user_profile.persona else None
        offerings = user_profile.persona.offerings if user_profile.persona else None

        if not requirements and not offerings:
            raise HTTPException(status_code=400, detail="No requirements or offerings found in persona")

        logger.info(f"[SYNC] Found requirements ({len(requirements) if requirements else 0} chars) and offerings ({len(offerings) if offerings else 0} chars)")

        # Clear old matches
        logger.info(f"[SYNC] Clearing old matches for user {user_id}")
        UserMatches.clear_user_matches(user_id)
        NotifiedMatchPairs.clear_user_pairs(user_id)

        # Generate embeddings
        logger.info(f"[SYNC] Generating embeddings for user {user_id}")
        success = embedding_service.store_user_embeddings(
            user_id=user_id,
            requirements=requirements or "",
            offerings=offerings or ""
        )

        if success:
            logger.info(f"[SYNC] Embeddings generated successfully for user {user_id}")

            # Also trigger matching
            from app.services.match_sync_service import match_sync_service
            logger.info(f"[SYNC] Triggering match calculation for user {user_id}")
            match_result = match_sync_service.sync_user_matches(user_id)

            return {
                "code": 200,
                "message": "Embeddings and matches generated successfully",
                "user_id": user_id,
                "embeddings": True,
                "matches": match_result
            }
        else:
            raise HTTPException(status_code=500, detail="Embedding generation failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SYNC] Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/generate-multi-vector-embeddings")
async def generate_multi_vector_embeddings(request: dict):
    """
    Generate multi-vector embeddings (6 dimensions) for a user.

    This creates the embeddings needed for HYBRID matching:
    - primary_goal (20% weight)
    - industry (25% weight)
    - stage (20% weight)
    - geography (15% weight)
    - engagement_style (10% weight)
    - dealbreakers (10% weight)

    Requires user to have persona data in DynamoDB.
    """
    import os
    from app.services.multi_vector_matcher import multi_vector_matcher
    from app.adapters.dynamodb import UserProfile

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        # Get user's persona from DynamoDB
        logger.info(f"[MULTI-VECTOR] Getting persona for user {user_id}")
        user_profile = UserProfile.get(user_id)

        if not user_profile or not user_profile.persona:
            raise HTTPException(status_code=404, detail=f"No persona found for user {user_id}")

        persona = user_profile.persona

        # Build persona_data dict with all relevant fields
        persona_data = {
            "primary_goal": getattr(persona, 'archetype', '') or getattr(persona, 'user_type', '') or '',
            "industry": getattr(persona, 'industry', '') or getattr(persona, 'focus', '') or '',
            "stage": getattr(persona, 'designation', '') or '',
            "geography": '',  # Often not collected
            "engagement_style": getattr(persona, 'engagement_style', '') or '',
            "dealbreakers": '',  # Often not collected
            "requirements": getattr(persona, 'requirements', '') or getattr(persona, 'what_theyre_looking_for', '') or '',
            "offerings": getattr(persona, 'offerings', '') or '',
        }

        # If primary_goal is empty, try to infer from profile_essence
        if not persona_data["primary_goal"]:
            profile_essence = getattr(persona, 'profile_essence', '') or ''
            if 'investor' in profile_essence.lower():
                persona_data["primary_goal"] = "investor"
            elif 'founder' in profile_essence.lower():
                persona_data["primary_goal"] = "founder"
            elif 'advisor' in profile_essence.lower():
                persona_data["primary_goal"] = "advisor"

        # If industry is empty, try to extract from profile_essence
        if not persona_data["industry"]:
            profile_essence = getattr(persona, 'profile_essence', '') or ''
            industries = ['technology', 'healthcare', 'fintech', 'saas', 'ai', 'biotech', 'retail', 'ecommerce']
            for ind in industries:
                if ind in profile_essence.lower():
                    persona_data["industry"] = ind
                    break

        logger.info(f"[MULTI-VECTOR] Building embeddings for user {user_id}")
        logger.info(f"[MULTI-VECTOR] Persona data: primary_goal={persona_data['primary_goal']}, industry={persona_data['industry']}")

        # Generate requirements embeddings (6 dimensions)
        req_results = multi_vector_matcher.store_multi_vector_embeddings(
            user_id=user_id,
            persona_data=persona_data,
            direction="requirements"
        )

        # Generate offerings embeddings (6 dimensions)
        off_results = multi_vector_matcher.store_multi_vector_embeddings(
            user_id=user_id,
            persona_data=persona_data,
            direction="offerings"
        )

        # Count embeddings generated
        req_count = sum(1 for v in req_results.values() if v) if req_results else 0
        off_count = sum(1 for v in off_results.values() if v) if off_results else 0
        total_count = req_count + off_count

        logger.info(f"[MULTI-VECTOR] Generated {total_count} embeddings for user {user_id} (req: {req_count}, off: {off_count})")

        return {
            "code": 200,
            "message": f"Generated {total_count} multi-vector embeddings",
            "user_id": user_id,
            "requirements_embeddings": req_count,
            "offerings_embeddings": off_count,
            "total_embeddings": total_count,
            "persona_data": {
                "primary_goal": persona_data["primary_goal"] or "(empty)",
                "industry": persona_data["industry"] or "(empty)",
                "stage": persona_data["stage"] or "(empty)"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MULTI-VECTOR] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/backfill-multi-vector-embeddings")
async def backfill_multi_vector_embeddings(request: dict, background_tasks: BackgroundTasks):
    """
    Backfill multi-vector embeddings for ALL users with personas.

    This runs in the background to avoid timeout.
    Returns immediately with a job ID to check status.
    """
    import os
    import uuid
    from app.adapters.dynamodb import UserProfile

    admin_key = os.getenv("ADMIN_API_KEY", "migrate-2connect-2026")
    if request.get("admin_key") != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Start background task
    async def run_backfill():
        from app.services.multi_vector_matcher import multi_vector_matcher

        logger.info(f"[BACKFILL {job_id}] Starting multi-vector embedding backfill")

        # Get all users from DynamoDB
        users = list(UserProfile.scan())
        logger.info(f"[BACKFILL {job_id}] Found {len(users)} users")

        stats = {"total": len(users), "processed": 0, "success": 0, "skipped": 0, "errors": []}

        for user in users:
            user_id = user.user_id

            try:
                if not user.persona:
                    stats["skipped"] += 1
                    continue

                persona = user.persona
                persona_data = {
                    "primary_goal": getattr(persona, 'archetype', '') or getattr(persona, 'user_type', '') or '',
                    "industry": getattr(persona, 'industry', '') or getattr(persona, 'focus', '') or '',
                    "stage": getattr(persona, 'designation', '') or '',
                    "geography": '',
                    "engagement_style": getattr(persona, 'engagement_style', '') or '',
                    "dealbreakers": '',
                    "requirements": getattr(persona, 'requirements', '') or getattr(persona, 'what_theyre_looking_for', '') or '',
                    "offerings": getattr(persona, 'offerings', '') or '',
                }

                # Generate embeddings
                multi_vector_matcher.store_multi_vector_embeddings(user_id, persona_data, "requirements")
                multi_vector_matcher.store_multi_vector_embeddings(user_id, persona_data, "offerings")

                stats["success"] += 1

            except Exception as e:
                stats["errors"].append({"user_id": user_id, "error": str(e)[:100]})

            stats["processed"] += 1

            if stats["processed"] % 10 == 0:
                logger.info(f"[BACKFILL {job_id}] Progress: {stats['processed']}/{stats['total']}")

        logger.info(f"[BACKFILL {job_id}] Complete: {stats['success']} success, {stats['skipped']} skipped, {len(stats['errors'])} errors")

    background_tasks.add_task(run_backfill)

    return {
        "code": 200,
        "message": "Backfill started in background",
        "job_id": job_id,
        "note": "Check logs for progress"
    }
