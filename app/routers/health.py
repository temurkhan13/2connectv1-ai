"""
Health check routes.
"""
from fastapi import APIRouter
from app.schemas.common import HealthResponse
from app.services.llm_slot_extractor import llm_slot_extractor
from app.adapters.postgresql import postgresql_adapter
import logging
import os
from datetime import datetime, timedelta

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint at /health."""
    return HealthResponse(
        success=True,
        data={"status": "healthy"},
        message="OK"
    )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check_v1():
    """Health check endpoint at /api/v1/health (for Render)."""
    return HealthResponse(
        success=True,
        data={"status": "healthy"},
        message="OK"
    )


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        success=True,
        data={"message": "Welcome to 2Connect AI API"},
        message="OK"
    )


@router.get("/verify")
async def verify_system():
    """
    Verification endpoint - checks all common issues at once.
    Use this BEFORE testing to ensure fixes are applied.
    """
    checks = {
        "llm_cleanup_function_exists": False,
        "llm_prompt_has_critical_rules": False,
        "postgresql_connection": False,
        "user_count": 0,
        "users_with_embeddings": 0,
    }
    issues = []

    # Check 1: LLM cleanup function exists
    try:
        if hasattr(llm_slot_extractor, '_clean_follow_up_question'):
            checks["llm_cleanup_function_exists"] = True
        else:
            issues.append("_clean_follow_up_question method missing - LLM may say 'The user is...'")
    except Exception as e:
        issues.append(f"LLM extractor check failed: {e}")

    # Check 2: PostgreSQL connection and user count
    try:
        stats = postgresql_adapter.get_embedding_stats()
        checks["postgresql_connection"] = True
        checks["user_count"] = stats.get("total_users", 0)
        checks["users_with_embeddings"] = stats.get("users_with_embeddings", 0)

        if checks["user_count"] == 0:
            issues.append("No users in database - onboarding may not be working")
        if checks["users_with_embeddings"] == 0:
            issues.append("No users with embeddings - matching will return empty")
    except Exception as e:
        issues.append(f"PostgreSQL check failed: {e}")

    # Check 3: Verify prompt has critical rules (check file content)
    try:
        import inspect
        source = inspect.getsource(llm_slot_extractor._build_system_prompt)
        if "CRITICAL: follow_up_question Rules" in source:
            checks["llm_prompt_has_critical_rules"] = True
        else:
            issues.append("LLM prompt missing 'CRITICAL: follow_up_question Rules' - may repeat user info")
    except Exception as e:
        issues.append(f"Prompt check failed: {e}")

    all_ok = len(issues) == 0

    return {
        "status": "ok" if all_ok else "issues_found",
        "checks": checks,
        "issues": issues,
        "recommendation": "All checks passed!" if all_ok else "Fix issues before testing"
    }


@router.get("/admin/system-health")
async def get_system_health():
    """
    Comprehensive system health endpoint for dashboard.
    Returns green/red status for all AI components.

    Categories:
    - ai_service: Core AI functionality (LLM, embeddings, Redis)
    - onboarding: Onboarding components (slot extraction, persona, conversational)
    - matching: Matching components (multi-vector, bidirectional, intent, temporal)
    """
    import psycopg2
    import redis
    from openai import OpenAI

    health = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "ai_service": {
            "status": "healthy",
            "components": {}
        },
        "onboarding": {
            "status": "healthy",
            "components": {}
        },
        "matching": {
            "status": "healthy",
            "components": {}
        }
    }

    issues = []

    # ===== AI SERVICE COMPONENTS =====

    # 1. OpenAI/LLM Connection
    try:
        client = OpenAI()
        # Quick test - just check we can create client
        health["ai_service"]["components"]["llm_openai"] = {
            "status": "healthy",
            "detail": "OpenAI client initialized"
        }
    except Exception as e:
        health["ai_service"]["components"]["llm_openai"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("LLM/OpenAI connection failed")

    # 2. PostgreSQL (AI DB with pgvector)
    try:
        ai_db_url = os.getenv('DATABASE_URL')
        if ai_db_url:
            conn = psycopg2.connect(ai_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_embeddings")
            embed_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            health["ai_service"]["components"]["postgresql_pgvector"] = {
                "status": "healthy",
                "detail": f"{embed_count} embeddings stored"
            }
        else:
            health["ai_service"]["components"]["postgresql_pgvector"] = {
                "status": "error",
                "detail": "DATABASE_URL not configured"
            }
            issues.append("PostgreSQL not configured")
    except Exception as e:
        health["ai_service"]["components"]["postgresql_pgvector"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("PostgreSQL connection failed")

    # 3. DynamoDB
    try:
        from app.adapters.dynamodb import UserProfile
        # Quick scan to verify connection
        count = 0
        for _ in UserProfile.scan(limit=1):
            count += 1
        health["ai_service"]["components"]["dynamodb"] = {
            "status": "healthy",
            "detail": "DynamoDB connection active"
        }
    except Exception as e:
        health["ai_service"]["components"]["dynamodb"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("DynamoDB connection failed")

    # 4. Redis (Celery broker)
    try:
        redis_url = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL', ''))
        if redis_url:
            r = redis.from_url(redis_url)
            r.ping()
            health["ai_service"]["components"]["redis_celery"] = {
                "status": "healthy",
                "detail": "Redis connection active"
            }
        else:
            health["ai_service"]["components"]["redis_celery"] = {
                "status": "warning",
                "detail": "REDIS_URL not configured (Celery may not work)"
            }
    except Exception as e:
        health["ai_service"]["components"]["redis_celery"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Redis connection failed")

    # 5. Backend PostgreSQL (for sync)
    try:
        backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        if backend_db_url:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            health["ai_service"]["components"]["backend_postgresql"] = {
                "status": "healthy",
                "detail": f"{user_count} users in backend"
            }
        else:
            health["ai_service"]["components"]["backend_postgresql"] = {
                "status": "warning",
                "detail": "RECIPROCITY_BACKEND_DB_URL not configured"
            }
    except Exception as e:
        health["ai_service"]["components"]["backend_postgresql"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Backend PostgreSQL connection failed")

    # ===== ONBOARDING COMPONENTS =====

    # 1. Slot Extractor (LLM-based)
    try:
        if hasattr(llm_slot_extractor, '_clean_follow_up_question'):
            health["onboarding"]["components"]["slot_extractor"] = {
                "status": "healthy",
                "detail": "LLM slot extractor initialized with cleanup"
            }
        else:
            health["onboarding"]["components"]["slot_extractor"] = {
                "status": "warning",
                "detail": "Cleanup function missing"
            }
    except Exception as e:
        health["onboarding"]["components"]["slot_extractor"] = {
            "status": "error",
            "detail": str(e)[:100]
        }

    # 2. Persona Generation
    try:
        from app.services.persona_service import persona_service
        health["onboarding"]["components"]["persona_generation"] = {
            "status": "healthy",
            "detail": "PersonaService available"
        }
    except Exception as e:
        health["onboarding"]["components"]["persona_generation"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Persona generation service failed")

    # 3. Conversational Onboarding (context manager)
    try:
        from app.services.context_manager import context_manager
        health["onboarding"]["components"]["conversational_onboarding"] = {
            "status": "healthy",
            "detail": "Context manager available"
        }
    except Exception as e:
        health["onboarding"]["components"]["conversational_onboarding"] = {
            "status": "error",
            "detail": str(e)[:100]
        }

    # 4. Progressive Disclosure
    try:
        from app.services.progressive_disclosure import ProgressiveDisclosure
        health["onboarding"]["components"]["progressive_disclosure"] = {
            "status": "healthy",
            "detail": "Progressive disclosure available"
        }
    except Exception as e:
        health["onboarding"]["components"]["progressive_disclosure"] = {
            "status": "error",
            "detail": str(e)[:100]
        }

    # 5. Resume Processing
    try:
        from app.services.resume_service import ResumeService
        health["onboarding"]["components"]["resume_processing"] = {
            "status": "healthy",
            "detail": "Resume service available"
        }
    except Exception as e:
        health["onboarding"]["components"]["resume_processing"] = {
            "status": "warning",
            "detail": str(e)[:100]
        }

    # ===== MATCHING COMPONENTS =====

    # 1. Embedding Service (basic)
    try:
        from app.services.embedding_service import embedding_service
        health["matching"]["components"]["embedding_service"] = {
            "status": "healthy",
            "detail": "Basic embedding service available"
        }
    except Exception as e:
        health["matching"]["components"]["embedding_service"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Embedding service failed")

    # 2. Multi-Vector Matcher
    try:
        from app.services.multi_vector_matcher import multi_vector_matcher
        health["matching"]["components"]["multi_vector_matcher"] = {
            "status": "healthy",
            "detail": "6-dimension matching available"
        }
    except Exception as e:
        health["matching"]["components"]["multi_vector_matcher"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Multi-vector matcher failed")

    # 3. Enhanced Matching (bidirectional + intent)
    try:
        from app.services.enhanced_matching_service import enhanced_matching_service
        health["matching"]["components"]["enhanced_matching"] = {
            "status": "healthy",
            "detail": "Bidirectional + intent classification"
        }
    except Exception as e:
        health["matching"]["components"]["enhanced_matching"] = {
            "status": "warning",
            "detail": str(e)[:100]
        }

    # 4. Inline Matching Service
    try:
        from app.services.inline_matching_service import inline_matching_service
        health["matching"]["components"]["inline_matching"] = {
            "status": "healthy",
            "detail": "Production matching service"
        }
    except Exception as e:
        health["matching"]["components"]["inline_matching"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Inline matching service failed")

    # 5. Match Sync Service
    try:
        from app.services.match_sync_service import match_sync_service
        health["matching"]["components"]["match_sync"] = {
            "status": "healthy",
            "detail": "Backend sync available"
        }
    except Exception as e:
        health["matching"]["components"]["match_sync"] = {
            "status": "error",
            "detail": str(e)[:100]
        }
        issues.append("Match sync service failed")

    # 6. AI Chat Service
    try:
        from app.services.ai_chat_service import AIChatService
        health["matching"]["components"]["ai_chat"] = {
            "status": "healthy",
            "detail": "AI-to-AI chat simulation available"
        }
    except Exception as e:
        health["matching"]["components"]["ai_chat"] = {
            "status": "warning",
            "detail": str(e)[:100]
        }

    # 7. Match Explanation Service
    try:
        from app.services.match_explanation_service import MatchExplanationService
        health["matching"]["components"]["match_explanation"] = {
            "status": "healthy",
            "detail": "Match explanations available"
        }
    except Exception as e:
        health["matching"]["components"]["match_explanation"] = {
            "status": "warning",
            "detail": str(e)[:100]
        }

    # Calculate overall status per category
    for category in ["ai_service", "onboarding", "matching"]:
        components = health[category]["components"]
        error_count = sum(1 for c in components.values() if c.get("status") == "error")
        warning_count = sum(1 for c in components.values() if c.get("status") == "warning")

        if error_count > 0:
            health[category]["status"] = "error"
        elif warning_count > 0:
            health[category]["status"] = "warning"
        else:
            health[category]["status"] = "healthy"

    # Calculate overall system status
    if any(health[c]["status"] == "error" for c in ["ai_service", "onboarding", "matching"]):
        health["overall_status"] = "error"
    elif any(health[c]["status"] == "warning" for c in ["ai_service", "onboarding", "matching"]):
        health["overall_status"] = "warning"

    health["issues"] = issues

    return health


@router.get("/admin/matching-diagnostics")
async def get_matching_diagnostics():
    """
    Comprehensive matching diagnostics endpoint for dashboard.
    Returns all users with their matching parameters and match details.

    For each user returns:
    - Basic info (id, email, name)
    - Embeddings (count, types, multi-vector coverage)
    - Intent classification
    - Dealbreakers
    - Matches with bidirectional scores
    - Activity boost / temporal weighting
    - Same objective blocking status
    """
    import psycopg2
    from app.adapters.dynamodb import UserProfile, UserMatches

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
            SELECT id, email, first_name, last_name, onboarding_status, created_at, updated_at
            FROM users
            WHERE onboarding_status = 'completed'
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Get all embeddings info in bulk
        embedding_info = {}
        multi_vector_dims = ['primary_goal', 'industry', 'stage', 'geography', 'engagement_style', 'dealbreakers']

        if ai_db_url:
            try:
                ai_conn = psycopg2.connect(ai_db_url)
                ai_cursor = ai_conn.cursor()
                ai_cursor.execute("""
                    SELECT user_id, embedding_type, created_at
                    FROM user_embeddings
                    ORDER BY user_id, embedding_type
                """)
                for row in ai_cursor.fetchall():
                    uid = str(row[0])
                    etype = row[1]
                    created = row[2]

                    if uid not in embedding_info:
                        embedding_info[uid] = {
                            "types": [],
                            "count": 0,
                            "multi_vector": {dim: {"req": False, "off": False} for dim in multi_vector_dims},
                            "last_created": None
                        }

                    embedding_info[uid]["types"].append(etype)
                    embedding_info[uid]["count"] += 1
                    embedding_info[uid]["last_created"] = str(created) if created else None

                    # Track multi-vector coverage
                    for dim in multi_vector_dims:
                        if etype == f"requirements_{dim}":
                            embedding_info[uid]["multi_vector"][dim]["req"] = True
                        elif etype == f"offerings_{dim}":
                            embedding_info[uid]["multi_vector"][dim]["off"] = True

                ai_cursor.close()
                ai_conn.close()
            except Exception as e:
                logger.error(f"Error fetching embeddings: {e}")

        # Get match counts and details from backend
        match_info = {}
        try:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    user_a_id, user_b_id,
                    user_a_decision, user_b_decision,
                    ai_remarks_after_chat, created_at
                FROM matches
                ORDER BY created_at DESC
            """)
            for row in cursor.fetchall():
                user_a = str(row[0])
                user_b = str(row[1])

                for uid, other_uid in [(user_a, user_b), (user_b, user_a)]:
                    if uid not in match_info:
                        match_info[uid] = {"count": 0, "matches": []}
                    match_info[uid]["count"] += 1
                    match_info[uid]["matches"].append({
                        "matched_with": other_uid,
                        "decision": row[2] if uid == user_a else row[3],
                        "ai_remarks": row[4][:100] if row[4] else None,
                        "created_at": str(row[5]) if row[5] else None
                    })

            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching matches: {e}")

        # Build user diagnostics
        for row in rows:
            user_id = str(row[0])

            # Get persona from DynamoDB for intent/dealbreakers
            persona_data = {}
            intent = "unknown"
            dealbreakers = []

            try:
                profile = UserProfile.get(user_id)
                if profile.persona:
                    p = profile.persona
                    persona_data = {
                        "archetype": getattr(p, 'archetype', None),
                        "user_type": getattr(p, 'user_type', None),
                        "focus": getattr(p, 'focus', None),
                        "requirements": getattr(p, 'requirements', None),
                        "offerings": getattr(p, 'offerings', None)
                    }

                    # Infer intent from archetype/user_type
                    archetype = (persona_data.get("archetype") or "").lower()
                    user_type = (persona_data.get("user_type") or "").lower()

                    if "investor" in archetype or "investor" in user_type:
                        intent = "INVESTOR_FOUNDER"
                    elif "founder" in archetype or "founder" in user_type:
                        intent = "FOUNDER_INVESTOR"
                    elif "mentor" in archetype or "advisor" in user_type:
                        intent = "MENTOR_MENTEE"
                    elif "cofounder" in archetype:
                        intent = "COFOUNDER"
                    else:
                        intent = "NETWORKING"

                    # Extract dealbreakers (if stored)
                    # TODO: Add dealbreakers field to persona if not present

            except Exception:
                pass

            # Get DynamoDB matches for score details
            dynamo_matches = []
            try:
                stored = UserMatches.get_user_matches(user_id)
                if stored:
                    req_matches = stored.get("requirements_matches", [])
                    for m in req_matches[:10]:  # Limit to top 10
                        dynamo_matches.append({
                            "user_id": m.get("user_id"),
                            "similarity_score": m.get("similarity_score"),
                            "match_type": m.get("match_type", "requirements"),
                            "forward_score": m.get("forward_score"),
                            "reverse_score": m.get("reverse_score")
                        })
            except Exception:
                pass

            # Calculate activity/temporal boost
            last_active = row[6]  # updated_at
            activity_boost = 1.0
            days_inactive = 0
            if last_active:
                days_inactive = (datetime.utcnow() - last_active.replace(tzinfo=None)).days
                if days_inactive <= 7:
                    activity_boost = 1.1  # +10% for active users
                elif days_inactive <= 30:
                    activity_boost = 1.0  # neutral
                else:
                    activity_boost = 0.85  # -15% decay

            # Get embedding info
            emb = embedding_info.get(user_id, {
                "types": [],
                "count": 0,
                "multi_vector": {dim: {"req": False, "off": False} for dim in multi_vector_dims},
                "last_created": None
            })

            # Calculate multi-vector coverage
            mv_complete = sum(1 for dim in multi_vector_dims
                            if emb["multi_vector"].get(dim, {}).get("req")
                            and emb["multi_vector"].get(dim, {}).get("off"))

            # Build user entry
            users.append({
                "user_id": user_id,
                "email": row[1],
                "name": f"{row[2] or ''} {row[3] or ''}".strip() or None,
                "onboarding_status": row[4],
                "created_at": str(row[5]) if row[5] else None,
                "last_active_at": str(row[6]) if row[6] else None,

                # Embeddings
                "embeddings": {
                    "count": emb["count"],
                    "types": emb["types"],
                    "has_basic": "requirements" in emb["types"] and "offerings" in emb["types"],
                    "multi_vector_complete": mv_complete,
                    "multi_vector_total": len(multi_vector_dims),
                    "multi_vector_details": emb["multi_vector"]
                },

                # Intent classification
                "intent_classification": {
                    "inferred_intent": intent,
                    "archetype": persona_data.get("archetype"),
                    "user_type": persona_data.get("user_type")
                },

                # Dealbreakers
                "dealbreakers": dealbreakers,

                # Temporal/activity weighting
                "temporal_weighting": {
                    "days_inactive": days_inactive,
                    "activity_boost": activity_boost,
                    "boost_reason": "Active (<7d)" if days_inactive <= 7 else
                                   "Neutral (7-30d)" if days_inactive <= 30 else
                                   "Decay (>30d)"
                },

                # Same objective blocking
                "same_objective_blocking": {
                    "enabled": True,
                    "intent": intent,
                    "would_block": [intent]  # Users with same intent would be blocked
                },

                # Matches
                "matches": {
                    "backend_count": match_info.get(user_id, {}).get("count", 0),
                    "backend_matches": match_info.get(user_id, {}).get("matches", [])[:5],
                    "dynamo_matches": dynamo_matches[:5],
                    "bidirectional_scores": dynamo_matches  # Same data, includes forward/reverse
                },

                # Persona summary
                "persona": {
                    "requirements_preview": (persona_data.get("requirements") or "")[:200],
                    "offerings_preview": (persona_data.get("offerings") or "")[:200],
                    "focus": persona_data.get("focus")
                }
            })

        return {
            "code": 200,
            "message": f"Found {len(users)} completed users",
            "result": users,
            "summary": {
                "total_users": len(users),
                "with_multi_vector": len([u for u in users if u["embeddings"]["multi_vector_complete"] == 6]),
                "with_basic_embeddings": len([u for u in users if u["embeddings"]["has_basic"]]),
                "with_matches": len([u for u in users if u["matches"]["backend_count"] > 0]),
                "intent_breakdown": {
                    intent: len([u for u in users if u["intent_classification"]["inferred_intent"] == intent])
                    for intent in ["INVESTOR_FOUNDER", "FOUNDER_INVESTOR", "MENTOR_MENTEE", "COFOUNDER", "NETWORKING", "unknown"]
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in matching diagnostics: {e}")
        return {"code": 500, "message": str(e), "result": []}
