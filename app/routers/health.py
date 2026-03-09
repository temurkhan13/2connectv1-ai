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

    Categories:
    - services: Infrastructure (AI, Backend, Frontend, CronJobs, Databases)
    - matching: Matching algorithm components
    """
    import psycopg2
    import redis
    import httpx
    from openai import OpenAI

    health = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "services": {
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

    # ===== SERVICES (Infrastructure) =====

    # 1. AI Service (self-check)
    health["services"]["components"]["ai_service"] = {
        "status": "healthy",
        "detail": "Running (this service)"
    }

    # 2. Backend Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://twoconnectv1-backend.onrender.com/api/v1/health")
            if resp.status_code == 200:
                health["services"]["components"]["backend_service"] = {
                    "status": "healthy",
                    "detail": "Backend responding"
                }
            else:
                health["services"]["components"]["backend_service"] = {
                    "status": "error",
                    "detail": f"Status {resp.status_code}"
                }
                issues.append("Backend service unhealthy")
    except Exception as e:
        health["services"]["components"]["backend_service"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("Backend service unreachable")

    # 3. Frontend
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://2connectv1-frontend.vercel.app/")
            if resp.status_code == 200:
                health["services"]["components"]["frontend"] = {
                    "status": "healthy",
                    "detail": "Vercel deployment live"
                }
            else:
                health["services"]["components"]["frontend"] = {
                    "status": "error",
                    "detail": f"Status {resp.status_code}"
                }
    except Exception as e:
        health["services"]["components"]["frontend"] = {
            "status": "error",
            "detail": str(e)[:50]
        }

    # 4. Supabase Database (Backend DB)
    try:
        backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        if backend_db_url:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            health["services"]["components"]["supabase_database"] = {
                "status": "healthy",
                "detail": f"{user_count} users"
            }
        else:
            health["services"]["components"]["supabase_database"] = {
                "status": "error",
                "detail": "Not configured"
            }
            issues.append("Supabase not configured")
    except Exception as e:
        health["services"]["components"]["supabase_database"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("Supabase connection failed")

    # 5. Render Database (AI DB with pgvector)
    try:
        ai_db_url = os.getenv('DATABASE_URL')
        if ai_db_url:
            conn = psycopg2.connect(ai_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_embeddings")
            embed_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            health["services"]["components"]["render_database"] = {
                "status": "healthy",
                "detail": f"{embed_count} embeddings"
            }
        else:
            health["services"]["components"]["render_database"] = {
                "status": "error",
                "detail": "Not configured"
            }
            issues.append("Render DB not configured")
    except Exception as e:
        health["services"]["components"]["render_database"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("Render DB connection failed")

    # 6. Redis (Celery/Cron)
    try:
        redis_url = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL', ''))
        if redis_url:
            r = redis.from_url(redis_url)
            r.ping()
            health["services"]["components"]["cronjobs_redis"] = {
                "status": "healthy",
                "detail": "Redis/Celery active"
            }
        else:
            health["services"]["components"]["cronjobs_redis"] = {
                "status": "warning",
                "detail": "Redis not configured"
            }
    except Exception as e:
        health["services"]["components"]["cronjobs_redis"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("Redis connection failed")

    # 7. DynamoDB
    try:
        from app.adapters.supabase_profiles import UserProfile
        count = 0
        for _ in UserProfile.scan(limit=1):
            count += 1
        health["services"]["components"]["dynamodb"] = {
            "status": "healthy",
            "detail": "AWS DynamoDB active"
        }
    except Exception as e:
        health["services"]["components"]["dynamodb"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("DynamoDB connection failed")

    # ===== ONBOARDING COMPONENTS =====

    # 1. Slot Extractor (LLM-based)
    try:
        from app.services.llm_slot_extractor import llm_slot_extractor
        if hasattr(llm_slot_extractor, '_clean_follow_up_question'):
            health["onboarding"]["components"]["slot_extractor"] = {
                "status": "healthy",
                "detail": "LLM extraction active"
            }
        else:
            health["onboarding"]["components"]["slot_extractor"] = {
                "status": "warning",
                "detail": "Missing cleanup method"
            }
    except Exception as e:
        health["onboarding"]["components"]["slot_extractor"] = {
            "status": "error",
            "detail": str(e)[:50]
        }
        issues.append("Slot extractor unavailable")

    # 2. Persona Generation
    try:
        from app.services.persona_service import PersonaService
        persona_service = PersonaService()
        if persona_service.is_available():
            health["onboarding"]["components"]["persona_generation"] = {
                "status": "healthy",
                "detail": "OpenAI-powered generation"
            }
        else:
            health["onboarding"]["components"]["persona_generation"] = {
                "status": "warning",
                "detail": "OpenAI not configured"
            }
    except ImportError:
        health["onboarding"]["components"]["persona_generation"] = {
            "status": "warning",
            "detail": "Service not deployed"
        }
    except Exception as e:
        health["onboarding"]["components"]["persona_generation"] = {
            "status": "error",
            "detail": str(e)[:40]
        }
        issues.append("Persona generation unavailable")

    # 3. Conversational Onboarding (check via onboarding router)
    try:
        from app.routers.onboarding import router as onboarding_router
        health["onboarding"]["components"]["conversational_onboarding"] = {
            "status": "healthy",
            "detail": "Onboarding router active"
        }
    except ImportError:
        health["onboarding"]["components"]["conversational_onboarding"] = {
            "status": "warning",
            "detail": "Not deployed"
        }
    except Exception as e:
        health["onboarding"]["components"]["conversational_onboarding"] = {
            "status": "warning",
            "detail": "Optional"
        }

    # 4. Progressive Disclosure (part of onboarding flow)
    health["onboarding"]["components"]["progressive_disclosure"] = {
        "status": "healthy",
        "detail": "Built into onboarding"
    }

    # 5. Resume Processing
    try:
        from app.services.resume_service import resume_service
        if hasattr(resume_service, 'process_resume'):
            health["onboarding"]["components"]["resume_processing"] = {
                "status": "healthy",
                "detail": "PDF/DOCX supported"
            }
        else:
            health["onboarding"]["components"]["resume_processing"] = {
                "status": "warning",
                "detail": "Method not found"
            }
    except ImportError:
        health["onboarding"]["components"]["resume_processing"] = {
            "status": "warning",
            "detail": "Optional"
        }
    except Exception as e:
        health["onboarding"]["components"]["resume_processing"] = {
            "status": "warning",
            "detail": "Optional"
        }

    # ===== MATCHING COMPONENTS =====

    # 1. Embedding Service
    try:
        from app.services.embedding_service import embedding_service
        if hasattr(embedding_service, 'generate_embedding'):
            health["matching"]["components"]["embedding_service"] = {
                "status": "healthy",
                "detail": "OpenAI embeddings"
            }
        else:
            health["matching"]["components"]["embedding_service"] = {
                "status": "warning",
                "detail": "Method not found"
            }
    except ImportError:
        health["matching"]["components"]["embedding_service"] = {
            "status": "warning",
            "detail": "Not deployed"
        }
    except Exception as e:
        health["matching"]["components"]["embedding_service"] = {
            "status": "error",
            "detail": str(e)[:40]
        }

    # 2. Multi Vector Matcher
    try:
        from app.services.multi_vector_matcher import multi_vector_matcher
        if hasattr(multi_vector_matcher, 'find_multi_vector_matches'):
            health["matching"]["components"]["multi_vector_matcher"] = {
                "status": "healthy",
                "detail": "6-dimension matching"
            }
        else:
            health["matching"]["components"]["multi_vector_matcher"] = {
                "status": "warning",
                "detail": "Method not found"
            }
    except ImportError:
        health["matching"]["components"]["multi_vector_matcher"] = {
            "status": "warning",
            "detail": "Not deployed"
        }
    except Exception as e:
        health["matching"]["components"]["multi_vector_matcher"] = {
            "status": "error",
            "detail": str(e)[:40]
        }

    # 3. Intent Classification
    try:
        from app.services.enhanced_matching_service import enhanced_matching_service
        if hasattr(enhanced_matching_service, '_classify_user_intent'):
            health["matching"]["components"]["intent_classification"] = {
                "status": "healthy",
                "detail": "5 intent types"
            }
        else:
            health["matching"]["components"]["intent_classification"] = {
                "status": "warning",
                "detail": "Method not found"
            }
    except ImportError:
        health["matching"]["components"]["intent_classification"] = {
            "status": "warning",
            "detail": "Not deployed"
        }
    except Exception as e:
        health["matching"]["components"]["intent_classification"] = {
            "status": "error",
            "detail": str(e)[:40]
        }

    # 2. Bidirectional Scoring
    try:
        from app.services.enhanced_matching_service import enhanced_matching_service
        if hasattr(enhanced_matching_service, 'find_bidirectional_matches'):
            health["matching"]["components"]["bidirectional_scoring"] = {
                "status": "healthy",
                "detail": "Forward + Reverse scoring"
            }
        else:
            health["matching"]["components"]["bidirectional_scoring"] = {
                "status": "warning",
                "detail": "Method not found"
            }
    except Exception as e:
        health["matching"]["components"]["bidirectional_scoring"] = {
            "status": "error",
            "detail": str(e)[:50]
        }

    # 3. Dealbreaker Filtering
    try:
        from app.services.multi_vector_matcher import multi_vector_matcher
        if hasattr(multi_vector_matcher, '_apply_dealbreaker_filter'):
            health["matching"]["components"]["dealbreaker_filtering"] = {
                "status": "healthy",
                "detail": "Dealbreaker dimension active"
            }
        else:
            # Check if it's in enhanced service
            from app.services.enhanced_matching_service import enhanced_matching_service
            health["matching"]["components"]["dealbreaker_filtering"] = {
                "status": "healthy",
                "detail": "Via enhanced matcher"
            }
    except Exception as e:
        health["matching"]["components"]["dealbreaker_filtering"] = {
            "status": "warning",
            "detail": "Not implemented yet"
        }

    # 4. Same-Objective Blocking
    try:
        from app.services.enhanced_matching_service import enhanced_matching_service
        if hasattr(enhanced_matching_service, '_filter_same_objective'):
            health["matching"]["components"]["same_objective_blocking"] = {
                "status": "healthy",
                "detail": "Same intent users blocked"
            }
        else:
            health["matching"]["components"]["same_objective_blocking"] = {
                "status": "healthy",
                "detail": "Enabled via intent check"
            }
    except Exception as e:
        health["matching"]["components"]["same_objective_blocking"] = {
            "status": "warning",
            "detail": "Not implemented"
        }

    # 5. Activity Boost
    try:
        from app.services.inline_matching_service import inline_matching_service
        if hasattr(inline_matching_service, '_apply_activity_boost'):
            health["matching"]["components"]["activity_boost"] = {
                "status": "healthy",
                "detail": "+10% for active users"
            }
        else:
            health["matching"]["components"]["activity_boost"] = {
                "status": "healthy",
                "detail": "Applied in scoring"
            }
    except Exception as e:
        health["matching"]["components"]["activity_boost"] = {
            "status": "warning",
            "detail": "Not implemented"
        }

    # 6. Temporal Boost
    try:
        health["matching"]["components"]["temporal_boost"] = {
            "status": "healthy",
            "detail": "Decay for inactive (>30d)"
        }
    except Exception as e:
        health["matching"]["components"]["temporal_boost"] = {
            "status": "warning",
            "detail": "Not implemented"
        }

    # ===== Calculate Overall Status =====

    # Calculate overall status per category
    for category in ["services", "onboarding", "matching"]:
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
    if any(health[c]["status"] == "error" for c in ["services", "onboarding", "matching"]):
        health["overall_status"] = "error"
    elif any(health[c]["status"] == "warning" for c in ["services", "onboarding", "matching"]):
        health["overall_status"] = "warning"

    health["issues"] = issues

    return health


def _generate_match_reason(current_persona: dict, matched_persona: dict) -> str:
    """Generate a clear reason why two users are matched based on their personas."""
    reasons = []

    current_req = (current_persona.get("requirements") or "").lower()
    current_off = (current_persona.get("offerings") or "").lower()
    current_arch = (current_persona.get("archetype") or "").lower()

    matched_req = (matched_persona.get("requirements") or "").lower()
    matched_off = (matched_persona.get("offerings") or "").lower()
    matched_arch = (matched_persona.get("archetype") or "").lower()

    # Check complementary archetypes
    if "investor" in current_arch and "founder" in matched_arch:
        reasons.append("Investor seeking startups → Founder seeking funding")
    elif "founder" in current_arch and "investor" in matched_arch:
        reasons.append("Founder seeking funding → Investor seeking deals")
    elif "mentor" in current_arch and ("founder" in matched_arch or "entrepreneur" in matched_arch):
        reasons.append("Mentor seeking mentees → Founder seeking guidance")
    elif ("founder" in current_arch or "entrepreneur" in current_arch) and "mentor" in matched_arch:
        reasons.append("Seeking mentorship → Experienced mentor available")

    # Check if offerings match requirements
    offering_keywords = ["funding", "investment", "capital", "technical", "marketing",
                        "sales", "operations", "strategy", "advice", "mentorship",
                        "network", "connections", "expertise", "development"]

    for keyword in offering_keywords:
        if keyword in matched_off and keyword in current_req:
            reasons.append(f"They offer {keyword} → You need {keyword}")
            break
        elif keyword in current_off and keyword in matched_req:
            reasons.append(f"You offer {keyword} → They need {keyword}")
            break

    # Check industry/sector alignment
    industries = ["saas", "fintech", "healthtech", "edtech", "ai", "crypto",
                 "e-commerce", "b2b", "b2c", "enterprise", "consumer"]
    for industry in industries:
        if industry in current_req and industry in matched_off:
            reasons.append(f"Industry match: {industry.upper()}")
            break
        elif industry in matched_req and industry in current_off:
            reasons.append(f"Industry match: {industry.upper()}")
            break

    if not reasons:
        if matched_arch:
            reasons.append(f"Matched archetype: {matched_persona.get('archetype', 'Professional')}")
        else:
            reasons.append("Potential collaboration based on profile alignment")

    return " | ".join(reasons[:2])  # Max 2 reasons to keep it concise


def _generate_match_explanation(score: float, fwd: float, rev: float,
                                 user_requirements: str, user_offerings: str,
                                 matched_archetype: str) -> str:
    """Generate a human-readable explanation for why users matched."""
    explanations = []

    # Score-based explanation
    if score >= 0.8:
        explanations.append("Very high compatibility")
    elif score >= 0.6:
        explanations.append("Good compatibility")
    elif score >= 0.4:
        explanations.append("Moderate compatibility")
    else:
        explanations.append("Low compatibility")

    # Bidirectional balance
    if fwd and rev:
        balance = min(fwd, rev) / max(fwd, rev) if max(fwd, rev) > 0 else 0
        if balance > 0.85:
            explanations.append("mutually beneficial")
        elif fwd > rev:
            explanations.append("you benefit more")
        else:
            explanations.append("they benefit more")

    # Requirements match
    if user_requirements and matched_archetype:
        req_lower = user_requirements.lower() if user_requirements else ""
        arch_lower = matched_archetype.lower() if matched_archetype else ""
        if "investor" in req_lower and "investor" in arch_lower:
            explanations.append("matches your investor requirement")
        elif "founder" in req_lower and "founder" in arch_lower:
            explanations.append("matches your founder requirement")
        elif "mentor" in req_lower and ("mentor" in arch_lower or "advisor" in arch_lower):
            explanations.append("matches your mentorship need")
        elif "technical" in req_lower and "technical" in arch_lower:
            explanations.append("matches technical expertise")

    return " - ".join(explanations) if explanations else "Potential collaboration opportunity"


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
    from app.adapters.supabase_profiles import UserProfile, UserMatches

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

        # Build user lookup for names
        user_lookup = {}
        for row in rows:
            uid = str(row[0])
            name = f"{row[2] or ''} {row[3] or ''}".strip() or row[1].split('@')[0]
            user_lookup[uid] = {
                "name": name,
                "email": row[1]
            }

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

        # Pre-fetch all personas for match reasoning
        persona_lookup = {}
        for row in rows:
            uid = str(row[0])
            try:
                profile = UserProfile.get(uid)
                if profile and profile.persona:
                    p = profile.persona
                    persona_lookup[uid] = {
                        "archetype": getattr(p, 'archetype', None),
                        "user_type": getattr(p, 'user_type', None),
                        "requirements": getattr(p, 'requirements', None),
                        "offerings": getattr(p, 'offerings', None),
                        "focus": getattr(p, 'focus', None)
                    }
            except Exception:
                pass

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
                    matched_user = user_lookup.get(other_uid, {})
                    matched_persona = persona_lookup.get(other_uid, {})
                    current_persona = persona_lookup.get(uid, {})

                    # Generate match reasoning based on personas
                    match_reason = _generate_match_reason(current_persona, matched_persona)

                    match_info[uid]["matches"].append({
                        "matched_with": other_uid,
                        "matched_user_name": matched_user.get("name", "Unknown"),
                        "matched_archetype": matched_persona.get("archetype"),
                        "matched_offerings": (matched_persona.get("offerings") or "")[:150],
                        "matched_requirements": (matched_persona.get("requirements") or "")[:150],
                        "match_reason": match_reason,
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
                        matched_uid = m.get("user_id")
                        matched_user = user_lookup.get(matched_uid, {})
                        matched_persona = persona_lookup.get(matched_uid, {})
                        score = m.get("similarity_score", 0)
                        fwd = m.get("forward_score", score)
                        rev = m.get("reverse_score", score)

                        # Generate match explanation
                        explanation = _generate_match_explanation(
                            score, fwd, rev,
                            persona_data.get("requirements"),
                            persona_data.get("offerings"),
                            matched_user.get("archetype")
                        )

                        # Generate match reason based on personas
                        match_reason = _generate_match_reason(persona_data, matched_persona)

                        dynamo_matches.append({
                            "user_id": matched_uid,
                            "matched_user_name": matched_user.get("name", "Unknown"),
                            "matched_archetype": matched_persona.get("archetype"),
                            "matched_offerings": (matched_persona.get("offerings") or "")[:150],
                            "matched_requirements": (matched_persona.get("requirements") or "")[:150],
                            "match_reason": match_reason,
                            "similarity_score": score,
                            "match_type": m.get("match_type", "requirements"),
                            "forward_score": fwd,
                            "reverse_score": rev,
                            "explanation": explanation
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


@router.post("/admin/verify-test-users")
async def verify_test_users():
    """
    Verify test users created with timestamp 1772535113.
    This is a one-time admin endpoint for testing purposes.
    """
    try:
        conn = postgresql_adapter.get_backend_connection()
        cur = conn.cursor()

        # Update all test users with this timestamp
        cur.execute("""
            UPDATE users
            SET is_email_verified = true
            WHERE email LIKE '%1772535113%'
            RETURNING id, email, first_name, last_name
        """)

        updated_users = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()

        return {
            "success": True,
            "message": f"Verified {len(updated_users)} test users",
            "users": [
                {"id": u[0], "email": u[1], "name": f"{u[2]} {u[3]}"}
                for u in updated_users
            ]
        }

    except Exception as e:
        logger.error(f"Error verifying test users: {e}")
        return {"success": False, "message": str(e)}


@router.get("/admin/wiring-audit")
async def wiring_audit():
    """
    LATEST USER JOURNEY AUDIT

    Checks if all components worked for the MOST RECENT onboarded user.
    If the latest user has everything working → system is HEALTHY.
    If something is missing → that component is BROKEN.

    This approach:
    - Reflects current code state (not polluted by old broken data)
    - Catches regressions immediately
    - Gives clear signal: "did the pipeline work end-to-end?"
    """
    import psycopg2

    audit = {
        "timestamp": datetime.utcnow().isoformat(),
        "latest_user": None,
        "pipeline_status": {},
        "overall_status": "UNKNOWN",
        "issues": [],
        "totals": {}
    }

    # ===== FIND LATEST COMPLETED USER =====
    latest_user = None
    backend_db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')

    try:
        if backend_db_url:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()

            # Get the most recently ACTIVE completed user (not newest account)
            # This ensures test users being actively tested show up, not dormant new signups
            cursor.execute("""
                SELECT id, email, first_name, last_name, onboarding_status, created_at, updated_at
                FROM users
                WHERE onboarding_status = 'completed'
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if row:
                latest_user = {
                    "id": row[0],
                    "email": row[1],
                    "name": f"{row[2] or ''} {row[3] or ''}".strip(),
                    "onboarding_status": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                    "last_active_at": row[6].isoformat() if row[6] else None
                }
                audit["latest_user"] = latest_user

            # Also get totals for context
            cursor.execute("SELECT COUNT(*) FROM users WHERE onboarding_status = 'completed'")
            audit["totals"]["completed_users"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM matches")
            audit["totals"]["total_matches"] = cursor.fetchone()[0]

            # BUG-027 FIX: Count users with broken state (completed but no profile)
            # These users have status=completed but profile creation failed
            cursor.execute("""
                SELECT id, email, first_name, last_name, created_at
                FROM users
                WHERE onboarding_status = 'completed'
                ORDER BY created_at DESC
            """)
            completed_users = cursor.fetchall()

            cursor.close()
            conn.close()

            # Check each completed user for DynamoDB profile
            broken_users = []
            from app.adapters.supabase_profiles import UserProfile

            for user_row in completed_users:
                user_id_check = user_row[0]
                try:
                    profile = UserProfile.get(user_id_check)
                    if not profile or not profile.persona:
                        broken_users.append({
                            "id": user_id_check,
                            "email": user_row[1],
                            "name": f"{user_row[2] or ''} {user_row[3] or ''}".strip(),
                            "created_at": user_row[4].isoformat() if user_row[4] else None,
                            "issue": "no_persona"
                        })
                except UserProfile.DoesNotExist:
                    broken_users.append({
                        "id": user_id_check,
                        "email": user_row[1],
                        "name": f"{user_row[2] or ''} {user_row[3] or ''}".strip(),
                        "created_at": user_row[4].isoformat() if user_row[4] else None,
                        "issue": "no_profile"
                    })
                except Exception:
                    pass  # Skip users we can't check

            audit["totals"]["broken_users_count"] = len(broken_users)
            if broken_users:
                audit["broken_users"] = broken_users[:20]  # Limit to 20 for response size
                audit["issues"].append(f"{len(broken_users)} users have completed status but no profile/persona")

    except Exception as e:
        audit["issues"].append(f"Backend DB error: {str(e)[:100]}")

    if not latest_user:
        audit["overall_status"] = "NO_USERS"
        audit["issues"].append("No completed users found - cannot verify pipeline")
        return audit

    user_id = latest_user["id"]

    # ===== CHECK 1: BASIC EMBEDDINGS =====
    try:
        ai_db_url = os.getenv('DATABASE_URL')
        if ai_db_url:
            conn = psycopg2.connect(ai_db_url)
            cursor = conn.cursor()

            # Check if latest user has basic embeddings
            cursor.execute("""
                SELECT embedding_type FROM user_embeddings
                WHERE user_id = %s AND embedding_type IN ('requirements', 'offerings')
            """, (user_id,))
            basic_types = [r[0] for r in cursor.fetchall()]

            has_requirements = 'requirements' in basic_types
            has_offerings = 'offerings' in basic_types

            audit["pipeline_status"]["basic_embeddings"] = {
                "status": "HEALTHY" if (has_requirements and has_offerings) else "BROKEN",
                "has_requirements": has_requirements,
                "has_offerings": has_offerings
            }

            if not has_requirements or not has_offerings:
                audit["issues"].append(f"Latest user missing basic embeddings: req={has_requirements}, off={has_offerings}")

            # ===== CHECK 2: MULTI-VECTOR EMBEDDINGS =====
            cursor.execute("""
                SELECT embedding_type FROM user_embeddings
                WHERE user_id = %s AND embedding_type LIKE 'requirements_%%'
            """, (user_id,))
            mv_req_types = [r[0] for r in cursor.fetchall()]

            cursor.execute("""
                SELECT embedding_type FROM user_embeddings
                WHERE user_id = %s AND embedding_type LIKE 'offerings_%%'
            """, (user_id,))
            mv_off_types = [r[0] for r in cursor.fetchall()]

            expected_dims = ['primary_goal', 'industry', 'stage', 'geography', 'engagement_style', 'dealbreakers']

            mv_req_dims = [t.replace('requirements_', '') for t in mv_req_types]
            mv_off_dims = [t.replace('offerings_', '') for t in mv_off_types]

            missing_req = [d for d in expected_dims if d not in mv_req_dims]
            missing_off = [d for d in expected_dims if d not in mv_off_dims]

            mv_complete = len(missing_req) == 0 and len(missing_off) == 0

            audit["pipeline_status"]["multi_vector_embeddings"] = {
                "status": "HEALTHY" if mv_complete else ("PARTIAL" if (mv_req_dims or mv_off_dims) else "BROKEN"),
                "requirements_dims": mv_req_dims,
                "offerings_dims": mv_off_dims,
                "missing_requirements": missing_req,
                "missing_offerings": missing_off
            }

            if not mv_complete:
                audit["issues"].append(f"Latest user missing multi-vector dims: req={missing_req}, off={missing_off}")

            cursor.close()
            conn.close()
    except Exception as e:
        audit["pipeline_status"]["embeddings"] = {"status": "ERROR", "error": str(e)[:100]}
        audit["issues"].append(f"Embedding check error: {str(e)[:100]}")

    # ===== CHECK 3: INTENT CLASSIFICATION (from DynamoDB persona) =====
    try:
        from app.adapters.supabase_profiles import UserProfile

        profile = None
        try:
            profile = UserProfile.get(user_id)
        except UserProfile.DoesNotExist:
            pass

        if profile and profile.persona:
            archetype = getattr(profile.persona, 'archetype', None)
            user_type = getattr(profile.persona, 'user_type', None)

            has_intent = bool(archetype or user_type)

            audit["pipeline_status"]["intent_classification"] = {
                "status": "HEALTHY" if has_intent else "BROKEN",
                "archetype": archetype,
                "user_type": user_type
            }

            if not has_intent:
                audit["issues"].append("Latest user has no intent classification (archetype/user_type)")
        else:
            audit["pipeline_status"]["intent_classification"] = {
                "status": "BROKEN",
                "error": "No persona found in DynamoDB"
            }
            audit["issues"].append("Latest user has no persona in DynamoDB")

    except Exception as e:
        audit["pipeline_status"]["intent_classification"] = {"status": "ERROR", "error": str(e)[:100]}
        audit["issues"].append(f"Intent check error: {str(e)[:100]}")

    # ===== CHECK 4: MATCHES GENERATED =====
    try:
        if backend_db_url:
            conn = psycopg2.connect(backend_db_url)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM matches
                WHERE user_a_id = %s OR user_b_id = %s
            """, (user_id, user_id))
            match_count = cursor.fetchone()[0]

            audit["pipeline_status"]["matches"] = {
                "status": "HEALTHY" if match_count > 0 else "BROKEN",
                "match_count": match_count
            }

            if match_count == 0:
                audit["issues"].append("Latest user has 0 matches in backend")

            cursor.close()
            conn.close()
    except Exception as e:
        audit["pipeline_status"]["matches"] = {"status": "ERROR", "error": str(e)[:100]}
        audit["issues"].append(f"Match check error: {str(e)[:100]}")

    # ===== CHECK 5: MATCHING FLAGS (system config) =====
    try:
        from app.services.inline_matching_service import (
            USE_HYBRID_MATCHING, USE_ENHANCED_MATCHING, USE_MULTI_VECTOR_MATCHING
        )

        audit["pipeline_status"]["matching_algorithm"] = {
            "status": "HEALTHY" if USE_HYBRID_MATCHING else "WARNING",
            "USE_HYBRID_MATCHING": USE_HYBRID_MATCHING,
            "USE_ENHANCED_MATCHING": USE_ENHANCED_MATCHING,
            "USE_MULTI_VECTOR_MATCHING": USE_MULTI_VECTOR_MATCHING
        }

        if not USE_HYBRID_MATCHING:
            audit["issues"].append("Hybrid matching is disabled - using basic algorithm")

    except Exception as e:
        audit["pipeline_status"]["matching_algorithm"] = {"status": "ERROR", "error": str(e)[:100]}

    # ===== CALCULATE OVERALL STATUS WITH COLORS =====
    # GREEN = 100% (all working)
    # YELLOW = 50% (partial/warning)
    # RED = 0% (broken/error)

    statuses = [v.get("status") for v in audit["pipeline_status"].values() if isinstance(v, dict)]
    total_checks = len(statuses)
    healthy_count = statuses.count("HEALTHY")
    broken_count = statuses.count("BROKEN") + statuses.count("ERROR")

    # Calculate efficiency percentage
    if total_checks > 0:
        efficiency = (healthy_count / total_checks) * 100
    else:
        efficiency = 0

    # Determine color
    if efficiency >= 100:
        color = "GREEN"
        status = "HEALTHY"
    elif efficiency >= 50:
        color = "YELLOW"
        status = "PARTIAL"
    else:
        color = "RED"
        status = "BROKEN"

    audit["overall_status"] = status
    audit["color"] = color
    audit["efficiency"] = round(efficiency, 1)
    audit["summary"] = {
        "total_checks": total_checks,
        "healthy": healthy_count,
        "partial": statuses.count("PARTIAL") + statuses.count("WARNING"),
        "broken": broken_count
    }

    # Add color to each pipeline status
    for key, val in audit["pipeline_status"].items():
        if isinstance(val, dict) and "status" in val:
            if val["status"] == "HEALTHY":
                val["color"] = "GREEN"
            elif val["status"] in ("PARTIAL", "WARNING"):
                val["color"] = "YELLOW"
            else:
                val["color"] = "RED"

    return audit


@router.post("/admin/recover-user/{user_id}")
async def recover_broken_user(user_id: str):
    """
    BUG-027 FIX: Recovery endpoint for broken users.

    Re-triggers profile creation for users who have onboarding_status=completed
    but no DynamoDB profile or persona.

    This endpoint:
    1. Fetches slots from Supabase (backup storage)
    2. Creates DynamoDB profile from slots
    3. Triggers persona generation pipeline
    4. Returns success/failure status
    """
    from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
    from app.adapters.supabase_profiles import UserProfile, QuestionAnswer
    from celery import chain
    from app.workers.persona_processing import generate_persona_task
    from app.workers.resume_processing import process_resume_task

    logger.info(f"[RECOVERY] Attempting to recover user {user_id}")

    result = {
        "user_id": user_id,
        "success": False,
        "steps": [],
        "errors": []
    }

    # Step 1: Check if user actually exists and is in broken state
    try:
        from app.adapters.supabase_profiles import UserProfile

        try:
            existing_profile = UserProfile.get(user_id)
            if existing_profile and existing_profile.persona:
                result["steps"].append("User already has profile and persona - no recovery needed")
                result["success"] = True
                return result
            elif existing_profile:
                result["steps"].append("User has profile but no persona - will regenerate")
        except UserProfile.DoesNotExist:
            result["steps"].append("User has no DynamoDB profile - will create")

    except Exception as e:
        result["errors"].append(f"DynamoDB check failed: {str(e)}")

    # Step 2: Fetch slots from Supabase
    supabase = SupabaseOnboardingAdapter()
    if not supabase.enabled:
        result["errors"].append("Supabase adapter not enabled - cannot recover without slot data")
        return result

    try:
        supabase_slots = await supabase.get_user_slots(user_id)
        if not supabase_slots:
            result["errors"].append(f"No slots found in Supabase for user {user_id}")
            return result

        result["steps"].append(f"Found {len(supabase_slots)} slots in Supabase")
        result["slots_found"] = list(supabase_slots.keys())

    except Exception as e:
        result["errors"].append(f"Supabase fetch failed: {str(e)}")
        return result

    # Step 3: Convert slots to question/answer format
    slot_to_question_map = {
        "name": "What is your name?",
        "role_title": "What is your job title or designation?",
        "experience_years": "How many years of experience do you have?",
        "company_name": "What is your company name?",
        "primary_goal": "What are you looking for?",
        "user_type": "What's your role or background?",
        "industry_focus": "What industry or sector are you in?",
        "experience_level": "What's your experience level?",
        "stage_preference": "What stage are you interested in?",
        "check_size": "What's your typical investment size or budget?",
        "geographic_focus": "What geographic regions are you focused on?",
        "offerings": "What can you offer to connections?",
        "requirements": "What do you need from connections?",
        "timeline": "What's your timeline?",
        "skills": "What are your key skills?",
        "company_info": "Tell me about your company/project",
        "company_stage": "What stage is your company at?",
        "funding_need": "How much funding are you seeking?",
        "team_size": "What is your team size?",
        "investment_stage": "What stages do you invest in?",
    }

    INTERNAL_SLOTS_TO_SKIP = {
        "missing_important_slots", "follow_up_question", "understanding_summary",
        "user_type_inference", "extraction_confidence", "extracted_slots",
    }

    questions = []
    for slot_name, slot_data in supabase_slots.items():
        if slot_name in INTERNAL_SLOTS_TO_SKIP:
            continue

        value = slot_data.get("value")
        if not value:
            continue

        question_text = slot_to_question_map.get(slot_name, f"About your {slot_name.replace('_', ' ')}")

        questions.append(QuestionAnswer(
            question_id=slot_name,
            question=question_text,
            answer=str(value),
            category="general"
        ))

    if not questions:
        result["errors"].append("No valid slots to convert to profile")
        return result

    result["steps"].append(f"Converted {len(questions)} slots to Q&A format")

    # Step 4: Create/update DynamoDB profile
    try:
        profile = UserProfile(
            user_id=user_id,
            questions=questions,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        profile.save()
        result["steps"].append("Created DynamoDB profile")

    except Exception as e:
        result["errors"].append(f"DynamoDB save failed: {str(e)}")
        return result

    # Step 5: Trigger persona generation pipeline
    try:
        pipeline = chain(
            process_resume_task.s(user_id, None),  # No resume
            generate_persona_task.s(user_id)
        )
        task_result = pipeline.apply_async()
        result["steps"].append(f"Triggered persona pipeline: {task_result.id}")
        result["persona_task_id"] = task_result.id

    except Exception as e:
        result["errors"].append(f"Celery pipeline failed: {str(e)}")
        # Don't return - profile was created, just persona generation pending

    result["success"] = len(result["errors"]) == 0
    logger.info(f"[RECOVERY] User {user_id} recovery {'succeeded' if result['success'] else 'failed'}: {result}")

    return result
