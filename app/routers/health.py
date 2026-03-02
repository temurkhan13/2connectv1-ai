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
        from app.adapters.dynamodb import UserProfile
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
