"""
LLM-Scored Matching Service.

Architecture:
  1. Cosine pre-filter: user's embedding → top N candidates (fast, free)
  2. LLM judge: reads both profiles, scores each pair 0-100 (accurate, understands direction)
  3. Dealbreaker filter + rank by LLM score

No intent classification. No keyword rules. No penalty/boost system.
The LLM reads what each person offers and needs, and decides if they should meet.
"""
import json
import logging
import math
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.adapters.postgresql import postgresql_adapter
from app.adapters.supabase_profiles import UserProfile, UserMatches
from app.services.llm_fallback import call_with_fallback

logger = logging.getLogger(__name__)

# Configuration
COSINE_PRE_FILTER_LIMIT = int(os.getenv('LLM_MATCH_PREFILTER_LIMIT', '50'))
COSINE_THRESHOLD = float(os.getenv('LLM_MATCH_COSINE_THRESHOLD', '0.30'))
LLM_SCORE_MIN = int(os.getenv('LLM_MATCH_SCORE_MIN', '30'))
DEFAULT_MATCH_LIMIT = int(os.getenv('LLM_MATCH_LIMIT', '30'))
MAX_PARALLEL_LLM = int(os.getenv('LLM_MATCH_PARALLEL', '5'))

SCORING_PROMPT = """You are a professional networking match evaluator for a platform where people connect for business — investing, hiring, mentoring, partnerships, consulting, co-founding, and more.

Given two user profiles, evaluate whether introducing them would create mutual value.

USER A:
What they need: {user_a_requirements}

What they offer: {user_a_offerings}

USER B:
What they need: {user_b_requirements}

What they offer: {user_b_offerings}

Evaluate:
1. Does A offer something B specifically needs? Check their STATED requirements, not assumptions.
2. Does B offer something A specifically needs? Check their STATED requirements, not assumptions.
3. Do the specifics align? (industry, geography, stage, check size, role type, seniority)
4. Would BOTH sides see value in this introduction?

IMPORTANT: Score based on what each person needs and offers — both STATED and IMPLIED.
- Read their stated requirements first.
- Then look at their Role, Focus, and Background context (in square brackets) to understand implied needs.
  Example: A "Series B CEO scaling a payments company" implies they need engineers, product leaders, and board members — even if their stated need is "investors."
  Example: A "VP Engineering seeking a role" implies they need hiring companies and recruiters — not other job seekers.
- If User A said they want "to connect with other engineers for networking" — then matching with peers IS valid.
- General industry overlap alone is NOT a match. There must be a specific value exchange — stated or clearly implied by their role and stage.

Respond with ONLY a JSON object, nothing else — no explanation text before or after:
{{"score": <0-100>, "reason": "<one sentence>"}}

Scoring guide:
90-100: Both sides get exactly what they asked for, specifics align perfectly
70-89: Strong alignment, one side benefits more or minor specifics differ
50-69: Some value exists but significant gaps (wrong geography, wrong stage, wrong role)
30-49: Weak — only surface-level relevance, no specific value exchange
0-29: No meaningful connection — neither side gets what they specifically asked for"""


@dataclass
class LLMMatch:
    """A single match result from LLM-scored matching."""
    user_id: str
    cosine_score: float
    llm_score: int
    reason: str
    match_type: str = 'llm_scored'


def _score_pair(
    user_a_req: str, user_a_off: str,
    user_b_req: str, user_b_off: str,
) -> Dict[str, Any]:
    """Score a single pair using LLM. Returns {"score": int, "reason": str}."""
    try:
        prompt = SCORING_PROMPT.format(
            user_a_requirements=user_a_req[:2000],
            user_a_offerings=user_a_off[:2000],
            user_b_requirements=user_b_req[:2000],
            user_b_offerings=user_b_off[:2000],
        )

        response = call_with_fallback(
            service="matching",
            system_prompt="You are a match scoring system. Respond with ONLY a JSON object, no other text.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )

        # Parse JSON response — handle multiple LLM response formats
        import re
        text = response.strip()

        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        code_block = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if code_block:
            text = code_block.group(1)
        elif text.startswith("```"):
            # Fallback: split on backticks
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # Try direct JSON parse
        try:
            result = json.loads(text)
            score = int(result.get("score", 0))
            reason = str(result.get("reason", ""))
            return {"score": max(0, min(100, score)), "reason": reason}
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract JSON object from surrounding text (LLM sometimes adds explanation)
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', text)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                score = int(result.get("score", 0))
                reason = str(result.get("reason", ""))
                return {"score": max(0, min(100, score)), "reason": reason}
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: extract score and reason separately via regex
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if score_match:
            score = int(score_match.group(1))
            reason = reason_match.group(1) if reason_match else "parse fallback"
            return {"score": max(0, min(100, score)), "reason": reason}

        logger.warning(f"LLM returned unparseable response: {response[:200]}")
        return {"score": 0, "reason": "parse_error"}
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return {"score": 0, "reason": f"error: {str(e)[:50]}"}


def _get_profile_text(user_id: str) -> Optional[Dict[str, str]]:
    """Get full profile context for a user — not just requirements/offerings."""
    try:
        profile = UserProfile.get(user_id)
        if not profile or not profile.persona:
            return None
        p = profile.persona

        # Build a rich context string so the LLM understands WHO this person is,
        # not just what they stated they need. A "Series B CEO" implies hiring needs
        # even if their stated requirement is "investors."
        context_parts = []
        designation = getattr(p, 'designation', '') or ''
        if designation:
            context_parts.append(f"Role: {designation}")
        focus = getattr(p, 'focus', '') or ''
        if focus:
            context_parts.append(f"Focus: {focus}")
        essence = getattr(p, 'profile_essence', '') or ''
        if essence:
            context_parts.append(f"Background: {essence}")

        context = " | ".join(context_parts) if context_parts else ""

        requirements = getattr(p, 'requirements', '') or ''
        offerings = getattr(p, 'offerings', '') or ''

        # Prepend context to both so the LLM sees the full picture
        if context:
            requirements = f"[{context}]\n{requirements}" if requirements else context
            offerings = f"[{context}]\n{offerings}" if offerings else context

        return {
            "requirements": requirements,
            "offerings": offerings,
        }
    except Exception:
        return None


def find_matches(
    user_id: str,
    limit: int = DEFAULT_MATCH_LIMIT,
    cosine_limit: int = COSINE_PRE_FILTER_LIMIT,
) -> List[LLMMatch]:
    """
    Find matches using cosine pre-filter + LLM scoring.

    Step 1: Cosine finds top N candidates in similar space
    Step 2: LLM reads both profiles, scores each pair
    Step 3: Dealbreaker filter + rank by LLM score
    """
    try:
        # 1. Get user's embedding for cosine pre-filter
        user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not user_embeddings:
            logger.warning(f"[LLMMatch] No embeddings for {user_id}")
            return []

        req_data = user_embeddings.get('requirements')
        if not req_data:
            logger.warning(f"[LLMMatch] No requirements embedding for {user_id}")
            return []

        user_vec = req_data['vector_data']

        # 2. Cosine pre-filter: find candidates whose OFFERINGS match our REQUIREMENTS
        # BUG FIX: was searching 'requirements' (same-direction = people who need same things)
        # Correct: search 'offerings' (cross-direction = people who offer what we need)
        cosine_results = postgresql_adapter.find_similar_users(
            query_vector=user_vec,
            embedding_type='offerings',
            threshold=COSINE_THRESHOLD,
            exclude_user_id=user_id,
        )

        # Take top N by cosine
        cosine_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        candidates = cosine_results[:cosine_limit]

        logger.info(f"[LLMMatch] {user_id[:12]}...: {len(cosine_results)} cosine results, taking top {len(candidates)}")

        if not candidates:
            return []

        # 3. Get user's profile text
        user_profile = _get_profile_text(user_id)
        if not user_profile:
            logger.warning(f"[LLMMatch] No profile text for {user_id}")
            return []

        # 4. LLM score each candidate (parallelized)
        matches = []

        def score_candidate(candidate):
            cand_id = candidate['user_id']
            cosine_score = candidate['similarity_score']

            cand_profile = _get_profile_text(cand_id)
            if not cand_profile:
                return None

            result = _score_pair(
                user_a_req=user_profile['requirements'],
                user_a_off=user_profile['offerings'],
                user_b_req=cand_profile['requirements'],
                user_b_off=cand_profile['offerings'],
            )

            if result['score'] >= LLM_SCORE_MIN:
                return LLMMatch(
                    user_id=cand_id,
                    cosine_score=cosine_score,
                    llm_score=result['score'],
                    reason=result['reason'],
                )
            return None

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_LLM) as executor:
            futures = {executor.submit(score_candidate, c): c for c in candidates}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        matches.append(result)
                except Exception as e:
                    logger.error(f"[LLMMatch] Candidate scoring error: {e}")

        # 5. Dealbreaker filter
        matches = _apply_dealbreaker_filter(user_id, matches)

        # 6. Rank by LLM score
        matches.sort(key=lambda m: m.llm_score, reverse=True)
        result = matches[:limit]

        logger.info(f"[LLMMatch] {user_id[:12]}...: {len(result)} matches (from {len(candidates)} candidates)")
        return result

    except Exception as e:
        logger.error(f"[LLMMatch] Error: {e}", exc_info=True)
        return []


def find_and_store_matches(user_id: str, limit: int = DEFAULT_MATCH_LIMIT) -> Dict[str, Any]:
    """Find matches and store in match cache. Legacy-compatible interface."""
    matches = find_matches(user_id, limit=limit)

    legacy_matches = []
    for m in matches:
        legacy_matches.append({
            'user_id': m.user_id,
            'match_type': m.match_type,
            'cosine_score': m.cosine_score,
            'llm_score': m.llm_score,
            'reason': m.reason,
            'similarity_score': m.llm_score / 100.0,  # legacy compat (0-1 scale)
            'combined_score': m.llm_score / 100.0,
        })

    if not legacy_matches:
        return {
            'success': True, 'user_id': user_id,
            'total_matches': 0, 'requirements_matches': [],
            'offerings_matches': [], 'stored': True
        }

    try:
        stored = UserMatches.store_user_matches(user_id, {
            'requirements_matches': legacy_matches,
            'offerings_matches': [],
            'algorithm': 'llm_scored_v1',
            'total_matches': len(legacy_matches),
        })
    except Exception as e:
        logger.warning(f"[LLMMatch] Failed to store: {e}")
        stored = False

    return {
        'success': True, 'user_id': user_id,
        'total_matches': len(legacy_matches),
        'requirements_matches': legacy_matches,
        'offerings_matches': [], 'stored': stored,
    }


def _apply_dealbreaker_filter(user_id: str, matches: List[LLMMatch]) -> List[LLMMatch]:
    """Remove matches violating user's stated dealbreakers."""
    try:
        from app.adapters.supabase_onboarding import supabase_onboarding_adapter
        slots = supabase_onboarding_adapter.get_user_slots_sync(user_id)
        if not slots:
            return matches

        db_slot = slots.get('dealbreakers', {})
        db_text = db_slot.get('value', '') if isinstance(db_slot, dict) else str(db_slot)
        if not db_text or not db_text.strip():
            return matches

        dealbreakers = [d.strip().lower() for d in db_text.replace(';', ',').split(',') if d.strip()]
        if not dealbreakers:
            return matches

        filtered = []
        for m in matches:
            try:
                profile = UserProfile.get(m.user_id)
                if not profile or not profile.persona:
                    filtered.append(m)
                    continue
                p = profile.persona
                text = ' '.join([
                    getattr(p, 'archetype', '') or '',
                    getattr(p, 'designation', '') or '',
                    getattr(p, 'focus', '') or '',
                    getattr(p, 'offerings', '') or '',
                ]).lower()
                if not any(db in text for db in dealbreakers):
                    filtered.append(m)
            except Exception:
                filtered.append(m)
        return filtered
    except Exception:
        return matches
