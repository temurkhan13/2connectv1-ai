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
COSINE_THRESHOLD_PRIMARY = float(os.getenv('LLM_MATCH_COSINE_THRESHOLD', '0.30'))
COSINE_THRESHOLD_SECONDARY = float(os.getenv('LLM_MATCH_COSINE_SECONDARY', '0.40'))
COSINE_THRESHOLD_DIMENSION = float(os.getenv('LLM_MATCH_COSINE_DIMENSION', '0.50'))
CANDIDATE_HARD_CAP = int(os.getenv('LLM_MATCH_CANDIDATE_CAP', '60'))
LLM_SCORE_MIN = int(os.getenv('LLM_MATCH_SCORE_MIN', '30'))
DEFAULT_MATCH_LIMIT = int(os.getenv('LLM_MATCH_LIMIT', '30'))
MAX_PARALLEL_LLM = int(os.getenv('LLM_MATCH_PARALLEL', '5'))
MIN_DIMENSION_WORDS = 15  # Skip dimension embeddings with less than this many words of source text

SCORING_PROMPT = """You are a professional networking match evaluator. Score whether introducing two people would create real value based on what they SPECIFICALLY said they need.

USER A:
What they need: {user_a_requirements}
What they offer: {user_a_offerings}

USER B:
What they need: {user_b_requirements}
What they offer: {user_b_offerings}

SCORING PROCESS — follow these steps IN ORDER:

STEP 1: What does User A specifically need? Read their requirements literally.
STEP 2: Does User B OFFER that specific thing? Not something vaguely related — the actual thing.
STEP 3: What does User B specifically need? Read their requirements literally.
STEP 4: Does User A OFFER that specific thing?
STEP 5: Do the specifics align? (industry, geography, stage, check size, role level, etc.)

SCORE BANDS — your score MUST fall in the correct band:
90-100: BOTH users directly deliver what the other specifically asked for. Specifics align (stage, geography, industry, check size).
70-89:  ONE user clearly delivers what the other needs. The reverse direction has some but weaker value.
50-69:  Neither user directly delivers what the other asked for, but there is a plausible indirect connection (e.g., shared industry knowledge, potential introduction to someone else).
30-49:  Surface-level similarity only. Same industry or geography but no specific value exchange.
0-29:   No meaningful connection.

HARD RULES — violations mean automatic score cap:
1. If BOTH users are seeking the SAME thing (both raising capital, both seeking jobs, both looking for co-founders) and NEITHER offers what the other needs → score MUST be below 35. Peer networking is worth 25-35, not 60+.
2. If your reason contains words like "misaligned", "neither can fulfill", "both seeking", "no direct overlap" → your score MUST be below 45. Your reason and score must agree.
3. Do NOT invent relationships neither user asked for. If nobody mentioned mentorship, do not score based on "potential mentorship value." If nobody mentioned networking, do not score based on "peer networking."
4. Check size / stage mismatch: A $5K-$25K pre-seed investor matched with a Series A founder raising $2M+ → score below 40. The capital gap is too large.
5. Geography mismatch: If a user specified a required geography (e.g., "must be in Southeast Asia") and the match is elsewhere → score below 40.

SELF-CHECK before responding:
- Re-read what User A SPECIFICALLY said they need. Does User B offer EXACTLY that? If no → your score should be below 50.
- Re-read what User B SPECIFICALLY said they need. Does User A offer EXACTLY that? If no → your score should be below 70 (one-directional at best).
- Does your reason match your score? If your reason describes problems, your score must reflect those problems.

IMPLIED NEEDS (use carefully):
- A "Series B CEO scaling a company" implies hiring needs — matching with qualified engineers/executives IS valid.
- A "VP Engineering seeking a role" implies they need hiring companies — NOT other job seekers.
- Only use implied needs when the implication is obvious from role + stage. Do not stretch.

Respond with ONLY a JSON object:
{{"score": <0-100>, "reason": "<one sentence explaining the specific value exchange or why it's weak>"}}"""


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
) -> List[LLMMatch]:
    """
    Find matches using multi-direction cosine pre-filter + LLM scoring.

    Pre-filter (cosine — free, fast):
      Search 1: my requirements → their offerings (who offers what I need)
      Search 2: my offerings → their requirements (who needs what I offer)
      Future: industry→industry, stage→stage when dimension embeddings exist

    Cosine is FILTER ONLY — pass/fail gate. All candidates that pass go to LLM
    with equal standing. LLM decides the actual score.

    LLM scoring (Claude API — the only cost):
      Reads both profiles, scores 0-100, returns reason.
    """
    try:
        # 1. Get user's embeddings
        user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not user_embeddings:
            logger.warning(f"[LLMMatch] No embeddings for {user_id}")
            return []

        req_data = user_embeddings.get('requirements')
        off_data = user_embeddings.get('offerings')

        if not req_data:
            logger.warning(f"[LLMMatch] No requirements embedding for {user_id}")
            return []

        # 2. Multi-direction cosine pre-filter
        # Each search is a simple find_similar_users() call — no complex SQL
        # Cosine is boolean gate only: pass threshold = candidate, fail = skip
        candidate_map = {}  # user_id → best cosine score (for logging only)

        # Search 1 (PRIMARY): my requirements → their offerings
        # "Who offers what I need?" — always runs, lowest threshold
        results_req_off = postgresql_adapter.find_similar_users(
            query_vector=req_data['vector_data'],
            embedding_type='offerings',
            threshold=COSINE_THRESHOLD_PRIMARY,
            exclude_user_id=user_id,
        )
        for r in results_req_off:
            uid = r['user_id']
            candidate_map[uid] = max(candidate_map.get(uid, 0), r['similarity_score'])

        logger.info(f"[LLMMatch] {user_id[:12]}... Search 1 (req→off): {len(results_req_off)} candidates")

        # Search 2 (SECONDARY): my offerings → their requirements
        # "Who needs what I offer?" — higher threshold, catches reverse matches
        if off_data:
            results_off_req = postgresql_adapter.find_similar_users(
                query_vector=off_data['vector_data'],
                embedding_type='requirements',
                threshold=COSINE_THRESHOLD_SECONDARY,
                exclude_user_id=user_id,
            )
            for r in results_off_req:
                uid = r['user_id']
                candidate_map[uid] = max(candidate_map.get(uid, 0), r['similarity_score'])

            logger.info(f"[LLMMatch] {user_id[:12]}... Search 2 (off→req): {len(results_off_req)} candidates")

        # Dimension searches: industry, stage, geography — skip if user has no embedding for that dimension
        DIMENSION_SEARCHES = [
            ('focus_slot_industry_focus', 'focus_slot_industry_focus'),
            ('focus_slot_stage_preference', 'focus_slot_stage_preference'),
            ('focus_slot_geography', 'focus_slot_geography'),
        ]
        for user_emb_type, search_emb_type in DIMENSION_SEARCHES:
            dim_data = user_embeddings.get(user_emb_type)
            if not dim_data:
                continue  # User has no embedding for this dimension — skip, don't penalize
            # Check minimum text length (from metadata) — skip thin embeddings
            text_len = dim_data.get('metadata', {}).get('text_length', 0) if isinstance(dim_data.get('metadata'), dict) else 0
            if text_len > 0 and text_len < MIN_DIMENSION_WORDS * 5:  # ~15 words * ~5 chars/word
                continue
            try:
                dim_results = postgresql_adapter.find_similar_users(
                    query_vector=dim_data['vector_data'],
                    embedding_type=search_emb_type,
                    threshold=COSINE_THRESHOLD_DIMENSION,
                    exclude_user_id=user_id,
                )
                for r in dim_results:
                    uid = r['user_id']
                    candidate_map[uid] = max(candidate_map.get(uid, 0), r['similarity_score'])
                if dim_results:
                    logger.info(f"[LLMMatch] {user_id[:12]}... Dimension {user_emb_type}: {len(dim_results)} candidates")
            except Exception as dim_err:
                logger.debug(f"[LLMMatch] Dimension search {user_emb_type} failed: {dim_err}")

        # 3. Dedup + hard cap
        # Sort by best cosine score, take top CANDIDATE_HARD_CAP
        all_candidates = [
            {'user_id': uid, 'similarity_score': score}
            for uid, score in candidate_map.items()
        ]
        all_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        candidates = all_candidates[:CANDIDATE_HARD_CAP]

        logger.info(f"[LLMMatch] {user_id[:12]}... Total unique: {len(all_candidates)}, capped to: {len(candidates)}")

        if not candidates:
            return []

        # 4. Get user's profile text for LLM
        user_profile = _get_profile_text(user_id)
        if not user_profile:
            logger.warning(f"[LLMMatch] No profile text for {user_id}")
            return []

        # 5. LLM score each candidate (parallelized)
        # Cosine score is NOT passed to LLM — it was just the gate
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

        # 6. Dealbreaker filter
        matches = _apply_dealbreaker_filter(user_id, matches)

        # 7. Rank by LLM score
        matches.sort(key=lambda m: m.llm_score, reverse=True)
        result = matches[:limit]

        logger.info(f"[LLMMatch] {user_id[:12]}...: {len(result)} final matches (from {len(candidates)} candidates)")
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
