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
from app.adapters.supabase_onboarding import supabase_onboarding_adapter
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

# Reciprocity hybrid (Apr-17): if a user has >= MIN_RECIPROCAL_MATCHES candidates
# with complementary primary_goal, we hard-filter to those (quality mode). If
# fewer, we fall back to including non-reciprocal candidates as backfill to
# avoid the empty-state trap for users in narrow goal pools (e.g. Seek
# Mentorship has only 2 potential reciprocals in the whole user base).
MIN_RECIPROCAL_MATCHES = int(os.getenv('LLM_MATCH_MIN_RECIPROCAL', '3'))

# Primary-goal reciprocity matrix. Each key maps to the set of partner goals
# that can exchange real value with that goal.
#   • Finance pairs: Raise Funding ↔ Invest in Startups
#   • Hiring pairs: Recruit ↔ Find New Job ↔ Hire Talent
#   • Mentorship pairs: Seek Mentorship ↔ Offer Mentorship
#   • Self-symmetric: Seek Networking, Explore Partnerships, Find Co-founder
#   • Service providers reciprocate with anyone whose goal implies a buyer
PRIMARY_GOAL_RECIPROCITY = {
    "Raise Funding":        {"Invest in Startups", "Offer Services"},
    "Find Co-founder":      {"Find Co-founder"},
    "Seek Mentorship":      {"Offer Mentorship"},
    "Offer Mentorship":     {"Seek Mentorship", "Find New Job"},
    "Explore Partnerships": {"Explore Partnerships", "Launch Product", "Offer Services"},
    "Invest in Startups":   {"Raise Funding", "Find Co-founder", "Launch Product"},
    "Offer Services":       {"Raise Funding", "Hire Talent", "Launch Product"},
    "Recruit":              {"Find New Job", "Hire Talent"},
    "Find New Job":         {"Recruit", "Hire Talent"},
    "Seek Networking":      {"Seek Networking"},
    "Hire Talent":          {"Find New Job", "Recruit", "Offer Services"},
    "Launch Product":       {"Explore Partnerships", "Invest in Startups", "Offer Services", "Raise Funding"},
}

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
{{"score": <0-100>, "reason": "<one sentence explaining the specific value exchange or why it's weak>", "breakdown": {{"role_fit": <0-100>, "stage_match": <0-100>, "geography_match": <0-100>, "industry_match": <0-100>}}}}

The breakdown dimensions (each 0-100, reflect per-dimension agreement independent of overall score):
- role_fit: Does the role type / person type align with what the other asked for?
- stage_match: Do the stages (seed/A/B, experience levels, hiring level) align?
- geography_match: Do the geographies / timezones / remote preferences align?
- industry_match: Does the industry / domain / sector align?"""


@dataclass
class LLMMatch:
    """A single match result from LLM-scored matching."""
    user_id: str
    cosine_score: float
    llm_score: int
    reason: str
    match_type: str = 'llm_scored'
    score_breakdown: Optional[Dict[str, int]] = None
    # Apr-17: primary_goal reciprocity flag. True when partner's primary_goal is
    # in the reciprocity matrix of the user's primary_goal (value-exchange
    # possible). False when partner got in as backfill under the hybrid filter.
    # None when reciprocity could not be evaluated (missing goal slots, unknown
    # enum value). Forward-compatible — downstream stores/ignores as needed.
    reciprocal: Optional[bool] = None


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
            breakdown = result.get("breakdown") if isinstance(result.get("breakdown"), dict) else None
            return {"score": max(0, min(100, score)), "reason": reason, "breakdown": breakdown}
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
        # Try multiple reason patterns: simple quotes, escaped quotes, single quotes
        reason = ""
        for pattern in [
            r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"',  # handles escaped quotes
            r'"reason"\s*:\s*"([^"]{1,500})',         # truncated at 500 chars
            r"'reason'\s*:\s*'([^']*)'",              # single quotes
        ]:
            reason_match = re.search(pattern, text, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1)
                break
        if score_match:
            score = int(score_match.group(1))
            return {"score": max(0, min(100, score)), "reason": reason or "no reason parsed", "breakdown": None}

        logger.warning(f"LLM returned unparseable response: {response[:200]}")
        return {"score": 0, "reason": "parse_error", "breakdown": None}
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return {"score": 0, "reason": f"error: {str(e)[:50]}", "breakdown": None}


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

        # 3.5 CHEAP FILTER: Remove candidates whose user_type doesn't match what this user seeks
        # Uses seeking_user_types slot (extracted during onboarding) to filter before LLM scoring
        try:
            user_slots = supabase_onboarding_adapter.get_user_slots_sync(user_id)
            seeking_raw = user_slots.get('seeking_user_types', {}).get('value', '')
            if seeking_raw:
                # ISSUE-7 FIX: Handle three storage formats defensively
                #   1. JSON list: '["Recruiter"]'   (preferred)
                #   2. Python list repr: "['Recruiter']" (legacy — from str(list))
                #   3. Semicolon-separated: "Recruiter; Founder"
                import json as _json
                import ast as _ast
                seeking_types = None
                if isinstance(seeking_raw, list):
                    seeking_types = seeking_raw
                elif isinstance(seeking_raw, str) and seeking_raw.startswith('['):
                    try:
                        seeking_types = _json.loads(seeking_raw)
                    except (ValueError, _json.JSONDecodeError):
                        try:
                            seeking_types = _ast.literal_eval(seeking_raw)
                        except (ValueError, SyntaxError):
                            seeking_types = None
                if not seeking_types:
                    seeking_types = [s.strip() for s in str(seeking_raw).split(';') if s.strip()]

                # Normalize to lowercase for comparison
                seeking_lower = {s.lower().strip() for s in seeking_types if s}

                if seeking_lower:
                    pre_filter_count = len(candidates)
                    filtered = []
                    for cand in candidates:
                        cand_slots = supabase_onboarding_adapter.get_user_slots_sync(cand['user_id'])
                        cand_type_raw = cand_slots.get('user_type', {}).get('value', '')
                        cand_type = cand_type_raw.lower().strip() if cand_type_raw else ''

                        # No user_type = exclude from results
                        # All users should have user_type via onboarding or backfill
                        if not cand_type:
                            continue

                        # Check if candidate's user_type matches any of the seeking types
                        match_found = False
                        for seek in seeking_lower:
                            if seek in cand_type or cand_type in seek:
                                match_found = True
                                break
                            # Also check individual parts: "Founder/Entrepreneur" → check "founder" and "entrepreneur"
                            for part in cand_type.split('/'):
                                part = part.strip()
                                if not part:
                                    continue
                                if seek in part or part in seek:
                                    match_found = True
                                    break
                            if match_found:
                                break

                        if match_found:
                            filtered.append(cand)

                    candidates = filtered
                    removed = pre_filter_count - len(candidates)
                    logger.info(f"[LLMMatch] {user_id[:12]}... Cheap filter: {pre_filter_count} → {len(candidates)} ({removed} removed, seeking: {seeking_types})")
        except Exception as filter_err:
            logger.warning(f"[LLMMatch] Cheap filter failed, skipping: {filter_err}")

        # 3.6 RECIPROCITY HYBRID (Apr-17): split candidates by primary_goal
        # reciprocity, then either hard-filter (if enough reciprocal partners
        # exist) or backfill with non-reciprocal candidates (fallback for
        # narrow-goal users). Each candidate carries a `reciprocal: bool` flag
        # forward so downstream consumers can tier the display later.
        try:
            user_goal_raw = user_slots.get('primary_goal', {}).get('value', '')
            user_goal = str(user_goal_raw).strip() if user_goal_raw else ''
            reciprocal_goals = PRIMARY_GOAL_RECIPROCITY.get(user_goal, set())

            if reciprocal_goals and candidates:
                reciprocal_cands = []
                adjacent_cands = []
                for cand in candidates:
                    try:
                        partner_slots = supabase_onboarding_adapter.get_user_slots_sync(cand['user_id'])
                        partner_goal_raw = partner_slots.get('primary_goal', {}).get('value', '')
                        partner_goal = str(partner_goal_raw).strip() if partner_goal_raw else ''
                    except Exception:
                        partner_goal = ''
                    if partner_goal and partner_goal in reciprocal_goals:
                        cand['reciprocal'] = True
                        reciprocal_cands.append(cand)
                    else:
                        cand['reciprocal'] = False
                        adjacent_cands.append(cand)

                pre_recip = len(candidates)
                if len(reciprocal_cands) >= MIN_RECIPROCAL_MATCHES:
                    # Quality mode: hard filter to reciprocal only
                    candidates = reciprocal_cands
                    logger.info(
                        f"[LLMMatch] {user_id[:12]}... Reciprocity HARD: "
                        f"{pre_recip} → {len(candidates)} reciprocal "
                        f"(user_goal={user_goal!r}, min={MIN_RECIPROCAL_MATCHES})"
                    )
                else:
                    # Fallback mode: keep reciprocal + top-cosine adjacent
                    adjacent_cands.sort(key=lambda c: c.get('similarity_score', 0), reverse=True)
                    budget = pre_recip - len(reciprocal_cands)
                    candidates = reciprocal_cands + adjacent_cands[:budget]
                    logger.info(
                        f"[LLMMatch] {user_id[:12]}... Reciprocity SOFT (fallback): "
                        f"{len(reciprocal_cands)} reciprocal + {min(budget, len(adjacent_cands))} adjacent "
                        f"(user_goal={user_goal!r}, threshold={MIN_RECIPROCAL_MATCHES})"
                    )
            elif user_goal:
                # Unknown user_goal (junk value not in enum) — skip reciprocity,
                # tag all as None so downstream knows we couldn't evaluate.
                for cand in candidates:
                    cand['reciprocal'] = None
                logger.info(
                    f"[LLMMatch] {user_id[:12]}... Reciprocity SKIPPED: "
                    f"user_goal={user_goal!r} not in PRIMARY_GOAL_RECIPROCITY matrix"
                )
            else:
                for cand in candidates:
                    cand['reciprocal'] = None
                logger.info(f"[LLMMatch] {user_id[:12]}... Reciprocity SKIPPED: no user primary_goal")
        except Exception as recip_err:
            logger.warning(f"[LLMMatch] Reciprocity hybrid failed, skipping: {recip_err}")

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
                    score_breakdown=result.get('breakdown'),
                    # Apr-17: forward the reciprocity flag set by the hybrid
                    # filter into the stored match so downstream UI can tier
                    # Primary vs Adjacent matches.
                    reciprocal=candidate.get('reciprocal'),
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


def _generate_match_explanations(user_id: str, matches: list) -> list:
    """Generate LLM explanations for each match pair using full persona data.

    Called once after matching. Explanations are stored with the match
    so the user sees them instantly without a live LLM call.
    """
    from app.routers.match import _get_user_persona
    from app.services.llm_service import get_llm_service
    import asyncio

    llm_service = get_llm_service()
    user_persona = _get_user_persona(user_id)

    for match in matches:
        try:
            match_user_id = match['user_id']
            match_persona = _get_user_persona(match_user_id)

            # Build scores for the prompt
            scores = {
                'req_to_off': match.get('similarity_score', 0.5),
                'off_to_req': match.get('similarity_score', 0.5),
                'industry_match': 0.7 if user_persona.get('industry', '').lower() == match_persona.get('industry', '').lower() else 0.4,
                'overall_score': match.get('similarity_score', 0.5),
            }

            # Run async LLM call synchronously (we're in a background worker)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run,
                            llm_service.generate_match_explanation(user_persona, match_persona, scores)
                        ).result(timeout=30)
                else:
                    result = loop.run_until_complete(
                        llm_service.generate_match_explanation(user_persona, match_persona, scores)
                    )
            except RuntimeError:
                result = asyncio.run(
                    llm_service.generate_match_explanation(user_persona, match_persona, scores)
                )

            match['headline'] = result.get('headline', '')
            match['key_points'] = result.get('key_points', [])
            match['explanation'] = result.get('summary', '')
            match['synergy_areas'] = result.get('synergy_areas', [])
            match['friction_points'] = result.get('friction_points', [])
            match['talking_points'] = result.get('talking_points', [])

            logger.info(f"[LLMMatch] Generated explanation for {user_id[:8]}↔{match_user_id[:8]}")

        except Exception as e:
            logger.warning(f"[LLMMatch] Failed to generate explanation for match {match.get('user_id', '?')}: {e}")
            match['headline'] = ''
            match['key_points'] = []
            match['explanation'] = match.get('reason', '')
            match['synergy_areas'] = [match.get('reason', '')] if match.get('reason') else []
            match['friction_points'] = []
            match['talking_points'] = []

    return matches


def find_and_store_matches(user_id: str, limit: int = DEFAULT_MATCH_LIMIT) -> Dict[str, Any]:
    """Find matches, store immediately (no explanations), then backfill explanations async.

    Split into two phases for fast match display:
    Phase 1 (immediate): Find + store + sync → user sees matches in <2 min
    Phase 2 (background): Generate explanations in batches → cached for instant display
    """
    matches = find_matches(user_id, limit=limit)

    legacy_matches = []
    for m in matches:
        legacy_matches.append({
            'user_id': m.user_id,
            'match_type': m.match_type,
            'cosine_score': m.cosine_score,
            'llm_score': m.llm_score,
            'reason': m.reason,
            'similarity_score': m.llm_score / 100.0,
            'combined_score': m.llm_score / 100.0,
            'score_breakdown': m.score_breakdown,
            'reciprocal': m.reciprocal,
            # No explanation yet — will be backfilled async
            # Backend will generate on-demand if user clicks before backfill completes
        })

    if not legacy_matches:
        return {
            'success': True, 'user_id': user_id,
            'total_matches': 0, 'requirements_matches': [],
            'offerings_matches': [], 'stored': True
        }

    # Phase 1: Store matches immediately WITHOUT explanations
    # This allows sync to backend so user sees matches right away
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

    logger.info(f"[LLMMatch] Phase 1 complete: {len(legacy_matches)} matches stored for {user_id[:8]} (no explanations yet)")

    # Phase 2: Kick off background explanation generation
    # Runs in a separate thread so find_and_store_matches returns immediately
    import threading
    thread = threading.Thread(
        target=_backfill_explanations_async,
        args=(user_id, legacy_matches),
        name=f"explain-{user_id[:8]}",
        daemon=True,
    )
    thread.start()
    logger.info(f"[LLMMatch] Phase 2 started: explanation backfill thread for {user_id[:8]}")

    return {
        'success': True, 'user_id': user_id,
        'total_matches': len(legacy_matches),
        'requirements_matches': legacy_matches,
        'offerings_matches': [], 'stored': stored,
    }


def _backfill_explanations_async(user_id: str, matches: list, batch_size: int = 5):
    """Generate explanations in batches and re-sync to backend after each batch.

    Runs in a background thread. Matches are already visible on the dashboard.
    As explanations are generated, they get synced so the cached version is available
    when the user clicks to view them.

    Args:
        user_id: The user whose matches need explanations
        matches: List of match dicts (already stored/synced without explanations)
        batch_size: Number of explanations to generate before re-syncing
    """
    import time
    from app.routers.match import _get_user_persona
    from app.services.llm_service import get_llm_service
    import asyncio

    logger.info(f"[ExplainBackfill] Starting for {user_id[:8]}: {len(matches)} matches, batch_size={batch_size}")

    llm_service = get_llm_service()
    user_persona = _get_user_persona(user_id)

    completed = 0
    failed = 0

    for i, match in enumerate(matches):
        try:
            match_user_id = match['user_id']
            match_persona = _get_user_persona(match_user_id)

            scores = {
                'req_to_off': match.get('similarity_score', 0.5),
                'off_to_req': match.get('similarity_score', 0.5),
                'industry_match': 0.7 if user_persona.get('industry', '').lower() == match_persona.get('industry', '').lower() else 0.4,
                'overall_score': match.get('similarity_score', 0.5),
            }

            # Run async LLM call synchronously
            try:
                result = asyncio.run(
                    llm_service.generate_match_explanation(user_persona, match_persona, scores)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    llm_service.generate_match_explanation(user_persona, match_persona, scores)
                )
                loop.close()

            match['headline'] = result.get('headline', '')
            match['key_points'] = result.get('key_points', [])
            match['explanation'] = result.get('summary', '')
            match['synergy_areas'] = result.get('synergy_areas', [])
            match['friction_points'] = result.get('friction_points', [])
            match['talking_points'] = result.get('talking_points', [])
            completed += 1

            logger.info(f"[ExplainBackfill] {completed}/{len(matches)} — {user_id[:8]}↔{match_user_id[:8]} done")

        except Exception as e:
            logger.warning(f"[ExplainBackfill] Failed for {match.get('user_id', '?')}: {e}")
            match['headline'] = ''
            match['key_points'] = []
            match['explanation'] = match.get('reason', '')
            match['synergy_areas'] = [match.get('reason', '')] if match.get('reason') else []
            match['friction_points'] = []
            match['talking_points'] = []
            failed += 1

        # Re-sync to backend after each batch
        if (i + 1) % batch_size == 0 or (i + 1) == len(matches):
            try:
                # Update the match cache with explanations so far
                UserMatches.store_user_matches(user_id, {
                    'requirements_matches': matches,
                    'offerings_matches': [],
                    'algorithm': 'llm_scored_v1',
                    'total_matches': len(matches),
                })

                # Re-sync to backend — replaces existing match rows with explanation data
                from app.services.match_sync_service import match_sync_service
                sync_result = match_sync_service.sync_matches_to_backend(
                    user_id=user_id,
                    matches={
                        'requirements_matches': matches,
                        'offerings_matches': [],
                    }
                )
                batch_num = (i + 1) // batch_size
                if sync_result.get('success'):
                    logger.info(f"[ExplainBackfill] Batch {batch_num} synced: {completed} explanations for {user_id[:8]}")
                else:
                    logger.warning(f"[ExplainBackfill] Batch {batch_num} sync failed: {sync_result.get('error')}")
            except Exception as sync_err:
                logger.warning(f"[ExplainBackfill] Sync error after batch: {sync_err}")

            # Small delay between batches to avoid API rate limits
            if (i + 1) < len(matches):
                time.sleep(1)

    logger.info(f"[ExplainBackfill] Complete for {user_id[:8]}: {completed} generated, {failed} failed out of {len(matches)}")


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
