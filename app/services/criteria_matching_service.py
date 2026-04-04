"""
DISABLED — Superseded by llm_matching_service.py

Criteria-Based Matching Service.
This approach used ideal_match embeddings for bidirectional cosine search.
Replaced by LLM-scored matching which reads profiles directly.

TODO: Delete this file once llm_matching_service.py is confirmed stable.

Original description:
Replaces the rule-based enhanced_matching_service with a simpler architecture:
  FORWARD: my ideal_match embedding → cosine search against all profile embeddings
  REVERSE: my profile embedding → cosine search against all ideal_match embeddings
  MERGE:   geometric mean of both directions
  FILTER:  dealbreakers only

No intent classification. No keyword rules. No penalty/boost system.
"""
import math
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.adapters.postgresql import postgresql_adapter
from app.adapters.supabase_profiles import UserProfile, UserMatches
from app.services.ideal_match_service import generate_ideal_match_profile, needs_regeneration

logger = logging.getLogger(__name__)

# Minimum score in either direction to be considered
REVERSE_MIN_THRESHOLD = float(os.getenv('REVERSE_MIN_THRESHOLD', '0.20'))
# Minimum similarity threshold for pgvector search
SEARCH_THRESHOLD = float(os.getenv('CRITERIA_SEARCH_THRESHOLD', '0.30'))
# Default max matches to return
DEFAULT_LIMIT = int(os.getenv('CRITERIA_MATCH_LIMIT', '30'))


@dataclass
class CriteriaMatch:
    """A single match result from criteria-based matching."""
    user_id: str
    forward_score: float   # how well they match MY ideal
    reverse_score: float   # how well I match THEIR ideal
    combined_score: float  # geometric mean
    match_type: str = 'criteria_bidirectional'


def find_matches(
    user_id: str,
    limit: int = DEFAULT_LIMIT,
    threshold: float = SEARCH_THRESHOLD,
) -> List[CriteriaMatch]:
    """
    Find matches using criteria-based bidirectional search.

    FORWARD: user's ideal_match embedding → cosine against all requirements embeddings
             "Who does this user want to meet?"

    REVERSE: user's requirements embedding → cosine against all ideal_match embeddings
             "Who wants to meet someone like this user?"

    MERGE: geometric mean of both directions. Only include if both > REVERSE_MIN_THRESHOLD.

    POST-FILTER: dealbreakers only.
    """
    try:
        # 1. Ensure ideal match profile is generated and up to date
        if needs_regeneration(user_id):
            logger.info(f"[CriteriaMatch] Regenerating ideal match for {user_id}")
            generate_ideal_match_profile(user_id)

        # 2. Get user's embeddings (returns dict keyed by embedding_type)
        user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
        if not user_embeddings:
            logger.warning(f"[CriteriaMatch] No embeddings for {user_id}")
            return []

        ideal_match_data = user_embeddings.get('ideal_match')
        requirements_data = user_embeddings.get('requirements')
        ideal_match_vec = ideal_match_data['vector_data'] if ideal_match_data else None
        requirements_vec = requirements_data['vector_data'] if requirements_data else None

        if ideal_match_vec is None:
            logger.warning(f"[CriteriaMatch] No ideal_match embedding for {user_id}")
            return []
        if requirements_vec is None:
            logger.warning(f"[CriteriaMatch] No requirements embedding for {user_id}")
            return []

        # 3. FORWARD: my ideal_match → all requirements embeddings
        #    "Find people who ARE what I need"
        forward_results = postgresql_adapter.find_similar_users(
            query_vector=ideal_match_vec,
            embedding_type='requirements',
            threshold=threshold,
            exclude_user_id=user_id
        )
        forward_scores = {r['user_id']: r['similarity_score'] for r in forward_results}

        # 4. REVERSE: my requirements → all ideal_match embeddings
        #    "Find people who NEED someone like me"
        reverse_results = postgresql_adapter.find_similar_users(
            query_vector=requirements_vec,
            embedding_type='ideal_match',
            threshold=threshold,
            exclude_user_id=user_id
        )
        reverse_scores = {r['user_id']: r['similarity_score'] for r in reverse_results}

        logger.info(
            f"[CriteriaMatch] {user_id}: {len(forward_scores)} forward, "
            f"{len(reverse_scores)} reverse candidates"
        )

        # 5. MERGE: geometric mean, both directions must pass minimum
        all_candidates = set(forward_scores.keys()) | set(reverse_scores.keys())
        matches = []

        for cand_id in all_candidates:
            fwd = forward_scores.get(cand_id, 0.0)
            rev = reverse_scores.get(cand_id, 0.0)

            # Both directions must have minimum signal
            if fwd < REVERSE_MIN_THRESHOLD or rev < REVERSE_MIN_THRESHOLD:
                continue

            combined = math.sqrt(fwd * rev)

            matches.append(CriteriaMatch(
                user_id=cand_id,
                forward_score=fwd,
                reverse_score=rev,
                combined_score=combined,
            ))

        # 6. POST-FILTER: dealbreakers
        matches = _apply_dealbreaker_filter(user_id, matches)

        # 7. Rank and limit
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        result = matches[:limit]

        logger.info(f"[CriteriaMatch] Returning {len(result)} matches for {user_id}")
        return result

    except Exception as e:
        logger.error(f"[CriteriaMatch] Error finding matches for {user_id}: {e}", exc_info=True)
        return []


def find_and_store_matches(user_id: str, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    """
    Find matches and store them in the match cache.
    Compatible with the legacy matching adapter interface.
    """
    matches = find_matches(user_id, limit=limit)

    if not matches:
        return {
            'success': True,
            'user_id': user_id,
            'total_matches': 0,
            'requirements_matches': [],
            'offerings_matches': [],
            'stored': True
        }

    # Format for storage
    legacy_matches = []
    for m in matches:
        legacy_matches.append({
            'user_id': m.user_id,
            'match_type': m.match_type,
            'forward_score': m.forward_score,
            'reverse_score': m.reverse_score,
            'combined_score': m.combined_score,
            'similarity_score': m.combined_score,  # legacy compat
        })

    # Store in match cache
    try:
        stored = UserMatches.store_user_matches(user_id, {
            'requirements_matches': legacy_matches,
            'offerings_matches': [],
            'algorithm': 'criteria_v1',
            'total_matches': len(legacy_matches)
        })
    except Exception as e:
        logger.warning(f"[CriteriaMatch] Failed to store matches: {e}")
        stored = False

    return {
        'success': True,
        'user_id': user_id,
        'total_matches': len(legacy_matches),
        'requirements_matches': legacy_matches,
        'offerings_matches': [],
        'stored': stored
    }


def _apply_dealbreaker_filter(
    user_id: str,
    matches: List[CriteriaMatch]
) -> List[CriteriaMatch]:
    """Remove matches that violate user's stated dealbreakers."""
    try:
        # Get user's dealbreakers from onboarding slots
        from app.adapters.supabase_onboarding import supabase_onboarding_adapter
        slots = supabase_onboarding_adapter.get_user_slots_sync(user_id)
        if not slots:
            return matches

        dealbreaker_slot = slots.get('dealbreakers', {})
        dealbreaker_text = dealbreaker_slot.get('value', '') if isinstance(dealbreaker_slot, dict) else str(dealbreaker_slot)

        if not dealbreaker_text or not dealbreaker_text.strip():
            return matches

        # Parse dealbreakers (semicolon or comma separated)
        dealbreakers = [
            d.strip().lower()
            for d in dealbreaker_text.replace(';', ',').split(',')
            if d.strip()
        ]

        if not dealbreakers:
            return matches

        logger.info(f"[CriteriaMatch] Checking {len(dealbreakers)} dealbreakers for {user_id}")

        filtered = []
        blocked = 0
        for m in matches:
            try:
                profile = UserProfile.get(m.user_id)
                if not profile or not profile.persona:
                    filtered.append(m)
                    continue

                # Build candidate text to check against
                p = profile.persona
                candidate_text = ' '.join([
                    getattr(p, 'archetype', '') or '',
                    getattr(p, 'designation', '') or '',
                    getattr(p, 'focus', '') or '',
                    getattr(p, 'profile_essence', '') or '',
                    getattr(p, 'offerings', '') or '',
                ]).lower()

                has_violation = any(db in candidate_text for db in dealbreakers)
                if has_violation:
                    blocked += 1
                else:
                    filtered.append(m)
            except Exception:
                filtered.append(m)  # On error, keep the match

        if blocked > 0:
            logger.info(f"[CriteriaMatch] Blocked {blocked} matches by dealbreakers for {user_id}")

        return filtered

    except Exception as e:
        logger.debug(f"[CriteriaMatch] Dealbreaker check failed for {user_id}: {e}")
        return matches


# Singleton-style accessor
def get_criteria_matching_service():
    """Return module-level functions for matching."""
    return {
        'find_matches': find_matches,
        'find_and_store_matches': find_and_store_matches,
    }
