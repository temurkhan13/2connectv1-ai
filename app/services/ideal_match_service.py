"""
DISABLED — Superseded by llm_matching_service.py

Ideal Match Profile Service.
This approach (embed ideal match description → cosine search) was tested and found
to have the same directionality limitations as regular embedding matching.
Replaced by LLM-scored matching: cosine pre-filter → LLM reads both profiles → scores directly.

TODO: Delete this file once llm_matching_service.py is confirmed stable.

Original description:
Generates a description of each user's ideal match using LLM,
then embeds it for criteria-based matching via cosine similarity.

Architecture:
  AI Summary → LLM describes ideal match → embed → store
  Matching: ideal_match embedding ↔ profile embeddings (bidirectional)

No intent classification. No keyword rules. No penalty/boost system.
The LLM comprehends who the user needs and describes that person.
"""
import hashlib
import logging
import os
from typing import Optional, Dict, Any

from app.services.llm_fallback import call_with_fallback
from app.services.embedding_service import embedding_service
from app.adapters.postgresql import postgresql_adapter

logger = logging.getLogger(__name__)

IDEAL_MATCH_PROMPT = """Read this user's profile carefully. Based ONLY on what they explicitly state they want and need, describe their ideal match — the person they should meet on this professional networking platform.

Describe the ideal match AS A REAL PERSON who exists on this platform:
- Who they are (role, title, background, years of experience)
- What they do (their work, business, expertise area)
- What they offer (skills, resources, access, capital, network, knowledge)
- Why they would want to connect with this user (what mutual value exchange looks like)

CRITICAL RULES:
1. Describe the match's IDENTITY and what they OFFER — NOT their problems or struggles.
   WRONG: "This person is struggling with customer acquisition..."
   RIGHT: "This person is an e-commerce founder running a $3M ARR brand with strong product but no marketing team..."

2. Do NOT invent specifics the user didn't mention.
   If user says "e-commerce clients" — write about e-commerce business owners.
   Do NOT add Shopify, fashion, Series A, London — unless the user said so.
   Use GENERAL language where the user was general.
   Use SPECIFIC language where the user was specific.

3. If the user has MULTIPLE needs (e.g., needs investors AND a technical partner),
   describe BOTH types of ideal matches in the same description.

4. Write in third person, same style and tone as a real user profile on the platform.

User's Profile:
{profile_text}"""

MIN_DESCRIPTION_WORDS = 200
MAX_RETRIES = 2
RETRY_PROMPT_SUFFIX = "\n\nIMPORTANT: Your previous description was too brief. Write a MORE detailed profile covering their role, background, what they do, what they offer, and why they'd connect with this user. Be thorough."


def _build_profile_text(user_id: str) -> Optional[str]:
    """Build the profile text from user_profiles persona fields."""
    try:
        conn = postgresql_adapter.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT persona_profile_essence, persona_requirements,
                   persona_offerings, persona_designation, persona_focus,
                   persona_what_looking_for, persona_engagement_style
            FROM user_profiles WHERE user_id = %s
        """, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            logger.warning(f"No profile found for user {user_id}")
            return None

        parts = []
        labels = [
            ("Profile", row[0]),
            ("What they're looking for", row[1]),
            ("What they offer", row[2]),
            ("Designation", row[3]),
            ("Focus areas", row[4]),
            ("Seeking", row[5]),
            ("Engagement style", row[6]),
        ]
        for label, value in labels:
            if value and value.strip() and value.strip() != "Not specified":
                parts.append(f"{label}: {value.strip()}")

        if not parts:
            logger.warning(f"Empty profile for user {user_id}")
            return None

        return "\n\n".join(parts)

    except Exception as e:
        logger.error(f"Error reading profile for {user_id}: {e}")
        return None


def _compute_hash(text: str) -> str:
    """SHA256 hash of the profile text for cache invalidation."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:64]


def generate_ideal_match_profile(user_id: str, force: bool = False) -> Optional[str]:
    """
    Generate an ideal match description for a user using LLM.

    Args:
        user_id: User identifier
        force: If True, regenerate even if cached version exists

    Returns:
        The generated ideal match profile text, or None on failure
    """
    # 1. Build profile text
    profile_text = _build_profile_text(user_id)
    if not profile_text:
        return None

    profile_hash = _compute_hash(profile_text)

    # 2. Check cache — skip if profile hasn't changed
    if not force:
        try:
            conn = postgresql_adapter.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ideal_match_hash, ideal_match_profile FROM user_profiles WHERE user_id = %s",
                (user_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row and row[0] == profile_hash and row[1]:
                logger.info(f"[IdealMatch] Cache hit for {user_id} — profile unchanged")
                return row[1]
        except Exception as e:
            logger.debug(f"[IdealMatch] Cache check failed for {user_id}: {e}")

    # 3. Generate via LLM
    logger.info(f"[IdealMatch] Generating ideal match profile for {user_id} ({len(profile_text)} chars)")

    description = None
    prompt = IDEAL_MATCH_PROMPT.format(profile_text=profile_text)

    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt > 0:
                prompt = IDEAL_MATCH_PROMPT.format(profile_text=profile_text) + RETRY_PROMPT_SUFFIX
                logger.info(f"[IdealMatch] Retry {attempt}/{MAX_RETRIES} for {user_id} (too short)")

            result = call_with_fallback(
                service="matching",
                system_prompt="You are an expert at understanding professional profiles and describing ideal connections.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.3,
            )

            if result and len(result.split()) >= MIN_DESCRIPTION_WORDS:
                description = result.strip()
                break
            elif result:
                logger.warning(
                    f"[IdealMatch] Description too short for {user_id}: "
                    f"{len(result.split())} words (min {MIN_DESCRIPTION_WORDS})"
                )
                description = result.strip()  # Keep for potential retry improvement

        except Exception as e:
            logger.error(f"[IdealMatch] LLM call failed for {user_id} (attempt {attempt}): {e}")

    if not description:
        logger.error(f"[IdealMatch] All attempts failed for {user_id}")
        return None

    word_count = len(description.split())
    logger.info(f"[IdealMatch] Generated {word_count} word description for {user_id}")

    # 4. Store the description and hash
    try:
        conn = postgresql_adapter.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_profiles
            SET ideal_match_profile = %s, ideal_match_hash = %s
            WHERE user_id = %s
        """, (description, profile_hash, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[IdealMatch] Stored description for {user_id}")
    except Exception as e:
        logger.error(f"[IdealMatch] Failed to store description for {user_id}: {e}")
        return None

    # 5. Generate and store embedding
    try:
        embedding = embedding_service.generate_embedding(description)
        if embedding:
            postgresql_adapter.store_embedding(
                user_id=user_id,
                embedding_type='ideal_match',
                vector_data=embedding,
                metadata={'word_count': word_count, 'hash': profile_hash}
            )
            logger.info(f"[IdealMatch] Stored ideal_match embedding for {user_id}")
        else:
            logger.error(f"[IdealMatch] Embedding generation failed for {user_id}")
    except Exception as e:
        logger.error(f"[IdealMatch] Embedding storage failed for {user_id}: {e}")

    return description


def needs_regeneration(user_id: str) -> bool:
    """Check if user's ideal match profile needs regeneration."""
    profile_text = _build_profile_text(user_id)
    if not profile_text:
        return False

    current_hash = _compute_hash(profile_text)

    try:
        conn = postgresql_adapter.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ideal_match_hash FROM user_profiles WHERE user_id = %s",
            (user_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row or not row[0]:
            return True  # No cached version
        return row[0] != current_hash
    except Exception:
        return True
