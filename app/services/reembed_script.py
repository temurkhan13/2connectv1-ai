"""
Migration script: Re-embed all user profiles with current embedding model.

Reads all user profiles from Supabase, generates new embeddings using the
currently configured model (gemini-embedding-2-preview @ 1536 dims), and
overwrites existing embeddings in pgvector.

IMPORTANT: Run the database migration first (clears old embeddings, adds conversation_text):
    supabase/migrations/20260404_upgrade_embeddings_1536.sql

Usage:
    # Dry run (count only)
    python -m app.services.reembed_script --dry-run

    # Full run
    python -m app.services.reembed_script

    # With batch size
    python -m app.services.reembed_script --batch-size 20
"""
import os
import sys
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_reembedding(dry_run: bool = True, batch_size: int = 10) -> Dict[str, Any]:
    """
    Re-embed all users with the current embedding model.

    Args:
        dry_run: If True, only count and report — don't write anything
        batch_size: Number of users to process before sleeping (rate limit protection)

    Returns:
        Summary dict with counts
    """
    from app.adapters.supabase_profiles import UserProfile
    from app.services.embedding_service import embedding_service
    from app.services.multi_vector_embedding_service import multi_vector_service

    logger.info(f"=== RE-EMBEDDING ALL USERS ===")
    logger.info(f"Model: {embedding_service.model_name}")
    logger.info(f"Dimension: {embedding_service.embedding_dimension}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Batch size: {batch_size}")

    stats = {
        "total_profiles": 0,
        "profiles_with_persona": 0,
        "profiles_skipped_no_text": 0,
        "embeddings_generated": 0,
        "embeddings_failed": 0,
        "multi_vector_generated": 0,
        "model": embedding_service.model_name,
        "dimension": embedding_service.embedding_dimension,
        "dry_run": dry_run,
    }

    # Scan all profiles
    profiles = list(UserProfile.scan())
    stats["total_profiles"] = len(profiles)
    logger.info(f"Found {len(profiles)} user profiles")

    # Batch-fetch all AI summaries from user_summaries table
    # The AI summary is the rich content (1000+ words) that should be embedded,
    # NOT the thin persona requirements/offerings fields
    from app.adapters.postgresql import postgresql_adapter
    ai_summaries = {}
    try:
        conn = postgresql_adapter.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT ON (user_id) user_id, summary
            FROM user_summaries
            WHERE status = 'approved'
            ORDER BY user_id, created_at DESC
        """)
        for row in cursor.fetchall():
            ai_summaries[str(row[0])] = row[1]
        cursor.close()
        conn.close()
        logger.info(f"Loaded {len(ai_summaries)} AI summaries from user_summaries")
    except Exception as e:
        logger.error(f"Failed to load AI summaries: {e}")

    batch_count = 0

    for i, profile in enumerate(profiles):
        user_id = profile.user_id

        # Use the full AI summary as the embedding source — it contains
        # profile essence, strategy, focus, requirements, offerings, everything.
        # This is 1000+ words of rich content vs ~400 words from persona fields alone.
        ai_summary = ai_summaries.get(user_id, '')

        # Fallback to persona requirements + offerings if no AI summary
        requirements_text = ""
        offerings_text = ""
        if not ai_summary and profile.persona:
            p = profile.persona
            requirements_text = getattr(p, 'requirements', '') or ''
            offerings_text = getattr(p, 'offerings', '') or ''

        if not ai_summary and not requirements_text and not offerings_text:
            stats["profiles_skipped_no_text"] += 1
            logger.debug(f"[{i+1}/{len(profiles)}] Skipping {user_id[:8]} — no text to embed")
            continue

        stats["profiles_with_persona"] += 1

        if dry_run:
            source = f"AI summary ({len(ai_summary.split())} words)" if ai_summary else f"persona (req={len(requirements_text)}, off={len(offerings_text)} chars)"
            logger.info(f"[{i+1}/{len(profiles)}] Would re-embed {user_id[:8]} — {source}")
            continue

        # === LIVE: Generate and store new embeddings ===
        try:
            if ai_summary:
                # Embed the full AI summary as both requirements and offerings embedding
                # The cosine pre-filter searches req→offerings cross-direction,
                # so both need to exist for bidirectional matching
                success = embedding_service.store_user_embeddings(
                    user_id=user_id,
                    requirements=ai_summary,
                    offerings=ai_summary
                )
            else:
                # Fallback: use persona fields
                success = embedding_service.store_user_embeddings(
                    user_id=user_id,
                    requirements=requirements_text,
                    offerings=offerings_text
                )

            if success:
                stats["embeddings_generated"] += 1
                source = "AI summary" if ai_summary else "persona"
                logger.info(f"[{i+1}/{len(profiles)}] Re-embedded {user_id[:8]} from {source}")
            else:
                stats["embeddings_failed"] += 1
                logger.warning(f"[{i+1}/{len(profiles)}] Embedding failed for {user_id[:8]}")

            # Rate limit protection: pause between batches
            batch_count += 1
            if batch_count >= batch_size:
                batch_count = 0
                logger.info(f"Batch complete, sleeping 2s for rate limit protection...")
                time.sleep(2)

        except Exception as e:
            stats["embeddings_failed"] += 1
            logger.error(f"[{i+1}/{len(profiles)}] Failed to re-embed {user_id[:8]}: {e}")

    # Summary
    logger.info(f"=== RE-EMBEDDING COMPLETE ===")
    logger.info(f"Total profiles: {stats['total_profiles']}")
    logger.info(f"With persona text: {stats['profiles_with_persona']}")
    logger.info(f"Skipped (no text): {stats['profiles_skipped_no_text']}")
    logger.info(f"Embeddings generated: {stats['embeddings_generated']}")
    logger.info(f"Embeddings failed: {stats['embeddings_failed']}")
    logger.info(f"Multi-vector dims: {stats['multi_vector_generated']}")

    return stats


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Re-embed all user profiles")
    parser.add_argument("--dry-run", action="store_true", default=False,
                       help="Count profiles without generating embeddings")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Users per batch before rate limit pause")
    args = parser.parse_args()

    result = run_reembedding(dry_run=args.dry_run, batch_size=args.batch_size)
    print(f"\n{'='*50}")
    print(f"Result: {result}")
