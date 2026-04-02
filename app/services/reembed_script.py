"""
One-time migration script: Re-embed all user profiles with current embedding model.

Reads all user profiles from Supabase, generates new embeddings using the
currently configured model (Gemini text-embedding-004), and overwrites
existing embeddings in pgvector.

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

    batch_count = 0

    for i, profile in enumerate(profiles):
        user_id = profile.user_id

        # Extract text from persona
        requirements_text = ""
        offerings_text = ""
        user_type = None

        if profile.persona:
            p = profile.persona
            requirements_text = getattr(p, 'requirements', '') or ''
            offerings_text = getattr(p, 'offerings', '') or ''
            user_type = getattr(p, 'archetype', None) or getattr(p, 'primary_goal', None)

            # If requirements/offerings are empty, try to build from other fields
            if not requirements_text:
                parts = []
                if getattr(p, 'focus', ''): parts.append(f"Focus: {p.focus}")
                if getattr(p, 'designation', ''): parts.append(f"Role: {p.designation}")
                requirements_text = ". ".join(parts)

            if not offerings_text:
                parts = []
                if getattr(p, 'expertise', ''): parts.append(f"Expertise: {p.expertise}")
                if getattr(p, 'designation', ''): parts.append(f"Background: {p.designation}")
                offerings_text = ". ".join(parts)

        if not requirements_text and not offerings_text:
            stats["profiles_skipped_no_text"] += 1
            logger.debug(f"[{i+1}/{len(profiles)}] Skipping {user_id[:8]} — no text to embed")
            continue

        stats["profiles_with_persona"] += 1

        if dry_run:
            logger.info(f"[{i+1}/{len(profiles)}] Would re-embed {user_id[:8]} "
                       f"(req={len(requirements_text)} chars, off={len(offerings_text)} chars, type={user_type})")
            continue

        # === LIVE: Generate and store new embeddings ===
        try:
            # 1. Basic embeddings (requirements + offerings)
            success = embedding_service.store_user_embeddings(
                user_id=user_id,
                requirements=requirements_text,
                offerings=offerings_text
            )

            if success:
                stats["embeddings_generated"] += 1
            else:
                stats["embeddings_failed"] += 1
                logger.warning(f"[{i+1}/{len(profiles)}] Basic embedding failed for {user_id[:8]}")
                continue

            # 2. Multi-vector dimension embeddings
            try:
                mv_result = multi_vector_service.generate_multi_vector_embeddings(
                    user_id=user_id,
                    requirements_text=requirements_text,
                    offerings_text=offerings_text,
                    store_in_db=True,
                    user_type=user_type
                )
                dim_count = len(mv_result.get("dimensions", {}))
                stats["multi_vector_generated"] += dim_count
                logger.info(f"[{i+1}/{len(profiles)}] Re-embedded {user_id[:8]} — "
                           f"basic=OK, multi_vector={dim_count} dims, type={user_type}")
            except Exception as mv_err:
                logger.warning(f"[{i+1}/{len(profiles)}] Multi-vector failed for {user_id[:8]}: {mv_err}")
                # Basic embedding succeeded, that's enough

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
