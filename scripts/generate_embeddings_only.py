"""
Generate embeddings for all users from their AI summaries.
Embeddings only — no matching, no LLM calls.
Uses Gemini free tier (gemini-embedding-2-preview @ 1536 dims).

Usage:
    python scripts/generate_embeddings_only.py --dry-run
    python scripts/generate_embeddings_only.py
"""
import os
import sys
import time
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('.env.production', override=True)

# Force correct embedding config
os.environ['GEMINI_EMBEDDING_MODEL'] = 'models/gemini-embedding-2-preview'
os.environ['EMBEDDING_DIMENSION'] = '1536'
os.environ['USE_GEMINI_EMBEDDINGS'] = 'true'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from AI summaries')
    parser.add_argument('--dry-run', action='store_true', help='Count only, no writes')
    parser.add_argument('--batch-size', type=int, default=15, help='Users per batch before rate limit pause')
    args = parser.parse_args()

    from app.adapters.postgresql import postgresql_adapter
    from app.adapters.supabase_profiles import UserProfile
    from app.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService()
    logger.info(f"Model: {embedding_service.model_name}")
    logger.info(f"Dimension: {embedding_service.embedding_dimension}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    # Load all AI summaries from user_summaries table
    # This is the rich 1000+ word content generated from onboarding
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
        logger.info(f"Loaded {len(ai_summaries)} AI summaries")
    except Exception as e:
        logger.error(f"Failed to load AI summaries: {e}")
        return

    # Load all profiles
    profiles = list(UserProfile.scan())
    logger.info(f"Found {len(profiles)} user profiles")

    generated = 0
    failed = 0
    skipped = 0
    batch_count = 0

    for i, profile in enumerate(profiles):
        user_id = profile.user_id

        # Get AI summary — only process users who have one
        ai_summary = ai_summaries.get(user_id, '')

        if ai_summary:
            # Full AI summary is embedded for both directions
            # It contains requirements, offerings, profile essence, strategy — everything
            requirements_text = ai_summary
            offerings_text = ai_summary
            source = "AI summary"
            word_count = len(ai_summary.split())
        else:
            # Skip users without AI summary — persona fallback disabled for now
            skipped += 1
            logger.debug(f"[{i+1}/{len(profiles)}] {user_id[:8]} — skipped (no AI summary)")
            continue

        if args.dry_run:
            logger.info(f"[{i+1}/{len(profiles)}] {user_id[:8]} — {source} ({word_count} words)")
            continue

        try:
            success = embedding_service.store_user_embeddings(
                user_id=user_id,
                requirements=requirements_text,
                offerings=offerings_text
            )
            if success:
                generated += 1
                logger.info(f"[{i+1}/{len(profiles)}] {user_id[:8]} — {source} ({word_count} words) ✓")
            else:
                failed += 1
                logger.warning(f"[{i+1}/{len(profiles)}] {user_id[:8]} — FAILED")

            batch_count += 1
            if batch_count >= args.batch_size:
                batch_count = 0
                time.sleep(2)

        except Exception as e:
            failed += 1
            logger.error(f"[{i+1}/{len(profiles)}] {user_id[:8]} — ERROR: {e}")

    logger.info("=" * 50)
    logger.info(f"Total: {len(profiles)}")
    logger.info(f"Generated: {generated}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")


if __name__ == '__main__':
    main()
