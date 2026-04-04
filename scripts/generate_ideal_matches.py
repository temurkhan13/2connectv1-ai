"""
DISABLED — Used by ideal_match_service.py which is superseded by llm_matching_service.py
TODO: Delete when llm_matching_service is confirmed stable.

Backfill script: Generate ideal match profiles for all existing users.

Reads all user profiles with completed personas, generates the LLM-based
ideal match description, embeds it, and stores both.

Usage:
    # Dry run (count only)
    python scripts/generate_ideal_matches.py --dry-run

    # Full run
    python scripts/generate_ideal_matches.py

    # Test on specific users
    python scripts/generate_ideal_matches.py --users "user_id_1,user_id_2"

    # Force regeneration (even if cached)
    python scripts/generate_ideal_matches.py --force
"""
import os
import sys
import time
import json
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.production'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate ideal match profiles for all users')
    parser.add_argument('--dry-run', action='store_true', help='Count only, no generation')
    parser.add_argument('--users', type=str, help='Comma-separated user IDs to process')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if cached')
    parser.add_argument('--batch-size', type=int, default=10, help='Pause after N users (rate limit)')
    args = parser.parse_args()

    from app.adapters.supabase_profiles import UserProfile
    from app.services.ideal_match_service import generate_ideal_match_profile

    # Get users to process
    if args.users:
        user_ids = [uid.strip() for uid in args.users.split(',')]
        logger.info(f"Processing {len(user_ids)} specified users")
    else:
        # Get all users with completed personas
        profiles = list(UserProfile.scan())
        user_ids = [
            p.user_id for p in profiles
            if p.persona_status == 'completed' and p.persona and getattr(p.persona, 'name', None)
        ]
        logger.info(f"Found {len(user_ids)} users with completed personas")

    if args.dry_run:
        logger.info(f"DRY RUN — would generate ideal match profiles for {len(user_ids)} users")
        return

    # Progress tracking
    progress_file = os.path.join(os.path.dirname(__file__), '.ideal_match_progress.json')
    done = set()
    failed = []

    if os.path.exists(progress_file) and not args.force:
        with open(progress_file) as f:
            data = json.load(f)
            done = set(data.get('done', []))
            failed = data.get('failed', [])
        logger.info(f"Resuming: {len(done)} already done, {len(failed)} previously failed")

    remaining = [uid for uid in user_ids if uid not in done]
    logger.info(f"Processing {len(remaining)} users ({len(done)} already done)")

    batch_count = 0
    for i, user_id in enumerate(remaining):
        try:
            result = generate_ideal_match_profile(user_id, force=args.force)
            if result:
                word_count = len(result.split())
                logger.info(f"[{i+1}/{len(remaining)}] OK {user_id[:12]}... ({word_count} words)")
                done.add(user_id)
            else:
                logger.warning(f"[{i+1}/{len(remaining)}] FAILED {user_id[:12]}... (returned None)")
                failed.append(user_id)

        except Exception as e:
            logger.error(f"[{i+1}/{len(remaining)}] ERROR {user_id[:12]}...: {e}")
            failed.append(user_id)

        # Save progress
        with open(progress_file, 'w') as f:
            json.dump({'done': list(done), 'failed': failed}, f)

        # Rate limit protection
        batch_count += 1
        if batch_count >= args.batch_size:
            batch_count = 0
            logger.info(f"Batch complete, sleeping 2s...")
            time.sleep(2)

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total:   {len(user_ids)}")
    logger.info(f"Done:    {len(done)}")
    logger.info(f"Failed:  {len(failed)}")
    if failed:
        logger.info(f"Failed IDs: {failed[:10]}...")


if __name__ == '__main__':
    main()
