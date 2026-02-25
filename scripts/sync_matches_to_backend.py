#!/usr/bin/env python3
"""
Sync Matches to Backend

Fetches matches from AI service and sends them to backend webhook
to persist in the matches table.

Usage:
    python scripts/sync_matches_to_backend.py --user USER_ID
    python scripts/sync_matches_to_backend.py --all
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.services.match_sync_service import match_sync_service


def sync_user(user_id: str) -> dict:
    """Sync matches for a single user."""
    return match_sync_service.sync_user_matches(user_id)


def sync_all_users():
    """Sync matches for all users in the backend."""
    import psycopg2

    # Get all users from backend
    conn = psycopg2.connect(
        os.getenv('RECIPROCITY_BACKEND_DB_URL',
                  'postgresql://postgres:postgres@localhost:5432/reciprocity_db')
    )
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users')
    user_ids = [str(row[0]) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    logger.info(f"Found {len(user_ids)} users to sync")

    results = {'synced': 0, 'failed': 0, 'no_matches': 0, 'total_matches': 0}

    for i, user_id in enumerate(user_ids, 1):
        logger.info(f"[{i}/{len(user_ids)}] Processing {user_id}")
        result = sync_user(user_id)

        if result.get('success'):
            count = result.get('count', 0)
            results['total_matches'] += count
            if count > 0:
                results['synced'] += 1
            else:
                results['no_matches'] += 1
        else:
            results['failed'] += 1

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Sync matches to backend')
    parser.add_argument('--user', type=str, help='Sync specific user ID')
    parser.add_argument('--all', action='store_true', help='Sync all users')

    args = parser.parse_args()

    if args.user:
        result = sync_user(args.user)
        logger.info(f"Result: {result}")
    elif args.all:
        results = sync_all_users()
        logger.info(f"Sync complete: {results}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
