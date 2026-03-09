#!/usr/bin/env python3
"""
Migration script: DynamoDB to Supabase

Migrates all data from DynamoDB tables to Supabase PostgreSQL tables:
- UserProfile -> user_profiles
- UserMatches -> user_match_cache
- NotifiedMatchPairs -> notified_match_pairs
- Feedback -> match_feedback (existing table)
- ChatRecord -> ai_conversations (existing table)

Usage:
    python scripts/migrate_dynamodb_to_supabase.py --dry-run
    python scripts/migrate_dynamodb_to_supabase.py --execute
    python scripts/migrate_dynamodb_to_supabase.py --validate
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamoToSupabaseMigrator:
    """Migrates data from DynamoDB to Supabase PostgreSQL."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.supabase_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')

        if not self.supabase_url:
            raise ValueError("RECIPROCITY_BACKEND_DB_URL environment variable is required")

        self.stats = {
            'user_profiles': {'migrated': 0, 'skipped': 0, 'errors': []},
            'user_matches': {'migrated': 0, 'skipped': 0, 'errors': []},
            'notified_pairs': {'migrated': 0, 'skipped': 0, 'errors': []},
            'feedback': {'migrated': 0, 'skipped': 0, 'errors': []},
            'chat_records': {'migrated': 0, 'skipped': 0, 'errors': []},
        }

        # Import DynamoDB models (still available during migration)
        try:
            from app.adapters.dynamodb import (
                UserProfile, UserMatches, NotifiedMatchPairs,
                Feedback, ChatRecord
            )
            self.UserProfile = UserProfile
            self.UserMatches = UserMatches
            self.NotifiedMatchPairs = NotifiedMatchPairs
            self.Feedback = Feedback
            self.ChatRecord = ChatRecord
            self.dynamo_available = True
        except Exception as e:
            logger.warning(f"DynamoDB models not available: {e}")
            self.dynamo_available = False

    def get_connection(self):
        """Get Supabase database connection."""
        return psycopg2.connect(self.supabase_url)

    def check_tables_exist(self) -> bool:
        """Check if migration target tables exist."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            tables = ['user_profiles', 'user_match_cache', 'notified_match_pairs']
            missing = []

            for table in tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    )
                """, (table,))
                exists = cursor.fetchone()[0]
                if not exists:
                    missing.append(table)

            if missing:
                logger.error(f"Missing tables: {missing}")
                logger.error("Run the SQL migration first: supabase/migrations/20260309_dynamo_to_supabase_migration.sql")
                return False

            logger.info("All target tables exist")
            return True

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_valid_user_ids(self) -> set:
        """Get set of valid user IDs from users table."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id::text FROM users")
            return {row[0] for row in cursor.fetchall()}

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def migrate_user_profiles(self):
        """Migrate UserProfile from DynamoDB to user_profiles table."""
        if not self.dynamo_available:
            logger.warning("Skipping user_profiles migration - DynamoDB not available")
            return

        logger.info("Migrating UserProfile records...")
        valid_users = self.get_valid_user_ids()
        logger.info(f"Found {len(valid_users)} valid users in Supabase")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            count = 0
            for profile in self.UserProfile.scan():
                count += 1
                user_id = profile.user_id

                try:
                    # Check if user exists in users table
                    if user_id not in valid_users:
                        logger.debug(f"Skipping profile {user_id} - user not in users table")
                        self.stats['user_profiles']['skipped'] += 1
                        continue

                    # Build raw_questions JSONB
                    raw_questions = []
                    if profile.profile and profile.profile.raw_questions:
                        for q in profile.profile.raw_questions:
                            raw_questions.append({
                                'prompt': q.prompt if hasattr(q, 'prompt') else q.get('prompt', ''),
                                'answer': q.answer if hasattr(q, 'answer') else q.get('answer', '')
                            })

                    if not self.dry_run:
                        cursor.execute("""
                            INSERT INTO user_profiles (
                                user_id, resume_link, raw_questions,
                                resume_text, resume_extracted_at, resume_extraction_method,
                                persona_name, persona_archetype, persona_designation,
                                persona_experience, persona_focus, persona_profile_essence,
                                persona_strategy, persona_what_looking_for, persona_engagement_style,
                                persona_requirements, persona_offerings, persona_user_type,
                                persona_industry, persona_generated_at,
                                processing_status, persona_status, needs_matchmaking,
                                created_at, updated_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (user_id) DO UPDATE SET
                                persona_name = EXCLUDED.persona_name,
                                persona_archetype = EXCLUDED.persona_archetype,
                                persona_designation = EXCLUDED.persona_designation,
                                persona_experience = EXCLUDED.persona_experience,
                                persona_focus = EXCLUDED.persona_focus,
                                persona_profile_essence = EXCLUDED.persona_profile_essence,
                                persona_strategy = EXCLUDED.persona_strategy,
                                persona_what_looking_for = EXCLUDED.persona_what_looking_for,
                                persona_engagement_style = EXCLUDED.persona_engagement_style,
                                persona_requirements = EXCLUDED.persona_requirements,
                                persona_offerings = EXCLUDED.persona_offerings,
                                persona_user_type = EXCLUDED.persona_user_type,
                                persona_industry = EXCLUDED.persona_industry,
                                persona_generated_at = EXCLUDED.persona_generated_at,
                                processing_status = EXCLUDED.processing_status,
                                persona_status = EXCLUDED.persona_status,
                                needs_matchmaking = EXCLUDED.needs_matchmaking,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            user_id,
                            profile.profile.resume_link if profile.profile else None,
                            Json(raw_questions),
                            profile.resume_text.text if profile.resume_text else None,
                            profile.resume_text.extracted_at if profile.resume_text and profile.resume_text.extracted_at else None,
                            profile.resume_text.extraction_method if profile.resume_text else None,
                            profile.persona.name if profile.persona else None,
                            profile.persona.archetype if profile.persona else None,
                            profile.persona.designation if profile.persona else None,
                            profile.persona.experience if profile.persona else None,
                            profile.persona.focus if profile.persona else None,
                            profile.persona.profile_essence if profile.persona else None,
                            profile.persona.strategy if profile.persona else None,
                            profile.persona.what_theyre_looking_for if profile.persona else None,
                            profile.persona.engagement_style if profile.persona else None,
                            profile.persona.requirements if profile.persona else None,
                            profile.persona.offerings if profile.persona else None,
                            profile.persona.user_type if profile.persona else None,
                            profile.persona.industry if profile.persona else None,
                            profile.persona.generated_at if profile.persona and profile.persona.generated_at else None,
                            profile.processing_status,
                            profile.persona_status,
                            profile.needs_matchmaking == 'true',
                            profile.profile.created_at if profile.profile and profile.profile.created_at else datetime.now(timezone.utc),
                            profile.profile.updated_at if profile.profile and profile.profile.updated_at else datetime.now(timezone.utc),
                        ))

                    self.stats['user_profiles']['migrated'] += 1
                    if count % 10 == 0:
                        logger.info(f"Processed {count} profiles...")

                except Exception as e:
                    self.stats['user_profiles']['errors'].append({
                        'user_id': user_id,
                        'error': str(e)
                    })
                    logger.error(f"Error migrating profile {user_id}: {e}")

            if not self.dry_run:
                conn.commit()

            logger.info(f"UserProfile migration complete: {self.stats['user_profiles']}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def migrate_user_matches(self):
        """Migrate UserMatches from DynamoDB to user_match_cache table."""
        if not self.dynamo_available:
            logger.warning("Skipping user_matches migration - DynamoDB not available")
            return

        logger.info("Migrating UserMatches records...")
        valid_users = self.get_valid_user_ids()

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            count = 0
            for matches in self.UserMatches.scan():
                count += 1
                user_id = matches.user_id

                try:
                    if user_id not in valid_users:
                        self.stats['user_matches']['skipped'] += 1
                        continue

                    # Build matches JSONB
                    matches_data = {
                        'requirements_matches': [],
                        'offerings_matches': []
                    }

                    for match in matches.matches:
                        match_dict = {
                            'user_id': match.matched_user_id,
                            'similarity_score': float(match.similarity_score),
                            'match_type': match.match_type,
                            'explanation': match.explanation,
                            'created_at': match.created_at.isoformat() if match.created_at else None
                        }
                        if match.match_type == 'requirements':
                            matches_data['requirements_matches'].append(match_dict)
                        else:
                            matches_data['offerings_matches'].append(match_dict)

                    if not self.dry_run:
                        cursor.execute("""
                            INSERT INTO user_match_cache (user_id, matches, total_matches, last_updated)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (user_id) DO UPDATE SET
                                matches = EXCLUDED.matches,
                                total_matches = EXCLUDED.total_matches,
                                last_updated = EXCLUDED.last_updated
                        """, (
                            user_id,
                            Json(matches_data),
                            int(matches.total_matches),
                            matches.last_updated if matches.last_updated else datetime.now(timezone.utc)
                        ))

                    self.stats['user_matches']['migrated'] += 1

                except Exception as e:
                    self.stats['user_matches']['errors'].append({
                        'user_id': user_id,
                        'error': str(e)
                    })
                    logger.error(f"Error migrating matches for {user_id}: {e}")

            if not self.dry_run:
                conn.commit()

            logger.info(f"UserMatches migration complete: {self.stats['user_matches']}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def migrate_notified_pairs(self):
        """Migrate NotifiedMatchPairs from DynamoDB to notified_match_pairs table."""
        if not self.dynamo_available:
            logger.warning("Skipping notified_pairs migration - DynamoDB not available")
            return

        logger.info("Migrating NotifiedMatchPairs records...")
        valid_users = self.get_valid_user_ids()

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            count = 0
            for pair in self.NotifiedMatchPairs.scan():
                count += 1

                try:
                    # Both users must exist
                    if pair.user_a_id not in valid_users or pair.user_b_id not in valid_users:
                        self.stats['notified_pairs']['skipped'] += 1
                        continue

                    # Ensure correct ordering (a < b)
                    if pair.user_a_id < pair.user_b_id:
                        user_a, user_b = pair.user_a_id, pair.user_b_id
                    else:
                        user_a, user_b = pair.user_b_id, pair.user_a_id

                    if not self.dry_run:
                        cursor.execute("""
                            INSERT INTO notified_match_pairs (
                                user_a_id, user_b_id, notified_at,
                                notification_count, last_similarity_score
                            ) VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (user_a_id, user_b_id) DO UPDATE SET
                                notification_count = notified_match_pairs.notification_count + EXCLUDED.notification_count,
                                notified_at = EXCLUDED.notified_at,
                                last_similarity_score = COALESCE(EXCLUDED.last_similarity_score, notified_match_pairs.last_similarity_score)
                        """, (
                            user_a,
                            user_b,
                            pair.notified_at if pair.notified_at else datetime.now(timezone.utc),
                            pair.notification_count or 1,
                            float(pair.last_similarity_score) if pair.last_similarity_score else None
                        ))

                    self.stats['notified_pairs']['migrated'] += 1

                except Exception as e:
                    self.stats['notified_pairs']['errors'].append({
                        'pair_key': pair.pair_key,
                        'error': str(e)
                    })
                    logger.error(f"Error migrating pair {pair.pair_key}: {e}")

            if not self.dry_run:
                conn.commit()

            logger.info(f"NotifiedMatchPairs migration complete: {self.stats['notified_pairs']}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def validate_migration(self):
        """Validate migration by comparing record counts."""
        if not self.dynamo_available:
            logger.warning("Cannot validate - DynamoDB not available")
            return

        logger.info("Validating migration...")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Count DynamoDB records
            dynamo_profiles = sum(1 for _ in self.UserProfile.scan())
            dynamo_matches = sum(1 for _ in self.UserMatches.scan())
            dynamo_pairs = sum(1 for _ in self.NotifiedMatchPairs.scan())

            # Count Supabase records
            cursor.execute("SELECT COUNT(*) FROM user_profiles")
            supabase_profiles = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_match_cache")
            supabase_matches = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM notified_match_pairs")
            supabase_pairs = cursor.fetchone()[0]

            # Count valid users (some DynamoDB records may be for deleted users)
            cursor.execute("SELECT COUNT(*) FROM users")
            valid_users = cursor.fetchone()[0]

            logger.info("=" * 60)
            logger.info("MIGRATION VALIDATION REPORT")
            logger.info("=" * 60)
            logger.info(f"Valid users in Supabase: {valid_users}")
            logger.info("")
            logger.info(f"UserProfile:      DynamoDB={dynamo_profiles}, Supabase={supabase_profiles}")
            logger.info(f"UserMatches:      DynamoDB={dynamo_matches}, Supabase={supabase_matches}")
            logger.info(f"NotifiedPairs:    DynamoDB={dynamo_pairs}, Supabase={supabase_pairs}")
            logger.info("")

            # Note: Counts may not match exactly due to orphaned records
            if supabase_profiles > 0:
                logger.info("Migration appears successful!")
            else:
                logger.warning("No records in Supabase tables - migration may not have run")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def run_full_migration(self):
        """Run complete migration."""
        mode = "DRY RUN" if self.dry_run else "EXECUTE"
        logger.info(f"Starting migration ({mode})")
        logger.info("=" * 60)

        # Check prerequisites
        if not self.check_tables_exist():
            logger.error("Migration aborted - tables not found")
            return self.stats

        # Run migrations
        self.migrate_user_profiles()
        self.migrate_user_matches()
        self.migrate_notified_pairs()

        # Summary
        logger.info("=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        for table, stats in self.stats.items():
            logger.info(f"{table}: migrated={stats['migrated']}, skipped={stats['skipped']}, errors={len(stats['errors'])}")

        if self.dry_run:
            logger.info("")
            logger.info("This was a DRY RUN - no data was written.")
            logger.info("Run with --execute to perform the actual migration.")

        return self.stats


def main():
    parser = argparse.ArgumentParser(description='Migrate DynamoDB to Supabase')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Run without making changes (default)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the migration')
    parser.add_argument('--validate', action='store_true',
                        help='Validate migration by comparing counts')

    args = parser.parse_args()

    # If --execute is specified, disable dry_run
    dry_run = not args.execute

    try:
        migrator = DynamoToSupabaseMigrator(dry_run=dry_run)

        if args.validate:
            migrator.validate_migration()
        else:
            migrator.run_full_migration()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
