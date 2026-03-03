"""
Backfill missing personas for users who completed onboarding before slot persistence fix.

Target users (from admin dashboard):
- jaredwbonham@gmail.com
- jeishukran@gmail.com
- jose@2connect.ai
- shane@2connect.ai
- ryan@stbl.io

All have 0/11 slots but completed onboarding (conversation exists in DynamoDB).

This script:
1. Fetches conversation history from DynamoDB
2. Re-runs slot extraction (replays conversation)
3. Persists slots to Supabase (now that bug is fixed)
4. Generates persona + embeddings
5. Triggers matching
6. Creates user summary for AI Summary page

Usage:
    # Dry-run (no changes)
    python scripts/backfill_missing_personas.py --dry-run

    # Run for real
    python scripts/backfill_missing_personas.py

    # Single user test
    python scripts/backfill_missing_personas.py --user jaredwbonham@gmail.com
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv

from app.services.slot_extraction.context_manager import ContextManager
from app.services.slot_extraction.llm_slot_extractor import LLMSlotExtractor
from app.services.profile_generation_service import ProfileGenerationService
from app.services.inline_matching_service import InlineMatchingService
from app.services.match_sync_service import MatchSyncService
from app.adapters.supabase_adapter import SupabaseAdapter
from app.adapters.postgresql_adapter import PostgreSQLAdapter

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target users (from admin dashboard screenshot)
TARGET_USERS = [
    "jaredwbonham@gmail.com",
    "jeishukran@gmail.com",
    "jose@2connect.ai",
    "shane@2connect.ai",
    "ryan@stbl.io"
]


class PersonaBackfiller:
    """Backfill missing personas for users with 0/11 slots."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.supabase = SupabaseAdapter()
        self.postgresql = PostgreSQLAdapter()

        # Initialize DynamoDB
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.profiles_table = self.dynamodb.Table('reciprocity_user_profiles')

        # Services (lazy init to avoid unnecessary API calls in dry-run)
        self._extractor = None
        self._profile_service = None
        self._matching_service = None
        self._match_sync = None

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = LLMSlotExtractor()
        return self._extractor

    @property
    def profile_service(self):
        if self._profile_service is None:
            self._profile_service = ProfileGenerationService()
        return self._profile_service

    @property
    def matching_service(self):
        if self._matching_service is None:
            self._matching_service = InlineMatchingService()
        return self._matching_service

    @property
    def match_sync(self):
        if self._match_sync is None:
            self._match_sync = MatchSyncService()
        return self._match_sync

    def get_user_id_by_email(self, email: str) -> Optional[str]:
        """Get user_id from PostgreSQL by email."""
        try:
            query = "SELECT id FROM user_profiles WHERE email = %s LIMIT 1"
            result = self.postgresql.execute_query(query, (email,))
            if result:
                return result[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error fetching user_id for {email}: {e}")
            return None

    def fetch_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """Fetch conversation history from DynamoDB."""
        try:
            response = self.profiles_table.get_item(Key={'user_id': user_id})
            profile = response.get('Item', {})
            messages = profile.get('message_history', [])

            logger.info(f"Fetched {len(messages)} messages for user {user_id}")
            return messages
        except Exception as e:
            logger.error(f"Error fetching conversation for {user_id}: {e}")
            return []

    def replay_conversation_and_extract_slots(
        self,
        user_id: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Replay conversation through slot extraction pipeline.

        Returns:
            Dict of extracted slots with confidence scores
        """
        logger.info(f"Replaying {len(messages)} messages for slot extraction...")

        # Initialize context manager (same as in onboarding flow)
        context_manager = ContextManager(user_id=user_id)

        # Replay conversation message by message
        for i, message in enumerate(messages):
            if message.get('role') == 'user':
                user_text = message.get('content', '')
                logger.debug(f"Message {i+1}: User said: {user_text[:100]}...")

                # Add to context
                context_manager.add_user_message(user_text)

                # Extract slots from accumulated context
                # (This is what happens in conversational_onboarding endpoint)
                extracted = self.extractor.extract_slots(
                    conversation_history=context_manager.get_messages(),
                    current_slots=context_manager.get_current_slots()
                )

                # Update context manager with extracted slots
                for slot_name, slot_data in extracted.items():
                    if slot_data.get('value') is not None:
                        context_manager.update_slot(
                            slot_name=slot_name,
                            value=slot_data['value'],
                            confidence=slot_data.get('confidence', 0.0)
                        )
                        logger.debug(f"  Extracted: {slot_name} = {slot_data['value']}")

        # Get final slots after replaying all messages
        final_slots = context_manager.get_current_slots()
        slot_count = len([s for s in final_slots.values() if s.get('value') is not None])

        logger.info(f"Extraction complete: {slot_count}/11 slots extracted")
        return final_slots

    def verify_slots_persisted(self, user_id: str) -> int:
        """Verify slots were saved to Supabase."""
        try:
            saved_slots = self.supabase.get_user_onboarding_answers(user_id)
            return len(saved_slots)
        except Exception as e:
            logger.error(f"Error verifying slots for {user_id}: {e}")
            return 0

    def generate_persona(self, user_id: str, slots: Dict[str, Any]) -> Optional[Dict]:
        """Generate persona from slots."""
        try:
            persona = self.profile_service.generate_profile_from_slots(
                user_id=user_id,
                slots=slots
            )
            logger.info(f"Generated persona: {persona.get('profile_type', 'unknown')} profile")
            return persona
        except Exception as e:
            logger.error(f"Error generating persona for {user_id}: {e}")
            return None

    def generate_embeddings(self, user_id: str, persona: Dict) -> bool:
        """Generate embeddings for matching."""
        try:
            embeddings = self.matching_service.generate_embeddings(persona)
            logger.info("Generated 6-dimension embeddings for matching")
            return True
        except Exception as e:
            logger.error(f"Error generating embeddings for {user_id}: {e}")
            return False

    def trigger_matching(self, user_id: str, persona: Dict) -> int:
        """Find and sync matches."""
        try:
            # Find matches
            matches = self.matching_service.find_matches_for_user(
                user_id=user_id,
                persona=persona,
                top_k=20
            )

            if matches:
                # Sync to backend PostgreSQL
                self.match_sync.sync_matches_to_backend(
                    user_id=user_id,
                    matches=matches
                )
                logger.info(f"Found and synced {len(matches)} matches")
                return len(matches)
            else:
                logger.info("No matches found")
                return 0
        except Exception as e:
            logger.error(f"Error triggering matching for {user_id}: {e}")
            return 0

    def create_user_summary(self, user_id: str, slots: Dict[str, Any]) -> bool:
        """Create markdown summary for AI Summary page."""
        try:
            # Extract values from slots (same as onboarding.py lines 551-567)
            profile_type = slots.get("user_type", {}).get("value", "User")
            industry = slots.get("industry_focus", {}).get("value", "Not specified")
            goal = slots.get("primary_goal", {}).get("value", "Not specified")
            stage = slots.get("stage_preference", {}).get("value", "Not specified")
            geography = slots.get("geographic_focus", {}).get("value", "Not specified")
            offerings = slots.get("offerings", {}).get("value", "Not specified")
            requirements = slots.get("requirements", {}).get("value", "Not specified")

            # Generate markdown (same as onboarding.py lines 570-593)
            summary_markdown = f"""# {profile_type} Profile

## Primary Goal
{goal}

## Industry Focus
{industry}

## Stage/Experience
{stage}

## Geography
{geography}

## What I Can Offer
{offerings}

## What I'm Looking For
{requirements}

---

*This summary was generated based on your onboarding responses. You can update it anytime from your profile settings.*
"""

            # Save to PostgreSQL
            summary_id = self.postgresql.create_user_summary(
                user_id=user_id,
                summary=summary_markdown,
                status='draft',
                urgency='ongoing'
            )

            if summary_id:
                logger.info(f"Created user_summary {summary_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error creating user summary for {user_id}: {e}")
            return False

    def backfill_user(self, email: str) -> Dict[str, Any]:
        """
        Backfill a single user's missing data.

        Returns:
            Dict with status and metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {email}")
        logger.info(f"{'='*60}")

        result = {
            "email": email,
            "user_id": None,
            "status": "pending",
            "slots_extracted": 0,
            "slots_persisted": 0,
            "persona_generated": False,
            "embeddings_generated": False,
            "matches_found": 0,
            "summary_created": False,
            "error": None
        }

        try:
            # Step 1: Get user_id
            user_id = self.get_user_id_by_email(email)
            if not user_id:
                result["status"] = "user_not_found"
                result["error"] = f"No user found with email {email}"
                logger.error(result["error"])
                return result

            result["user_id"] = user_id
            logger.info(f"Found user_id: {user_id}")

            # Step 2: Fetch conversation history
            messages = self.fetch_conversation_history(user_id)
            if not messages:
                result["status"] = "no_messages"
                result["error"] = "No conversation history found in DynamoDB"
                logger.error(result["error"])
                return result

            logger.info(f"Found {len(messages)} messages in conversation")

            if self.dry_run:
                logger.info("[DRY-RUN] Would replay conversation and extract slots")
                result["status"] = "dry_run_complete"
                return result

            # Step 3: Re-extract slots by replaying conversation
            logger.info("Re-extracting slots from conversation...")
            slots = self.replay_conversation_and_extract_slots(user_id, messages)
            result["slots_extracted"] = len([s for s in slots.values() if s.get('value') is not None])

            # Step 4: Verify slots were persisted
            result["slots_persisted"] = self.verify_slots_persisted(user_id)
            logger.info(f"Verified: {result['slots_persisted']}/11 slots persisted to Supabase")

            if result["slots_persisted"] == 0:
                result["status"] = "slots_not_persisted"
                result["error"] = "Slots extracted but not persisted to Supabase"
                logger.error(result["error"])
                return result

            # Step 5: Generate persona
            logger.info("Generating persona...")
            persona = self.generate_persona(user_id, slots)
            if persona:
                result["persona_generated"] = True
            else:
                result["status"] = "persona_failed"
                result["error"] = "Persona generation failed"
                logger.error(result["error"])
                return result

            # Step 6: Generate embeddings
            logger.info("Generating embeddings...")
            result["embeddings_generated"] = self.generate_embeddings(user_id, persona)

            # Step 7: Trigger matching
            logger.info("Finding matches...")
            result["matches_found"] = self.trigger_matching(user_id, persona)

            # Step 8: Create user summary
            logger.info("Creating user summary...")
            result["summary_created"] = self.create_user_summary(user_id, slots)

            # Success!
            result["status"] = "success"
            logger.info(f"✅ User {email} successfully backfilled!")
            logger.info(f"   Slots: {result['slots_persisted']}/11")
            logger.info(f"   Matches: {result['matches_found']}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ Error processing {email}: {e}", exc_info=True)

        return result

    def backfill_all_users(self, target_emails: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Backfill all target users.

        Args:
            target_emails: List of emails to process (defaults to TARGET_USERS)

        Returns:
            Summary of results
        """
        emails = target_emails or TARGET_USERS

        logger.info(f"\n{'='*60}")
        logger.info(f"BACKFILL MIGRATION - {datetime.now().isoformat()}")
        logger.info(f"Mode: {'DRY-RUN' if self.dry_run else 'PRODUCTION'}")
        logger.info(f"Target users: {len(emails)}")
        logger.info(f"{'='*60}\n")

        results = []
        for email in emails:
            result = self.backfill_user(email)
            results.append(result)

        # Generate summary report
        summary = {
            "total_users": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] not in ["success", "dry_run_complete"]]),
            "dry_run": self.dry_run,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("MIGRATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total users: {summary['total_users']}")
        logger.info(f"✅ Successful: {summary['successful']}")
        logger.info(f"❌ Failed: {summary['failed']}")

        if summary['successful'] > 0:
            logger.info(f"\nSuccessfully backfilled:")
            for r in results:
                if r["status"] == "success":
                    logger.info(f"  • {r['email']} - {r['slots_persisted']}/11 slots, {r['matches_found']} matches")

        if summary['failed'] > 0:
            logger.info(f"\nFailed:")
            for r in results:
                if r["status"] not in ["success", "dry_run_complete"]:
                    logger.info(f"  • {r['email']} - {r['status']}: {r['error']}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing personas for users with 0/11 slots"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode (no changes made)'
    )
    parser.add_argument(
        '--user',
        type=str,
        help='Process single user by email'
    )

    args = parser.parse_args()

    # Initialize backfiller
    backfiller = PersonaBackfiller(dry_run=args.dry_run)

    # Process single user or all target users
    if args.user:
        result = backfiller.backfill_user(args.user)
        return 0 if result["status"] == "success" else 1
    else:
        summary = backfiller.backfill_all_users()
        return 0 if summary["successful"] == summary["total_users"] else 1


if __name__ == "__main__":
    sys.exit(main())
