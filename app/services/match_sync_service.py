"""
Match Sync Service.

Automatically syncs matches from AI service to backend database.
This ensures matches are persisted and visible in the frontend.
"""
import os
import uuid
import logging
import requests
from typing import Dict, Any, List, Set, Optional
from dotenv import load_dotenv

# Load .env file to ensure env vars are available
load_dotenv()

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv('RECIPROCITY_BACKEND_URL', 'http://localhost:3000')
BACKEND_API_KEY = os.getenv('BACKEND_API_KEY', 'dev-webhook-key')


def is_valid_uuid(val: str) -> bool:
    """Check if a string is a valid UUID (RFC 4122 compliant)."""
    try:
        parsed = uuid.UUID(str(val))
        # Filter out test/fake UUIDs that have invalid variant bits
        # These pass Python's uuid.UUID() but fail class-validator's @IsUUID()
        # Valid variant must be in [8, 9, A, B] (RFC 4122)
        # Also filter out obvious test patterns
        hex_str = str(parsed).replace('-', '')
        # Check for repeating digit patterns (test UUIDs)
        if len(set(hex_str)) <= 2:  # All same digit or just two different
            return False
        # Check variant bits (character 17 in the 32-char hex, position 16 zero-indexed)
        # For standard UUIDs, this should be 8, 9, a, or b
        variant_char = hex_str[16].lower()
        if variant_char not in ['8', '9', 'a', 'b']:
            return False
        return True
    except (ValueError, TypeError):
        return False


class MatchSyncService:
    """Service to sync matches from AI service to backend database."""

    def __init__(self):
        self.backend_url = BACKEND_URL
        self.api_key = BACKEND_API_KEY
        self._valid_user_ids_cache: Optional[Set[str]] = None
        self._cache_ttl = 300  # 5 minutes

    def get_valid_user_ids_from_backend(self) -> Set[str]:
        """Get set of valid user IDs from backend database."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                os.getenv('RECIPROCITY_BACKEND_DB_URL',
                         'postgresql://postgres:postgres@localhost:5432/reciprocity_db')
            )
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users')
            valid_ids = set(str(row[0]) for row in cursor.fetchall())
            cursor.close()
            conn.close()
            return valid_ids
        except Exception as e:
            logger.warning(f"Could not fetch valid user IDs: {e}")
            return set()

    def calculate_user_matches(self, user_id: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Calculate matches for a user using the matching service."""
        from app.services.matching_service import matching_service

        try:
            result = matching_service.get_all_user_matches(user_id, threshold)

            # Handle Pydantic model response
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.dict()
            return result
        except Exception as e:
            logger.error(f"Error calculating matches for {user_id}: {e}")
            return {'success': False, 'error': str(e)}

    def sync_matches_to_backend(
        self,
        user_id: str,
        matches: Dict[str, Any],
        valid_user_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Send matches to backend webhook to persist them."""
        batch_id = str(uuid.uuid4())

        # Validate source user ID
        if not is_valid_uuid(user_id):
            logger.error(f"Source user_id is not a valid UUID: {user_id}")
            return {'success': False, 'error': 'Invalid source user ID'}

        # Get valid user IDs from backend if not provided
        if valid_user_ids is None:
            valid_user_ids = self.get_valid_user_ids_from_backend()
            # Filter to only valid UUIDs
            valid_user_ids = {uid for uid in valid_user_ids if is_valid_uuid(uid)}
            logger.info(f"Found {len(valid_user_ids)} valid users in backend")

        # Convert matches to webhook format
        match_pairs = []
        seen_pairs = set()
        skipped_invalid = 0

        # Add requirements matches (what user needs vs others' offerings)
        for m in matches.get('requirements_matches', []):
            target_id = m.get('user_id')
            if not target_id:
                continue
            # Skip invalid UUIDs (test data, etc.)
            if not is_valid_uuid(target_id):
                skipped_invalid += 1
                continue
            # Skip users that don't exist in backend
            if target_id not in valid_user_ids:
                skipped_invalid += 1
                continue
            # Skip duplicates
            if target_id in seen_pairs:
                continue
            seen_pairs.add(target_id)
            match_pairs.append({
                'user_a_id': user_id,
                'user_b_id': target_id,
                'user_a_designation': '',  # Backend requires string, not null
                'user_b_designation': '',
            })

        # Add offerings matches (what user offers vs others' needs)
        for m in matches.get('offerings_matches', []):
            target_id = m.get('user_id')
            if not target_id:
                continue
            # Skip invalid UUIDs (test data, etc.)
            if not is_valid_uuid(target_id):
                skipped_invalid += 1
                continue
            # Skip users that don't exist in backend
            if target_id not in valid_user_ids:
                skipped_invalid += 1
                continue
            # Skip duplicates
            if target_id in seen_pairs:
                continue
            seen_pairs.add(target_id)
            match_pairs.append({
                'user_a_id': user_id,
                'user_b_id': target_id,
                'user_a_designation': '',
                'user_b_designation': '',
            })

        if skipped_invalid > 0:
            logger.info(f"Skipped {skipped_invalid} matches with users not in backend")

        if not match_pairs:
            logger.info(f"No matches to sync for user {user_id}")
            return {'success': True, 'message': 'No matches to sync', 'count': 0}

        # Send to backend webhook
        endpoint = f"{self.backend_url}/api/v1/webhooks/matches-ready"
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
        }
        payload = {
            'batch_id': batch_id,
            'matches': match_pairs,
        }

        logger.info(f"Sending {len(match_pairs)} matches to backend for user {user_id}")

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully synced {len(match_pairs)} matches for user {user_id}")
            return {'success': True, 'count': len(match_pairs), 'response': response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to sync matches: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {'success': False, 'error': str(e)}

    def sync_user_matches(self, user_id: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Calculate and sync matches for a user - the main entry point.

        This is called after onboarding completion to ensure the user's
        matches are immediately available in the frontend.

        Args:
            user_id: The user's ID
            threshold: Minimum similarity score for matches

        Returns:
            Dict with success status and match count
        """
        logger.info(f"Starting match sync for user {user_id}")

        # Calculate matches
        matches = self.calculate_user_matches(user_id, threshold)

        if not matches.get('success', True):
            logger.error(f"Failed to calculate matches for {user_id}")
            return {'success': False, 'error': 'Failed to calculate matches'}

        total = matches.get('total_matches', 0)
        logger.info(f"Found {total} potential matches for user {user_id}")

        if total == 0:
            return {'success': True, 'message': 'No matches found', 'count': 0}

        # Sync to backend
        return self.sync_matches_to_backend(user_id, matches)


# Singleton instance
match_sync_service = MatchSyncService()
