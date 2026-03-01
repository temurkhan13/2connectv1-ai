"""
Inline Bidirectional Matching Service.

PURPOSE:
--------
Provides IMMEDIATE match calculation when a user completes onboarding.
This replaces the old pattern of waiting 4 hours for the cron job.

ARCHITECTURE DECISION (March 2026):
-----------------------------------
The cron job (`scheduled_matching.py`) remains for:
- Recalculating matches for existing users when embeddings change
- Batch processing for platform-wide updates

This inline service handles:
- Immediate match calculation for NEW users completing onboarding
- Bidirectional updates (adds new user to existing users' match lists)
- Real-time sync to backend

FLOW:
-----
1. User completes onboarding
2. Embeddings are stored
3. `calculate_and_sync_matches_bidirectional()` is called
4. New user's matches are calculated and synced
5. Existing matched users' lists are updated to include new user
6. Matches are synced to backend immediately

DEPRECATION NOTE:
-----------------
The old approach of relying solely on the 4-hour cron job for new users
is deprecated. New users should always get immediate matches via this service.
The cron job is now only for maintenance/recalculation.
"""
import os
import uuid
import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger(__name__)


class InlineMatchingService:
    """
    Service for immediate match calculation on onboarding completion.

    Unlike the scheduled matching worker which batches users, this service
    processes a single user immediately and handles bidirectional updates.
    """

    def __init__(self):
        # BUG FIX: Increased default from 0.3 to 0.5 for more meaningful matches
        # 0.3 was too permissive, causing everyone to match with everyone
        self.threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
        self.max_matches = int(os.getenv("MAX_INLINE_MATCHES", "50"))

    def calculate_and_sync_matches_bidirectional(
        self,
        user_id: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate matches for a new user and update bidirectional relationships.

        This is the main entry point called from onboarding completion.

        Steps:
        1. Calculate matches for the new user (who they should see)
        2. Update existing users' match lists to include this new user
        3. Sync all affected matches to backend

        Args:
            user_id: The new user's ID
            threshold: Minimum similarity score (default from env)

        Returns:
            Dict with success status, match counts, and sync results
        """
        from app.services.matching_service import matching_service
        from app.services.match_sync_service import match_sync_service
        from app.adapters.dynamodb import UserMatches, NotifiedMatchPairs

        if threshold is None:
            threshold = self.threshold

        result = {
            "success": False,
            "user_id": user_id,
            "new_user_matches": 0,
            "reciprocal_updates": 0,
            "backend_synced": False,
            "errors": []
        }

        try:
            logger.info(f"[INLINE MATCH] Starting bidirectional matching for user {user_id}")

            # STEP 1: Calculate matches for the new user
            # This finds who the new user should see in their Discover page
            # Pass the threshold to ensure consistent filtering
            matches_result = matching_service.find_and_store_user_matches(user_id, threshold=threshold)

            if not matches_result.get("success"):
                result["errors"].append(f"Failed to calculate matches: {matches_result.get('message')}")
                return result

            total_matches = matches_result.get("total_matches", 0)
            result["new_user_matches"] = total_matches
            logger.info(f"[INLINE MATCH] Found {total_matches} matches for new user {user_id}")

            # STEP 2: Sync new user's matches to backend
            try:
                sync_result = match_sync_service.sync_matches_to_backend(user_id, matches_result)
                if sync_result.get("success"):
                    result["backend_synced"] = True
                    logger.info(f"[INLINE MATCH] Synced {sync_result.get('count', 0)} matches to backend for user {user_id}")
                else:
                    result["errors"].append(f"Backend sync failed: {sync_result.get('error')}")
            except Exception as e:
                result["errors"].append(f"Backend sync error: {str(e)}")
                logger.warning(f"[INLINE MATCH] Backend sync failed for {user_id}: {e}")

            # STEP 3: Bidirectional updates - add new user to existing users' match lists
            # This ensures existing users can discover the new user
            reciprocal_count = self._update_reciprocal_matches(
                user_id=user_id,
                matches=matches_result,
                threshold=threshold
            )
            result["reciprocal_updates"] = reciprocal_count

            # STEP 4: Sync affected users' matches to backend
            # We need to sync matches for users whose lists were updated
            affected_users = self._get_affected_users(matches_result)
            synced_count = 0

            for affected_user_id in affected_users[:10]:  # Limit to avoid overwhelming backend
                try:
                    stored_matches = UserMatches.get_user_matches(affected_user_id)
                    if stored_matches:
                        sync_result = match_sync_service.sync_matches_to_backend(
                            affected_user_id,
                            stored_matches
                        )
                        if sync_result.get("success"):
                            synced_count += 1
                except Exception as e:
                    logger.warning(f"[INLINE MATCH] Failed to sync affected user {affected_user_id}: {e}")

            logger.info(f"[INLINE MATCH] Synced {synced_count} affected users to backend")

            result["success"] = True
            result["message"] = (
                f"Bidirectional matching complete: "
                f"{total_matches} matches for new user, "
                f"{reciprocal_count} reciprocal updates, "
                f"{synced_count} affected users synced"
            )

            logger.info(f"[INLINE MATCH] Complete for {user_id}: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"[INLINE MATCH] Error for user {user_id}: {str(e)}")
            result["errors"].append(str(e))
            return result

    def _update_reciprocal_matches(
        self,
        user_id: str,
        matches: Dict[str, Any],
        threshold: float
    ) -> int:
        """
        Update existing users' match lists to include the new user.

        When User A joins and matches with User B:
        - User A already has User B in their matches (from step 1)
        - This method adds User A to User B's matches

        This is the key difference from the old approach which only
        calculated one-way matches.
        """
        from app.adapters.dynamodb import UserMatches, UserProfile

        updated_count = 0
        processed_users = set()

        # Process requirements matches (new user needs X, matched user offers X)
        for match in matches.get("requirements_matches", []):
            matched_user_id = match.get("user_id")
            if not matched_user_id or matched_user_id in processed_users:
                continue

            similarity_score = match.get("similarity_score", 0.0)
            if similarity_score < threshold:
                continue

            try:
                # Get matched user's current matches
                stored = UserMatches.get_user_matches(matched_user_id) or {
                    "requirements_matches": [],
                    "offerings_matches": []
                }

                # Check if new user already exists in their list
                already_exists = any(
                    m.get("user_id") == user_id
                    for m in stored.get("offerings_matches", [])
                )

                if not already_exists:
                    # Add new user to matched user's offerings_matches
                    # (matched user offers what new user needs)
                    offerings_matches = stored.get("offerings_matches", [])
                    offerings_matches.append({
                        "user_id": user_id,
                        "similarity_score": similarity_score,
                        "match_type": "offerings_to_requirements",
                        "explanation": f"This user needs what you offer (score: {similarity_score:.2f})"
                    })

                    stored["offerings_matches"] = offerings_matches
                    UserMatches.store_user_matches(matched_user_id, stored)
                    updated_count += 1
                    logger.info(f"[INLINE MATCH] Added {user_id} to {matched_user_id}'s offerings_matches")

                processed_users.add(matched_user_id)

            except Exception as e:
                logger.warning(f"[INLINE MATCH] Failed to update reciprocal for {matched_user_id}: {e}")

        # Process offerings matches (new user offers X, matched user needs X)
        for match in matches.get("offerings_matches", []):
            matched_user_id = match.get("user_id")
            if not matched_user_id or matched_user_id in processed_users:
                continue

            similarity_score = match.get("similarity_score", 0.0)
            if similarity_score < threshold:
                continue

            try:
                # Get matched user's current matches
                stored = UserMatches.get_user_matches(matched_user_id) or {
                    "requirements_matches": [],
                    "offerings_matches": []
                }

                # Check if new user already exists in their list
                already_exists = any(
                    m.get("user_id") == user_id
                    for m in stored.get("requirements_matches", [])
                )

                if not already_exists:
                    # Add new user to matched user's requirements_matches
                    # (matched user needs what new user offers)
                    requirements_matches = stored.get("requirements_matches", [])
                    requirements_matches.append({
                        "user_id": user_id,
                        "similarity_score": similarity_score,
                        "match_type": "requirements_to_offerings",
                        "explanation": f"This user offers what you need (score: {similarity_score:.2f})"
                    })

                    stored["requirements_matches"] = requirements_matches
                    UserMatches.store_user_matches(matched_user_id, stored)
                    updated_count += 1
                    logger.info(f"[INLINE MATCH] Added {user_id} to {matched_user_id}'s requirements_matches")

                processed_users.add(matched_user_id)

            except Exception as e:
                logger.warning(f"[INLINE MATCH] Failed to update reciprocal for {matched_user_id}: {e}")

        return updated_count

    def _get_affected_users(self, matches: Dict[str, Any]) -> List[str]:
        """Get list of user IDs whose match lists were potentially updated."""
        affected = set()

        for match in matches.get("requirements_matches", []):
            if match.get("user_id"):
                affected.add(match["user_id"])

        for match in matches.get("offerings_matches", []):
            if match.get("user_id"):
                affected.add(match["user_id"])

        return list(affected)


# Singleton instance
inline_matching_service = InlineMatchingService()
