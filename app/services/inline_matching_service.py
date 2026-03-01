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

ENHANCED MATCHING FEATURES (March 2026):
----------------------------------------
When USE_ENHANCED_MATCHING=true, this service uses:
- Intent Classification: Detects INVESTOR↔FOUNDER, MENTOR↔MENTEE pairs
- Dealbreaker Filtering: Excludes matches based on user's hard no's
- Same-Objective Blocking: Prevents INVESTOR↔INVESTOR, TALENT↔TALENT matches
- Activity Boost: Active users ranked higher
- Temporal Boost: New users get visibility boost for 14 days
- Bidirectional Scoring: Both parties must benefit (geometric mean)

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

# Feature flags for advanced matching
USE_ENHANCED_MATCHING = os.getenv("USE_ENHANCED_MATCHING", "false").lower() == "true"
USE_MULTI_VECTOR_MATCHING = os.getenv("USE_MULTI_VECTOR_MATCHING", "false").lower() == "true"


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

        ENHANCED MATCHING (when USE_ENHANCED_MATCHING=true):
        Uses intent classification, dealbreaker filtering, same-objective blocking,
        activity boost, and temporal boost for smarter matches.

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
            "algorithm": "basic",
            "errors": []
        }

        try:
            logger.info(f"[INLINE MATCH] Starting bidirectional matching for user {user_id}")

            # STEP 1: Calculate matches using the appropriate algorithm
            if USE_ENHANCED_MATCHING:
                # Use enhanced matching with intent classification, dealbreakers, etc.
                matches_result = self._calculate_enhanced_matches(user_id, threshold)
                result["algorithm"] = "enhanced_bidirectional"
            elif USE_MULTI_VECTOR_MATCHING:
                # Use multi-vector weighted matching (6 dimensions)
                matches_result = self._calculate_multi_vector_matches(user_id, threshold)
                result["algorithm"] = "multi_vector"
            else:
                # Use basic 2-vector matching
                matches_result = matching_service.find_and_store_user_matches(user_id, threshold=threshold)
                result["algorithm"] = "basic"

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

    def _calculate_enhanced_matches(
        self,
        user_id: str,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Calculate matches using the enhanced bidirectional matching service.

        Features enabled:
        - Intent Classification (INVESTOR↔FOUNDER, MENTOR↔MENTEE, etc.)
        - Dealbreaker Filtering (when ENFORCE_HARD_DEALBREAKERS=true)
        - Same-Objective Blocking (when BLOCK_SAME_OBJECTIVE=true)
        - Activity Boost (active users ranked higher)
        - Temporal Boost (new users get visibility)
        - Bidirectional Scoring (geometric mean of forward/reverse)

        Returns dict in same format as matching_service.find_and_store_user_matches()
        """
        from app.services.enhanced_matching_service import enhanced_matching_service
        from app.adapters.dynamodb import UserMatches

        try:
            logger.info(f"[INLINE MATCH] Using ENHANCED matching for user {user_id}")

            # Get enhanced bidirectional matches
            matches = enhanced_matching_service.find_bidirectional_matches(
                user_id=user_id,
                threshold=threshold,
                limit=self.max_matches,
                include_explanations=True
            )

            if not matches:
                logger.warning(f"[INLINE MATCH] Enhanced matching found 0 matches for {user_id}")
                return {
                    "success": True,
                    "total_matches": 0,
                    "requirements_matches": [],
                    "offerings_matches": [],
                    "algorithm": "enhanced_bidirectional"
                }

            # Convert BidirectionalMatch objects to legacy format for compatibility
            # All enhanced matches go into requirements_matches (they're bidirectional)
            requirements_matches = []
            offerings_matches = []

            for m in matches:
                match_data = {
                    "user_id": m.user_id,
                    "similarity_score": m.final_score,  # Use final score with all factors
                    "match_type": "enhanced_bidirectional",
                    "forward_score": m.forward_score,
                    "reverse_score": m.reverse_score,
                    "intent_quality": m.intent_match_quality,
                    "activity_boost": m.activity_boost,
                    "temporal_boost": m.temporal_boost,
                    "explanation": m.match_reasons[0] if m.match_reasons else "Strong bidirectional match"
                }

                # Put in both lists for reciprocal updates to work correctly
                # Forward = their offerings match my requirements
                # Reverse = my offerings match their requirements
                if m.forward_score >= threshold:
                    requirements_matches.append(match_data)
                if m.reverse_score >= threshold:
                    offerings_matches.append(match_data)

            # Store in DynamoDB
            matches_to_store = {
                "requirements_matches": requirements_matches,
                "offerings_matches": offerings_matches,
                "algorithm": "enhanced_bidirectional",
                "threshold": threshold
            }
            UserMatches.store_user_matches(user_id, matches_to_store)

            logger.info(
                f"[INLINE MATCH] Enhanced matching complete for {user_id}: "
                f"{len(requirements_matches)} requirements, {len(offerings_matches)} offerings"
            )

            return {
                "success": True,
                "total_matches": len(set(m["user_id"] for m in requirements_matches + offerings_matches)),
                "requirements_matches": requirements_matches,
                "offerings_matches": offerings_matches,
                "algorithm": "enhanced_bidirectional"
            }

        except Exception as e:
            logger.error(f"[INLINE MATCH] Enhanced matching failed for {user_id}: {e}")
            # Fall back to basic matching
            logger.info(f"[INLINE MATCH] Falling back to basic matching for {user_id}")
            from app.services.matching_service import matching_service
            return matching_service.find_and_store_user_matches(user_id, threshold=threshold)

    def _calculate_multi_vector_matches(
        self,
        user_id: str,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Calculate matches using multi-vector weighted matching (6 dimensions).

        Dimensions:
        - primary_goal (20% weight)
        - industry (25% weight)
        - stage (20% weight)
        - geography (15% weight)
        - engagement_style (10% weight)
        - dealbreakers (10% weight)

        Returns dict in same format as matching_service.find_and_store_user_matches()
        """
        from app.services.multi_vector_matcher import MultiVectorMatcher, MatchTier
        from app.adapters.dynamodb import UserMatches

        try:
            logger.info(f"[INLINE MATCH] Using MULTI-VECTOR matching for user {user_id}")

            matcher = MultiVectorMatcher()

            # Map threshold to tier: 0.7+ = STRONG, 0.55+ = WORTH_EXPLORING, else LOW
            if threshold >= 0.7:
                min_tier = MatchTier.STRONG
            elif threshold >= 0.55:
                min_tier = MatchTier.WORTH_EXPLORING
            else:
                min_tier = MatchTier.LOW

            # Find multi-vector matches
            matches = matcher.find_multi_vector_matches(
                user_id=user_id,
                min_tier=min_tier,
                limit=self.max_matches
            )

            if not matches:
                logger.warning(f"[INLINE MATCH] Multi-vector matching found 0 matches for {user_id}")
                return {
                    "success": True,
                    "total_matches": 0,
                    "requirements_matches": [],
                    "offerings_matches": [],
                    "algorithm": "multi_vector"
                }

            # Convert MultiVectorMatchResult objects to legacy format
            requirements_matches = []
            for m in matches:
                if m.tier != MatchTier.LOW:
                    match_data = {
                        "user_id": m.user_id,
                        "similarity_score": m.total_score,
                        "match_type": "multi_vector",
                        "tier": m.tier.value,
                        "dimension_scores": [
                            {"dimension": ds.dimension, "score": ds.weighted_score}
                            for ds in m.dimension_scores
                        ],
                        "explanation": m.explanation or f"Multi-vector {m.tier.value} match"
                    }
                    requirements_matches.append(match_data)

            # Store in DynamoDB
            matches_to_store = {
                "requirements_matches": requirements_matches,
                "offerings_matches": [],  # Multi-vector currently does requirements only
                "algorithm": "multi_vector",
                "threshold": threshold
            }
            UserMatches.store_user_matches(user_id, matches_to_store)

            logger.info(
                f"[INLINE MATCH] Multi-vector matching complete for {user_id}: "
                f"{len(requirements_matches)} matches"
            )

            return {
                "success": True,
                "total_matches": len(requirements_matches),
                "requirements_matches": requirements_matches,
                "offerings_matches": [],
                "algorithm": "multi_vector"
            }

        except Exception as e:
            logger.error(f"[INLINE MATCH] Multi-vector matching failed for {user_id}: {e}")
            # Fall back to basic matching
            logger.info(f"[INLINE MATCH] Falling back to basic matching for {user_id}")
            from app.services.matching_service import matching_service
            return matching_service.find_and_store_user_matches(user_id, threshold=threshold)


# Singleton instance
inline_matching_service = InlineMatchingService()
