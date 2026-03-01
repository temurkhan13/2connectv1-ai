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
USE_HYBRID_MATCHING = os.getenv("USE_HYBRID_MATCHING", "false").lower() == "true"

# HYBRID = Multi-Vector base (6 dimensions) + Enhanced features (intent, bidirectional, etc.)
# When HYBRID is enabled, it combines the best of both systems


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
            # Priority: HYBRID > ENHANCED > MULTI_VECTOR > BASIC
            if USE_HYBRID_MATCHING:
                # HYBRID: Multi-vector base (6 dimensions) + Enhanced features
                # Combines the best of both systems
                matches_result = self._calculate_hybrid_matches(user_id, threshold)
                result["algorithm"] = "hybrid_full"
            elif USE_ENHANCED_MATCHING:
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

    def _calculate_hybrid_matches(
        self,
        user_id: str,
        threshold: float
    ) -> Dict[str, Any]:
        """
        HYBRID MATCHING: Combines Multi-Vector (6 dimensions) + Enhanced Features.

        This is the most comprehensive matching algorithm:

        MULTI-VECTOR BASE (6 dimensions with weights):
        - primary_goal (20% weight) - User's main objective
        - industry (25% weight) - Industry/sector focus
        - stage (20% weight) - Investment/company stage
        - geography (15% weight) - Geographic preferences
        - engagement_style (10% weight) - Communication preferences
        - dealbreakers (10% weight) - Hard exclusion criteria

        ENHANCED FEATURES (applied on top):
        - Intent Classification (INVESTOR↔FOUNDER, MENTOR↔MENTEE detection)
        - Bidirectional Scoring (geometric mean: both parties must benefit)
        - Dealbreaker Filtering (when ENFORCE_HARD_DEALBREAKERS=true)
        - Same-Objective Blocking (when BLOCK_SAME_OBJECTIVE=true)
        - Activity Boost (active users ranked higher)
        - Temporal Boost (new users get visibility for 14 days)

        Returns dict in same format as matching_service.find_and_store_user_matches()
        """
        import math
        from app.services.multi_vector_matcher import MultiVectorMatcher, MatchTier
        from app.services.enhanced_matching_service import (
            enhanced_matching_service,
            IntentClassifier,
            MatchIntent,
            INTENT_SCORING_CONFIGS
        )
        from app.adapters.dynamodb import UserProfile, UserMatches

        try:
            logger.info(f"[INLINE MATCH] Using HYBRID matching for user {user_id}")

            # STEP 1: Get multi-vector base scores (6 dimensions)
            matcher = MultiVectorMatcher()
            multi_vector_matches = matcher.find_multi_vector_matches(
                user_id=user_id,
                min_tier=MatchTier.LOW,  # Get all, we'll filter later
                limit=self.max_matches * 2  # Get more, we'll filter down
            )

            if not multi_vector_matches:
                logger.warning(f"[INLINE MATCH] Hybrid matching found 0 multi-vector candidates for {user_id}")
                return {
                    "success": True,
                    "total_matches": 0,
                    "requirements_matches": [],
                    "offerings_matches": [],
                    "algorithm": "hybrid_full"
                }

            # STEP 2: Get user's persona for enhanced features
            try:
                user_profile = UserProfile.get(user_id)
                user_persona = user_profile.persona
            except Exception as e:
                logger.warning(f"[INLINE MATCH] Could not load persona for {user_id}: {e}")
                user_persona = None

            # STEP 3: Classify user intent
            intent_classifier = IntentClassifier()
            if user_persona:
                user_intent, intent_confidence = intent_classifier.classify({
                    "what_theyre_looking_for": getattr(user_persona, "what_theyre_looking_for", ""),
                    "requirements": getattr(user_persona, "requirements", ""),
                    "offerings": getattr(user_persona, "offerings", ""),
                    "archetype": getattr(user_persona, "archetype", ""),
                    "focus": getattr(user_persona, "focus", "")
                })
            else:
                user_intent, intent_confidence = MatchIntent.GENERAL, 0.5

            # STEP 4: Get user's dealbreakers
            user_dealbreakers = []
            if user_persona:
                # Extract dealbreakers from persona
                dealbreaker_text = getattr(user_persona, "what_theyre_looking_for", "") or ""
                # Look for "not interested in", "avoid", "no" patterns
                import re
                patterns = [
                    r"not interested in ([^,\.]+)",
                    r"avoid ([^,\.]+)",
                    r"no ([^,\.]+) please"
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, dealbreaker_text.lower())
                    user_dealbreakers.extend(matches)

            # STEP 5: Apply enhanced features to each multi-vector match
            enhanced_matches = []
            blocked_count = 0
            dealbreaker_count = 0

            for mv_match in multi_vector_matches:
                candidate_id = mv_match.user_id
                base_score = mv_match.total_score  # Multi-vector weighted score

                # Skip if below base threshold
                if base_score < threshold * 0.7:  # Allow some below threshold for boosts
                    continue

                # Get candidate persona for intent matching
                try:
                    candidate_profile = UserProfile.get(candidate_id)
                    candidate_persona = candidate_profile.persona
                except Exception:
                    candidate_persona = None

                # 5a. Classify candidate intent
                if candidate_persona:
                    candidate_intent, _ = intent_classifier.classify({
                        "what_theyre_looking_for": getattr(candidate_persona, "what_theyre_looking_for", ""),
                        "requirements": getattr(candidate_persona, "requirements", ""),
                        "offerings": getattr(candidate_persona, "offerings", ""),
                        "archetype": getattr(candidate_persona, "archetype", ""),
                        "focus": getattr(candidate_persona, "focus", "")
                    })
                else:
                    candidate_intent = MatchIntent.GENERAL

                # 5b. Check intent complementarity
                intent_quality = self._get_intent_quality(user_intent, candidate_intent)

                # 5c. Check same-objective blocking
                if enhanced_matching_service.block_same_objective:
                    if self._should_block_same_objective(user_intent, candidate_intent):
                        blocked_count += 1
                        continue

                # 5d. Check dealbreakers
                if enhanced_matching_service.enforce_hard_dealbreakers and user_dealbreakers:
                    if candidate_persona:
                        candidate_text = " ".join([
                            str(getattr(candidate_persona, "focus", "") or ""),
                            str(getattr(candidate_persona, "archetype", "") or ""),
                        ]).lower()
                        violated = any(db in candidate_text for db in user_dealbreakers)
                        if violated:
                            dealbreaker_count += 1
                            continue

                # 5e. Calculate bidirectional adjustment
                # Use enhanced service's bidirectional calculation
                bidirectional_factor = 1.0
                try:
                    bidirectional_matches = enhanced_matching_service.find_bidirectional_matches(
                        user_id=user_id,
                        threshold=0.3,  # Low threshold, we're filtering ourselves
                        limit=1,
                        candidate_ids=[candidate_id],
                        include_explanations=False
                    )
                    if bidirectional_matches:
                        bm = bidirectional_matches[0]
                        # Geometric mean of forward and reverse
                        bidirectional_factor = math.sqrt(bm.forward_score * bm.reverse_score) / base_score
                        bidirectional_factor = max(0.5, min(1.5, bidirectional_factor))  # Cap adjustment
                except Exception:
                    pass  # Use default factor

                # 5f. Calculate activity boost
                activity_boost = enhanced_matching_service._calculate_activity_boost(candidate_id)

                # 5g. Calculate temporal boost
                temporal_boost = enhanced_matching_service._calculate_temporal_boost(candidate_id)

                # STEP 6: Calculate final hybrid score
                # Base (multi-vector) × intent × bidirectional × boosts
                final_score = (
                    base_score *
                    (0.7 + 0.3 * intent_quality) *  # Intent can boost by 30%
                    bidirectional_factor *
                    (0.9 + 0.1 * activity_boost) *  # Activity can boost by 10%
                    (0.95 + 0.05 * temporal_boost)   # Temporal can boost by 5%
                )

                # Only include if above threshold
                if final_score >= threshold:
                    match_data = {
                        "user_id": candidate_id,
                        "similarity_score": round(final_score, 4),
                        "match_type": "hybrid_full",
                        # Multi-vector components
                        "base_score": round(base_score, 4),
                        "tier": mv_match.tier.value,
                        "dimension_scores": [
                            {"dimension": ds.dimension, "score": round(ds.weighted_score, 4)}
                            for ds in mv_match.dimension_scores
                        ],
                        # Enhanced components
                        "intent_quality": round(intent_quality, 2),
                        "user_intent": user_intent.value,
                        "candidate_intent": candidate_intent.value,
                        "activity_boost": round(activity_boost, 2),
                        "temporal_boost": round(temporal_boost, 2),
                        "bidirectional_factor": round(bidirectional_factor, 2),
                        "explanation": self._generate_hybrid_explanation(
                            mv_match, user_intent, candidate_intent, intent_quality
                        )
                    }
                    enhanced_matches.append(match_data)

            # Sort by final score
            enhanced_matches.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Limit to max_matches
            enhanced_matches = enhanced_matches[:self.max_matches]

            # Store in DynamoDB
            matches_to_store = {
                "requirements_matches": enhanced_matches,
                "offerings_matches": enhanced_matches,  # Bidirectional, same list
                "algorithm": "hybrid_full",
                "threshold": threshold
            }
            UserMatches.store_user_matches(user_id, matches_to_store)

            logger.info(
                f"[INLINE MATCH] Hybrid matching complete for {user_id}: "
                f"{len(enhanced_matches)} matches (blocked: {blocked_count}, dealbreakers: {dealbreaker_count})"
            )

            return {
                "success": True,
                "total_matches": len(enhanced_matches),
                "requirements_matches": enhanced_matches,
                "offerings_matches": enhanced_matches,
                "algorithm": "hybrid_full",
                "stats": {
                    "candidates_evaluated": len(multi_vector_matches),
                    "blocked_same_objective": blocked_count,
                    "blocked_dealbreakers": dealbreaker_count
                }
            }

        except Exception as e:
            logger.error(f"[INLINE MATCH] Hybrid matching failed for {user_id}: {e}")
            # Fall back to enhanced matching
            logger.info(f"[INLINE MATCH] Falling back to enhanced matching for {user_id}")
            return self._calculate_enhanced_matches(user_id, threshold)

    def _get_intent_quality(self, user_intent: 'MatchIntent', candidate_intent: 'MatchIntent') -> float:
        """
        Calculate intent complementarity score (0-1).

        Perfect matches (1.0):
        - INVESTOR_FOUNDER ↔ FOUNDER_INVESTOR
        - MENTOR_MENTEE ↔ MENTEE_MENTOR
        - TALENT_SEEKING ↔ OPPORTUNITY_SEEKING

        Good matches (0.8):
        - COFOUNDER ↔ COFOUNDER
        - PARTNERSHIP ↔ PARTNERSHIP

        Neutral (0.5):
        - GENERAL ↔ anything

        Poor matches (0.3):
        - Same-seeking intents (INVESTOR_FOUNDER ↔ INVESTOR_FOUNDER)
        """
        from app.services.enhanced_matching_service import MatchIntent

        # Perfect complementary pairs
        complementary_pairs = {
            (MatchIntent.INVESTOR_FOUNDER, MatchIntent.FOUNDER_INVESTOR): 1.0,
            (MatchIntent.FOUNDER_INVESTOR, MatchIntent.INVESTOR_FOUNDER): 1.0,
            (MatchIntent.MENTOR_MENTEE, MatchIntent.MENTEE_MENTOR): 1.0,
            (MatchIntent.MENTEE_MENTOR, MatchIntent.MENTOR_MENTEE): 1.0,
            (MatchIntent.TALENT_SEEKING, MatchIntent.OPPORTUNITY_SEEKING): 1.0,
            (MatchIntent.OPPORTUNITY_SEEKING, MatchIntent.TALENT_SEEKING): 1.0,
        }

        # Same-type matches (both seeking the same thing)
        same_type_pairs = {
            (MatchIntent.COFOUNDER, MatchIntent.COFOUNDER): 0.9,
            (MatchIntent.PARTNERSHIP, MatchIntent.PARTNERSHIP): 0.85,
        }

        pair = (user_intent, candidate_intent)

        if pair in complementary_pairs:
            return complementary_pairs[pair]
        if pair in same_type_pairs:
            return same_type_pairs[pair]
        if user_intent == MatchIntent.GENERAL or candidate_intent == MatchIntent.GENERAL:
            return 0.5
        if user_intent == candidate_intent:
            return 0.3  # Both seeking same thing (e.g., both INVESTOR_FOUNDER)
        return 0.4  # Unrelated intents

    def _should_block_same_objective(self, user_intent: 'MatchIntent', candidate_intent: 'MatchIntent') -> bool:
        """
        Check if two users should be blocked from matching due to same objective.

        Blocks:
        - Two investors both looking for founders
        - Two founders both raising
        - Two job seekers
        - Two hiring companies
        """
        from app.services.enhanced_matching_service import MatchIntent

        blocked_same_pairs = [
            MatchIntent.INVESTOR_FOUNDER,  # Two investors
            MatchIntent.FOUNDER_INVESTOR,  # Two founders raising
            MatchIntent.OPPORTUNITY_SEEKING,  # Two job seekers
            MatchIntent.TALENT_SEEKING,  # Two hiring companies
            MatchIntent.MENTEE_MENTOR,  # Two mentees seeking mentors
        ]

        return user_intent == candidate_intent and user_intent in blocked_same_pairs

    def _generate_hybrid_explanation(
        self,
        mv_match: Any,
        user_intent: 'MatchIntent',
        candidate_intent: 'MatchIntent',
        intent_quality: float
    ) -> str:
        """Generate human-readable explanation for hybrid match."""
        from app.services.enhanced_matching_service import MatchIntent

        # Base explanation from tier
        tier_explanations = {
            "perfect": "Exceptional match across all dimensions",
            "strong": "Strong compatibility",
            "worth_exploring": "Worth exploring",
            "low": "Potential connection"
        }
        base = tier_explanations.get(mv_match.tier.value, "Match found")

        # Add intent context
        if intent_quality >= 0.9:
            if user_intent == MatchIntent.INVESTOR_FOUNDER and candidate_intent == MatchIntent.FOUNDER_INVESTOR:
                return f"{base} — Investor meets Founder seeking funding"
            elif user_intent == MatchIntent.FOUNDER_INVESTOR and candidate_intent == MatchIntent.INVESTOR_FOUNDER:
                return f"{base} — Founder meets active Investor"
            elif user_intent == MatchIntent.MENTOR_MENTEE and candidate_intent == MatchIntent.MENTEE_MENTOR:
                return f"{base} — Mentor meets eager Mentee"
            elif user_intent == MatchIntent.MENTEE_MENTOR and candidate_intent == MatchIntent.MENTOR_MENTEE:
                return f"{base} — Mentee meets experienced Mentor"
            else:
                return f"{base} — Complementary objectives"
        elif intent_quality >= 0.7:
            return f"{base} — Good alignment on goals"
        else:
            return base


# Singleton instance
inline_matching_service = InlineMatchingService()
