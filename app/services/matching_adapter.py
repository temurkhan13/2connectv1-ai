"""
Adapter to use enhanced_matching_service with legacy interface.
Converts between BidirectionalMatch and legacy Dict format.

This adapter allows gradual migration from legacy matching to enhanced
bidirectional matching with intent classification.
"""
import logging
from typing import Dict, List, Any

from app.services.enhanced_matching_service import enhanced_matching_service
from app.adapters.dynamodb import UserMatches

logger = logging.getLogger(__name__)


class EnhancedMatchingAdapter:
    """Wraps enhanced matching with legacy-compatible interface."""

    def find_and_store_user_matches(self, user_id: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Find matches using enhanced service, return legacy format.

        This method provides backward compatibility with the legacy matching
        interface while using the enhanced bidirectional matching under the hood.

        Args:
            user_id: User ID to find matches for
            threshold: Minimum match score threshold (default 0.6)

        Returns:
            Dict with legacy format: success, user_id, total_matches,
            requirements_matches, offerings_matches, stored
        """
        try:
            logger.info(f"[ENHANCED] Finding bidirectional matches for user {user_id}")

            # Use enhanced bidirectional matching
            matches = enhanced_matching_service.find_bidirectional_matches(
                user_id=user_id,
                threshold=threshold,
                limit=50,
                include_explanations=True
            )

            if not matches:
                logger.info(f"[ENHANCED] No matches found for user {user_id}")
                return {
                    'success': True,
                    'user_id': user_id,
                    'total_matches': 0,
                    'requirements_matches': [],
                    'offerings_matches': [],
                    'stored': True
                }

            # Format for API
            formatted = enhanced_matching_service.format_matches_for_api(matches)

            logger.info(f"[ENHANCED] Found {len(formatted)} matches for user {user_id}")

            # Convert to legacy format for storage compatibility
            legacy_matches = []
            for m in formatted:
                scores = m.get('scores', {})
                explanation = m.get('explanation', {})
                match_reasons = explanation.get('match_reasons', [])

                legacy_matches.append({
                    'user_id': m['user_id'],
                    'similarity_score': scores.get('final', 0.0),
                    'match_type': 'enhanced_bidirectional',
                    'forward_score': scores.get('forward', 0.0),
                    'reverse_score': scores.get('reverse', 0.0),
                    'intent_quality': scores.get('intent_match_quality', 1.0),
                    'activity_boost': scores.get('activity_boost', 1.0),
                    'explanation': match_reasons[0] if match_reasons else ''
                })

            # Store in DynamoDB using existing UserMatches model
            try:
                stored = UserMatches.store_user_matches(user_id, {
                    'requirements_matches': legacy_matches,
                    'offerings_matches': [],  # Enhanced doesn't distinguish
                    'algorithm': 'enhanced_v1',
                    'total_matches': len(legacy_matches)
                })
            except Exception as e:
                logger.warning(f"[ENHANCED] Failed to store matches in DynamoDB: {e}")
                stored = False

            return {
                'success': True,
                'user_id': user_id,
                'total_matches': len(legacy_matches),
                'requirements_matches': legacy_matches,
                'offerings_matches': [],
                'stored': stored
            }

        except Exception as e:
            logger.error(f"[ENHANCED] Matching failed for {user_id}: {e}", exc_info=True)
            # Return failure result instead of raising to allow graceful degradation
            return {
                'success': False,
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'stored': False,
                'error': str(e)
            }


# Singleton instance
enhanced_matching_adapter = EnhancedMatchingAdapter()
