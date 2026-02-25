"""User matching service."""
import logging
from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel

from app.services.embedding_service import embedding_service
from app.services.match_explanation_service import match_explanation_service
from app.services.multi_vector_matcher import multi_vector_matcher, MatchTier
from app.adapters.postgresql import postgresql_adapter
from app.adapters.dynamodb import UserMatches, NotifiedMatchPairs


# Pydantic Models
class MatchResult(BaseModel):
    """Match result with explanation."""
    user_id: str
    similarity_score: float
    match_type: str
    explanation: Optional[str] = None


class UserMatchesResponse(BaseModel):
    """Response for user matches."""
    success: bool
    user_id: str
    total_matches: int
    requirements_matches: List[MatchResult]
    offerings_matches: List[MatchResult]
    message: str

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for handling user matching operations."""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.explanation_service = match_explanation_service
    
    
    def find_user_matches(self, user_id: str, similarity_threshold: Optional[float] = None, upper_threshold: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Find matches for a user using stored embeddings."""
        try:
            logger.info(f"Finding matches for user {user_id}")
            
            # Use custom threshold or service default
            threshold = similarity_threshold if similarity_threshold is not None else self.embedding_service.similarity_threshold
            
            # Get upper threshold from env if not provided (Point 2: Upper bound adjustment)
            if upper_threshold is None:
                upper_threshold_env = os.getenv("SIMILARITY_UPPER_THRESHOLD")
                if upper_threshold_env:
                    upper_threshold = float(upper_threshold_env)
                    logger.info(f"Using upper threshold: {upper_threshold}")
            
            # Get user's embeddings
            user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
            
            if not user_embeddings:
                logger.warning(f"No embeddings found for user {user_id}")
                return {'requirements_matches': [], 'offerings_matches': []}
            
            results = {
                'requirements_matches': [],
                'offerings_matches': []
            }
            
            # Find matches for user's requirements (against others' offerings)
            if user_embeddings.get('requirements'):
                req_vector = user_embeddings['requirements']['vector_data']
                req_matches = postgresql_adapter.find_similar_users(
                    query_vector=req_vector,
                    embedding_type='offerings',  # Search in others' offerings
                    threshold=threshold,
                    upper_threshold=upper_threshold,
                    exclude_user_id=user_id
                )
                results['requirements_matches'] = req_matches
            
            # Find matches for user's offerings (against others' requirements)
            if user_embeddings.get('offerings'):
                off_vector = user_embeddings['offerings']['vector_data']
                off_matches = postgresql_adapter.find_similar_users(
                    query_vector=off_vector,
                    embedding_type='requirements',  # Search in others' requirements
                    threshold=threshold,
                    upper_threshold=upper_threshold,
                    exclude_user_id=user_id
                )
                results['offerings_matches'] = off_matches
            
            total_matches = len(results['requirements_matches']) + len(results['offerings_matches'])
            logger.info(f"Found {total_matches} total matches for user {user_id} (threshold: {threshold}, upper: {upper_threshold})")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find matches for user {user_id}: {str(e)}")
            return {'requirements_matches': [], 'offerings_matches': []}
    
    def find_matches_with_threshold(self, user_id: str, similarity_threshold: float) -> Dict[str, Any]:
        """Find matches with custom similarity threshold."""
        return self.find_user_matches(user_id, similarity_threshold)
    
    def format_match_results(self, matches: List[Dict[str, Any]], match_type: str) -> List[MatchResult]:
        """Format raw match results for API response."""
        formatted_matches = []
        for match in matches:
            formatted_matches.append(MatchResult(
                user_id=match['user_id'],
                similarity_score=match['similarity_score'],
                match_type=match_type,
                explanation=self.explanation_service.generate_explanation(
                    match['similarity_score'], 
                    match_type
                )
            ))
        return formatted_matches
    
    def get_all_user_matches(self, user_id: str, similarity_threshold: float) -> UserMatchesResponse:
        """Get all matches for a user (requirements and offerings)."""
        try:
            logger.info(f"Finding all matches for user: {user_id}")
            
            # Find matches
            matches_result = self.find_matches_with_threshold(user_id, similarity_threshold)
            
            # Format results
            requirements_matches = self.format_match_results(
                matches_result['requirements_matches'], 
                "requirements_to_offerings"
            )
            
            offerings_matches = self.format_match_results(
                matches_result['offerings_matches'], 
                "offerings_to_requirements"
            )
            
            total_matches = len(requirements_matches) + len(offerings_matches)
            
            return UserMatchesResponse(
                success=True,
                user_id=user_id,
                total_matches=total_matches,
                requirements_matches=requirements_matches,
                offerings_matches=offerings_matches,
                message=f"Found {total_matches} matches using embedding system"
            )
            
        except Exception as e:
            logger.error(f"Error finding all matches for user {user_id}: {str(e)}")
            raise e
    
    def get_requirements_matches(self, user_id: str, similarity_threshold: float) -> Dict[str, Any]:
        """Get matches for user's requirements vs others' offerings."""
        try:
            logger.info(f"Finding requirements matches for user: {user_id}")
            
            # Find matches
            matches_result = self.find_matches_with_threshold(user_id, similarity_threshold)
            
            # Format requirements matches only
            requirements_matches = self.format_match_results(
                matches_result['requirements_matches'], 
                "requirements_to_offerings"
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "match_type": "requirements_to_offerings",
                "total_matches": len(requirements_matches),
                "matches": requirements_matches,
                "message": f"Found {len(requirements_matches)} users whose offerings match your requirements"
            }
            
        except Exception as e:
            logger.error(f"Error finding requirements matches for user {user_id}: {str(e)}")
            raise e
    
    def get_offerings_matches(self, user_id: str, similarity_threshold: float) -> Dict[str, Any]:
        """Get matches for user's offerings vs others' requirements."""
        try:
            logger.info(f"Finding offerings matches for user: {user_id}")
            
            # Find matches
            matches_result = self.find_matches_with_threshold(user_id, similarity_threshold)
            
            # Format offerings matches only
            offerings_matches = self.format_match_results(
                matches_result['offerings_matches'], 
                "offerings_to_requirements"
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "match_type": "offerings_to_requirements",
                "total_matches": len(offerings_matches),
                "matches": offerings_matches,
                "message": f"Found {len(offerings_matches)} users who need what you're offering"
            }
            
        except Exception as e:
            logger.error(f"Error finding offerings matches for user {user_id}: {str(e)}")
            raise e
    
    def get_matching_stats(self) -> Dict[str, Any]:
        """Get matching system statistics."""
        try:
            stats = self.embedding_service.get_stats()
            return {
                "success": True,
                "data": stats,
                "message": "Matching system statistics"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise e
    
    def find_and_store_user_matches(self, user_id: str) -> Dict[str, Any]:
        """Find matches and store them in DynamoDB."""
        try:
            logger.info(f"Finding and storing matches for user {user_id}")
            
            matches_result = self.find_user_matches(user_id, 0.0)
            
            # Format the result like other methods
            requirements_matches = self.format_match_results(
                matches_result['requirements_matches'], 
                "requirements_to_offerings"
            )
            
            offerings_matches = self.format_match_results(
                matches_result['offerings_matches'], 
                "offerings_to_requirements"
            )
            
            total_matches = len(requirements_matches) + len(offerings_matches)
            
            # Prepare formatted result
            formatted_result = {
                'success': True,
                'user_id': user_id,
                'total_matches': total_matches,
                'requirements_matches': [match.dict() for match in requirements_matches],
                'offerings_matches': [match.dict() for match in offerings_matches],
                'message': f"Found {total_matches} matches using embedding system"
            }
            
            # Store matches (even if 0 matches, to create/update the record)
            store_success = UserMatches.store_user_matches(user_id, formatted_result)
            formatted_result['stored'] = store_success
            
            if store_success:
                if total_matches > 0:
                    logger.info(f"Successfully stored {total_matches} matches for user {user_id}")
                else:
                    logger.info(f"Successfully stored record with 0 matches for user {user_id}")
            else:
                logger.error(f"Failed to store matches for user {user_id}")
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error finding and storing matches for user {user_id}: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'stored': False,
                'message': f"Error: {str(e)}"
            }

    def find_and_store_user_matches_multi_vector(
        self,
        user_id: str,
        min_tier: MatchTier = MatchTier.WORTH_EXPLORING,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Find matches using multi-vector weighted algorithm and store in DynamoDB.

        Uses 6-dimension weighted matching (primary_goal, industry, stage, geography,
        engagement_style, dealbreakers) instead of simple 2-vector similarity.

        Returns same format as find_and_store_user_matches for compatibility.
        """
        try:
            logger.info(f"Finding multi-vector matches for user {user_id}")

            # Get matches using multi-vector algorithm
            multi_results = multi_vector_matcher.find_multi_vector_matches(
                user_id=user_id,
                min_tier=min_tier,
                limit=limit
            )

            # Convert to requirements_matches format (multi-vector doesn't distinguish direction)
            # All matches are considered "requirements" matches since multi-vector evaluates
            # overall compatibility rather than needs vs offerings
            requirements_matches = []
            for result in multi_results:
                # Build explanation from dimension scores
                dimension_breakdown = "; ".join([
                    f"{ds.dimension}: {ds.similarity:.0%}"
                    for ds in result.dimension_scores
                    if ds.similarity > 0.3
                ])

                explanation = result.explanation or f"Match tier: {result.tier.value}"
                if dimension_breakdown:
                    explanation += f" ({dimension_breakdown})"

                requirements_matches.append({
                    'user_id': result.user_id,
                    'similarity_score': result.total_score,
                    'match_type': 'multi_vector',
                    'explanation': explanation,
                    # Include multi-vector specific data
                    'tier': result.tier.value,
                    'dimension_scores': {
                        ds.dimension: {
                            'score': ds.similarity,
                            'weight': ds.weight,
                            'weighted_score': ds.weighted_score
                        }
                        for ds in result.dimension_scores
                    }
                })

            total_matches = len(requirements_matches)

            # Prepare formatted result (compatible with existing code)
            formatted_result = {
                'success': True,
                'user_id': user_id,
                'total_matches': total_matches,
                'requirements_matches': requirements_matches,
                'offerings_matches': [],  # Multi-vector doesn't use this distinction
                'match_algorithm': 'multi_vector',  # Flag for tracking
                'message': f"Found {total_matches} matches using multi-vector algorithm"
            }

            # Store matches
            store_success = UserMatches.store_user_matches(user_id, formatted_result)
            formatted_result['stored'] = store_success

            if store_success:
                if total_matches > 0:
                    logger.info(f"Stored {total_matches} multi-vector matches for user {user_id}")
                else:
                    logger.info(f"Stored record with 0 multi-vector matches for user {user_id}")
            else:
                logger.error(f"Failed to store multi-vector matches for user {user_id}")

            return formatted_result

        except Exception as e:
            logger.error(f"Error in multi-vector matching for user {user_id}: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'stored': False,
                'match_algorithm': 'multi_vector',
                'message': f"Error: {str(e)}"
            }

    def get_stored_matches(self, user_id: str) -> Dict[str, Any]:
        """Get stored matches from DynamoDB."""
        try:
            logger.info(f"Retrieving stored matches for user {user_id}")
            matches = UserMatches.get_user_matches(user_id)
            
            if matches:
                return {
                    'success': True,
                    **matches
                }
            else:
                return {
                    'success': False,
                    'user_id': user_id,
                    'total_matches': 0,
                    'requirements_matches': [],
                    'offerings_matches': [],
                    'message': 'No stored matches found'
                }
        except Exception as e:
            logger.error(f"Error retrieving stored matches for user {user_id}: {str(e)}")
            return {
                'success': False,
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'message': f"Error: {str(e)}"
            }
    
    def update_reciprocal_matches(self, source_user_id: str, source_matches: Dict[str, Any], skip_already_notified: bool = True) -> List[Dict[str, Any]]:
        """Update reciprocal matches when a user is processed. Skips already-notified pairs."""
        try:
            from app.adapters.dynamodb import UserProfile
            
            logger.info(f"Updating reciprocal matches for source user {source_user_id}")
            
            # Get source user's designation
            try:
                source_profile = UserProfile.get(source_user_id)
                source_designation = source_profile.persona.designation if source_profile.persona else ""
            except Exception:
                source_designation = ""
            
            match_pairs = []
            updated_users = []
            skipped_already_notified = 0
            skipped_over_limit = 0
            notify_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
            
            # Per-user match limit (over-matching prevention)
            max_matches_per_cycle = int(os.getenv("MAX_MATCHES_PER_CYCLE", "10"))
            
            # Process requirements matches (source needs something, matched user offers it)
            for req_match in source_matches.get('requirements_matches', []):
                matched_user_id = req_match['user_id']
                similarity_score = req_match.get('similarity_score', 0.0)
                
                # VALIDATION 1: Check if pair already notified before (duplicate prevention)
                if skip_already_notified and NotifiedMatchPairs.is_pair_notified(source_user_id, matched_user_id):
                    logger.info(f"Skipping already notified pair: {source_user_id} <-> {matched_user_id}")
                    skipped_already_notified += 1
                    continue
                
                # VALIDATION 2: Check match limit for this cycle (over-matching prevention)
                if len(match_pairs) >= max_matches_per_cycle:
                    logger.info(f"Match limit reached ({max_matches_per_cycle}), skipping remaining matches")
                    skipped_over_limit += 1
                    continue
                
                # Get matched user's stored matches
                stored = UserMatches.get_user_matches(matched_user_id)
                if stored is None:
                    stored = {
                        'requirements_matches': [],
                        'offerings_matches': []
                    }
                
                # Add source_user to matched user's offerings_matches (reciprocal)
                # Source has requirements, so matched user's offerings matched with source's requirements
                new_offering_match = {
                    'user_id': source_user_id,
                    'similarity_score': similarity_score,
                    'match_type': 'offerings',
                    'explanation': req_match.get('explanation', '')
                }
                
                # Check if already exists in current stored matches
                already_exists = any(
                    m['user_id'] == source_user_id 
                    for m in stored.get('offerings_matches', [])
                )
                
                if not already_exists:
                    offerings_matches = stored.get('offerings_matches', [])
                    offerings_matches.append(new_offering_match)
                    
                    # Update stored matches
                    updated_data = {
                        'requirements_matches': stored.get('requirements_matches', []),
                        'offerings_matches': offerings_matches
                    }
                    UserMatches.store_user_matches(matched_user_id, updated_data)
                    updated_users.append(matched_user_id)
                    # Flag OLD user for scheduled notification
                    try:
                        old_user_profile = UserProfile.get(matched_user_id)
                        old_user_profile.needs_matchmaking = "true"
                        old_user_profile.save()
                        logger.info(f"Set needs_matchmaking='true' for OLD user {matched_user_id}")
                    except Exception as e:
                        logger.error(f"Failed to set needs_matchmaking for {matched_user_id}: {str(e)}")
                
                # Get matched user's designation (regardless of existing match)
                try:
                    matched_profile = UserProfile.get(matched_user_id)
                    matched_designation = matched_profile.persona.designation if matched_profile.persona else ""
                except Exception:
                    matched_designation = ""
                
                # Add to match pairs only if similarity meets threshold
                if similarity_score >= notify_threshold:
                    match_pairs.append({
                        'user_a_id': source_user_id,
                        'user_a_designation': source_designation,
                        'user_b_id': matched_user_id,
                        'user_b_designation': matched_designation
                    })
            
            # Process offerings matches (source offers something, matched user needs it)
            for off_match in source_matches.get('offerings_matches', []):
                matched_user_id = off_match['user_id']
                similarity_score = off_match.get('similarity_score', 0.0)
                
                # VALIDATION 1: Check if pair already notified before (duplicate prevention)
                if skip_already_notified and NotifiedMatchPairs.is_pair_notified(source_user_id, matched_user_id):
                    logger.info(f"Skipping already notified pair: {source_user_id} <-> {matched_user_id}")
                    skipped_already_notified += 1
                    continue
                
                # VALIDATION 2: Check match limit for this cycle (over-matching prevention)
                if len(match_pairs) >= max_matches_per_cycle:
                    logger.info(f"Match limit reached ({max_matches_per_cycle}), skipping remaining matches")
                    skipped_over_limit += 1
                    continue
                
                # Get matched user's stored matches
                stored = UserMatches.get_user_matches(matched_user_id)
                if stored is None:
                    stored = {
                        'requirements_matches': [],
                        'offerings_matches': []
                    }
                
                # Add source_user to matched user's requirements_matches (reciprocal)
                new_requirement_match = {
                    'user_id': source_user_id,
                    'similarity_score': similarity_score,
                    'match_type': 'requirements',
                    'explanation': off_match.get('explanation', '')
                }
                
                # Check if already exists in current stored matches
                already_exists = any(
                    m['user_id'] == source_user_id 
                    for m in stored.get('requirements_matches', [])
                )
                
                if not already_exists:
                    requirements_matches = stored.get('requirements_matches', [])
                    requirements_matches.append(new_requirement_match)
                    
                    # Update stored matches
                    updated_data = {
                        'requirements_matches': requirements_matches,
                        'offerings_matches': stored.get('offerings_matches', [])
                    }
                    UserMatches.store_user_matches(matched_user_id, updated_data)
                    
                    # Set needs_matchmaking only if similarity meets threshold
                    try:
                        if similarity_score >= notify_threshold:
                            old_user_profile = UserProfile.get(matched_user_id)
                            old_user_profile.needs_matchmaking = "true"
                            old_user_profile.save()
                            logger.info(f"Set needs_matchmaking='true' for OLD user {matched_user_id}")
                    except Exception as e:
                        logger.error(f"Failed to set needs_matchmaking for {matched_user_id}: {str(e)}")
                    
                    if matched_user_id not in updated_users:
                        updated_users.append(matched_user_id)
                    
                    # Get matched user's designation
                    try:
                        matched_profile = UserProfile.get(matched_user_id)
                        matched_designation = matched_profile.persona.designation if matched_profile.persona else ""
                    except:
                        matched_designation = ""
                    
                    # Add to match pairs (only if not already added from requirements)
                    pair_exists = any(
                        (p['user_a_id'] == source_user_id and p['user_b_id'] == matched_user_id)
                        for p in match_pairs
                    )
                    if not pair_exists and similarity_score >= notify_threshold:
                        match_pairs.append({
                            'user_a_id': source_user_id,
                            'user_a_designation': source_designation,
                            'user_b_id': matched_user_id,
                            'user_b_designation': matched_designation
                        })
            
            logger.info(f"Updated {len(updated_users)} reciprocal matches for source user {source_user_id}")
            logger.info(f"Created {len(match_pairs)} match pairs for notification")
            logger.info(f"Skipped {skipped_already_notified} already notified pairs, {skipped_over_limit} over limit")
            
            return match_pairs
            
        except Exception as e:
            logger.error(f"Error updating reciprocal matches for user {source_user_id}: {str(e)}")
            return []


# Create singleton instance
matching_service = MatchingService()