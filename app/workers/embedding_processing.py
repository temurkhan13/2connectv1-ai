"""
Celery worker for embedding generation tasks.
"""
from celery import current_app
from app.core.celery import celery_app
from app.services.embedding_service import embedding_service
from app.services.matching_service import matching_service
from app.adapters.dynamodb import UserProfile, UserMatches, NotifiedMatchPairs
from app.services.notification_service import NotificationService
import logging
import uuid
import os

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='generate_embeddings')
def generate_embeddings_task(self, user_id: str):
    """
    Generate and store embeddings for a user's requirements and offerings.
    
    This task runs after persona generation is completed.
    
    Args:
        user_id: ID of the user to generate embeddings for
        
    Returns:
        Dictionary with task status and results
    """
    try:
        logger.info(f"Starting embedding generation for user {user_id}")
        
        # Get user profile from DynamoDB
        try:
            user_profile = UserProfile.get(user_id)
        except UserProfile.DoesNotExist:
            logger.error(f"User profile not found for user {user_id}")
            return {
                "success": False,
                "user_id": user_id,
                "message": "User profile not found"
            }
        
        # Check if persona is completed
        if user_profile.persona_status != 'completed':
            logger.error(f"Persona not completed for user {user_id}. Status: {user_profile.persona_status}")
            return {
                "success": False,
                "user_id": user_id,
                "message": f"Persona not completed. Status: {user_profile.persona_status}"
            }
        
        # Get requirements and offerings from persona
        requirements = user_profile.persona.requirements if user_profile.persona else None
        offerings = user_profile.persona.offerings if user_profile.persona else None
        
        if not requirements and not offerings:
            logger.warning(f"No requirements or offerings found for user {user_id}")
            return {
                "success": False,
                "user_id": user_id,
                "message": "No requirements or offerings found"
            }
        
        logger.info(f"Found requirements ({len(requirements) if requirements else 0} chars) and offerings ({len(offerings) if offerings else 0} chars)")
        
        # CLEAR OLD DATA: Remove old matches and notified pairs before fresh calculation
        # This ensures updated profile gets completely fresh matching
        logger.info(f"Clearing old matches and notified pairs for user {user_id}")
        UserMatches.clear_user_matches(user_id)
        cleared_pairs = NotifiedMatchPairs.clear_user_pairs(user_id)
        logger.info(f"Cleared old data: matches removed, {cleared_pairs} notified pairs removed")
        
        # Generate and store embeddings using hybrid service (SentenceTransformers + pgvector)
        logger.info(f"Generating embeddings for user {user_id} using hybrid service")
        
        success = embedding_service.store_user_embeddings(
            user_id=user_id,
            requirements=requirements or "",
            offerings=offerings or ""
        )
        
        if success:
            logger.info(f"Successfully generated and stored embeddings for user {user_id}")
            
            # NEW: Find and store matches after embeddings are ready
            logger.info(f"Finding and storing matches for user {user_id}")
            
            try:
                # Find and store matches
                matches_result = matching_service.find_and_store_user_matches(user_id)
                
                if matches_result['success']:
                    if matches_result.get('stored'):
                        logger.info(f"Successfully found and stored {matches_result['total_matches']} matches for user {user_id}")
                    else:
                        logger.info(f"Found {matches_result['total_matches']} matches for user {user_id} but storage failed")
                    
                    # Update OLD users' stored matches with this NEW user
                    # This ensures OLD users will be notified by scheduled worker
                    logger.info(f"Updating reciprocal matches for OLD users (requirements only)")
                    
                    # Only pass requirements_matches for reciprocal update
                    requirements_only = {
                        'requirements_matches': matches_result.get('requirements_matches', []),
                        'offerings_matches': []  # Don't update offerings
                    }
                    
                    match_pairs = matching_service.update_reciprocal_matches(
                        source_user_id=user_id,
                        source_matches=requirements_only
                    )
                    logger.info(f"Updated stored matches for {len(match_pairs)} OLD users")
                    
                    # Send notification to THIS user about THEIR matches (requirements only)
                    # Format: user_id + array of matches
                    # Endpoint: /api/v1/webhooks/user-matches-ready
                    notification_service = NotificationService()
                    
                    if notification_service.is_configured():
                        # Use the current task ID as batch_id
                        batch_id = self.request.id if hasattr(self, 'request') and self.request else str(uuid.uuid4())
                        
                        notify_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
                        requirements_matches = [
                            m for m in matches_result.get('requirements_matches', [])
                            if m.get('similarity_score', 0.0) >= notify_threshold
                        ]
                        
                        # Send notification using user-matches-ready endpoint
                        notification_result = notification_service.send_matches_ready_notification(
                            user_id=user_id,
                            batch_id=batch_id,
                            matches=requirements_matches  # Array of matched users
                        )
                        
                        if notification_result.get("success"):
                            logger.info(f"Successfully sent matches notification for user {user_id} with {len(requirements_matches)} matches")
                        else:
                            logger.error(f"Failed to send matches notification for user {user_id}: {notification_result.get('message')}")
                    else:
                        logger.warning(f"Backend notification not configured, skipping matches notification for user {user_id}")
                else:
                    logger.warning(f"No matches found for user {user_id}")
                    
            except Exception as e:
                logger.error(f"Error in matching/notification process for user {user_id}: {str(e)}")
                # Don't fail the entire task if matching/notification fails
                matches_result = {'success': False, 'total_matches': 0, 'stored': False}
        else:
            logger.error(f"Failed to generate/store embeddings for user {user_id}")
            matches_result = {'success': False, 'total_matches': 0, 'stored': False}
        
        # Pipeline completed - this is the final step
        logger.info(f"Complete pipeline finished for user {user_id}")
        
        # Return final result with match info
        return {
            "success": success,
            "user_id": user_id,
            "pipeline_completed": True,
            "matches_stored": matches_result.get('stored', False),
            "total_matches": matches_result.get('total_matches', 0),
            "message": "Pipeline completed successfully" if success else "Pipeline completed with errors"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in embedding generation for user {user_id}: {str(e)}")
        return {
            "success": False,
            "user_id": user_id,
            "message": f"Unexpected error: {str(e)}"
        }
