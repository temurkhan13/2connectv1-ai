"""
Celery worker for scheduled matchmaking tasks.

Processes users with needs_matchmaking='true' by retrieving stored matches,
performing reciprocal updates, sending a single batch notification, and
marking users as processed.

Supports three matching algorithms:
- Simple 2-vector (requirements vs offerings) - default
- Multi-vector weighted (6 dimensions) - set USE_MULTI_VECTOR_MATCHING=true
- Enhanced bidirectional with intent classification - set USE_ENHANCED_MATCHING=true
"""
from app.adapters.dynamodb import UserProfile, NotifiedMatchPairs, UserMatches
from app.services.matching_service import matching_service
from app.services.notification_service import NotificationService
from app.core.celery import celery_app
import uuid
import logging
import json
import os

logger = logging.getLogger(__name__)

# Feature flag for enhanced matching with intent classification
USE_ENHANCED_MATCHING = os.getenv("USE_ENHANCED_MATCHING", "false").lower() == "true"

if USE_ENHANCED_MATCHING:
    from app.services.matching_adapter import enhanced_matching_adapter
    logger.info("[ENHANCED MATCHING] Enabled - using bidirectional matching with intent classification")




@celery_app.task(bind=True, name='process_reciprocal_updates_and_notify')
def process_reciprocal_updates_and_notify_task(self, batch_id: str, phase2_results: list):
    """
    PHASE 3 TASK: Process reciprocal updates and send batch notification.
    
    This task runs AFTER all embeddings and matches are complete (Phase 2).
    It ensures all embeddings exist before any reciprocal updates happen.
    
    Workflow:
    1. For each successfully processed user, update reciprocal matches
    2. Collect all match pairs
    3. Send single batch notification to backend
    4. Mark all users as processed (needs_matchmaking='false')
    
    Args:
        batch_id: Batch identifier for tracking
        phase2_results: List of results from Phase 2 (embeddings + matches)
        
    Returns:
        Notification result summary
    """
    try:
        logger.info(f"[PHASE 3] Starting reciprocal updates for {len(phase2_results)} users")
        
        all_match_pairs = []
        successful_users = []
        failed_users = []
        
        # Process each user's reciprocal updates
        for result in phase2_results:
            if not isinstance(result, dict):
                continue
                
            user_id = result.get('user_id')
            
            if not result.get('success'):
                failed_users.append(user_id)
                logger.warning(f"[PHASE 3] User {user_id}: Skipping (Phase 2 failed) - {result.get('message')}")
                continue
            
            try:
                # Get the matches data from Phase 2
                matches_data = result.get('matches_data', {})
                
                if not matches_data:
                    logger.warning(f"[PHASE 3] User {user_id}: No matches data found")
                    successful_users.append(user_id)
                    continue
                
                # Perform reciprocal updates (ONLY for requirements_matches)
                logger.info(f"[PHASE 3] Updating reciprocal matches for user {user_id} (requirements only)")
                
                # Only pass requirements_matches for notification
                requirements_only = {
                    'requirements_matches': matches_data.get('requirements_matches', []),
                    'offerings_matches': []  # Don't notify offerings
                }
                
                match_pairs = matching_service.update_reciprocal_matches(
                    source_user_id=user_id,
                    source_matches=requirements_only
                )
                
                all_match_pairs.extend(match_pairs)
                successful_users.append(user_id)
                logger.info(f"[PHASE 3] User {user_id}: Created {len(match_pairs)} match pairs from requirements matches")
                
                # Mark user as processed
                try:
                    user = UserProfile.get(user_id)
                    user.needs_matchmaking = "false"
                    user.save()
                    logger.info(f"[PHASE 3] User {user_id}: Marked as processed (needs_matchmaking='false')")
                except Exception as e:
                    logger.error(f"[PHASE 3] Failed to update needs_matchmaking for user {user_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"[PHASE 3] Error processing reciprocal updates for user {user_id}: {str(e)}")
                failed_users.append(user_id)
        
        logger.info(f"[PHASE 3] Reciprocal updates complete")
        logger.info(f"[PHASE 3] Total match pairs: {len(all_match_pairs)}")
        logger.info(f"[PHASE 3] Successful: {len(successful_users)}, Failed: {len(failed_users)}")
        logger.info(f"[PHASE 3] Preparing batch notification with {len(all_match_pairs)} pairs (pre-dedupe)")

        filtered_pairs = []
        backfilled_count = 0
        for pair in all_match_pairs:
            try:
                # Skip if already notified previously
                if NotifiedMatchPairs.is_pair_notified(pair['user_a_id'], pair['user_b_id']):
                    continue

                # Backfill ledger for pairs that likely were sent via per-user webhook
                # If either side's stored matches already include the other, treat as already notified
                a_matches = UserMatches.get_user_matches(pair['user_a_id']) or {}
                b_matches = UserMatches.get_user_matches(pair['user_b_id']) or {}

                a_has_b = any(m.get('user_id') == pair['user_b_id'] for m in a_matches.get('requirements_matches', []))
                b_has_a = any(m.get('user_id') == pair['user_a_id'] for m in b_matches.get('requirements_matches', []))

                if a_has_b or b_has_a:
                    NotifiedMatchPairs.mark_pair_notified(pair['user_a_id'], pair['user_b_id'], pair.get('similarity_score'))
                    backfilled_count += 1
                    continue

                filtered_pairs.append(pair)
            except Exception as e:
                logger.error(f"[PHASE 3] Error during dedupe/backfill for pair {pair.get('user_a_id')} <-> {pair.get('user_b_id')}: {str(e)}")

        logger.info(f"[PHASE 3] Dedupe complete: {len(filtered_pairs)} pairs to notify, {backfilled_count} backfilled as already notified")

        webhook_match_pairs = []
        for pair in filtered_pairs:
            webhook_match_pairs.append({
                'user_a_id': pair['user_a_id'],
                'user_a_designation': pair['user_a_designation'],
                'user_b_id': pair['user_b_id'],
                'user_b_designation': pair['user_b_designation']
            })
        
        notification_service = NotificationService()
        notification_result = notification_service.send_batch_matches_notification(
            batch_id=batch_id,
            match_pairs=webhook_match_pairs
        )
        
        # POINT 4: Mark pairs as notified AFTER successful notification
        # This prevents duplicate notifications in future cycles
        pairs_marked = 0
        if notification_result.get('success', False) and filtered_pairs:
            logger.info(f"[PHASE 3] Marking {len(filtered_pairs)} pairs as notified...")
            for pair in filtered_pairs:
                try:
                    NotifiedMatchPairs.mark_pair_notified(
                        user_id_1=pair['user_a_id'],
                        user_id_2=pair['user_b_id'],
                        similarity_score=pair.get('similarity_score')
                    )
                    pairs_marked += 1
                except Exception as e:
                    logger.error(f"[PHASE 3] Failed to mark pair as notified: {pair['user_a_id']} <-> {pair['user_b_id']}: {str(e)}")
            logger.info(f"[PHASE 3] Successfully marked {pairs_marked}/{len(filtered_pairs)} pairs as notified")
        
        return {
            "success": True,
            "batch_id": batch_id,
            "total_match_pairs": len(filtered_pairs),
            "successful_users": len(successful_users),
            "failed_users": len(failed_users),
            "notification_sent": notification_result.get('success', False),
            "pairs_marked_notified": pairs_marked,
            "webhook_attempted": True,
            "notification_message": notification_result.get('message', ''),
            "message": f"Processed {len(successful_users)} users, created {len(all_match_pairs)} match pairs, {pairs_marked} marked as notified"
        }
            
    except Exception as e:
        logger.error(f"[PHASE 3] Error in reciprocal updates and notification: {str(e)}")
        return {
            "success": False,
            "message": f"Phase 3 error: {str(e)}"
        }


@celery_app.task(bind=True, name='scheduled_matchmaking')
def scheduled_matchmaking_task(self):
    """
    Process users with needs_matchmaking='true' and send batch notifications.
    Retrieves stored matches, performs reciprocal updates, batches notifications,
    and marks users as processed.
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING SCHEDULED MATCHMAKING TASK (3-PHASE SEQUENTIAL)")
        logger.info("=" * 80)
        
        users = list(UserProfile.scan(UserProfile.needs_matchmaking == "true"))
        logger.info(f"Found {len(users)} users needing matchmaking")
        
        batch_id = self.request.id if hasattr(self, 'request') and self.request else str(uuid.uuid4())
        logger.info(f"Batch ID: {batch_id}")
        
        if not users:
            notification_service = NotificationService()
            endpoint = f"{notification_service.backend_url}/api/v1/webhooks/matches-ready"
            headers = notification_service._get_headers()
            payload = {"batch_id": batch_id, "match_pairs": []}
            logger.info(f"Sending batch matches notification to {endpoint} for batch {batch_id}")
            logger.info("Sending 0 match pairs")
            logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
            logger.info(f"Request headers: {headers}")
            notification_result = notification_service.send_batch_matches_notification(
                batch_id=batch_id,
                match_pairs=[]
            )
            logger.info(f"Backend response status: {('200' if notification_result.get('success') else 'error')}")
            return {
                "success": notification_result.get('success', False),
                "batch_id": batch_id,
                "total_users": 0,
                "match_pairs_notified": 0,
                "notification_sent": notification_result.get('success', False),
                "notification_message": notification_result.get('message', ''),
                "message": "No users need matchmaking, empty batch notification sent"
            }
        logger.info(f"Batch ID: {batch_id}")
        
        # RECALCULATE matches for users (clear old + fresh calculate)
        phase2_results = []
        recalculate_matches = os.getenv("SCHEDULED_RECALCULATE_MATCHES", "true").lower() == "true"
        use_multi_vector = os.getenv("USE_MULTI_VECTOR_MATCHING", "false").lower() == "true"

        logger.info("")
        logger.info("=" * 80)
        if recalculate_matches:
            algorithm = "MULTI-VECTOR (6 dimensions)" if use_multi_vector else "SIMPLE (2-vector)"
            logger.info(f"RECALCULATING MATCHES using {algorithm} algorithm")
        else:
            logger.info("RETRIEVING STORED MATCHES")
        logger.info("=" * 80)

        for user in users:
            try:
                user_id = user.user_id

                if recalculate_matches:
                    # CLEAR old matches and notified pairs
                    logger.info(f"Clearing old data for user {user_id}")
                    UserMatches.clear_user_matches(user_id)
                    cleared_pairs = NotifiedMatchPairs.clear_user_pairs(user_id)
                    logger.info(f"Cleared old matches and {cleared_pairs} notified pairs for user {user_id}")

                    # RECALCULATE fresh matches using configured algorithm
                    if USE_ENHANCED_MATCHING:
                        logger.info(f"Recalculating ENHANCED BIDIRECTIONAL matches for user {user_id}")
                        matches_result = enhanced_matching_adapter.find_and_store_user_matches(user_id)
                    elif use_multi_vector:
                        logger.info(f"Recalculating MULTI-VECTOR matches for user {user_id}")
                        matches_result = matching_service.find_and_store_user_matches_multi_vector(user_id)
                    else:
                        logger.info(f"Recalculating matches for user {user_id}")
                        matches_result = matching_service.find_and_store_user_matches(user_id)
                    
                    if matches_result.get('success') and matches_result.get('total_matches', 0) > 0:
                        phase2_results.append({
                            "success": True,
                            "user_id": user_id,
                            "matches_data": {
                                'requirements_matches': matches_result.get('requirements_matches', []),
                                'offerings_matches': matches_result.get('offerings_matches', [])
                            }
                        })
                        logger.info(f"Recalculated {matches_result.get('total_matches', 0)} matches for user {user_id}")
                    else:
                        logger.info(f"No matches found for user {user_id} after recalculation")
                else:
                    # OLD behavior: just retrieve stored matches
                    stored_matches = UserMatches.get_user_matches(user_id)
                    if stored_matches and stored_matches.get('total_matches', 0) > 0:
                        phase2_results.append({
                            "success": True,
                            "user_id": user_id,
                            "matches_data": {
                                'requirements_matches': stored_matches.get('requirements_matches', []),
                                'offerings_matches': stored_matches.get('offerings_matches', [])
                            }
                        })
                        logger.info(f"Retrieved matches for user {user_id}")
                    else:
                        logger.info(f"No stored matches for user {user_id}")
                        
            except Exception as e:
                logger.error(f"Error processing matches for {user.user_id}: {str(e)}")
        
        successful_phase2 = sum(1 for r in phase2_results if r.get('success'))
        failed_phase2 = len(phase2_results) - successful_phase2
        logger.info(f"Users ready for reciprocal updates: {successful_phase2}")
        
        # ========================================================================
        # PHASE 3: Reciprocal Updates + Notification
        # ========================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[PHASE 3] STARTING: Reciprocal Updates + Notification")
        logger.info("=" * 80)
        
        notification_result = process_reciprocal_updates_and_notify_task(batch_id, phase2_results)
        
        logger.info(f"[PHASE 3] COMPLETED: {notification_result.get('message')}")
        
        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("SCHEDULED MATCHMAKING TASK COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total users processed: {len(users)}")
        logger.info(f"Users with stored matches: {successful_phase2}")
        logger.info(f"Phase 3 matches: {notification_result.get('match_pairs_count', 0)}")
        logger.info(f"Notification sent: {notification_result.get('notification_sent', False)}")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "batch_id": batch_id,
            "total_users": len(users),
            "phase2_successful": successful_phase2,
            "phase2_failed": failed_phase2,
            "matches_notified": notification_result.get('match_pairs_count', 0),
            "notification_sent": notification_result.get('notification_sent', False),
            "message": f"Processing complete: {successful_phase2}/{len(users)} users had stored matches, {notification_result.get('match_pairs_count', 0)} matches"
        }
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"SCHEDULED MATCHMAKING TASK FAILED: {str(e)}")
        logger.error("=" * 80)
        return {
            "success": False,
            "message": f"Task failed: {str(e)}"
        }
