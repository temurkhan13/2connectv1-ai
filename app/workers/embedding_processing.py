"""
Celery worker for embedding generation tasks.
"""
from celery import current_app
from app.core.celery import celery_app
from app.services.embedding_service import embedding_service
from app.services.multi_vector_embedding_service import multi_vector_service
from app.services.matching_service import matching_service
from app.adapters.supabase_profiles import UserProfile, UserMatches, NotifiedMatchPairs
from app.adapters.supabase_onboarding import supabase_onboarding_adapter
from app.adapters.postgresql import postgresql_adapter
from app.services.use_case_templates import get_onboarding_slots, get_template
from app.services.notification_service import NotificationService
import logging
import uuid
import os

logger = logging.getLogger(__name__)


def _extract_objective_from_profile(user_profile: UserProfile) -> str:
    """
    Extract primary_goal/objective from profile for objective-specific embeddings.

    BUG-092 FIX: Query Supabase onboarding_answers table for extracted slots,
    not raw_questions (which contains question text, not slot names).

    Aligns with ObjectiveTypes from use_case_templates.py:
    - fundraising, investing, hiring, partnership
    - mentorship, cofounder, product_launch, networking, job_search

    Checks multiple sources in priority order:
    1. Supabase extracted slots (primary_goal)
    2. Raw questions (fallback for backwards compat)
    3. user_type inference
    """
    user_id = user_profile.user_id

    # PRIORITY 1: Get from Supabase extracted slots (correct source)
    # BUG-092 FIX: This is where extracted slot values actually live
    try:
        extracted_slots = supabase_onboarding_adapter.get_user_slots_sync(user_id)
        if extracted_slots and 'primary_goal' in extracted_slots:
            goal_value = extracted_slots['primary_goal'].get('value', '')
            if goal_value:
                goal_lower = goal_value.lower()
                logger.info(f"[Objective] Found primary_goal in Supabase: '{goal_value}'")

                # Map to ObjectiveType enum values
                if 'invest' in goal_lower:
                    return 'investing'
                if 'fund' in goal_lower or 'raise' in goal_lower or 'capital' in goal_lower:
                    return 'fundraising'
                if 'hire' in goal_lower or 'recruit' in goal_lower or 'talent' in goal_lower:
                    return 'hiring'
                if 'partner' in goal_lower:
                    return 'partnership'
                if 'mentor' in goal_lower:
                    return 'mentorship'
                if 'cofounder' in goal_lower or 'co-founder' in goal_lower:
                    return 'cofounder'
                if 'launch' in goal_lower or 'product' in goal_lower:
                    return 'product_launch'
                if 'network' in goal_lower or 'connect' in goal_lower:
                    return 'networking'
                if 'job' in goal_lower or 'career' in goal_lower or 'role' in goal_lower or 'position' in goal_lower:
                    return 'job_search'

                # If no keyword match, log and return as-is for potential future mapping
                logger.info(f"[Objective] No keyword match for '{goal_value}', returning as-is")
                return goal_lower
    except Exception as e:
        logger.warning(f"[Objective] Supabase query failed for user {user_id}: {e}")

    # PRIORITY 2: Fallback to raw_questions (backwards compat)
    if user_profile.profile and user_profile.profile.raw_questions:
        for q in user_profile.profile.raw_questions:
            q_dict = q.as_dict() if hasattr(q, 'as_dict') else q
            code = q_dict.get('code', '').lower()
            # Check for primary_goal slot
            if 'primary_goal' in code or 'objective' in code or 'goal' in code:
                answer = q_dict.get('answer', '')
                if answer:
                    return answer.strip().lower()

    # PRIORITY 3: Try user_type which may indicate objective
    if user_profile.profile and user_profile.profile.raw_questions:
        for q in user_profile.profile.raw_questions:
            q_dict = q.as_dict() if hasattr(q, 'as_dict') else q
            code = q_dict.get('code', '').lower()
            if 'user_type' in code:
                answer = q_dict.get('answer', '')
                if answer:
                    # Map user_type to likely objective
                    answer_lower = answer.strip().lower()
                    if 'founder' in answer_lower or 'entrepreneur' in answer_lower:
                        return 'fundraising'
                    if 'investor' in answer_lower or 'angel' in answer_lower or 'vc' in answer_lower:
                        return 'investing'
                    return answer_lower

    # Default to None (will use universal dimensions only)
    logger.info(f"[Objective] No objective found for user {user_id}, using universal")
    return None


def _generate_focus_slot_embeddings(user_id: str, objective: str) -> int:
    """
    Generate embeddings for focus slots specific to the user's objective.

    This creates direct embeddings from extracted slot VALUES (not narrative text),
    enabling more precise matching on structured fields like check_size, funding_need, etc.

    Args:
        user_id: User identifier
        objective: User's primary objective (e.g., "investing", "fundraising")

    Returns:
        Number of focus slot embeddings generated
    """
    if not objective:
        logger.info(f"[FocusSlots] No objective for user {user_id}, skipping focus slot embeddings")
        return 0

    try:
        # Get the focus slots for this objective's template
        focus_slots = get_onboarding_slots(objective)
        if not focus_slots:
            logger.info(f"[FocusSlots] No focus slots defined for objective '{objective}'")
            return 0

        logger.info(f"[FocusSlots] Generating embeddings for {len(focus_slots)} focus slots (objective: {objective})")

        # Get extracted slot values from Supabase
        extracted_slots = supabase_onboarding_adapter.get_user_slots_sync(user_id)
        if not extracted_slots:
            logger.warning(f"[FocusSlots] No extracted slots found for user {user_id}")
            return 0

        logger.info(f"[FocusSlots] Found {len(extracted_slots)} extracted slots for user {user_id}")

        # Generate embedding for each focus slot that has a value
        generated_count = 0
        for slot_name in focus_slots:
            slot_data = extracted_slots.get(slot_name)
            if not slot_data:
                continue

            slot_value = slot_data.get("value")
            if not slot_value:
                continue

            # Convert value to string if needed (handle lists, dicts)
            if isinstance(slot_value, list):
                text_value = ", ".join(str(v) for v in slot_value)
            elif isinstance(slot_value, dict):
                text_value = str(slot_value)
            else:
                text_value = str(slot_value)

            # Skip empty values
            if not text_value.strip():
                continue

            # Generate embedding for this slot value
            embedding = embedding_service.generate_embedding(text_value)
            if not embedding:
                logger.warning(f"[FocusSlots] Failed to generate embedding for slot '{slot_name}'")
                continue

            # Store as focus_slot_{slot_name}
            embedding_type = f"focus_slot_{slot_name}"
            success = postgresql_adapter.store_embedding(
                user_id=user_id,
                embedding_type=embedding_type,
                vector_data=embedding,
                metadata={
                    "slot_name": slot_name,
                    "slot_value": text_value[:500],  # Truncate for metadata
                    "objective": objective,
                    "source": "extracted_slot"
                }
            )

            if success:
                generated_count += 1
                logger.info(f"[FocusSlots] Stored {embedding_type} embedding for user {user_id[:8]}...")
            else:
                logger.warning(f"[FocusSlots] Failed to store {embedding_type} embedding")

        logger.info(f"[FocusSlots] Generated {generated_count} focus slot embeddings for user {user_id}")
        return generated_count

    except Exception as e:
        logger.error(f"[FocusSlots] Error generating focus slot embeddings for user {user_id}: {e}")
        return 0


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
        
        # Extract objective for objective-specific embeddings
        objective = _extract_objective_from_profile(user_profile)
        logger.info(f"User {user_id} objective for embeddings: {objective or 'universal'}")

        # Generate and store embeddings using hybrid service (SentenceTransformers + pgvector)
        logger.info(f"Generating embeddings for user {user_id} using hybrid service")

        success = embedding_service.store_user_embeddings(
            user_id=user_id,
            requirements=requirements or "",
            offerings=offerings or ""
        )

        # Generate multi-vector embeddings with objective-specific dimensions
        if success:
            try:
                logger.info(f"Generating multi-vector embeddings for user {user_id} (objective: {objective})")
                mv_result = multi_vector_service.generate_multi_vector_embeddings(
                    user_id=user_id,
                    requirements_text=requirements or "",
                    offerings_text=offerings or "",
                    store_in_db=True,
                    user_type=objective  # user_type param accepts objective for backward compat
                )
                dim_count = len(mv_result.get('dimensions', {}))
                logger.info(f"Generated {dim_count} multi-vector dimension embeddings for user {user_id}")
            except Exception as mv_error:
                # Don't fail the task if multi-vector fails - basic embeddings are still good
                logger.warning(f"Multi-vector embedding generation failed for user {user_id}: {mv_error}")

            # Generate DIRECTIONAL multi-vector embeddings (requirements_X and offerings_X)
            # These are used by the hybrid matcher for bidirectional scoring
            try:
                from app.services.multi_vector_matcher import multi_vector_matcher
                from app.adapters.supabase_onboarding import SupabaseOnboardingAdapter
                adapter = SupabaseOnboardingAdapter()
                slots = adapter.get_user_slots_sync(user_id)

                persona_data = {
                    "primary_goal": slots.get("primary_goal", {}).get("value", "") if isinstance(slots.get("primary_goal"), dict) else str(slots.get("primary_goal", "")),
                    "industry": slots.get("industry_focus", {}).get("value", "") if isinstance(slots.get("industry_focus"), dict) else str(slots.get("industry_focus", "")),
                    "stage": slots.get("company_stage", {}).get("value", "") if isinstance(slots.get("company_stage"), dict) else str(slots.get("company_stage", "")),
                    "geography": slots.get("geography", {}).get("value", "") if isinstance(slots.get("geography"), dict) else str(slots.get("geography", "")),
                    "engagement_style": slots.get("engagement_style", {}).get("value", "") if isinstance(slots.get("engagement_style"), dict) else str(slots.get("engagement_style", "")),
                    "dealbreakers": slots.get("dealbreakers", {}).get("value", "") if isinstance(slots.get("dealbreakers"), dict) else str(slots.get("dealbreakers", "")),
                    "requirements": requirements or "",
                    "offerings": offerings or "",
                }

                req_results = multi_vector_matcher.store_multi_vector_embeddings(
                    user_id=user_id, persona_data=persona_data, direction="requirements"
                )
                off_results = multi_vector_matcher.store_multi_vector_embeddings(
                    user_id=user_id, persona_data=persona_data, direction="offerings"
                )
                dir_count = sum(1 for v in {**req_results, **off_results}.values() if v)
                logger.info(f"Generated {dir_count} directional multi-vector embeddings for user {user_id}")
            except Exception as dir_error:
                logger.warning(f"Directional multi-vector embedding generation failed for user {user_id}: {dir_error}")

            # Generate focus slot embeddings (direct embeddings from extracted slot values)
            try:
                focus_slot_count = _generate_focus_slot_embeddings(user_id, objective)
                if focus_slot_count > 0:
                    logger.info(f"Generated {focus_slot_count} focus slot embeddings for user {user_id}")
            except Exception as fs_error:
                # Don't fail the task if focus slot embeddings fail
                logger.warning(f"Focus slot embedding generation failed for user {user_id}: {fs_error}")

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
