"""
Celery worker for persona generation tasks.
"""
from celery import current_app
from app.core.celery import celery_app
from app.services.persona_service import PersonaService
from app.adapters.dynamodb import UserProfile
from app.services.notification_service import NotificationService
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='generate_persona')
def generate_persona_task(self, user_id: str, send_notification: bool = True):
    """
    Generate persona for a user using AI.
    
    Args:
        user_id: User ID
        send_notification: Whether to send persona ready notification (default: True)
    """
    try:
        logger.info(f"Starting persona generation for user: {user_id}")
        
        # Get user profile
        user_profile = UserProfile.get(user_id)
        
        # Extract questions and resume text from user profile
        questions = [q.as_dict() for q in user_profile.profile.raw_questions]
        resume_text = user_profile.resume_text.text if user_profile.resume_text and user_profile.resume_text.text else ""
        
        if not resume_text:
            logger.info(f"No resume text found for user {user_id}, generating persona from questions only")
        else:
            logger.info(f"Resume text found for user {user_id}, generating persona from questions and resume")
        
        # Initialize persona service
        persona_service = PersonaService()
        
        # Generate persona using AI
        logger.info(f"Generating persona for user {user_id}...")
        persona_data = persona_service.generate_persona_sync(questions, resume_text)
        
        if persona_data:
            # Extract persona from the new structure
            persona = persona_data.get('persona', {})
            requirements = persona_data.get('requirements', '')
            offerings = persona_data.get('offerings', '')
            
            # Debug logging
            logger.info(f"Generated persona: {persona.get('name', 'N/A')}")
            logger.info(f"Requirements length: {len(requirements)} characters")
            logger.info(f"Offerings length: {len(offerings)} characters")
            
            # Store persona data with requirements and offerings using update method
            # Note: "strategy" is the role-agnostic field that maps to investment_philosophy in DB
            user_profile.update(
                actions=[
                    UserProfile.persona.name.set(persona.get('name')),
                    UserProfile.persona.archetype.set(persona.get('archetype')),
                    UserProfile.persona.experience.set(persona.get('experience')),
                    UserProfile.persona.focus.set(persona.get('focus')),
                    UserProfile.persona.profile_essence.set(persona.get('profile_essence')),
                    # Strategy field replaces investment_philosophy (role-agnostic)
                    UserProfile.persona.investment_philosophy.set(persona.get('strategy', persona.get('investment_philosophy'))),
                    UserProfile.persona.what_theyre_looking_for.set(persona.get('what_theyre_looking_for')),
                    UserProfile.persona.engagement_style.set(persona.get('engagement_style')),
                    UserProfile.persona.designation.set(persona.get('designation')),
                    UserProfile.persona.requirements.set(requirements),
                    UserProfile.persona.offerings.set(offerings),
                    UserProfile.persona.generated_at.set(datetime.utcnow()),
                    UserProfile.persona_status.set('completed')
                ]
            )
            
            logger.info(f"Successfully stored persona data for user {user_id}")
            logger.info(f"Successfully generated persona for user {user_id}: {persona.get('name')}")
            
            # Send notification to backend (only if requested)
            logger.info(f"Notification check for user {user_id}: send_notification={send_notification}")
            notification_service = NotificationService()
            logger.info(f"NotificationService backend_url={notification_service.backend_url}, is_configured={notification_service.is_configured()}")

            if send_notification:
                logger.info(f"Sending persona ready notification for user: {user_id}")

                if notification_service.is_configured():
                    notification_result = notification_service.send_persona_ready_notification(user_id)
                    if notification_result.get("success"):
                        logger.info(f"Successfully notified backend for user {user_id}")
                    else:
                        logger.error(f"Failed to notify backend for user {user_id}: {notification_result.get('message')}")
                else:
                    logger.warning(f"Backend notification not configured, skipping for user {user_id}")
            else:
                logger.info(f"Skipping persona ready notification for user {user_id} (notification disabled)")
            
            logger.info(f"Persona generation completed successfully for user: {user_id}")

            # AUTO-TRIGGER EMBEDDING GENERATION
            # This ensures embeddings are generated regardless of backend notification status
            try:
                from app.core.celery import celery_app
                logger.info(f"Auto-triggering embedding generation for user: {user_id}")
                celery_app.send_task('generate_embeddings', args=[user_id])
                logger.info(f"Embedding generation task queued for user: {user_id}")
            except Exception as embed_error:
                logger.error(f"Failed to trigger embedding generation for user {user_id}: {embed_error}")
                # Don't fail the persona task if embedding trigger fails

            # Return result dict for next task in chain
            return {
                "success": True,
                "user_id": user_id,
                "message": "Persona generated successfully"
            }
        else:
            logger.error(f"Failed to generate persona for user {user_id}")
            user_profile.update(actions=[UserProfile.persona_status.set('failed')])
            user_profile.save()
            # Return failure dict but allow chain to continue
            return {
                "success": False,
                "user_id": user_id,
                "message": "Persona generation failed"
            }
            
    except UserProfile.DoesNotExist:
        logger.error(f"User profile {user_id} not found for persona generation")
        return {
            "success": False,
            "user_id": user_id,
            "message": "User profile not found"
        }
    except Exception as e:
        logger.exception(f"Error generating persona for user {user_id}: {e}")
        try:
            user_profile = UserProfile.get(user_id)
            user_profile.update(actions=[UserProfile.persona_status.set('failed')])
            user_profile.save()
        except Exception:
            pass
        return {
            "success": False,
            "user_id": user_id,
            "message": f"Error: {str(e)}"
        }
