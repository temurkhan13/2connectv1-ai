"""
Celery worker for persona generation tasks.
"""
from celery import current_app
from app.core.celery import celery_app
from app.services.persona_service import PersonaService
from app.adapters.dynamodb import UserProfile
from app.services.notification_service import NotificationService
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def _convert_persona_to_markdown(persona: Dict[str, Any], requirements: str, offerings: str) -> str:
    """
    Convert persona data to markdown format for storage.

    This is used by the direct DB write fallback to ensure AI summary
    is created even if the webhook notification fails.
    """
    parts = []

    if persona.get('name'):
        parts.append(f"# {persona['name']}")

    if persona.get('archetype'):
        parts.append(f"**Archetype:** {persona['archetype']}")

    if persona.get('designation'):
        parts.append(f"**Designation:** {persona['designation']}")

    if persona.get('experience'):
        parts.append(f"**Experience:** {persona['experience']}")

    if persona.get('focus'):
        parts.append(f"\n## Focus\n{persona['focus']}")

    if persona.get('profile_essence'):
        parts.append(f"\n## Profile Essence\n{persona['profile_essence']}")

    # Strategy field (role-agnostic, replaces investment_philosophy)
    strategy = persona.get('strategy') or persona.get('investment_philosophy')
    if strategy:
        parts.append(f"\n## Strategy\n{strategy}")

    if persona.get('what_theyre_looking_for'):
        parts.append(f"\n## What They're Looking For\n{persona['what_theyre_looking_for']}")

    if persona.get('engagement_style'):
        parts.append(f"\n## Engagement Style\n{persona['engagement_style']}")

    if requirements:
        parts.append(f"\n## Requirements\n{requirements}")

    if offerings:
        parts.append(f"\n## Offerings\n{offerings}")

    return "\n\n".join(parts)

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

            # BUG-009 + BUG-013 FIX: Ensure requirements/offerings are strings for DynamoDB
            # DynamoDB SerializationException occurs if these are complex objects (lists, dicts)
            # Convert lists properly (join with semicolons) instead of str() which gives "['item1', 'item2']"
            if isinstance(requirements, list):
                requirements = "; ".join(str(item).strip() for item in requirements if item)
                logger.info(f"BUG-013 FIX: Converted requirements from list to string in persona_processing")
            elif not isinstance(requirements, str):
                requirements = str(requirements) if requirements else ''

            if isinstance(offerings, list):
                offerings = "; ".join(str(item).strip() for item in offerings if item)
                logger.info(f"BUG-013 FIX: Converted offerings from list to string in persona_processing")
            elif not isinstance(offerings, str):
                offerings = str(offerings) if offerings else ''

            # BUG-024 FIX: Ensure ALL persona fields are strings before DynamoDB write
            # Convert any non-string values to strings to prevent SerializationException
            def ensure_string(value):
                """Convert any value to string safely."""
                if value is None:
                    return ''
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    # Join list items with semicolons
                    return "; ".join(str(item).strip() for item in value if item)
                if isinstance(value, dict):
                    # Convert dict to readable string
                    return "; ".join(f"{k}: {v}" for k, v in value.items())
                # For numbers, booleans, etc.
                return str(value)

            # Sanitize all persona fields
            persona_sanitized = {
                key: ensure_string(value)
                for key, value in persona.items()
            }

            # Debug logging
            logger.info(f"Generated persona: {persona_sanitized.get('name', 'N/A')}")
            logger.info(f"Requirements type: {type(requirements)}, length: {len(requirements)} characters")
            logger.info(f"Offerings type: {type(offerings)}, length: {len(offerings)} characters")
            
            # Store persona data with requirements and offerings using update method
            # Note: "strategy" is the role-agnostic field that maps to investment_philosophy in DB
            # IMPORTANT: All .get() calls must provide default empty strings to prevent DynamoDB SerializationException
            # BUG-024 FIX: Use persona_sanitized (all strings) instead of raw persona
            user_profile.update(
                actions=[
                    UserProfile.persona.name.set(persona_sanitized.get('name', '')),
                    UserProfile.persona.archetype.set(persona_sanitized.get('archetype', '')),
                    UserProfile.persona.experience.set(persona_sanitized.get('experience', '')),
                    UserProfile.persona.focus.set(persona_sanitized.get('focus', '')),
                    UserProfile.persona.profile_essence.set(persona_sanitized.get('profile_essence', '')),
                    # Strategy field replaces investment_philosophy (role-agnostic)
                    UserProfile.persona.investment_philosophy.set(persona_sanitized.get('strategy') or persona_sanitized.get('investment_philosophy') or ''),
                    UserProfile.persona.what_theyre_looking_for.set(persona_sanitized.get('what_theyre_looking_for', '')),
                    UserProfile.persona.engagement_style.set(persona_sanitized.get('engagement_style', '')),
                    UserProfile.persona.designation.set(persona_sanitized.get('designation', '')),
                    UserProfile.persona.requirements.set(requirements or ''),
                    UserProfile.persona.offerings.set(offerings or ''),
                    UserProfile.persona.generated_at.set(datetime.utcnow()),
                    UserProfile.persona_status.set('completed')
                ]
            )
            
            logger.info(f"Successfully stored persona data for user {user_id}")
            logger.info(f"Successfully generated persona for user {user_id}: {persona_sanitized.get('name')}")

            # Generate markdown summary for backend storage
            # BUG-024 FIX: Use sanitized persona
            markdown_summary = _convert_persona_to_markdown(persona_sanitized, requirements, offerings)

            # DIRECT DB WRITE (March 2026) - Ensure AI summary is created regardless of webhook status
            # This is a fallback to prevent "No AI summary found" issues
            try:
                from app.adapters.postgresql import postgresql_adapter
                summary_id = postgresql_adapter.create_user_summary(
                    user_id=user_id,
                    summary=markdown_summary,
                    status='approved',
                    urgency='ongoing'
                )
                if summary_id:
                    logger.info(f"Created user_summary directly in PostgreSQL for user {user_id}")
                else:
                    logger.warning(f"Failed to create user_summary in PostgreSQL for user {user_id}")
            except Exception as db_error:
                logger.error(f"Error creating user_summary in PostgreSQL for user {user_id}: {db_error}")

            # Send notification to backend (only if requested) - may be redundant now but keeps webhook flow
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
            # NOTE: Using .delay() instead of send_task() so CELERY_TASK_ALWAYS_EAGER works
            try:
                from app.workers.embedding_processing import generate_embeddings_task
                logger.info(f"Auto-triggering embedding generation for user: {user_id}")
                generate_embeddings_task.delay(user_id)
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
