"""User service for registration and profile management."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from celery import chain

from app.adapters.dynamodb import UserProfile, QuestionAnswer
from app.schemas.user import UserRegistrationRequest, UserRegistrationResponse
from app.workers.embedding_processing import generate_embeddings_task
from app.workers.persona_processing import generate_persona_task
from app.workers.resume_processing import process_resume_task

logger = logging.getLogger(__name__)


class UserService:
    """Service for user-related operations."""
    
    def register_user(self, request: UserRegistrationRequest) -> UserRegistrationResponse:
        """Register a new user or update an existing user profile."""
        logger.info(f"Registering user {request.user_id} (update={getattr(request, 'update', False)})")
        try:
            user_id = request.user_id
            is_update = getattr(request, "update", False)
            
            if is_update:
                # Update flow: update existing user and trigger pipeline
                try:
                    user_profile = UserProfile.get(user_id)
                    # Update profile fields
                    user_profile.profile.resume_link = request.resume_link
                    user_profile.profile.raw_questions = [
                        QuestionAnswer(prompt=q.prompt, answer=str(q.answer)) for q in request.questions
                    ]
                    user_profile.profile.updated_at = datetime.utcnow()
                    
                    # Trigger immediate pipeline for updated data
                    user_profile.needs_matchmaking = "false"
                    user_profile.save()
                    logger.info(f"User {user_id} updated, triggering pipeline")
                    
                    workflow = chain(
                        process_resume_task.s(user_id, request.resume_link),
                        generate_persona_task.s()
                    )
                    result = workflow.apply_async()
                    logger.debug(f"Pipeline started: {result.id}")
                    
                except UserProfile.DoesNotExist:
                    logger.warning(f"User {user_id} does not exist")
                    return UserRegistrationResponse(
                        success=False,
                        user_id="",
                        message=f"User {user_id} does not exist. Use update=false to create new user."
                    )
            else:
                # Onboarding flow: create new user and trigger pipeline
                user_profile = UserProfile.create_user(
                    user_id=user_id,
                    resume_link=request.resume_link,
                    questions=[q.model_dump() for q in request.questions]
                )
                user_profile.needs_matchmaking = "false"
                user_profile.save()  # Save BEFORE starting async tasks
                logger.info(f"Starting pipeline for new user: {user_id}")
                
                # Create task chain: resume -> persona
                # NOTE: Embeddings generation removed from automatic workflow
                # Backend should call /user/approve-summary endpoint to trigger embeddings generation
                workflow = chain(
                    process_resume_task.s(user_id, request.resume_link),
                    generate_persona_task.s()
                    # generate_embeddings_task.s()  # Commented - now triggered via /user/approve-summary
                )
                result = workflow.apply_async()
                logger.debug(f"Pipeline started: {result.id}")
            
            logger.info(f"User {user_id} registered/updated with {len(request.questions)} questions")
            
            return UserRegistrationResponse(
                success=True,
                user_id=user_id,
                message="User registered/updated successfully."
            )
            
        except Exception as e:
            logger.error(f"User registration failed: {str(e)}")
            return UserRegistrationResponse(
                success=False,
                user_id="",
                message=f"Registration failed: {str(e)}"
            )
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID."""
        try:
            user_profile = UserProfile.get(user_id)
            return user_profile.to_dict()
        except UserProfile.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error retrieving user profile: {str(e)}")
            return None
    
    def update_user_status(self, user_id: str, status: str) -> bool:
        """Update user processing status."""
        try:
            user_profile = UserProfile.get(user_id)
            user_profile.processing_status = status
            user_profile.save()
            return True
        except UserProfile.DoesNotExist:
            logger.warning(f"User profile {user_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating user status: {str(e)}")
            return False
