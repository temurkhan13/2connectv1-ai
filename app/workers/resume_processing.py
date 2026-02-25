"""
Celery worker for resume processing tasks.
"""
from celery import current_app
from app.core.celery import celery_app
from app.services.resume_service import ResumeService
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='process_resume')
def process_resume_task(self, user_id: str, resume_link: Optional[str]):
    """
    Background task to process resume using LangChain loaders and store parsed text.
    After completion, notifies pipeline orchestrator.
    """
    try:
        logger.info(f"Starting resume processing for user: {user_id}")
        
        # Process resume
        resume_service = ResumeService()
        result = resume_service.process_resume(user_id, resume_link)
        
        if result["success"]:
            if result["skipped"]:
                logger.info(f"Resume processing skipped for user {user_id}: {result['message']}")
            else:
                logger.info(f"Resume processing completed for user {user_id}: {result['message']}")
        else:
            logger.error(f"Resume processing failed for user {user_id}: {result['message']}")
        
        # Always return user_id for next task in chain
        return user_id
            
    except Exception as e:
        logger.exception(f"Error in resume processing task for user {user_id}: {e}")
        # Still return user_id to continue chain even on error
        return user_id