"""
AI Chat Processing Worker for simulating AI-to-AI conversations.
"""
import logging
import uuid
from app.core.celery import celery_app
from app.services.ai_chat_service import AIChatService
from app.services.notification_service import NotificationService
from app.adapters.dynamodb import ChatRecord

logger = logging.getLogger(__name__)


@celery_app.task(name="ai_chat_processing.simulate_ai_chat")
def simulate_ai_chat_task(
    initiator_id: str,
    responder_id: str,
    match_id: str,
    template: str = None
):
    """
    Celery task to simulate AI-to-AI chat conversation between two users.
    
    This task runs asynchronously to:
    1. Simulate a full conversation between two user personas
    2. Store the conversation in DynamoDB
    3. Send a webhook notification with the results
    
    Args:
        initiator_id: Initiator user ID
        responder_id: Responder user ID
        match_id: Match identifier
        template: Optional custom starting message
    """
    logger.info(f"Starting AI chat simulation task for match {match_id}")
    logger.info(f"Initiator: {initiator_id}, Responder: {responder_id}")
    
    try:
        # Initialize services
        chat_service = AIChatService()
        notification_service = NotificationService()
        
        # Simulate the conversation
        logger.info("Simulating AI-to-AI conversation...")
        result = chat_service.simulate_conversation(
            initiator_id=initiator_id,
            responder_id=responder_id,
            match_id=match_id,
            template=template
        )
        
        logger.info(f"Conversation simulation completed. Messages: {len(result['conversation_data'])}")
        logger.info(f"Compatibility score: {result['compatibility_score']}")
        
        # Generate unique chat_id
        chat_id = str(uuid.uuid4())
        logger.info(f"Generated chat_id: {chat_id}")
        
        # Store conversation in DynamoDB
        logger.info("Storing conversation in DynamoDB...")
        ChatRecord.store_chat(
            chat_id=chat_id,
            match_id=match_id,
            initiator_id=initiator_id,
            responder_id=responder_id,
            conversation_data=result['conversation_data'],
            ai_remarks=result['ai_remarks'],
            compatibility_score=result['compatibility_score']
        )
        logger.info("Conversation stored successfully")
        
        # Send webhook notification
        logger.info("Sending webhook notification...")
        webhook_result = notification_service.send_ai_chat_ready_notification(
            initiator_id=initiator_id,
            responder_id=responder_id,
            match_id=match_id,
            ai_remarks=result['ai_remarks'],
            compatibility_score=result['compatibility_score'],
            conversation_data=result['conversation_data']
        )
        
        if webhook_result['success']:
            logger.info("Webhook notification sent successfully")
        else:
            logger.warning(f"Webhook notification failed: {webhook_result['message']}")
        
        logger.info(f"AI chat simulation task completed successfully for match {match_id}")
        
        return {
            "success": True,
            "match_id": match_id,
            "message_count": len(result['conversation_data']),
            "compatibility_score": result['compatibility_score'],
            "webhook_sent": webhook_result['success']
        }
        
    except Exception as e:
        logger.exception(f"Error in AI chat simulation task for match {match_id}: {e}")
        return {
            "success": False,
            "match_id": match_id,
            "error": str(e)
        }

