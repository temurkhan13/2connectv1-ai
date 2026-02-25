"""Feedback processing service."""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from app.adapters.dynamodb import Feedback, ChatRecord
from app.schemas.user import FeedbackRequest
from app.services.persona_service import update_persona_vector_with_feedback

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for processing user feedback on matches/chats."""
    
    def process_feedback(self, data: FeedbackRequest) -> Dict[str, Any]:
        """Process user feedback: save to DB and update persona embeddings."""
        try:
            # Generate unique feedback_id
            feedback_id = str(uuid.uuid4())
            
            # Save feedback in DynamoDB
            feedback_item = Feedback(
                feedback_id=feedback_id,
                user_id=data.user_id,
                type=data.type,
                target_id=data.id,
                feedback=data.feedback,
                created_at=datetime.utcnow()
            )
            feedback_item.save()
            
            # Prepare context based on feedback type
            context = None
            if data.type == "chat":
                # Fetch chat conversation for context
                try:
                    chat = ChatRecord.get(data.id)
                    context = {
                        "chat_id": chat.chat_id,
                        "conversation": [
                            {
                                "sender_id": getattr(msg, "sender_id", msg.get("sender_id") if isinstance(msg, dict) else None),
                                "content": getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else None)
                            }
                            for msg in chat.conversation_data
                        ],
                        "ai_remarks": getattr(chat, "ai_remarks", None),
                        "compatibility_score": getattr(chat, "compatibility_score", None)
                    }
                    logger.debug(f"Retrieved chat context for feedback: {data.id}")
                except ChatRecord.DoesNotExist:
                    logger.warning(f"Chat record not found: {data.id}")
                    context = None
                except Exception as e:
                    logger.error(f"Error fetching chat context: {e}")
                    context = None
            
            # Update persona vector using feedback
            update_persona_vector_with_feedback(
                user_id=data.user_id,
                feedback=data.feedback,
                feedback_type=data.type,
                match_context=context
            )
            
            return {
                "success": True,
                "message": "Feedback saved and persona updated",
                "data": data.dict()
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "success": False,
                "message": f"Failed to process feedback: {str(e)}"
            }