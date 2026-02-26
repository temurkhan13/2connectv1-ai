"""
Notification service for sending data to external backend services.
"""
from typing import Dict, Any, Optional, List, Set
import requests
import os
import json
import logging
import uuid
import psycopg2
from app.adapters.dynamodb import UserProfile

logger = logging.getLogger(__name__)

# Cache for valid backend user IDs
_valid_user_ids_cache: Optional[Set[str]] = None
_cache_timestamp: float = 0
_CACHE_TTL = 300  # 5 minutes


def get_valid_backend_user_ids() -> Set[str]:
    """
    Get set of valid user IDs from backend database.

    Uses a 5-minute cache to avoid hitting the database on every notification.
    This ensures we only send matches for users that exist in the backend.
    """
    global _valid_user_ids_cache, _cache_timestamp
    import time

    current_time = time.time()
    if _valid_user_ids_cache is not None and (current_time - _cache_timestamp) < _CACHE_TTL:
        return _valid_user_ids_cache

    try:
        conn = psycopg2.connect(
            os.getenv('RECIPROCITY_BACKEND_DB_URL',
                     'postgresql://postgres:postgres@localhost:5432/reciprocity_db')
        )
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users')
        valid_ids = set(str(row[0]) for row in cursor.fetchall())
        cursor.close()
        conn.close()

        _valid_user_ids_cache = valid_ids
        _cache_timestamp = current_time
        logger.info(f"Refreshed valid backend user IDs cache: {len(valid_ids)} users")
        return valid_ids
    except Exception as e:
        logger.warning(f"Could not fetch valid user IDs from backend: {e}")
        # Return cached value if available, otherwise empty set
        return _valid_user_ids_cache if _valid_user_ids_cache else set()


class NotificationService:
    """Service for sending notifications to external services."""

    def __init__(self):
        """Initialize notification service."""
        self.backend_url = os.getenv('RECIPROCITY_BACKEND_URL')
        # Use same name as backend for consistency (AI_SERVICE_WEBHOOK_API_KEY)
        self.webhook_api_key = os.getenv('AI_SERVICE_WEBHOOK_API_KEY') or os.getenv('WEBHOOK_API_KEY')
    
    def is_configured(self) -> bool:
        """Check if backend URL is configured."""
        return bool(self.backend_url)
    
    def _get_headers(self) -> dict:
        """Get headers for webhook requests including API key if configured."""
        headers = {"Content-Type": "application/json"}
        if self.webhook_api_key:
            headers["X-API-KEY"] = self.webhook_api_key
        return headers
    
    def _convert_persona_to_markdown(self, persona_data: Dict[str, Any]) -> str:
        """
        Convert persona data dictionary to markdown formatted string.
        
        Args:
            persona_data: Dictionary containing persona fields
            
        Returns:
            Markdown formatted string
        """
        markdown_parts = []
        
        # Add name and archetype at the top if available
        if persona_data.get('name'):
            markdown_parts.append(f"# {persona_data['name']}")
        
        if persona_data.get('archetype'):
            markdown_parts.append(f"**Archetype:** {persona_data['archetype']}")
        
        if persona_data.get('designation'):
            markdown_parts.append(f"**Designation:** {persona_data['designation']}")
        
        if persona_data.get('experience'):
            markdown_parts.append(f"**Experience:** {persona_data['experience']}")
        
        # Add sections with headers
        if persona_data.get('focus'):
            markdown_parts.append(f"\n## Focus\n{persona_data['focus']}")
        
        if persona_data.get('profile_essence'):
            markdown_parts.append(f"\n## Profile Essence\n{persona_data['profile_essence']}")
        
        if persona_data.get('investment_philosophy'):
            markdown_parts.append(f"\n## Investment Philosophy\n{persona_data['investment_philosophy']}")
        
        if persona_data.get('what_theyre_looking_for'):
            markdown_parts.append(f"\n## What They're Looking For\n{persona_data['what_theyre_looking_for']}")
        
        if persona_data.get('engagement_style'):
            markdown_parts.append(f"\n## Engagement Style\n{persona_data['engagement_style']}")
        
        if persona_data.get('requirements'):
            markdown_parts.append(f"\n## Requirements\n{persona_data['requirements']}")
        
        if persona_data.get('offerings'):
            markdown_parts.append(f"\n## Offerings\n{persona_data['offerings']}")
        
        return "\n\n".join(markdown_parts)
    
    def send_persona_ready_notification(self, user_id: str) -> Dict[str, Any]:
        """
        Send persona ready notification to backend.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with notification result
        """
        try:
            if not self.is_configured():
                logger.warning(f"Backend URL not configured, skipping notification for user {user_id}")
                return {
                    "success": False,
                    "message": "Backend URL not configured"
                }
            
            # Get user profile with persona data
            try:
                user_profile = UserProfile.get(user_id)
                profile_data = user_profile.to_dict()
                
                # Extract persona data
                persona_data = profile_data.get('persona', {})
                
                if not persona_data:
                    logger.error(f"No persona data found for user {user_id}")
                    return {
                        "success": False,
                        "message": "No persona data available"
                    }
                
            except UserProfile.DoesNotExist:
                logger.error(f"User profile {user_id} not found")
                return {
                    "success": False,
                    "message": "User profile not found"
                }
            
            # Convert persona data to markdown format
            markdown_summary = self._convert_persona_to_markdown(persona_data)
            
            # Prepare payload
            payload = {
                "user_id": user_id,
                "summary": markdown_summary
            }
            
            # Send notification to backend
            # Note: backend_url already contains /api/v1 suffix
            endpoint = f"{self.backend_url}/webhooks/summary-ready"
            headers = self._get_headers()
            # SECURITY: Don't log payloads (contain PII) or headers (contain API keys)
            logger.info(f"Sending persona ready notification for user {user_id}")

            response = requests.post(
                endpoint,
                json=payload,
                timeout=30,
                headers=headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully notified backend for user {user_id}")
                logger.debug(f"Backend response status: {response.status_code}")
                
                # Try to parse JSON response, handle empty or invalid responses gracefully
                response_data = None
                try:
                    if response.content and response.text.strip():
                        response_data = response.json()
                        logger.debug("Backend response parsed as JSON")
                    else:
                        logger.debug("Backend response is empty")
                except ValueError:
                    # If JSON parsing fails, use the raw text
                    logger.debug("Backend response is not JSON")
                    response_data = response.text

                return {
                    "success": True,
                    "message": "Notification sent successfully",
                    "status_code": response.status_code,
                    "response": response_data,
                    "raw_response": response.text
                }
            else:
                logger.error(f"Backend notification failed for user {user_id}: status={response.status_code}")
                return {
                    "success": False,
                    "message": f"Backend returned {response.status_code}",
                    "status_code": response.status_code,
                    "response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.exception(f"Request error sending notification for user {user_id}: {e}")
            return {
                "success": False,
                "message": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"Error sending notification for user {user_id}: {e}")
            return {
                "success": False,
                "message": f"Notification error: {str(e)}"
            }
    
    def send_matches_ready_notification(self, user_id: str, batch_id: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send matches ready notification to backend.
        
        Args:
            user_id: User identifier
            batch_id: Worker/batch identifier (typically the Celery task ID)
            matches: List of requirement matches
            
        Returns:
            Dict with notification result
        """
        try:
            if not self.is_configured():
                logger.warning(f"Backend URL not configured, skipping matches notification for user {user_id}")
                return {
                    "success": False,
                    "message": "Backend URL not configured"
                }

            # CRITICAL: Get valid user IDs from backend to avoid foreign key violations
            # This prevents 500 errors when matched users don't exist in backend database
            valid_user_ids = get_valid_backend_user_ids()

            # Prepare matches payload - only include users that exist in backend
            matches_payload = []
            skipped_count = 0
            for match in matches:
                target_user_id = match.get('user_id')

                # Skip matches where target user doesn't exist in backend
                if valid_user_ids and target_user_id not in valid_user_ids:
                    skipped_count += 1
                    continue

                designation = ""
                # Fetch designation from user's persona in DynamoDB
                try:
                    target_profile = UserProfile.get(target_user_id)
                    if target_profile.persona and hasattr(target_profile.persona, 'designation'):
                        designation = target_profile.persona.designation or ""
                except UserProfile.DoesNotExist:
                    logger.warning(f"User profile not found for matched user {target_user_id}")
                except Exception as e:
                    logger.warning(f"Error fetching designation for user {target_user_id}: {str(e)}")
                matches_payload.append({
                    "target_user_id": target_user_id,
                    "target_user_designation": designation,
                })

            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} matches (users not in backend) for user {user_id}")

            if not matches_payload:
                logger.info(f"No valid matches to notify for user {user_id}")
                return {
                    "success": True,
                    "message": "No valid matches to notify",
                    "skipped": skipped_count
                }
            
            # Prepare notification payload
            payload = {
                "batch_id": batch_id,
                "user_id": user_id,
                "matches": matches_payload
            }
            
            # Send notification to backend
            # Note: backend_url already contains /api/v1 suffix
            endpoint = f"{self.backend_url}/webhooks/user-matches-ready"
            headers = self._get_headers()
            # SECURITY: Don't log payloads or headers
            logger.info(f"Sending matches notification for user {user_id} ({len(matches_payload)} matches)")

            response = requests.post(
                endpoint,
                json=payload,
                timeout=30,
                headers=headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully notified backend for user {user_id}")

                # Try to parse JSON response, handle empty or invalid responses gracefully
                response_data = None
                try:
                    if response.content and response.text.strip():
                        response_data = response.json()
                    else:
                        response_data = {}
                except json.JSONDecodeError:
                    response_data = {"raw_response": response.text}

                return {
                    "success": True,
                    "message": "Matches notification sent successfully",
                    "response": response_data
                }
            else:
                logger.error(f"Backend returned error for matches notification: status={response.status_code}")
                return {
                    "success": False,
                    "message": f"Backend error: {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error sending matches notification for user {user_id}: {e}")
            return {
                "success": False,
                "message": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"Error sending matches notification for user {user_id}: {e}")
            return {
                "success": False,
                "message": f"Notification error: {str(e)}"
            }
    
    def send_batch_matches_notification(self, batch_id: str, match_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send batch matches notification to backend for reciprocal updates.
        
        This is used by the scheduled worker to notify the backend about new match pairs
        that were created when processing updated users or new users.
        
        Args:
            batch_id: Worker/batch identifier (typically the Celery task ID)
            match_pairs: List of match pair dictionaries with structure:
                [
                    {
                        "user_a_id": "user_123",
                        "user_a_designation": "Software Engineer",
                        "user_b_id": "user_456",
                        "user_b_designation": "Product Manager"
                    },
                    ...
                ]
            
        Returns:
            Dict with notification result
        """
        try:
            if not self.is_configured():
                logger.warning(f"Backend URL not configured, skipping batch matches notification")
                return {
                    "success": False,
                    "message": "Backend URL not configured"
                }
            
            # Always send, even if empty
            
            # Prepare notification payload (scheduled worker contract)
            payload = {
                "batch_id": batch_id,
                "matches": match_pairs
            }
            
            # Send notification to backend
            # Note: backend_url already contains /api/v1 suffix
            endpoint = f"{self.backend_url}/webhooks/matches-ready"
            headers = self._get_headers()
            # SECURITY: Don't log payloads or headers
            logger.info(f"Sending batch matches notification for batch {batch_id} ({len(match_pairs)} matches)")

            response = requests.post(
                endpoint,
                json=payload,
                timeout=30,
                headers=headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully sent batch matches notification for batch {batch_id}")

                # Try to parse JSON response
                response_data = None
                try:
                    if response.content and response.text.strip():
                        response_data = response.json()
                    else:
                        response_data = {}
                except json.JSONDecodeError:
                    response_data = {"raw_response": response.text}

                return {
                    "success": True,
                    "message": "Batch matches notification sent successfully",
                    "match_pairs_count": len(match_pairs),
                    "response": response_data
                }
            else:
                logger.error(f"Backend returned error for batch matches notification: status={response.status_code}")
                return {
                    "success": False,
                    "message": f"Backend error: {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error sending batch matches notification: {e}")
            return {
                "success": False,
                "message": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"Error sending batch matches notification: {e}")
            return {
                "success": False,
                "message": f"Notification error: {str(e)}"
            }
    
    def send_ai_chat_ready_notification(
        self,
        initiator_id: str,
        responder_id: str,
        match_id: str,
        ai_remarks: str,
        compatibility_score: int,
        conversation_data: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Send AI chat ready notification to backend.
        
        Args:
            initiator_id: Initiator user ID
            responder_id: Responder user ID
            match_id: Match identifier
            ai_remarks: AI-generated remarks
            compatibility_score: Compatibility score
            conversation_data: List of conversation messages
            
        Returns:
            Dict with notification result
        """
        try:
            if not self.is_configured():
                logger.warning(f"Backend URL not configured, skipping AI chat notification for match {match_id}")
                return {
                    "success": False,
                    "message": "Backend URL not configured"
                }
            
            # Prepare payload
            payload = {
                "initiator_id": initiator_id,
                "responder_id": responder_id,
                "match_id": match_id,
                "ai_remarks": ai_remarks,
                "compatibility_score": compatibility_score,
                "conversation_data": conversation_data
            }
            
            # Send notification to backend
            # Note: backend_url already contains /api/v1 suffix
            endpoint = f"{self.backend_url}/webhooks/ai-chat-ready"
            headers = self._get_headers()
            # SECURITY: Don't log payloads (contain conversation data) or headers
            logger.info(f"Sending AI chat ready notification for match {match_id}")

            response = requests.post(
                endpoint,
                json=payload,
                timeout=30,
                headers=headers
            )

            if response.status_code == 200:
                logger.info(f"Successfully sent AI chat ready notification for match {match_id}")

                # Try to parse JSON response
                response_data = None
                try:
                    if response.content and response.text.strip():
                        response_data = response.json()
                    else:
                        response_data = {}
                except json.JSONDecodeError:
                    response_data = {"raw_response": response.text}

                return {
                    "success": True,
                    "message": "AI chat notification sent successfully",
                    "response": response_data
                }
            else:
                logger.error(f"Backend returned error for AI chat notification: status={response.status_code}")
                return {
                    "success": False,
                    "message": f"Backend error: {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error sending AI chat notification for match {match_id}: {e}")
            return {
                "success": False,
                "message": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"Error sending AI chat notification for match {match_id}: {e}")
            return {
                "success": False,
                "message": f"Notification error: {str(e)}"
            }
