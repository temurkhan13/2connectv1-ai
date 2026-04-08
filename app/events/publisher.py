"""
Redis Event Publisher for 2Connect AI Service.

Publishes events to Redis pub/sub channels for cross-service communication.
Backend subscribes to these events to trigger actions like push notifications.

Events:
- matches_ready: Published when a user's matches are calculated and synced
- onboarding_complete: Published when a user completes onboarding

Channel naming: 2connect:events:<event_name>
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Redis URL from environment (same as used by cache.py)
REDIS_URL = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))

# Event channels
CHANNELS = {
    "matches_ready": "2connect:events:matches_ready",
    "onboarding_complete": "2connect:events:onboarding_complete",
    "match_accepted": "2connect:events:match_accepted",
}


class EventPublisher:
    """
    Publishes events to Redis for cross-service communication.

    Uses Redis pub/sub which is fire-and-forget. If no subscribers
    are listening, events are dropped (this is by design - events
    are notifications, not guaranteed delivery).
    """

    def __init__(self):
        self._client = None
        self._connected = False
        self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection for publishing."""
        if self._connected and self._client:
            return True

        try:
            import redis

            redis_kwargs = {
                "decode_responses": True,
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
            }

            # Support rediss:// URLs (Upstash, etc.)
            if REDIS_URL.startswith("rediss://"):
                redis_kwargs["ssl_cert_reqs"] = "none"

            self._client = redis.from_url(REDIS_URL, **redis_kwargs)
            self._client.ping()
            self._connected = True
            logger.info("[EventPublisher] Connected to Redis")
            return True

        except Exception as e:
            logger.warning(f"[EventPublisher] Redis connection failed: {e}")
            self._connected = False
            return False

    def publish(self, channel: str, data: Dict[str, Any]) -> bool:
        """
        Publish an event to a Redis channel.

        Args:
            channel: Channel name (use CHANNELS dict keys)
            data: Event payload (will be JSON serialized)

        Returns:
            True if published successfully, False otherwise
        """
        if not self._connected:
            if not self._connect():
                logger.warning(f"[EventPublisher] Cannot publish - not connected")
                return False

        try:
            channel_name = CHANNELS.get(channel, channel)
            message = json.dumps({
                **data,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "ai_service"
            })

            subscribers = self._client.publish(channel_name, message)
            logger.info(f"[EventPublisher] Published to {channel_name}: {subscribers} subscribers")
            return True

        except Exception as e:
            logger.error(f"[EventPublisher] Failed to publish: {e}")
            self._connected = False
            return False

    def publish_matches_ready(
        self,
        user_id: str,
        match_count: int,
        algorithm: str = "unknown",
        reciprocal_updates: int = 0,
        trigger: str = "onboarding"
    ) -> bool:
        """
        Publish event when a user's matches are ready.

        This triggers:
        - Push notification to user (message varies by trigger)
        - Real-time update if user is on dashboard

        Args:
            user_id: The user who now has matches
            match_count: Number of matches found
            algorithm: Which matching algorithm was used
            reciprocal_updates: How many existing users got reciprocal updates
            trigger: What caused the match generation:
                     "onboarding" — first-time after completing onboarding
                     "cron" — scheduled periodic re-matching
                     "profile_edit" — user edited their profile/summary

        Returns:
            True if published successfully
        """
        return self.publish("matches_ready", {
            "user_id": user_id,
            "match_count": match_count,
            "algorithm": algorithm,
            "reciprocal_updates": reciprocal_updates,
            "trigger": trigger,
            "event_type": "matches_ready"
        })

    def publish_onboarding_complete(
        self,
        user_id: str,
        session_id: str,
        slots_filled: int
    ) -> bool:
        """
        Publish event when a user completes onboarding.

        This can trigger:
        - Welcome push notification
        - Analytics tracking

        Args:
            user_id: The user who completed onboarding
            session_id: The onboarding session ID
            slots_filled: Number of slots filled during onboarding

        Returns:
            True if published successfully
        """
        return self.publish("onboarding_complete", {
            "user_id": user_id,
            "session_id": session_id,
            "slots_filled": slots_filled,
            "event_type": "onboarding_complete"
        })

    def publish_match_accepted(
        self,
        user_a_id: str,
        user_b_id: str,
        match_id: str
    ) -> bool:
        """
        Publish event when a match is accepted.

        This triggers:
        - Push notification to the other user
        - Analytics tracking

        Args:
            user_a_id: User who accepted
            user_b_id: User to be notified
            match_id: The match ID

        Returns:
            True if published successfully
        """
        return self.publish("match_accepted", {
            "user_a_id": user_a_id,
            "user_b_id": user_b_id,
            "match_id": match_id,
            "event_type": "match_accepted"
        })


# Singleton instance
event_publisher = EventPublisher()
