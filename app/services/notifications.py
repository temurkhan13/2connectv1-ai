"""
Smart Notifications Service.

Manages notifications with intelligent batching, priority handling,
and user preference management.

Key features:
1. Multi-channel notifications (in-app, email, push)
2. Smart batching to reduce noise
3. Priority-based delivery
4. User preferences and quiet hours
5. Read/unread tracking
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    NEW_MATCH = "new_match"
    MESSAGE_RECEIVED = "message_received"
    MATCH_ACCEPTED = "match_accepted"
    MEETING_REMINDER = "meeting_reminder"
    PROFILE_VIEW = "profile_view"
    FEEDBACK_REQUEST = "feedback_request"
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    WEEKLY_DIGEST = "weekly_digest"
    MATCH_EXPIRING = "match_expiring"
    CONVERSATION_STALLED = "conversation_stalled"


class NotificationPriority(int, Enum):
    """Priority levels for notifications."""
    URGENT = 1       # Immediate delivery
    HIGH = 2         # Within 5 minutes
    MEDIUM = 3       # Can be batched hourly
    LOW = 4          # Can be batched daily
    DIGEST = 5       # Only in digests


class NotificationChannel(str, Enum):
    """Delivery channels."""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"


class NotificationStatus(str, Enum):
    """Status of a notification."""
    PENDING = "pending"      # Not yet delivered
    DELIVERED = "delivered"  # Sent to channel
    READ = "read"            # User has seen it
    DISMISSED = "dismissed"  # User dismissed
    EXPIRED = "expired"      # Past expiry time


@dataclass
class Notification:
    """A single notification."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    body: str
    status: NotificationStatus
    created_at: datetime
    channels: List[NotificationChannel]
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    action_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserNotificationPreferences:
    """User's notification preferences."""
    user_id: str
    enabled_channels: List[NotificationChannel]
    enabled_types: List[NotificationType]
    quiet_hours_start: Optional[time] = None
    quiet_hours_end: Optional[time] = None
    email_digest_only: bool = False
    batch_non_urgent: bool = True
    timezone: str = "UTC"


@dataclass
class NotificationBatch:
    """A batch of notifications for delivery."""
    batch_id: str
    user_id: str
    channel: NotificationChannel
    notifications: List[Notification]
    created_at: datetime


class NotificationService:
    """
    Manages notifications with smart delivery.

    Handles batching, priority, user preferences,
    and multi-channel delivery.
    """

    # Default channel config by notification type
    DEFAULT_CHANNELS = {
        NotificationType.NEW_MATCH: [NotificationChannel.IN_APP, NotificationChannel.PUSH],
        NotificationType.MESSAGE_RECEIVED: [NotificationChannel.IN_APP, NotificationChannel.PUSH],
        NotificationType.MATCH_ACCEPTED: [NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        NotificationType.MEETING_REMINDER: [NotificationChannel.IN_APP, NotificationChannel.EMAIL, NotificationChannel.PUSH],
        NotificationType.PROFILE_VIEW: [NotificationChannel.IN_APP],
        NotificationType.FEEDBACK_REQUEST: [NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        NotificationType.SYSTEM_ANNOUNCEMENT: [NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        NotificationType.WEEKLY_DIGEST: [NotificationChannel.EMAIL],
        NotificationType.MATCH_EXPIRING: [NotificationChannel.IN_APP, NotificationChannel.PUSH],
        NotificationType.CONVERSATION_STALLED: [NotificationChannel.IN_APP]
    }

    # Default priority by type
    DEFAULT_PRIORITY = {
        NotificationType.NEW_MATCH: NotificationPriority.HIGH,
        NotificationType.MESSAGE_RECEIVED: NotificationPriority.HIGH,
        NotificationType.MATCH_ACCEPTED: NotificationPriority.MEDIUM,
        NotificationType.MEETING_REMINDER: NotificationPriority.URGENT,
        NotificationType.PROFILE_VIEW: NotificationPriority.LOW,
        NotificationType.FEEDBACK_REQUEST: NotificationPriority.LOW,
        NotificationType.SYSTEM_ANNOUNCEMENT: NotificationPriority.MEDIUM,
        NotificationType.WEEKLY_DIGEST: NotificationPriority.DIGEST,
        NotificationType.MATCH_EXPIRING: NotificationPriority.MEDIUM,
        NotificationType.CONVERSATION_STALLED: NotificationPriority.LOW
    }

    # Notification templates
    TEMPLATES = {
        NotificationType.NEW_MATCH: {
            "title": "New Match!",
            "body": "You have a new {tier} match with {name}. Check out their profile!"
        },
        NotificationType.MESSAGE_RECEIVED: {
            "title": "New Message",
            "body": "{name} sent you a message"
        },
        NotificationType.MATCH_ACCEPTED: {
            "title": "Connection Accepted",
            "body": "{name} accepted your connection request. Start a conversation!"
        },
        NotificationType.MEETING_REMINDER: {
            "title": "Meeting Reminder",
            "body": "You have a meeting with {name} in {time_until}"
        },
        NotificationType.PROFILE_VIEW: {
            "title": "Profile View",
            "body": "Someone viewed your profile"
        },
        NotificationType.FEEDBACK_REQUEST: {
            "title": "How was your connection?",
            "body": "Share feedback on your conversation with {name}"
        },
        NotificationType.MATCH_EXPIRING: {
            "title": "Match Expiring Soon",
            "body": "Your match with {name} will expire in {time_until}. Connect now!"
        },
        NotificationType.CONVERSATION_STALLED: {
            "title": "Continue your conversation?",
            "body": "Your conversation with {name} has been quiet. Say hello!"
        }
    }

    def __init__(self):
        # Storage (in production, use database)
        self._notifications: Dict[str, List[Notification]] = defaultdict(list)
        self._preferences: Dict[str, UserNotificationPreferences] = {}
        self._pending_batches: Dict[str, List[Notification]] = defaultdict(list)

        # Configuration
        self.batch_window_minutes = int(os.getenv("NOTIFICATION_BATCH_WINDOW", "60"))
        self.max_notifications_per_batch = int(os.getenv("MAX_NOTIFICATIONS_BATCH", "10"))
        self.notification_expiry_days = int(os.getenv("NOTIFICATION_EXPIRY_DAYS", "30"))

    def create_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        context: Dict[str, Any],
        priority: Optional[NotificationPriority] = None,
        channels: Optional[List[NotificationChannel]] = None,
        action_url: Optional[str] = None
    ) -> Notification:
        """
        Create and queue a notification.

        Args:
            user_id: User to notify
            notification_type: Type of notification
            context: Context for template rendering
            priority: Optional priority override
            channels: Optional channels override
            action_url: Optional action URL

        Returns:
            Notification record
        """
        # Get preferences
        prefs = self._get_or_create_preferences(user_id)

        # Check if type is enabled
        if notification_type not in prefs.enabled_types:
            logger.debug(f"Notification type {notification_type} disabled for user {user_id}")
            return None

        # Determine channels
        if channels is None:
            channels = self.DEFAULT_CHANNELS.get(notification_type, [NotificationChannel.IN_APP])

        # Filter by user's enabled channels
        channels = [c for c in channels if c in prefs.enabled_channels]

        if not channels:
            logger.debug(f"No enabled channels for notification to user {user_id}")
            return None

        # Determine priority
        if priority is None:
            priority = self.DEFAULT_PRIORITY.get(notification_type, NotificationPriority.MEDIUM)

        # Render template
        template = self.TEMPLATES.get(notification_type, {"title": "Notification", "body": ""})
        title = template["title"]
        body = template["body"].format(**context) if context else template["body"]

        # Create notification
        notification_id = f"notif_{user_id}_{datetime.utcnow().timestamp()}"
        now = datetime.utcnow()

        notification = Notification(
            notification_id=notification_id,
            user_id=user_id,
            notification_type=notification_type,
            priority=priority,
            title=title,
            body=body,
            status=NotificationStatus.PENDING,
            created_at=now,
            channels=channels,
            expires_at=now + timedelta(days=self.notification_expiry_days),
            action_url=action_url,
            metadata=context
        )

        # Queue for delivery
        self._queue_notification(notification, prefs)

        logger.info(f"Created notification {notification_id} for user {user_id}")
        return notification

    def _queue_notification(
        self,
        notification: Notification,
        prefs: UserNotificationPreferences
    ) -> None:
        """Queue notification for delivery."""
        # Check quiet hours
        if self._is_quiet_hours(prefs):
            # Defer non-urgent notifications
            if notification.priority.value > NotificationPriority.URGENT.value:
                self._pending_batches[notification.user_id].append(notification)
                return

        # Immediate delivery for urgent
        if notification.priority == NotificationPriority.URGENT:
            self._deliver_notification(notification)
            return

        # Batch non-urgent if preference set
        if prefs.batch_non_urgent and notification.priority.value >= NotificationPriority.MEDIUM.value:
            self._pending_batches[notification.user_id].append(notification)
        else:
            self._deliver_notification(notification)

    def _is_quiet_hours(self, prefs: UserNotificationPreferences) -> bool:
        """Check if currently in user's quiet hours."""
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return False

        now = datetime.utcnow().time()
        start = prefs.quiet_hours_start
        end = prefs.quiet_hours_end

        if start <= end:
            return start <= now <= end
        else:
            # Quiet hours span midnight
            return now >= start or now <= end

    def _deliver_notification(self, notification: Notification) -> None:
        """Deliver notification to channels."""
        notification.status = NotificationStatus.DELIVERED
        notification.delivered_at = datetime.utcnow()
        self._notifications[notification.user_id].append(notification)

        # In production, this would dispatch to actual channels
        for channel in notification.channels:
            logger.debug(f"Delivering notification {notification.notification_id} to {channel.value}")

    def process_batches(self) -> int:
        """
        Process pending notification batches.

        Call this periodically (e.g., every hour).

        Returns:
            Number of batches processed
        """
        processed = 0

        for user_id, pending in list(self._pending_batches.items()):
            if not pending:
                continue

            # Group by channel
            by_channel = defaultdict(list)
            for notif in pending:
                for channel in notif.channels:
                    by_channel[channel].append(notif)

            # Create and deliver batches
            for channel, notifs in by_channel.items():
                batch = NotificationBatch(
                    batch_id=f"batch_{user_id}_{datetime.utcnow().timestamp()}",
                    user_id=user_id,
                    channel=channel,
                    notifications=notifs[:self.max_notifications_per_batch],
                    created_at=datetime.utcnow()
                )

                self._deliver_batch(batch)
                processed += 1

            # Clear pending
            self._pending_batches[user_id] = []

        logger.info(f"Processed {processed} notification batches")
        return processed

    def _deliver_batch(self, batch: NotificationBatch) -> None:
        """Deliver a batch of notifications."""
        for notification in batch.notifications:
            notification.status = NotificationStatus.DELIVERED
            notification.delivered_at = datetime.utcnow()
            self._notifications[notification.user_id].append(notification)

        logger.debug(
            f"Delivered batch {batch.batch_id} with {len(batch.notifications)} notifications"
        )

    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        notification_type: Optional[NotificationType] = None,
        limit: int = 50
    ) -> List[Notification]:
        """
        Get user's notifications.

        Args:
            user_id: User ID
            unread_only: Only return unread
            notification_type: Filter by type
            limit: Max notifications to return

        Returns:
            List of notifications
        """
        notifications = self._notifications.get(user_id, [])

        # Filter
        if unread_only:
            notifications = [n for n in notifications if n.status != NotificationStatus.READ]

        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]

        # Remove expired
        now = datetime.utcnow()
        notifications = [n for n in notifications if not n.expires_at or n.expires_at > now]

        # Sort by created_at descending
        notifications = sorted(notifications, key=lambda n: n.created_at, reverse=True)

        return notifications[:limit]

    def mark_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read."""
        notifications = self._notifications.get(user_id, [])
        for notif in notifications:
            if notif.notification_id == notification_id:
                notif.status = NotificationStatus.READ
                notif.read_at = datetime.utcnow()
                return True
        return False

    def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read."""
        notifications = self._notifications.get(user_id, [])
        count = 0
        now = datetime.utcnow()
        for notif in notifications:
            if notif.status != NotificationStatus.READ:
                notif.status = NotificationStatus.READ
                notif.read_at = now
                count += 1
        return count

    def dismiss(self, user_id: str, notification_id: str) -> bool:
        """Dismiss a notification."""
        notifications = self._notifications.get(user_id, [])
        for notif in notifications:
            if notif.notification_id == notification_id:
                notif.status = NotificationStatus.DISMISSED
                return True
        return False

    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications."""
        notifications = self._notifications.get(user_id, [])
        return sum(1 for n in notifications if n.status not in [NotificationStatus.READ, NotificationStatus.DISMISSED])

    def set_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> UserNotificationPreferences:
        """
        Set user notification preferences.

        Args:
            user_id: User ID
            preferences: Preferences dictionary

        Returns:
            Updated preferences
        """
        prefs = self._get_or_create_preferences(user_id)

        if "enabled_channels" in preferences:
            prefs.enabled_channels = [
                NotificationChannel(c) for c in preferences["enabled_channels"]
            ]

        if "enabled_types" in preferences:
            prefs.enabled_types = [
                NotificationType(t) for t in preferences["enabled_types"]
            ]

        if "quiet_hours_start" in preferences and preferences["quiet_hours_start"]:
            prefs.quiet_hours_start = time.fromisoformat(preferences["quiet_hours_start"])

        if "quiet_hours_end" in preferences and preferences["quiet_hours_end"]:
            prefs.quiet_hours_end = time.fromisoformat(preferences["quiet_hours_end"])

        if "email_digest_only" in preferences:
            prefs.email_digest_only = preferences["email_digest_only"]

        if "batch_non_urgent" in preferences:
            prefs.batch_non_urgent = preferences["batch_non_urgent"]

        if "timezone" in preferences:
            prefs.timezone = preferences["timezone"]

        self._preferences[user_id] = prefs
        logger.info(f"Updated notification preferences for user {user_id}")

        return prefs

    def get_preferences(self, user_id: str) -> UserNotificationPreferences:
        """Get user's notification preferences."""
        return self._get_or_create_preferences(user_id)

    def _get_or_create_preferences(self, user_id: str) -> UserNotificationPreferences:
        """Get or create default preferences."""
        if user_id not in self._preferences:
            self._preferences[user_id] = UserNotificationPreferences(
                user_id=user_id,
                enabled_channels=list(NotificationChannel),
                enabled_types=list(NotificationType)
            )
        return self._preferences[user_id]

    def notification_to_dict(self, notification: Notification) -> Dict[str, Any]:
        """Convert notification to dictionary for API."""
        return {
            "notification_id": notification.notification_id,
            "type": notification.notification_type.value,
            "priority": notification.priority.name.lower(),
            "title": notification.title,
            "body": notification.body,
            "status": notification.status.value,
            "created_at": notification.created_at.isoformat(),
            "delivered_at": notification.delivered_at.isoformat() if notification.delivered_at else None,
            "read_at": notification.read_at.isoformat() if notification.read_at else None,
            "action_url": notification.action_url,
            "channels": [c.value for c in notification.channels]
        }

    def preferences_to_dict(self, prefs: UserNotificationPreferences) -> Dict[str, Any]:
        """Convert preferences to dictionary for API."""
        return {
            "user_id": prefs.user_id,
            "enabled_channels": [c.value for c in prefs.enabled_channels],
            "enabled_types": [t.value for t in prefs.enabled_types],
            "quiet_hours_start": prefs.quiet_hours_start.isoformat() if prefs.quiet_hours_start else None,
            "quiet_hours_end": prefs.quiet_hours_end.isoformat() if prefs.quiet_hours_end else None,
            "email_digest_only": prefs.email_digest_only,
            "batch_non_urgent": prefs.batch_non_urgent,
            "timezone": prefs.timezone
        }


# Global instance
notification_service = NotificationService()
