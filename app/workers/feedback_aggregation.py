"""
Feedback Aggregation Worker.

Celery task that runs periodically to:
1. Aggregate match_feedback patterns per user
2. Learn user preferences from approve/decline decisions
3. Update user_preferences_learned table

Phase 2.1: Feedback Learning Loop (Pattern Learning Component)
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor

from app.celery_app import celery_app

logger = logging.getLogger(__name__)

# Minimum feedback records needed before learning patterns
MIN_FEEDBACK_THRESHOLD = 5

# Preference types we track
PREFERENCE_TYPES = [
    "industry",
    "company_stage",
    "geography",
    "engagement_style",
    "expertise_area"
]


def get_backend_connection():
    """Get connection to backend database (where match_feedback lives)."""
    db_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
    if not db_url:
        raise ValueError("RECIPROCITY_BACKEND_DB_URL not configured")
    return psycopg2.connect(db_url)


@celery_app.task(
    name="feedback_aggregation_task",
    bind=True,
    max_retries=3,
    default_retry_delay=300
)
def feedback_aggregation_task(self) -> Dict[str, Any]:
    """
    Aggregate feedback patterns and update learned preferences.

    This task should run weekly via Celery Beat.
    """
    logger.info("Starting feedback aggregation task")
    start_time = datetime.utcnow()

    results = {
        "users_processed": 0,
        "preferences_updated": 0,
        "skipped_insufficient_data": 0,
        "errors": []
    }

    try:
        conn = get_backend_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get users with sufficient feedback
        cursor.execute("""
            SELECT
                user_id,
                COUNT(*) as feedback_count,
                COUNT(*) FILTER (WHERE decision = 'approved') as approved_count,
                COUNT(*) FILTER (WHERE decision = 'declined') as declined_count
            FROM match_feedback
            GROUP BY user_id
            HAVING COUNT(*) >= %s
        """, (MIN_FEEDBACK_THRESHOLD,))

        users_to_process = cursor.fetchall()
        logger.info(f"Found {len(users_to_process)} users with sufficient feedback")

        for user_row in users_to_process:
            user_id = str(user_row['user_id'])
            try:
                # Get detailed feedback for this user
                cursor.execute("""
                    SELECT
                        decision,
                        reason_tags,
                        other_user_attributes,
                        created_at
                    FROM match_feedback
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (user_id,))

                feedback_records = cursor.fetchall()

                # Analyze patterns
                patterns = analyze_feedback_patterns(feedback_records)

                # Update preferences
                for pref_type, pattern_data in patterns.items():
                    if pattern_data['sample_count'] >= 3:  # Need at least 3 samples
                        update_user_preference(
                            cursor,
                            user_id,
                            pref_type,
                            pattern_data
                        )
                        results["preferences_updated"] += 1

                results["users_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                results["errors"].append({
                    "user_id": user_id,
                    "error": str(e)
                })

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Feedback aggregation failed: {e}")
        results["errors"].append({"global": str(e)})

    results["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Feedback aggregation complete: {results}")

    return results


def analyze_feedback_patterns(feedback_records: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze feedback records to extract preference patterns.

    Returns dict of preference_type -> {
        positive_patterns: [...],
        negative_patterns: [...],
        confidence: float,
        sample_count: int
    }
    """
    patterns = {}

    # Track attribute occurrences per decision
    approved_attributes = defaultdict(lambda: defaultdict(int))
    declined_attributes = defaultdict(lambda: defaultdict(int))
    declined_reasons = defaultdict(int)

    for record in feedback_records:
        decision = record.get('decision', '')
        reason_tags = record.get('reason_tags') or []
        other_attrs = record.get('other_user_attributes') or {}

        # Count reason tags for declines
        if decision == 'declined':
            for tag in reason_tags:
                declined_reasons[tag] += 1

        # Extract attributes from other_user_attributes
        if other_attrs:
            attrs_dict = approved_attributes if decision == 'approved' else declined_attributes

            # Extract common attributes
            if 'industry' in other_attrs:
                attrs_dict['industry'][other_attrs['industry']] += 1
            if 'company_stage' in other_attrs:
                attrs_dict['company_stage'][other_attrs['company_stage']] += 1
            if 'location' in other_attrs:
                attrs_dict['geography'][other_attrs['location']] += 1

    # Build patterns for each preference type
    for pref_type in PREFERENCE_TYPES:
        positive = dict(approved_attributes.get(pref_type, {}))
        negative = dict(declined_attributes.get(pref_type, {}))

        total_samples = sum(positive.values()) + sum(negative.values())

        if total_samples > 0:
            # Calculate confidence based on consistency
            if positive or negative:
                # Higher confidence if patterns are clear (e.g., always decline certain industries)
                max_neg = max(negative.values()) if negative else 0
                max_pos = max(positive.values()) if positive else 0
                total = max_neg + max_pos
                confidence = (max(max_neg, max_pos) / total) if total > 0 else 0.5
            else:
                confidence = 0.5

            patterns[pref_type] = {
                "positive_patterns": [
                    {"value": k, "count": v, "weight": v / max(sum(positive.values()), 1)}
                    for k, v in sorted(positive.items(), key=lambda x: -x[1])[:10]
                ],
                "negative_patterns": [
                    {"value": k, "count": v, "weight": v / max(sum(negative.values()), 1)}
                    for k, v in sorted(negative.items(), key=lambda x: -x[1])[:10]
                ],
                "confidence": min(0.95, confidence),
                "sample_count": total_samples
            }

    # Add reason_tags as a special preference type
    if declined_reasons:
        total_declines = sum(declined_reasons.values())
        patterns["decline_reasons"] = {
            "positive_patterns": [],
            "negative_patterns": [
                {"value": k, "count": v, "weight": v / total_declines}
                for k, v in sorted(declined_reasons.items(), key=lambda x: -x[1])
            ],
            "confidence": 0.8,
            "sample_count": total_declines
        }

    return patterns


def update_user_preference(
    cursor,
    user_id: str,
    preference_type: str,
    pattern_data: Dict
) -> bool:
    """
    Upsert user preference to user_preferences_learned table.
    """
    try:
        cursor.execute("""
            INSERT INTO user_preferences_learned (
                id, user_id, preference_type,
                positive_patterns, negative_patterns,
                confidence, sample_count,
                last_trained_at, created_at, updated_at
            ) VALUES (
                gen_random_uuid(), %s, %s,
                %s, %s,
                %s, %s,
                NOW(), NOW(), NOW()
            )
            ON CONFLICT (user_id, preference_type) DO UPDATE SET
                positive_patterns = EXCLUDED.positive_patterns,
                negative_patterns = EXCLUDED.negative_patterns,
                confidence = EXCLUDED.confidence,
                sample_count = EXCLUDED.sample_count,
                last_trained_at = NOW(),
                updated_at = NOW()
        """, (
            user_id,
            preference_type,
            json.dumps(pattern_data.get('positive_patterns', [])),
            json.dumps(pattern_data.get('negative_patterns', [])),
            pattern_data.get('confidence', 0.5),
            pattern_data.get('sample_count', 0)
        ))

        logger.debug(f"Updated preference {preference_type} for user {user_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to update preference: {e}")
        return False


@celery_app.task(name="check_feedback_data_health")
def check_feedback_data_health() -> Dict[str, Any]:
    """
    Health check for feedback data.
    Returns counts and stats for monitoring.
    """
    try:
        conn = get_backend_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_feedback,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(*) FILTER (WHERE decision = 'approved') as approved_count,
                COUNT(*) FILTER (WHERE decision = 'declined') as declined_count,
                COUNT(*) FILTER (WHERE array_length(reason_tags, 1) > 0) as with_reasons,
                MIN(created_at) as oldest_feedback,
                MAX(created_at) as newest_feedback
            FROM match_feedback
        """)

        stats = cursor.fetchone()

        # Get users ready for learning (5+ feedback)
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id) as ready_users
            FROM match_feedback
            GROUP BY user_id
            HAVING COUNT(*) >= %s
        """, (MIN_FEEDBACK_THRESHOLD,))

        ready_result = cursor.fetchone()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "total_feedback": stats['total_feedback'],
            "unique_users": stats['unique_users'],
            "approved_count": stats['approved_count'],
            "declined_count": stats['declined_count'],
            "with_reasons": stats['with_reasons'],
            "oldest_feedback": str(stats['oldest_feedback']) if stats['oldest_feedback'] else None,
            "newest_feedback": str(stats['newest_feedback']) if stats['newest_feedback'] else None,
            "users_ready_for_learning": ready_result['ready_users'] if ready_result else 0,
            "min_threshold": MIN_FEEDBACK_THRESHOLD
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
