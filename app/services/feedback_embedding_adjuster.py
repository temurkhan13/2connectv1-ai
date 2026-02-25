"""
Feedback-Driven Embedding Adjuster.

Implements the key improvement: embeddings are adjusted TOWARDS successful matches
and AWAY from unsuccessful ones, creating a self-improving matching system.

Key insight: Instead of just scaling embeddings uniformly, we:
1. Get the embedding of the matched user
2. Calculate the direction vector between users
3. Adjust the user's embedding towards/away from that direction

This creates a feedback loop where:
- Positive feedback on frank <-> grace → frank's embedding moves towards grace's
- Negative feedback → embeddings move apart
- Over time, the system learns what "good matches" look like for each user

Author: Claude Code
Date: February 2026
"""
import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.adapters.postgresql import postgresql_adapter
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingAdjustmentConfig:
    """Configuration for embedding adjustment."""
    # Learning rates
    positive_learning_rate: float = 0.05   # Move 5% towards good match
    negative_learning_rate: float = 0.03   # Move 3% away from bad match
    max_adjustment_per_feedback: float = 0.1  # Max 10% adjustment per feedback

    # Decay settings
    adjustment_decay: float = 0.95         # Each adjustment is 95% of previous

    # Normalization
    maintain_norm: bool = True             # Keep vector magnitude constant

    # History
    max_adjustment_history: int = 50       # Track last 50 adjustments

    @classmethod
    def from_env(cls) -> 'EmbeddingAdjustmentConfig':
        return cls(
            positive_learning_rate=float(os.getenv("POSITIVE_LEARNING_RATE", "0.05")),
            negative_learning_rate=float(os.getenv("NEGATIVE_LEARNING_RATE", "0.03")),
            max_adjustment_per_feedback=float(os.getenv("MAX_ADJUSTMENT_PER_FEEDBACK", "0.1")),
        )


class FeedbackType(str, Enum):
    """Types of feedback."""
    VERY_POSITIVE = "very_positive"   # "This was perfect!"
    POSITIVE = "positive"              # "Good match"
    NEUTRAL = "neutral"                # "It was okay"
    NEGATIVE = "negative"              # "Not quite right"
    VERY_NEGATIVE = "very_negative"    # "Terrible match"


# =============================================================================
# FEEDBACK EMBEDDING ADJUSTER
# =============================================================================

class FeedbackEmbeddingAdjuster:
    """
    Adjusts user embeddings based on match feedback.

    The key insight is that we don't just scale embeddings - we move them
    towards successful matches and away from unsuccessful ones.

    Example:
    - Frank (data scientist) matches with Grace (health-tech founder)
    - Frank gives positive feedback
    - Frank's "requirements" embedding moves TOWARDS Grace's "offerings" embedding
    - Frank's "offerings" embedding moves TOWARDS Grace's "requirements" embedding

    This creates a virtuous cycle where the more feedback users give,
    the better their future matches become.
    """

    def __init__(self, config: Optional[EmbeddingAdjustmentConfig] = None):
        self.config = config or EmbeddingAdjustmentConfig.from_env()
        self._adjustment_history: Dict[str, List[Dict]] = {}

    def process_match_feedback(
        self,
        user_id: str,
        matched_user_id: str,
        feedback_type: FeedbackType,
        feedback_text: str = "",
        adjust_both_users: bool = False
    ) -> Dict[str, Any]:
        """
        Process feedback and adjust embeddings accordingly.

        Args:
            user_id: User providing feedback
            matched_user_id: User being reviewed
            feedback_type: Type of feedback (positive/negative/etc)
            feedback_text: Optional text feedback for analysis
            adjust_both_users: Whether to also adjust the matched user's embeddings

        Returns:
            Dict with adjustment results
        """
        try:
            logger.info(
                f"Processing {feedback_type.value} feedback from {user_id} "
                f"about match with {matched_user_id}"
            )

            # Get both users' embeddings
            user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
            match_embeddings = postgresql_adapter.get_user_embeddings(matched_user_id)

            if not user_embeddings or not match_embeddings:
                return {
                    "success": False,
                    "message": "Could not retrieve embeddings for one or both users"
                }

            # Calculate adjustment direction and magnitude
            adjustment_magnitude = self._calculate_adjustment_magnitude(feedback_type)

            results = {
                "success": True,
                "user_id": user_id,
                "matched_user_id": matched_user_id,
                "feedback_type": feedback_type.value,
                "adjustment_magnitude": adjustment_magnitude,
                "adjustments": []
            }

            # Adjust user's requirements towards/away from match's offerings
            if user_embeddings.get('requirements') and match_embeddings.get('offerings'):
                req_result = self._adjust_embedding(
                    user_id=user_id,
                    embedding_type='requirements',
                    user_vector=user_embeddings['requirements']['vector_data'],
                    target_vector=match_embeddings['offerings']['vector_data'],
                    adjustment_magnitude=adjustment_magnitude,
                    is_positive=(adjustment_magnitude > 0)
                )
                results["adjustments"].append({
                    "type": "requirements_to_offerings",
                    **req_result
                })

            # Adjust user's offerings towards/away from match's requirements
            if user_embeddings.get('offerings') and match_embeddings.get('requirements'):
                off_result = self._adjust_embedding(
                    user_id=user_id,
                    embedding_type='offerings',
                    user_vector=user_embeddings['offerings']['vector_data'],
                    target_vector=match_embeddings['requirements']['vector_data'],
                    adjustment_magnitude=adjustment_magnitude,
                    is_positive=(adjustment_magnitude > 0)
                )
                results["adjustments"].append({
                    "type": "offerings_to_requirements",
                    **off_result
                })

            # Optionally adjust the matched user as well (symmetric learning)
            if adjust_both_users and adjustment_magnitude > 0:
                # For positive feedback, also move matched user towards this user
                # This creates symmetric learning
                reverse_magnitude = adjustment_magnitude * 0.5  # Half the adjustment

                if match_embeddings.get('requirements') and user_embeddings.get('offerings'):
                    self._adjust_embedding(
                        user_id=matched_user_id,
                        embedding_type='requirements',
                        user_vector=match_embeddings['requirements']['vector_data'],
                        target_vector=user_embeddings['offerings']['vector_data'],
                        adjustment_magnitude=reverse_magnitude,
                        is_positive=True
                    )

            # Record adjustment history
            self._record_adjustment(user_id, matched_user_id, feedback_type, adjustment_magnitude)

            logger.info(
                f"Applied {len(results['adjustments'])} embedding adjustments "
                f"for user {user_id} based on feedback"
            )

            return results

        except Exception as e:
            logger.error(f"Error processing match feedback: {e}")
            return {"success": False, "message": str(e)}

    def _calculate_adjustment_magnitude(self, feedback_type: FeedbackType) -> float:
        """
        Calculate adjustment magnitude based on feedback type.

        Returns positive value for positive feedback (move towards),
        negative value for negative feedback (move away).
        """
        magnitudes = {
            FeedbackType.VERY_POSITIVE: self.config.positive_learning_rate * 1.5,
            FeedbackType.POSITIVE: self.config.positive_learning_rate,
            FeedbackType.NEUTRAL: 0.0,  # No adjustment for neutral
            FeedbackType.NEGATIVE: -self.config.negative_learning_rate,
            FeedbackType.VERY_NEGATIVE: -self.config.negative_learning_rate * 1.5
        }

        magnitude = magnitudes.get(feedback_type, 0.0)

        # Clamp to max adjustment
        return max(
            -self.config.max_adjustment_per_feedback,
            min(self.config.max_adjustment_per_feedback, magnitude)
        )

    def _adjust_embedding(
        self,
        user_id: str,
        embedding_type: str,
        user_vector: List[float],
        target_vector: List[float],
        adjustment_magnitude: float,
        is_positive: bool
    ) -> Dict[str, Any]:
        """
        Adjust a single embedding towards or away from a target.

        For positive feedback:
            new_vector = old_vector + magnitude * (target - old_vector)
            = old_vector * (1 - magnitude) + target * magnitude

        For negative feedback:
            new_vector = old_vector - magnitude * (target - old_vector)
            = old_vector * (1 + magnitude) - target * magnitude

        This moves the vector along the direction towards/away from the target.
        """
        try:
            # Parse vectors if needed
            if isinstance(user_vector, str):
                import json
                user_vector = json.loads(user_vector)
            if isinstance(target_vector, str):
                import json
                target_vector = json.loads(target_vector)

            user_vector = [float(v) for v in user_vector]
            target_vector = [float(v) for v in target_vector]

            if len(user_vector) != len(target_vector):
                return {
                    "success": False,
                    "message": "Vector dimension mismatch"
                }

            # Calculate original norm for normalization later
            original_norm = self._vector_norm(user_vector)

            # Calculate direction vector (target - user)
            direction = [t - u for t, u in zip(target_vector, user_vector)]

            # Apply adjustment
            abs_magnitude = abs(adjustment_magnitude)

            if is_positive:
                # Move towards target
                adjusted_vector = [
                    u + abs_magnitude * d
                    for u, d in zip(user_vector, direction)
                ]
            else:
                # Move away from target
                adjusted_vector = [
                    u - abs_magnitude * d
                    for u, d in zip(user_vector, direction)
                ]

            # Normalize to maintain original magnitude
            if self.config.maintain_norm:
                adjusted_vector = self._normalize_to_norm(adjusted_vector, original_norm)

            # Store updated embedding
            success = postgresql_adapter.store_embedding(
                user_id=user_id,
                embedding_type=embedding_type,
                vector_data=adjusted_vector,
                metadata={
                    "feedback_adjusted": True,
                    "adjustment_type": "towards" if is_positive else "away",
                    "adjustment_magnitude": adjustment_magnitude,
                    "adjusted_at": datetime.utcnow().isoformat()
                }
            )

            # Calculate how much the embedding actually moved
            movement = self._vector_distance(user_vector, adjusted_vector)

            return {
                "success": success,
                "movement_distance": round(movement, 6),
                "direction": "towards" if is_positive else "away",
                "magnitude_applied": abs_magnitude
            }

        except Exception as e:
            logger.error(f"Error adjusting embedding: {e}")
            return {"success": False, "message": str(e)}

    def _vector_norm(self, vector: List[float]) -> float:
        """Calculate L2 norm of vector."""
        return math.sqrt(sum(v * v for v in vector))

    def _vector_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate Euclidean distance between vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _normalize_to_norm(self, vector: List[float], target_norm: float) -> List[float]:
        """Normalize vector to have specific L2 norm."""
        current_norm = self._vector_norm(vector)
        if current_norm == 0:
            return vector

        scale = target_norm / current_norm
        return [v * scale for v in vector]

    def _record_adjustment(
        self,
        user_id: str,
        matched_user_id: str,
        feedback_type: FeedbackType,
        magnitude: float
    ) -> None:
        """Record adjustment in history for analytics."""
        if user_id not in self._adjustment_history:
            self._adjustment_history[user_id] = []

        self._adjustment_history[user_id].append({
            "matched_user_id": matched_user_id,
            "feedback_type": feedback_type.value,
            "magnitude": magnitude,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Trim history to max size
        if len(self._adjustment_history[user_id]) > self.config.max_adjustment_history:
            self._adjustment_history[user_id] = \
                self._adjustment_history[user_id][-self.config.max_adjustment_history:]

    def get_adjustment_history(self, user_id: str) -> List[Dict]:
        """Get adjustment history for a user."""
        return self._adjustment_history.get(user_id, [])

    def get_adjustment_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about adjustments for a user."""
        history = self.get_adjustment_history(user_id)

        if not history:
            return {
                "total_adjustments": 0,
                "positive_count": 0,
                "negative_count": 0,
                "average_magnitude": 0
            }

        positive = [h for h in history if h["magnitude"] > 0]
        negative = [h for h in history if h["magnitude"] < 0]

        return {
            "total_adjustments": len(history),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "average_magnitude": sum(h["magnitude"] for h in history) / len(history),
            "last_adjustment": history[-1]["timestamp"] if history else None
        }


# =============================================================================
# INTEGRATION WITH FEEDBACK SERVICE
# =============================================================================

def classify_feedback_text(feedback_text: str) -> FeedbackType:
    """
    Classify feedback text into a FeedbackType.

    This is a simple keyword-based classifier. Could be enhanced with ML.
    """
    text_lower = feedback_text.lower()

    # Very positive indicators
    very_positive_keywords = [
        "perfect", "excellent", "amazing", "fantastic", "exactly",
        "love", "incredible", "best", "ideal", "wonderful"
    ]
    if any(kw in text_lower for kw in very_positive_keywords):
        return FeedbackType.VERY_POSITIVE

    # Very negative indicators
    very_negative_keywords = [
        "terrible", "awful", "horrible", "worst", "waste",
        "completely wrong", "disaster", "never", "hate"
    ]
    if any(kw in text_lower for kw in very_negative_keywords):
        return FeedbackType.VERY_NEGATIVE

    # Positive indicators
    positive_keywords = [
        "good", "great", "helpful", "nice", "useful",
        "interested", "relevant", "valuable", "like"
    ]
    if any(kw in text_lower for kw in positive_keywords):
        return FeedbackType.POSITIVE

    # Negative indicators
    negative_keywords = [
        "not great", "could be better", "mismatch", "wrong",
        "not interested", "not quite", "disappointing", "poor"
    ]
    if any(kw in text_lower for kw in negative_keywords):
        return FeedbackType.NEGATIVE

    # Default to neutral
    return FeedbackType.NEUTRAL


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

feedback_embedding_adjuster = FeedbackEmbeddingAdjuster()
