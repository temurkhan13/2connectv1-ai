"""
Feedback Collection and Learning Loop Service.

Comprehensive system for collecting, analyzing, and applying
user feedback to continuously improve matching quality.

Key features:
1. Structured feedback collection
2. Pattern analysis across feedback
3. Learning signal extraction
4. Model adjustment recommendations
5. Feedback effectiveness tracking
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

from app.services.feedback_learner import FeedbackLearner, FeedbackSentiment
from app.services.multi_vector_matcher import MatchTier

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    MATCH_RATING = "match_rating"          # Overall match quality
    CONNECTION_OUTCOME = "connection_outcome"  # What happened after connecting
    DIMENSION_FEEDBACK = "dimension_feedback"  # Feedback on specific dimensions
    SUGGESTION = "suggestion"              # User suggestions
    COMPLAINT = "complaint"                # Issues/complaints


class FeedbackRating(int, Enum):
    """Rating scale for feedback."""
    EXCELLENT = 5
    GOOD = 4
    OKAY = 3
    POOR = 2
    TERRIBLE = 1


class ConnectionOutcome(str, Enum):
    """Outcomes after users connect."""
    SUCCESSFUL_DEAL = "successful_deal"    # Business outcome achieved
    ONGOING_RELATIONSHIP = "ongoing_relationship"  # Continuing to talk
    VALUABLE_CONVERSATION = "valuable_conversation"  # Good chat, no more
    NO_RESPONSE = "no_response"            # Other party didn't respond
    MUTUAL_PASS = "mutual_pass"            # Both decided not to proceed
    NEGATIVE_EXPERIENCE = "negative_experience"  # Bad experience


@dataclass
class StructuredFeedback:
    """Structured feedback record."""
    feedback_id: str
    user_id: str
    match_user_id: str
    feedback_type: FeedbackType
    rating: Optional[FeedbackRating] = None
    outcome: Optional[ConnectionOutcome] = None
    dimension_ratings: Dict[str, int] = field(default_factory=dict)
    free_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    match_tier: Optional[MatchTier] = None
    match_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackPattern:
    """Detected pattern in feedback."""
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    affected_dimension: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class LearningSignal:
    """Signal extracted for model improvement."""
    signal_type: str
    dimension: str
    direction: str  # "increase_weight", "decrease_weight", "no_change"
    magnitude: float  # 0-1, how strong the signal
    evidence_count: int
    confidence: float


@dataclass
class FeedbackAnalytics:
    """Analytics summary for feedback."""
    period_start: datetime
    period_end: datetime
    total_feedback: int
    avg_rating: float
    rating_distribution: Dict[int, int]
    outcome_distribution: Dict[str, int]
    patterns_detected: List[FeedbackPattern]
    learning_signals: List[LearningSignal]
    tier_performance: Dict[str, Dict[str, Any]]


class FeedbackCollector:
    """
    Collects and validates user feedback.

    Provides structured forms for different feedback types
    and validates submissions.
    """

    def __init__(self):
        # Storage (in production, use database)
        self._feedback: List[StructuredFeedback] = []

        # Validation
        self.required_fields = {
            FeedbackType.MATCH_RATING: ["rating"],
            FeedbackType.CONNECTION_OUTCOME: ["outcome"],
            FeedbackType.DIMENSION_FEEDBACK: ["dimension_ratings"],
            FeedbackType.SUGGESTION: ["free_text"],
            FeedbackType.COMPLAINT: ["free_text"]
        }

    def collect_match_rating(
        self,
        user_id: str,
        match_user_id: str,
        rating: int,
        match_tier: Optional[MatchTier] = None,
        match_score: Optional[float] = None,
        comment: Optional[str] = None
    ) -> StructuredFeedback:
        """
        Collect rating feedback on a match.

        Args:
            user_id: User providing feedback
            match_user_id: User being rated
            rating: 1-5 rating
            match_tier: Tier of the match
            match_score: Original match score
            comment: Optional comment

        Returns:
            StructuredFeedback record
        """
        feedback = StructuredFeedback(
            feedback_id=f"fb_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            match_user_id=match_user_id,
            feedback_type=FeedbackType.MATCH_RATING,
            rating=FeedbackRating(min(5, max(1, rating))),
            free_text=comment,
            match_tier=match_tier,
            match_score=match_score
        )

        self._feedback.append(feedback)
        logger.info(f"Collected match rating from {user_id}: {rating}/5")

        return feedback

    def collect_connection_outcome(
        self,
        user_id: str,
        match_user_id: str,
        outcome: str,
        details: Optional[str] = None,
        match_tier: Optional[MatchTier] = None
    ) -> StructuredFeedback:
        """
        Collect outcome feedback after connection.

        Args:
            user_id: User providing feedback
            match_user_id: Connected user
            outcome: Connection outcome
            details: Optional details
            match_tier: Original match tier

        Returns:
            StructuredFeedback record
        """
        feedback = StructuredFeedback(
            feedback_id=f"fb_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            match_user_id=match_user_id,
            feedback_type=FeedbackType.CONNECTION_OUTCOME,
            outcome=ConnectionOutcome(outcome) if outcome in [o.value for o in ConnectionOutcome] else None,
            free_text=details,
            match_tier=match_tier
        )

        self._feedback.append(feedback)
        logger.info(f"Collected connection outcome from {user_id}: {outcome}")

        return feedback

    def collect_dimension_feedback(
        self,
        user_id: str,
        match_user_id: str,
        dimension_ratings: Dict[str, int],
        comment: Optional[str] = None
    ) -> StructuredFeedback:
        """
        Collect feedback on specific matching dimensions.

        Args:
            user_id: User providing feedback
            match_user_id: Matched user
            dimension_ratings: Rating per dimension (1-5)
            comment: Optional comment

        Returns:
            StructuredFeedback record
        """
        # Validate ratings
        validated_ratings = {
            k: min(5, max(1, v)) for k, v in dimension_ratings.items()
        }

        feedback = StructuredFeedback(
            feedback_id=f"fb_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            match_user_id=match_user_id,
            feedback_type=FeedbackType.DIMENSION_FEEDBACK,
            dimension_ratings=validated_ratings,
            free_text=comment
        )

        self._feedback.append(feedback)
        logger.info(
            f"Collected dimension feedback from {user_id}: "
            f"{len(dimension_ratings)} dimensions rated"
        )

        return feedback

    def get_user_feedback(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[StructuredFeedback]:
        """Get all feedback from a user."""
        return [
            f for f in self._feedback
            if f.user_id == user_id
        ][-limit:]

    def get_feedback_for_match(
        self,
        user_id: str,
        match_user_id: str
    ) -> List[StructuredFeedback]:
        """Get all feedback for a specific match."""
        return [
            f for f in self._feedback
            if f.user_id == user_id and f.match_user_id == match_user_id
        ]


class FeedbackAnalyzer:
    """
    Analyzes feedback to extract patterns and learning signals.

    Identifies systematic issues and opportunities for
    algorithm improvement.
    """

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def analyze_period(
        self,
        days: int = 30
    ) -> FeedbackAnalytics:
        """
        Analyze feedback over a time period.

        Args:
            days: Number of days to analyze

        Returns:
            FeedbackAnalytics summary
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [f for f in self.collector._feedback if f.timestamp >= cutoff]

        # Basic stats
        ratings = [f.rating.value for f in recent if f.rating]
        avg_rating = statistics.mean(ratings) if ratings else 0.0

        rating_dist = defaultdict(int)
        for r in ratings:
            rating_dist[r] += 1

        outcome_dist = defaultdict(int)
        for f in recent:
            if f.outcome:
                outcome_dist[f.outcome.value] += 1

        # Detect patterns
        patterns = self._detect_patterns(recent)

        # Extract learning signals
        signals = self._extract_learning_signals(recent)

        # Analyze by tier
        tier_perf = self._analyze_by_tier(recent)

        return FeedbackAnalytics(
            period_start=cutoff,
            period_end=datetime.utcnow(),
            total_feedback=len(recent),
            avg_rating=avg_rating,
            rating_distribution=dict(rating_dist),
            outcome_distribution=dict(outcome_dist),
            patterns_detected=patterns,
            learning_signals=signals,
            tier_performance=tier_perf
        )

    def _detect_patterns(
        self,
        feedback: List[StructuredFeedback]
    ) -> List[FeedbackPattern]:
        """Detect patterns in feedback."""
        patterns = []

        # Pattern: Low ratings correlated with specific dimension
        dim_ratings = defaultdict(list)
        for f in feedback:
            if f.dimension_ratings:
                for dim, rating in f.dimension_ratings.items():
                    dim_ratings[dim].append(rating)

        for dim, ratings in dim_ratings.items():
            if len(ratings) >= 5:
                avg = statistics.mean(ratings)
                if avg < 3.0:
                    patterns.append(FeedbackPattern(
                        pattern_type="low_dimension_satisfaction",
                        description=f"Users consistently rate {dim} matching low",
                        frequency=len([r for r in ratings if r < 3]),
                        confidence=0.7,
                        affected_dimension=dim,
                        recommendation=f"Review {dim} matching algorithm"
                    ))

        # Pattern: High tier matches with low ratings
        tier_ratings = defaultdict(list)
        for f in feedback:
            if f.match_tier and f.rating:
                tier_ratings[f.match_tier.value].append(f.rating.value)

        for tier, ratings in tier_ratings.items():
            if len(ratings) >= 5:
                avg = statistics.mean(ratings)
                if tier in ["perfect", "strong"] and avg < 3.5:
                    patterns.append(FeedbackPattern(
                        pattern_type="tier_mismatch",
                        description=f"{tier.title()} tier matches receiving poor ratings",
                        frequency=len([r for r in ratings if r < 3]),
                        confidence=0.8,
                        recommendation="Recalibrate tier thresholds"
                    ))

        # Pattern: Connection outcomes
        outcomes = [f.outcome.value for f in feedback if f.outcome]
        if outcomes:
            no_response_rate = outcomes.count("no_response") / len(outcomes)
            if no_response_rate > 0.3:
                patterns.append(FeedbackPattern(
                    pattern_type="high_no_response",
                    description="High rate of non-responses after matching",
                    frequency=int(no_response_rate * 100),
                    confidence=0.75,
                    recommendation="Consider improving ice breakers or timing"
                ))

        return patterns

    def _extract_learning_signals(
        self,
        feedback: List[StructuredFeedback]
    ) -> List[LearningSignal]:
        """Extract signals for model improvement."""
        signals = []

        # Aggregate dimension feedback
        dim_feedback = defaultdict(list)
        for f in feedback:
            if f.dimension_ratings:
                for dim, rating in f.dimension_ratings.items():
                    dim_feedback[dim].append(rating)

        for dim, ratings in dim_feedback.items():
            if len(ratings) >= 3:
                avg = statistics.mean(ratings)

                # Determine direction and magnitude
                if avg >= 4.0:
                    direction = "increase_weight"
                    magnitude = (avg - 3) / 2  # 0.5 to 1.0
                elif avg <= 2.0:
                    direction = "decrease_weight"
                    magnitude = (3 - avg) / 2  # 0.5 to 1.0
                else:
                    direction = "no_change"
                    magnitude = 0.0

                if direction != "no_change":
                    signals.append(LearningSignal(
                        signal_type="weight_adjustment",
                        dimension=dim,
                        direction=direction,
                        magnitude=magnitude,
                        evidence_count=len(ratings),
                        confidence=min(0.9, 0.5 + (len(ratings) * 0.05))
                    ))

        return signals

    def _analyze_by_tier(
        self,
        feedback: List[StructuredFeedback]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by match tier."""
        tier_data = defaultdict(lambda: {
            "count": 0, "ratings": [], "outcomes": []
        })

        for f in feedback:
            if f.match_tier:
                tier = f.match_tier.value
                tier_data[tier]["count"] += 1
                if f.rating:
                    tier_data[tier]["ratings"].append(f.rating.value)
                if f.outcome:
                    tier_data[tier]["outcomes"].append(f.outcome.value)

        result = {}
        for tier, data in tier_data.items():
            ratings = data["ratings"]
            outcomes = data["outcomes"]

            positive_outcomes = [
                "successful_deal", "ongoing_relationship", "valuable_conversation"
            ]
            positive_count = sum(1 for o in outcomes if o in positive_outcomes)

            result[tier] = {
                "match_count": data["count"],
                "avg_rating": statistics.mean(ratings) if ratings else 0,
                "positive_outcome_rate": positive_count / len(outcomes) if outcomes else 0,
                "feedback_count": len(ratings) + len(outcomes)
            }

        return result


class FeedbackLoop:
    """
    Main feedback loop service.

    Coordinates feedback collection, analysis, and application
    of learnings to improve matching.
    """

    def __init__(self):
        self.collector = FeedbackCollector()
        self.analyzer = FeedbackAnalyzer(self.collector)
        self.learner = FeedbackLearner()

        # Configuration
        self.auto_apply_threshold = float(
            os.getenv("FEEDBACK_AUTO_APPLY_THRESHOLD", "0.8")
        )
        self.min_samples_for_learning = int(
            os.getenv("MIN_FEEDBACK_SAMPLES", "10")
        )

    def submit_match_feedback(
        self,
        user_id: str,
        match_user_id: str,
        rating: int,
        comment: Optional[str] = None,
        match_tier: Optional[str] = None,
        match_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit and process match feedback.

        Args:
            user_id: User providing feedback
            match_user_id: Matched user
            rating: 1-5 rating
            comment: Optional comment
            match_tier: Original match tier
            match_score: Original match score

        Returns:
            Processing result
        """
        tier = MatchTier(match_tier) if match_tier else None

        # Collect the feedback
        feedback = self.collector.collect_match_rating(
            user_id=user_id,
            match_user_id=match_user_id,
            rating=rating,
            match_tier=tier,
            match_score=match_score,
            comment=comment
        )

        # Apply to user's embeddings if significant
        if rating <= 2 or rating >= 4:
            sentiment = "positive" if rating >= 4 else "negative"
            self.learner.process_feedback(
                user_id=user_id,
                feedback_text=comment or f"Match rated {rating}/5",
                feedback_type="match",
                match_context={"match_user_id": match_user_id, "tier": match_tier}
            )

        return {
            "success": True,
            "feedback_id": feedback.feedback_id,
            "applied_learning": rating <= 2 or rating >= 4
        }

    def submit_outcome_feedback(
        self,
        user_id: str,
        match_user_id: str,
        outcome: str,
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit connection outcome feedback."""
        feedback = self.collector.collect_connection_outcome(
            user_id=user_id,
            match_user_id=match_user_id,
            outcome=outcome,
            details=details
        )

        # Apply stronger learning for definitive outcomes
        if outcome in ["successful_deal", "ongoing_relationship"]:
            self.learner.process_feedback(
                user_id=user_id,
                feedback_text=f"Successful outcome: {outcome}",
                feedback_type="match",
                match_context={"outcome": outcome}
            )

        return {
            "success": True,
            "feedback_id": feedback.feedback_id
        }

    def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback analytics."""
        analytics = self.analyzer.analyze_period(days)

        return {
            "period": {
                "start": analytics.period_start.isoformat(),
                "end": analytics.period_end.isoformat()
            },
            "summary": {
                "total_feedback": analytics.total_feedback,
                "avg_rating": round(analytics.avg_rating, 2)
            },
            "rating_distribution": analytics.rating_distribution,
            "outcome_distribution": analytics.outcome_distribution,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                    "confidence": round(p.confidence, 2),
                    "recommendation": p.recommendation
                }
                for p in analytics.patterns_detected
            ],
            "learning_signals": [
                {
                    "type": s.signal_type,
                    "dimension": s.dimension,
                    "direction": s.direction,
                    "magnitude": round(s.magnitude, 2),
                    "evidence_count": s.evidence_count,
                    "confidence": round(s.confidence, 2)
                }
                for s in analytics.learning_signals
            ],
            "tier_performance": analytics.tier_performance
        }

    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for matching improvement."""
        analytics = self.analyzer.analyze_period(30)
        recommendations = []

        # From patterns
        for pattern in analytics.patterns_detected:
            if pattern.recommendation and pattern.confidence >= 0.7:
                recommendations.append({
                    "type": "pattern_based",
                    "priority": "high" if pattern.confidence >= 0.8 else "medium",
                    "issue": pattern.description,
                    "recommendation": pattern.recommendation,
                    "confidence": pattern.confidence
                })

        # From learning signals
        for signal in analytics.learning_signals:
            if signal.magnitude >= 0.5 and signal.evidence_count >= 5:
                recommendations.append({
                    "type": "weight_adjustment",
                    "priority": "medium",
                    "dimension": signal.dimension,
                    "direction": signal.direction,
                    "magnitude": signal.magnitude,
                    "recommendation": f"Consider {signal.direction.replace('_', ' ')} "
                                    f"for {signal.dimension} dimension"
                })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 2))

        return recommendations

    def close_loop(self) -> Dict[str, Any]:
        """
        Close the feedback loop by applying accumulated learnings.

        This should be called periodically (e.g., daily) to
        apply feedback learnings to the system.
        """
        analytics = self.analyzer.analyze_period(7)  # Last week

        applied_changes = []

        # Apply high-confidence learning signals
        for signal in analytics.learning_signals:
            if (signal.confidence >= self.auto_apply_threshold and
                signal.evidence_count >= self.min_samples_for_learning):

                # Log the change (in production, this would modify config)
                applied_changes.append({
                    "dimension": signal.dimension,
                    "change": signal.direction,
                    "magnitude": signal.magnitude
                })
                logger.info(
                    f"Applied learning: {signal.direction} for {signal.dimension} "
                    f"(magnitude: {signal.magnitude:.2f})"
                )

        return {
            "loop_closed": True,
            "timestamp": datetime.utcnow().isoformat(),
            "changes_applied": len(applied_changes),
            "changes": applied_changes,
            "total_feedback_processed": analytics.total_feedback
        }


# Global instance
feedback_loop = FeedbackLoop()
