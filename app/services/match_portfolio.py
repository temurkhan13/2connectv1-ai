"""
Match Portfolio Service.

Manages a user's collection of matches with filtering,
sorting, status tracking, and portfolio analytics.

Key features:
1. Match collection management
2. Status tracking (new, viewed, connected, archived)
3. Filtering and sorting
4. Portfolio analytics
5. Saved searches and preferences
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

from app.services.multi_vector_matcher import MatchTier

logger = logging.getLogger(__name__)


class MatchStatus(str, Enum):
    """Status of a match in user's portfolio."""
    NEW = "new"                      # Just matched, not viewed
    VIEWED = "viewed"                # User has seen the card
    SAVED = "saved"                  # User saved for later
    CONNECTED = "connected"          # Users are in conversation
    MEETING_SCHEDULED = "meeting_scheduled"  # Call/meeting planned
    COMPLETED = "completed"          # Outcome reached (deal, partnership, etc.)
    ARCHIVED = "archived"            # User dismissed/archived
    BLOCKED = "blocked"              # User blocked this match


class SortOption(str, Enum):
    """Options for sorting matches."""
    SCORE_DESC = "score_desc"        # Highest score first
    SCORE_ASC = "score_asc"          # Lowest score first
    NEWEST = "newest"                # Most recent first
    OLDEST = "oldest"                # Oldest first
    LAST_ACTIVITY = "last_activity"  # Most recent activity
    STATUS = "status"                # By status


class FilterCriteria(str, Enum):
    """Criteria for filtering matches."""
    STATUS = "status"
    TIER = "tier"
    SCORE_MIN = "score_min"
    SCORE_MAX = "score_max"
    DATE_AFTER = "date_after"
    DATE_BEFORE = "date_before"
    HAS_CONVERSATION = "has_conversation"


@dataclass
class PortfolioMatch:
    """A match in user's portfolio."""
    match_id: str
    user_id: str  # Portfolio owner
    match_user_id: str  # Matched user
    status: MatchStatus
    tier: MatchTier
    score: float
    matched_at: datetime
    last_activity: datetime
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioStats:
    """Statistics for a user's portfolio."""
    total_matches: int
    matches_by_status: Dict[str, int]
    matches_by_tier: Dict[str, int]
    avg_score: float
    connection_rate: float
    active_conversations: int
    archived_count: int
    new_this_week: int
    new_this_month: int


@dataclass
class PortfolioView:
    """View of filtered/sorted portfolio."""
    user_id: str
    matches: List[PortfolioMatch]
    total_count: int
    page: int
    page_size: int
    filters_applied: Dict[str, Any]
    sort_applied: SortOption


class MatchPortfolio:
    """
    Manages a user's match portfolio.

    Provides CRUD operations, filtering, sorting,
    and analytics for match collections.
    """

    def __init__(self):
        # Storage (in production, use database)
        self._portfolios: Dict[str, List[PortfolioMatch]] = defaultdict(list)

        # Configuration
        self.default_page_size = int(os.getenv("PORTFOLIO_PAGE_SIZE", "20"))
        self.max_page_size = int(os.getenv("PORTFOLIO_MAX_PAGE_SIZE", "100"))

    def add_match(
        self,
        user_id: str,
        match_user_id: str,
        tier: MatchTier,
        score: float,
        metadata: Optional[Dict] = None
    ) -> PortfolioMatch:
        """
        Add a new match to user's portfolio.

        Args:
            user_id: Portfolio owner
            match_user_id: Matched user
            tier: Match tier
            score: Match score
            metadata: Optional metadata

        Returns:
            PortfolioMatch record
        """
        match_id = f"pm_{user_id}_{match_user_id}_{datetime.utcnow().timestamp()}"
        now = datetime.utcnow()

        match = PortfolioMatch(
            match_id=match_id,
            user_id=user_id,
            match_user_id=match_user_id,
            status=MatchStatus.NEW,
            tier=tier,
            score=score,
            matched_at=now,
            last_activity=now,
            metadata=metadata or {}
        )

        self._portfolios[user_id].append(match)
        logger.info(f"Added match {match_id} to portfolio for user {user_id}")

        return match

    def get_match(
        self,
        user_id: str,
        match_user_id: str
    ) -> Optional[PortfolioMatch]:
        """Get a specific match from portfolio."""
        matches = self._portfolios.get(user_id, [])
        for match in matches:
            if match.match_user_id == match_user_id:
                return match
        return None

    def update_status(
        self,
        user_id: str,
        match_user_id: str,
        new_status: MatchStatus
    ) -> bool:
        """
        Update match status.

        Args:
            user_id: Portfolio owner
            match_user_id: Matched user
            new_status: New status

        Returns:
            True if updated
        """
        match = self.get_match(user_id, match_user_id)
        if match:
            match.status = new_status
            match.last_activity = datetime.utcnow()
            logger.info(f"Updated match {match.match_id} status to {new_status.value}")
            return True
        return False

    def add_note(
        self,
        user_id: str,
        match_user_id: str,
        note: str
    ) -> bool:
        """Add a note to a match."""
        match = self.get_match(user_id, match_user_id)
        if match:
            match.notes = note
            match.last_activity = datetime.utcnow()
            return True
        return False

    def add_tag(
        self,
        user_id: str,
        match_user_id: str,
        tag: str
    ) -> bool:
        """Add a tag to a match."""
        match = self.get_match(user_id, match_user_id)
        if match and tag not in match.tags:
            match.tags.append(tag)
            match.last_activity = datetime.utcnow()
            return True
        return False

    def remove_tag(
        self,
        user_id: str,
        match_user_id: str,
        tag: str
    ) -> bool:
        """Remove a tag from a match."""
        match = self.get_match(user_id, match_user_id)
        if match and tag in match.tags:
            match.tags.remove(tag)
            return True
        return False

    def link_conversation(
        self,
        user_id: str,
        match_user_id: str,
        conversation_id: str
    ) -> bool:
        """Link a conversation to a match."""
        match = self.get_match(user_id, match_user_id)
        if match:
            match.conversation_id = conversation_id
            if match.status == MatchStatus.NEW or match.status == MatchStatus.VIEWED:
                match.status = MatchStatus.CONNECTED
            match.last_activity = datetime.utcnow()
            return True
        return False

    def get_portfolio(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: SortOption = SortOption.NEWEST,
        page: int = 1,
        page_size: Optional[int] = None
    ) -> PortfolioView:
        """
        Get filtered and sorted portfolio view.

        Args:
            user_id: Portfolio owner
            filters: Filter criteria
            sort_by: Sort option
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            PortfolioView with matches
        """
        page_size = min(page_size or self.default_page_size, self.max_page_size)
        matches = self._portfolios.get(user_id, []).copy()

        # Apply filters
        if filters:
            matches = self._apply_filters(matches, filters)

        # Apply sorting
        matches = self._apply_sort(matches, sort_by)

        # Paginate
        total_count = len(matches)
        start = (page - 1) * page_size
        end = start + page_size
        page_matches = matches[start:end]

        return PortfolioView(
            user_id=user_id,
            matches=page_matches,
            total_count=total_count,
            page=page,
            page_size=page_size,
            filters_applied=filters or {},
            sort_applied=sort_by
        )

    def _apply_filters(
        self,
        matches: List[PortfolioMatch],
        filters: Dict[str, Any]
    ) -> List[PortfolioMatch]:
        """Apply filters to matches."""
        result = matches

        # Status filter
        if FilterCriteria.STATUS.value in filters:
            status_values = filters[FilterCriteria.STATUS.value]
            if isinstance(status_values, str):
                status_values = [status_values]
            result = [m for m in result if m.status.value in status_values]

        # Tier filter
        if FilterCriteria.TIER.value in filters:
            tier_values = filters[FilterCriteria.TIER.value]
            if isinstance(tier_values, str):
                tier_values = [tier_values]
            result = [m for m in result if m.tier.value in tier_values]

        # Score filters
        if FilterCriteria.SCORE_MIN.value in filters:
            min_score = float(filters[FilterCriteria.SCORE_MIN.value])
            result = [m for m in result if m.score >= min_score]

        if FilterCriteria.SCORE_MAX.value in filters:
            max_score = float(filters[FilterCriteria.SCORE_MAX.value])
            result = [m for m in result if m.score <= max_score]

        # Date filters
        if FilterCriteria.DATE_AFTER.value in filters:
            after = datetime.fromisoformat(filters[FilterCriteria.DATE_AFTER.value])
            result = [m for m in result if m.matched_at >= after]

        if FilterCriteria.DATE_BEFORE.value in filters:
            before = datetime.fromisoformat(filters[FilterCriteria.DATE_BEFORE.value])
            result = [m for m in result if m.matched_at <= before]

        # Conversation filter
        if FilterCriteria.HAS_CONVERSATION.value in filters:
            has_conv = filters[FilterCriteria.HAS_CONVERSATION.value]
            if has_conv:
                result = [m for m in result if m.conversation_id]
            else:
                result = [m for m in result if not m.conversation_id]

        return result

    def _apply_sort(
        self,
        matches: List[PortfolioMatch],
        sort_by: SortOption
    ) -> List[PortfolioMatch]:
        """Apply sorting to matches."""
        if sort_by == SortOption.SCORE_DESC:
            return sorted(matches, key=lambda m: m.score, reverse=True)
        elif sort_by == SortOption.SCORE_ASC:
            return sorted(matches, key=lambda m: m.score)
        elif sort_by == SortOption.NEWEST:
            return sorted(matches, key=lambda m: m.matched_at, reverse=True)
        elif sort_by == SortOption.OLDEST:
            return sorted(matches, key=lambda m: m.matched_at)
        elif sort_by == SortOption.LAST_ACTIVITY:
            return sorted(matches, key=lambda m: m.last_activity, reverse=True)
        elif sort_by == SortOption.STATUS:
            status_order = {s: i for i, s in enumerate(MatchStatus)}
            return sorted(matches, key=lambda m: status_order.get(m.status, 99))
        return matches

    def get_stats(self, user_id: str) -> PortfolioStats:
        """
        Get portfolio statistics.

        Args:
            user_id: Portfolio owner

        Returns:
            PortfolioStats summary
        """
        matches = self._portfolios.get(user_id, [])
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        # Count by status
        status_counts = defaultdict(int)
        for m in matches:
            status_counts[m.status.value] += 1

        # Count by tier
        tier_counts = defaultdict(int)
        for m in matches:
            tier_counts[m.tier.value] += 1

        # Calculate averages
        scores = [m.score for m in matches]
        avg_score = statistics.mean(scores) if scores else 0.0

        # Connection rate
        connected_statuses = [MatchStatus.CONNECTED, MatchStatus.MEETING_SCHEDULED, MatchStatus.COMPLETED]
        connected = sum(1 for m in matches if m.status in connected_statuses)
        connection_rate = connected / len(matches) if matches else 0.0

        # Active conversations
        active = sum(1 for m in matches if m.conversation_id and m.status == MatchStatus.CONNECTED)

        # New counts
        new_week = sum(1 for m in matches if m.matched_at >= week_ago)
        new_month = sum(1 for m in matches if m.matched_at >= month_ago)

        return PortfolioStats(
            total_matches=len(matches),
            matches_by_status=dict(status_counts),
            matches_by_tier=dict(tier_counts),
            avg_score=avg_score,
            connection_rate=connection_rate,
            active_conversations=active,
            archived_count=status_counts.get(MatchStatus.ARCHIVED.value, 0),
            new_this_week=new_week,
            new_this_month=new_month
        )

    def get_by_tag(self, user_id: str, tag: str) -> List[PortfolioMatch]:
        """Get matches by tag."""
        matches = self._portfolios.get(user_id, [])
        return [m for m in matches if tag in m.tags]

    def get_all_tags(self, user_id: str) -> List[Tuple[str, int]]:
        """Get all tags with counts."""
        matches = self._portfolios.get(user_id, [])
        tag_counts = defaultdict(int)
        for m in matches:
            for tag in m.tags:
                tag_counts[tag] += 1
        return sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

    def bulk_update_status(
        self,
        user_id: str,
        match_user_ids: List[str],
        new_status: MatchStatus
    ) -> int:
        """
        Bulk update status for multiple matches.

        Args:
            user_id: Portfolio owner
            match_user_ids: List of match user IDs
            new_status: New status to apply

        Returns:
            Number of matches updated
        """
        updated = 0
        for match_user_id in match_user_ids:
            if self.update_status(user_id, match_user_id, new_status):
                updated += 1
        return updated

    def archive_old_matches(
        self,
        user_id: str,
        days_inactive: int = 90
    ) -> int:
        """
        Archive matches that have been inactive.

        Args:
            user_id: Portfolio owner
            days_inactive: Days of inactivity threshold

        Returns:
            Number of matches archived
        """
        cutoff = datetime.utcnow() - timedelta(days=days_inactive)
        matches = self._portfolios.get(user_id, [])
        archived = 0

        for match in matches:
            if (match.last_activity < cutoff and
                match.status not in [MatchStatus.ARCHIVED, MatchStatus.BLOCKED, MatchStatus.COMPLETED]):
                match.status = MatchStatus.ARCHIVED
                archived += 1

        if archived:
            logger.info(f"Archived {archived} inactive matches for user {user_id}")

        return archived

    def portfolio_view_to_dict(self, view: PortfolioView) -> Dict[str, Any]:
        """Convert portfolio view to dictionary for API."""
        return {
            "user_id": view.user_id,
            "matches": [
                {
                    "match_id": m.match_id,
                    "match_user_id": m.match_user_id,
                    "status": m.status.value,
                    "tier": m.tier.value,
                    "score": round(m.score, 2),
                    "matched_at": m.matched_at.isoformat(),
                    "last_activity": m.last_activity.isoformat(),
                    "notes": m.notes,
                    "tags": m.tags,
                    "has_conversation": m.conversation_id is not None
                }
                for m in view.matches
            ],
            "pagination": {
                "total_count": view.total_count,
                "page": view.page,
                "page_size": view.page_size,
                "total_pages": (view.total_count + view.page_size - 1) // view.page_size
            },
            "filters_applied": view.filters_applied,
            "sort_applied": view.sort_applied.value
        }

    def stats_to_dict(self, stats: PortfolioStats) -> Dict[str, Any]:
        """Convert stats to dictionary for API."""
        return {
            "total_matches": stats.total_matches,
            "by_status": stats.matches_by_status,
            "by_tier": stats.matches_by_tier,
            "avg_score": round(stats.avg_score, 2),
            "connection_rate": round(stats.connection_rate, 2),
            "active_conversations": stats.active_conversations,
            "archived_count": stats.archived_count,
            "new_this_week": stats.new_this_week,
            "new_this_month": stats.new_this_month
        }


# Global instance
match_portfolio = MatchPortfolio()
