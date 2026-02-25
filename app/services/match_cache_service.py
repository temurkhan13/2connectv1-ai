"""
Pre-computed Match Index (Match Cache Service).

Implements caching layer for match results to improve performance.
Instead of running pgvector similarity searches for every request,
we pre-compute matches and cache them.

Strategy:
1. On user creation/update: Compute matches and cache them
2. On match request: Return cached results (fast)
3. Background job: Periodically refresh cache for active users
4. Cache invalidation: When user updates their profile

Author: Claude Code
Date: February 2026
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""
    ttl_seconds: int = 3600           # 1 hour default TTL
    max_matches_per_user: int = 50    # Max matches to cache per user
    refresh_threshold_seconds: int = 1800  # Refresh if older than 30 min
    enabled: bool = True

    @classmethod
    def from_env(cls) -> 'CacheConfig':
        return cls(
            ttl_seconds=int(os.getenv("MATCH_CACHE_TTL", "3600")),
            max_matches_per_user=int(os.getenv("MATCH_CACHE_MAX_MATCHES", "50")),
            refresh_threshold_seconds=int(os.getenv("MATCH_CACHE_REFRESH_THRESHOLD", "1800")),
            enabled=os.getenv("MATCH_CACHE_ENABLED", "true").lower() == "true"
        )


class MatchCacheService:
    """
    Service for caching pre-computed match results.

    Uses Redis if available, falls back to in-memory cache.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig.from_env()
        self._memory_cache: Dict[str, Dict] = {}
        self._redis_client = None

        # Try to connect to Redis
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection if available."""
        try:
            import redis
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()
                logger.info("Connected to Redis for match caching")
            else:
                logger.info("REDIS_URL not set, using in-memory cache")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self._redis_client = None

    def _cache_key(self, user_id: str) -> str:
        """Generate cache key for user matches."""
        return f"matches:v2:{user_id}"

    def get_cached_matches(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached matches for a user.

        Returns:
            Cached match data or None if not found/expired
        """
        if not self.config.enabled:
            return None

        cache_key = self._cache_key(user_id)

        try:
            if self._redis_client:
                data = self._redis_client.get(cache_key)
                if data:
                    cached = json.loads(data)
                    logger.debug(f"Cache hit for user {user_id}")
                    return cached
            else:
                # In-memory fallback
                if cache_key in self._memory_cache:
                    cached = self._memory_cache[cache_key]
                    # Check TTL
                    cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
                    if datetime.utcnow() - cached_at < timedelta(seconds=self.config.ttl_seconds):
                        logger.debug(f"Memory cache hit for user {user_id}")
                        return cached
                    else:
                        # Expired
                        del self._memory_cache[cache_key]

        except Exception as e:
            logger.error(f"Error getting cached matches for {user_id}: {e}")

        return None

    def cache_matches(
        self,
        user_id: str,
        matches: List[Dict[str, Any]],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Cache match results for a user.

        Args:
            user_id: User identifier
            matches: List of match results
            metadata: Optional metadata

        Returns:
            True if cached successfully
        """
        if not self.config.enabled:
            return False

        cache_key = self._cache_key(user_id)

        try:
            cache_data = {
                "user_id": user_id,
                "matches": matches[:self.config.max_matches_per_user],
                "total_matches": len(matches),
                "cached_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            if self._redis_client:
                self._redis_client.setex(
                    cache_key,
                    self.config.ttl_seconds,
                    json.dumps(cache_data)
                )
                logger.debug(f"Cached {len(matches)} matches for user {user_id} in Redis")
            else:
                # In-memory fallback
                self._memory_cache[cache_key] = cache_data
                logger.debug(f"Cached {len(matches)} matches for user {user_id} in memory")

            return True

        except Exception as e:
            logger.error(f"Error caching matches for {user_id}: {e}")
            return False

    def invalidate_cache(self, user_id: str) -> bool:
        """
        Invalidate cached matches for a user.

        Should be called when:
        - User updates their profile
        - User provides feedback
        - Admin manually invalidates
        """
        cache_key = self._cache_key(user_id)

        try:
            if self._redis_client:
                self._redis_client.delete(cache_key)
            elif cache_key in self._memory_cache:
                del self._memory_cache[cache_key]

            logger.info(f"Invalidated match cache for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error invalidating cache for {user_id}: {e}")
            return False

    def invalidate_all(self) -> int:
        """
        Invalidate all cached matches.

        Returns:
            Number of entries invalidated
        """
        count = 0
        try:
            if self._redis_client:
                keys = self._redis_client.keys("matches:v2:*")
                if keys:
                    count = self._redis_client.delete(*keys)
            else:
                count = len(self._memory_cache)
                self._memory_cache.clear()

            logger.info(f"Invalidated {count} match cache entries")
            return count

        except Exception as e:
            logger.error(f"Error invalidating all cache: {e}")
            return 0

    def needs_refresh(self, user_id: str) -> bool:
        """
        Check if user's cache needs refresh.

        Returns True if:
        - Cache doesn't exist
        - Cache is older than refresh threshold
        """
        cached = self.get_cached_matches(user_id)
        if not cached:
            return True

        try:
            cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            age = (datetime.utcnow() - cached_at).total_seconds()
            return age > self.config.refresh_threshold_seconds
        except Exception:
            return True

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self.config.enabled,
            "backend": "redis" if self._redis_client else "memory",
            "ttl_seconds": self.config.ttl_seconds
        }

        try:
            if self._redis_client:
                keys = self._redis_client.keys("matches:v2:*")
                stats["cached_users"] = len(keys)
                info = self._redis_client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            else:
                stats["cached_users"] = len(self._memory_cache)
                # Estimate memory usage
                memory_estimate = sum(
                    len(json.dumps(v)) for v in self._memory_cache.values()
                )
                stats["memory_estimate_bytes"] = memory_estimate

        except Exception as e:
            stats["error"] = str(e)

        return stats


# =============================================================================
# CACHED MATCHING SERVICE WRAPPER
# =============================================================================

class CachedMatchingService:
    """
    Wrapper around matching services that adds caching.

    Usage:
    1. First checks cache for matches
    2. If cache miss, computes matches using enhanced_matching_service
    3. Caches the results for future requests
    """

    def __init__(self):
        self.cache = MatchCacheService()
        self._matching_service = None

    @property
    def matching_service(self):
        """Lazy load matching service to avoid circular imports."""
        if self._matching_service is None:
            from app.services.enhanced_matching_service import enhanced_matching_service
            self._matching_service = enhanced_matching_service
        return self._matching_service

    def find_matches(
        self,
        user_id: str,
        force_refresh: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find matches with caching.

        Args:
            user_id: User identifier
            force_refresh: Force recomputation even if cached
            **kwargs: Additional arguments for matching service

        Returns:
            Match results (from cache or freshly computed)
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self.cache.get_cached_matches(user_id)
            if cached:
                return {
                    "success": True,
                    "from_cache": True,
                    "cached_at": cached.get("cached_at"),
                    "matches": cached.get("matches", []),
                    "total_matches": cached.get("total_matches", 0)
                }

        # Cache miss or force refresh - compute matches
        try:
            matches = self.matching_service.find_bidirectional_matches(
                user_id=user_id,
                include_explanations=True,
                **kwargs
            )

            # Format for caching
            formatted_matches = self.matching_service.format_matches_for_api(matches)

            # Cache the results
            self.cache.cache_matches(
                user_id=user_id,
                matches=formatted_matches,
                metadata={"computed_at": datetime.utcnow().isoformat()}
            )

            return {
                "success": True,
                "from_cache": False,
                "matches": formatted_matches,
                "total_matches": len(formatted_matches)
            }

        except Exception as e:
            logger.error(f"Error computing matches for {user_id}: {e}")
            return {
                "success": False,
                "from_cache": False,
                "error": str(e),
                "matches": [],
                "total_matches": 0
            }

    def refresh_user_cache(self, user_id: str) -> bool:
        """Force refresh cache for a user."""
        self.cache.invalidate_cache(user_id)
        result = self.find_matches(user_id, force_refresh=True)
        return result.get("success", False)

    def invalidate_on_profile_update(self, user_id: str) -> None:
        """
        Called when user profile is updated.
        Invalidates their cache and also caches of users who matched with them.
        """
        # Invalidate user's own cache
        self.cache.invalidate_cache(user_id)

        # Note: For a complete solution, we'd also invalidate caches
        # of users who had this user in their matches. This would require
        # maintaining a reverse index. For simplicity, we rely on TTL
        # to eventually refresh those caches.
        logger.info(f"Invalidated cache for user {user_id} due to profile update")


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

match_cache_service = MatchCacheService()
cached_matching_service = CachedMatchingService()
