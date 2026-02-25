"""
Redis caching utility for the Reciprocity AI platform.
Provides persistent caching with automatic serialization for embeddings and other data.
"""
import os
import json
import hashlib
import logging
from typing import Any, Optional, List, Union
from functools import wraps
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '86400'))  # 24 hours default
EMBEDDING_CACHE_TTL = int(os.getenv('EMBEDDING_CACHE_TTL', '604800'))  # 7 days for embeddings


class RedisCache:
    """
    Redis cache client with connection pooling and automatic fallback.
    Handles serialization of complex types like embedding vectors.
    """

    def __init__(self, url: str = REDIS_URL, enabled: bool = CACHE_ENABLED):
        self.enabled = enabled
        self._client: Optional[redis.Redis] = None
        self._url = url
        self._connected = False

        if self.enabled:
            self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection with proper error handling."""
        if self._connected and self._client:
            return True

        try:
            # Support rediss:// URLs (Upstash, etc.) which require ssl_cert_reqs
            redis_kwargs = {
                "decode_responses": False,  # We handle encoding ourselves
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
                "retry_on_timeout": True
            }
            if self._url.startswith("rediss://"):
                redis_kwargs["ssl_cert_reqs"] = "none"

            self._client = redis.from_url(self._url, **redis_kwargs)
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info("Redis cache connected successfully")
            return True
        except RedisError as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self._connected = False
            return False

    @staticmethod
    def _generate_key(prefix: str, data: str) -> str:
        """Generate a cache key using SHA256 hash."""
        hash_value = hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
        return f"{prefix}:{hash_value}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        Returns None if key not found or cache is disabled/unavailable.
        """
        if not self.enabled or not self._connected:
            return None

        try:
            value = self._client.get(key)
            if value is None:
                return None

            # Deserialize
            return json.loads(value.decode('utf-8'))
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> bool:
        """
        Set a value in cache with TTL.
        Returns True if successful, False otherwise.
        """
        if not self.enabled or not self._connected:
            return False

        try:
            # Serialize value
            serialized = json.dumps(value).encode('utf-8')
            self._client.setex(key, ttl, serialized)
            return True
        except (RedisError, TypeError) as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.enabled or not self._connected:
            return False

        try:
            self._client.delete(key)
            return True
        except RedisError as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        Embeddings are stored with a longer TTL.
        """
        key = self._generate_key("embed", text)
        return self.get(key)

    def set_embedding(self, text: str, embedding: List[float], ttl: int = EMBEDDING_CACHE_TTL) -> bool:
        """
        Cache an embedding for text.
        Uses embedding-specific TTL (default 7 days).
        """
        key = self._generate_key("embed", text)
        return self.set(key, embedding, ttl)

    def get_persona(self, user_id: str) -> Optional[dict]:
        """Get cached persona for user."""
        key = f"persona:{user_id}"
        return self.get(key)

    def set_persona(self, user_id: str, persona: dict, ttl: int = CACHE_TTL_SECONDS) -> bool:
        """Cache persona for user."""
        key = f"persona:{user_id}"
        return self.set(key, persona, ttl)

    def invalidate_user(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a user.
        Returns number of keys deleted.
        """
        if not self.enabled or not self._connected:
            return 0

        try:
            pattern = f"*:{user_id}*"
            keys = list(self._client.scan_iter(match=pattern, count=100))
            if keys:
                return self._client.delete(*keys)
            return 0
        except RedisError as e:
            logger.warning(f"Cache invalidation error for user {user_id}: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        if not self._connected:
            return {"enabled": True, "connected": False}

        try:
            info = self._client.info('memory')
            return {
                "enabled": True,
                "connected": True,
                "used_memory": info.get('used_memory_human', 'unknown'),
                "max_memory": info.get('maxmemory_human', 'unlimited'),
                "keys": self._client.dbsize()
            }
        except RedisError as e:
            return {"enabled": True, "connected": False, "error": str(e)}


# Global cache instance
cache = RedisCache()


def cached(prefix: str, ttl: int = CACHE_TTL_SECONDS):
    """
    Decorator for caching function results.
    Uses first argument as cache key basis.

    Example:
        @cached("question_mod", ttl=3600)
        def modify_question(text: str) -> str:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache.enabled or not cache._connected:
                return func(*args, **kwargs)

            # Use first arg as key basis
            key_data = str(args[0]) if args else str(kwargs)
            key = cache._generate_key(prefix, key_data)

            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl)

            return result
        return wrapper
    return decorator
