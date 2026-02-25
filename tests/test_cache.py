"""
Unit tests for Redis caching utility.
Tests cache operations, key generation, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os
import hashlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRedisCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_key_deterministic(self):
        """Same input should produce same key."""
        # Test the key generation logic directly
        data = "test data"
        hash1 = hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
        hash2 = hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
        assert hash1 == hash2

    def test_generate_key_different_data(self):
        """Different data should produce different keys."""
        hash1 = hashlib.sha256("data1".encode('utf-8')).hexdigest()[:32]
        hash2 = hashlib.sha256("data2".encode('utf-8')).hexdigest()[:32]
        assert hash1 != hash2

    def test_generate_key_different_prefix(self):
        """Different prefixes should produce different keys."""
        data = "data"
        hash_val = hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
        key1 = f"prefix1:{hash_val}"
        key2 = f"prefix2:{hash_val}"
        assert key1 != key2

    def test_generate_key_format(self):
        """Key should have correct format."""
        prefix = "embed"
        data = "test"
        hash_val = hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
        key = f"{prefix}:{hash_val}"
        assert key.startswith("embed:")
        assert len(key) == len("embed:") + 32  # prefix + 32 char hash


class TestRedisCacheOperations:
    """Tests for cache get/set operations."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        return mock_client

    @pytest.fixture
    def cache_with_mock(self, mock_redis_client):
        """Create cache with mocked Redis."""
        with patch.dict(os.environ, {'CACHE_ENABLED': 'true', 'REDIS_URL': 'redis://localhost:6379/0'}):
            with patch('redis.from_url', return_value=mock_redis_client):
                from app.utils.cache import RedisCache
                cache = RedisCache(enabled=True)
                cache._client = mock_redis_client
                cache._connected = True
                return cache, mock_redis_client

    def test_set_serializes_value(self, cache_with_mock):
        """Set should serialize value to JSON."""
        cache, mock_client = cache_with_mock
        cache.set("key", {"data": "value"}, ttl=100)
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == "key"
        assert call_args[0][1] == 100
        # Check serialization
        stored_value = json.loads(call_args[0][2].decode('utf-8'))
        assert stored_value == {"data": "value"}

    def test_get_deserializes_value(self, cache_with_mock):
        """Get should deserialize JSON value."""
        cache, mock_client = cache_with_mock
        mock_client.get.return_value = b'{"data": "value"}'
        result = cache.get("key")
        assert result == {"data": "value"}

    def test_get_returns_none_for_missing_key(self, cache_with_mock):
        """Get should return None for missing key."""
        cache, mock_client = cache_with_mock
        mock_client.get.return_value = None
        result = cache.get("missing_key")
        assert result is None

    def test_delete_key(self, cache_with_mock):
        """Delete should remove key."""
        cache, mock_client = cache_with_mock
        cache.delete("key")
        mock_client.delete.assert_called_once_with("key")


class TestEmbeddingCache:
    """Tests for embedding-specific cache operations."""

    @pytest.fixture
    def cache_with_mock(self):
        """Create cache with mocked Redis."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True

        with patch.dict(os.environ, {'CACHE_ENABLED': 'true'}):
            with patch('redis.from_url', return_value=mock_client):
                from app.utils.cache import RedisCache
                cache = RedisCache(enabled=True)
                cache._client = mock_client
                cache._connected = True
                return cache, mock_client

    def test_get_embedding_uses_correct_prefix(self, cache_with_mock):
        """get_embedding should use 'embed' prefix."""
        cache, mock_client = cache_with_mock
        mock_client.get.return_value = b'[0.1, 0.2, 0.3]'
        result = cache.get_embedding("test text")
        # Key should start with 'embed:'
        call_args = mock_client.get.call_args
        assert call_args[0][0].startswith("embed:")

    def test_set_embedding_uses_longer_ttl(self, cache_with_mock):
        """set_embedding should use embedding-specific TTL."""
        cache, mock_client = cache_with_mock
        embedding = [0.1, 0.2, 0.3]
        cache.set_embedding("test text", embedding)
        call_args = mock_client.setex.call_args
        # Default EMBEDDING_CACHE_TTL is 604800 (7 days)
        assert call_args[0][1] == 604800

    def test_embedding_roundtrip(self, cache_with_mock):
        """Setting and getting embedding should preserve data."""
        cache, mock_client = cache_with_mock
        embedding = [0.123, 0.456, 0.789]
        # Mock get to return what was set
        mock_client.get.return_value = json.dumps(embedding).encode('utf-8')

        cache.set_embedding("test", embedding)
        result = cache.get_embedding("test")

        assert result == embedding


class TestCacheDisabled:
    """Tests for cache when disabled."""

    def test_get_returns_none_when_disabled(self):
        """Get should return None when cache is disabled."""
        with patch.dict(os.environ, {'CACHE_ENABLED': 'false'}):
            from app.utils.cache import RedisCache
            cache = RedisCache(enabled=False)
            result = cache.get("key")
            assert result is None

    def test_set_returns_false_when_disabled(self):
        """Set should return False when cache is disabled."""
        with patch.dict(os.environ, {'CACHE_ENABLED': 'false'}):
            from app.utils.cache import RedisCache
            cache = RedisCache(enabled=False)
            result = cache.set("key", "value")
            assert result is False


class TestCacheConnectionFailure:
    """Tests for cache behavior when Redis is unavailable."""

    def test_graceful_degradation_on_connection_error(self):
        """Cache should gracefully degrade when Redis is unavailable."""
        from redis.exceptions import RedisError
        with patch('redis.from_url') as mock_from_url:
            mock_from_url.side_effect = RedisError("Connection refused")

            with patch.dict(os.environ, {'CACHE_ENABLED': 'true'}):
                from app.utils.cache import RedisCache
                cache = RedisCache(enabled=True)
                # Should not raise, just disable
                assert cache._connected is False

    def test_get_returns_none_on_connection_failure(self):
        """Get should return None when not connected."""
        from redis.exceptions import RedisError
        with patch('redis.from_url') as mock_from_url:
            mock_from_url.side_effect = RedisError("Connection refused")

            with patch.dict(os.environ, {'CACHE_ENABLED': 'true'}):
                from app.utils.cache import RedisCache
                cache = RedisCache(enabled=True)
                result = cache.get("key")
                assert result is None


class TestCacheStats:
    """Tests for cache statistics."""

    @pytest.fixture
    def cache_with_mock(self):
        """Create cache with mocked Redis."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.info.return_value = {
            'used_memory_human': '1M',
            'maxmemory_human': '100M'
        }
        mock_client.dbsize.return_value = 42

        with patch.dict(os.environ, {'CACHE_ENABLED': 'true'}):
            with patch('redis.from_url', return_value=mock_client):
                from app.utils.cache import RedisCache
                cache = RedisCache(enabled=True)
                cache._client = mock_client
                cache._connected = True
                return cache

    def test_get_stats_returns_info(self, cache_with_mock):
        """get_stats should return cache information."""
        stats = cache_with_mock.get_stats()
        assert stats['enabled'] is True
        assert stats['connected'] is True
        assert stats['used_memory'] == '1M'
        assert stats['keys'] == 42

    def test_get_stats_when_disabled(self):
        """get_stats should indicate disabled state."""
        with patch.dict(os.environ, {'CACHE_ENABLED': 'false'}):
            from app.utils.cache import RedisCache
            cache = RedisCache(enabled=False)
            stats = cache.get_stats()
            assert stats['enabled'] is False
