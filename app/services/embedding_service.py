"""
Main embedding service: OpenAI embedding + pgvector storage.
Single, reliable service for all embedding needs.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from cachetools import LRUCache

from typing import Union
from sentence_transformers import SentenceTransformer

from app.adapters.postgresql import postgresql_adapter
from app.utils.cache import cache

load_dotenv()

logger = logging.getLogger(__name__)

# Maximum entries in local embedding cache to prevent memory leaks
# Each 768-dim embedding is ~6KB, so 1000 entries = ~6MB max
LOCAL_CACHE_MAX_SIZE = int(os.getenv('EMBEDDING_LOCAL_CACHE_SIZE', '1000'))

# Lazy import to avoid circular dependency
_versioning_service = None


def _get_versioning_service():
    """Lazy load versioning service to avoid circular imports."""
    global _versioning_service
    if _versioning_service is None:
        from app.services.embedding_versioning import embedding_versioning
        _versioning_service = embedding_versioning
    return _versioning_service

class EmbeddingService:
    """
    Main embedding service for all embedding needs:
    - OpenAI embeddings (via API)
    - pgvector for persistent storage
    - Single, reliable service for all users
    """

    def __init__(self):
        # Read similarity threshold and model config from environment
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '768'))
        self.st_model = SentenceTransformer(self.model_name.replace('sentence-transformers/', ''))
        try:
            dim = getattr(self.st_model, 'get_sentence_embedding_dimension', None)
            if callable(dim):
                self.embedding_dimension = int(dim())
        except Exception:
            pass
        logger.info(f"Using SentenceTransformers model: {self.model_name} with dimension: {self.embedding_dimension}")
        # Use Redis cache with LRU in-memory fallback (bounded to prevent memory leaks)
        self._local_cache: LRUCache = LRUCache(maxsize=LOCAL_CACHE_MAX_SIZE)
        self._cache_hits = 0
        self._cache_misses = 0

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using SentenceTransformers.
        Uses Redis cache for persistence with in-memory fallback.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding generation")
                return None

            cleaned_text = text.strip()

            # Try Redis cache first (persistent across restarts)
            cached_embedding = cache.get_embedding(cleaned_text)
            if cached_embedding is not None:
                self._cache_hits += 1
                logger.debug(f"Redis cache hit for embedding (hits: {self._cache_hits})")
                return cached_embedding

            # Try local cache as fallback
            local_key = f"embed:{hash(cleaned_text)}"
            if local_key in self._local_cache:
                self._cache_hits += 1
                logger.debug(f"Local cache hit for embedding (hits: {self._cache_hits})")
                return self._local_cache[local_key]

            self._cache_misses += 1
            logger.debug(f"Cache miss for embedding (misses: {self._cache_misses})")

            # Generate new embedding
            try:
                embedding_vector = self.st_model.encode(cleaned_text).tolist()
            except Exception as e:
                logger.error(f"Failed to generate SentenceTransformers embedding: {str(e)}")
                return None

            if embedding_vector:
                # Store in Redis cache (persistent)
                cache.set_embedding(cleaned_text, embedding_vector)
                # Also store in local cache (fast access)
                self._local_cache[local_key] = embedding_vector

            return embedding_vector
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None

    def store_user_embeddings(self, user_id: str, requirements: str, offerings: str) -> bool:
        """
        Generate and store embeddings for user's requirements and offerings.

        Args:
            user_id: User identifier
            requirements: User's requirements text
            offerings: User's offerings text

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Generating and storing embeddings for user {user_id}")
            success_count = 0

            # Get version metadata for tracking
            versioning = _get_versioning_service()
            version_metadata = versioning.get_version_metadata() if versioning else {}

            if requirements:
                req_embedding = self.generate_embedding(requirements)
                if req_embedding:
                    metadata = {
                        'text_length': len(requirements),
                        **version_metadata
                    }
                    if postgresql_adapter.store_embedding(
                        user_id=user_id,
                        embedding_type='requirements',
                        vector_data=req_embedding,
                        metadata=metadata
                    ):
                        success_count += 1
                        logger.info(f"Stored requirements embedding for user {user_id}")

            if offerings:
                off_embedding = self.generate_embedding(offerings)
                if off_embedding:
                    metadata = {
                        'text_length': len(offerings),
                        **version_metadata
                    }
                    if postgresql_adapter.store_embedding(
                        user_id=user_id,
                        embedding_type='offerings',
                        vector_data=off_embedding,
                        metadata=metadata
                    ):
                        success_count += 1
                        logger.info(f"Stored offerings embedding for user {user_id}")

            logger.info(f"Successfully stored {success_count} embeddings for user {user_id}")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error storing user embeddings for {user_id}: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics including cache performance and versioning.

        Returns:
            Dictionary of stats about embeddings, storage, cache, and versions.
        """
        try:
            all_embeddings = postgresql_adapter.get_all_user_embeddings()

            req_count = len([e for e in all_embeddings if e['embedding_type'] == 'requirements'])
            off_count = len([e for e in all_embeddings if e['embedding_type'] == 'offerings'])

            # Calculate cache hit rate
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

            # Get versioning stats
            versioning = _get_versioning_service()
            version_stats = versioning.get_version_stats() if versioning else {}

            return {
                'method': 'SentenceTransformers + pgvector',
                'model': self.model_name,
                'dimension': self.embedding_dimension,
                'total_embeddings': len(all_embeddings),
                'requirements_embeddings': req_count,
                'offerings_embeddings': off_count,
                'similarity_threshold': self.similarity_threshold,
                'storage': 'PostgreSQL pgvector',
                'cache': {
                    'redis_stats': cache.get_stats(),
                    'local_cache_size': self._local_cache.currsize,
                    'local_cache_max_size': LOCAL_CACHE_MAX_SIZE,
                    'cache_hits': self._cache_hits,
                    'cache_misses': self._cache_misses,
                    'hit_rate_percent': round(hit_rate, 2)
                },
                'versioning': version_stats,
                'cpu_only': True,
                'reliable': True
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'method': 'SentenceTransformers + pgvector',
                'error': str(e)
            }
        
# Global service instance
embedding_service = EmbeddingService()
