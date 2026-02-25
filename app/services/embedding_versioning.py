"""
Embedding Versioning Service.

Tracks model versions for embeddings to ensure consistency and enable
automatic regeneration when models change.

Key features:
- Version tracking per embedding
- Stale embedding detection
- Batch regeneration support
- Version history for auditing
"""
import os
import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from app.adapters.postgresql import postgresql_adapter

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingVersion:
    """Represents an embedding model version."""
    model_name: str
    model_version: str
    dimension: int
    version_hash: str
    created_at: datetime

    @classmethod
    def from_env(cls) -> 'EmbeddingVersion':
        """Create version from current environment settings."""
        model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        model_version = os.getenv('EMBEDDING_MODEL_VERSION', '1.0.0')
        dimension = int(os.getenv('EMBEDDING_DIMENSION', '768'))

        # Create deterministic hash of model configuration
        config_str = f"{model_name}|{model_version}|{dimension}"
        version_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return cls(
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            version_hash=version_hash,
            created_at=datetime.utcnow()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "version_hash": self.version_hash,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingVersion':
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "unknown"),
            model_version=data.get("model_version", "unknown"),
            dimension=data.get("dimension", 768),
            version_hash=data.get("version_hash", ""),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        )


class EmbeddingVersioningService:
    """
    Service for managing embedding versions.

    Ensures embeddings are regenerated when the model changes,
    tracks version history, and identifies stale embeddings.
    """

    def __init__(self):
        self.current_version = EmbeddingVersion.from_env()
        logger.info(
            f"Embedding versioning initialized: "
            f"model={self.current_version.model_name}, "
            f"version={self.current_version.model_version}, "
            f"hash={self.current_version.version_hash}"
        )

    def get_current_version(self) -> EmbeddingVersion:
        """Get the current embedding model version."""
        return self.current_version

    def get_version_metadata(self) -> Dict[str, Any]:
        """Get version metadata for embedding storage."""
        return {
            "embedding_version": self.current_version.to_dict()
        }

    def is_embedding_current(self, embedding_metadata: Dict[str, Any]) -> bool:
        """
        Check if an embedding was created with the current model version.

        Args:
            embedding_metadata: Metadata from stored embedding

        Returns:
            True if embedding is current, False if stale
        """
        if not embedding_metadata:
            return False

        version_info = embedding_metadata.get("embedding_version", {})
        stored_hash = version_info.get("version_hash", "")

        return stored_hash == self.current_version.version_hash

    def get_stale_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find embeddings that need regeneration due to model changes.

        Args:
            limit: Maximum number of stale embeddings to return

        Returns:
            List of embedding records that are stale
        """
        try:
            # Get all embeddings
            all_embeddings = postgresql_adapter.get_all_user_embeddings()

            stale = []
            for emb in all_embeddings:
                metadata = emb.get("metadata", {})
                if not self.is_embedding_current(metadata):
                    stale.append({
                        "user_id": emb["user_id"],
                        "embedding_type": emb["embedding_type"],
                        "created_at": emb.get("created_at"),
                        "stored_version": metadata.get("embedding_version", {}).get("version_hash", "none"),
                        "current_version": self.current_version.version_hash
                    })

                    if len(stale) >= limit:
                        break

            logger.info(f"Found {len(stale)} stale embeddings (limit={limit})")
            return stale

        except Exception as e:
            logger.error(f"Error finding stale embeddings: {e}")
            return []

    def get_version_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embedding versions in the database.

        Returns:
            Dict with version distribution and stats
        """
        try:
            all_embeddings = postgresql_adapter.get_all_user_embeddings()

            version_counts: Dict[str, int] = {}
            total = 0
            current = 0
            stale = 0

            for emb in all_embeddings:
                total += 1
                metadata = emb.get("metadata", {})
                version_info = metadata.get("embedding_version", {})
                version_hash = version_info.get("version_hash", "unknown")

                version_counts[version_hash] = version_counts.get(version_hash, 0) + 1

                if version_hash == self.current_version.version_hash:
                    current += 1
                else:
                    stale += 1

            return {
                "current_version": self.current_version.to_dict(),
                "total_embeddings": total,
                "current_embeddings": current,
                "stale_embeddings": stale,
                "stale_percentage": (stale / total * 100) if total > 0 else 0,
                "version_distribution": version_counts
            }

        except Exception as e:
            logger.error(f"Error getting version stats: {e}")
            return {
                "error": str(e),
                "current_version": self.current_version.to_dict()
            }

    def mark_for_regeneration(self, user_id: str, embedding_type: str) -> bool:
        """
        Mark an embedding for regeneration.

        Args:
            user_id: User identifier
            embedding_type: Type of embedding to regenerate

        Returns:
            True if marked successfully
        """
        try:
            # Get current embedding
            user_embeddings = postgresql_adapter.get_user_embeddings(user_id)
            emb = user_embeddings.get(embedding_type)

            if not emb:
                logger.warning(f"No {embedding_type} embedding found for user {user_id}")
                return False

            # Update metadata to mark for regeneration
            metadata = emb.get("metadata", {})
            metadata["needs_regeneration"] = True
            metadata["regeneration_requested_at"] = datetime.utcnow().isoformat()
            metadata["regeneration_reason"] = "model_version_change"

            # Store with updated metadata
            return postgresql_adapter.store_embedding(
                user_id=user_id,
                embedding_type=embedding_type,
                vector_data=emb["vector_data"],
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error marking embedding for regeneration: {e}")
            return False

    def needs_regeneration(self, embedding_metadata: Dict[str, Any]) -> bool:
        """
        Check if an embedding needs regeneration.

        Args:
            embedding_metadata: Metadata from stored embedding

        Returns:
            True if regeneration is needed
        """
        if not embedding_metadata:
            return True

        # Check explicit regeneration flag
        if embedding_metadata.get("needs_regeneration", False):
            return True

        # Check version
        return not self.is_embedding_current(embedding_metadata)


# Global instance
embedding_versioning = EmbeddingVersioningService()


def get_versioned_metadata() -> Dict[str, Any]:
    """
    Get metadata dict with current version info.

    Use this when storing embeddings to include version tracking.
    """
    return embedding_versioning.get_version_metadata()
