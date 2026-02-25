"""
Multi-Vector Embedding Service.

Implements multi-dimensional embeddings for more precise matching.
Instead of a single embedding for requirements/offerings, we create
separate embeddings for different aspects:

1. Skills/Expertise - Technical capabilities
2. Industry/Domain - Sector focus
3. Stage/Phase - Company stage preference
4. Culture/Style - Work style and values

This allows for more nuanced matching where different dimensions
can have different weights based on user intent.

Author: Claude Code
Date: February 2026
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.adapters.postgresql import postgresql_adapter
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING DIMENSIONS
# =============================================================================

class EmbeddingDimension:
    """Embedding dimension types."""
    SKILLS = "skills"           # Technical skills and expertise
    INDUSTRY = "industry"       # Industry/domain focus
    STAGE = "stage"             # Company stage (seed, series A, etc.)
    CULTURE = "culture"         # Work culture and style
    REQUIREMENTS = "requirements"  # Traditional requirements
    OFFERINGS = "offerings"     # Traditional offerings


@dataclass
class DimensionConfig:
    """Configuration for a dimension."""
    name: str
    weight: float              # Importance weight (0-2)
    keywords: List[str]        # Keywords to identify this dimension
    extraction_prompt: str     # Prompt template for extraction


# Default dimension configurations
DIMENSION_CONFIGS = {
    EmbeddingDimension.SKILLS: DimensionConfig(
        name="Skills & Expertise",
        weight=1.0,
        keywords=[
            "skills", "expertise", "proficient", "experience with", "know how",
            "python", "javascript", "machine learning", "data science", "cloud",
            "aws", "gcp", "kubernetes", "react", "backend", "frontend", "devops"
        ],
        extraction_prompt="Extract the technical skills and expertise mentioned."
    ),
    EmbeddingDimension.INDUSTRY: DimensionConfig(
        name="Industry Focus",
        weight=1.2,  # Slightly higher weight
        keywords=[
            "industry", "sector", "market", "fintech", "healthtech", "edtech",
            "saas", "b2b", "b2c", "e-commerce", "healthcare", "finance",
            "real estate", "logistics", "media", "entertainment"
        ],
        extraction_prompt="Extract the industry or sector focus."
    ),
    EmbeddingDimension.STAGE: DimensionConfig(
        name="Stage Preference",
        weight=1.5,  # High weight - stage mismatch is critical
        keywords=[
            "stage", "seed", "pre-seed", "series a", "series b", "growth",
            "early-stage", "startup", "scale-up", "enterprise", "funding round"
        ],
        extraction_prompt="Extract the company stage preference."
    ),
    EmbeddingDimension.CULTURE: DimensionConfig(
        name="Culture & Style",
        weight=0.8,  # Lower weight
        keywords=[
            "culture", "remote", "hybrid", "office", "flexible", "fast-paced",
            "collaborative", "autonomous", "work-life", "startup culture",
            "corporate", "hands-on", "hands-off", "mentorship"
        ],
        extraction_prompt="Extract work culture and style preferences."
    )
}


# =============================================================================
# MULTI-VECTOR EMBEDDING SERVICE
# =============================================================================

class MultiVectorEmbeddingService:
    """
    Service for creating and managing multi-dimensional embeddings.

    This creates separate embeddings for different aspects of a user's
    profile, allowing for more nuanced matching.
    """

    def __init__(self):
        self.dimension_configs = DIMENSION_CONFIGS
        self.base_embedding_service = embedding_service

    def extract_dimension_text(
        self,
        full_text: str,
        dimension: str
    ) -> str:
        """
        Extract text relevant to a specific dimension from full text.

        Uses keyword matching to identify relevant sentences.
        Could be enhanced with LLM-based extraction.
        """
        config = self.dimension_configs.get(dimension)
        if not config:
            return full_text

        # Split into sentences
        sentences = full_text.replace('\n', '. ').split('. ')

        # Find sentences containing dimension keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in config.keywords):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            return ". ".join(relevant_sentences)

        # If no matches, return original text (fallback)
        return full_text

    def generate_multi_vector_embeddings(
        self,
        user_id: str,
        requirements_text: str,
        offerings_text: str,
        store_in_db: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate multiple embeddings for different dimensions.

        Args:
            user_id: User identifier
            requirements_text: User's requirements text
            offerings_text: User's offerings text
            store_in_db: Whether to store in database

        Returns:
            Dict with dimension embeddings
        """
        results = {
            "user_id": user_id,
            "dimensions": {},
            "generated_at": datetime.utcnow().isoformat()
        }

        # Generate traditional embeddings first
        req_embedding = self.base_embedding_service.generate_embedding(requirements_text)
        off_embedding = self.base_embedding_service.generate_embedding(offerings_text)

        if req_embedding:
            results["dimensions"]["requirements"] = {
                "vector": req_embedding,
                "dimension": len(req_embedding),
                "source": "full_text"
            }
            if store_in_db:
                postgresql_adapter.store_embedding(
                    user_id=user_id,
                    embedding_type="requirements",
                    vector_data=req_embedding,
                    metadata={"dimension": "requirements", "multi_vector": True}
                )

        if off_embedding:
            results["dimensions"]["offerings"] = {
                "vector": off_embedding,
                "dimension": len(off_embedding),
                "source": "full_text"
            }
            if store_in_db:
                postgresql_adapter.store_embedding(
                    user_id=user_id,
                    embedding_type="offerings",
                    vector_data=off_embedding,
                    metadata={"dimension": "offerings", "multi_vector": True}
                )

        # Generate dimension-specific embeddings
        combined_text = f"{requirements_text}\n{offerings_text}"

        for dim_name, config in self.dimension_configs.items():
            try:
                # Extract dimension-specific text
                dim_text = self.extract_dimension_text(combined_text, dim_name)

                if not dim_text.strip():
                    continue

                # Generate embedding for this dimension
                dim_embedding = self.base_embedding_service.generate_embedding(dim_text)

                if dim_embedding:
                    emb_type = f"{dim_name}_combined"
                    results["dimensions"][emb_type] = {
                        "vector": dim_embedding,
                        "dimension": len(dim_embedding),
                        "weight": config.weight,
                        "source": "extracted"
                    }

                    if store_in_db:
                        postgresql_adapter.store_embedding(
                            user_id=user_id,
                            embedding_type=emb_type,
                            vector_data=dim_embedding,
                            metadata={
                                "dimension": dim_name,
                                "weight": config.weight,
                                "multi_vector": True
                            }
                        )

            except Exception as e:
                logger.error(f"Error generating {dim_name} embedding for {user_id}: {e}")

        logger.info(
            f"Generated {len(results['dimensions'])} dimension embeddings "
            f"for user {user_id}"
        )

        return results

    def calculate_weighted_similarity(
        self,
        user_embeddings: Dict[str, Any],
        match_embeddings: Dict[str, Any],
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted similarity across multiple dimensions.

        Args:
            user_embeddings: User's multi-dimensional embeddings
            match_embeddings: Match's multi-dimensional embeddings
            dimension_weights: Optional custom weights per dimension

        Returns:
            Tuple of (weighted_score, per_dimension_scores)
        """
        from numpy import dot
        from numpy.linalg import norm

        def cosine_similarity(v1, v2) -> float:
            """Calculate cosine similarity between two vectors."""
            try:
                v1_arr = [float(x) for x in v1]
                v2_arr = [float(x) for x in v2]
                dot_product = sum(a * b for a, b in zip(v1_arr, v2_arr))
                norm1 = sum(a * a for a in v1_arr) ** 0.5
                norm2 = sum(b * b for b in v2_arr) ** 0.5
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot_product / (norm1 * norm2)
            except Exception:
                return 0.0

        dimension_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Get common dimensions
        user_dims = set(user_embeddings.keys())
        match_dims = set(match_embeddings.keys())
        common_dims = user_dims & match_dims

        for dim in common_dims:
            user_vec = user_embeddings[dim].get('vector_data') or user_embeddings[dim].get('vector')
            match_vec = match_embeddings[dim].get('vector_data') or match_embeddings[dim].get('vector')

            if not user_vec or not match_vec:
                continue

            # Calculate similarity
            similarity = cosine_similarity(user_vec, match_vec)
            dimension_scores[dim] = similarity

            # Get weight
            if dimension_weights and dim in dimension_weights:
                weight = dimension_weights[dim]
            else:
                # Use default config weight
                base_dim = dim.replace('_combined', '').replace('_requirements', '').replace('_offerings', '')
                config = self.dimension_configs.get(base_dim)
                weight = config.weight if config else 1.0

            total_weight += weight
            weighted_sum += similarity * weight

        if total_weight == 0:
            return 0.0, dimension_scores

        weighted_score = weighted_sum / total_weight

        return weighted_score, dimension_scores


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

multi_vector_service = MultiVectorEmbeddingService()
