"""
Multi-Vector Matching Service.

Implements weighted multi-dimensional matching using 6 separate embedding categories:
- primary_goal: User's main objective (investor, founder, advisor, etc.)
- industry: Industry/sector focus
- stage: Investment/company stage
- geography: Geographic preferences
- engagement_style: Communication and engagement preferences
- dealbreakers: Hard exclusion criteria

Each dimension is matched separately and combined with configurable weights.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
from enum import Enum

from app.services.embedding_service import embedding_service
from app.adapters.postgresql import postgresql_adapter

logger = logging.getLogger(__name__)


class MatchTier(str, Enum):
    """Match quality tiers based on weighted score."""
    PERFECT = "perfect"              # 85%+ weighted score
    STRONG = "strong"                # 70-84% weighted score
    WORTH_EXPLORING = "worth_exploring"  # 55-69% weighted score
    LOW = "low"                      # Below 55%


@dataclass
class DimensionWeight:
    """Configurable weight for a matching dimension."""
    dimension: str
    weight: float
    required: bool = False  # If true, 0 similarity = no match


@dataclass
class MultiVectorConfig:
    """Configuration for multi-vector matching."""
    dimensions: List[DimensionWeight] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> 'MultiVectorConfig':
        """Load configuration from environment variables."""
        return cls(
            dimensions=[
                DimensionWeight("primary_goal", float(os.getenv("MATCH_WEIGHT_PRIMARY_GOAL", "0.20")), required=True),
                DimensionWeight("industry", float(os.getenv("MATCH_WEIGHT_INDUSTRY", "0.25")), required=False),
                DimensionWeight("stage", float(os.getenv("MATCH_WEIGHT_STAGE", "0.20")), required=False),
                DimensionWeight("geography", float(os.getenv("MATCH_WEIGHT_GEOGRAPHY", "0.15")), required=False),
                DimensionWeight("engagement_style", float(os.getenv("MATCH_WEIGHT_ENGAGEMENT", "0.10")), required=False),
                DimensionWeight("dealbreakers", float(os.getenv("MATCH_WEIGHT_DEALBREAKERS", "0.10")), required=False),
            ]
        )

    def get_total_weight(self) -> float:
        """Get sum of all weights (should be 1.0)."""
        return sum(d.weight for d in self.dimensions)

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.get_total_weight()
        if total > 0:
            for dim in self.dimensions:
                dim.weight = dim.weight / total


class DimensionScore(BaseModel):
    """Score for a single matching dimension."""
    dimension: str
    similarity: float
    weight: float
    weighted_score: float
    matched: bool


class MultiVectorMatchResult(BaseModel):
    """Result from multi-vector matching."""
    user_id: str
    total_score: float
    tier: MatchTier
    dimension_scores: List[DimensionScore]
    explanation: Optional[str] = None

    @property
    def is_match(self) -> bool:
        """Check if this is a valid match."""
        return self.tier != MatchTier.LOW


class MultiVectorMatcher:
    """
    Multi-dimensional matching using separate embeddings per dimension.

    Instead of compressing all user data into a single vector, this matcher:
    1. Maintains separate embeddings for each dimension (industry, stage, etc.)
    2. Computes similarity for each dimension independently
    3. Combines scores using configurable weights
    4. Returns tiered match quality (Perfect, Strong, Potential, Weak)
    """

    def __init__(self, config: Optional[MultiVectorConfig] = None):
        self.config = config or MultiVectorConfig.from_env()
        self.config.normalize_weights()
        self.embedding_service = embedding_service

    def extract_dimension_text(self, persona_data: Dict[str, Any], dimension: str) -> Optional[str]:
        """
        Extract text for a specific dimension from persona data.

        Args:
            persona_data: User's persona data
            dimension: The dimension to extract (e.g., 'industry', 'stage')

        Returns:
            Extracted text for the dimension, or None if not available
        """
        # Map dimensions to persona fields
        dimension_mappings = {
            "primary_goal": ["primary_goal", "objective", "looking_for"],
            "industry": ["industry", "sector", "focus_area", "industry_focus"],
            "stage": ["stage", "investment_stage", "company_stage", "funding_stage"],
            "geography": ["geography", "location", "region", "geographic_focus"],
            "engagement_style": ["engagement_style", "communication_preference", "collaboration_style"],
            "dealbreakers": ["dealbreakers", "exclusions", "not_interested_in", "avoid"],
        }

        # Try to find matching fields
        field_names = dimension_mappings.get(dimension, [dimension])
        for field_name in field_names:
            if field_name in persona_data and persona_data[field_name]:
                value = persona_data[field_name]
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value)
                return str(value)

        # Fallback: try to extract from requirements/offerings text
        for key in ["requirements", "offerings"]:
            if key in persona_data and persona_data[key]:
                # Could use NLP to extract specific dimension
                # For now, return the full text as fallback
                return None

        return None

    def store_multi_vector_embeddings(
        self,
        user_id: str,
        persona_data: Dict[str, Any],
        direction: str = "requirements"
    ) -> Dict[str, bool]:
        """
        Store separate embeddings for each dimension.

        Args:
            user_id: User identifier
            persona_data: User's persona data
            direction: 'requirements' (what user needs) or 'offerings' (what user provides)

        Returns:
            Dictionary of {dimension: success} for each stored embedding
        """
        results = {}

        for dim_config in self.config.dimensions:
            dimension = dim_config.dimension
            text = self.extract_dimension_text(persona_data, dimension)

            if text:
                # Generate embedding for this dimension
                embedding = self.embedding_service.generate_embedding(text)

                if embedding:
                    # Store with dimension-specific type
                    embedding_type = f"{direction}_{dimension}"
                    success = postgresql_adapter.store_embedding(
                        user_id=user_id,
                        embedding_type=embedding_type,
                        vector_data=embedding,
                        metadata={
                            "dimension": dimension,
                            "direction": direction,
                            "text_length": len(text)
                        }
                    )
                    results[dimension] = success
                else:
                    results[dimension] = False
                    logger.warning(f"Failed to generate embedding for {user_id}/{dimension}")
            else:
                results[dimension] = False
                logger.debug(f"No data for dimension {dimension} for user {user_id}")

        return results

    def get_user_dimension_embeddings(
        self,
        user_id: str,
        direction: str = "requirements"
    ) -> Dict[str, List[float]]:
        """
        Get all dimension embeddings for a user.

        Args:
            user_id: User identifier
            direction: 'requirements' or 'offerings'

        Returns:
            Dictionary of {dimension: embedding_vector}
        """
        embeddings = {}

        # Get all embeddings for user in one call (more efficient)
        all_embeddings = postgresql_adapter.get_user_embeddings(user_id)

        if not all_embeddings:
            return embeddings

        for dim_config in self.config.dimensions:
            dimension = dim_config.dimension
            embedding_type = f"{direction}_{dimension}"

            # Check if this specific dimension exists
            if embedding_type in all_embeddings:
                emb_data = all_embeddings[embedding_type]
                if emb_data:
                    embeddings[dimension] = emb_data.get('vector_data', [])

        return embeddings

    def compute_dimension_similarity(
        self,
        user_vector: List[float],
        candidate_vector: List[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            user_vector: User's embedding vector
            candidate_vector: Candidate's embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        if not user_vector or not candidate_vector:
            return 0.0

        if len(user_vector) != len(candidate_vector):
            logger.warning("Vector dimension mismatch")
            return 0.0

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(user_vector, candidate_vector))
        norm_a = sum(a * a for a in user_vector) ** 0.5
        norm_b = sum(b * b for b in candidate_vector) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def calculate_weighted_score(
        self,
        dimension_similarities: Dict[str, float]
    ) -> Tuple[float, List[DimensionScore]]:
        """
        Calculate weighted total score from dimension similarities.

        Args:
            dimension_similarities: Dictionary of {dimension: similarity}

        Returns:
            Tuple of (total_weighted_score, list_of_dimension_scores)
        """
        dimension_scores = []
        total_weighted_score = 0.0

        for dim_config in self.config.dimensions:
            dimension = dim_config.dimension
            similarity = dimension_similarities.get(dimension, 0.0)
            weighted = similarity * dim_config.weight

            dimension_scores.append(DimensionScore(
                dimension=dimension,
                similarity=similarity,
                weight=dim_config.weight,
                weighted_score=weighted,
                matched=similarity > 0.5
            ))

            total_weighted_score += weighted

        return total_weighted_score, dimension_scores

    def score_to_tier(self, score: float) -> MatchTier:
        """Convert weighted score to match tier.

        Thresholds aligned with user expectations:
        - 80%+ = Perfect Match (exceptional alignment)
        - 65-79% = Strong Match (high compatibility)
        - 45-64% = Worth Exploring (good potential)
        - <45% = Low (limited overlap, not shown by default)
        """
        if score >= 0.80:
            return MatchTier.PERFECT
        elif score >= 0.65:
            return MatchTier.STRONG
        elif score >= 0.45:
            return MatchTier.WORTH_EXPLORING
        else:
            return MatchTier.LOW

    def find_multi_vector_matches(
        self,
        user_id: str,
        min_tier: MatchTier = MatchTier.WORTH_EXPLORING,
        limit: int = 20
    ) -> List[MultiVectorMatchResult]:
        """
        Find matches using multi-vector similarity.

        Args:
            user_id: User identifier
            min_tier: Minimum match tier to include
            limit: Maximum number of matches to return

        Returns:
            List of MultiVectorMatchResult sorted by total score
        """
        try:
            # 1. Prepare weights config for DB query
            weights_map = {d.dimension: d.weight for d in self.config.dimensions}
            
            # 2. Execute DB-optimized query
            # This returns top candidates sorted by weighted score
            db_matches = postgresql_adapter.find_multi_vector_matches(
                user_id=user_id,
                dimension_weights=weights_map,
                limit=limit * 2  # Fetch extra to allow for post-filtering if needed
            )
            
            if not db_matches:
                logger.info(f"No multi-vector matches found via DB for {user_id}")
                return []

            # 3. Process results into MultiVectorMatchResult objects
            matches = []
            for row in db_matches:
                candidate_id = row['user_id']
                total_score = float(row['total_score'])
                
                # Dimension scores come as a JSON dict from the DB query
                # {'industry': 0.8, 'stage': 0.5}
                raw_scores_map = row.get('dimension_scores', {})
                
                dimension_scores = []
                
                # Reconstruct DimensionScore objects
                for dim_config in self.config.dimensions:
                    dim = dim_config.dimension
                    similarity = float(raw_scores_map.get(dim, 0.0))
                    weighted = similarity * dim_config.weight
                    
                    dimension_scores.append(DimensionScore(
                        dimension=dim,
                        similarity=similarity,
                        weight=dim_config.weight,
                        weighted_score=weighted,
                        matched=similarity > 0.5
                    ))

                # Check tiers and required dimensions
                tier = self.score_to_tier(total_score)
                
                # Check required dimensions
                # Note: The DB query sums everything, but "required" means 
                # we might want to strictly filter out if a specific dim is 0.
                # The prompt implies we should respect the 'required' flag.
                required_met = all(
                    raw_scores_map.get(d.dimension, 0) > 0.3
                    for d in self.config.dimensions
                    if d.required
                )

                if required_met and self._tier_meets_minimum(tier, min_tier):
                    matches.append(MultiVectorMatchResult(
                        user_id=candidate_id,
                        total_score=total_score,
                        tier=tier,
                        dimension_scores=dimension_scores,
                        explanation=self._generate_explanation(dimension_scores, tier)
                    ))

            # Sort by score and limit
            matches.sort(key=lambda m: m.total_score, reverse=True)
            return matches[:limit]

        except Exception as e:
            logger.error(f"Error in multi-vector matching for {user_id}: {e}")
            return []

    def _tier_meets_minimum(self, tier: MatchTier, min_tier: MatchTier) -> bool:
        """Check if a tier meets the minimum requirement."""
        tier_order = [MatchTier.LOW, MatchTier.WORTH_EXPLORING, MatchTier.STRONG, MatchTier.PERFECT]
        return tier_order.index(tier) >= tier_order.index(min_tier)

    def _generate_explanation(
        self,
        dimension_scores: List[DimensionScore],
        tier: MatchTier
    ) -> str:
        """Generate human-readable explanation for match."""
        # Find strongest dimensions
        strong = [d for d in dimension_scores if d.similarity >= 0.7]
        moderate = [d for d in dimension_scores if 0.5 <= d.similarity < 0.7]

        parts = []
        if tier == MatchTier.PERFECT:
            parts.append("Excellent match across all dimensions.")
        elif tier == MatchTier.STRONG:
            parts.append("Strong compatibility.")

        if strong:
            dims = ", ".join(d.dimension.replace("_", " ") for d in strong[:3])
            parts.append(f"Best alignment: {dims}.")

        return " ".join(parts) if parts else "Compatible based on profile analysis."


# Global instance
multi_vector_matcher = MultiVectorMatcher()
