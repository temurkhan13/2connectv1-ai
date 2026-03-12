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
    """Embedding dimension types aligned with ObjectiveTypes from use_case_templates.py."""
    # Universal dimensions (all users)
    SKILLS = "skills"           # Technical skills and expertise
    INDUSTRY = "industry"       # Industry/domain focus
    STAGE = "stage"             # Company stage (seed, series A, etc.)
    CULTURE = "culture"         # Work culture and style
    REQUIREMENTS = "requirements"  # Traditional requirements
    OFFERINGS = "offerings"     # Traditional offerings

    # FUNDRAISING objective dimensions (founders seeking investment)
    TRACTION = "traction"       # Traction metrics, users, revenue
    MARKET = "market"           # Market size, opportunity, TAM
    TEAM = "team"               # Team composition, background
    FUNDING = "funding"         # Funding needs, runway, use of funds

    # INVESTING objective dimensions (investors seeking deals)
    CHECK_SIZE = "check_size"   # Investment amount preferences
    PORTFOLIO = "portfolio"     # Portfolio companies, track record
    THESIS = "thesis"           # Investment thesis, focus areas

    # HIRING objective dimensions (companies looking to hire)
    ROLE_TYPE = "role_type"     # Type of role being hired
    COMPENSATION = "compensation"  # Salary, equity, benefits
    TEAM_SIZE = "team_size"     # Current team size and growth

    # PARTNERSHIP objective dimensions (business partnerships)
    STRATEGIC_GOALS = "strategic_goals"  # Partnership objectives
    VALUE_EXCHANGE = "value_exchange"    # What each party brings

    # MENTORSHIP objective dimensions (seeking mentors/advisors)
    EXPERTISE = "expertise"       # Domain expertise, advisory areas
    ENGAGEMENT = "engagement"     # Engagement style, time commitment

    # COFOUNDER objective dimensions (finding co-founders)
    VISION = "vision"           # Company vision and direction
    COMMITMENT = "commitment"   # Time commitment, equity expectations
    COMPLEMENTARY = "complementary"  # Complementary skills needed

    # PRODUCT_LAUNCH objective dimensions (launching products)
    GTM = "gtm"                 # Go-to-market strategy
    RESOURCES = "resources"     # Resources and partnerships needed


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
# OBJECTIVE-SPECIFIC DIMENSION CONFIGS
# Aligned with ObjectiveTypes from use_case_templates.py
# =============================================================================

# 1. FUNDRAISING - Founders seeking investment
# Focus slots: funding_need, company_stage, industry_focus, geography, timeline
FUNDRAISING_DIMENSIONS = {
    EmbeddingDimension.TRACTION: DimensionConfig(
        name="Traction & Metrics",
        weight=1.4,
        keywords=[
            "users", "customers", "revenue", "mrr", "arr", "growth", "traction",
            "retention", "engagement", "dau", "mau", "conversion", "sales",
            "paying customers", "downloads", "signups", "active users"
        ],
        extraction_prompt="Extract traction metrics, user numbers, and revenue."
    ),
    EmbeddingDimension.MARKET: DimensionConfig(
        name="Market Opportunity",
        weight=1.3,
        keywords=[
            "market", "tam", "sam", "som", "opportunity", "billion", "million",
            "market size", "addressable", "growing", "trend", "disruption",
            "incumbent", "competition", "competitive advantage", "moat"
        ],
        extraction_prompt="Extract market size, opportunity, and competitive positioning."
    ),
    EmbeddingDimension.TEAM: DimensionConfig(
        name="Team & Background",
        weight=1.2,
        keywords=[
            "team", "co-founder", "cofounder", "founding team", "engineer",
            "designer", "cto", "ceo", "experience", "background", "google",
            "facebook", "meta", "amazon", "microsoft", "stripe", "employees"
        ],
        extraction_prompt="Extract team composition and backgrounds."
    ),
    EmbeddingDimension.FUNDING: DimensionConfig(
        name="Funding & Runway",
        weight=1.5,
        keywords=[
            "raise", "raising", "funding", "investment", "seed", "series",
            "pre-seed", "runway", "valuation", "cap", "safe", "convertible",
            "round", "lead", "follow", "syndicate", "angel", "vc"
        ],
        extraction_prompt="Extract funding needs, round size, and terms."
    )
}

# 2. INVESTING - Investors seeking deals
# Focus slots: check_size, stage_preference, industry_focus, investment_thesis, geography
INVESTING_DIMENSIONS = {
    EmbeddingDimension.CHECK_SIZE: DimensionConfig(
        name="Check Size",
        weight=1.5,
        keywords=[
            "check size", "ticket", "invest", "write", "minimum", "maximum",
            "average", "typical", "range", "k", "million", "25k", "50k",
            "100k", "250k", "500k", "1m", "2m", "5m"
        ],
        extraction_prompt="Extract investment check size preferences."
    ),
    EmbeddingDimension.PORTFOLIO: DimensionConfig(
        name="Portfolio & Track Record",
        weight=1.1,
        keywords=[
            "portfolio", "invested", "backed", "companies", "exits", "ipo",
            "acquisition", "unicorn", "returns", "fund", "aum", "deployed",
            "track record", "previous investments"
        ],
        extraction_prompt="Extract portfolio companies and track record."
    ),
    EmbeddingDimension.THESIS: DimensionConfig(
        name="Investment Thesis",
        weight=1.4,
        keywords=[
            "thesis", "focus", "interested", "looking for", "prefer",
            "conviction", "bet", "trend", "vertical", "horizontal",
            "platform", "marketplace", "saas", "deep tech", "ai", "climate"
        ],
        extraction_prompt="Extract investment thesis and focus areas."
    )
}

# 3. HIRING - Companies looking to hire
# Focus slots: role_type, team_size, industry_focus, geography, engagement_style
HIRING_DIMENSIONS = {
    EmbeddingDimension.ROLE_TYPE: DimensionConfig(
        name="Role Type",
        weight=1.4,
        keywords=[
            "hiring", "role", "position", "engineer", "designer", "manager",
            "developer", "analyst", "specialist", "coordinator", "director",
            "vp", "head of", "c-level", "executive", "senior", "junior"
        ],
        extraction_prompt="Extract the type of role being hired for."
    ),
    EmbeddingDimension.COMPENSATION: DimensionConfig(
        name="Compensation Package",
        weight=1.3,
        keywords=[
            "salary", "compensation", "equity", "stock", "options", "bonus",
            "benefits", "package", "budget", "range", "offer", "k", "per year"
        ],
        extraction_prompt="Extract compensation and benefits offered."
    ),
    EmbeddingDimension.TEAM_SIZE: DimensionConfig(
        name="Team & Growth",
        weight=1.1,
        keywords=[
            "team", "employees", "headcount", "growing", "scaling", "expand",
            "department", "division", "startup", "enterprise", "company size"
        ],
        extraction_prompt="Extract team size and growth plans."
    )
}

# 4. PARTNERSHIP - Business partnerships
# Focus slots: primary_goal, industry_focus, company_stage, geography, engagement_style
PARTNERSHIP_DIMENSIONS = {
    EmbeddingDimension.STRATEGIC_GOALS: DimensionConfig(
        name="Strategic Goals",
        weight=1.4,
        keywords=[
            "partner", "partnership", "strategic", "collaboration", "alliance",
            "joint venture", "integration", "distribution", "channel", "reseller",
            "co-marketing", "white label", "licensing", "synergy"
        ],
        extraction_prompt="Extract partnership objectives and strategic goals."
    ),
    EmbeddingDimension.VALUE_EXCHANGE: DimensionConfig(
        name="Value Exchange",
        weight=1.3,
        keywords=[
            "offer", "bring", "provide", "capabilities", "resources", "access",
            "network", "expertise", "technology", "customers", "market",
            "mutual", "benefit", "value proposition", "complementary"
        ],
        extraction_prompt="Extract what each party brings to the partnership."
    )
}

# 5. MENTORSHIP - Seeking mentors/advisors
# Focus slots: primary_goal, industry_focus, experience_years, engagement_style
MENTORSHIP_DIMENSIONS = {
    EmbeddingDimension.EXPERTISE: DimensionConfig(
        name="Domain Expertise",
        weight=1.4,
        keywords=[
            "expert", "specialist", "domain", "vertical", "advise", "consult",
            "mentor", "coach", "board", "strategic", "operational", "growth",
            "sales", "marketing", "product", "engineering", "finance", "legal"
        ],
        extraction_prompt="Extract advisory expertise and domain focus."
    ),
    EmbeddingDimension.ENGAGEMENT: DimensionConfig(
        name="Engagement Model",
        weight=1.2,
        keywords=[
            "hours", "monthly", "weekly", "retainer", "equity", "advisory shares",
            "board seat", "formal", "informal", "calls", "meetings", "available",
            "commitment", "part-time", "fractional", "time"
        ],
        extraction_prompt="Extract advisory engagement style and availability."
    )
}

# 6. COFOUNDER - Finding co-founders
# Focus slots: primary_goal, industry_focus, company_stage, engagement_style, experience_years
COFOUNDER_DIMENSIONS = {
    EmbeddingDimension.VISION: DimensionConfig(
        name="Vision & Direction",
        weight=1.5,
        keywords=[
            "vision", "mission", "goal", "direction", "build", "create",
            "company", "startup", "idea", "opportunity", "problem", "solution",
            "industry", "market", "disrupt", "transform"
        ],
        extraction_prompt="Extract company vision and direction."
    ),
    EmbeddingDimension.COMMITMENT: DimensionConfig(
        name="Commitment Level",
        weight=1.4,
        keywords=[
            "full-time", "part-time", "commitment", "dedicated", "equity",
            "split", "ownership", "sweat equity", "vesting", "cliff",
            "timeline", "when", "start", "available"
        ],
        extraction_prompt="Extract commitment level and equity expectations."
    ),
    EmbeddingDimension.COMPLEMENTARY: DimensionConfig(
        name="Complementary Skills",
        weight=1.3,
        keywords=[
            "skills", "technical", "business", "operations", "sales", "marketing",
            "product", "engineering", "design", "finance", "legal", "complementary",
            "looking for", "need", "missing", "gap"
        ],
        extraction_prompt="Extract complementary skills needed in co-founder."
    )
}

# 7. PRODUCT_LAUNCH - Launching products
# Focus slots: primary_goal, user_type, industry_focus, company_stage, geography, requirements, offerings, team_size
PRODUCT_LAUNCH_DIMENSIONS = {
    EmbeddingDimension.GTM: DimensionConfig(
        name="Go-to-Market",
        weight=1.4,
        keywords=[
            "launch", "go-to-market", "gtm", "rollout", "release", "beta",
            "mvp", "product market fit", "customers", "acquisition", "growth",
            "distribution", "channel", "sales", "marketing", "outbound"
        ],
        extraction_prompt="Extract go-to-market strategy and launch plans."
    ),
    EmbeddingDimension.RESOURCES: DimensionConfig(
        name="Resources Needed",
        weight=1.3,
        keywords=[
            "need", "looking for", "resource", "partner", "help", "support",
            "marketing", "sales", "funding", "distribution", "channel",
            "advisor", "expertise", "network", "connections"
        ],
        extraction_prompt="Extract resources and partnerships needed for launch."
    )
}

# 8. NETWORKING - General networking (uses universal dimensions only)
# Focus slots: primary_goal, industry_focus, user_type, geography
NETWORKING_DIMENSIONS = {}  # Uses universal dimensions

# =============================================================================
# OBJECTIVE TO DIMENSIONS MAPPING
# Matches ObjectiveType enum values from use_case_templates.py
# =============================================================================

OBJECTIVE_DIMENSION_MAPPING = {
    # Primary objective types (enum values)
    "fundraising": FUNDRAISING_DIMENSIONS,
    "investing": INVESTING_DIMENSIONS,
    "hiring": HIRING_DIMENSIONS,
    "partnership": PARTNERSHIP_DIMENSIONS,
    "mentorship": MENTORSHIP_DIMENSIONS,
    "cofounder": COFOUNDER_DIMENSIONS,
    "product_launch": PRODUCT_LAUNCH_DIMENSIONS,
    "networking": NETWORKING_DIMENSIONS,

    # Common keyword aliases for fuzzy matching
    "seeking investment": FUNDRAISING_DIMENSIONS,
    "raise funding": FUNDRAISING_DIMENSIONS,
    "raise capital": FUNDRAISING_DIMENSIONS,
    "looking for investors": FUNDRAISING_DIMENSIONS,

    "looking to invest": INVESTING_DIMENSIONS,
    "angel investor": INVESTING_DIMENSIONS,
    "vc": INVESTING_DIMENSIONS,
    "venture capital": INVESTING_DIMENSIONS,

    "hire": HIRING_DIMENSIONS,
    "hiring talent": HIRING_DIMENSIONS,
    "recruit": HIRING_DIMENSIONS,
    "looking to hire": HIRING_DIMENSIONS,

    "seeking partnership": PARTNERSHIP_DIMENSIONS,
    "business partnership": PARTNERSHIP_DIMENSIONS,
    "strategic alliance": PARTNERSHIP_DIMENSIONS,

    "seeking mentorship": MENTORSHIP_DIMENSIONS,
    "find mentor": MENTORSHIP_DIMENSIONS,
    "advisor": MENTORSHIP_DIMENSIONS,
    "looking for guidance": MENTORSHIP_DIMENSIONS,

    "find cofounder": COFOUNDER_DIMENSIONS,
    "co-founder": COFOUNDER_DIMENSIONS,
    "founding team": COFOUNDER_DIMENSIONS,
    "looking for cofounder": COFOUNDER_DIMENSIONS,

    "launch product": PRODUCT_LAUNCH_DIMENSIONS,
    "product rollout": PRODUCT_LAUNCH_DIMENSIONS,
    "go-to-market": PRODUCT_LAUNCH_DIMENSIONS,
    "gtm": PRODUCT_LAUNCH_DIMENSIONS,

    "network": NETWORKING_DIMENSIONS,
    "connect": NETWORKING_DIMENSIONS,
    "general networking": NETWORKING_DIMENSIONS,
}


def get_dimensions_for_objective(objective: str) -> Dict[str, DimensionConfig]:
    """
    Get objective-specific embedding dimensions.

    Args:
        objective: The user's primary_goal/objective (e.g., "fundraising", "hiring")

    Returns:
        Dict of dimension name to DimensionConfig for that objective
    """
    if not objective:
        return {}

    objective_lower = objective.lower().strip()

    # Direct match
    if objective_lower in OBJECTIVE_DIMENSION_MAPPING:
        return OBJECTIVE_DIMENSION_MAPPING[objective_lower]

    # Partial match (e.g., "seeking investment for my startup" contains "seeking investment")
    for key, dimensions in OBJECTIVE_DIMENSION_MAPPING.items():
        if key in objective_lower or objective_lower in key:
            return dimensions

    return {}


# =============================================================================
# MULTI-VECTOR EMBEDDING SERVICE
# =============================================================================

class MultiVectorEmbeddingService:
    """
    Service for creating and managing multi-dimensional embeddings.

    This creates separate embeddings for different aspects of a user's
    profile, allowing for more nuanced matching.

    Enhanced (March 2026): Now supports objective-specific embedding dimensions
    aligned with ObjectiveTypes from use_case_templates.py:
    - FUNDRAISING, INVESTING, HIRING, PARTNERSHIP
    - MENTORSHIP, COFOUNDER, PRODUCT_LAUNCH, NETWORKING
    """

    def __init__(self):
        self.universal_dimensions = DIMENSION_CONFIGS
        self.dimension_configs = DIMENSION_CONFIGS  # Keep for backward compatibility
        self.base_embedding_service = embedding_service

    def get_all_dimensions_for_user(self, user_type: Optional[str] = None) -> Dict[str, DimensionConfig]:
        """
        Get all applicable dimensions for a user (universal + objective-specific).

        Args:
            user_type: The user's primary_goal/objective (e.g., "fundraising", "hiring")
                       Note: Parameter kept as user_type for backward compatibility

        Returns:
            Combined dict of universal and objective-specific dimensions
        """
        # Start with universal dimensions
        all_dims = dict(self.universal_dimensions)

        # Add objective-specific dimensions if provided
        if user_type:
            objective_dims = get_dimensions_for_objective(user_type)
            all_dims.update(objective_dims)

        return all_dims

    def extract_dimension_text(
        self,
        full_text: str,
        dimension: str,
        dimension_config: Optional[DimensionConfig] = None
    ) -> str:
        """
        Extract text relevant to a specific dimension from full text.

        Uses keyword matching to identify relevant sentences.
        Could be enhanced with LLM-based extraction.

        Args:
            full_text: The combined text to extract from
            dimension: The dimension name
            dimension_config: Optional config (if not in universal configs)
        """
        config = dimension_config or self.dimension_configs.get(dimension)
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
        store_in_db: bool = True,
        user_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate multiple embeddings for different dimensions.

        Enhanced (March 2026): Now generates role-specific embeddings based on
        user_type. For example, founders get traction/market/team/funding
        dimensions while investors get check_size/portfolio/thesis dimensions.

        Args:
            user_id: User identifier
            requirements_text: User's requirements text
            offerings_text: User's offerings text
            store_in_db: Whether to store in database
            user_type: User's role (founder, investor, job_seeker, advisor)

        Returns:
            Dict with dimension embeddings including role-specific ones
        """
        results = {
            "user_id": user_id,
            "user_type": user_type,
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
                    metadata={"dimension": "requirements", "multi_vector": True, "user_type": user_type}
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
                    metadata={"dimension": "offerings", "multi_vector": True, "user_type": user_type}
                )

        # Generate dimension-specific embeddings
        combined_text = f"{requirements_text}\n{offerings_text}"

        # Get ALL applicable dimensions (universal + role-specific)
        all_dimensions = self.get_all_dimensions_for_user(user_type)

        # Track which are objective-specific for logging
        objective_specific_dims = get_dimensions_for_objective(user_type) if user_type else {}

        for dim_name, config in all_dimensions.items():
            try:
                # Extract dimension-specific text
                dim_text = self.extract_dimension_text(combined_text, dim_name, config)

                if not dim_text.strip():
                    continue

                # Generate embedding for this dimension
                dim_embedding = self.base_embedding_service.generate_embedding(dim_text)

                if dim_embedding:
                    emb_type = f"{dim_name}_combined"
                    is_role_specific = dim_name in objective_specific_dims

                    results["dimensions"][emb_type] = {
                        "vector": dim_embedding,
                        "dimension": len(dim_embedding),
                        "weight": config.weight,
                        "source": "extracted",
                        "role_specific": is_role_specific
                    }

                    if store_in_db:
                        postgresql_adapter.store_embedding(
                            user_id=user_id,
                            embedding_type=emb_type,
                            vector_data=dim_embedding,
                            metadata={
                                "dimension": dim_name,
                                "weight": config.weight,
                                "multi_vector": True,
                                "user_type": user_type,
                                "role_specific": is_role_specific
                            }
                        )

            except Exception as e:
                logger.error(f"Error generating {dim_name} embedding for {user_id}: {e}")

        # Log summary
        universal_count = sum(1 for d in results["dimensions"].values() if not d.get("role_specific"))
        role_count = sum(1 for d in results["dimensions"].values() if d.get("role_specific"))

        logger.info(
            f"Generated {len(results['dimensions'])} dimension embeddings for user {user_id} "
            f"(universal: {universal_count}, role-specific: {role_count}, user_type: {user_type})"
        )

        return results

    def calculate_weighted_similarity(
        self,
        user_embeddings: Dict[str, Any],
        match_embeddings: Dict[str, Any],
        dimension_weights: Optional[Dict[str, float]] = None,
        user_type: Optional[str] = None,
        match_user_type: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted similarity across multiple dimensions.

        Enhanced (March 2026): Now supports role-specific dimension weights.

        Args:
            user_embeddings: User's multi-dimensional embeddings
            match_embeddings: Match's multi-dimensional embeddings
            dimension_weights: Optional custom weights per dimension
            user_type: User's role for weight lookup
            match_user_type: Match's role for weight lookup

        Returns:
            Tuple of (weighted_score, per_dimension_scores)
        """
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

        # Build combined dimension config lookup (universal + both users' role-specific)
        all_configs = dict(self.universal_dimensions)
        if user_type:
            all_configs.update(get_dimensions_for_objective(user_type))
        if match_user_type:
            all_configs.update(get_dimensions_for_objective(match_user_type))

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
                # Use config weight (check role-specific first, then universal)
                base_dim = dim.replace('_combined', '').replace('_requirements', '').replace('_offerings', '')
                config = all_configs.get(base_dim)
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
