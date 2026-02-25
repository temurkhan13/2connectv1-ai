"""
Advanced filtering service for persona matching and discovery
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime, timedelta
import math

class FilterOperator(Enum):
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    IN = "in"
    NOT_IN = "nin"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    EXISTS = "exists"
    RANGE = "range"

class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"

@dataclass
class FilterCriteria:
    field: str
    operator: FilterOperator
    value: Any
    weight: float = 1.0  # For weighted scoring

@dataclass
class SortCriteria:
    field: str
    order: SortOrder = SortOrder.DESC

@dataclass
class PaginationCriteria:
    page: int = 1
    per_page: int = 20
    max_per_page: int = 100

class PersonaFilteringService:
    """Advanced filtering service with multiple matching strategies"""
    
    def __init__(self):
        self.industry_categories = {
            "Technology": ["SaaS", "AI/ML", "Software", "Hardware", "Cybersecurity", "IoT"],
            "Finance": ["Fintech", "Banking", "Insurance", "Investment", "Crypto", "Blockchain"],
            "Healthcare": ["Biotech", "MedTech", "Digital Health", "Pharma", "Telemedicine"],
            "E-commerce": ["Retail Tech", "MarketPlace", "B2B Commerce", "Supply Chain"],
            "Media": ["Social Media", "Content", "Gaming", "Entertainment", "EdTech"],
            "Professional Services": ["Consulting", "Legal Tech", "HR Tech", "Marketing Tech"]
        }
        
        self.role_levels = {
            "Executive": ["CEO", "CTO", "CFO", "COO", "President", "VP"],
            "Leadership": ["Director", "Head of", "VP", "Lead", "Principal"],
            "Senior": ["Senior", "Sr.", "Staff", "Principal"],
            "Mid-Level": ["Manager", "Lead", "Specialist"],
            "Individual Contributor": ["Engineer", "Developer", "Analyst", "Designer", "Consultant"]
        }
        
        self.experience_mapping = {
            "Junior": (0, 3),
            "Mid": (3, 7),
            "Senior": (7, 12),
            "Executive": (12, 30)
        }

    def build_smart_filters(self, 
                          query_params: Dict[str, Any],
                          user_context: Optional[Dict[str, Any]] = None) -> List[FilterCriteria]:
        """Build intelligent filters based on query parameters and user context"""
        
        filters = []
        
        # Basic field filters
        for field, value in query_params.items():
            if value is None:
                continue
                
            filter_criteria = self._parse_field_filter(field, value)
            if filter_criteria:
                filters.extend(filter_criteria)
        
        # Add contextual filters based on user's profile
        if user_context:
            contextual_filters = self._build_contextual_filters(user_context)
            filters.extend(contextual_filters)
        
        return filters

    def _parse_field_filter(self, field: str, value: Any) -> List[FilterCriteria]:
        """Parse individual field filter with smart operator detection"""
        
        filters = []
        
        # Handle special field mappings
        field_mapping = {
            "industries": ["industries_of_interest", "industry_focus_areas"],
            "skills": ["core_strength_1", "core_strength_2", "core_strength_3"],
            "roles": ["preferred_roles_titles"],
            "objectives": ["core_objectives.primary_goal"]
        }
        
        target_fields = field_mapping.get(field, [field])
        
        for target_field in target_fields:
            # Age range handling
            if field in ["age", "age_range"]:
                if isinstance(value, dict):
                    if "min" in value:
                        filters.append(FilterCriteria(target_field, FilterOperator.GREATER_THAN_EQUAL, value["min"]))
                    if "max" in value:
                        filters.append(FilterCriteria(target_field, FilterOperator.LESS_THAN_EQUAL, value["max"]))
                elif isinstance(value, str) and "-" in value:
                    # Handle "25-35" format
                    min_age, max_age = map(int, value.split("-"))
                    filters.append(FilterCriteria(target_field, FilterOperator.RANGE, (min_age, max_age)))
            
            # List values (IN operator)
            elif isinstance(value, list):
                filters.append(FilterCriteria(target_field, FilterOperator.IN, value))
            
            # String contains (case-insensitive)
            elif isinstance(value, str):
                if value.startswith("!"):
                    # Negation
                    filters.append(FilterCriteria(target_field, FilterOperator.NOT_CONTAINS, value[1:]))
                elif "*" in value or "?" in value:
                    # Wildcard to regex
                    regex_pattern = value.replace("*", ".*").replace("?", ".")
                    filters.append(FilterCriteria(target_field, FilterOperator.REGEX, regex_pattern))
                else:
                    filters.append(FilterCriteria(target_field, FilterOperator.CONTAINS, value))
            
            # Boolean exists check
            elif isinstance(value, bool):
                filters.append(FilterCriteria(target_field, FilterOperator.EXISTS, value))
            
            # Numeric comparison
            elif isinstance(value, (int, float)):
                filters.append(FilterCriteria(target_field, FilterOperator.EQUALS, value))
        
        return filters

    def _build_contextual_filters(self, user_context: Dict[str, Any]) -> List[FilterCriteria]:
        """Build filters based on user's own profile for better matching"""
        
        filters = []
        
        # Industry affinity - find people in similar or complementary industries
        user_industries = user_context.get("industries_of_interest", [])
        if user_industries:
            # Find industry categories
            related_industries = set()
            for industry in user_industries:
                for category, industries in self.industry_categories.items():
                    if industry in industries:
                        related_industries.update(industries)
            
            if related_industries:
                filters.append(FilterCriteria(
                    "industries_of_interest", 
                    FilterOperator.IN, 
                    list(related_industries),
                    weight=0.8
                ))
        
        # Role complementarity
        user_role = user_context.get("preferred_roles_titles", [])
        if user_role and isinstance(user_role, list) and len(user_role) > 0:
            primary_role = user_role[0]
            complementary_roles = self._get_complementary_roles(primary_role)
            if complementary_roles:
                filters.append(FilterCriteria(
                    "preferred_roles_titles",
                    FilterOperator.IN,
                    complementary_roles,
                    weight=0.9
                ))
        
        # Objective compatibility
        user_objective = user_context.get("core_objectives", {}).get("primary_goal")
        if user_objective:
            compatible_objectives = self._get_compatible_objectives(user_objective)
            if compatible_objectives:
                filters.append(FilterCriteria(
                    "core_objectives.primary_goal",
                    FilterOperator.IN,
                    compatible_objectives,
                    weight=1.0
                ))
        
        return filters

    def _get_complementary_roles(self, role: str) -> List[str]:
        """Get roles that complement the given role"""
        
        complementary_mapping = {
            "CEO": ["CTO", "CFO", "VP Engineering", "VP Sales", "VP Marketing"],
            "CTO": ["CEO", "VP Product", "Engineering Manager", "Head of Data"],
            "CFO": ["CEO", "VP Sales", "Head of Business Development"],
            "Founder": ["Co-founder", "Advisor", "Investor", "Mentor"],
            "Engineer": ["Product Manager", "Designer", "Engineering Manager"],
            "Product Manager": ["Engineer", "Designer", "Data Scientist"],
            "Sales": ["Marketing", "Business Development", "Customer Success"],
            "Marketing": ["Sales", "Product Marketing", "Growth"],
            "Investor": ["Founder", "Advisor", "Board Member"],
            "Mentor": ["Founder", "Early-stage entrepreneur"]
        }
        
        # Direct mapping
        for key, complements in complementary_mapping.items():
            if key.lower() in role.lower():
                return complements
        
        # Role level matching
        user_level = self._get_role_level(role)
        if user_level:
            return self.role_levels.get(user_level, [])
        
        return []

    def _get_compatible_objectives(self, objective: str) -> List[str]:
        """Get objectives that are compatible with the given objective"""
        
        compatibility_mapping = {
            "Find a technical co-founder": ["Seeking co-founder", "Looking for partnership", "Technical collaboration"],
            "Seeking mentorship": ["Offering mentorship", "Advisor opportunities"],
            "Offering mentorship": ["Seeking mentorship", "Learning & development"],
            "Raising capital": ["Finding investment opportunities", "Angel investing"],
            "Finding investment opportunities": ["Raising capital", "Seeking funding"],
            "Generating leads": ["Strategic partnerships", "Business development"],
            "Finding job opportunities": ["Recruiting", "Hiring", "Team building"],
            "Recruiting": ["Finding job opportunities", "Career opportunities"]
        }
        
        return compatibility_mapping.get(objective, [])

    def _get_role_level(self, role: str) -> Optional[str]:
        """Determine the level of a role"""
        
        role_lower = role.lower()
        for level, role_keywords in self.role_levels.items():
            for keyword in role_keywords:
                if keyword.lower() in role_lower:
                    return level
        return None

    def apply_filters_to_mongo_query(self, filters: List[FilterCriteria]) -> Dict[str, Any]:
        """Convert FilterCriteria to MongoDB query"""
        
        mongo_query = {}
        or_conditions = []
        
        for filter_criteria in filters:
            field = filter_criteria.field
            operator = filter_criteria.operator
            value = filter_criteria.value
            
            condition = self._build_mongo_condition(field, operator, value)
            
            if condition:
                if filter_criteria.weight < 1.0:
                    # Optional condition (for OR logic)
                    or_conditions.append(condition)
                else:
                    # Required condition (for AND logic)
                    if field in mongo_query:
                        # Merge conditions for same field
                        if isinstance(mongo_query[field], dict) and isinstance(condition[field], dict):
                            mongo_query[field].update(condition[field])
                    else:
                        mongo_query.update(condition)
        
        # Add OR conditions if any
        if or_conditions:
            if "$or" in mongo_query:
                mongo_query["$or"].extend(or_conditions)
            else:
                mongo_query["$or"] = or_conditions
        
        return mongo_query

    def _build_mongo_condition(self, field: str, operator: FilterOperator, value: Any) -> Dict[str, Any]:
        """Build individual MongoDB condition"""
        
        if operator == FilterOperator.EQUALS:
            return {field: value}
        elif operator == FilterOperator.NOT_EQUALS:
            return {field: {"$ne": value}}
        elif operator == FilterOperator.GREATER_THAN:
            return {field: {"$gt": value}}
        elif operator == FilterOperator.GREATER_THAN_EQUAL:
            return {field: {"$gte": value}}
        elif operator == FilterOperator.LESS_THAN:
            return {field: {"$lt": value}}
        elif operator == FilterOperator.LESS_THAN_EQUAL:
            return {field: {"$lte": value}}
        elif operator == FilterOperator.IN:
            return {field: {"$in": value}}
        elif operator == FilterOperator.NOT_IN:
            return {field: {"$nin": value}}
        elif operator == FilterOperator.CONTAINS:
            return {field: {"$regex": re.escape(str(value)), "$options": "i"}}
        elif operator == FilterOperator.NOT_CONTAINS:
            return {field: {"$not": {"$regex": re.escape(str(value)), "$options": "i"}}}
        elif operator == FilterOperator.REGEX:
            return {field: {"$regex": value, "$options": "i"}}
        elif operator == FilterOperator.EXISTS:
            return {field: {"$exists": value}}
        elif operator == FilterOperator.RANGE:
            min_val, max_val = value
            return {field: {"$gte": min_val, "$lte": max_val}}
        
        return {}

    def calculate_match_score(self, 
                            persona: Dict[str, Any], 
                            filters: List[FilterCriteria],
                            user_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate a match score between 0-1 for a persona against filters"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for filter_criteria in filters:
            field_value = self._get_nested_field_value(persona, filter_criteria.field)
            match_score = self._calculate_field_match_score(
                field_value, 
                filter_criteria.operator, 
                filter_criteria.value
            )
            
            weighted_score = match_score * filter_criteria.weight
            total_score += weighted_score
            total_weight += filter_criteria.weight
        
        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0.0
        
        # Add compatibility bonus if user context is provided
        if user_context:
            compatibility_bonus = self._calculate_compatibility_bonus(persona, user_context)
            normalized_score = min(1.0, normalized_score + compatibility_bonus * 0.1)
        
        return normalized_score

    def _get_nested_field_value(self, document: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested field path (e.g., 'core_objectives.primary_goal')"""
        
        value = document
        for field_part in field_path.split('.'):
            if isinstance(value, dict) and field_part in value:
                value = value[field_part]
            else:
                return None
        return value

    def _calculate_field_match_score(self, field_value: Any, operator: FilterOperator, target_value: Any) -> float:
        """Calculate match score for a single field"""
        
        if field_value is None:
            return 0.0
        
        if operator == FilterOperator.EQUALS:
            return 1.0 if field_value == target_value else 0.0
        elif operator == FilterOperator.IN:
            if isinstance(field_value, list):
                # List intersection
                common_items = set(field_value) & set(target_value)
                return len(common_items) / len(target_value) if target_value else 0.0
            else:
                return 1.0 if field_value in target_value else 0.0
        elif operator == FilterOperator.CONTAINS:
            if isinstance(field_value, str) and isinstance(target_value, str):
                return 1.0 if target_value.lower() in field_value.lower() else 0.0
            return 0.0
        elif operator == FilterOperator.RANGE:
            if isinstance(field_value, (int, float)):
                min_val, max_val = target_value
                if min_val <= field_value <= max_val:
                    return 1.0
                else:
                    # Partial score based on distance
                    range_size = max_val - min_val
                    if field_value < min_val:
                        distance = min_val - field_value
                    else:
                        distance = field_value - max_val
                    return max(0.0, 1.0 - (distance / range_size))
            return 0.0
        elif operator == FilterOperator.EXISTS:
            return 1.0 if (field_value is not None) == target_value else 0.0
        
        return 0.0

    def _calculate_compatibility_bonus(self, persona: Dict[str, Any], user_context: Dict[str, Any]) -> float:
        """Calculate compatibility bonus based on complementary attributes"""
        
        bonus = 0.0
        
        # Industry compatibility
        persona_industries = persona.get("industries_of_interest", [])
        user_industries = user_context.get("industries_of_interest", [])
        if persona_industries and user_industries:
            common_industries = set(persona_industries) & set(user_industries)
            if common_industries:
                bonus += 0.2
        
        # Communication style compatibility
        persona_style = persona.get("communication_style")
        user_style = user_context.get("communication_style")
        if persona_style and user_style and persona_style == user_style:
            bonus += 0.1
        
        # Geographic proximity (if location data available)
        # This would require location fields in the data
        
        return min(0.5, bonus)  # Cap bonus at 0.5

    def get_advanced_recommendations(self, 
                                   user_profile: Dict[str, Any],
                                   available_personas: List[Dict[str, Any]],
                                   max_results: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """Get advanced recommendations with scoring"""
        
        # Build contextual filters
        contextual_filters = self._build_contextual_filters(user_profile)
        
        # Calculate scores for all personas
        scored_personas = []
        for persona in available_personas:
            score = self.calculate_match_score(persona, contextual_filters, user_profile)
            if score > 0.1:  # Minimum threshold
                scored_personas.append((persona, score))
        
        # Sort by score and return top results
        scored_personas.sort(key=lambda x: x[1], reverse=True)
        return scored_personas[:max_results]