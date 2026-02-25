"""
Simple match explanation service.
Provides clear explanations for why users matched.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MatchExplanationService:
    """Simple service for generating match explanations."""
    
    def __init__(self):
        """Initialize the match explanation service."""
        pass
    
    def generate_explanation(self, similarity_score: float, match_type: str) -> str:
        """
        Generate simple explanation for a match.
        
        Args:
            similarity_score: Numerical similarity score (0-1)
            match_type: Type of match (requirements_to_offerings, etc.)
            
        Returns:
            Human-readable explanation string
        """
        try:
            # Quality assessment
            if similarity_score >= 0.8:
                quality = "Excellent match"
            elif similarity_score >= 0.6:
                quality = "Strong match"
            elif similarity_score >= 0.4:
                quality = "Good match"
            else:
                quality = "Moderate match"
            
            # Match type explanation
            if match_type == "requirements_to_offerings":
                type_desc = "Your requirements align well with their offerings"
            elif match_type == "offerings_to_requirements":
                type_desc = "Your offerings match what they're looking for"
            else:
                type_desc = "Similar interests and goals"
            
            return f"{quality} ({similarity_score:.2f}). {type_desc}."
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Match score: {similarity_score:.2f}"
    
    def find_common_keywords(self, text1: str, text2: str) -> List[str]:
        """
        Find common meaningful keywords between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            List of common keywords
        """
        try:
            # Simple keyword extraction
            words1 = set(word.lower().strip() for word in text1.split() if len(word) > 3)
            words2 = set(word.lower().strip() for word in text2.split() if len(word) > 3)
            
            # Remove common stopwords
            stopwords = {
                'this', 'that', 'with', 'from', 'they', 'have', 'will', 
                'been', 'were', 'your', 'their', 'what', 'when', 'where',
                'would', 'could', 'should', 'about', 'which', 'there'
            }
            
            words1 -= stopwords
            words2 -= stopwords
            
            # Find intersection
            common = words1.intersection(words2)
            
            return sorted(list(common))[:8]  # Top 8 keywords
            
        except Exception as e:
            logger.error(f"Error finding common keywords: {str(e)}")
            return []


# Global service instance
match_explanation_service = MatchExplanationService()
