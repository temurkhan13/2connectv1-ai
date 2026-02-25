"""
Service for predicting and validating user answers with fuzzy matching.
"""
import os
import re
import logging
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for predicting answers with fuzzy matching and LLM fallback."""
    
    def __init__(self):
        """Initialize prediction service."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        # Check if model requires max_completion_tokens
        self.use_max_completion_tokens = self.model.startswith(('o1', 'o3')) or 'gpt-4.1-mini' in self.model
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings (0.0 to 1.0)."""
        return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()
    
    def _is_significant_word_match(self, user_input: str, option_text: str, min_length: int = 3) -> float:
        """
        Check if user input matches option text using word boundaries.
        Returns a similarity boost (0.0 to 0.85) based on match quality.

        A significant match requires:
        1. Input is at least min_length characters
        2. Input matches a complete word or is a significant prefix/suffix
        3. Match is a substantial portion of the option text
        """
        if len(user_input) < min_length:
            return 0.0

        user_lower = user_input.lower().strip()
        option_lower = option_text.lower().strip()

        # Exact match (case-insensitive)
        if user_lower == option_lower:
            return 1.0

        # Word boundary matching: check if user input matches a complete word
        # Use word boundary regex to avoid partial matches like "I" in "Investor"
        word_pattern = r'\b' + re.escape(user_lower) + r'\b'
        if re.search(word_pattern, option_lower):
            # Full word match - high confidence
            return 0.85

        # Check if user input is a significant prefix (at least 50% of option)
        if option_lower.startswith(user_lower) and len(user_lower) >= len(option_lower) * 0.5:
            return 0.75

        # Check if option starts with user's input as a word
        option_words = option_lower.split()
        user_words = user_lower.split()

        # Check if any option word starts with user's complete input
        for option_word in option_words:
            if option_word.startswith(user_lower) and len(user_lower) >= 3:
                return 0.7

        # Check if all user words match beginning of option words
        if len(user_words) > 0:
            matches = 0
            for user_word in user_words:
                if len(user_word) >= 2:  # Min 2 chars per word
                    for option_word in option_words:
                        if option_word.startswith(user_word):
                            matches += 1
                            break
            if matches == len(user_words) and matches > 0:
                return 0.65

        return 0.0

    def find_best_match(self, user_response: str, options: List[dict]) -> Optional[dict]:
        """
        Find the best matching option using fuzzy matching.
        Returns the option with highest similarity if above threshold (0.6).
        """
        if not options or not user_response:
            return None

        user_response_cleaned = user_response.strip()
        best_match = None
        best_score = 0.0
        threshold = 0.6  # 60% similarity threshold

        for option in options:
            # Check both label and value
            label = option.get('label', '')
            value = option.get('value', '')

            # Calculate base similarity with both label and value
            label_similarity = self.calculate_similarity(user_response, label)
            value_similarity = self.calculate_similarity(user_response, value)

            # Take the maximum base similarity
            similarity = max(label_similarity, value_similarity)

            # Apply word boundary matching bonus (replaces buggy substring matching)
            label_word_match = self._is_significant_word_match(user_response_cleaned, label)
            value_word_match = self._is_significant_word_match(user_response_cleaned, value)
            word_match_bonus = max(label_word_match, value_word_match)

            # Use the higher of base similarity or word match score
            similarity = max(similarity, word_match_bonus)

            if similarity > best_score:
                best_score = similarity
                best_match = option

        # Return match only if similarity is above threshold
        if best_score >= threshold:
            return {
                'option': best_match,
                'score': best_score
            }

        return None
    
    def generate_fallback_text(self, user_response: str, options: List[dict]) -> str:
        """Generate a helpful fallback message using LLM when no match is found."""
        # Format options for the prompt
        options_text = "\n".join([
            f"- {opt.get('label', opt.get('value', ''))}"
            for opt in options
        ])
        
        system_prompt = """You are a helpful assistant that guides users to select from available options.
        When a user provides an invalid input, politely inform them that their input doesn't match any available options
        and remind them of the valid options. Keep the message concise and friendly."""
        
        user_prompt = f"""The user entered: "{user_response}"

However, the available options are:
{options_text}

Please generate a friendly, concise message (1-2 sentences) informing the user that their input doesn't match any available options and they should select from the provided options.

Return only the message text, nothing else."""
        
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
            }
            
            # Use appropriate parameter based on model
            if self.use_max_completion_tokens:
                request_params["max_completion_tokens"] = 100
            else:
                request_params["max_tokens"] = 100
            
            response = self.client.chat.completions.create(**request_params)
            
            fallback_text = response.choices[0].message.content.strip()
            # Remove any quotes if present
            fallback_text = fallback_text.strip('"').strip("'").strip()
            return fallback_text
        except Exception as e:
            error_str = str(e)
            # If we get the max_tokens error, retry with max_completion_tokens
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                logger.warning(f"Model {self.model} requires max_completion_tokens, retrying...")
                try:
                    request_params = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_completion_tokens": 100
                    }
                    response = self.client.chat.completions.create(**request_params)
                    fallback_text = response.choices[0].message.content.strip()
                    fallback_text = fallback_text.strip('"').strip("'").strip()
                    return fallback_text
                except Exception as retry_error:
                    logger.error(f"OpenAI API error on retry: {str(retry_error)}")
                    return f"Your input '{user_response}' doesn't match any available options. Please select from: {options_text}"
            else:
                logger.error(f"Error generating fallback text: {error_str}")
                return f"Your input '{user_response}' doesn't match any available options. Please select from: {options_text}"
    
    def predict_answer(self, user_response: str, options: List[dict]) -> dict:
        """
        Predict the correct answer from user input and available options.
        
        Returns:
            {
                "predicted_answer": str or None,
                "valid_answer": bool or None,
                "fallback_text": str
            }
        """
        if not user_response or not user_response.strip():
            return {
                "predicted_answer": None,
                "valid_answer": None,
                "fallback_text": "Please provide a response."
            }
        
        # Try to find a match
        match_result = self.find_best_match(user_response, options)
        
        if match_result:
            # Match found - return the predicted answer
            matched_option = match_result['option']
            # Use label if available, otherwise value
            predicted_answer = matched_option.get('label', matched_option.get('value', ''))
            
            return {
                "predicted_answer": predicted_answer,
                "valid_answer": True,
                "fallback_text": ""
            }
        else:
            # No match found - generate fallback text using LLM
            fallback_text = self.generate_fallback_text(user_response, options)
            
            return {
                "predicted_answer": None,
                "valid_answer": None,
                "fallback_text": fallback_text
            }

