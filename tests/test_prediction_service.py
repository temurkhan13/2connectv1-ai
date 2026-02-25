"""
Unit tests for PredictionService.
Tests fuzzy matching and answer prediction functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPredictionServiceWordMatching:
    """Tests for the word boundary matching fix."""

    @pytest.fixture
    def service(self):
        """Create PredictionService with mocked OpenAI client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('app.services.prediction_service.OpenAI'):
                from app.services.prediction_service import PredictionService
                return PredictionService()

    def test_single_char_does_not_match(self, service):
        """Single character 'I' should NOT match 'Investor'."""
        result = service._is_significant_word_match("I", "Investor")
        assert result == 0.0, "Single char 'I' should not match 'Investor'"

    def test_single_char_does_not_match_a(self, service):
        """Single character 'a' should NOT match 'Angel Investor'."""
        result = service._is_significant_word_match("a", "Angel Investor")
        assert result == 0.0, "Single char 'a' should not match 'Angel Investor'"

    def test_short_substring_does_not_match(self, service):
        """Short substring 'in' should NOT match 'Investment'."""
        result = service._is_significant_word_match("in", "Investment Banking")
        assert result == 0.0, "Short 'in' should not match 'Investment Banking'"

    def test_word_boundary_match(self, service):
        """Complete word 'Angel' should match 'Angel Investor'."""
        result = service._is_significant_word_match("Angel", "Angel Investor")
        assert result >= 0.85, "Complete word 'Angel' should match with high score"

    def test_word_boundary_match_lowercase(self, service):
        """Case-insensitive word match 'angel' should match 'Angel Investor'."""
        result = service._is_significant_word_match("angel", "Angel Investor")
        assert result >= 0.85, "Case-insensitive 'angel' should match"

    def test_exact_match(self, service):
        """Exact match should return 1.0."""
        result = service._is_significant_word_match("Investment Banking", "Investment Banking")
        assert result == 1.0, "Exact match should return 1.0"

    def test_significant_prefix_match(self, service):
        """Significant prefix 'Invest' should match 'Investment' with decent score."""
        result = service._is_significant_word_match("Invest", "Investment")
        # 'Invest' is 6 chars, 'Investment' is 10 chars, so 60% - should get some score
        assert result >= 0.65, "Prefix 'Invest' should get reasonable score"

    def test_minimum_length_enforced(self, service):
        """Strings shorter than min_length should return 0."""
        result = service._is_significant_word_match("AB", "ABCD", min_length=3)
        assert result == 0.0, "String shorter than min_length should return 0"


class TestPredictionServiceFuzzyMatching:
    """Tests for overall fuzzy matching behavior."""

    @pytest.fixture
    def service(self):
        """Create PredictionService with mocked OpenAI client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('app.services.prediction_service.OpenAI'):
                from app.services.prediction_service import PredictionService
                return PredictionService()

    @pytest.fixture
    def sample_options(self):
        return [
            {"label": "Investor", "value": "investor"},
            {"label": "Entrepreneur", "value": "entrepreneur"},
            {"label": "Angel Investor", "value": "angel_investor"},
            {"label": "Venture Capitalist", "value": "vc"},
        ]

    def test_find_best_match_exact(self, service, sample_options):
        """Exact match should return the option."""
        result = service.find_best_match("Investor", sample_options)
        assert result is not None
        assert result['option']['label'] == "Investor"
        assert result['score'] >= 0.9

    def test_find_best_match_case_insensitive(self, service, sample_options):
        """Match should be case-insensitive."""
        result = service.find_best_match("investor", sample_options)
        assert result is not None
        assert result['option']['label'] == "Investor"

    def test_find_best_match_no_false_positive(self, service, sample_options):
        """Single 'I' should NOT match 'Investor'."""
        result = service.find_best_match("I", sample_options)
        # Should either return None or have score < 0.6 threshold
        if result is not None:
            assert result['score'] < 0.6, "Single 'I' should not pass threshold"

    def test_find_best_match_partial(self, service, sample_options):
        """Partial match 'Entre' should match 'Entrepreneur'."""
        result = service.find_best_match("Entre", sample_options)
        # Might not pass threshold but shouldn't false-positive
        if result is not None:
            assert result['option']['label'] == "Entrepreneur"

    def test_find_best_match_empty_options(self, service):
        """Empty options should return None."""
        result = service.find_best_match("test", [])
        assert result is None

    def test_find_best_match_empty_response(self, service, sample_options):
        """Empty user response should return None."""
        result = service.find_best_match("", sample_options)
        assert result is None

    def test_similarity_calculation(self, service):
        """Test similarity ratio calculation."""
        # Exact match
        assert service.calculate_similarity("test", "test") == 1.0
        # Different strings
        assert service.calculate_similarity("test", "xyz") < 0.5
        # Similar strings
        similarity = service.calculate_similarity("testing", "test")
        assert 0.5 < similarity < 1.0


class TestPredictionServicePredictAnswer:
    """Tests for the main predict_answer method."""

    @pytest.fixture
    def service(self):
        """Create PredictionService with mocked OpenAI client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('app.services.prediction_service.OpenAI') as mock_openai:
                # Mock the chat completion
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="Please select from the available options."))]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                from app.services.prediction_service import PredictionService
                return PredictionService()

    @pytest.fixture
    def sample_options(self):
        return [
            {"label": "Option A", "value": "a"},
            {"label": "Option B", "value": "b"},
            {"label": "Option C", "value": "c"},
        ]

    def test_predict_answer_exact_match(self, service, sample_options):
        """Exact match should return predicted_answer and valid_answer=True."""
        result = service.predict_answer("Option A", sample_options)
        assert result["predicted_answer"] == "Option A"
        assert result["valid_answer"] is True
        assert result["fallback_text"] == ""

    def test_predict_answer_no_match(self, service, sample_options):
        """No match should return fallback text."""
        result = service.predict_answer("Something completely different", sample_options)
        assert result["predicted_answer"] is None
        assert result["valid_answer"] is None
        assert len(result["fallback_text"]) > 0

    def test_predict_answer_empty_input(self, service, sample_options):
        """Empty input should return appropriate response."""
        result = service.predict_answer("", sample_options)
        assert result["predicted_answer"] is None
        assert result["valid_answer"] is None
        assert "Please provide a response" in result["fallback_text"]

    def test_predict_answer_whitespace_input(self, service, sample_options):
        """Whitespace-only input should return appropriate response."""
        result = service.predict_answer("   ", sample_options)
        assert result["predicted_answer"] is None
        assert result["valid_answer"] is None
