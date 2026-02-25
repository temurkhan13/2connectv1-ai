"""
Unit tests for Pydantic schemas and validation.
Tests input validation, sanitization, and constraints.
"""
import pytest
from pydantic import ValidationError
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.user import (
    FeedbackRequest,
    ApproveSummaryRequest,
    InitiateAIChatRequest,
    ConversationMessage,
    UserRegistrationRequest,
    QuestionAnswer
)
from app.schemas.question import (
    QuestionPayload,
    PreviousUserResponse,
    OptionItem as QuestionOptionItem
)
from app.schemas.prediction import (
    PredictAnswerPayload,
    OptionItem as PredictionOptionItem
)


class TestFeedbackRequestValidation:
    """Tests for FeedbackRequest schema validation."""

    def test_valid_feedback_request(self):
        """Valid feedback request should pass validation."""
        request = FeedbackRequest(
            user_id="123e4567-e89b-12d3-a456-426614174000",
            type="match",
            id="123e4567-e89b-12d3-a456-426614174001",
            feedback="This was a great match!"
        )
        assert request.user_id == "123e4567-e89b-12d3-a456-426614174000"
        assert request.type == "match"

    def test_invalid_uuid_user_id(self):
        """Invalid UUID for user_id should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                user_id="not-a-uuid",
                type="match",
                id="123e4567-e89b-12d3-a456-426614174001",
                feedback="Test feedback"
            )
        assert "Invalid UUID format" in str(exc_info.value)

    def test_invalid_uuid_id(self):
        """Invalid UUID for id should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                user_id="123e4567-e89b-12d3-a456-426614174000",
                type="match",
                id="invalid-id",
                feedback="Test feedback"
            )
        assert "Invalid UUID format" in str(exc_info.value)

    def test_invalid_feedback_type(self):
        """Invalid feedback type should fail validation."""
        with pytest.raises(ValidationError):
            FeedbackRequest(
                user_id="123e4567-e89b-12d3-a456-426614174000",
                type="invalid",
                id="123e4567-e89b-12d3-a456-426614174001",
                feedback="Test feedback"
            )

    def test_feedback_length_limit(self):
        """Feedback exceeding max length should fail."""
        long_feedback = "x" * 5001  # MAX_FEEDBACK_LENGTH is 5000
        with pytest.raises(ValidationError):
            FeedbackRequest(
                user_id="123e4567-e89b-12d3-a456-426614174000",
                type="match",
                id="123e4567-e89b-12d3-a456-426614174001",
                feedback=long_feedback
            )

    def test_feedback_sanitization(self):
        """HTML in feedback should be escaped."""
        request = FeedbackRequest(
            user_id="123e4567-e89b-12d3-a456-426614174000",
            type="match",
            id="123e4567-e89b-12d3-a456-426614174001",
            feedback="<script>alert('xss')</script>"
        )
        assert "<script>" not in request.feedback
        assert "&lt;script&gt;" in request.feedback

    def test_feedback_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        request = FeedbackRequest(
            user_id="123e4567-e89b-12d3-a456-426614174000",
            type="chat",
            id="123e4567-e89b-12d3-a456-426614174001",
            feedback="  test feedback  "
        )
        assert request.feedback == "test feedback"


class TestApproveSummaryRequestValidation:
    """Tests for ApproveSummaryRequest schema validation."""

    def test_valid_request(self):
        """Valid request should pass."""
        request = ApproveSummaryRequest(
            user_id="123e4567-e89b-12d3-a456-426614174000"
        )
        assert request.user_id == "123e4567-e89b-12d3-a456-426614174000"

    def test_invalid_uuid(self):
        """Invalid UUID should fail."""
        with pytest.raises(ValidationError) as exc_info:
            ApproveSummaryRequest(user_id="not-a-uuid")
        assert "Invalid UUID format" in str(exc_info.value)


class TestInitiateAIChatRequestValidation:
    """Tests for InitiateAIChatRequest schema validation."""

    def test_valid_request(self):
        """Valid request should pass."""
        request = InitiateAIChatRequest(
            initiator_id="123e4567-e89b-12d3-a456-426614174000",
            responder_id="123e4567-e89b-12d3-a456-426614174001",
            match_id="123e4567-e89b-12d3-a456-426614174002"
        )
        assert request.template is None

    def test_template_sanitization(self):
        """Template with HTML should be sanitized."""
        request = InitiateAIChatRequest(
            initiator_id="123e4567-e89b-12d3-a456-426614174000",
            responder_id="123e4567-e89b-12d3-a456-426614174001",
            match_id="123e4567-e89b-12d3-a456-426614174002",
            template="<img src='x' onerror='alert(1)'>"
        )
        assert "<img" not in request.template
        assert "&lt;img" in request.template

    def test_template_length_limit(self):
        """Template exceeding max length should fail."""
        long_template = "x" * 2001  # MAX_TEMPLATE_LENGTH is 2000
        with pytest.raises(ValidationError):
            InitiateAIChatRequest(
                initiator_id="123e4567-e89b-12d3-a456-426614174000",
                responder_id="123e4567-e89b-12d3-a456-426614174001",
                match_id="123e4567-e89b-12d3-a456-426614174002",
                template=long_template
            )


class TestConversationMessageValidation:
    """Tests for ConversationMessage schema validation."""

    def test_valid_message(self):
        """Valid message should pass."""
        msg = ConversationMessage(
            sender_id="123e4567-e89b-12d3-a456-426614174000",
            content="Hello, nice to meet you!"
        )
        assert msg.content == "Hello, nice to meet you!"

    def test_content_sanitization(self):
        """Content with HTML should be sanitized."""
        msg = ConversationMessage(
            sender_id="123e4567-e89b-12d3-a456-426614174000",
            content="<b>Bold</b> text"
        )
        assert "<b>" not in msg.content
        assert "&lt;b&gt;" in msg.content

    def test_empty_content_fails(self):
        """Empty content should fail."""
        with pytest.raises(ValidationError):
            ConversationMessage(
                sender_id="123e4567-e89b-12d3-a456-426614174000",
                content=""
            )


class TestQuestionPayloadValidation:
    """Tests for QuestionPayload schema validation."""

    def test_valid_payload(self):
        """Valid payload should pass."""
        payload = QuestionPayload(
            question_id="q1",
            code="Q001",
            prompt="What is your goal?",
            suggestion_chips="Tell me more"
        )
        assert payload.question_id == "q1"

    def test_prompt_length_limit(self):
        """Prompt exceeding max length should fail."""
        long_prompt = "x" * 5001  # MAX_PROMPT_LENGTH is 5000
        with pytest.raises(ValidationError):
            QuestionPayload(
                question_id="q1",
                code="Q001",
                prompt=long_prompt,
                suggestion_chips="chip"
            )

    def test_previous_responses_limit(self):
        """More than 50 previous responses should fail."""
        many_responses = [
            PreviousUserResponse(
                question_id=f"q{i}",
                ai_text="text",
                prompt="prompt",
                suggestion_chips="chips"
            )
            for i in range(51)
        ]
        with pytest.raises(ValidationError) as exc_info:
            QuestionPayload(
                question_id="q1",
                code="Q001",
                prompt="prompt",
                suggestion_chips="chips",
                previous_user_response=many_responses
            )
        assert "Too many previous responses" in str(exc_info.value)


class TestPredictAnswerPayloadValidation:
    """Tests for PredictAnswerPayload schema validation."""

    def test_valid_payload(self):
        """Valid payload should pass."""
        payload = PredictAnswerPayload(
            options=[
                PredictionOptionItem(label="Option A", value="a"),
                PredictionOptionItem(label="Option B", value="b")
            ],
            user_response="I choose A"
        )
        assert len(payload.options) == 2

    def test_user_response_sanitization(self):
        """User response with HTML should be sanitized."""
        payload = PredictAnswerPayload(
            options=[PredictionOptionItem(label="Option A", value="a")],
            user_response="<script>bad</script>"
        )
        assert "<script>" not in payload.user_response

    def test_empty_options_fails(self):
        """Empty options should fail."""
        with pytest.raises(ValidationError):
            PredictAnswerPayload(
                options=[],
                user_response="test"
            )

    def test_too_many_options_fails(self):
        """More than max options should fail."""
        many_options = [
            PredictionOptionItem(label=f"Opt{i}", value=f"v{i}")
            for i in range(51)  # MAX_OPTIONS_COUNT is 50
        ]
        with pytest.raises(ValidationError):
            PredictAnswerPayload(
                options=many_options,
                user_response="test"
            )


class TestUserRegistrationValidation:
    """Tests for UserRegistrationRequest schema validation."""

    def test_valid_registration(self):
        """Valid registration should pass."""
        request = UserRegistrationRequest(
            user_id="123e4567-e89b-12d3-a456-426614174000",
            questions=[
                QuestionAnswer(prompt="What is your goal?", answer="To grow")
            ]
        )
        assert request.user_id == "123e4567-e89b-12d3-a456-426614174000"

    def test_invalid_uuid(self):
        """Invalid UUID should fail."""
        with pytest.raises(ValidationError):
            UserRegistrationRequest(
                user_id="not-valid-uuid",
                questions=[
                    QuestionAnswer(prompt="Test?", answer="Test")
                ]
            )
