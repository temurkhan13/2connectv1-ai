"""
Pytest configuration and shared fixtures for Reciprocity AI tests.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure environment variables are set for testing."""
    env_vars = {
        'OPENAI_API_KEY': 'test-openai-key',
        'OPENAI_MODEL': 'gpt-4.1-mini',
        'CACHE_ENABLED': 'false',
        'RATE_LIMIT_ENABLED': 'false',
        'SIMILARITY_THRESHOLD': '0.7',
        'EMBEDDING_MODEL': 'sentence-transformers/all-mpnet-base-v2',
        'EMBEDDING_DIMENSION': '768',
        'LOG_LEVEL': 'WARNING',
        # DynamoDB env vars (LocalStack on port 4566)
        'DYNAMO_PROFILE_TABLE_NAME': 'test-profiles',
        'DYNAMO_QUESTIONS_TABLE_NAME': 'test-questions',
        'AWS_REGION': 'us-east-1',
        'AWS_ACCESS_KEY_ID': 'test-access-key',
        'AWS_SECRET_ACCESS_KEY': 'test-secret-key',
        'AWS_ENDPOINT_URL': 'http://localhost:4566',
        # PostgreSQL env vars (Docker on port 5433)
        'DATABASE_URL': 'postgresql://reciprocity_user:reciprocity_pass@localhost:5433/reciprocity_ai',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5433',
        'POSTGRES_USER': 'reciprocity_user',
        'POSTGRES_PASSWORD': 'reciprocity_pass',
        'POSTGRES_DB': 'reciprocity_ai',
        # Redis env vars (Docker on port 6380)
        'REDIS_URL': 'redis://localhost:6380/0',
        'CELERY_BROKER_URL': 'redis://localhost:6380/0',
        # API settings
        'API_KEY': 'test-api-key',
        'BACKEND_API_KEY': 'test-backend-key',
        'BACKEND_WEBHOOK_URL': 'http://localhost:3000/webhooks',
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def sample_uuid():
    """Return a valid UUID for testing."""
    return "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture
def sample_options():
    """Return sample options for testing."""
    return [
        {"label": "Option A", "value": "a"},
        {"label": "Option B", "value": "b"},
        {"label": "Option C", "value": "c"},
    ]


@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client."""
    with patch('openai.OpenAI') as mock:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_redis():
    """Return a mocked Redis client."""
    with patch('redis.Redis') as mock:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock.return_value = mock_client
        yield mock_client
