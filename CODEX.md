# Reciprocity AI - Codex Instructions

## Project Overview

FastAPI-based AI service for a professional networking platform. Handles user persona generation, semantic matching via embeddings, and feedback learning.

## Tech Stack

- **Framework:** FastAPI + Celery (async task processing)
- **Database:** PostgreSQL with pgvector (embeddings), LocalStack DynamoDB (profiles/matches)
- **AI/ML:** OpenAI GPT-4.1-mini (persona generation), SentenceTransformers all-mpnet-base-v2 (768-dim embeddings)
- **Python:** 3.12

## Architecture

```
app/
├── adapters/          # Database adapters (DynamoDB, PostgreSQL)
├── middleware/        # Auth (API key), rate limiting, error handling
├── routers/           # FastAPI endpoints
├── schemas/           # Pydantic models
├── services/          # Business logic
├── workers/           # Celery tasks (resume, persona, embeddings)
└── main.py            # FastAPI app entry
```

## Key Services

| Service | Purpose |
|---------|---------|
| `persona_service.py` | Generate AI personas from onboarding answers |
| `matching_service.py` | Find matches using embedding similarity |
| `enhanced_matching_service.py` | Bidirectional matching with intent classification |
| `embedding_service.py` | Generate/store 768-dim vectors |
| `feedback_learner.py` | Adjust embeddings based on user feedback |

## Code Conventions

1. **Logging:** Use `logger = logging.getLogger(__name__)`, never `print()`
2. **Exceptions:** Use `except Exception:` not bare `except:`
3. **Docstrings:** Keep concise, one line preferred
4. **Imports:** Standard library first, then third-party, then local
5. **Type hints:** Required on all public methods

## Security Requirements

- API key auth via `X-API-KEY` header
- CORS restricted to specific origins/methods
- No secrets in code (use `.env`)
- UUID validation on user IDs

## Running Locally

```bash
# Start services
docker-compose up -d  # LocalStack, PostgreSQL

# Run API
uvicorn app.main:app --reload --port 8000

# Run Celery (Windows requires --pool=solo)
celery -A app.core.celery worker --loglevel=info --pool=solo
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_matching.py -v
```

## Audit Checklist

When auditing this codebase, check for:

1. **Security**
   - [ ] No hardcoded credentials
   - [ ] Proper input validation
   - [ ] CORS properly configured
   - [ ] API key middleware on all routes

2. **Code Quality**
   - [ ] No `print()` statements (use logger)
   - [ ] No bare `except:` clauses
   - [ ] Proper error handling (no `raise e`)
   - [ ] Connections properly closed

3. **Performance**
   - [ ] No N+1 queries
   - [ ] Proper connection pooling
   - [ ] Appropriate caching

4. **Style**
   - [ ] Consistent naming
   - [ ] Concise docstrings
   - [ ] No commented-out code blocks
   - [ ] Imports organized
