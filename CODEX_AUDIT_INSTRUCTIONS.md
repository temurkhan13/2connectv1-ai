# Codex Audit Instructions — Reciprocity AI Platform

## Project Overview

This is a **FastAPI-based AI matching platform** that connects professionals based on their goals, skills, and offerings. Users chat with an AI during onboarding, the system extracts their persona, generates embeddings, and matches them with compatible users.

**Tech Stack:**
- Python 3.12 + FastAPI + Pydantic v2
- PostgreSQL with pgvector for vector similarity search
- Redis for caching (7-day TTL for embeddings)
- DynamoDB (via LocalStack) for user profiles
- Celery for async task processing
- OpenAI GPT-4.1-mini for question modification

---

## Audit Scope

Please perform a comprehensive code audit covering:

### 1. Security Audit
- [ ] API key authentication implementation (`middleware/auth.py`, `routers/*.py`)
- [ ] Input validation and sanitization (`schemas/*.py`)
- [ ] SQL injection prevention in PostgreSQL queries
- [ ] XSS prevention in user-submitted content
- [ ] Rate limiting implementation (`middleware/rate_limit.py`)
- [ ] Secrets handling (check for hardcoded keys, proper env var usage)
- [ ] CORS configuration (`main.py`)

### 2. Code Quality
- [ ] Type annotations completeness
- [ ] Error handling patterns (try/except usage, error messages)
- [ ] Logging practices (are we logging sensitive data?)
- [ ] Dead code or unused imports
- [ ] Code duplication that should be refactored
- [ ] Docstring coverage for public functions

### 3. Architecture Review
- [ ] Service layer separation (`services/*.py`)
- [ ] Dependency injection patterns
- [ ] Database adapter patterns (`adapters/dynamodb.py`, `adapters/postgres.py`)
- [ ] Celery task design (`workers/*.py`)
- [ ] Configuration management (`core/config.py`)

### 4. Performance
- [ ] N+1 query patterns
- [ ] Proper use of async/await
- [ ] Caching strategy effectiveness
- [ ] Embedding generation efficiency
- [ ] Database indexing recommendations

### 5. Testing
- [ ] Test coverage gaps (check `tests/` directory)
- [ ] Mock patterns (are we testing real behavior?)
- [ ] Edge cases not covered
- [ ] Integration test completeness

---

## Key Files to Review

```
app/
├── main.py                    # FastAPI app entry point
├── core/
│   ├── config.py              # Configuration management
│   └── celery.py              # Celery configuration
├── middleware/
│   ├── auth.py                # API key authentication
│   └── rate_limit.py          # Rate limiting
├── routers/
│   ├── question.py            # Question modification endpoints
│   ├── prediction.py          # Answer prediction endpoints
│   ├── user.py                # User registration/profile endpoints
│   ├── matching.py            # Match finding endpoints
│   └── health.py              # Health check
├── schemas/
│   ├── question.py            # Request/response schemas
│   ├── prediction.py
│   └── user.py
├── services/
│   ├── question_service.py    # OpenAI integration for questions
│   ├── prediction_service.py  # Answer prediction logic
│   ├── user_service.py        # User management
│   ├── matching_service.py    # Embedding-based matching
│   ├── multi_vector_matcher.py # 6-dimension matching algorithm
│   ├── ice_breakers.py        # Conversation starter generation
│   ├── match_explanation.py   # Match explanation generation
│   ├── feedback_learner.py    # ML-based preference learning
│   └── slot_extraction.py     # Extract slots from chat
├── adapters/
│   ├── dynamodb.py            # DynamoDB/PynamoDB models
│   └── postgres.py            # PostgreSQL with pgvector
├── workers/
│   ├── persona_processing.py  # Async persona generation
│   ├── embedding_processing.py # Async embedding generation
│   └── scheduled_matching.py  # Batch matching jobs
└── utils/
    └── cache.py               # Redis caching utilities

tests/
├── test_cache.py
├── test_prediction_service.py
├── test_schemas.py
├── test_routers.py
└── conftest.py

scripts/
├── test_full_journey.py       # Full user journey test (14 tests)
└── verify_e2e.py              # E2E infrastructure verification
```

---

## Specific Questions to Answer

1. **Are there any security vulnerabilities?** (OWASP Top 10)
2. **Is the OpenAI API key properly protected?** (not logged, not exposed)
3. **Are database queries safe from injection?**
4. **Is the rate limiting sufficient for production?**
5. **Are there any race conditions in async code?**
6. **Is the embedding caching strategy optimal?**
7. **Are there any memory leaks in long-running workers?**
8. **Is error handling consistent across all endpoints?**
9. **Are webhook callbacks secure?** (HMAC signing, etc.)
10. **What would you change before deploying to production?**

---

## Expected Output Format

Please provide your audit as a structured report:

```markdown
# Reciprocity AI — Code Audit Report

## Executive Summary
[High-level findings: critical issues, major concerns, overall assessment]

## Critical Issues (Must Fix)
[Security vulnerabilities, data exposure risks, crashes]

## Major Issues (Should Fix)
[Performance problems, architectural concerns, missing validation]

## Minor Issues (Nice to Fix)
[Code style, minor optimizations, documentation gaps]

## Recommendations
[Prioritized list of improvements]

## Files Changed
[If you made any fixes, list them here with brief descriptions]
```

---

## Commands to Run

```bash
# Navigate to project
cd /path/to/reciprocity-ai

# Install dependencies (if needed)
pip install -r requirements.txt

# Run tests to understand current state
python -m pytest tests/ -v

# Run the full journey test
python scripts/test_full_journey.py

# Check for obvious issues
python -m py_compile app/**/*.py
```

---

## Notes

- The project uses Pydantic v2 (not v1) — validator syntax is different
- DynamoDB adapter uses PynamoDB library
- PostgreSQL uses psycopg2 + raw SQL (not SQLAlchemy ORM)
- Rate limiting uses slowapi library
- All tests currently pass (70 unit + 14 journey + 8 E2E = 92 total)

---

*Generated: February 10, 2026*
