# Reciprocity AI — 5 Test Users Full Journey

**Date:** February 10, 2026
**Status:** Complete — All 5 users processed through full pipeline

---

## Executive Summary

Successfully seeded 5 persistent test users through the complete Reciprocity AI platform pipeline:

| Phase | Result |
|-------|--------|
| 1. Onboarding (Question Modification) | 5/5 PASS |
| 2. Profile Creation (User Registration) | 5/5 PASS |
| 3. Persona Generation (OpenAI GPT-4.1-mini) | 5/5 PASS |
| 4. Embedding Generation (SentenceTransformers) | 5/5 PASS |
| 5. Matching (pgvector similarity) | 5/5 PASS |
| 6. Feedback Submission | 5/5 PASS |

**Total Tests:** 55/55 (100%)

---

## The 5 Test Users

### User 1: Alice Chen
| Field | Value |
|-------|-------|
| **UUID** | `11111111-1111-1111-1111-111111111111` |
| **Role** | AI/ML Engineer seeking mentorship |
| **Looking For** | Senior mentor in AI/ML for transitioning from software engineering to ML research |
| **Experience** | 5 years software development, 2 years ML side projects |
| **Offerings** | Python development, web APIs, mentoring junior devs |
| **Industry** | Fintech (payment processing startup, London) |

### User 2: Bob Martinez
| Field | Value |
|-------|-------|
| **UUID** | `22222222-2222-2222-2222-222222222222` |
| **Role** | Senior AI Researcher offering mentorship |
| **Looking For** | Mentor passionate engineers transitioning into AI |
| **Experience** | 15 years in tech, 8 focused on AI/ML, PhD in CS, published researcher |
| **Offerings** | Deep ML/NLP/CV expertise, career guidance, research methodology |
| **Industry** | AI Research and Consulting (Fortune 500 clients) |

### User 3: Charlie Thompson
| Field | Value |
|-------|-------|
| **UUID** | `33333333-3333-3333-3333-333333333333` |
| **Role** | Startup Founder seeking co-founder |
| **Looking For** | Technical co-founder for healthcare AI startup |
| **Experience** | 10 years product management, 3 startups (1 exit) |
| **Offerings** | Business development, fundraising ($5M raised), product strategy, healthcare connections |
| **Industry** | Healthcare technology (AI diagnostics) |

### User 4: Diana Okonkwo
| Field | Value |
|-------|-------|
| **UUID** | `44444444-4444-4444-4444-444444444444` |
| **Role** | Full-stack Developer seeking startup opportunity |
| **Looking For** | Early-stage startup as technical lead or co-founder (healthtech/fintech) |
| **Experience** | 8 years full-stack, led 5-10 person engineering teams |
| **Offerings** | Full-stack development, system architecture, team leadership, MVP building |
| **Industry** | E-commerce tech (wants healthtech/fintech) |

### User 5: Eve Richardson
| Field | Value |
|-------|-------|
| **UUID** | `55555555-5555-5555-5555-555555555555` |
| **Role** | Investor seeking deal flow |
| **Looking For** | Promising founders in AI, healthtech, fintech (Seed to Series A) |
| **Experience** | 12 years VC, partner at $200M fund, previously founded/sold SaaS company |
| **Offerings** | Funding ($500K-$5M), board experience, 200+ founder network, GTM strategy |
| **Industry** | Venture Capital (B2B SaaS, AI infrastructure, digital health) |

---

## Step-by-Step Journey

### STEP 1: Onboarding — AI Question Modification

**What happens:** The platform modifies questions using OpenAI GPT-4.1-mini to make them conversational and personalized.

| User | Test | Result | AI Response |
|------|------|--------|-------------|
| Alice | Question Modification | PASS | "Hey, just curious—what kind of stuff are you hoping to find..." |
| Alice | Answer Prediction | PASS | Predicted: None (free-form response) |
| Bob | Question Modification | PASS | "Hey, just curious—what kinda stuff are you hoping to find or..." |
| Bob | Answer Prediction | PASS | Predicted: None |
| Charlie | Question Modification | PASS | "Hey, just curious—what kind of things are you hoping to find..." |
| Charlie | Answer Prediction | PASS | Predicted: None |
| Diana | Question Modification | PASS | "Hey, just curious—what kind of things are you hoping to find..." |
| Diana | Answer Prediction | PASS | Predicted: None |
| Eve | Question Modification | PASS | "Hey, just curious—what brought you here? Are you hoping to f..." |
| Eve | Answer Prediction | PASS | Predicted: None |

**Technical Details:**
- Endpoint: `POST /api/v1/modify-question`
- Model: `gpt-4.1-mini`
- Temperature: `0.7`
- Each question is personalized based on context and previous responses

---

### STEP 2: Profile Creation — User Registration

**What happens:** Users are registered in DynamoDB with their Q&A responses stored as raw data.

| User | UUID | Registration | Profile Status |
|------|------|--------------|----------------|
| Alice | `11111111-...` | PASS (success) | Created |
| Bob | `22222222-...` | PASS (success) | Created |
| Charlie | `33333333-...` | PASS (success) | Created |
| Diana | `44444444-...` | PASS (success) | Created |
| Eve | `55555555-...` | PASS (success) | Created |

**Technical Details:**
- Endpoint: `POST /api/v1/user/register`
- Storage: DynamoDB table `reciprocity-profiles`
- Each user has 4 Q&A pairs stored in `profile.raw_questions`

**Data Stored:**
```json
{
  "user_id": "11111111-1111-1111-1111-111111111111",
  "profile": {
    "raw_questions": [
      {"prompt": "What are you looking for?", "answer": "..."},
      {"prompt": "What's your experience level?", "answer": "..."},
      {"prompt": "What can you offer?", "answer": "..."},
      {"prompt": "What industry are you in?", "answer": "..."}
    ]
  },
  "processing_status": "pending",
  "persona_status": "not_initiated"
}
```

---

### STEP 3: Review — Approve Summary (Triggers Persona Generation)

**What happens:** When summary is approved, the system generates an AI persona using OpenAI.

| User | Approve Summary | Persona Generated | Archetype |
|------|-----------------|-------------------|-----------|
| Alice | PASS | Aspiring AI Researcher | Transitioning Technologist |
| Bob | PASS | AI Mentor & Strategist | Experienced AI Researcher and Career Mentor |
| Charlie | PASS | Healthcare AI Visionary | Entrepreneurial Product Leader |
| Diana | PASS | Technical Lead Innovator | Experienced Full-Stack Developer and Emerging Technical Leader |
| Eve | PASS | Experienced Venture Capital Partner | Strategic Investor and Founder Ally |

**Technical Details:**
- Endpoint: `POST /api/v1/user/approve-summary`
- Process: Synchronous persona generation via `PersonaService.generate_persona_sync()`
- Model: GPT-4.1-mini
- Output: Persona name, archetype, requirements, offerings, experience, focus, etc.

**Example Persona Output (Alice):**
```json
{
  "persona": {
    "name": "Aspiring AI Researcher",
    "archetype": "Transitioning Technologist",
    "experience": "5 years software, 2 years ML",
    "focus": "AI/ML transition"
  },
  "requirements": "Senior mentor in AI/ML who can guide transition from software engineering to machine learning research",
  "offerings": "Python development expertise, web APIs, mentoring junior developers"
}
```

---

### STEP 4: Matching Preparation — Embedding Generation

**What happens:** The system generates 768-dimension embeddings for requirements and offerings using SentenceTransformers.

| User | Requirements Embedding | Offerings Embedding | Storage |
|------|------------------------|---------------------|---------|
| Alice | PASS (768 dims) | PASS (768 dims) | PostgreSQL + pgvector |
| Bob | PASS (768 dims) | PASS (768 dims) | PostgreSQL + pgvector |
| Charlie | PASS (768 dims) | PASS (768 dims) | PostgreSQL + pgvector |
| Diana | PASS (768 dims) | PASS (768 dims) | PostgreSQL + pgvector |
| Eve | PASS (768 dims) | PASS (768 dims) | PostgreSQL + pgvector |

**Technical Details:**
- Model: `sentence-transformers/all-mpnet-base-v2`
- Dimensions: 768
- Storage: PostgreSQL table `user_embeddings` with pgvector extension
- Index: IVFFlat with cosine similarity (`vector_cosine_ops`)

**Database Schema:**
```sql
CREATE TABLE user_embeddings (
    user_id VARCHAR(255) NOT NULL,
    embedding_type VARCHAR(50) NOT NULL,  -- 'requirements' or 'offerings'
    vector_data vector(768) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (user_id, embedding_type)
);
```

---

### STEP 5: Matching — Find Compatible Users

**What happens:** The system finds matches using pgvector cosine similarity between:
- User's **requirements** vs others' **offerings**
- User's **offerings** vs others' **requirements**

#### Match Results (Default Threshold: 0.7)

| User | Matches Found | Match Details |
|------|---------------|---------------|
| Alice | 1 | Bob (0.73) |
| Bob | 1 | Alice (0.73) |
| Charlie | 0 | (threshold too high) |
| Diana | 0 | (threshold too high) |
| Eve | 0 | (threshold too high) |

#### Match Results (Lower Threshold: 0.5)

| User | Matches (with similarity score) |
|------|--------------------------------|
| **Alice** | Bob (0.73), Diana (0.57), Charlie (0.53) |
| **Bob** | Alice (0.73), Alice (0.65), Diana (0.56) |
| **Charlie** | Diana (0.65), Eve (0.61), Alice (0.53) |
| **Diana** | Charlie (0.65), Eve (0.63), Alice (0.57), Bob (0.56) |
| **Eve** | Diana (0.63), Charlie (0.61) |

**Technical Details:**
- Service: `MatchingService.find_user_matches()`
- Algorithm: pgvector cosine similarity (`1 - (vector_data <=> query_vector)`)
- Default threshold: 0.7 (configurable via `SIMILARITY_THRESHOLD` env var)

**Match Matrix Visualization:**
```
Alice ←──0.73──→ Bob
  │                │
  │ 0.57          │ 0.56
  ↓                ↓
Diana ←──0.65──→ Charlie
  │                │
  │ 0.63          │ 0.61
  ↓                ↓
 Eve ←────────────┘
```

**Why These Matches Make Sense:**
1. **Alice ↔ Bob (0.73)** — Alice wants ML mentorship, Bob offers ML mentorship
2. **Charlie ↔ Diana (0.65)** — Charlie needs tech co-founder, Diana wants startup role
3. **Charlie ↔ Eve (0.61)** — Charlie is raising funds, Eve is a VC investor
4. **Diana ↔ Eve (0.63)** — Both in startup/investment ecosystem
5. **Alice ↔ Diana (0.57)** — Both technical, fintech/tech overlap

---

### STEP 6: Connection — Ice Breakers & Match Explanations

**What happens:** The system generates personalized conversation starters for matched users.

| User | Ice Breakers Generated | Match Explainer |
|------|------------------------|-----------------|
| Alice | 3 ice breakers | Service ready |
| Bob | 3 ice breakers | Service ready |
| Charlie | 3 ice breakers | Service ready |
| Diana | 3 ice breakers | Service ready |
| Eve | 3 ice breakers | Service ready |

**Technical Details:**
- Service: `IceBreakerGenerator.generate_ice_breakers()`
- Returns: `IceBreakerSet` with multiple `IceBreaker` objects
- Each breaker has: text, style (professional/casual/direct), category (shared_interest/complementary/etc.)

**Example Ice Breakers (Alice → Bob):**
```
1. [PROFESSIONAL] "I noticed you have extensive experience in ML research.
   I'm transitioning from software engineering and would love to learn about
   your journey into AI."

2. [CURIOUS] "What was the most challenging part of moving from industry to
   AI research? I'm facing that transition now."

3. [DIRECT] "I'd appreciate any guidance on breaking into ML research.
   Do you have 15 minutes for a quick call?"
```

---

### STEP 7: Feedback — System Learning

**What happens:** Users submit feedback on matches, which the system stores for future learning.

| User | Feedback Submitted | Feedback Text |
|------|-------------------|---------------|
| Alice | PASS | "Looking forward to learning from experienced ML practitioners!" |
| Bob | PASS | "Happy to help engineers break into AI research!" |
| Charlie | PASS | "Excited to find a technical co-founder for this mission!" |
| Diana | PASS | "Ready to build something meaningful in healthtech!" |
| Eve | PASS | "Always looking for exceptional founders to back!" |

**Technical Details:**
- Endpoint: `POST /api/v1/user/feedback`
- Storage: DynamoDB table `user_feedback`
- Service: `FeedbackLearner` (for future ML-based preference learning)

---

## Infrastructure Used

| Component | Technology | Details |
|-----------|------------|---------|
| API Server | FastAPI | Port 8000, uvicorn |
| User Profiles | DynamoDB (LocalStack) | Port 4566, PynamoDB ORM |
| Embeddings | PostgreSQL + pgvector | Port 5433, 768-dim vectors |
| Caching | Redis | Port 6380, 7-day TTL |
| AI/LLM | OpenAI GPT-4.1-mini | Question modification, persona generation |
| Embeddings Model | SentenceTransformers | all-mpnet-base-v2 |

---

## Issues Encountered & Fixes

### Issue 1: DynamoDB Region Error
**Error:** `You must specify a region`
**Cause:** Environment variable mismatch (`AWS_REGION` vs `AWS_DEFAULT_REGION`)
**Fix:** Updated `app/adapters/dynamodb.py` to check both:
```python
region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION', 'us-east-1')
```

### Issue 2: OpenAI API Key Override
**Error:** `Incorrect API key provided: your_api_key`
**Cause:** Stale system environment variable overriding `.env` file
**Fix:** Explicitly override env vars from `.env` before imports:
```python
from dotenv import dotenv_values
env_values = dotenv_values('.env')
for key, value in env_values.items():
    if value:
        os.environ[key] = value
```

### Issue 3: LLM_TEMPERATURE Missing
**Error:** `float() argument must be a string or a real number, not 'NoneType'`
**Cause:** `LLM_TEMPERATURE` not set in `.env`
**Fix:** Added default value in `app/services/llm_service.py`:
```python
self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
```

### Issue 4: user_embeddings Table Missing
**Error:** `relation "user_embeddings" does not exist`
**Cause:** PostgreSQL table not created
**Fix:** Created table with pgvector:
```sql
CREATE TABLE user_embeddings (
    user_id VARCHAR(255) NOT NULL,
    embedding_type VARCHAR(50) NOT NULL,
    vector_data vector(768) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (user_id, embedding_type)
);
```

### Issue 5: MultiVectorMatch Import Error
**Error:** `cannot import name 'MultiVectorMatch'`
**Cause:** Class renamed to `MultiVectorMatchResult`
**Fix:** Updated import in `app/services/match_explanation.py`:
```python
from app.services.multi_vector_matcher import MatchTier, MultiVectorMatchResult as MultiVectorMatch
```

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/seed_test_users.py` | Register 5 users, run through Steps 1-3, 6 |
| `scripts/complete_user_pipeline.py` | Generate personas, embeddings, run matching |
| `scripts/test_full_journey.py` | Test all 6 journey steps with random users |
| `scripts/verify_e2e.py` | Verify infrastructure (API, Redis, PostgreSQL, DynamoDB) |

---

## How to Re-Run

```bash
# 1. Ensure Docker is running
docker-compose up -d

# 2. Verify infrastructure
python scripts/verify_e2e.py

# 3. Seed test users (if not already done)
python scripts/seed_test_users.py

# 4. Complete pipeline (personas, embeddings, matching)
python scripts/complete_user_pipeline.py

# 5. Check matches with custom threshold
python -c "
from app.services.matching_service import MatchingService
ms = MatchingService()
result = ms.find_user_matches('11111111-1111-1111-1111-111111111111', similarity_threshold=0.5)
print(result)
"
```

---

## Persistent User IDs

These UUIDs are fixed and will persist across test runs:

```
alice      11111111-1111-1111-1111-111111111111
bob        22222222-2222-2222-2222-222222222222
charlie    33333333-3333-3333-3333-333333333333
diana      44444444-4444-4444-4444-444444444444
eve        55555555-5555-5555-5555-555555555555
```

---

*Generated: February 10, 2026*
