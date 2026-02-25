# Reciprocity Platform - Complete Codebase Audit Instructions

## Your Role
You are auditing the **complete Reciprocity platform** - an AI-powered professional networking application consisting of 3 codebases. Your task is to review for quality, consistency, security, integration correctness, and completeness.

---

## Platform Overview

**What it is:** A networking platform that uses AI to match professionals based on their objectives, then facilitates AI-mediated conversations before connecting them.

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│   AI Service    │
│  React + Vite   │     │  NestJS + PG    │     │ FastAPI + Celery│
│   Port: 5173    │     │   Port: 3000    │     │   Port: 8000    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                       │
                                ▼                       ▼
                        ┌───────────────┐       ┌───────────────┐
                        │  PostgreSQL   │       │ Redis + PGVec │
                        └───────────────┘       └───────────────┘
```

---

## The 3 Repositories

| Repo | Path | Tech Stack | Purpose |
|------|------|------------|---------|
| **Frontend** | `C:\Users\hp\reciprocity-frontend` | React 19, TypeScript, Vite, TailwindCSS, React Query | User interface |
| **Backend** | `C:\Users\hp\reciprocity-backend` | NestJS, Sequelize-TypeScript, PostgreSQL, Bull | API, business logic, data |
| **AI Service** | `C:\Users\hp\reciprocity-ai` | FastAPI, Celery, pgvector, OpenAI/Claude | ML, matching, embeddings |

---

## Repository 1: Frontend (`reciprocity-frontend`)

### Tech Stack
- React 19, TypeScript, Vite
- TailwindCSS, React Query, React Router
- Stream Chat SDK, Firebase (push notifications)

### Structure
```
src/
├── components/
│   ├── common/         # Button, Badge, Spinner, Sidebar, ErrorBoundary
│   ├── matches/        # MatchCard, MatchExplanation, ScoreBreakdown, DeclineModal
│   ├── chat/           # FirstMessagePanel, ChatWindow
│   ├── aichat/         # AIChatResult, AIChatMessages
│   ├── onboarding/     # Footer, EditFooter, QuestionCard
│   ├── dashboard/      # MatchListing, StatsCards
│   └── discover/       # AnonymousProfileCard, SearchFilters (NEW)
├── pages/              # Route pages (Home, Matches, AIChat, Discover, etc.)
├── hooks/              # useMatch, useDiscover, useAnalytics, useOnboardingInput
├── services/           # API calls (matchService, discoverService, analyticsService)
├── contexts/           # AuthContext, OnboardingContext, SocketContext
├── types/              # TypeScript interfaces (match.ts, aiChat.ts, user.ts)
└── utils/              # Helpers, axios config, date formatting
```

### Key Files to Audit
```
src/App.tsx                                    # Route definitions
src/contexts/AuthContext.tsx                   # Auth state
src/components/matches/MatchCard.tsx           # Match display
src/components/matches/MatchExplanation.tsx    # NEW - Why this match?
src/components/matches/ScoreBreakdown.tsx      # NEW - 6-dimension scores
src/components/matches/DeclineModal.tsx        # NEW - Feedback collection
src/components/chat/FirstMessagePanel.tsx      # NEW - Ice breakers
src/components/aichat/AIChatResult.tsx         # NEW - Rich verdicts
src/pages/Discover/DiscoverPage.tsx            # NEW - Profile discovery
src/hooks/useMatch.ts                          # Match data hooks
src/hooks/useDiscover.ts                       # NEW - Discovery hooks
src/services/matchService.ts                   # Match API calls
src/types/match.ts                             # Match interfaces
```

---

## Repository 2: Backend (`reciprocity-backend`)

### Tech Stack
- NestJS, Sequelize-TypeScript, PostgreSQL
- Bull (Redis queues), class-validator
- Stream Chat (server SDK)

### Structure
```
src/
├── modules/
│   ├── auth/              # JWT auth, guards
│   ├── profile/           # User profiles
│   ├── onboarding/        # Onboarding flow
│   ├── dashboard/         # Match management, explanations, ice breakers
│   ├── ai-conversations/  # AI chat sessions
│   ├── chat/              # Direct messaging
│   ├── mail/              # Email service (Bull queue)
│   ├── discover/          # NEW - Profile discovery
│   └── analytics/         # NEW - Event tracking
├── common/
│   ├── entities/          # Sequelize models
│   ├── enums.ts           # Shared enums
│   └── guards/            # Auth guards
├── integration/
│   └── ai-service/        # Facade to AI microservice
└── database/
    └── migrations/        # Sequelize migrations
```

### Key Files to Audit
```
src/app.module.ts                              # Module imports
src/modules/dashboard/dashboard.service.ts     # Core match logic
src/modules/dashboard/dashboard.controller.ts  # Match endpoints
src/modules/ai-conversations/ai-conversations.service.ts
src/modules/discover/discover.service.ts       # NEW
src/modules/discover/discover.controller.ts    # NEW
src/modules/analytics/analytics.service.ts     # NEW
src/integration/ai-service/ai-service.facade.ts  # AI integration
src/common/entities/match.entity.ts
src/common/entities/user-summaries.entity.ts
src/common/entities/connection-interest.entity.ts  # NEW
src/common/entities/analytics-event.entity.ts      # NEW
database/migrations/20260210*.js               # NEW migrations
```

---

## Repository 3: AI Service (`reciprocity-ai`)

### Tech Stack
- FastAPI, Pydantic, SQLAlchemy
- Celery (async tasks), Redis
- pgvector (embeddings), OpenAI/Claude APIs

### Structure
```
app/
├── routers/
│   ├── health.py          # Health check endpoint
│   ├── matching.py        # Match scoring endpoints
│   ├── question.py        # Question modification
│   ├── prediction.py      # Match predictions
│   └── user.py            # User/persona endpoints
├── services/
│   ├── ai_chat_service.py       # AI conversation logic
│   ├── ai_conversation.py       # Conversation state
│   ├── matching_service.py      # Match scoring
│   ├── multi_vector_matcher.py  # 6-dimension scoring
│   ├── feedback_learner.py      # Learn from rejections
│   ├── ice_breakers.py          # Generate openers
│   ├── match_explanation.py     # Generate explanations
│   ├── persona_service.py       # Persona generation
│   ├── embedding_service.py     # Vector embeddings
│   ├── llm_service.py           # LLM abstraction
│   └── ...
├── workers/
│   ├── ai_chat_processing.py    # Async AI chat
│   ├── embedding_processing.py  # Async embeddings
│   ├── resume_processing.py     # Resume parsing
│   └── scheduled_matching.py    # Batch matching
├── schemas/                     # Pydantic models
├── prompts/                     # LLM prompts
│   └── persona_prompts.py
├── middleware/
│   ├── rate_limit.py
│   └── error_handling.py
└── utils/
    ├── cache.py
    └── logging_config.py
```

### Key Files to Audit
```
app/routers/matching.py                    # Match endpoints
app/routers/question.py                    # Question modification
app/services/ai_chat_service.py            # AI conversation
app/services/matching_service.py           # Match scoring
app/services/multi_vector_matcher.py       # 6-dimension scoring
app/services/feedback_learner.py           # Feedback learning
app/services/ice_breakers.py               # Ice breaker generation
app/services/match_explanation.py          # Explanation generation
app/services/persona_service.py            # Persona generation
app/services/llm_service.py                # LLM calls
app/workers/ai_chat_processing.py          # Celery tasks
app/prompts/persona_prompts.py             # System prompts
alembic/versions/*.py                      # DB migrations
```

---

## Integration Points to Verify

### Backend → AI Service
```
Backend calls AI service via HTTP:
- POST /api/v1/matching/score          # Get match scores
- POST /api/v1/matching/explanation    # Get match explanation
- POST /api/v1/question/modify         # Modify onboarding questions
- POST /api/v1/prediction/verdict      # Get AI chat verdict
- POST /api/v1/user/persona            # Generate persona
```

**Check:**
- Request/response schemas match between backend facade and AI routers
- Error handling for AI service failures
- Timeout configuration
- Retry logic

### Frontend → Backend
```
Frontend calls Backend via REST:
- GET  /dashboard/matches
- GET  /dashboard/match-explanation/:id
- GET  /dashboard/ice-breakers/:id
- POST /dashboard/decline-match/:id
- GET  /discover/search-profiles
- POST /discover/express-interest
- POST /analytics/track-event
```

**Check:**
- Types match between frontend and backend DTOs
- Auth token handling
- Error response handling
- Loading states

---

## New Features Added (Feb 2026)

### Phase 1: Quick Wins
| Feature | Frontend | Backend | AI |
|---------|----------|---------|-----|
| Match Explanation | `MatchExplanation.tsx` | `dashboard.service.ts` | `match_explanation.py` |
| Ice Breakers | `FirstMessagePanel.tsx` | `dashboard.service.ts` | `ice_breakers.py` |
| Rich Verdicts | `AIChatResult.tsx` | `ai-conversations.service.ts` | `ai_chat_service.py` |

### Phase 2: Core Intelligence
| Feature | Frontend | Backend | AI |
|---------|----------|---------|-----|
| Feedback Learning | `DeclineModal.tsx` | `dashboard.service.ts` | `feedback_learner.py` |
| Multi-Vector Scoring | Types in `match.ts` | Entity columns | `multi_vector_matcher.py` |
| Match Tiers | `MATCH_TIER_CONFIG` | Generated column | Scoring logic |

### Phase 3: Advanced Features
| Feature | Frontend | Backend | AI |
|---------|----------|---------|-----|
| Temporal Relevance | `UrgencySelector.tsx` | `user-summaries.entity.ts` | Freshness decay |
| Discovery System | `DiscoverPage.tsx` | `discover.module.ts` | N/A |
| Analytics Pipeline | `useAnalytics.ts` | `analytics.module.ts` | N/A |

---

## Audit Checklist

### 1. Cross-Repo Consistency
- [ ] API contracts match (frontend types ↔ backend DTOs ↔ AI schemas)
- [ ] Error codes are consistent
- [ ] Enum values match across repos
- [ ] Field naming conventions consistent (camelCase vs snake_case)

### 2. Security
- [ ] No secrets in code (API keys, passwords)
- [ ] Auth guards on all protected endpoints
- [ ] Input validation (SQL injection, XSS, prompt injection)
- [ ] Rate limiting on AI endpoints
- [ ] CORS configuration

### 3. Error Handling
- [ ] Frontend handles API errors gracefully
- [ ] Backend handles AI service failures
- [ ] AI service handles LLM failures
- [ ] Proper logging across all repos

### 4. Performance
- [ ] Frontend: React Query caching, lazy loading
- [ ] Backend: Database indexes, N+1 queries
- [ ] AI: Embedding caching, batch processing

### 5. Code Quality
- [ ] TypeScript: No `any` types
- [ ] Python: Type hints, docstrings
- [ ] Consistent patterns within each repo

---

## Commands to Run

### Frontend
```bash
cd C:/Users/hp/reciprocity-frontend
npm run lint                    # ESLint
npm run build                   # TypeScript + Vite build
```

### Backend
```bash
cd C:/Users/hp/reciprocity-backend
npm run lint                    # ESLint
npx tsc --noEmit               # TypeScript check
npm run test                    # Jest tests
npx sequelize-cli db:migrate:status
```

### AI Service
```bash
cd C:/Users/hp/reciprocity-ai
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m pytest tests/         # Run tests
python -m mypy app/             # Type checking (if configured)
alembic current                 # Migration status
```

---

## Output Format

Provide your audit results organized by repository:

```markdown
## Summary
- Critical issues: X
- High priority: X
- Medium priority: X

---

## Frontend Issues

### Critical
1. [FILE:LINE] Description

### High Priority
1. [FILE:LINE] Description

---

## Backend Issues

### Critical
1. [FILE:LINE] Description

### High Priority
1. [FILE:LINE] Description

---

## AI Service Issues

### Critical
1. [FILE:LINE] Description

### High Priority
1. [FILE:LINE] Description

---

## Integration Issues

### API Contract Mismatches
1. Description

### Missing Error Handling
1. Description

---

## Recommendations
1. Suggestion for improvement
```

---

## Questions to Answer

1. Do API contracts match across all three repos?
2. Is error handling consistent and complete?
3. Are there any security vulnerabilities?
4. Are there orphaned files or dead code?
5. Is the caching strategy appropriate?
6. Are LLM prompts properly secured against injection?
7. Is the feedback learning loop complete end-to-end?
8. Are database migrations in sync across Backend and AI?
