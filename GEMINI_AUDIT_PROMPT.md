# Gemini Audit Prompt - All 3 Repos

Copy and paste this directly into Gemini:

---

You are auditing **Reciprocity** - an AI-powered professional networking platform with **3 codebases**.

## Architecture
```
Frontend (React) → Backend (NestJS) → AI Service (FastAPI)
   :5173              :3000               :8000
```

## The 3 Repositories

| Repo | Path | Stack |
|------|------|-------|
| Frontend | `C:\Users\hp\reciprocity-frontend` | React 19, TypeScript, Vite |
| Backend | `C:\Users\hp\reciprocity-backend` | NestJS, Sequelize, PostgreSQL |
| AI | `C:\Users\hp\reciprocity-ai` | FastAPI, Celery, pgvector |

---

## Priority Files to Audit

### Frontend
```
src/components/matches/MatchExplanation.tsx    # NEW - "Why this match?"
src/components/matches/ScoreBreakdown.tsx      # NEW - 6-dimension scores
src/components/matches/DeclineModal.tsx        # NEW - Feedback modal
src/components/chat/FirstMessagePanel.tsx      # NEW - Ice breakers
src/components/aichat/AIChatResult.tsx         # NEW - Rich verdicts
src/pages/Discover/DiscoverPage.tsx            # NEW - Discovery
src/hooks/useMatch.ts
src/hooks/useDiscover.ts                       # NEW
src/services/matchService.ts
src/types/match.ts
```

### Backend
```
src/modules/dashboard/dashboard.service.ts     # Core match logic
src/modules/dashboard/dashboard.controller.ts
src/modules/discover/discover.service.ts       # NEW
src/modules/discover/discover.controller.ts    # NEW
src/modules/analytics/analytics.service.ts     # NEW
src/integration/ai-service/ai-service.facade.ts
src/common/entities/match.entity.ts
src/common/entities/connection-interest.entity.ts  # NEW
database/migrations/20260210*.js               # NEW migrations
```

### AI Service
```
app/routers/matching.py                        # Match scoring
app/routers/question.py                        # Question modification
app/services/ai_chat_service.py                # AI conversation
app/services/multi_vector_matcher.py           # 6-dimension scoring
app/services/feedback_learner.py               # Feedback learning
app/services/ice_breakers.py                   # Ice breaker gen
app/services/match_explanation.py              # Explanation gen
app/services/llm_service.py                    # LLM calls
app/prompts/persona_prompts.py                 # System prompts
```

---

## What to Look For

### Cross-Repo Integration
- API contracts match (Frontend types ↔ Backend DTOs ↔ AI schemas)
- Error handling for service failures
- Consistent naming (camelCase vs snake_case)

### Security
- No secrets in code
- Input validation (SQL injection, XSS, prompt injection)
- Auth guards on protected endpoints
- Rate limiting on AI endpoints

### Code Quality
- TypeScript: No `any` types
- Python: Type hints present
- Error handling complete

---

## Output Format

```markdown
## Summary
- Critical: X
- High: X
- Medium: X

## Frontend Issues
1. [FILE:LINE] Description

## Backend Issues
1. [FILE:LINE] Description

## AI Service Issues
1. [FILE:LINE] Description

## Integration Issues
1. API mismatch between X and Y

## Recommendations
1. Suggestion
```

---

## Commands

```bash
# Frontend
cd C:/Users/hp/reciprocity-frontend && npm run build

# Backend
cd C:/Users/hp/reciprocity-backend && npx tsc --noEmit

# AI Service
cd C:/Users/hp/reciprocity-ai && python -m pytest tests/
```

Start by reading the priority files listed above across all 3 repos and provide your audit findings.
