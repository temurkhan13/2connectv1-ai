# Reciprocity AI Service - Deployment Checklist

## Before Testing Any Changes

### 1. Rebuild Docker (if Python code changed)
```bash
cd /c/Users/hp/reciprocity-ai
docker-compose down ai-service
docker-compose up -d --build ai-service
```

### 2. Verify AI Service Health
```bash
curl http://localhost:8000/health
# Expected: {"success":true,"data":{"status":"healthy"}}
```

### 3. Verify LLM Response Format
Test the onboarding chat endpoint and check:
- Response does NOT contain "The user is..."
- Response does NOT contain "They are..."
- Response is a SHORT question only

### 4. Verify Onboarding Flow
- `/onboarding` should redirect to `/onboarding/chat`
- Use `?legacy=true` to access old fixed-question flow

### 5. Verify Matching Works
```bash
curl http://localhost:8000/api/v1/matching/{user_id}/matches
# Should return matches from AI service database
```

---

## Known Issues & Fixes

### Issue: LLM says "The user is a founder..."
**File:** `app/services/llm_slot_extractor.py`
**Fix:** `_clean_follow_up_question()` strips third-person preambles
**Verify:** Check `follow_up_question` in API response

### Issue: Onboarding uses old fixed flow
**File:** `reciprocity-frontend/src/pages/Onboarding/OnboardingPage.tsx`
**Fix:** Auto-redirect to `/onboarding/chat` unless `?legacy=true`

### Issue: No matches found (60 users, 0 matches)
**Cause:** Backend uses different DB than AI service
**Fix:** Backend calls AI service API, not direct DB
**Files:** `src/modules/matches/matches.service.ts`

### Issue: 100% progress but no redirect
**File:** `app/services/context_manager.py`
**Fix:** `_user_signals_completion()` detects "done", "no", etc.
