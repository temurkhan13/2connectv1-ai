# 2Connect Platform — User Testing Analysis
**Date:** March 3, 2026
**Reports Analyzed:** 2 full user journeys (Alice Anderson + Anonymous)

---

## Executive Summary

**Critical Finding:** Report 1 discovered a **BLOCKER bug** — after successful onboarding completion, users get trapped in an infinite onboarding loop. Report 2 did NOT encounter this bug, suggesting it's **intermittent or race-condition based**.

**Good News:**
- Conversational onboarding quality is excellent (7 contextual questions)
- Profile creation backend NOW WORKS (was failing in all previous tests)
- Core features (Discover, Filters, Dashboard) are functional

**Bad News:**
- Frontend state management bug creates permanent onboarding trap for some users
- Performance issues (80+ second message streaming)
- Inconsistent UX (modal shown/hidden randomly, metrics don't match)

---

## Issues by Severity

### 🚨 CRITICAL / BLOCKER (2 issues)

| # | Issue | Report | Impact |
|---|-------|--------|--------|
| **1** | **Onboarding loop after successful profile creation** | Report 1 | User permanently trapped. After "Creating your profile..." succeeds and dashboard loads, ALL navigation redirects back to /onboarding/chat. No workaround. |
| **2** | **Onboarding completion state not persisted** | Report 1 | Despite backend creating profile successfully (persona updated, activity logged), frontend `isOnboarded` flag (or equivalent) not saved. Route guard redirects indefinitely. |

**Root Cause Hypothesis:**
The backend fix that resolved profile creation timeout introduced a **frontend state management regression**:
- Backend: Profile creation ✅ works
- Frontend: Completion flag ❌ not persisted to localStorage/session/backend
- Result: Route guard thinks user incomplete → redirects to onboarding

**Why Report 2 didn't see this:**
- **Timing:** Report 1 tested earlier, Report 2 may have tested after a hotfix
- **Race condition:** WebSocket state update vs route guard check
- **Browser state:** Different caching/localStorage behavior

---

### ⚠️ HIGH (3 issues)

| # | Issue | Report | Details |
|---|-------|--------|---------|
| **3** | **Chat history wiped on onboarding re-entry** | Report 1 | When loop bug triggers, previous conversation (2 messages, 100% progress) completely erased. New welcome message streams from scratch. |
| **4** | **Progress regresses to 83% from 100%** | Report 1 | After achieving 100% and clicking "Complete & View Matches", redirect shows 83%. Progress calculation inconsistent. |
| **5** | **Onboarding extremely slow (80+ seconds for 3 sentences)** | Report 1 | Welcome message takes 80+ seconds to stream. Second message ~90 seconds. Poor first impression. Report 2 had 7 questions complete in reasonable time. |

**Note on #5:** Report 2 didn't mention streaming speed issues, suggesting:
- Network condition difference
- Server load during Report 1 test
- Claude Sonnet API latency spike

---

### 🔶 MEDIUM (4 issues)

| # | Issue | Report | Details |
|---|-------|--------|---------|
| **6** | **AI Summary page renders blank (Settings)** | Report 2 | Settings → AI Summary shows empty text box despite backend confirming persona generated ("Updated The Persona - 1h ago"). Edit/Approve buttons present but no content. Frontend fetch bug. |
| **7** | **Signup 409 error with no user feedback** | Report 1 | Email already exists → HTTP 409 returned, but signup form shows NO error message. User has no indication to switch to login. Form appears frozen. |
| **8** | **Profile review modal inconsistently shown** | Report 1 | Alice didn't get review modal. Previous tests: 3/5 got it, 2/5 skipped. Logic unclear. |
| **9** | **Inconsistent match score metrics** | Report 2 | Dashboard: "11-50%" (fluctuates), Matches page: "7%", different page loads show different values. Calculation or caching bug. |

---

### 🔷 LOW / INFO (4 issues)

| # | Issue | Report | Details |
|---|-------|--------|---------|
| **10** | **WebSocket 429 rate limit on initial load** | Report 2 | Stream Chat WebSocket hits rate limit → blank dashboard. Resolves on refresh. KNOWN ISSUE per test notes. |
| **11** | **"Failed to send message" error during onboarding** | Report 2 | Occasional error requiring retry. Encountered once. May be network/backend timeout. |
| **12** | **No AI-generated matches after 1+ hour** | Report 2 | Scheduled matching task (runs every 15 min per scheduled_matching.py) hasn't populated matches. May be expected for brand new users. |
| **13** | **Different button text in completion** | Report 1 | Shows "Complete & View Matches" instead of "I'm ready" seen in other sessions. Inconsistent UX. |

---

## What Worked Well ✅

| Feature | Evidence |
|---------|----------|
| **Conversational onboarding quality** | Report 2: 7 contextual questions, AI referenced prior answers ("decision support classification" callback), natural language rewriting excellent |
| **Profile creation backend** | Report 1: "Creating your profile..." succeeded (first time after 5/5 failures in previous tests) |
| **Discover search & filtering** | Report 2: 80 profiles loaded, search fast, filtered 80→6 with industry tags, "Clear all" works |
| **Real-time notifications** | Report 2: Notification bell showed "updated the persona - 1h ago", activity feed updated live |
| **Dashboard UI** | Both reports: Clean 4-stat overview, activity feed, navigation works |
| **Extraction & slot filling** | Report 2: Progress jumped 0%→41%→57%→64%→70%→80%→100%, complex answers parsed correctly |

---

## Critical Path Analysis

### Report 1 (Alice) — FAILED Journey
```
Signup (409, no error) → Login ✅ → Onboarding Chat ✅ (2 messages, 100%)
→ "Complete & View Matches" → Profile creation ✅ → Dashboard loads ✅
→ Click ANY link → ❌ REDIRECT TO ONBOARDING (83%, history wiped)
→ TRAPPED IN LOOP
```

### Report 2 (Anonymous) — SUCCESSFUL Journey
```
Signup ✅ → Login ✅ → Onboarding Chat ✅ (7 questions, 100%)
→ "Complete & View Matches" → Profile creation ✅ → Dashboard ✅
→ Navigate Discover/Matches/Inbox/Settings ✅
→ ALL FEATURES ACCESSIBLE
```

**Key Difference:** Report 2's frontend state persisted correctly. Report 1's did not.

---

## Comparison: Before vs After Enhanced Extraction

**From IMPLEMENTATION_SUMMARY.md (changes we just deployed):**
- Enhanced extraction hints for implicit offerings/requirements
- Question count limit (max 5 questions)

**Actual Results:**
- Report 1: 2 questions, 100% completion (BETTER than expected)
- Report 2: 7 questions, 100% completion (ABOVE the 5-question limit we set)

**Analysis:**
- Report 1 tested BEFORE our changes deployed (old extraction logic)
- Report 2 likely tested BEFORE our changes deployed (still asking 7 questions)
- OR: `MAX_ONBOARDING_QUESTIONS=5` env var not set on staging

**Action Item:** Verify our commit `9cb1f43` is deployed to staging and `MAX_ONBOARDING_QUESTIONS=5` is set.

---

## Root Cause: The Onboarding Loop Bug

### What Changed Recently
**Previous behavior (all past tests):**
- "Creating your profile..." → TIMEOUT/FAILURE ❌
- User stayed in chat (no redirect)
- Workaround: Manually navigate to /dashboard (worked)

**Current behavior (Report 1):**
- "Creating your profile..." → SUCCESS ✅
- Dashboard loads briefly ✅
- Navigation triggers redirect ❌
- Trapped in loop, no workaround ❌

### The Regression
A backend fix that resolved the profile creation timeout **introduced a frontend state bug**:

**Backend Flow (now working):**
```
1. POST /onboarding/finalize → success
2. Celery: generate_persona_task → DynamoDB persona created ✅
3. Webhook: POST /backend/users/:id/webhook → "Updated The Persona" ✅
4. Frontend: Redirect to /dashboard ✅
```

**Frontend Flow (broken):**
```
1. useOnboarding hook: is_complete = true (in-memory state) ✅
2. Navigate to /dashboard → renders ✅
3. Click Discover link → Route guard checks onboarding status...
4. ???: Checks backend endpoint OR localStorage OR session storage
5. Backend/storage says: is_complete = false ❌
6. Route guard: Redirect to /onboarding/chat ❌
7. Context lost: Chat history wiped, progress reset to 83% ❌
```

### Where the State Should Be
The `isOnboarded` flag needs to persist in **at least 2 of 3 places**:
1. **Backend:** `users` table → `onboarding_completed` column
2. **Frontend localStorage:** `onboarding_state` or similar
3. **Frontend context/store:** React state (already working in-memory)

**Likely culprit:** Backend `onboarding_completed` flag not set during `/onboarding/finalize` success, or route guard checks wrong endpoint.

---

## Files to Investigate (for loop bug)

### Frontend (reciprocity-frontend)
| File | Purpose | Check |
|------|---------|-------|
| `src/hooks/useOnboarding.ts` | Onboarding state hook | Does `finalize()` persist to backend/localStorage? |
| `src/lib/routes.ts` + `App.tsx` | Route guards | What condition checks if user completed onboarding? |
| `src/contexts/OnboardingContext.tsx` | Onboarding context (if exists) | Is state persisted on unmount? |

### Backend (reciprocity-backend)
| File | Purpose | Check |
|------|---------|-------|
| `src/modules/onboarding/onboarding.service.ts` | Finalize logic | Does `finalizeOnboarding()` set `user.onboarding_completed = true`? |
| `src/modules/users/users.entity.ts` | User schema | Is `onboarding_completed` field defined? |
| `src/modules/dashboard/dashboard.controller.ts` | Dashboard access guard | What checks if user can access dashboard? |

### AI Service (reciprocity-ai)
| File | Purpose | Check |
|------|---------|-------|
| `app/routers/onboarding.py` | `/finalize` endpoint | Does it notify backend of completion? |
| `app/services/context_manager.py` | Session finalization | Does `finalize_session()` trigger backend webhook? |

---

## Recommended Investigation Order

### 1. Reproduce the Loop Bug (HIGH PRIORITY)
```bash
# Test with fresh account
# Complete onboarding → Dashboard loads → Click Discover
# If redirect occurs → check browser DevTools:
# - Network tab: What endpoint does route guard call?
# - Console: Any errors?
# - Application tab: localStorage keys
```

### 2. Check Backend Completion Flag
```sql
-- Check if onboarding_completed exists and is set
SELECT id, email, onboarding_completed, created_at
FROM users
WHERE email = 'alice.anderson.1772535113@2connect-test.com';
```

### 3. Check Frontend State Persistence
```typescript
// In useOnboarding.ts or wherever finalize() is called
const finalize = async () => {
  const result = await api.post('/onboarding/finalize', sessionData);

  // Is this being called?
  localStorage.setItem('onboarding_complete', 'true');

  // Is this being called?
  await updateUserOnBackend({ onboarding_completed: true });
};
```

### 4. Check Route Guard Logic
```typescript
// In App.tsx or routes.ts
const ProtectedRoute = ({ children }) => {
  const { isComplete } = useOnboarding();

  // What is the source of truth here?
  // In-memory state? localStorage? Backend API call?
  if (!isComplete) {
    return <Navigate to="/onboarding/chat" />;
  }

  return children;
};
```

---

## Quick Wins (Can Fix Immediately)

### 1. Signup 409 Error Message
**File:** `reciprocity-frontend/src/pages/Auth/SignupPage.tsx`
```typescript
// Add error handling
try {
  await signup(email, password);
} catch (error) {
  if (error.response?.status === 409) {
    setError("This email is already registered. Please log in instead.");
  }
}
```

### 2. AI Summary Blank Page
**File:** `reciprocity-frontend/src/pages/Settings/AISummary.tsx`
```typescript
// Check if persona fetch is wired correctly
const { data: persona } = useQuery(['persona'], fetchPersona);

// If persona exists but not rendering, check:
<textarea value={persona?.summary || ''} />  // Add fallback
```

### 3. Message Streaming Timeout
**File:** `reciprocity-ai/app/routers/onboarding.py`
```python
# Check streaming chunk size
async def stream_response():
    for chunk in response.split(' '):
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.05)  # Too long? Reduce to 0.01
```

---

## Testing Checklist (Before Next Release)

- [ ] **Reproduce loop bug** with fresh account
- [ ] **Verify onboarding_completed flag** sets in database
- [ ] **Check route guard source of truth** (backend vs localStorage)
- [ ] **Test 409 error message** displays on signup
- [ ] **Verify AI Summary renders** persona text
- [ ] **Load test message streaming** (should be <10s for welcome message)
- [ ] **Confirm our enhanced extraction changes deployed** (check question count ≤5)
- [ ] **Set MAX_ONBOARDING_QUESTIONS=5** in staging .env

---

## Summary Table: Issues by Component

| Component | Critical | High | Medium | Low | Total |
|-----------|----------|------|--------|-----|-------|
| Onboarding state management | 2 | 2 | 1 | 1 | 6 |
| Message streaming | 0 | 1 | 0 | 1 | 2 |
| Settings/AI Summary | 0 | 0 | 1 | 0 | 1 |
| Signup/Auth | 0 | 0 | 1 | 0 | 1 |
| Matching/Metrics | 0 | 0 | 1 | 2 | 3 |
| **TOTAL** | **2** | **3** | **4** | **4** | **13** |

---

## Next Steps

1. **URGENT:** Fix onboarding loop bug (blocker for all users hitting it)
2. **HIGH:** Deploy our enhanced extraction changes (`9cb1f43`) to staging
3. **MEDIUM:** Fix AI Summary rendering, signup 409 error, streaming speed
4. **LOW:** Investigate scheduled matching delay, metric inconsistencies

---

## Open Questions

1. **Why did Report 2 not hit the loop bug?** Timing? Fix deployed between tests? Race condition?
2. **Are our enhanced extraction changes deployed yet?** Report 2 still saw 7 questions (above our 5 limit)
3. **What's the actual source of truth for onboarding completion?** Backend DB? localStorage? React state?
4. **Why does progress show 83% instead of 0% or 100%?** Cached partial state from Redis?
